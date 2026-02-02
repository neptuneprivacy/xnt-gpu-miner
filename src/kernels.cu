// Define before including headers to prevent extern declarations of d_gpu_range_*
#define KERNELS_DEFINING_RANGE_CONSTANTS

#include "kernels.cuh"
#include "gpu_resources.cuh"
#include "common.cuh"

// ===== GPU RANGE CONSTANTS =====
// Define the __constant__ variables here (declared extern in tip5.cuh for other TUs)
__constant__ uint64_t d_gpu_range_start;
__constant__ uint64_t d_gpu_range_size;

// ===== SHARED MEMORY LOOKUP TABLE =====
// Loaded once per block for S-box computation
__shared__ uint8_t s_lookup_table[256];

// ===== MINING OUTPUT BUFFERS =====

bool MiningOutputBuffers::allocate() {
    cudaError_t err;
    
    err = cudaMalloc(&d_solution_nonce, sizeof(uint64_t));
    if (err != cudaSuccess) return false;
    
    err = cudaMalloc(&d_solution_found, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_solution_nonce);
        d_solution_nonce = nullptr;
        return false;
    }
    
    err = cudaMalloc(&d_solution_path_a, MERKLE_TREE_HEIGHT_ * sizeof(Digest));
    if (err != cudaSuccess) {
        cudaFree(d_solution_nonce);
        cudaFree(d_solution_found);
        d_solution_nonce = nullptr;
        d_solution_found = nullptr;
        return false;
    }
    
    err = cudaMalloc(&d_solution_path_b, MERKLE_TREE_HEIGHT_ * sizeof(Digest));
    if (err != cudaSuccess) {
        cudaFree(d_solution_nonce);
        cudaFree(d_solution_found);
        cudaFree(d_solution_path_a);
        d_solution_nonce = nullptr;
        d_solution_found = nullptr;
        d_solution_path_a = nullptr;
        return false;
    }
    
    err = cudaMalloc(&d_solution_nonce_digest, sizeof(Digest));
    if (err != cudaSuccess) {
        cudaFree(d_solution_nonce);
        cudaFree(d_solution_found);
        cudaFree(d_solution_path_a);
        cudaFree(d_solution_path_b);
        d_solution_nonce = nullptr;
        d_solution_found = nullptr;
        d_solution_path_a = nullptr;
        d_solution_path_b = nullptr;
        return false;
    }
    
    err = cudaMalloc(&d_solution_final_hash, sizeof(Digest));
    if (err != cudaSuccess) {
        cudaFree(d_solution_nonce);
        cudaFree(d_solution_found);
        cudaFree(d_solution_path_a);
        cudaFree(d_solution_path_b);
        cudaFree(d_solution_nonce_digest);
        d_solution_nonce = nullptr;
        d_solution_found = nullptr;
        d_solution_path_a = nullptr;
        d_solution_path_b = nullptr;
        d_solution_nonce_digest = nullptr;
        return false;
    }
    
    return true;
}

void MiningOutputBuffers::free() {
    if (d_solution_nonce) {
        cudaFree(d_solution_nonce);
        d_solution_nonce = nullptr;
    }
    if (d_solution_found) {
        cudaFree(d_solution_found);
        d_solution_found = nullptr;
    }
    if (d_solution_path_a) {
        cudaFree(d_solution_path_a);
        d_solution_path_a = nullptr;
    }
    if (d_solution_path_b) {
        cudaFree(d_solution_path_b);
        d_solution_path_b = nullptr;
    }
    if (d_solution_nonce_digest) {
        cudaFree(d_solution_nonce_digest);
        d_solution_nonce_digest = nullptr;
    }
    if (d_solution_final_hash) {
        cudaFree(d_solution_final_hash);
        d_solution_final_hash = nullptr;
    }
}

bool MiningOutputBuffers::reset() {
    if (!d_solution_found) return false;
    
    cudaError_t err = cudaMemset(d_solution_found, 0, sizeof(int));
    return err == cudaSuccess;
}

// ===== HIGH-VRAM MINING KERNEL =====

__global__ void __launch_bounds__(256) parallel_mining_kernel_high_vram(
    const Digest* __restrict__ d_leafs,
    const Digest* __restrict__ d_internal_nodes,
    const Digest hash,
    const Digest target,
    const uint64_t start_nonce,
    const uint64_t num_nonces,
    const size_t num_leafs,
    const size_t merkle_height,
    const PowMastPaths mast_paths,
    const Digest leaf_prefix,
    const int consensus_rule_set,
    uint64_t* __restrict__ d_solution_nonce,
    int* __restrict__ d_solution_found,
    Digest* __restrict__ d_solution_path_a,
    Digest* __restrict__ d_solution_path_b,
    Digest* __restrict__ d_solution_nonce_digest,
    Digest* __restrict__ d_solution_final_hash) {
    
    // Load lookup table into shared memory (first warp)
    if (threadIdx.x < 256) {
        s_lookup_table[threadIdx.x] = LOOKUP_TABLE[threadIdx.x];
    }
    __syncthreads();
    
    // Check if solution already found
    if (*d_solution_found) return;
    
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    
    // Process nonces
    for (uint64_t idx = tid; idx < num_nonces; idx += stride) {
        // Early exit if solution found
        if (*d_solution_found) return;
        
        // Sequential nonce within GPU's range (using original working format)
        uint64_t nonce_value = d_gpu_range_start + start_nonce + idx;
        
        // Nonce digest - EXACT ORIGINAL FORMAT (this was working at 13-15 M/s!)
        Digest nonce_digest;
        nonce_digest.values[0] = nonce_value;
        nonce_digest.values[1] = (nonce_value >> 32);  // Upper bits in second limb
        nonce_digest.values[2] = 0;
        nonce_digest.values[3] = 0;
        nonce_digest.values[4] = 0;
        
        // Compute indices from index picker preimage and nonce
        uint64_t index_a, index_b;
        Pow_indices_device(hash, nonce_digest, index_a, index_b);
        
        // Paths are ALWAYS computed using original indices (matching Rust guess())
        // For HardforkAlpha, leaves are swapped during preprocessing, so paths use original indices
        // but the tree structure matches the swapped leaves
        uint64_t path_index_a = index_a;
        uint64_t path_index_b = index_b;
        
        // Compute path for index_a
        Digest path_a[MERKLE_TREE_HEIGHT_];
        size_t running_index_a = path_index_a + num_leafs;
        // After swap, Rust path() uses original index directly - leafs are swapped but tree structure matches
        size_t sibling_leaf_index_a = path_index_a ^ 1;
        path_a[0] = d_leafs[sibling_leaf_index_a];
        
        // Subsequent levels: internal nodes
        for (size_t level = 1; level < merkle_height; ++level) {
            running_index_a >>= 1;
            size_t sibling_index_a = running_index_a ^ 1;
            if (sibling_index_a < (MERKLE_NUM_LEAFS)) {
                path_a[level] = d_internal_nodes[sibling_index_a];
            } else {
                path_a[level] = Digest::default_digest();
            }
        }
        
        // Compute path for index_b
        Digest path_b[MERKLE_TREE_HEIGHT_];
        size_t running_index_b = path_index_b + num_leafs;
        // After swap, Rust path() uses original index directly - leafs are swapped but tree structure matches
        size_t sibling_leaf_index_b = path_index_b ^ 1;
        path_b[0] = d_leafs[sibling_leaf_index_b];
        
        for (size_t level = 1; level < merkle_height; ++level) {
            running_index_b >>= 1;
            size_t sibling_index_b = running_index_b ^ 1;
            if (sibling_index_b < (MERKLE_NUM_LEAFS)) {
                path_b[level] = d_internal_nodes[sibling_index_b];
            } else {
                path_b[level] = Digest::default_digest();
            }
        }
        
        // Get Merkle root (stored at last index in sequentially-built tree)
        // Tree is built sequentially: layer 0 at offset 0, root at last index
        // Total internal nodes = num_leafs - 1, so root is at index num_leafs - 2
        Digest merkle_root = d_internal_nodes[num_leafs - 2];
        
        // Compute POW hash using correct fast_mast_hash implementation
        Pow pow;
        pow.root = merkle_root;
        pow.nonce = nonce_digest;
        #pragma unroll
        for (int i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
            pow.path_a[i] = path_a[i];
            pow.path_b[i] = path_b[i];
        }
        
        Digest final_hash = mast_paths.fast_mast_hash_device(pow);
        
        // Check against target
        bool is_solution = true;
        for (int i = DIGEST_LEN - 1; i >= 0; --i) {
            if (final_hash.values[i] > target.values[i]) {
                is_solution = false;
                break;
            }
            if (final_hash.values[i] < target.values[i]) {
                break;
            }
        }
        
        if (is_solution) {
            int was = atomicCAS(d_solution_found, 0, 1);
            if (was == 0) {
                // Store first limb of nonce for basic tracking
                atomicExch((unsigned long long*)d_solution_nonce, nonce_digest.values[0]);
                // Store full nonce digest so host can submit all 5 limbs
                *d_solution_nonce_digest = nonce_digest;
                
                // Store the final_hash that met the threshold (for debugging)
                *d_solution_final_hash = final_hash;
                
                // Copy the already-computed paths
                #pragma unroll
                for (int i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
                    d_solution_path_a[i] = path_a[i];
                    d_solution_path_b[i] = path_b[i];
                }
                
                return;
            }
        }
    }
}

// ===== LOW-VRAM MINING KERNEL =====

__global__ void __launch_bounds__(256) parallel_mining_kernel_low_vram(
    const Digest* __restrict__ d_leafs,
    const Digest* __restrict__ d_internal_nodes,
    const Digest hash,
    const Digest target,
    const uint64_t start_nonce,
    const uint64_t num_nonces,
    const size_t num_leafs,
    const size_t merkle_height,
    const size_t stored_nodes_count,
    const PowMastPaths mast_paths,
    const Digest leaf_prefix,
    const int consensus_rule_set,
    uint64_t* __restrict__ d_solution_nonce,
    int* __restrict__ d_solution_found,
    Digest* __restrict__ d_solution_path_a,
    Digest* __restrict__ d_solution_path_b,
    Digest* __restrict__ d_solution_nonce_digest,
    Digest* __restrict__ d_solution_final_hash) {
    
    // Load lookup table into shared memory
    if (threadIdx.x < 256) {
        s_lookup_table[threadIdx.x] = LOOKUP_TABLE[threadIdx.x];
    }
    __syncthreads();
    
    if (*d_solution_found) return;
    
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    
    for (uint64_t idx = tid; idx < num_nonces; idx += stride) {
        if (*d_solution_found) return;
        
        uint64_t nonce_value = d_gpu_range_start + start_nonce + idx;
        
        // Nonce digest - MUST match Rust: Digest(bfe_array![0, 0, 0, 0, i])
        // The nonce value goes in the LAST limb (index 4), not the first!
        Digest nonce_digest;
        nonce_digest.values[0] = 0;
        nonce_digest.values[1] = 0;
        nonce_digest.values[2] = 0;
        nonce_digest.values[3] = 0;
        nonce_digest.values[4] = nonce_value;
        
        uint64_t index_a, index_b;
        Pow_indices_device(hash, nonce_digest, index_a, index_b);
        
        // Paths are ALWAYS computed using original indices (matching Rust guess())
        // For HardforkAlpha, leaves are swapped during preprocessing, so paths use original indices
        // but the tree structure matches the swapped leaves
        uint64_t path_index_a = index_a;
        uint64_t path_index_b = index_b;
        
        // Commitment is needed for get_internal_node_safe (for computing missing nodes)
        Digest commitment = mast_paths.commit_device();
        // leaf_prefix is passed as parameter (commitment for Reboot/Xnt, prev_block_digest for HardforkAlpha)
        
        // Build paths using stored internal nodes where available
        // For missing nodes, compute on-demand
        Digest path_a[MERKLE_TREE_HEIGHT_];
        Digest path_b[MERKLE_TREE_HEIGHT_];
        
        // Path B - use get_internal_node_safe for transparent stored/computed node access
        {
            size_t running_index = path_index_b + num_leafs;
            size_t sibling_leaf_index = path_index_b ^ 1;
            // Leaf level: compute from leaf_prefix (commitment in Reboot/Xnt, prev_block_digest in HardforkAlpha)
            // Use original index for computation (not bit-reversed)
            Digest sib = compute_leaf_from_commitment_device_parallel(leaf_prefix, sibling_leaf_index, num_leafs);
            path_b[0] = sib;
            
            // Internal nodes: use get_internal_node_safe (stored or computed)
            for (size_t level = 1; level < merkle_height; ++level) {
                running_index >>= 1;
                size_t sibling_index = running_index ^ 1;
                Digest node = get_internal_node_safe(d_internal_nodes, sibling_index, stored_nodes_count, commitment, leaf_prefix, num_leafs);
                path_b[level] = node;
            }
        }
        
        // Path A - use get_internal_node_safe for transparent stored/computed node access
        {
            size_t running_index = path_index_a + num_leafs;
            size_t sibling_leaf_index = path_index_a ^ 1;
            // Leaf level: compute from leaf_prefix (commitment in Reboot/Xnt, prev_block_digest in HardforkAlpha)
            // Use original index for computation (not bit-reversed)
            Digest sib = compute_leaf_from_commitment_device_parallel(leaf_prefix, sibling_leaf_index, num_leafs);
            path_a[0] = sib;
            
            // Internal nodes: use get_internal_node_safe (stored or computed)
            for (size_t level = 1; level < merkle_height; ++level) {
                running_index >>= 1;
                size_t sibling_index = running_index ^ 1;
                Digest node = get_internal_node_safe(d_internal_nodes, sibling_index, stored_nodes_count, commitment, leaf_prefix, num_leafs);
                path_a[level] = node;
            }
        }
        
        // Compute final hash using correct fast_mast_hash implementation
        // Root is at last index in sequentially-built tree
        Digest merkle_root = d_internal_nodes[stored_nodes_count - 1];
        Pow pow;
        pow.root = merkle_root;
        pow.nonce = nonce_digest;
        #pragma unroll
        for (int i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
            pow.path_a[i] = path_a[i];
            pow.path_b[i] = path_b[i];
        }
        
        Digest final_hash = mast_paths.fast_mast_hash_device(pow);
        
        bool is_solution = true;
        for (int i = DIGEST_LEN - 1; i >= 0; --i) {
            if (final_hash.values[i] > target.values[i]) {
                is_solution = false;
                break;
            }
            if (final_hash.values[i] < target.values[i]) {
                break;
            }
        }
        
        if (is_solution) {
            int was = atomicCAS(d_solution_found, 0, 1);
            if (was == 0) {
                atomicExch((unsigned long long*)d_solution_nonce, nonce_value);
                
                // Store the final_hash that met the threshold (for debugging)
                *d_solution_final_hash = final_hash;
                *d_solution_nonce_digest = nonce_digest;
                
                // Store paths - use get_internal_node_safe for both stored and computed nodes
                size_t running_index_a = path_index_a + num_leafs;
                size_t sibling_leaf_index_a = path_index_a ^ 1;
                // Use original index for computation (not bit-reversed)
                if (sibling_leaf_index_a < num_leafs) {
                    d_solution_path_a[0] = compute_leaf_from_commitment_device_parallel(leaf_prefix, sibling_leaf_index_a, num_leafs);
                } else {
                    d_solution_path_a[0] = Digest::default_digest();
                }
                for (size_t level = 1; level < merkle_height; ++level) {
                    running_index_a >>= 1;
                    size_t sibling_index_a = running_index_a ^ 1;
                    d_solution_path_a[level] = get_internal_node_safe(d_internal_nodes, sibling_index_a, stored_nodes_count, commitment, leaf_prefix, num_leafs);
                }
            
                size_t running_index_b = path_index_b + num_leafs;
                size_t sibling_leaf_index_b = path_index_b ^ 1;
                // Use original index for computation (not bit-reversed)
                if (sibling_leaf_index_b < num_leafs) {
                    d_solution_path_b[0] = compute_leaf_from_commitment_device_parallel(leaf_prefix, sibling_leaf_index_b, num_leafs);
                } else {
                    d_solution_path_b[0] = Digest::default_digest();
                }
                for (size_t level = 1; level < merkle_height; ++level) {
                    running_index_b >>= 1;
                    size_t sibling_index_b = running_index_b ^ 1;
                    d_solution_path_b[level] = get_internal_node_safe(d_internal_nodes, sibling_index_b, stored_nodes_count, commitment, leaf_prefix, num_leafs);
                }
            
                return;
            }
        }
    }
}

// ===== HOST MINING FUNCTIONS =====

void calculate_mining_launch_config(
    uint64_t num_nonces,
    int& threads_per_block,
    int& blocks_per_grid,
    int gpu_id) {
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    
    threads_per_block = MINING_THREADS_PER_BLOCK;
    
    // Calculate optimal number of blocks
    // Note: Temporarily disabled due to CUDA 13.0 compatibility issue
    // int min_grid_size, optimal_block_size;
    // cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &optimal_block_size,
    //                                     parallel_mining_kernel_high_vram, 0, 0);
    
    // Calculate maximum blocks to launch for optimal GPU utilization
    // Higher values allow more parallel blocks, improving performance for large nonce ranges
    // Note: This is effectively a multiplier to allow sufficient blocks to be launched.
    // The actual number of blocks is capped by needed_blocks and MAX_GRID_DIM_X.
    int num_sms = prop.multiProcessorCount;
    int blocks_per_sm = 128; // High value to allow maximum parallel blocks (capped by grid limits)
    int max_blocks = num_sms * blocks_per_sm;
    
    // Calculate blocks needed for nonces
    int needed_blocks = (num_nonces + threads_per_block - 1) / threads_per_block;
    
    // Cap at max blocks
    blocks_per_grid = std::min(needed_blocks, max_blocks);
    blocks_per_grid = std::min(blocks_per_grid, MAX_GRID_DIM_X);
}

uint64_t get_optimal_batch_size(int gpu_id, int target_duration_ms) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    
    uint64_t optimal = 1000000ULL;
    
    // Apply bounds
    const uint64_t min_batch = 100000;
    const uint64_t max_batch = 50000000;
    
    optimal = std::max(optimal, min_batch);
    optimal = std::min(optimal, max_batch);
    
    return optimal;
}

MiningKernelType select_mining_kernel(int gpu_id) {
    VramMode mode = detect_vram_mode(gpu_id);
    
    switch (mode) {
        case VramMode::HIGH_VRAM:
            return MiningKernelType::HIGH_VRAM;
        case VramMode::LOW_VRAM:
            return MiningKernelType::LOW_VRAM;
        default:
            return MiningKernelType::HIGH_VRAM;
    }
}

bool check_kernel_launch_errors(const char* kernel_name) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR(kernel_name, err);
        return false;
    }
    return true;
}

bool sync_and_check_errors(const char* stage) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        LOG_ERROR(stage, err);
        return false;
    }
    return true;
}

// ===== MAIN MINING FUNCTION =====

std::optional<MiningSolution> mine_pow_with_buffer(
    GuesserBuffer& buffer,
    const Digest& target,
    const PowMastPaths& mast_paths,
    uint64_t start_nonce,
    uint64_t max_nonces,
    int consensus_rule_set,
    bool* cancel_flag) {
    
    if (!buffer.is_valid()) {
        LOG_DEBUG("mine_pow_with_buffer: invalid buffer");
        return std::nullopt;
    }
    
    // Allocate output buffers
    MiningOutputBuffers output;
    if (!output.allocate()) {
        LOG_DEBUG("mine_pow_with_buffer: failed to allocate output buffers");
        return std::nullopt;
    }
    
    if (!output.reset()) {
        output.free();
        return std::nullopt;
    }
    
    // Get launch configuration
    int threads_per_block, blocks_per_grid;
    int gpu_id;
    cudaGetDevice(&gpu_id);
    calculate_mining_launch_config(max_nonces, threads_per_block, blocks_per_grid, gpu_id);
    
    // Calculate GPU's dedicated nonce range to avoid overlap with other GPUs
    // CRITICAL: Pass actual GPU count to ensure proper nonce space partitioning
    int actual_gpu_count = g_total_gpu_count.load();
    GpuNonceRange gpu_range = calculate_gpu_range(gpu_id, actual_gpu_count);
    
    // Copy range to constant memory (avoids register pressure from extra parameters)
    cudaError_t range_err = cudaMemcpyToSymbol(d_gpu_range_start, &gpu_range.range_start, sizeof(uint64_t));
    if (range_err != cudaSuccess) {
        LOG_ERROR("copy range_start", range_err);
        output.free();
        return std::nullopt;
    }
    range_err = cudaMemcpyToSymbol(d_gpu_range_size, &gpu_range.range_size, sizeof(uint64_t));
    if (range_err != cudaSuccess) {
        LOG_ERROR("copy range_size", range_err);
        output.free();
        return std::nullopt;
    }
    
    // Select and launch appropriate kernel
    MiningKernelType kernel_type = select_mining_kernel(gpu_id);
    
    switch (kernel_type) {
        case MiningKernelType::HIGH_VRAM:
            parallel_mining_kernel_high_vram<<<blocks_per_grid, threads_per_block>>>(
                buffer.d_leafs,
                buffer.d_merkle_tree,
                buffer.index_picker_preimage,
                target,
                start_nonce,
                max_nonces,
                buffer.num_leafs,
                MERKLE_TREE_HEIGHT_,
                mast_paths,
                buffer.hash, // leaf_prefix = commitment
                consensus_rule_set,
                output.d_solution_nonce,
                output.d_solution_found,
                output.d_solution_path_a,
                output.d_solution_path_b,
                output.d_solution_nonce_digest,
                output.d_solution_final_hash);
            break;
            
        case MiningKernelType::LOW_VRAM:
            parallel_mining_kernel_low_vram<<<blocks_per_grid, threads_per_block>>>(
                nullptr,
                buffer.d_merkle_tree,
                buffer.index_picker_preimage,
                target,
                start_nonce,
                max_nonces,
                buffer.num_leafs,
                MERKLE_TREE_HEIGHT_,
                buffer.tree_size, // stored_nodes_count
                mast_paths,
                buffer.hash,
                consensus_rule_set,
                output.d_solution_nonce,
                output.d_solution_found,
                output.d_solution_path_a,
                output.d_solution_path_b,
                output.d_solution_nonce_digest,
                output.d_solution_final_hash);
            break;
    }
    
    // Check for launch errors
    if (!check_kernel_launch_errors("mining_kernel")) {
        output.free();
        return std::nullopt;
    }
    
    // Synchronize and check for errors
    if (!sync_and_check_errors("mining_kernel_sync")) {
        output.free();
        return std::nullopt;
    }
    
    // Check if solution was found
    int solution_found = 0;
    cudaMemcpy(&solution_found, output.d_solution_found, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (solution_found) {
        // Copy solution data back to host
        Pow solution;
        
        uint64_t nonce_value;
        cudaMemcpy(&nonce_value, output.d_solution_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(&solution.nonce, output.d_solution_nonce_digest, sizeof(Digest), cudaMemcpyDeviceToHost);
        cudaMemcpy(solution.path_a, output.d_solution_path_a, 
                   MERKLE_TREE_HEIGHT_ * sizeof(Digest), cudaMemcpyDeviceToHost);
        cudaMemcpy(solution.path_b, output.d_solution_path_b,
                   MERKLE_TREE_HEIGHT_ * sizeof(Digest), cudaMemcpyDeviceToHost);
        
        // Copy the final_hash that kernel computed (for debugging)
        Digest kernel_final_hash;
        cudaMemcpy(&kernel_final_hash, output.d_solution_final_hash, sizeof(Digest), cudaMemcpyDeviceToHost);
        
        // Set root from buffer
        solution.root = buffer.merkle_root;
        
        // Return both solution and kernel_final_hash
        MiningSolution mining_solution;
        mining_solution.pow = solution;
        mining_solution.kernel_final_hash = kernel_final_hash;
        
        output.free();
        return mining_solution;
    }
    
    output.free();
    return std::nullopt;
}

// ===== MINING BATCH FUNCTION =====

MiningResult mine_batch(
    GuesserBuffer& buffer,
    const Digest& target,
    const PowMastPaths& mast_paths,
    uint64_t start_nonce,
    uint64_t batch_size,
    int consensus_rule_set,
    bool* cancel_flag) {
    
    MiningResult result;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto solution = mine_pow_with_buffer(
        buffer, target, mast_paths,
        start_nonce, batch_size, consensus_rule_set, cancel_flag);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    result.elapsed_seconds = duration.count() / 1e6;
    result.nonces_tested = batch_size;
    
    if (solution.has_value()) {
        result.solution_found = true;
        result.pow_solution = solution.value().pow;
        result.solution_hash = solution.value().kernel_final_hash;  // Use kernel's final_hash
    }
    
    return result;
}