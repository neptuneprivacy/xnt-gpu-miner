#include "kernels.cuh"
#include "gpu_resources.cuh"

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
    Digest* __restrict__ d_solution_nonce_digest) {
    
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
        
        uint64_t nonce_value = start_nonce + idx;
        
        // Create nonce digest
        Digest nonce_digest;
        nonce_digest.values[0] = nonce_value;
        nonce_digest.values[1] = 0;
        nonce_digest.values[2] = 0;
        nonce_digest.values[3] = 0;
        nonce_digest.values[4] = 0;
        
        // Compute indices from hash and nonce
        uint64_t index_a, index_b;
        Pow::indices(hash, nonce_digest, index_a, index_b);
        
        // Build Merkle paths and verify
        // For HIGH_VRAM mode, we have all data available
        
        // Compute path for index_a
        Digest path_a[MERKLE_TREE_HEIGHT_];
        size_t running_index_a = index_a;
        
        // First level: sibling leaf
        size_t sibling_leaf_a = running_index_a ^ 1;
        path_a[0] = d_leafs[sibling_leaf_a];
        running_index_a >>= 1;
        
        // Subsequent levels: internal nodes
        for (size_t level = 1; level < merkle_height; ++level) {
            size_t layer_start = 0;
            for (size_t l = 0; l < level - 1; ++l) {
                layer_start += num_leafs >> (l + 1);
            }
            size_t sibling_index = layer_start + (running_index_a ^ 1);
            path_a[level] = d_internal_nodes[sibling_index];
            running_index_a >>= 1;
        }
        
        // Compute path for index_b
        Digest path_b[MERKLE_TREE_HEIGHT_];
        size_t running_index_b = index_b;
        
        size_t sibling_leaf_b = running_index_b ^ 1;
        path_b[0] = d_leafs[sibling_leaf_b];
        running_index_b >>= 1;
        
        for (size_t level = 1; level < merkle_height; ++level) {
            size_t layer_start = 0;
            for (size_t l = 0; l < level - 1; ++l) {
                layer_start += num_leafs >> (l + 1);
            }
            size_t sibling_index = layer_start + (running_index_b ^ 1);
            path_b[level] = d_internal_nodes[sibling_index];
            running_index_b >>= 1;
        }
        
        // Get Merkle root (last internal node)
        Digest merkle_root = d_internal_nodes[num_leafs - 2];
        
        // Compute POW hash using MAST paths
        // Create temporary POW structure for encoding
        uint64_t state[STATE_SIZE];
        
        // Initialize state for variable length hashing
        tip5_sponge_init(state, Domain::VariableLength);
        
        // Absorb nonce
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) {
            state[i] = nonce_digest.values[i];
        }
        state[DIGEST_LEN] = 0;
        
        tip5_permutation(state);
        
        // Absorb paths and root (simplified - full implementation would hash complete encoding)
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) {
            state[i] = merkle_root.values[i];
        }
        
        tip5_permutation(state);
        
        // Combine with MAST paths for header hash
        Digest header_hash;
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) {
            header_hash.values[i] = state[i];
        }
        
        // Apply POW MAST path
        for (int i = 0; i < 3; ++i) {
            header_hash = tip5_hash_fixed_device(header_hash, mast_paths.pow[i]);
        }
        
        // Apply header MAST path
        for (int i = 0; i < 2; ++i) {
            header_hash = tip5_hash_fixed_device(header_hash, mast_paths.header[i]);
        }
        
        // Apply kernel MAST path
        Digest final_hash = tip5_hash_fixed_device(header_hash, mast_paths.kernel[0]);
        
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
            // Atomically claim this solution
            int was = atomicCAS(d_solution_found, 0, 1);
            if (was == 0) {
                // We won the race - store solution
                *d_solution_nonce = nonce_value;
                *d_solution_nonce_digest = nonce_digest;
                
                // Copy paths
                #pragma unroll
                for (int i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
                    d_solution_path_a[i] = path_a[i];
                    d_solution_path_b[i] = path_b[i];
                }
            }
            return;
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
    Digest* __restrict__ d_solution_nonce_digest) {
    
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
        
        uint64_t nonce_value = start_nonce + idx;
        
        Digest nonce_digest;
        nonce_digest.values[0] = nonce_value;
        nonce_digest.values[1] = 0;
        nonce_digest.values[2] = 0;
        nonce_digest.values[3] = 0;
        nonce_digest.values[4] = 0;
        
        uint64_t index_a, index_b;
        Pow::indices(hash, nonce_digest, index_a, index_b);
        
        // Compute leafs on-demand from commitment (leaf_prefix)
        Digest leaf_a = compute_leaf_from_commitment_device(leaf_prefix, index_a, num_leafs);
        Digest leaf_b = compute_leaf_from_commitment_device(leaf_prefix, index_b, num_leafs);
        
        // Build paths using stored internal nodes where available
        // For missing nodes, compute on-demand
        Digest path_a[MERKLE_TREE_HEIGHT_];
        Digest path_b[MERKLE_TREE_HEIGHT_];
        
        size_t running_a = index_a;
        size_t running_b = index_b;
        
        // First level: compute sibling leafs on-demand
        size_t sibling_a = running_a ^ 1;
        size_t sibling_b = running_b ^ 1;
        path_a[0] = compute_leaf_from_commitment_device(leaf_prefix, sibling_a, num_leafs);
        path_b[0] = compute_leaf_from_commitment_device(leaf_prefix, sibling_b, num_leafs);
        running_a >>= 1;
        running_b >>= 1;
        
        // Upper levels: use stored nodes or compute
        for (size_t level = 1; level < merkle_height; ++level) {
            size_t layer_start = 0;
            for (size_t l = 0; l < level - 1; ++l) {
                layer_start += num_leafs >> (l + 1);
            }
            
            size_t sibling_idx_a = layer_start + (running_a ^ 1);
            size_t sibling_idx_b = layer_start + (running_b ^ 1);
            
            path_a[level] = get_internal_node_safe(d_internal_nodes, sibling_idx_a,
                                                    stored_nodes_count, leaf_prefix,
                                                    leaf_prefix, num_leafs);
            path_b[level] = get_internal_node_safe(d_internal_nodes, sibling_idx_b,
                                                    stored_nodes_count, leaf_prefix,
                                                    leaf_prefix, num_leafs);
            
            running_a >>= 1;
            running_b >>= 1;
        }
        
        // Compute final hash (same as high_vram kernel)
        Digest merkle_root = d_internal_nodes[stored_nodes_count - 1];
        
        uint64_t state[STATE_SIZE];
        tip5_sponge_init(state, Domain::VariableLength);
        
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) {
            state[i] = nonce_digest.values[i];
        }
        state[DIGEST_LEN] = 0;
        tip5_permutation(state);
        
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) {
            state[i] = merkle_root.values[i];
        }
        tip5_permutation(state);
        
        Digest header_hash;
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) {
            header_hash.values[i] = state[i];
        }
        
        for (int i = 0; i < 3; ++i) {
            header_hash = tip5_hash_fixed_device(header_hash, mast_paths.pow[i]);
        }
        for (int i = 0; i < 2; ++i) {
            header_hash = tip5_hash_fixed_device(header_hash, mast_paths.header[i]);
        }
        Digest final_hash = tip5_hash_fixed_device(header_hash, mast_paths.kernel[0]);
        
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
                *d_solution_nonce = nonce_value;
                *d_solution_nonce_digest = nonce_digest;
                
                #pragma unroll
                for (int i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
                    d_solution_path_a[i] = path_a[i];
                    d_solution_path_b[i] = path_b[i];
                }
            }
            return;
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
    int min_grid_size, optimal_block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &optimal_block_size,
                                        parallel_mining_kernel_high_vram, 0, 0);
    
    // Use multiple of SM count for good occupancy
    int num_sms = prop.multiProcessorCount;
    int blocks_per_sm = 4; // Target occupancy
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

std::optional<Pow> mine_pow_with_buffer(
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
    
    // Select and launch appropriate kernel
    MiningKernelType kernel_type = select_mining_kernel(gpu_id);
    
    switch (kernel_type) {
        case MiningKernelType::HIGH_VRAM:
            parallel_mining_kernel_high_vram<<<blocks_per_grid, threads_per_block>>>(
                buffer.d_leafs,
                buffer.d_merkle_tree,
                buffer.hash,
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
                output.d_solution_nonce_digest);
            break;
            
        case MiningKernelType::LOW_VRAM:
            parallel_mining_kernel_low_vram<<<blocks_per_grid, threads_per_block>>>(
                nullptr,
                buffer.d_merkle_tree,
                buffer.hash,
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
                output.d_solution_nonce_digest);
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
        
        // Set root from buffer
        solution.root = buffer.merkle_root;
        
        output.free();
        return solution;
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
        result.pow_solution = solution.value();
    }
    
    return result;
}