// Define before including headers to prevent extern declarations of d_gpu_range_*
#define KERNELS_DEFINING_RANGE_CONSTANTS

#include "kernels.cuh"
#include "gpu_resources.cuh"
#include "common.cuh"

// ===== GPU RANGE CONSTANTS =====
// Define the __constant__ variables here (declared extern in tip5.cuh for other TUs)
__constant__ uint64_t d_gpu_range_start;
__constant__ uint64_t d_gpu_range_size;

// ===== TOP MERKLE TREE CACHE =====
// Cache top 8 levels of Merkle tree in constant memory for fast access
// Top 8 levels = 2^8 - 1 = 255 nodes = 255 * 40 bytes = 10,200 bytes
// These nodes are accessed by ALL threads, so caching eliminates global memory reads
// Constants TOP_TREE_CACHE_LEVELS and TOP_TREE_CACHE_SIZE defined in kernels.cuh
// Store as raw uint64_t since Digest has constructor (not allowed in __constant__)
__constant__ uint64_t d_top_tree_cache[TOP_TREE_CACHE_SIZE * DIGEST_LEN];

// Flag to track if cache is initialized for current job
static bool g_top_tree_cache_initialized = false;

// Helper to read cached node as Digest - defined here where d_top_tree_cache is visible
__device__ Digest get_cached_node(size_t index) {
    Digest d;
    size_t base = index * DIGEST_LEN;
    d.values[0] = d_top_tree_cache[base];
    d.values[1] = d_top_tree_cache[base + 1];
    d.values[2] = d_top_tree_cache[base + 2];
    d.values[3] = d_top_tree_cache[base + 3];
    d.values[4] = d_top_tree_cache[base + 4];
    return d;
}

// ===== SHARED MEMORY LOOKUP TABLE =====
// Loaded once per block for S-box computation
// OPTIMIZATION: Pad to 257 bytes to reduce bank conflicts (256 + 1 padding)
// This ensures adjacent threads don't hit the same bank
__shared__ uint8_t s_lookup_table[257];  // 256 + 1 padding for bank conflict reduction

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

// ===== GUESSER BUFFER MINING RESOURCES =====

bool GuesserBuffer::ensure_mining_resources() {
    // Create stream if not initialized
    if (!stream_initialized) {
        cudaError_t err = cudaStreamCreate(&mining_stream);
        if (err != cudaSuccess) {
            LOG_ERROR("cudaStreamCreate", err);
            return false;
        }
        stream_initialized = true;
    }
    
    // Allocate output buffers if not allocated
    if (!output_buffers_allocated) {
        cudaError_t err;
        
        err = cudaMalloc(&d_solution_nonce, sizeof(uint64_t));
        if (err != cudaSuccess) { LOG_ERROR("alloc d_solution_nonce", err); return false; }
        
        err = cudaMalloc(&d_solution_found, sizeof(int));
        if (err != cudaSuccess) { 
            cudaFree(d_solution_nonce); d_solution_nonce = nullptr;
            LOG_ERROR("alloc d_solution_found", err); 
            return false; 
        }
        
        err = cudaMalloc(&d_solution_path_a, MERKLE_TREE_HEIGHT_ * sizeof(Digest));
        if (err != cudaSuccess) {
            cudaFree(d_solution_nonce); d_solution_nonce = nullptr;
            cudaFree(d_solution_found); d_solution_found = nullptr;
            LOG_ERROR("alloc d_solution_path_a", err);
            return false;
        }
        
        err = cudaMalloc(&d_solution_path_b, MERKLE_TREE_HEIGHT_ * sizeof(Digest));
        if (err != cudaSuccess) {
            cudaFree(d_solution_nonce); d_solution_nonce = nullptr;
            cudaFree(d_solution_found); d_solution_found = nullptr;
            cudaFree(d_solution_path_a); d_solution_path_a = nullptr;
            LOG_ERROR("alloc d_solution_path_b", err);
            return false;
        }
        
        err = cudaMalloc(&d_solution_nonce_digest, sizeof(Digest));
        if (err != cudaSuccess) {
            cudaFree(d_solution_nonce); d_solution_nonce = nullptr;
            cudaFree(d_solution_found); d_solution_found = nullptr;
            cudaFree(d_solution_path_a); d_solution_path_a = nullptr;
            cudaFree(d_solution_path_b); d_solution_path_b = nullptr;
            LOG_ERROR("alloc d_solution_nonce_digest", err);
            return false;
        }
        
        err = cudaMalloc(&d_solution_final_hash, sizeof(Digest));
        if (err != cudaSuccess) {
            cudaFree(d_solution_nonce); d_solution_nonce = nullptr;
            cudaFree(d_solution_found); d_solution_found = nullptr;
            cudaFree(d_solution_path_a); d_solution_path_a = nullptr;
            cudaFree(d_solution_path_b); d_solution_path_b = nullptr;
            cudaFree(d_solution_nonce_digest); d_solution_nonce_digest = nullptr;
            LOG_ERROR("alloc d_solution_final_hash", err);
            return false;
        }
        
        output_buffers_allocated = true;
    }
    
    return true;
}

bool GuesserBuffer::reset_output_buffers() {
    if (!d_solution_found) return false;
    
    // Use async memset on the mining stream for better overlap
    cudaError_t err = cudaMemsetAsync(d_solution_found, 0, sizeof(int), mining_stream);
    return err == cudaSuccess;
}

// ===== TOP TREE CACHE FUNCTIONS =====

// Initialize the top tree cache from the merkle tree
// Call this once per job when the tree changes
bool initialize_top_tree_cache(const Digest* d_merkle_tree, size_t num_leafs) {
    if (g_top_tree_cache_initialized) return true;
    
    // Top tree cache stores the highest (smallest index) internal nodes
    // In our tree layout, the root is at index (num_leafs - 2)
    // We want to cache the top 8 levels = 255 nodes near the root
    // 
    // Tree indices (for num_leafs = 2^27):
    // Root is at index num_leafs - 2 = 134217726
    // Its children are at indices that hash to it
    //
    // Actually, we need to understand the tree layout:
    // The tree is stored with internal nodes indexed 0 to num_leafs-2
    // Index 0 is the first internal node (parent of leaves 0 and 1)
    // Root is at index num_leafs - 2
    //
    // For path computation, sibling_index starts large and gets smaller
    // At top levels (near root), sibling_index < 256 (for top 8 levels)
    //
    // So we cache indices 0 to TOP_TREE_CACHE_SIZE-1
    // Copy as raw uint64_t (same memory layout as Digest array)
    
    cudaError_t err = cudaMemcpyToSymbol(
        d_top_tree_cache, 
        d_merkle_tree,  // First TOP_TREE_CACHE_SIZE nodes (as raw bytes)
        TOP_TREE_CACHE_SIZE * DIGEST_LEN * sizeof(uint64_t),
        0,
        cudaMemcpyDeviceToDevice);
    
    if (err != cudaSuccess) {
        LOG_ERROR("initialize_top_tree_cache", err);
        return false;
    }
    
    g_top_tree_cache_initialized = true;
    return true;
}

// Reset cache flag when job changes
void reset_top_tree_cache() {
    g_top_tree_cache_initialized = false;
}

// Device function to get internal node - uses cache for small indices
__device__ __forceinline__ Digest get_internal_node_cached(
    const Digest* __restrict__ d_internal_nodes,
    size_t index,
    size_t num_leafs
) {
    // Check if this node is in our top-tree cache
    if (index < TOP_TREE_CACHE_SIZE) {
        return get_cached_node(index);
    }
    // Fall back to global memory
    return (index < num_leafs) ? d_internal_nodes[index] : Digest::default_digest();
}

// ===== HIGH-VRAM MINING KERNEL =====

__global__ void parallel_mining_kernel_high_vram(
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
    // OPTIMIZATION: Load with padding to reduce bank conflicts
    if (threadIdx.x < 256) {
        s_lookup_table[threadIdx.x] = LOOKUP_TABLE[threadIdx.x];
    }
    // Pad the last element to ensure proper alignment
    if (threadIdx.x == 0) {
        s_lookup_table[256] = 0;  // Padding byte
    }
    __syncthreads();
    
    // Check if solution already found
    if (*d_solution_found) return;
    
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    
    // Cache constant values outside loop to prevent repeated memory accesses
    // Merkle root is constant - read once and reuse to maintain 40 MH/s performance
    const Digest* __restrict__ root_ptr = &d_internal_nodes[num_leafs - 2];
    const Digest merkle_root = *root_ptr;  // Cache constant value
    
    // Process nonces
    for (uint64_t idx = tid; idx < num_nonces; idx += stride) {
        
        // Sequential nonce within GPU's range (using original working format)
        uint64_t nonce_value = d_gpu_range_start + start_nonce + idx;
        
        // Nonce digest - EXACT ORIGINAL FORMAT (this was working at 13-15 M/s!)
        Digest nonce_digest;
        nonce_digest.values[0] = nonce_value;
        nonce_digest.values[1] = 0;  // Upper bits in second limb
        nonce_digest.values[2] = 0;
        nonce_digest.values[3] = 0;
        nonce_digest.values[4] = 0;
        
        // Compute indices from index picker preimage and nonce
        uint64_t index_a, index_b;
        Pow_indices_device(hash, nonce_digest, index_a, index_b);
        
        // Compute POW hash directly from global memory - no Pow struct needed
        // This eliminates 2240 bytes of register pressure per thread
        // Uses top tree cache for fast access to upper Merkle levels
        Digest final_hash = fast_mast_hash_direct(
            mast_paths,
            nonce_digest,
            merkle_root,
            d_leafs,
            d_internal_nodes,
            index_a,  // path_index_a
            index_b,  // path_index_b
            num_leafs,
            merkle_height);
        
        // Check against target - optimized comparison
        // Most hashes will fail on the highest limb, so check it first
        bool is_solution = (final_hash.values[4] <= target.values[4]);
        if (is_solution && final_hash.values[4] == target.values[4]) {
            // Need to check lower limbs
            for (int i = 3; i >= 0; --i) {
                if (final_hash.values[i] > target.values[i]) {
                    is_solution = false;
                    break;
                }
                if (final_hash.values[i] < target.values[i]) {
                    break;
                }
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
                
                // Recompute paths for solution storage (solutions are rare, so this is fine)
                // Path A
                {
                    size_t running_index = index_a + num_leafs;
                    d_solution_path_a[0] = d_leafs[index_a ^ 1];
                    for (size_t level = 1; level < merkle_height; ++level) {
                        running_index >>= 1;
                        size_t sibling_index = running_index ^ 1;
                        d_solution_path_a[level] = (sibling_index < num_leafs) 
                            ? d_internal_nodes[sibling_index] 
                            : Digest::default_digest();
                    }
                }
                // Path B
                {
                    size_t running_index = index_b + num_leafs;
                    d_solution_path_b[0] = d_leafs[index_b ^ 1];
                    for (size_t level = 1; level < merkle_height; ++level) {
                        running_index >>= 1;
                        size_t sibling_index = running_index ^ 1;
                        d_solution_path_b[level] = (sibling_index < num_leafs) 
                            ? d_internal_nodes[sibling_index] 
                            : Digest::default_digest();
                    }
                }
                
                return;
            }
        }
    }
}

// ===== LOW-VRAM MINING KERNEL =====

__global__ void parallel_mining_kernel_low_vram(
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
    // OPTIMIZATION: Load with padding to reduce bank conflicts
    if (threadIdx.x < 256) {
        s_lookup_table[threadIdx.x] = LOOKUP_TABLE[threadIdx.x];
    }
    // Pad the last element to ensure proper alignment
    if (threadIdx.x == 0) {
        s_lookup_table[256] = 0;  // Padding byte
    }
    __syncthreads();
    
    if (*d_solution_found) return;
    
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    
    // Check solution flag less frequently to reduce global memory traffic and cache pollution
    // Increased from 64 to 4096 to improve sustained performance
    const uint64_t CHECK_INTERVAL = 4096;
    for (uint64_t idx = tid; idx < num_nonces; idx += stride) {
        // Early exit if solution found (check every N iterations for performance)
        if ((idx & (CHECK_INTERVAL - 1)) == 0 && *d_solution_found) return;
        
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
        
        // Compute final hash directly from global memory - no Pow struct or local arrays needed
        // This eliminates 2240 bytes of register pressure per thread (same as high VRAM version)
        // Root is at last index in sequentially-built tree
        const Digest* __restrict__ root_ptr = &d_internal_nodes[stored_nodes_count - 1];
        Digest merkle_root = *root_ptr;
        
        // Use optimized low VRAM version that computes nodes on-demand
        Digest final_hash = fast_mast_hash_direct_low_vram(
            mast_paths,
            nonce_digest,
            merkle_root,
            d_internal_nodes,
            path_index_a,  // path_index_a
            path_index_b,  // path_index_b
            num_leafs,
            merkle_height,
            stored_nodes_count,
            commitment,
            leaf_prefix);
        
        // Check against target - optimized comparison (low VRAM version)
        // Most hashes will fail on the highest limb, so check it first
        bool is_solution = (final_hash.values[4] <= target.values[4]);
        if (is_solution && final_hash.values[4] == target.values[4]) {
            // Need to check lower limbs
            for (int i = 3; i >= 0; --i) {
                if (final_hash.values[i] > target.values[i]) {
                    is_solution = false;
                    break;
                }
                if (final_hash.values[i] < target.values[i]) {
                    break;
                }
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
    
    threads_per_block = (g_block_size > 0) ? g_block_size : MINING_THREADS_PER_BLOCK;
    
    // Calculate optimal number of blocks
    // Note: Temporarily disabled due to CUDA 13.0 compatibility issue
    // int min_grid_size, optimal_block_size;
    // cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &optimal_block_size,
    //                                     parallel_mining_kernel_high_vram, 0, 0);
    
    // Use multiple of SM count for good occupancy
    // Architecture-specific tuning based on compute capability
    // SM 100/120 = Blackwell (RTX 5090), SM 89 = Ada (RTX 4090), SM 90 = Hopper
    int num_sms = prop.multiProcessorCount;
    int blocks_per_sm;
    if (prop.major >= 10) {
        // Blackwell architecture (RTX 5090) - use more blocks for better occupancy
        blocks_per_sm = 128;  // High value for maximum parallel blocks (capped by grid limits)
    } else if (prop.major == 9) {
        // Hopper architecture - use 6 blocks per SM
        blocks_per_sm = 6;
    } else {
        // Ampere/Ada - use default
        blocks_per_sm = 8;  // Increased from 4 to 8 for better performance
    }
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
    
    // Base batch size scaled by SM count and architecture
    // RTX 5090 (SM 120) has 192 SMs, ~21K CUDA cores
    uint64_t optimal;
    
    if (prop.major >= 10) {
        // Blackwell (RTX 5090) - larger batches for high SM count
        optimal = 80000000ULL; // 80M nonces for maximum GPU utilization
    } else if (prop.major == 9) {
        // Hopper - medium batch
        optimal = 40000000ULL;
    } else {
        // Ampere/Ada - default
        optimal = 30000000ULL;
    }
    
    // Apply bounds
    const uint64_t min_batch = 500000;
    const uint64_t max_batch = 100000000;  // Increased max for high-end GPUs
    
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
    
    // Ensure persistent mining resources are allocated (stream + output buffers)
    if (!buffer.ensure_mining_resources()) {
        LOG_DEBUG("mine_pow_with_buffer: failed to ensure mining resources");
        return std::nullopt;
    }
    
    // Reset the solution_found flag (async on stream)
    if (!buffer.reset_output_buffers()) {
        return std::nullopt;
    }
    
    // Get launch configuration
    int threads_per_block, blocks_per_grid;
    int gpu_id;
    cudaGetDevice(&gpu_id);
    calculate_mining_launch_config(max_nonces, threads_per_block, blocks_per_grid, gpu_id);
    
    // Calculate GPU's dedicated nonce range to avoid overlap with other GPUs
    // Only set once per buffer - the range doesn't change during mining
    if (!buffer.gpu_range_initialized) {
        int actual_gpu_count = g_total_gpu_count.load();
        GpuNonceRange gpu_range = calculate_gpu_range(gpu_id, actual_gpu_count);
        
        // Copy range to constant memory (avoids register pressure from extra parameters)
        cudaError_t range_err = cudaMemcpyToSymbol(d_gpu_range_start, &gpu_range.range_start, sizeof(uint64_t));
        if (range_err != cudaSuccess) {
            LOG_ERROR("copy range_start", range_err);
            return std::nullopt;
        }
        range_err = cudaMemcpyToSymbol(d_gpu_range_size, &gpu_range.range_size, sizeof(uint64_t));
        if (range_err != cudaSuccess) {
            LOG_ERROR("copy range_size", range_err);
            return std::nullopt;
        }
        
        // Initialize top tree cache (top 8 levels in constant memory)
        if (!initialize_top_tree_cache(buffer.d_merkle_tree, buffer.num_leafs)) {
            LOG_DEBUG("mine_pow_with_buffer: failed to initialize top tree cache");
            // Non-fatal - continue without cache
        }
        
        buffer.gpu_range_initialized = true;
    }
    
    // Select and launch appropriate kernel on the buffer's stream
    MiningKernelType kernel_type = select_mining_kernel(gpu_id);
    
    switch (kernel_type) {
        case MiningKernelType::HIGH_VRAM:
            parallel_mining_kernel_high_vram<<<blocks_per_grid, threads_per_block, 0, buffer.mining_stream>>>(
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
                buffer.d_solution_nonce,
                buffer.d_solution_found,
                buffer.d_solution_path_a,
                buffer.d_solution_path_b,
                buffer.d_solution_nonce_digest,
                buffer.d_solution_final_hash);
            break;
            
        case MiningKernelType::LOW_VRAM:
            parallel_mining_kernel_low_vram<<<blocks_per_grid, threads_per_block, 0, buffer.mining_stream>>>(
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
                buffer.d_solution_nonce,
                buffer.d_solution_found,
                buffer.d_solution_path_a,
                buffer.d_solution_path_b,
                buffer.d_solution_nonce_digest,
                buffer.d_solution_final_hash);
            break;
    }
    
    // Check for launch errors
    if (!check_kernel_launch_errors("mining_kernel")) {
        return std::nullopt;
    }
    
    // Check if solution was found
    // Note: cudaMemcpy implicitly synchronizes with the device, so no need for explicit sync
    int solution_found = 0;
    cudaError_t sync_err = cudaMemcpy(&solution_found, buffer.d_solution_found, sizeof(int), cudaMemcpyDeviceToHost);
    if (sync_err != cudaSuccess) {
        LOG_ERROR("mining_kernel memcpy sync", sync_err);
        return std::nullopt;
    }
    
    if (solution_found) {
        // Copy solution data back to host
        Pow solution;
        
        uint64_t nonce_value;
        cudaMemcpy(&nonce_value, buffer.d_solution_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(&solution.nonce, buffer.d_solution_nonce_digest, sizeof(Digest), cudaMemcpyDeviceToHost);
        cudaMemcpy(solution.path_a, buffer.d_solution_path_a, 
                   MERKLE_TREE_HEIGHT_ * sizeof(Digest), cudaMemcpyDeviceToHost);
        cudaMemcpy(solution.path_b, buffer.d_solution_path_b,
                   MERKLE_TREE_HEIGHT_ * sizeof(Digest), cudaMemcpyDeviceToHost);
        
        // Copy the final_hash that kernel computed (for debugging)
        Digest kernel_final_hash;
        cudaMemcpy(&kernel_final_hash, buffer.d_solution_final_hash, sizeof(Digest), cudaMemcpyDeviceToHost);
        
        // Set root from buffer
        solution.root = buffer.merkle_root;
        
        // Return both solution and kernel_final_hash
        MiningSolution mining_solution;
        mining_solution.pow = solution;
        mining_solution.kernel_final_hash = kernel_final_hash;
        
        return mining_solution;
    }
    
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