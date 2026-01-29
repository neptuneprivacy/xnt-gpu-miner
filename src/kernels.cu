#include "kernels.cuh"
#include "gpu_resources.cuh"
#include "common.cuh"

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
// Optimized: uses LUT-aware permutation (no per-call __syncthreads),
// streams MAST hash directly from Merkle lookups (eliminates ~6KB local memory),
// inlines index computation (avoids Digest temporaries).

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

    // Load lookup table into shared memory ONCE for the entire block
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        s_lookup_table[i] = LOOKUP_TABLE[i];
    }
    __syncthreads();

    if (*d_solution_found) return;

    const uint8_t* __restrict__ lut = s_lookup_table;
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;

    for (uint64_t idx = tid; idx < num_nonces; idx += stride) {
        if (*d_solution_found) return;

        uint64_t nonce_value = d_gpu_range_start + start_nonce + idx;

        // Nonce digest
        Digest nonce_digest;
        nonce_digest.values[0] = nonce_value;
        nonce_digest.values[1] = (nonce_value >> 32);
        nonce_digest.values[2] = 0;
        nonce_digest.values[3] = 0;
        nonce_digest.values[4] = 0;

        // === Inline index computation (63 permutations) ===
        uint64_t state[STATE_SIZE];
        // tip5_hash_fixed(hash, nonce_digest): init FixedLength sponge
        #pragma unroll
        for (int i = RATE; i < STATE_SIZE; ++i) state[i] = BFE_ONE; // FixedLength capacity
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) {
            state[i] = hash.values[i];
            state[i + DIGEST_LEN] = nonce_digest.values[i];
        }
        tip5_permutation_lut(state, lut);

        // 62 more iterations of hash_fixed_right_zero
        // Precompute x^7(BFE_ONE) once for the specialized first-round optimization
        const uint64_t x7_one = x7_computer(BFE_ONE);
        for (uint32_t iter = 1; iter < NUM_INDEX_REPETITIONS; ++iter) {
            // Keep state[0..4] (result from previous), zero right half, reset capacity
            state[5] = 0; state[6] = 0; state[7] = 0; state[8] = 0; state[9] = 0;
            state[10] = BFE_ONE; state[11] = BFE_ONE; state[12] = BFE_ONE;
            state[13] = BFE_ONE; state[14] = BFE_ONE; state[15] = BFE_ONE;
            tip5_permutation_lut_index(state, lut, x7_one);
        }

        uint64_t index_a = state[0] & MERKLE_INDEX_MASK;
        uint64_t index_b = state[1] & MERKLE_INDEX_MASK;

        // === Streaming MAST hash ===
        // Instead of storing path_a[27], path_b[27], Pow, encoding[280] (~6KB),
        // stream Merkle path data directly into the sponge as we read it.
        //
        // Encoding order: nonce(5) + path_b[0..26](135) + path_a[0..26](135) + root(5) = 280 words
        // RATE = 10 = 2 * DIGEST_LEN, so every 2 Digests fills one sponge block.
        // 280 / 10 = 28 full blocks -> 28 permutations, then 1 padding -> 29 total.

        // Re-init state for VariableLength sponge
        #pragma unroll
        for (int i = 0; i < STATE_SIZE; ++i) state[i] = BFE_ZERO;

        // --- Block 0: nonce + path_b[0] ---
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) state[i] = nonce_digest.values[i];
        {
            Digest sib = d_leafs[index_b ^ 1];
            #pragma unroll
            for (int i = 0; i < DIGEST_LEN; ++i) state[DIGEST_LEN + i] = sib.values[i];
        }
        tip5_permutation_lut(state, lut);

        // --- Blocks 1-13: path_b[1..26] ---
        {
            size_t running_b = (index_b + num_leafs) >> 1;
            for (size_t level = 1; level < merkle_height; ++level) {
                size_t sibling_b = running_b ^ 1;
                Digest node;
                if (sibling_b < MERKLE_NUM_LEAFS) {
                    node = d_internal_nodes[sibling_b];
                } else {
                    node = Digest::default_digest();
                }

                if (level & 1) {
                    // Odd level -> first half
                    #pragma unroll
                    for (int i = 0; i < DIGEST_LEN; ++i) state[i] = node.values[i];
                } else {
                    // Even level -> second half -> permute
                    #pragma unroll
                    for (int i = 0; i < DIGEST_LEN; ++i) state[DIGEST_LEN + i] = node.values[i];
                    tip5_permutation_lut(state, lut);
                }

                running_b >>= 1;
            }
        }

        // --- Blocks 14-27: path_a[0..26] + root ---
        {
            // path_a[0] = sibling leaf -> first half
            Digest sib_a = d_leafs[index_a ^ 1];
            #pragma unroll
            for (int i = 0; i < DIGEST_LEN; ++i) state[i] = sib_a.values[i];

            size_t running_a = (index_a + num_leafs) >> 1;
            for (size_t level = 1; level < merkle_height; ++level) {
                size_t sibling_a = running_a ^ 1;
                Digest node;
                if (sibling_a < MERKLE_NUM_LEAFS) {
                    node = d_internal_nodes[sibling_a];
                } else {
                    node = Digest::default_digest();
                }

                if (level & 1) {
                    // Odd level -> second half -> permute
                    #pragma unroll
                    for (int i = 0; i < DIGEST_LEN; ++i) state[DIGEST_LEN + i] = node.values[i];
                    tip5_permutation_lut(state, lut);
                } else {
                    // Even level -> first half
                    #pragma unroll
                    for (int i = 0; i < DIGEST_LEN; ++i) state[i] = node.values[i];
                }

                running_a >>= 1;
            }
        }

        // --- Root (completes block 28) ---
        {
            Digest merkle_root = d_internal_nodes[num_leafs - 2];
            #pragma unroll
            for (int i = 0; i < DIGEST_LEN; ++i) state[DIGEST_LEN + i] = merkle_root.values[i];
        }
        tip5_permutation_lut(state, lut);

        // --- Padding block (varlen finalization) ---
        state[0] = BFE_ONE;
        #pragma unroll
        for (int i = 1; i < RATE; ++i) state[i] = BFE_ZERO;
        tip5_permutation_lut(state, lut);
        // state[0..4] = pow_encoding_digest (29 permutations done)

        // === Continue MAST hash tree ===
        // header_mast_hash = hash_fixed(pow_encoding_digest, mast_paths.pow[0])
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) state[DIGEST_LEN + i] = mast_paths.pow[0].values[i];
        #pragma unroll
        for (int i = RATE; i < STATE_SIZE; ++i) state[i] = BFE_ONE; // FixedLength capacity
        tip5_permutation_lut(state, lut);

        // header_mast_hash = hash_fixed(result, mast_paths.pow[1])
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) state[DIGEST_LEN + i] = mast_paths.pow[1].values[i];
        #pragma unroll
        for (int i = RATE; i < STATE_SIZE; ++i) state[i] = BFE_ONE;
        tip5_permutation_lut(state, lut);

        // header_mast_hash = hash_fixed(mast_paths.pow[2], result)
        // NOTE: pow[2] is LEFT, result is RIGHT
        {
            uint64_t tmp[DIGEST_LEN];
            #pragma unroll
            for (int i = 0; i < DIGEST_LEN; ++i) tmp[i] = state[i];
            #pragma unroll
            for (int i = 0; i < DIGEST_LEN; ++i) {
                state[i] = mast_paths.pow[2].values[i];
                state[DIGEST_LEN + i] = tmp[i];
            }
        }
        #pragma unroll
        for (int i = RATE; i < STATE_SIZE; ++i) state[i] = BFE_ONE;
        tip5_permutation_lut(state, lut);
        // state[0..4] = header_mast_hash

        // kernel_mast_hash = hash_fixed(hash_varlen(header_mast_hash, 5), header[0])
        // First: hash_varlen of 5 words (VariableLength sponge, 1 block + padding)
        #pragma unroll
        for (int i = RATE; i < STATE_SIZE; ++i) state[i] = BFE_ZERO; // VariableLength
        // state[0..4] already has header_mast_hash
        state[DIGEST_LEN] = BFE_ONE; // padding at position 5
        #pragma unroll
        for (int i = DIGEST_LEN + 1; i < RATE; ++i) state[i] = BFE_ZERO;
        tip5_permutation_lut(state, lut);

        // hash_fixed(result, header[0])
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) state[DIGEST_LEN + i] = mast_paths.header[0].values[i];
        #pragma unroll
        for (int i = RATE; i < STATE_SIZE; ++i) state[i] = BFE_ONE;
        tip5_permutation_lut(state, lut);

        // hash_fixed(result, header[1])
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) state[DIGEST_LEN + i] = mast_paths.header[1].values[i];
        #pragma unroll
        for (int i = RATE; i < STATE_SIZE; ++i) state[i] = BFE_ONE;
        tip5_permutation_lut(state, lut);
        // state[0..4] = kernel_mast_hash

        // Final: hash_fixed(hash_varlen(kernel_mast_hash, 5), kernel[0])
        // hash_varlen of 5 words
        #pragma unroll
        for (int i = RATE; i < STATE_SIZE; ++i) state[i] = BFE_ZERO;
        state[DIGEST_LEN] = BFE_ONE;
        #pragma unroll
        for (int i = DIGEST_LEN + 1; i < RATE; ++i) state[i] = BFE_ZERO;
        tip5_permutation_lut(state, lut);

        // hash_fixed(result, kernel[0])
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) state[DIGEST_LEN + i] = mast_paths.kernel[0].values[i];
        #pragma unroll
        for (int i = RATE; i < STATE_SIZE; ++i) state[i] = BFE_ONE;
        tip5_permutation_lut(state, lut);
        // state[0..4] = final_hash

        // === Target comparison ===
        bool is_solution = true;
        #pragma unroll
        for (int i = DIGEST_LEN - 1; i >= 0; --i) {
            if (state[i] > target.values[i]) {
                is_solution = false;
                break;
            }
            if (state[i] < target.values[i]) {
                break;
            }
        }

        if (is_solution) {
            int was = atomicCAS(d_solution_found, 0, 1);
            if (was == 0) {
                atomicExch((unsigned long long*)d_solution_nonce, nonce_digest.values[0]);
                *d_solution_nonce_digest = nonce_digest;

                // Re-read paths from the Merkle tree for the solution output.
                // This only happens on solution found (extremely rare), so cost is negligible.
                {
                    size_t running_a = index_a + num_leafs;
                    d_solution_path_a[0] = d_leafs[index_a ^ 1];
                    for (size_t level = 1; level < merkle_height; ++level) {
                        running_a >>= 1;
                        size_t sibling_a = running_a ^ 1;
                        d_solution_path_a[level] = (sibling_a < MERKLE_NUM_LEAFS) ?
                            d_internal_nodes[sibling_a] : Digest::default_digest();
                    }
                }
                {
                    size_t running_b = index_b + num_leafs;
                    d_solution_path_b[0] = d_leafs[index_b ^ 1];
                    for (size_t level = 1; level < merkle_height; ++level) {
                        running_b >>= 1;
                        size_t sibling_b = running_b ^ 1;
                        d_solution_path_b[level] = (sibling_b < MERKLE_NUM_LEAFS) ?
                            d_internal_nodes[sibling_b] : Digest::default_digest();
                    }
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
    
    // Larger batches amortize launch overhead and improve GPU utilization
    uint64_t optimal = 4000000ULL;
    
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
                output.d_solution_nonce_digest);
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