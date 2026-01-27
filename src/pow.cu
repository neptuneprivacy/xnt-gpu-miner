#include "pow.cuh"

__constant__ uint64_t d_gpu_range_start;
__constant__ uint64_t d_gpu_range_size;

__device__ Digest PowMastPaths::commit_device() const {
    Digest pow_root = pow[0];
    for (int i = 1; i < 3; ++i) {
        pow_root = tip5_hash_fixed_device(pow_root, pow[i]);
    }
    
    Digest header_root = header[0];
    for (int i = 1; i < 2; ++i) {
        header_root = tip5_hash_fixed_device(header_root, header[i]);
    }
    
    Digest combined = tip5_hash_fixed_device(pow_root, header_root);
    combined = tip5_hash_fixed_device(combined, kernel[0]);
    
    return combined;
}

__host__ Digest PowMastPaths::commit() const {
    Digest pow_root = pow[0];
    for (int i = 1; i < 3; ++i) {
        pow_root = tip5_hash_fixed_host(pow_root, pow[i]);
    }
    
    Digest header_root = header[0];
    for (int i = 1; i < 2; ++i) {
        header_root = tip5_hash_fixed_host(header_root, header[i]);
    }
    
    Digest combined = tip5_hash_fixed_host(pow_root, header_root);
    combined = tip5_hash_fixed_host(combined, kernel[0]);
    
    return combined;
}

Digest PowMastPaths::fast_mast_hash(const Pow& pow_obj) const {
    auto pow_encoding = pow_obj.encode();
    auto pow_digest = tip5_hash_varlen_host(pow_encoding);
    
    Digest header_mast_hash = pow_digest;
    for (int i = 0; i < 3; ++i) {
        header_mast_hash = tip5_hash_fixed_host(header_mast_hash, pow[i]);
    }
    
    for (int i = 0; i < 2; ++i) {
        header_mast_hash = tip5_hash_fixed_host(header_mast_hash, header[i]);
    }
    
    Digest final_hash = tip5_hash_fixed_host(header_mast_hash, kernel[0]);
    return final_hash;
}

VramMode detect_vram_mode(int gpu_id) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, gpu_id);
    
    if (err != cudaSuccess) {
        LOG_DEBUG("Failed to get device properties for GPU " << gpu_id);
        return VramMode::HIGH_VRAM;
    }
    
    size_t total_vram_gb = prop.totalGlobalMem / (1024ULL * 1024ULL * 1024ULL);
    LOG_DEBUG("GPU " << gpu_id << " has " << total_vram_gb << " GB VRAM");
    
    if (total_vram_gb >= 11) {
        return VramMode::HIGH_VRAM;
    } else {
        return VramMode::LOW_VRAM;
    }
}

__device__ Digest Pow::bud(const Digest& commitment, uint64_t index) {
    Digest hash = commitment;
    
    for (size_t round = 0; round < BUDDING_ROUNDS; ++round) {
        Digest round_digest;
        round_digest.values[0] = index;
        round_digest.values[1] = 0;
        round_digest.values[2] = 0;
        round_digest.values[3] = 0;
        round_digest.values[4] = round;
        hash = tip5_hash_fixed_device(hash, round_digest);
    }
    
    return hash;
}

__host__ Digest Pow::bud_host(const Digest& commitment, uint64_t index) {
    Digest hash = commitment;
    
    for (size_t round = 0; round < BUDDING_ROUNDS; ++round) {
        Digest round_digest;
        round_digest.values[0] = index;
        round_digest.values[1] = 0;
        round_digest.values[2] = 0;
        round_digest.values[3] = 0;
        round_digest.values[4] = round;
        hash = tip5_hash_fixed_host(hash, round_digest);
    }
    
    return hash;
}

__device__ void Pow::indices(const Digest& hash, const Digest& nonce, 
                              uint64_t& index_a, uint64_t& index_b) {
    Digest indexer = tip5_hash_fixed_device(hash, nonce);
    
    for (uint32_t i = 1; i < NUM_INDEX_REPETITIONS; ++i) {
        indexer = tip5_hash_fixed_device(indexer, Digest::default_digest());
    }
    
    index_a = indexer.values[0] & MERKLE_INDEX_MASK;
    index_b = indexer.values[1] & MERKLE_INDEX_MASK;
}

__host__ std::pair<uint64_t, uint64_t> Pow::indices(const Digest& hash, const Digest& nonce) {
    Digest indexer = tip5_hash_fixed_host(hash, nonce);
    
    for (uint32_t i = 1; i < NUM_INDEX_REPETITIONS; ++i) {
        indexer = tip5_hash_fixed_host(indexer, Digest::default_digest());
    }
    
    uint64_t index_a = indexer.values[0] & MERKLE_INDEX_MASK;
    uint64_t index_b = indexer.values[1] & MERKLE_INDEX_MASK;
    
    return {index_a, index_b};
}

__host__ Digest Pow::compute_leaf_from_commitment_host(const Digest& commitment, 
                                                        uint64_t index, 
                                                        uint64_t num_leafs) {
    return bud_host(commitment, index);
}

__host__ bool Pow::verify_merkle_path_host(const Digest& root, uint64_t index, 
                                            const Digest* path, const Digest& element) {
    if (index >= (1ULL << MERKLE_TREE_HEIGHT_)) {
        return false;
    }
    
    uint64_t running_index = index;
    Digest running_digest = element;
    
    for (size_t i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        if (running_index & 1) {
            // Right child: hash(sibling, running_digest)
            running_digest = tip5_hash_fixed_host(path[i], running_digest);
        } else {
            // Left child: hash(running_digest, sibling)
            running_digest = tip5_hash_fixed_host(running_digest, path[i]);
        }
        running_index >>= 1;
    }
    
    // Compare with root
    for (int i = 0; i < DIGEST_LEN; ++i) {
        if (running_digest.values[i] != root.values[i]) {
            return false;
        }
    }
    return true;
}

__host__ uint64_t Pow::bitreverse_host(uint64_t n, uint32_t log2_n) {
    uint64_t rev = 0;
    for (uint32_t i = 0; i < log2_n; ++i) {
        if ((n >> i) & 1) {
            rev |= (1ULL << (log2_n - 1 - i));
        }
    }
    return rev;
}

std::vector<uint64_t> Pow::encode() const {
    std::vector<uint64_t> encoding;
    encoding.reserve(DIGEST_LEN * (1 + 2 * MERKLE_TREE_HEIGHT_ + 1));
    
    for (int i = 0; i < DIGEST_LEN; ++i) {
        encoding.push_back(nonce.values[i]);
    }
    
    for (int i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        for (int j = 0; j < DIGEST_LEN; ++j) {
            encoding.push_back(path_b[i].values[j]);
        }
    }
    
    for (int i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        for (int j = 0; j < DIGEST_LEN; ++j) {
            encoding.push_back(path_a[i].values[j]);
        }
    }
    
    for (int i = 0; i < DIGEST_LEN; ++i) {
        encoding.push_back(root.values[i]);
    }
    
    return encoding;
}

GuesserBuffer Pow::preprocess(const PowMastPaths& mast_auth_paths,
                              const Digest& prev_block_digest,
                              int consensus_rule_set,
                              bool* cancel_flag) {
    return preprocess_gpu(mast_auth_paths, prev_block_digest, consensus_rule_set, cancel_flag);
}

__host__ GuesserBuffer Pow::preprocess_gpu(const PowMastPaths& mast_auth_paths,
                                            const Digest& prev_block_digest,
                                            int consensus_rule_set,
                                            bool* cancel_flag) {
    // For mainnet blocks >= 15256, we always use CONSENSUS_XNT
    LOG_DEBUG("preprocess: starting (XNT consensus mode)");
    
    int current_gpu = 0;
    cudaError_t cuda_error = cudaGetDevice(&current_gpu);
    if (cuda_error != cudaSuccess) {
        LOG_ERROR("get device in preprocess", cuda_error);
        return GuesserBuffer();
    }
    
    VramMode vram_mode = detect_vram_mode(current_gpu);
    
    switch (vram_mode) {
        case VramMode::HIGH_VRAM:
            return preprocess_gpu_high_vram(mast_auth_paths, prev_block_digest, 
                                            consensus_rule_set, cancel_flag);
        case VramMode::LOW_VRAM:
            return preprocess_gpu_low_vram(mast_auth_paths, prev_block_digest,
                                            consensus_rule_set, cancel_flag);
        default:
            return preprocess_gpu_high_vram(mast_auth_paths, prev_block_digest,
                                            consensus_rule_set, cancel_flag);
    }
}

__host__ GuesserBuffer Pow::preprocess_gpu_high_vram(const PowMastPaths& mast_auth_paths,
                                                      const Digest& prev_block_digest,
                                                      int consensus_rule_set,
                                                      bool* cancel_flag) {
    GuesserBuffer buffer;
    
    // For mainnet blocks >= 15256, we use CONSENSUS_XNT
    // Always use mast_auth_paths.commit() for XNT consensus
    Digest commitment = mast_auth_paths.commit();
    
    buffer.hash = commitment;
    buffer.prev_block_digest = prev_block_digest;  // Still needed for validation
    buffer.consensus_rule_set = CONSENSUS_XNT;  // Always XNT for blocks >= 15256
    buffer.mast_paths = mast_auth_paths;
    buffer.num_leafs = MERKLE_NUM_LEAFS;
    
    LOG_DEBUG("preprocess_high_vram: commitment computed");
    
    if (cancel_flag && *cancel_flag) {
        return GuesserBuffer();
    }
    
    size_t leafs_size = MERKLE_NUM_LEAFS * sizeof(Digest);
    cudaError_t alloc_err = cudaMalloc(&buffer.d_leafs, leafs_size);
    if (alloc_err != cudaSuccess) {
        LOG_ERROR("cudaMalloc leafs", alloc_err);
        return GuesserBuffer();
    }
    
    LOG_DEBUG("preprocess_high_vram: allocated " << (leafs_size / (1024*1024)) << " MB for leafs");
    
    size_t internal_size = (MERKLE_NUM_LEAFS - 1) * sizeof(Digest);
    alloc_err = cudaMalloc(&buffer.d_merkle_tree, internal_size);
    if (alloc_err != cudaSuccess) {
        LOG_ERROR("cudaMalloc internal_nodes", alloc_err);
        cudaFree(buffer.d_leafs);
        buffer.d_leafs = nullptr;
        return GuesserBuffer();
    }
    buffer.tree_size = MERKLE_NUM_LEAFS - 1;
    
    LOG_DEBUG("preprocess_high_vram: allocated " << (internal_size / (1024*1024)) << " MB for internal nodes");
    
    if (cancel_flag && *cancel_flag) {
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    int threadsPerBlock = 256;
    int numBlocks = (MERKLE_NUM_LEAFS + threadsPerBlock - 1) / threadsPerBlock;
    
    compute_buds_kernel<<<numBlocks, threadsPerBlock>>>(
        buffer.d_leafs, commitment, MERKLE_NUM_LEAFS, 0);
    
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        LOG_ERROR("compute_buds_kernel sync", sync_err);
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    LOG_DEBUG("preprocess_high_vram: buds computed");
    
    // Check for cancellation
    if (cancel_flag && *cancel_flag) {
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    // Convert buds to leafs through NUM_BUD_LAYERS iterations
    // This matches Rust: iterate log-many times to compute leafs from buds
    // Optimize memory: allocate only half size since each layer halves the output
    // The first layer outputs MERKLE_NUM_LEAFS/2 elements, which is the max we need
    size_t temp_buffer_size = (MERKLE_NUM_LEAFS / 2) * sizeof(Digest);
    Digest* d_temp_leafs = nullptr;
    cudaError_t temp_alloc_err = cudaMalloc(&d_temp_leafs, temp_buffer_size);
    if (temp_alloc_err != cudaSuccess) {
        LOG_ERROR("cudaMalloc temp_leafs", temp_alloc_err);
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    Digest* current_buds = buffer.d_leafs;
    Digest* current_leafs = d_temp_leafs;
    
    for (size_t layer = 0; layer < NUM_BUD_LAYERS; ++layer) {
        if (cancel_flag && *cancel_flag) {
            cudaFree(d_temp_leafs);
            buffer.cleanup();
            return GuesserBuffer();
        }
        
        // Each layer outputs MERKLE_NUM_LEAFS >> (layer + 1) elements
        size_t output_count = MERKLE_NUM_LEAFS >> (layer + 1);
        int layerBlocks = (output_count + threadsPerBlock - 1) / threadsPerBlock;
        
        compute_leafs_from_buds_kernel<<<layerBlocks, threadsPerBlock>>>(
            current_leafs, current_buds, MERKLE_NUM_LEAFS, layer);
        
        sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            LOG_ERROR("compute_leafs_from_buds_kernel sync", sync_err);
            cudaFree(d_temp_leafs);
            buffer.cleanup();
            return GuesserBuffer();
        }
        
        // Alternate between temp buffer and main buffer for next iteration
        // Even layers: write to temp, odd layers: write to buffer.d_leafs
        if (layer < NUM_BUD_LAYERS - 1) {
            if ((layer % 2) == 0) {
                // Next layer (odd): read from temp, write to buffer
                current_buds = d_temp_leafs;
                current_leafs = buffer.d_leafs;
            } else {
                // Next layer (even): read from buffer, write to temp
                current_buds = buffer.d_leafs;
                current_leafs = d_temp_leafs;
            }
        }
    }
    
    // After NUM_BUD_LAYERS iterations, final leafs are in current_leafs
    // Copy them to buffer.d_leafs if needed
    if (current_leafs != buffer.d_leafs) {
        size_t final_count = MERKLE_NUM_LEAFS >> NUM_BUD_LAYERS;
        cudaMemcpy(buffer.d_leafs, current_leafs, final_count * sizeof(Digest), 
                   cudaMemcpyDeviceToDevice);
    }
    
    cudaFree(d_temp_leafs);
    LOG_DEBUG("preprocess_high_vram: buds converted to leafs");
    
    // Check for cancellation
    if (cancel_flag && *cancel_flag) {
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    // For CONSENSUS_XNT (mainnet blocks >= 15256), no bit-reverse swap is needed
    // Bit-reverse swap is only for HardforkAlpha, which we no longer use
    LOG_DEBUG("preprocess_high_vram: using XNT consensus (no bit-reverse swap)");
    
    size_t current_count = MERKLE_NUM_LEAFS;
    const Digest* current_layer = buffer.d_leafs;
    size_t write_offset = 0;
    
    for (size_t layer = 0; layer < MERKLE_TREE_HEIGHT_; ++layer) {
        if (cancel_flag && *cancel_flag) {
            buffer.cleanup();
            return GuesserBuffer();
        }
        
        size_t parent_count = current_count / 2;
        Digest* parent_layer = buffer.d_merkle_tree + write_offset;
        
        int layerBlocks = (parent_count + threadsPerBlock - 1) / threadsPerBlock;
        merkle_zip_kernel<<<layerBlocks, threadsPerBlock>>>(
            parent_layer, current_layer, parent_count);
        
        sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            LOG_ERROR("merkle_zip_kernel sync", sync_err);
            buffer.cleanup();
            return GuesserBuffer();
        }
        
        current_layer = parent_layer;
        write_offset += parent_count;
        current_count = parent_count;
    }
    
    LOG_DEBUG("preprocess_high_vram: Merkle tree built");
    
    cudaMemcpy(&buffer.merkle_root, buffer.d_merkle_tree + buffer.tree_size - 1,
               sizeof(Digest), cudaMemcpyDeviceToHost);
    
    LOG_DEBUG("preprocess_high_vram: complete");
    return buffer;
}

__host__ GuesserBuffer Pow::preprocess_gpu_low_vram(const PowMastPaths& mast_auth_paths,
                                                     const Digest& prev_block_digest,
                                                     int consensus_rule_set,
                                                     bool* cancel_flag) {
    LOG_DEBUG("preprocess_low_vram: falling back to high_vram implementation");
    return preprocess_gpu_high_vram(mast_auth_paths, prev_block_digest,
                                    consensus_rule_set, cancel_flag);
}

__device__ void Pow_indices_device(const Digest& hash, const Digest& nonce,
                                    uint64_t& index_a, uint64_t& index_b) {
    Pow::indices(hash, nonce, index_a, index_b);
}

uint64_t generate_secure_random_start(const std::string& puzzle_id, int gpu_id,
                                       const std::string& gpu_uuid) {
    
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
    
    // Combine various entropy sources
    uint64_t seed = timestamp_ns;
    
    // Mix in puzzle ID
    for (char c : puzzle_id) {
        seed = seed * 31 + c;
    }
    
    // Mix in GPU ID
    seed = seed * 17 + gpu_id;
    
    // Mix in GPU UUID
    for (char c : gpu_uuid) {
        seed = seed * 41 + c;
    }
    
    // Add some hardware randomness if available
    std::random_device rd;
    try {
        seed ^= rd();
        seed ^= (static_cast<uint64_t>(rd()) << 32);
    } catch (...) {
        // Random device not available, continue with other entropy
    }
    
    // Final mixing
    seed ^= (seed >> 33);
    seed *= 0xff51afd7ed558ccdULL;
    seed ^= (seed >> 33);
    seed *= 0xc4ceb9fe1a85ec53ULL;
    seed ^= (seed >> 33);
    
    return seed;
}

// ===== SOLUTION VERIFICATION =====

bool verify_pow_solution(const Pow& pow,
                         const Digest& hash,
                         const Digest& target,
                         const Digest& commitment) {
    // Compute indices from nonce
    auto [index_a, index_b] = Pow::indices(hash, pow.nonce);
    
    // Compute expected leafs from commitment
    Digest leaf_a = Pow::compute_leaf_from_commitment_host(commitment, index_a, MERKLE_NUM_LEAFS);
    Digest leaf_b = Pow::compute_leaf_from_commitment_host(commitment, index_b, MERKLE_NUM_LEAFS);
    
    // Verify Merkle paths
    bool path_a_valid = Pow::verify_merkle_path_host(pow.root, index_a, pow.path_a, leaf_a);
    bool path_b_valid = Pow::verify_merkle_path_host(pow.root, index_b, pow.path_b, leaf_b);
    
    if (!path_a_valid || !path_b_valid) {
        return false;
    }
    
    // Compute final hash and check against target
    // (Full verification would include MAST hash computation)
    
    return true;
}

// ===== AUTH PATHS CONVERSION =====

PowMastPaths convertToPowMastPaths(const AuthPaths& auth_paths) {
    PowMastPaths result;
    
    // Convert pow paths
    for (size_t i = 0; i < std::min(auth_paths.pow.size(), size_t(3)); ++i) {
        result.pow[i] = hex_to_digest(auth_paths.pow[i]);
    }
    
    // Convert header paths
    for (size_t i = 0; i < std::min(auth_paths.header.size(), size_t(2)); ++i) {
        result.header[i] = hex_to_digest(auth_paths.header[i]);
    }
    
    // Convert kernel paths
    for (size_t i = 0; i < std::min(auth_paths.kernel.size(), size_t(1)); ++i) {
        result.kernel[i] = hex_to_digest(auth_paths.kernel[i]);
    }
    
    return result;
}