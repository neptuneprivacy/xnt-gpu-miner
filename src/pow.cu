#include "pow.cuh"

// d_gpu_range_start and d_gpu_range_size are defined via extern in tip5.cuh
// CUDA 13+ treats extern __constant__ as static definition

__device__ Digest PowMastPaths::commit_device() const {
    // Match Rust: Tip5::hash_varlen over flattened pow, header, kernel digests
    uint64_t values[DIGEST_LEN * 6];
    size_t pos = 0;
    
    for (int i = 0; i < 3; ++i) {
        #pragma unroll
        for (int j = 0; j < DIGEST_LEN; ++j) {
            values[pos++] = pow[i].values[j];
        }
    }
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < DIGEST_LEN; ++j) {
            values[pos++] = header[i].values[j];
        }
    }
    #pragma unroll
    for (int j = 0; j < DIGEST_LEN; ++j) {
        values[pos++] = kernel[0].values[j];
    }
    
    return tip5_hash_varlen_device(values, pos);
}

__host__ Digest PowMastPaths::commit() const {
    // Match Rust: Tip5::hash_varlen over flattened pow, header, kernel digests
    std::vector<uint64_t> values;
    values.reserve(DIGEST_LEN * 6);
    
    auto append_digest = [&values](const Digest& d) {
        for (int i = 0; i < DIGEST_LEN; ++i) {
            values.push_back(d.values[i]);
        }
    };
    
    for (int i = 0; i < 3; ++i) {
        append_digest(pow[i]);
    }
    for (int i = 0; i < 2; ++i) {
        append_digest(header[i]);
    }
    append_digest(kernel[0]);
    
    return tip5_hash_varlen_host(values);
}

Digest PowMastPaths::fast_mast_hash(const Pow& pow_obj) const {
    // 39 permutations to get the block hash, when Merkle tree height is 20
    auto pow_encoding = pow_obj.encode();
    auto header_mast_hash = tip5_hash_fixed_host(tip5_hash_varlen_host(pow_encoding), this->pow[0]);
    header_mast_hash = tip5_hash_fixed_host(header_mast_hash, this->pow[1]);
    header_mast_hash = tip5_hash_fixed_host(this->pow[2], header_mast_hash);
    
    // Convert header_mast_hash to vector for varlen hash
    std::vector<uint64_t> header_encoding;
    for (const auto& val : header_mast_hash.values) {
        header_encoding.push_back(val);
    }
    auto kernel_mast_hash = tip5_hash_fixed_host(tip5_hash_varlen_host(header_encoding), this->header[0]);
    kernel_mast_hash = tip5_hash_fixed_host(kernel_mast_hash, this->header[1]);
    
    // Convert kernel_mast_hash to vector for varlen hash
    std::vector<uint64_t> kernel_encoding;
    for (const auto& val : kernel_mast_hash.values) {
        kernel_encoding.push_back(val);
    }
    return tip5_hash_fixed_host(tip5_hash_varlen_host(kernel_encoding), this->kernel[0]);
}

__device__ Digest PowMastPaths::fast_mast_hash_device(const Pow& pow_obj) const {
    // EXACT CPU LOGIC: Compute the encoding manually (correct size)
    constexpr size_t POW_ENCODING_WORDS = 5 + 2 * MERKLE_TREE_HEIGHT_ * 5 + 5;
    uint64_t encoding[POW_ENCODING_WORDS]; // nonce + paths + root
    int idx = 0;
    
    // Add nonce values first
    for (int i = 0; i < DIGEST_LEN; ++i) {
        if (idx < sizeof(encoding)/sizeof(encoding[0])) {
            encoding[idx++] = pow_obj.nonce.values[i];
        }
    }
    
    // Add path_b values
    for (int i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        for (int j = 0; j < DIGEST_LEN; ++j) {
            if (idx < sizeof(encoding)/sizeof(encoding[0])) {
                encoding[idx++] = pow_obj.path_b[i].values[j];
            }
        }
    }

    // Add path_a values
    for (int i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        for (int j = 0; j < DIGEST_LEN; ++j) {
            if (idx < sizeof(encoding)/sizeof(encoding[0])) {
                encoding[idx++] = pow_obj.path_a[i].values[j];
            }
        }
    }
    
    // Add root values
    for (int i = 0; i < DIGEST_LEN; ++i) {
        if (idx < sizeof(encoding)/sizeof(encoding[0])) {
            encoding[idx++] = pow_obj.root.values[i];
        }
    }
    
    // Now compute the fast mast hash exactly like CPU version
    auto pow_encoding_digest = tip5_hash_varlen_device(encoding, idx);
    auto header_mast_hash = tip5_hash_fixed_device(pow_encoding_digest, pow[0]);
    header_mast_hash = tip5_hash_fixed_device(header_mast_hash, pow[1]);
    header_mast_hash = tip5_hash_fixed_device(pow[2], header_mast_hash);
    
    // Convert header_mast_hash to array for varlen hash
    uint64_t header_encoding[5];
    for (int i = 0; i < DIGEST_LEN; ++i) {
        header_encoding[i] = header_mast_hash.values[i];
    }
    auto kernel_mast_hash = tip5_hash_fixed_device(tip5_hash_varlen_device(header_encoding, DIGEST_LEN), header[0]);
    kernel_mast_hash = tip5_hash_fixed_device(kernel_mast_hash, header[1]);
    
    // Convert kernel_mast_hash to array for varlen hash
    uint64_t kernel_encoding[5];
    for (int i = 0; i < DIGEST_LEN; ++i) {
        kernel_encoding[i] = kernel_mast_hash.values[i];
    }
    return tip5_hash_fixed_device(tip5_hash_varlen_device(kernel_encoding, DIGEST_LEN), kernel[0]);
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
    
    // Use fast right-zero hash variant for the bulk of repetitions
    for (uint32_t i = 1; i < NUM_INDEX_REPETITIONS; ++i) {
        indexer = tip5_hash_fixed_right_zero_device(indexer);
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
    // Match Rust: output size stays NUM_LEAFS every layer
    size_t temp_buffer_size = MERKLE_NUM_LEAFS * sizeof(Digest);
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
        
        // Each layer outputs MERKLE_NUM_LEAFS elements
        int layerBlocks = (MERKLE_NUM_LEAFS + threadsPerBlock - 1) / threadsPerBlock;
        
        compute_leafs_from_buds_kernel<<<layerBlocks, threadsPerBlock>>>(
            current_leafs, current_buds, MERKLE_NUM_LEAFS, layer);
        
        sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            LOG_ERROR("compute_leafs_from_buds_kernel sync", sync_err);
            cudaFree(d_temp_leafs);
            buffer.cleanup();
            return GuesserBuffer();
        }
        
        // Swap for next iteration
        std::swap(current_buds, current_leafs);
    }
    
    // After NUM_BUD_LAYERS iterations, final leafs are in current_buds
    // Copy them to buffer.d_leafs if needed
    if (current_buds != buffer.d_leafs) {
        cudaMemcpy(buffer.d_leafs, current_buds, temp_buffer_size,
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
    
    // Precompute index picker preimage: Tip5::hash_pair(root, mast_auth_paths.commit())
    buffer.index_picker_preimage = tip5_hash_fixed_host(buffer.merkle_root, commitment);
    
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

__device__ void Pow_indices_device(const Digest& hash, const Digest& nonce, uint64_t& index_a, uint64_t& index_b) {
    Digest indexer = tip5_hash_fixed_device(hash, nonce);
    // Use fast right-zero hash variant for the bulk of repetitions
    for (uint32_t i = 1; i < NUM_INDEX_REPETITIONS; ++i) {
        indexer = tip5_hash_fixed_right_zero_device(indexer);
    }

    index_a = indexer.values[0] & MERKLE_INDEX_MASK;
    index_b = indexer.values[1] & MERKLE_INDEX_MASK;
}

// Mining pool configuration
static int g_miner_id = 0;           // Unique miner ID in pool (0-65535)
static int g_total_miners = 1024;    // Total miners in pool (configurable)

// Generate cryptographically strong random start with multiple entropy sources
// Uses GPU UUID for hardware-unique identification instead of worker_id
uint64_t generate_secure_random_start(const std::string& puzzle_id, int gpu_id, const std::string& worker_id, const std::string& gpu_uuid) {
    // Entropy source 1: High-resolution timestamp (nanoseconds)
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    
    // Entropy source 2: Process ID (unique per miner instance)
    uint64_t process_id = static_cast<uint64_t>(getpid());
    
    // Entropy source 3: Thread ID (additional randomness)
    uint64_t thread_id = static_cast<uint64_t>(pthread_self());
    
    // Entropy source 4: Random device (hardware RNG if available)
    std::random_device rd;
    uint64_t hw_random = (static_cast<uint64_t>(rd()) << 32) | static_cast<uint64_t>(rd());
    
    // Entropy source 5: Memory address (ASLR provides randomness)
    uint64_t stack_addr = reinterpret_cast<uint64_t>(&now);
    
    // Combine all entropy sources with XOR and mixing
    uint64_t random_seed = timestamp_ns;
    random_seed ^= process_id * 0x9e3779b97f4a7c15ULL;  // Golden ratio
    random_seed ^= thread_id * 0x85ebca6b;
    random_seed ^= hw_random;
    random_seed ^= stack_addr;
    
    // Hash proposal_id (puzzle_id) to ensure different proposals get different nonce ranges
    // This ensures the same GPU mines different nonce ranges for different jobs/proposals
    uint64_t proposal_hash = 0;
    for (size_t i = 0; i < puzzle_id.length(); ++i) {
        proposal_hash = proposal_hash * 31 + (uint64_t)puzzle_id[i];
    }
    
    // NEW: Use GPU UUID AND proposal_id together for hardware-unique + job-unique identification
    // GPU UUID ensures different GPUs get different ranges
    // Proposal ID ensures the same GPU gets different ranges for different jobs/proposals
    uint64_t gpu_proposal_combined = 0;
    if (!gpu_uuid.empty()) {
        uint64_t uuid_hash = 0;
        for (size_t i = 0; i < gpu_uuid.length(); ++i) {
            uuid_hash = uuid_hash * 31 + (uint64_t)gpu_uuid[i];
        }
        // Combine GPU UUID and proposal_id hash together
        gpu_proposal_combined = uuid_hash ^ (proposal_hash * 0x9e3779b97f4a7c15ULL);
        random_seed ^= gpu_proposal_combined;  // GPU UUID + proposal_id provides uniqueness per GPU per job
    } else if (!worker_id.empty()) {
        // Fallback 1: Use worker_id from server if GPU UUID not available
        uint64_t worker_hash = 0;
        for (size_t i = 0; i < worker_id.length(); ++i) {
            worker_hash = worker_hash * 31 + (uint64_t)worker_id[i];
        }
        // Combine worker_id and proposal_id hash together
        gpu_proposal_combined = worker_hash ^ (proposal_hash * 0x9e3779b97f4a7c15ULL);
        random_seed ^= gpu_proposal_combined;
    } else {
        // Fallback 2: Use miner_id + gpu_id if neither UUID nor worker_id available
        uint64_t fallback_hash = (uint64_t)g_miner_id ^ ((uint64_t)gpu_id * 0x9e3779b97f4a7c15ULL);
        gpu_proposal_combined = fallback_hash ^ (proposal_hash * 0x9e3779b97f4a7c15ULL);
        random_seed ^= gpu_proposal_combined;
    }
    

    // Apply SplitMix64 hash mixing for better distribution
    random_seed ^= (random_seed >> 30);
    random_seed *= 0xbf58476d1ce4e5b9ULL;
    random_seed ^= (random_seed >> 27);
    random_seed *= 0x94d049bb133111ebULL;
    random_seed ^= (random_seed >> 31);
    
    return random_seed;
}

// GPU Range Calculation with Mining Pool Support
// Hierarchical partitioning: First by miner ID, then by GPU ID
// Supports miners with hundreds of GPUs and large pools

GpuNonceRange calculate_gpu_range(int gpu_id, int total_gpus) {
    // Get partition ID from environment (for nonce space partitioning)
    // Check both PARTITION_ID (new) and MINER_ID (legacy) for backward compatibility
    const char* env_partition_id = std::getenv("PARTITION_ID");
    if (env_partition_id) {
        int partition_val = std::atoi(env_partition_id);
        if (partition_val >= 0 && partition_val < 65536) {
            g_miner_id = partition_val;
        }
    } else {
        // Legacy support for MINER_ID env var
        const char* env_miner_id = std::getenv("MINER_ID");
        if (env_miner_id) {
            int miner_val = std::atoi(env_miner_id);
            if (miner_val >= 0 && miner_val < 65536) {
                g_miner_id = miner_val;
            }
        }
    }
    
    // Get total miners in pool
    const char* env_total_miners = std::getenv("POOL_SIZE");
    if (env_total_miners) {
        int pool_val = std::atoi(env_total_miners);
        if (pool_val > 0 && pool_val <= 65536) {  // Support up to 65K miners
            g_total_miners = pool_val;
        }
    }
    
    // Get GPUs per miner (support up to 36 GPUs per machine)
    // NOTE: This env var is OPTIONAL - if not set, uses the total_gpus parameter
    //       which should be the actual detected GPU count from the caller
    const char* env_gpus = std::getenv("NUM_GPUS");
    if (env_gpus) {
        int env_val = std::atoi(env_gpus);
        if (env_val > 0 && env_val <= 36) {  // Hardware limit: 36 GPUs per machine
            total_gpus = env_val;
        }
    }
    
    GpuNonceRange range;
    range.miner_id = g_miner_id;
    range.total_miners = g_total_miners;
    range.gpu_id = gpu_id;
    range.total_gpus = total_gpus;
    
    // Two-level partitioning:
    // Level 1: Divide by total miners in pool
    // Level 2: Divide miner's range by GPUs per miner
    
    const uint64_t usable_bits = 62;
    const uint64_t total_space = (1ULL << usable_bits);
    
    // Each miner gets: total_space / total_miners
    uint64_t miner_range_size = total_space / g_total_miners;
    uint64_t miner_range_start = miner_range_size * g_miner_id;
    
    // Each GPU within miner gets: miner_range / gpus_per_miner
    range.range_size = miner_range_size / total_gpus;
    range.range_start = miner_range_start + (range.range_size * gpu_id);
    
    return range;
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