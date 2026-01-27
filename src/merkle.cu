#include "merkle.cuh"

// ===== MTREE CLASS IMPLEMENTATION =====

MTree MTree::build_inplace(std::vector<Digest> leafs, 
                           std::vector<Digest> internal_nodes,
                           bool* cancel_flag) {
    // Calculate height
    size_t height = 0;
    size_t temp = leafs.size();
    while (temp > 1) {
        temp >>= 1;
        height++;
    }
    
    size_t num_sequential_layers = std::min(height, size_t(8));
    size_t seq_cutoff_height = std::max(size_t(1), height - num_sequential_layers);
    
    // First layer: connects leafs to internal nodes
    size_t range_layer_0_start = 1 << (height - 1);
    size_t range_layer_0_end = 1 << height;
    
    // EXACT CPU LOGIC: Use merkle_zip like CPU version
    merkle_zip(&internal_nodes[range_layer_0_start], leafs.data(), 
               range_layer_0_end - range_layer_0_start);
    
    if (cancel_flag && *cancel_flag) {
        return MTree();
    }
    
    // EXACT CPU LOGIC: Remaining layers connect internal nodes to internal nodes
    for (size_t layer = 1; layer < seq_cutoff_height; ++layer) {
        size_t parents_start = 1 << (height - 1 - layer);
        size_t mid_point = 1 << (height - layer);
        
        par_merkle_zip(&internal_nodes[parents_start], 
                      &internal_nodes[mid_point], 
                      mid_point - parents_start,
                      cancel_flag);
        
        if (cancel_flag && *cancel_flag) {
            return MTree();
        }
    }
    
    // EXACT CPU LOGIC: Do top of tree sequentially
    for (size_t layer = seq_cutoff_height; layer < height; ++layer) {
        size_t parents_start = 1 << (height - 1 - layer);
        size_t mid_point = 1 << (height - layer);
        
        merkle_zip(&internal_nodes[parents_start], 
                  &internal_nodes[mid_point], 
                  mid_point - parents_start);
    }
    
    return MTree(std::move(leafs), std::move(internal_nodes));
}

void MTree::merkle_zip(Digest* parents, const Digest* children, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        parents[i] = tip5_hash_fixed_host(children[2 * i], children[2 * i + 1]);
    }
}

void MTree::par_merkle_zip(Digest* parents, const Digest* children, 
                           size_t count, bool* cancel_flag) {
    // Simple parallel implementation using OpenMP if available
    // Otherwise falls back to sequential
    #pragma omp parallel for if(count > 1000)
    for (size_t i = 0; i < count; ++i) {
        if (cancel_flag && *cancel_flag) continue;
        parents[i] = tip5_hash_fixed_host(children[2 * i], children[2 * i + 1]);
    }
}

std::vector<Digest> MTree::path(size_t index) const {
    std::vector<Digest> path;
    size_t running_index = index + leafs_.size();
    path.push_back(leafs_[index ^ 1]);
    
    // Calculate tree height as log2 of number of leafs
    size_t tree_height = 0;
    size_t temp = leafs_.size();
    while (temp > 1) {
        temp >>= 1;
        tree_height++;
    }
    
    for (size_t i = 1; i < tree_height; ++i) {
        running_index >>= 1;
        path.push_back(internal_nodes_[running_index ^ 1]);
    }
    
    return path;
}

// ===== GPU PREPROCESSING KERNELS =====

__global__ void __launch_bounds__(256) bitreverse_swap_leafs_kernel(
    Digest* __restrict__ leafs,
    size_t num_leafs,
    uint32_t log2_n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_leafs) return;
    
    // Calculate bit-reversed index - match Rust bitreverse implementation exactly
    // Rust uses u32 and reverses only the lower log2_n bits
    uint32_t k = (uint32_t)idx;
    
    // Manual bit-reversal matching Rust implementation (for log2_n bits)
    // This is more accurate than __brev which reverses all 32 bits
    uint32_t rev_k = k;
    rev_k = ((rev_k & 0x55555555) << 1) | ((rev_k & 0xaaaaaaaa) >> 1);
    rev_k = ((rev_k & 0x33333333) << 2) | ((rev_k & 0xcccccccc) >> 2);
    rev_k = ((rev_k & 0x0f0f0f0f) << 4) | ((rev_k & 0xf0f0f0f0) >> 4);
    rev_k = ((rev_k & 0x00ff00ff) << 8) | ((rev_k & 0xff00ff00) >> 8);
    rev_k = __funnelshift_r(rev_k, rev_k, 16);  // rotate_right(16)
    rev_k = rev_k >> ((32 - log2_n) & 0x1f);
    
    // Only swap if k < rev_k (prevents double-swapping) - matches Rust swap_indices logic
    if (k < rev_k && rev_k < num_leafs) {
        Digest temp = leafs[k];
        leafs[k] = leafs[rev_k];
        leafs[rev_k] = temp;
    }
}

__global__ void __launch_bounds__(128) build_layer1_from_layer0_on_demand_kernel(
    Digest* __restrict__ internal_nodes,
    size_t height,
    size_t num_leafs,
    const Digest commitment
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t layer1_start = 1ULL << (height - 2); // Layer 1 start index (2^25)
    size_t layer1_count = 1ULL << (height - 2); // Number of nodes in layer 1  
    if (idx >= layer1_count) return;
    
    // Write index will be [layer1_start + idx]
    size_t write_idx = layer1_start + idx;
    
    // Layer 1 node at write_idx has children from layer 0
    size_t layer0_child_a = write_idx * 2;
    size_t layer0_child_b = layer0_child_a + 1;
    
    // Compute layer 0 children on-demand from leaves
    // Each layer 0 node is built from 2 leaves
    Digest layer0_node_a, layer0_node_b;
    
    // Compute layer 0 node a from its leaf children
    {
        uint64_t leaf_idx_a = (layer0_child_a - (1ULL << (height-1))) * 2;
        uint64_t leaf_idx_b = leaf_idx_a + 1;
        Digest leaf_a = compute_leaf_from_commitment_device_parallel(commitment, leaf_idx_a, num_leafs);
        Digest leaf_b = compute_leaf_from_commitment_device_parallel(commitment, leaf_idx_b, num_leafs);
        layer0_node_a = tip5_hash_fixed_device(leaf_a, leaf_b);
    }
    
    // Compute layer 0 node b from its leaf children
    {
        uint64_t leaf_idx_a = (layer0_child_b - (1ULL << (height-1))) * 2;
        uint64_t leaf_idx_b = leaf_idx_a + 1;
        Digest leaf_a = compute_leaf_from_commitment_device_parallel(commitment, leaf_idx_a, num_leafs);
        Digest leaf_b = compute_leaf_from_commitment_device_parallel(commitment, leaf_idx_b, num_leafs);
        layer0_node_b = tip5_hash_fixed_device(leaf_a, leaf_b);
    }
    
    // Hash layer 0 children to create layer 1 node
    internal_nodes[write_idx] = tip5_hash_fixed_device(layer0_node_a, layer0_node_b);
}

__global__ void __launch_bounds__(256) build_layer0_from_commitment_kernel(
    Digest* __restrict__ internal_nodes,
    size_t height,
    size_t num_leafs,
    const Digest commitment
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t range_layer_0_start = 1ULL << (height - 1);
    size_t layer_0_count = range_layer_0_start;
    if (idx >= layer_0_count) return;

    uint64_t leaf_a_index = (uint64_t)(idx * 2ULL);
    uint64_t leaf_b_index = leaf_a_index + 1ULL;

    Digest leaf_a = compute_leaf_from_commitment_device(commitment, leaf_a_index, num_leafs);
    Digest leaf_b = compute_leaf_from_commitment_device(commitment, leaf_b_index, num_leafs);

    internal_nodes[range_layer_0_start + idx] = tip5_hash_fixed_device(leaf_a, leaf_b);
}

__global__ void __launch_bounds__(256) compute_buds_kernel(
    Digest* __restrict__ buds,
    const Digest commitment,
    size_t segment_len,
    size_t base_index) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < segment_len) {
        size_t global_idx = base_index + idx;
        // Bud computation: 32 rounds of hashing (BUDDING_ROUNDS)
        // hash = Tip5::hash_pair(hash, Digest::new([index, 0, 0, 0, round]))
        Digest hash = commitment;
        for (size_t round = 0; round < BUDDING_ROUNDS; ++round) {
            // Create Digest with values [global_idx, 0, 0, 0, round]
            Digest round_digest;
            round_digest.values[0] = global_idx;
            round_digest.values[1] = 0;
            round_digest.values[2] = 0;
            round_digest.values[3] = 0;
            round_digest.values[4] = round;
            hash = tip5_hash_fixed_device(hash, round_digest);
        }
        buds[global_idx] = hash;
    }
}

__global__ void __launch_bounds__(256) compute_leafs_from_buds_kernel(
    Digest* __restrict__ leafs, 
    const Digest* __restrict__ buds, 
    size_t num_leafs, size_t layer) {
    // Fast 32-bit path when safe
    if (num_leafs <= 0xFFFFFFFFu && layer < 32) {
        uint32_t idx32 = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t n32 = static_cast<uint32_t>(num_leafs);
        if (idx32 < n32) {
            uint32_t stride32 = (1u << static_cast<unsigned int>(layer));
            uint32_t buddy32 = (idx32 + stride32) & (n32 - 1u);
            leafs[idx32] = tip5_hash_fixed_device(buds[idx32], buds[buddy32]);
        }
    } else {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_leafs) {
            size_t stride = (1ULL << layer);
            size_t mask = num_leafs - 1ULL; // num_leafs is power-of-two
            size_t buddy_index = (idx + stride) & mask;
            leafs[idx] = tip5_hash_fixed_device(buds[idx], buds[buddy_index]);
        }
    }
}

__global__ void __launch_bounds__(256) merkle_zip_kernel(
    Digest* __restrict__ parents, 
    const Digest* __restrict__ children, 
    size_t count) {
    if (count <= 0xFFFFFFFFu) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < static_cast<uint32_t>(count)) {
            parents[idx] = tip5_hash_fixed_device(children[2*idx], children[2*idx+1]);
        }
    } else {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < count) {
            parents[idx] = tip5_hash_fixed_device(children[2*idx], children[2*idx+1]);
        }
    }
}

// ===== DEVICE HELPER FUNCTIONS =====
// compute_leaf_from_commitment_device and compute_leaf_from_commitment_device_parallel
// are defined inline in merkle.cuh

__device__ __noinline__ Digest get_internal_node_safe(
    const Digest* d_internal_nodes,
    size_t node_index,
    size_t stored_nodes_count,
    const Digest& commitment,
    const Digest& leaf_prefix,  // Commitment (Reboot/Xnt) or prev_block_digest (HardforkAlpha)
    size_t num_leafs
) {
    // Fast path: If node is within stored buffer, return it directly
    if (node_index < stored_nodes_count) {
        return d_internal_nodes[node_index];
    }
    
    // Slow path: Compute layer 0 node on-demand if needed
    const size_t layer0_start = 1ULL << (MERKLE_TREE_HEIGHT_ - 1); // 2^26
    const size_t layer0_end = 1ULL << MERKLE_TREE_HEIGHT_;   // 2^27
    
    if (node_index >= layer0_start && node_index < layer0_end) {
        // Layer 0 node - compute from leaves
        size_t layer0_idx = node_index - layer0_start;
        uint64_t leaf_a_idx = layer0_idx * 2;
        uint64_t leaf_b_idx = leaf_a_idx + 1;
        
        // Use leaf_prefix (commitment for Reboot/Xnt, prev_block_digest for HardforkAlpha)
        Digest leaf_a = compute_leaf_from_commitment_device(leaf_prefix, leaf_a_idx, num_leafs);
        Digest leaf_b = compute_leaf_from_commitment_device(leaf_prefix, leaf_b_idx, num_leafs);
        
        // Hash and return
        return tip5_hash_fixed_device(leaf_a, leaf_b);
    }
    
    // Unknown index - return zero digest
    return Digest::default_digest();
}

__device__ void compute_merkle_path(
    Digest* path,
    size_t leaf_index,
    const Digest* d_internal_nodes,
    const Digest* d_leafs,
    size_t num_leafs,
    size_t merkle_height) {
    
    size_t running_index = leaf_index;
    
    // First level: sibling leaf
    size_t sibling_leaf_index = running_index ^ 1;
    if (d_leafs != nullptr && sibling_leaf_index < num_leafs) {
        path[0] = d_leafs[sibling_leaf_index];
    } else {
        path[0] = Digest::default_digest();
    }
    running_index >>= 1;
    
    // Subsequent levels: sibling internal nodes
    for (size_t i = 1; i < merkle_height; ++i) {
        size_t layer_start = layer_start_index_device(i - 1, num_leafs);
        size_t sibling_index = layer_start + (running_index ^ 1);
        
        path[i] = d_internal_nodes[sibling_index];
        running_index >>= 1;
    }
}

__device__ void compute_merkle_paths(
    Digest* path_a,
    Digest* path_b,
    size_t index_a,
    size_t index_b,
    const Digest* d_internal_nodes,
    const Digest* d_leafs,
    size_t num_leafs,
    size_t merkle_height) {
    
    compute_merkle_path(path_a, index_a, d_internal_nodes, d_leafs, num_leafs, merkle_height);
    compute_merkle_path(path_b, index_b, d_internal_nodes, d_leafs, num_leafs, merkle_height);
}