#include "merkle.cuh"

// ===== MTREE CLASS IMPLEMENTATION =====

MTree MTree::build_inplace(std::vector<Digest> leafs, 
                           std::vector<Digest> internal_nodes,
                           bool* cancel_flag) {
    MTree tree(std::move(leafs), std::move(internal_nodes));
    
    if (tree.leafs_.empty()) {
        return tree;
    }
    
    size_t height = calculate_tree_height(tree.leafs_.size());
    
    // Build tree layer by layer
    // Layer 0: hash pairs of leafs
    // Layer 1+: hash pairs of previous layer
    
    size_t num_sequential_layers = 4; // Use sequential for top layers
    size_t seq_cutoff_height = (height > num_sequential_layers) ? height - num_sequential_layers : 0;
    
    // Build lower layers (parallel)
    for (size_t layer = 0; layer < seq_cutoff_height; ++layer) {
        if (cancel_flag && *cancel_flag) {
            return MTree();
        }
        
        size_t parents_start = layer_start_index(layer, tree.leafs_.size());
        size_t count = layer_node_count(layer, tree.leafs_.size());
        
        if (layer == 0) {
            // First layer: hash pairs of leafs
            merkle_zip(tree.internal_nodes_.data() + parents_start, 
                      tree.leafs_.data(), count);
        } else {
            // Subsequent layers: hash pairs of previous layer
            size_t children_start = layer_start_index(layer - 1, tree.leafs_.size());
            merkle_zip(tree.internal_nodes_.data() + parents_start,
                      tree.internal_nodes_.data() + children_start, count);
        }
    }
    
    // Build upper layers (sequential for cache efficiency)
    for (size_t layer = seq_cutoff_height; layer < height; ++layer) {
        if (cancel_flag && *cancel_flag) {
            return MTree();
        }
        
        size_t parents_start = layer_start_index(layer, tree.leafs_.size());
        size_t count = layer_node_count(layer, tree.leafs_.size());
        
        if (layer == 0) {
            merkle_zip(tree.internal_nodes_.data() + parents_start,
                      tree.leafs_.data(), count);
        } else {
            size_t children_start = layer_start_index(layer - 1, tree.leafs_.size());
            merkle_zip(tree.internal_nodes_.data() + parents_start,
                      tree.internal_nodes_.data() + children_start, count);
        }
    }
    
    return tree;
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
    std::vector<Digest> result;
    
    if (leafs_.empty() || index >= leafs_.size()) {
        return result;
    }
    
    size_t tree_height = calculate_tree_height(leafs_.size());
    result.reserve(tree_height);
    
    size_t running_index = index;
    
    // First level: sibling leaf
    size_t sibling_leaf_index = (running_index ^ 1);
    if (sibling_leaf_index < leafs_.size()) {
        result.push_back(leafs_[sibling_leaf_index]);
    } else {
        result.push_back(Digest::default_digest());
    }
    running_index >>= 1;
    
    // Subsequent levels: sibling internal nodes
    for (size_t i = 0; i < tree_height - 1; ++i) {
        size_t layer_start = layer_start_index(i, leafs_.size());
        size_t sibling_index = layer_start + (running_index ^ 1);
        
        if (sibling_index < internal_nodes_.size()) {
            result.push_back(internal_nodes_[sibling_index]);
        } else {
            result.push_back(Digest::default_digest());
        }
        running_index >>= 1;
    }
    
    return result;
}

// ===== GPU PREPROCESSING KERNELS =====

__global__ void __launch_bounds__(256) bitreverse_swap_leafs_kernel(
    Digest* __restrict__ leafs,
    size_t num_leafs,
    uint32_t log2_n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_leafs / 2) return;
    
    // Compute bit-reversed index
    uint32_t k = static_cast<uint32_t>(idx);
    uint32_t rev_k = 0;
    
    #pragma unroll
    for (uint32_t i = 0; i < log2_n; ++i) {
        if ((k >> i) & 1) {
            rev_k |= (1u << (log2_n - 1 - i));
        }
    }
    
    // Only swap if k < rev_k (to avoid double-swapping)
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
    const Digest commitment) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t layer1_start = num_leafs / 2;  // Layer 1 starts after layer 0
    size_t layer1_count = num_leafs / 4;
    
    if (idx >= layer1_count) return;
    
    size_t write_idx = layer1_start + idx;
    
    // Layer 0 indices for children
    size_t layer0_child_a = 2 * idx;
    size_t layer0_child_b = 2 * idx + 1;
    
    // Compute layer 0 nodes on-demand
    // Each layer 0 node is hash of two leafs
    uint64_t leaf_idx_a = layer0_child_a * 2;
    uint64_t leaf_idx_b = layer0_child_a * 2 + 1;
    Digest leaf_a = compute_leaf_from_commitment_device(commitment, leaf_idx_a, num_leafs);
    Digest leaf_b = compute_leaf_from_commitment_device(commitment, leaf_idx_b, num_leafs);
    Digest node_a = tip5_hash_fixed_device(leaf_a, leaf_b);
    
    leaf_idx_a = layer0_child_b * 2;
    leaf_idx_b = layer0_child_b * 2 + 1;
    leaf_a = compute_leaf_from_commitment_device(commitment, leaf_idx_a, num_leafs);
    leaf_b = compute_leaf_from_commitment_device(commitment, leaf_idx_b, num_leafs);
    Digest node_b = tip5_hash_fixed_device(leaf_a, leaf_b);
    
    // Hash the two layer 0 nodes to get layer 1 node
    internal_nodes[write_idx] = tip5_hash_fixed_device(node_a, node_b);
}

__global__ void __launch_bounds__(256) build_layer0_from_commitment_kernel(
    Digest* __restrict__ internal_nodes,
    size_t height,
    size_t num_leafs,
    const Digest commitment) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t layer_0_count = num_leafs / 2;
    
    if (idx >= layer_0_count) return;
    
    // Compute two leaf indices for this layer 0 node
    uint64_t leaf_idx_a = idx * 2;
    uint64_t leaf_idx_b = idx * 2 + 1;
    
    // Compute leafs from commitment
    Digest leaf_a = compute_leaf_from_commitment_device(commitment, leaf_idx_a, num_leafs);
    Digest leaf_b = compute_leaf_from_commitment_device(commitment, leaf_idx_b, num_leafs);
    
    // Hash to get layer 0 node
    internal_nodes[idx] = tip5_hash_fixed_device(leaf_a, leaf_b);
}

__global__ void __launch_bounds__(256) compute_buds_kernel(
    Digest* __restrict__ buds,
    const Digest commitment,
    size_t segment_len,
    size_t base_index) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= segment_len) return;
    
    size_t global_idx = base_index + idx;
    
    // Compute bud using BUDDING_ROUNDS iterations
    Digest hash = commitment;
    for (size_t round = 0; round < BUDDING_ROUNDS; ++round) {
        Digest round_digest;
        round_digest.values[0] = global_idx;
        round_digest.values[1] = 0;
        round_digest.values[2] = 0;
        round_digest.values[3] = 0;
        round_digest.values[4] = round;
        hash = tip5_hash_fixed_device(hash, round_digest);
    }
    
    buds[idx] = hash;
}

__global__ void __launch_bounds__(256) compute_leafs_from_buds_kernel(
    Digest* __restrict__ leafs, 
    const Digest* __restrict__ buds, 
    size_t num_leafs, 
    size_t layer) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = static_cast<uint32_t>(num_leafs);
    
    if (idx >= n) return;
    
    // Match Rust exactly: *leaf = Tip5::hash_pair(buds[k], buds[(k + (1 << i)) % NUM_LEAFS])
    // For layer i, we hash buds[k] with buds[(k + (1 << i)) % NUM_LEAFS]
    size_t k = idx;
    size_t offset = 1ULL << layer;
    size_t right_idx = (k + offset) % n;
    
    leafs[idx] = tip5_hash_fixed_device(buds[k], buds[right_idx]);
}

__global__ void __launch_bounds__(256) merkle_zip_kernel(
    Digest* __restrict__ parents, 
    const Digest* __restrict__ children, 
    size_t count) {
    
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    parents[idx] = tip5_hash_fixed_device(children[2 * idx], children[2 * idx + 1]);
}

// ===== DEVICE HELPER FUNCTIONS =====
// compute_leaf_from_commitment_device and compute_leaf_from_commitment_device_parallel
// are defined inline in merkle.cuh

__device__ __noinline__ Digest get_internal_node_safe(
    const Digest* d_internal_nodes,
    size_t node_index,
    size_t stored_nodes_count,
    const Digest& commitment,
    const Digest& leaf_prefix,
    size_t num_leafs) {
    
    // If node is stored, return it directly
    if (node_index < stored_nodes_count) {
        return d_internal_nodes[node_index];
    }
    
    // Otherwise, compute on demand
    // This is used in low-VRAM mode where not all nodes are stored
    
    // Determine which layer this node is in
    size_t layer = 0;
    size_t layer_start = 0;
    size_t layer_count = num_leafs / 2;
    
    while (layer_start + layer_count <= node_index) {
        layer_start += layer_count;
        layer_count /= 2;
        layer++;
    }
    
    size_t index_in_layer = node_index - layer_start;
    
    if (layer == 0) {
        // Layer 0: hash two adjacent leafs
        uint64_t leaf_a_idx = index_in_layer * 2;
        uint64_t leaf_b_idx = leaf_a_idx + 1;
        
        Digest leaf_a = compute_leaf_from_commitment_device(commitment, leaf_a_idx, num_leafs);
        Digest leaf_b = compute_leaf_from_commitment_device(commitment, leaf_b_idx, num_leafs);
        
        return tip5_hash_fixed_device(leaf_a, leaf_b);
    } else {
        // Higher layers: recursively compute children
        size_t child_layer_start = layer_start_index_device(layer - 1, num_leafs);
        size_t child_a_idx = child_layer_start + index_in_layer * 2;
        size_t child_b_idx = child_a_idx + 1;
        
        Digest child_a = get_internal_node_safe(d_internal_nodes, child_a_idx, 
                                                 stored_nodes_count, commitment, 
                                                 leaf_prefix, num_leafs);
        Digest child_b = get_internal_node_safe(d_internal_nodes, child_b_idx,
                                                 stored_nodes_count, commitment,
                                                 leaf_prefix, num_leafs);
        
        return tip5_hash_fixed_device(child_a, child_b);
    }
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