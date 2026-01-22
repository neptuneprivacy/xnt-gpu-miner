#ifndef XNT_MERKLE_CUH
#define XNT_MERKLE_CUH

#include "digest.cuh"

// ===== MERKLE TREE CLASS =====
// Host-side Merkle tree implementation for preprocessing
class MTree {
private:
    std::vector<Digest> leafs_;
    std::vector<Digest> internal_nodes_;

public:
    // Default constructor
    MTree() = default;
    
    // Constructor with pre-allocated storage
    MTree(std::vector<Digest> leafs, std::vector<Digest> internal_nodes)
        : leafs_(std::move(leafs)), internal_nodes_(std::move(internal_nodes)) {}
    
    // Build tree in-place from leafs
    // internal_nodes should be pre-allocated with the right size
    static MTree build_inplace(std::vector<Digest> leafs, 
                              std::vector<Digest> internal_nodes,
                              bool* cancel_flag = nullptr);
    
    // Zip operation: compute parent hashes from child pairs
    // parents[i] = hash(children[2*i], children[2*i+1])
    static void merkle_zip(Digest* parents, const Digest* children, size_t count);
    
    // Parallel zip operation with cancellation support
    static void par_merkle_zip(Digest* parents, const Digest* children, 
                              size_t count, bool* cancel_flag = nullptr);
    
    // Get the root hash
    Digest root() const {
        if (internal_nodes_.empty()) {
            return Digest::default_digest();
        }
        return internal_nodes_.back();
    }
    
    // Get number of leafs
    size_t num_leafs() const {
        return leafs_.size();
    }
    
    // Get authentication path for a leaf at given index
    std::vector<Digest> path(size_t index) const;
    
    // Access internal nodes (for GPU transfer)
    const std::vector<Digest>& internal_nodes() const { return internal_nodes_; }
    std::vector<Digest>& internal_nodes() { return internal_nodes_; }
    
    // Access leafs
    const std::vector<Digest>& leafs() const { return leafs_; }
    std::vector<Digest>& leafs() { return leafs_; }
};

// ===== GPU PREPROCESSING KERNELS =====

// Bit-reverse permutation of leafs for Merkle tree construction
// Swaps leafs[k] with leafs[bitreverse(k)] for all k < bitreverse(k)
__global__ void __launch_bounds__(256) bitreverse_swap_leafs_kernel(
    Digest* __restrict__ leafs,
    size_t num_leafs,
    uint32_t log2_n);

// Build layer 1 from layer 0 with on-demand leaf computation
// Used when layer 0 is not stored (low-VRAM mode)
__global__ void __launch_bounds__(128) build_layer1_from_layer0_on_demand_kernel(
    Digest* __restrict__ internal_nodes,
    size_t height,
    size_t num_leafs,
    const Digest commitment);

// Build layer 0 (leaf pairs) from commitment
// Each thread computes hash of two adjacent leafs
__global__ void __launch_bounds__(256) build_layer0_from_commitment_kernel(
    Digest* __restrict__ internal_nodes,
    size_t height,
    size_t num_leafs,
    const Digest commitment);

// Compute buds (intermediate values before leafs)
// bud[i] = hash^32(commitment, [i, 0, 0, 0, round])
__global__ void __launch_bounds__(256) compute_buds_kernel(
    Digest* __restrict__ buds,
    const Digest commitment,
    size_t segment_len,
    size_t base_index);

// Compute leafs from buds using tree reduction
// Each layer hashes pairs of buds/intermediate values
__global__ void __launch_bounds__(256) compute_leafs_from_buds_kernel(
    Digest* __restrict__ leafs, 
    const Digest* __restrict__ buds, 
    size_t num_leafs, 
    size_t layer);

// Standard Merkle zip kernel
// parents[i] = hash(children[2*i], children[2*i+1])
__global__ void __launch_bounds__(256) merkle_zip_kernel(
    Digest* __restrict__ parents, 
    const Digest* __restrict__ children, 
    size_t count);

// ===== DEVICE HELPER FUNCTIONS =====

// Compute leaf from commitment using bud computation
// leaf[index] = bud(commitment, index)
__device__ __forceinline__ Digest compute_leaf_from_commitment_device(
    const Digest commitment, 
    uint64_t base_index, 
    size_t num_leafs);

// Parallel version for warp-cooperative computation
__device__ __forceinline__ Digest compute_leaf_from_commitment_device_parallel(
    const Digest commitment, 
    uint64_t base_index, 
    size_t num_leafs);

// Get internal node from tree, with on-demand computation if needed
// Used when not all layers are stored in memory
__device__ __noinline__ Digest get_internal_node_safe(
    const Digest* d_internal_nodes,
    size_t node_index,
    size_t stored_nodes_count,
    const Digest& commitment,
    const Digest& leaf_prefix,
    size_t num_leafs);

// Compute Merkle path for a leaf
// Fills path array with sibling nodes from leaf to root
__device__ void compute_merkle_path(
    Digest* path,
    size_t leaf_index,
    const Digest* d_internal_nodes,
    const Digest* d_leafs,
    size_t num_leafs,
    size_t merkle_height);

// Compute both Merkle paths for two leaf indices
__device__ void compute_merkle_paths(
    Digest* path_a,
    Digest* path_b,
    size_t index_a,
    size_t index_b,
    const Digest* d_internal_nodes,
    const Digest* d_leafs,
    size_t num_leafs,
    size_t merkle_height);

// ===== HOST HELPER FUNCTIONS =====

// Calculate tree height from number of leafs
inline size_t calculate_tree_height(size_t num_leafs) {
    if (num_leafs == 0) return 0;
    size_t height = 0;
    size_t temp = num_leafs;
    while (temp > 1) {
        temp >>= 1;
        height++;
    }
    return height;
}

// Calculate number of internal nodes for a tree
inline size_t calculate_internal_nodes_count(size_t num_leafs) {
    // For a complete binary tree with n leafs:
    // Internal nodes = n - 1 (not including root in some representations)
    // Here we store all layers from leafs up to root
    if (num_leafs == 0) return 0;
    return num_leafs - 1;
}

// Calculate range of internal nodes for a specific layer
// Layer 0: leaf pairs, Layer height-1: root
inline void calculate_layer_range(size_t layer, size_t height, size_t num_leafs,
                                  size_t& start, size_t& count) {
    // Layer 0 starts at internal_nodes[num_leafs/2 - 1] (or similar based on layout)
    // This depends on the specific tree layout used
    size_t nodes_at_layer = num_leafs >> (layer + 1);
    if (layer == 0) {
        start = 0;
        count = num_leafs / 2;
    } else {
        start = num_leafs / 2;
        for (size_t l = 1; l < layer; ++l) {
            start += num_leafs >> (l + 1);
        }
        count = nodes_at_layer;
    }
}

// Bit-reverse a value with given number of bits
inline uint64_t bitreverse_host(uint64_t n, uint32_t log2_n) {
    uint64_t result = 0;
    for (uint32_t i = 0; i < log2_n; ++i) {
        if ((n >> i) & 1) {
            result |= (1ULL << (log2_n - 1 - i));
        }
    }
    return result;
}

// ===== TREE LAYOUT CONSTANTS =====

// The internal nodes are stored in a specific layout:
// For a tree with 2^k leafs:
// - Layer 0 (closest to leafs): indices [0, 2^(k-1) - 1], stores hash(leaf[2i], leaf[2i+1])
// - Layer 1: indices [2^(k-1), 2^(k-1) + 2^(k-2) - 1]
// - ...
// - Layer k-1 (root): index [2^k - 2]

// Calculate the starting index of a layer in the internal nodes array
inline size_t layer_start_index(size_t layer, size_t num_leafs) {
    if (layer == 0) return 0;
    size_t start = 0;
    for (size_t l = 0; l < layer; ++l) {
        start += num_leafs >> (l + 1);
    }
    return start;
}

// Calculate the number of nodes at a specific layer
inline size_t layer_node_count(size_t layer, size_t num_leafs) {
    return num_leafs >> (layer + 1);
}

#endif // XNT_MERKLE_CUH