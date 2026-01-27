#ifndef XNT_MERKLE_CUH
#define XNT_MERKLE_CUH

#include "digest.cuh"

class MTree {
private:
    std::vector<Digest> leafs_;
    std::vector<Digest> internal_nodes_;

public:
    MTree() = default;
    
    MTree(std::vector<Digest> leafs, std::vector<Digest> internal_nodes)
        : leafs_(std::move(leafs)), internal_nodes_(std::move(internal_nodes)) {}
    
    static MTree build_inplace(std::vector<Digest> leafs, 
                              std::vector<Digest> internal_nodes,
                              bool* cancel_flag = nullptr);
    
    static void merkle_zip(Digest* parents, const Digest* children, size_t count);
    
    static void par_merkle_zip(Digest* parents, const Digest* children, 
                              size_t count, bool* cancel_flag = nullptr);
    
    Digest root() const {
        if (internal_nodes_.empty()) return Digest::default_digest();
        return internal_nodes_.back();
    }
    
    size_t num_leafs() const { return leafs_.size(); }
    
    std::vector<Digest> path(size_t index) const;
    
    const std::vector<Digest>& internal_nodes() const { return internal_nodes_; }
    std::vector<Digest>& internal_nodes() { return internal_nodes_; }
    const std::vector<Digest>& leafs() const { return leafs_; }
    std::vector<Digest>& leafs() { return leafs_; }
};

__global__ void __launch_bounds__(256) bitreverse_swap_leafs_kernel(
    Digest* __restrict__ leafs,
    size_t num_leafs,
    uint32_t log2_n);

__global__ void __launch_bounds__(128) build_layer1_from_layer0_on_demand_kernel(
    Digest* __restrict__ internal_nodes,
    size_t height,
    size_t num_leafs,
    const Digest commitment);

__global__ void __launch_bounds__(256) build_layer0_from_commitment_kernel(
    Digest* __restrict__ internal_nodes,
    size_t height,
    size_t num_leafs,
    const Digest commitment);

__global__ void __launch_bounds__(256) compute_buds_kernel(
    Digest* __restrict__ buds,
    const Digest commitment,
    size_t segment_len,
    size_t base_index);

__global__ void __launch_bounds__(256) compute_leafs_from_buds_kernel(
    Digest* __restrict__ leafs, 
    const Digest* __restrict__ buds, 
    size_t num_leafs, 
    size_t layer);

__global__ void __launch_bounds__(256) merkle_zip_kernel(
    Digest* __restrict__ parents, 
    const Digest* __restrict__ children, 
    size_t count);

__device__ __forceinline__ Digest compute_leaf_from_commitment_device(
    const Digest commitment, uint64_t base_index, size_t num_leafs) {
    const int kLevels = (int)NUM_BUD_LAYERS; // 5
    const int span = 1 << kLevels;           // 32
    Digest buf[1 << NUM_BUD_LAYERS];

    // Compute all buds for this leaf
    for (int i = 0; i < span; ++i) {
        uint64_t idx = (base_index + (uint64_t)i) % (uint64_t)num_leafs;
        // Bud computation: 32 rounds of hashing (BUDDING_ROUNDS)
        // hash = Tip5::hash_pair(hash, Digest::new([index, 0, 0, 0, round]))
        Digest hash = commitment;
        for (size_t round = 0; round < BUDDING_ROUNDS; ++round) {
            // Create Digest with values [idx, 0, 0, 0, round]
            Digest round_digest;
            round_digest.values[0] = idx;
            round_digest.values[1] = 0;
            round_digest.values[2] = 0;
            round_digest.values[3] = 0;
            round_digest.values[4] = round;
            hash = tip5_hash_fixed_device(hash, round_digest);
        }
        buf[i] = hash;
    }

    // Pairwise fold with doubling stride each level
    for (int level = 0, step = 1; level < kLevels; ++level, step <<= 1) {
        for (int i = 0; i < span; i += (step << 1)) {
            buf[i] = tip5_hash_fixed_device(buf[i], buf[i + step]);
        }
    }
    return buf[0];
}

// Parallel version - same implementation but may use shared memory optimizations in future
__device__ __forceinline__ Digest compute_leaf_from_commitment_device_parallel(
    const Digest commitment, uint64_t base_index, size_t num_leafs) {
    // Use the same implementation as the regular version to avoid memory access issues
    return compute_leaf_from_commitment_device(commitment, base_index, num_leafs);
}

__device__ __noinline__ Digest get_internal_node_safe(
    const Digest* d_internal_nodes,
    size_t node_index,
    size_t stored_nodes_count,
    const Digest& commitment,
    const Digest& leaf_prefix,
    size_t num_leafs);

__device__ void compute_merkle_path(
    Digest* path,
    size_t leaf_index,
    const Digest* d_internal_nodes,
    const Digest* d_leafs,
    size_t num_leafs,
    size_t merkle_height);

__device__ void compute_merkle_paths(
    Digest* path_a,
    Digest* path_b,
    size_t index_a,
    size_t index_b,
    const Digest* d_internal_nodes,
    const Digest* d_leafs,
    size_t num_leafs,
    size_t merkle_height);

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

inline size_t calculate_internal_nodes_count(size_t num_leafs) {
    if (num_leafs == 0) return 0;
    return num_leafs - 1;
}

inline uint64_t bitreverse_host(uint64_t n, uint32_t log2_n) {
    uint64_t result = 0;
    for (uint32_t i = 0; i < log2_n; ++i) {
        if ((n >> i) & 1) {
            result |= (1ULL << (log2_n - 1 - i));
        }
    }
    return result;
}

inline size_t layer_start_index(size_t layer, size_t num_leafs) {
    if (layer == 0) return 0;
    size_t start = 0;
    for (size_t l = 0; l < layer; ++l) {
        start += num_leafs >> (l + 1);
    }
    return start;
}

__device__ __forceinline__ size_t layer_start_index_device(size_t layer, size_t num_leafs) {
    if (layer == 0) return 0;
    size_t start = 0;
    for (size_t l = 0; l < layer; ++l) {
        start += num_leafs >> (l + 1);
    }
    return start;
}

inline size_t layer_node_count(size_t layer, size_t num_leafs) {
    return num_leafs >> (layer + 1);
}

#endif
