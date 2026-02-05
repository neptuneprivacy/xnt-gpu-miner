#ifndef XNT_KERNELS_CUH
#define XNT_KERNELS_CUH

#include "pow.cuh"

struct GuesserBuffer;
struct PowMastPaths;
struct Digest;

constexpr int MINING_THREADS_PER_BLOCK = 256;
constexpr int PREPROCESSING_THREADS_PER_BLOCK = 256;
constexpr int MERKLE_THREADS_PER_BLOCK = 256;
constexpr int MAX_GRID_DIM_X = 65535;
constexpr size_t SHARED_LUT_SIZE = 256;

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
    Digest* __restrict__ d_solution_nonce_digest);

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
    Digest* __restrict__ d_solution_nonce_digest);

__device__ Digest fast_mast_hash_device(
    const Pow& pow,
    const PowMastPaths& mast_paths,
    uint64_t state[STATE_SIZE]);

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

struct MiningSolution {
    Pow pow;
    Digest kernel_final_hash;  // The final_hash that kernel computed and checked
};

std::optional<MiningSolution> mine_pow_with_buffer(
    GuesserBuffer& buffer,
    const Digest& target,
    const PowMastPaths& mast_paths,
    uint64_t start_nonce,
    uint64_t max_nonces,
    int consensus_rule_set,
    bool* cancel_flag = nullptr);

struct MiningResult {
    bool solution_found;
    Pow pow_solution;
    Digest solution_hash;
    uint64_t nonces_tested;
    double elapsed_seconds;

    MiningResult() : solution_found(false), nonces_tested(0), elapsed_seconds(0.0) {}
};

MiningResult mine_batch(
    GuesserBuffer& buffer,
    const Digest& target,
    const PowMastPaths& mast_paths,
    uint64_t start_nonce,
    uint64_t batch_size,
    int consensus_rule_set,
    bool* cancel_flag = nullptr);

void calculate_mining_launch_config(
    uint64_t num_nonces,
    int& threads_per_block,
    int& blocks_per_grid,
    int gpu_id = 0);

uint64_t get_optimal_batch_size(int gpu_id, int target_duration_ms = 900);

struct MiningOutputBuffers {
    uint64_t* d_solution_nonce;
    int* d_solution_found;
    Digest* d_solution_path_a;
    Digest* d_solution_path_b;
    Digest* d_solution_nonce_digest;
    Digest* d_solution_final_hash;  // Store the final_hash that kernel computed

    MiningOutputBuffers()
        : d_solution_nonce(nullptr)
        , d_solution_found(nullptr)
        , d_solution_path_a(nullptr)
        , d_solution_path_b(nullptr)
        , d_solution_nonce_digest(nullptr)
        , d_solution_final_hash(nullptr) {}

    bool allocate();
    void free();
    bool reset();
};

enum class MiningKernelType {
    HIGH_VRAM,
    LOW_VRAM
};

MiningKernelType select_mining_kernel(int gpu_id);

inline const char* kernel_type_name(MiningKernelType type) {
    switch (type) {
        case MiningKernelType::HIGH_VRAM: return "HIGH_VRAM";
        case MiningKernelType::LOW_VRAM: return "LOW_VRAM";
        default: return "UNKNOWN";
    }
}

bool check_kernel_launch_errors(const char* kernel_name);
bool sync_and_check_errors(const char* stage);

// Top tree cache - constants defined in kernels.cu
constexpr size_t TOP_TREE_CACHE_LEVELS = 10;
constexpr size_t TOP_TREE_CACHE_SIZE = (1 << TOP_TREE_CACHE_LEVELS) - 1;  // 1023 nodes

// Declared in kernels.cu, extern here for use in pow.cu
// Store as raw uint64_t array since Digest has constructor (not allowed in __constant__)
extern __constant__ uint64_t d_top_tree_cache[TOP_TREE_CACHE_SIZE * DIGEST_LEN];

// Helper to read cached node as Digest
__device__ __forceinline__ Digest get_cached_node(size_t index) {
    Digest d;
    size_t base = index * DIGEST_LEN;
    d.values[0] = d_top_tree_cache[base];
    d.values[1] = d_top_tree_cache[base + 1];
    d.values[2] = d_top_tree_cache[base + 2];
    d.values[3] = d_top_tree_cache[base + 3];
    d.values[4] = d_top_tree_cache[base + 4];
    return d;
}

// Top tree cache functions
bool initialize_top_tree_cache(const Digest* d_merkle_tree, size_t num_leafs);
void reset_top_tree_cache();

// ===== MAST PATHS CACHE =====
// Cache PowMastPaths in constant memory for fast broadcast to all threads
// PowMastPaths = 6 Digests = 30 uint64_t = 240 bytes (tiny compared to 64KB limit)
// Accessed 8 times per hash in MAST chain, same for ALL threads
constexpr size_t MAST_PATHS_SIZE = 30;  // 6 Digests * 5 uint64_t each

extern __constant__ uint64_t d_mast_paths_cache[MAST_PATHS_SIZE];

// Helper to read cached mast_paths as PowMastPaths structure
__device__ __forceinline__ void get_cached_mast_paths_digests(
    Digest& pow0, Digest& pow1, Digest& pow2,
    Digest& header0, Digest& header1,
    Digest& kernel0
) {
    // pow[0] at offset 0
    pow0.values[0] = d_mast_paths_cache[0];
    pow0.values[1] = d_mast_paths_cache[1];
    pow0.values[2] = d_mast_paths_cache[2];
    pow0.values[3] = d_mast_paths_cache[3];
    pow0.values[4] = d_mast_paths_cache[4];
    // pow[1] at offset 5
    pow1.values[0] = d_mast_paths_cache[5];
    pow1.values[1] = d_mast_paths_cache[6];
    pow1.values[2] = d_mast_paths_cache[7];
    pow1.values[3] = d_mast_paths_cache[8];
    pow1.values[4] = d_mast_paths_cache[9];
    // pow[2] at offset 10
    pow2.values[0] = d_mast_paths_cache[10];
    pow2.values[1] = d_mast_paths_cache[11];
    pow2.values[2] = d_mast_paths_cache[12];
    pow2.values[3] = d_mast_paths_cache[13];
    pow2.values[4] = d_mast_paths_cache[14];
    // header[0] at offset 15
    header0.values[0] = d_mast_paths_cache[15];
    header0.values[1] = d_mast_paths_cache[16];
    header0.values[2] = d_mast_paths_cache[17];
    header0.values[3] = d_mast_paths_cache[18];
    header0.values[4] = d_mast_paths_cache[19];
    // header[1] at offset 20
    header1.values[0] = d_mast_paths_cache[20];
    header1.values[1] = d_mast_paths_cache[21];
    header1.values[2] = d_mast_paths_cache[22];
    header1.values[3] = d_mast_paths_cache[23];
    header1.values[4] = d_mast_paths_cache[24];
    // kernel[0] at offset 25
    kernel0.values[0] = d_mast_paths_cache[25];
    kernel0.values[1] = d_mast_paths_cache[26];
    kernel0.values[2] = d_mast_paths_cache[27];
    kernel0.values[3] = d_mast_paths_cache[28];
    kernel0.values[4] = d_mast_paths_cache[29];
}

// Mast paths cache functions
bool initialize_mast_paths_cache(const PowMastPaths& mast_paths);
void reset_mast_paths_cache();

#endif
