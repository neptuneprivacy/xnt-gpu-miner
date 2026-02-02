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

// Performance tuning constants
constexpr int BLOCKS_PER_SM_DEFAULT = 4;
constexpr int BLOCKS_PER_SM_BLACKWELL = 8;  // RTX 5090 benefits from more blocks

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

#endif
