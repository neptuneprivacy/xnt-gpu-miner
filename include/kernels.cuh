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
    Digest* __restrict__ d_solution_nonce_digest,
    Digest* __restrict__ d_solution_final_hash);

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
    Digest* __restrict__ d_solution_nonce_digest,
    Digest* __restrict__ d_solution_final_hash);

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

// Top tree cache - cache internal Merkle nodes in constant memory for fast access.
// Sized to fill remaining constant memory after ROUND_CONSTANTS, MDS_COEFF, LOOKUP_TABLE,
// and d_gpu_range_* (~42KB used). Each node = 5 x uint64_t = 40 bytes.
// 1614 nodes x 40 bytes = 64,560 bytes, total constant usage ~65,520 bytes (< 64KB limit).
constexpr size_t TOP_TREE_CACHE_SIZE = 1614;

// d_top_tree_cache is defined in kernels.cu only
// Helper function is defined in kernels.cu where it can access the constant
__device__ Digest get_cached_node(size_t index);

// Top tree cache functions
bool initialize_top_tree_cache(const Digest* d_merkle_tree, size_t num_leafs);
void reset_top_tree_cache();

// ============================================================================
// Double-Buffered Async Mining
// ============================================================================
// This allows overlapping kernel execution with solution checking.
// While one batch is being mined on the GPU, we can check if the previous
// batch found a solution, achieving better GPU utilization.

struct AsyncMiningSlot {
    // Device-side output buffers (separate from GuesserBuffer's to allow double-buffering)
    uint64_t* d_solution_nonce;
    int* d_solution_found;
    Digest* d_solution_path_a;
    Digest* d_solution_path_b;
    Digest* d_solution_nonce_digest;
    Digest* d_solution_final_hash;
    
    // Host-side pinned memory for async copy
    int* h_solution_found_pinned;
    
    // CUDA stream and event for this slot
    cudaStream_t stream;
    cudaEvent_t completion_event;
    
    // State tracking
    bool allocated;
    bool kernel_launched;
    uint64_t batch_start_nonce;
    uint64_t batch_size;
    
    AsyncMiningSlot() 
        : d_solution_nonce(nullptr)
        , d_solution_found(nullptr)
        , d_solution_path_a(nullptr)
        , d_solution_path_b(nullptr)
        , d_solution_nonce_digest(nullptr)
        , d_solution_final_hash(nullptr)
        , h_solution_found_pinned(nullptr)
        , stream(nullptr)
        , completion_event(nullptr)
        , allocated(false)
        , kernel_launched(false)
        , batch_start_nonce(0)
        , batch_size(0) {}
    
    bool allocate();
    void free();
    bool reset();
};

struct DoubleBufferedMiner {
    AsyncMiningSlot slots[2];
    int current_slot;  // Which slot to launch next kernel on
    bool initialized;
    
    DoubleBufferedMiner() : current_slot(0), initialized(false) {}
    
    bool initialize();
    void cleanup();
    
    // Launch kernel on current slot (non-blocking)
    bool launch_async(
        GuesserBuffer& buffer,
        const Digest& target,
        const PowMastPaths& mast_paths,
        uint64_t start_nonce,
        uint64_t num_nonces,
        int consensus_rule_set);
    
    // Check if the OTHER slot (previous batch) has completed and found a solution
    // Returns: 0 = no solution, 1 = solution found, -1 = not ready yet
    int check_previous_slot(MiningSolution* out_solution, GuesserBuffer& buffer);
    
    // Wait for current slot to complete (blocking)
    bool wait_current_slot();
    
    // Swap to next slot
    void swap_slots() { current_slot = 1 - current_slot; }
    
    // Get previous slot index
    int previous_slot() const { return 1 - current_slot; }
};

// Async mining function - launches kernel and returns immediately
bool launch_mining_kernel_async(
    AsyncMiningSlot& slot,
    GuesserBuffer& buffer,
    const Digest& target,
    const PowMastPaths& mast_paths,
    uint64_t start_nonce,
    uint64_t num_nonces,
    int consensus_rule_set);

// Check if async kernel completed and get result
// Returns: 0 = no solution, 1 = solution found, -1 = kernel not finished
int check_async_mining_result(
    AsyncMiningSlot& slot,
    MiningSolution* out_solution,
    GuesserBuffer& buffer);

#endif
