#ifndef XNT_POW_CUH
#define XNT_POW_CUH

#include "merkle.cuh"

class Pow;
struct GuesserBuffer;
struct PowMastPaths;
struct MiningOutputBuffers;

struct PowMastPaths {
    Digest pow[3];
    Digest header[2];
    Digest kernel[1];
    
    __device__ __host__ PowMastPaths() {
        for (int i = 0; i < 3; ++i) pow[i] = Digest::default_digest();
        for (int i = 0; i < 2; ++i) header[i] = Digest::default_digest();
        kernel[0] = Digest::default_digest();
    }
    
    __device__ Digest commit_device() const;
    __host__ Digest commit() const;
    Digest fast_mast_hash(const Pow& pow_obj) const;
    __device__ __noinline__ Digest fast_mast_hash_device(const Pow& pow_obj) const;
};

class GuesserBuffer {
public:
    Digest merkle_root;
    Digest hash;
    Digest index_picker_preimage;
    Digest prev_block_digest;
    Digest* d_merkle_tree;
    size_t tree_size;
    Digest* d_leafs;
    size_t num_leafs;
    int consensus_rule_set;
    PowMastPaths mast_paths;
    
    // Persistent mining resources to avoid per-batch alloc/free overhead
    cudaStream_t mining_stream;
    bool stream_initialized;
    
    // Persistent output buffers (allocated on first use)
    uint64_t* d_solution_nonce;
    int* d_solution_found;
    Digest* d_solution_path_a;
    Digest* d_solution_path_b;
    Digest* d_solution_nonce_digest;
    Digest* d_solution_final_hash;
    bool output_buffers_allocated;
    
    // Phase-split intermediate buffer (index_a, index_b per nonce; 16 bytes/nonce)
    uint64_t* d_phase_indices;
    size_t d_phase_indices_capacity;
    
    // GPU range initialization (only need to set once per GPU)
    bool gpu_range_initialized;
    // L2 persistence hint set for d_leafs (P0.3 optimization)
    bool l2_persist_set;
    
    GuesserBuffer() 
        : merkle_root()
        , hash()
        , index_picker_preimage()
        , prev_block_digest()
        , d_merkle_tree(nullptr)
        , tree_size(0)
        , d_leafs(nullptr)
        , num_leafs(0)
        , consensus_rule_set(CONSENSUS_XNT)
        , mast_paths()
        , mining_stream(nullptr)
        , stream_initialized(false)
        , d_solution_nonce(nullptr)
        , d_solution_found(nullptr)
        , d_solution_path_a(nullptr)
        , d_solution_path_b(nullptr)
        , d_solution_nonce_digest(nullptr)
        , d_solution_final_hash(nullptr)
        , output_buffers_allocated(false)
        , d_phase_indices(nullptr)
        , d_phase_indices_capacity(0)
        , gpu_range_initialized(false)
        , l2_persist_set(false) {}
    
    ~GuesserBuffer() { cleanup(); }
    
    void cleanup() {
        if (d_merkle_tree) {
            cudaFree(d_merkle_tree);
            d_merkle_tree = nullptr;
            tree_size = 0;
        }
        if (d_leafs) {
            cudaFree(d_leafs);
            d_leafs = nullptr;
            num_leafs = 0;
        }
        cleanup_mining_resources();
    }
    
    void cleanup_mining_resources() {
        if (d_solution_nonce) { cudaFree(d_solution_nonce); d_solution_nonce = nullptr; }
        if (d_solution_found) { cudaFree(d_solution_found); d_solution_found = nullptr; }
        if (d_solution_path_a) { cudaFree(d_solution_path_a); d_solution_path_a = nullptr; }
        if (d_solution_path_b) { cudaFree(d_solution_path_b); d_solution_path_b = nullptr; }
        if (d_solution_nonce_digest) { cudaFree(d_solution_nonce_digest); d_solution_nonce_digest = nullptr; }
        if (d_solution_final_hash) { cudaFree(d_solution_final_hash); d_solution_final_hash = nullptr; }
        if (d_phase_indices) { cudaFree(d_phase_indices); d_phase_indices = nullptr; }
        d_phase_indices_capacity = 0;
        output_buffers_allocated = false;
        
        if (stream_initialized && mining_stream) {
            cudaStreamDestroy(mining_stream);
            mining_stream = nullptr;
            stream_initialized = false;
        }
    }
    
    bool ensure_mining_resources();  // Allocates stream and output buffers if needed
    bool reset_output_buffers();     // Resets solution_found flag
    
    GuesserBuffer(const GuesserBuffer&) = delete;
    GuesserBuffer& operator=(const GuesserBuffer&) = delete;
    
    GuesserBuffer(GuesserBuffer&& other) noexcept
        : merkle_root(other.merkle_root)
        , hash(other.hash)
        , index_picker_preimage(other.index_picker_preimage)
        , prev_block_digest(other.prev_block_digest)
        , d_merkle_tree(other.d_merkle_tree)
        , tree_size(other.tree_size)
        , d_leafs(other.d_leafs)
        , num_leafs(other.num_leafs)
        , consensus_rule_set(other.consensus_rule_set)
        , mast_paths(other.mast_paths)
        , mining_stream(other.mining_stream)
        , stream_initialized(other.stream_initialized)
        , d_solution_nonce(other.d_solution_nonce)
        , d_solution_found(other.d_solution_found)
        , d_solution_path_a(other.d_solution_path_a)
        , d_solution_path_b(other.d_solution_path_b)
        , d_solution_nonce_digest(other.d_solution_nonce_digest)
        , d_solution_final_hash(other.d_solution_final_hash)
        , output_buffers_allocated(other.output_buffers_allocated)
        , d_phase_indices(other.d_phase_indices)
        , d_phase_indices_capacity(other.d_phase_indices_capacity)
        , gpu_range_initialized(other.gpu_range_initialized)
        , l2_persist_set(other.l2_persist_set) {
        other.d_merkle_tree = nullptr;
        other.tree_size = 0;
        other.d_leafs = nullptr;
        other.num_leafs = 0;
        other.mining_stream = nullptr;
        other.stream_initialized = false;
        other.d_solution_nonce = nullptr;
        other.d_solution_found = nullptr;
        other.d_solution_path_a = nullptr;
        other.d_solution_path_b = nullptr;
        other.d_solution_nonce_digest = nullptr;
        other.d_solution_final_hash = nullptr;
        other.d_phase_indices = nullptr;
        other.d_phase_indices_capacity = 0;
        other.output_buffers_allocated = false;
        other.gpu_range_initialized = false;
        other.l2_persist_set = false;
    }
    
    GuesserBuffer& operator=(GuesserBuffer&& other) noexcept {
        if (this != &other) {
            cleanup();
            merkle_root = other.merkle_root;
            hash = other.hash;
            index_picker_preimage = other.index_picker_preimage;
            prev_block_digest = other.prev_block_digest;
            d_merkle_tree = other.d_merkle_tree;
            tree_size = other.tree_size;
            d_leafs = other.d_leafs;
            num_leafs = other.num_leafs;
            consensus_rule_set = other.consensus_rule_set;
            mast_paths = other.mast_paths;
            mining_stream = other.mining_stream;
            stream_initialized = other.stream_initialized;
            d_solution_nonce = other.d_solution_nonce;
            d_solution_found = other.d_solution_found;
            d_solution_path_a = other.d_solution_path_a;
            d_solution_path_b = other.d_solution_path_b;
            d_solution_nonce_digest = other.d_solution_nonce_digest;
            d_solution_final_hash = other.d_solution_final_hash;
            d_phase_indices = other.d_phase_indices;
            d_phase_indices_capacity = other.d_phase_indices_capacity;
            output_buffers_allocated = other.output_buffers_allocated;
            gpu_range_initialized = other.gpu_range_initialized;
            l2_persist_set = other.l2_persist_set;
            other.d_merkle_tree = nullptr;
            other.tree_size = 0;
            other.d_leafs = nullptr;
            other.num_leafs = 0;
            other.mining_stream = nullptr;
            other.stream_initialized = false;
            other.d_solution_nonce = nullptr;
            other.d_solution_found = nullptr;
            other.d_solution_path_a = nullptr;
            other.d_solution_path_b = nullptr;
            other.d_solution_nonce_digest = nullptr;
            other.d_solution_final_hash = nullptr;
            other.d_phase_indices = nullptr;
            other.d_phase_indices_capacity = 0;
            other.output_buffers_allocated = false;
        }
        return *this;
    }
    
    bool is_valid() const { return d_merkle_tree != nullptr && tree_size > 0; }
};

enum class VramMode {
    HIGH_VRAM,
    LOW_VRAM
};

VramMode detect_vram_mode(int gpu_id = 0);

inline const char* vram_mode_name(VramMode mode) {
    switch (mode) {
        case VramMode::HIGH_VRAM: return "HIGH_VRAM";
        case VramMode::LOW_VRAM: return "LOW_VRAM";
        default: return "UNKNOWN";
    }
}

class Pow {
public:
    static constexpr size_t NUM_LEAFS = MERKLE_NUM_LEAFS;
    
    Digest root;
    Digest path_a[MERKLE_TREE_HEIGHT_];
    Digest path_b[MERKLE_TREE_HEIGHT_];
    Digest nonce;
    
    __device__ __host__ Pow() : root(), nonce() {
        for (int i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
            path_a[i] = Digest::default_digest();
            path_b[i] = Digest::default_digest();
        }
    }
    
    __device__ static Digest bud(const Digest& commitment, uint64_t index);
    __host__ static Digest bud_host(const Digest& commitment, uint64_t index);
    
    __device__ static void indices(const Digest& hash, const Digest& nonce, 
                                   uint64_t& index_a, uint64_t& index_b);
    __host__ static std::pair<uint64_t, uint64_t> indices(const Digest& hash, const Digest& nonce);
    
    __host__ static Digest compute_leaf_from_commitment_host(
        const Digest& commitment, uint64_t index, uint64_t num_leafs);
    
    __host__ static bool verify_merkle_path_host(
        const Digest& root, uint64_t index, const Digest* path, const Digest& element);
    
    __host__ static uint64_t bitreverse_host(uint64_t n, uint32_t log2_n);
    
    std::vector<uint64_t> encode() const;
    
    static GuesserBuffer preprocess(
        const PowMastPaths& mast_auth_paths,
        const Digest& prev_block_digest,
        int consensus_rule_set = CONSENSUS_XNT,
        bool* cancel_flag = nullptr);
    
    __host__ static GuesserBuffer preprocess_gpu(
        const PowMastPaths& mast_auth_paths,
        const Digest& prev_block_digest,
        int consensus_rule_set = CONSENSUS_XNT,
        bool* cancel_flag = nullptr);
    
    __host__ static GuesserBuffer preprocess_gpu_high_vram(
        const PowMastPaths& mast_auth_paths,
        const Digest& prev_block_digest,
        int consensus_rule_set = CONSENSUS_XNT,
        bool* cancel_flag = nullptr);
    
    __host__ static GuesserBuffer preprocess_gpu_low_vram(
        const PowMastPaths& mast_auth_paths,
        const Digest& prev_block_digest,
        int consensus_rule_set = CONSENSUS_XNT,
        bool* cancel_flag = nullptr);
};

__device__ void Pow_indices_device(
    const Digest& hash, const Digest& nonce, 
    uint64_t& index_a, uint64_t& index_b);

// Direct MAST hash - reads paths from global memory without building Pow struct
__device__ __noinline__ Digest fast_mast_hash_direct_low_vram(
    const PowMastPaths& mast_paths,
    const Digest& nonce,
    const Digest& root,
    const Digest* __restrict__ d_internal_nodes,
    uint64_t path_index_a,
    uint64_t path_index_b,
    size_t num_leafs,
    size_t merkle_height,
    size_t stored_nodes_count,
    const Digest& commitment,
    const Digest& leaf_prefix);

__device__ __noinline__ Digest fast_mast_hash_direct(
    const PowMastPaths& mast_paths,
    const Digest& nonce,
    const Digest& root,
    const Digest* __restrict__ d_leafs,
    const Digest* __restrict__ d_internal_nodes,
    uint64_t path_index_a,
    uint64_t path_index_b,
    size_t num_leafs,
    size_t merkle_height);

uint64_t generate_secure_random_start(
    const std::string& puzzle_id, int gpu_id, 
    const std::string& worker_id = "", const std::string& gpu_uuid = "");

// GPU Range Calculation with Mining Pool Support
struct GpuNonceRange {
    uint64_t range_start;
    uint64_t range_size;
    int miner_id;
    int total_miners;
    int gpu_id;
    int total_gpus;
};

GpuNonceRange calculate_gpu_range(int gpu_id, int total_gpus = 8);

// d_gpu_range_start and d_gpu_range_size are declared in tip5.cuh

bool verify_pow_solution(
    const Pow& pow, const Digest& hash, 
    const Digest& target, const Digest& commitment);

struct AuthPaths {
    std::vector<std::string> pow;
    std::vector<std::string> header;
    std::vector<std::string> kernel;
};

PowMastPaths convertToPowMastPaths(const AuthPaths& auth_paths);

#endif
