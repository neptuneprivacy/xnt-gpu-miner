#ifndef XNT_POW_CUH
#define XNT_POW_CUH

#include "merkle.cuh"

class Pow;
struct GuesserBuffer;
struct PowMastPaths;

struct PowMastPaths {
    Digest pow[3];
    Digest header[2];
    Digest kernel[1];
    
    __device__ __host__ PowMastPaths() {
        #pragma unroll
        for (int i = 0; i < 3; ++i) pow[i] = Digest::default_digest();
        #pragma unroll
        for (int i = 0; i < 2; ++i) header[i] = Digest::default_digest();
        kernel[0] = Digest::default_digest();
    }
    
    __device__ Digest commit_device() const;
    __host__ Digest commit() const;
    Digest fast_mast_hash(const Pow& pow_obj) const;
    __device__ Digest fast_mast_hash_device(const Pow& pow_obj) const;
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
        , mast_paths() {}
    
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
    }
    
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
        , mast_paths(other.mast_paths) {
        other.d_merkle_tree = nullptr;
        other.tree_size = 0;
        other.d_leafs = nullptr;
        other.num_leafs = 0;
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
            other.d_merkle_tree = nullptr;
            other.tree_size = 0;
            other.d_leafs = nullptr;
            other.num_leafs = 0;
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
        #pragma unroll
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

uint64_t generate_secure_random_start(
    const std::string& puzzle_id, int gpu_id, const std::string& gpu_uuid);

extern __constant__ uint64_t d_gpu_range_start;
extern __constant__ uint64_t d_gpu_range_size;

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
