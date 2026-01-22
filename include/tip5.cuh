#ifndef XNT_TIP5_CUH
#define XNT_TIP5_CUH

#include "common.cuh"

// ===== TIP5 ALGORITHM CONSTANTS =====
constexpr int STATE_SIZE = 16;
constexpr int NUM_ROUNDS = 5;
constexpr int DIGEST_LEN = 5;
constexpr int RATE = 10;
constexpr int CAPACITY = 6;

// ===== GOLDILOCKS FIELD CONSTANTS =====
constexpr uint64_t GOLDILOCKS_MODULUS = 0xFFFFFFFF00000001ULL;
constexpr uint64_t R2 = 0xFFFFFFFE00000001ULL;
constexpr uint64_t LOWER_MASK = 0xFFFFFFFFULL;
constexpr uint64_t BFE_ONE = 0x0000000000000001ULL;
constexpr uint64_t BFE_ZERO = 0x0000000000000000ULL;
[[maybe_unused]] constexpr size_t CHECKPOINT_DISTANCE = 1 << 19;

// ===== MERKLE/POW CONSTANTS =====
constexpr uint32_t NUM_INDEX_REPETITIONS = 63;
constexpr size_t BUDDING_ROUNDS = 32;
constexpr size_t NUM_BUD_LAYERS = 5;
[[maybe_unused]] constexpr size_t BUDS_PER_LEAF = 1 << NUM_BUD_LAYERS;
constexpr size_t MERKLE_TREE_HEIGHT_ = 27;  // POW_MEMORY_PARAMETER = 1 << 27
constexpr uint64_t MERKLE_INDEX_MASK = (1ULL << MERKLE_TREE_HEIGHT_) - 1ULL;
constexpr size_t MERKLE_NUM_LEAFS = (1ULL << MERKLE_TREE_HEIGHT_);

// ===== DOMAIN CONSTANTS =====
enum class Domain : uint64_t {
    FixedLength = BFE_ONE,
    VariableLength = BFE_ZERO
};

// ===== CONSTANT MEMORY DECLARATIONS =====
// These are defined in tip5.cu and accessed via extern
extern __constant__ uint32_t MDS_COEFF[STATE_SIZE];
extern __constant__ uint64_t ROUND_CONSTANTS[NUM_ROUNDS][STATE_SIZE];
extern __constant__ uint8_t LOOKUP_TABLE[256];
extern __constant__ uint64_t d_gpu_range_start;
extern __constant__ uint64_t d_gpu_range_size;

// ===== HOST LOOKUP TABLES =====
extern const uint8_t LOOKUP_TABLE_HOST[256];
extern const uint32_t MDS_COEFF_HOST[16];
extern const uint64_t ROUND_CONSTANTS_HOST[5][16];

// ===== DEVICE HELPER FUNCTIONS (BIT OPERATIONS) =====
__device__ __forceinline__ uint64_t fast_bitreverse_64(uint64_t val) {
    return __brevll(val);  // 64-bit hardware instruction
}

__device__ __forceinline__ uint64_t fast_bitreverse_64(uint64_t val, uint32_t bits) {
    return __brevll(val) >> (64 - bits);
}

// ===== MONTGOMERY REDUCTION =====
// Device version using PTX intrinsics
__device__ __forceinline__ uint64_t montyred_from_parts(uint64_t xh, uint64_t xl) {
    uint64_t shifted = xl << 32;
    uint64_t a = xl - shifted;
    bool e = a > xl;
    
    uint64_t b = a + xh;
    bool c = b < a;
    
    bool c2 = c || e;
    uint64_t r = c2 ? (b + GOLDILOCKS_MODULUS) : (b >= GOLDILOCKS_MODULUS ? b - GOLDILOCKS_MODULUS : b);
    return r;
}

// Host version using 128-bit arithmetic
__host__ inline uint64_t montyred_host(__uint128_t x) {
    uint64_t xl = static_cast<uint64_t>(x);
    uint64_t xh = static_cast<uint64_t>(x >> 64);
    
    uint64_t shifted = xl << 32;
    uint64_t a = xl - shifted;
    bool e = a > xl;
    
    uint64_t b = a + xh;
    bool c = b < a;
    
    uint64_t r = (c || e) ? (b + GOLDILOCKS_MODULUS) : b;
    if (r >= GOLDILOCKS_MODULUS) r -= GOLDILOCKS_MODULUS;
    return r;
}

// ===== FIELD ADDITION =====
__device__ __forceinline__ uint64_t fast_field_add(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    bool overflow = sum < a;
    bool needs_reduction = sum >= GOLDILOCKS_MODULUS;
    return (overflow || needs_reduction) ? (sum - GOLDILOCKS_MODULUS + (overflow ? GOLDILOCKS_MODULUS : 0)) : sum;
}

__host__ inline uint64_t fast_field_add_host(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    if (sum < a || sum >= GOLDILOCKS_MODULUS) {
        return sum - GOLDILOCKS_MODULUS + (sum < a ? GOLDILOCKS_MODULUS : 0);
    }
    return sum;
}

// ===== FIELD MULTIPLICATION (HOST) =====
__host__ inline uint64_t field_mul_host(uint64_t a, uint64_t b) {
    __uint128_t prod = static_cast<__uint128_t>(a) * b;
    return montyred_host(prod);
}

// ===== X^7 COMPUTATION =====
// Computes x^7 using field multiplication for S-box
__device__ __forceinline__ uint64_t x7_computer(uint64_t x) {
    uint64_t lo2, hi2;
    asm("mul.lo.u64 %0, %2, %2;\n\t"
        "mul.hi.u64 %1, %2, %2;" : "=l"(lo2), "=l"(hi2) : "l"(x));
    uint64_t x2 = montyred_from_parts(hi2, lo2);
    
    uint64_t lo4, hi4;
    asm("mul.lo.u64 %0, %2, %2;\n\t"
        "mul.hi.u64 %1, %2, %2;" : "=l"(lo4), "=l"(hi4) : "l"(x2));
    uint64_t x4 = montyred_from_parts(hi4, lo4);
    
    uint64_t lo6, hi6;
    asm("mul.lo.u64 %0, %2, %3;\n\t"
        "mul.hi.u64 %1, %2, %3;" : "=l"(lo6), "=l"(hi6) : "l"(x4), "l"(x2));
    uint64_t x6 = montyred_from_parts(hi6, lo6);
    
    uint64_t lo7, hi7;
    asm("mul.lo.u64 %0, %2, %3;\n\t"
        "mul.hi.u64 %1, %2, %3;" : "=l"(lo7), "=l"(hi7) : "l"(x6), "l"(x));
    return montyred_from_parts(hi7, lo7);
}

__host__ inline uint64_t x7_computer_host(uint64_t x) {
    uint64_t x2 = field_mul_host(x, x);
    uint64_t x4 = field_mul_host(x2, x2);
    uint64_t x6 = field_mul_host(x4, x2);
    return field_mul_host(x6, x);
}

// ===== S-BOX LOOKUP =====
// Device version using shared memory lookup table
__device__ __forceinline__ uint64_t split_lookup_shared(uint64_t element_in, const uint8_t* __restrict__ shared_lut) {
    uint64_t lo1, hi1;
    asm("mul.lo.u64 %0, %2, %3;\n\t"
        "mul.hi.u64 %1, %2, %3;" : "=l"(lo1), "=l"(hi1) : "l"(element_in), "l"(R2));
    uint64_t reduced_in = montyred_from_parts(hi1, lo1);
    
    uint8_t addr0 = reduced_in & 0xFF;
    uint8_t addr1 = (reduced_in >> 8) & 0xFF;
    uint8_t addr2 = (reduced_in >> 16) & 0xFF;
    uint8_t addr3 = (reduced_in >> 24) & 0xFF;
    uint8_t addr4 = (reduced_in >> 32) & 0xFF;
    uint8_t addr5 = (reduced_in >> 40) & 0xFF;
    uint8_t addr6 = (reduced_in >> 48) & 0xFF;
    uint8_t addr7 = (reduced_in >> 56) & 0xFF;
    
    uint64_t b0 = shared_lut[addr0];
    uint64_t b1 = shared_lut[addr1];
    uint64_t b2 = shared_lut[addr2];
    uint64_t b3 = shared_lut[addr3];
    uint64_t b4 = shared_lut[addr4];
    uint64_t b5 = shared_lut[addr5];
    uint64_t b6 = shared_lut[addr6];
    uint64_t b7 = shared_lut[addr7];
    
    uint64_t sbox_out = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | 
                        (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56);
    return x7_computer(sbox_out);
}

// Host version
__host__ inline uint64_t split_lookup_host(uint64_t element_in) {
    __uint128_t stage1 = static_cast<__uint128_t>(element_in) * R2;
    uint64_t reduced_in = montyred_host(stage1);
    
    uint64_t sbox_out = 0;
    for (int i = 0; i < 8; ++i) {
        uint8_t byte = (reduced_in >> (i * 8)) & 0xFF;
        sbox_out |= static_cast<uint64_t>(LOOKUP_TABLE_HOST[byte]) << (i * 8);
    }
    
    __uint128_t stage3 = static_cast<__uint128_t>(sbox_out) * sbox_out;
    return x7_computer_host(sbox_out);
}

// ===== MDS COEFFICIENT LOOKUP =====
__device__ __forceinline__ uint32_t get_mds_coeff(int i, int j) {
    int idx = (STATE_SIZE - i + j) % STATE_SIZE;
    return MDS_COEFF[idx];
}

__host__ inline uint32_t get_mds_coeff_host(int i, int j) {
    int idx = (STATE_SIZE - i + j) % STATE_SIZE;
    return MDS_COEFF_HOST[idx];
}

// ===== TIP5 LAYER FUNCTIONS =====
// S-box layer (applies S-box to first 4 elements, lookup to rest)
__device__ void sbox_layer(const uint64_t* __restrict__ state_in, uint64_t* __restrict__ state_out);
__device__ void mds_layer(const uint64_t* state_in, uint64_t* state_out);
__device__ void round_constants_layer(int round_index, const uint64_t* state_in, uint64_t* state_out);
__device__ void generated_function(const uint64_t* input, uint64_t* output);

__host__ void sbox_layer_host(const uint64_t* state_in, uint64_t* state_out);
__host__ void mds_layer_host(const uint64_t* state_in, uint64_t* state_out);
__host__ void round_constants_layer_host(int round_index, const uint64_t* state_in, uint64_t* state_out);

// ===== TIP5 PERMUTATION =====
__device__ void tip5_permutation(uint64_t* state);
__host__ void tip5_permutation_host(uint64_t* state);

// ===== TIP5 SPONGE CONSTRUCTION =====
__device__ __forceinline__ void tip5_sponge_init(uint64_t* __restrict__ state, Domain domain) {
    uint64_t capacity_val = static_cast<uint64_t>(domain);
    
    #pragma unroll
    for (int i = 0; i < RATE; ++i) {
        state[i] = 0;
    }
    state[RATE] = capacity_val;
    #pragma unroll
    for (int i = RATE + 1; i < STATE_SIZE; ++i) {
        state[i] = 0;
    }
}

__device__ __forceinline__ void tip5_sponge_absorb_chunk(uint64_t* __restrict__ state, 
                                                          const uint64_t* __restrict__ chunk, 
                                                          size_t len) {
    for (size_t i = 0; i < len && i < RATE; ++i) {
        state[i] = chunk[i];
    }
    tip5_permutation(state);
}

__device__ __forceinline__ void tip5_sponge_squeeze(uint64_t* __restrict__ state, 
                                                     uint64_t* __restrict__ digest) {
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        digest[i] = state[i];
    }
}

// Host versions
__host__ void tip5_sponge_init_host(uint64_t* state, Domain domain);
__host__ void tip5_sponge_permutation_host(uint64_t* state);
__host__ void tip5_sponge_absorb_chunk_host(uint64_t* state, const uint64_t* chunk, size_t len);
__host__ void tip5_sponge_squeeze_host(uint64_t* state, uint64_t* digest);

// ===== INITIALIZATION =====
// Initialize constant memory on the GPU (call once at startup)
void tip5_init_constants();

#endif // XNT_TIP5_CUH