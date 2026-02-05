#ifndef XNT_TIP5_CUH
#define XNT_TIP5_CUH

#include "common.cuh"

#ifdef _WIN32
    #ifdef _MSC_VER
        #include <intrin.h>
        struct host_uint128 {
            uint64_t low;
            uint64_t high;
            host_uint128() : low(0), high(0) {}
            host_uint128(uint64_t l) : low(l), high(0) {}
            host_uint128(uint64_t l, uint64_t h) : low(l), high(h) {}
        };
    #endif
#endif

constexpr int STATE_SIZE = 16;
constexpr int NUM_ROUNDS = 5;
constexpr int DIGEST_LEN = 5;
constexpr int RATE = 10;
constexpr int CAPACITY = 6;

constexpr uint64_t GOLDILOCKS_MODULUS = 0xFFFFFFFF00000001ULL;
constexpr uint64_t R2 = 0xFFFFFFFE00000001ULL;
constexpr uint64_t BFE_ONE = 0x0000000000000001ULL;
constexpr uint64_t BFE_ZERO = 0x0000000000000000ULL;

constexpr uint32_t NUM_INDEX_REPETITIONS = 63;
constexpr size_t BUDDING_ROUNDS = 32;
constexpr size_t NUM_BUD_LAYERS = 5;
constexpr size_t MERKLE_TREE_HEIGHT_ = 27;
constexpr uint64_t MERKLE_INDEX_MASK = (1ULL << MERKLE_TREE_HEIGHT_) - 1ULL;
constexpr size_t MERKLE_NUM_LEAFS = (1ULL << MERKLE_TREE_HEIGHT_);

enum class Domain : uint64_t {
    FixedLength = BFE_ONE,
    VariableLength = BFE_ZERO
};

// These are defined in tip5.cu
#ifndef TIP5_DEFINING_CONSTANTS
extern __constant__ uint32_t MDS_COEFF[STATE_SIZE];
extern __constant__ uint64_t ROUND_CONSTANTS[NUM_ROUNDS][STATE_SIZE];
extern __constant__ uint8_t LOOKUP_TABLE[256];
#endif

// These are defined in kernels.cu
#ifndef KERNELS_DEFINING_RANGE_CONSTANTS
extern __constant__ uint64_t d_gpu_range_start;
extern __constant__ uint64_t d_gpu_range_size;
#endif

extern const uint8_t LOOKUP_TABLE_HOST[256];
extern const uint32_t MDS_COEFF_HOST[16];
extern const uint64_t ROUND_CONSTANTS_HOST[5][16];

__device__ __forceinline__ uint64_t fast_bitreverse_64(uint64_t val) {
    return __brevll(val);
}

__device__ __forceinline__ uint64_t fast_bitreverse_64(uint64_t val, uint32_t bits) {
    return __brevll(val) >> (64 - bits);
}

// Optimized Montgomery reduction for Goldilocks prime p = 2^64 - 2^32 + 1
// Input: 128-bit value (xh, xl) where result = (xh * 2^64 + xl) * R^(-1) mod p
// Uses PTX setp/selp for truly branchless execution
__device__ __forceinline__ uint64_t montyred_from_parts(uint64_t xh, uint64_t xl) {
    // Step 1: t = xl + (xl << 32), detect carry
    uint64_t xl_lo = xl << 32;
    uint64_t t = xl + xl_lo;

    // Branchless carry detection: carry = (t < xl) ? 1 : 0
    uint64_t carry;
    asm("{\n\t"
        ".reg .pred p;\n\t"
        "setp.lt.u64 p, %1, %2;\n\t"
        "selp.u64 %0, 1, 0, p;\n\t"
        "}" : "=l"(carry) : "l"(t), "l"(xl));

    // Step 2: u = t - (t >> 32) - carry
    uint64_t t_hi = t >> 32;
    uint64_t u = t - t_hi - carry;

    // Step 3: result = xh - u, with conditional correction
    uint64_t result = xh - u;

    // Branchless underflow correction: if xh < u, subtract 0xFFFFFFFF
    uint64_t correction;
    asm("{\n\t"
        ".reg .pred q;\n\t"
        "setp.lt.u64 q, %1, %2;\n\t"
        "selp.u64 %0, 4294967295, 0, q;\n\t"
        "}" : "=l"(correction) : "l"(xh), "l"(u));
    result -= correction;

    return result;
}

#ifdef _WIN32
    #ifdef _MSC_VER
        __host__ inline uint64_t montyred_host(const host_uint128& x) {
            uint64_t xl = x.low;
            uint64_t xh = x.high;
    #else
        __host__ inline uint64_t montyred_host(__uint128_t x) {
            uint64_t xl = static_cast<uint64_t>(x);
            uint64_t xh = static_cast<uint64_t>(x >> 64);
    #endif
#else
__host__ inline uint64_t montyred_host(__uint128_t x) {
    uint64_t xl = static_cast<uint64_t>(x);
    uint64_t xh = static_cast<uint64_t>(x >> 64);
#endif
    uint64_t shifted = xl << 32;
    uint64_t a = xl + shifted;
    bool e = (a < xl) || (a < shifted);

    uint64_t b = a - (a >> 32);
    if (e) b -= 1;

    bool c = (xh < b);
    uint64_t r = xh - b;

    return r - (0xFFFFFFFFULL & (c ? 0xFFFFFFFFFFFFFFFFULL : 0));
}

__device__ __forceinline__ uint64_t fast_field_add(uint64_t a, uint64_t b) {
    // OPTIMIZATION #11: Simplified logic - only one condition needed
    uint64_t sum = a + b;
    bool overflow = (sum < a);
    bool needs_reduction = overflow || (sum >= GOLDILOCKS_MODULUS);
    return sum - (needs_reduction ? GOLDILOCKS_MODULUS : 0);
}

__host__ inline uint64_t fast_field_add_host(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    if (sum < a || sum >= GOLDILOCKS_MODULUS) {
        return sum - GOLDILOCKS_MODULUS + (sum < a ? GOLDILOCKS_MODULUS : 0);
    }
    return sum;
}

__host__ inline uint64_t field_mul_host(uint64_t a, uint64_t b) {
#ifdef _WIN32
    #ifdef _MSC_VER
        unsigned __int64 high;
        unsigned __int64 low = _umul128(a, b, &high);
        host_uint128 prod(low, high);
    #else
        __uint128_t prod = static_cast<__uint128_t>(a) * b;
    #endif
#else
    __uint128_t prod = static_cast<__uint128_t>(a) * b;
#endif
    return montyred_host(prod);
}

__device__ __forceinline__ uint64_t x7_computer_pipelined(uint64_t x) {
    uint64_t lo2 = x * x;
    uint64_t hi2 = __umul64hi(x, x);
    uint64_t x2 = montyred_from_parts(hi2, lo2);

    uint64_t lo4 = x2 * x2;
    uint64_t hi4 = __umul64hi(x2, x2);
    uint64_t x4 = montyred_from_parts(hi4, lo4);

    uint64_t lo6 = x4 * x2;
    uint64_t hi6 = __umul64hi(x4, x2);
    uint64_t x6 = montyred_from_parts(hi6, lo6);

    uint64_t lo7 = x6 * x;
    uint64_t hi7 = __umul64hi(x6, x);
    return montyred_from_parts(hi7, lo7);
}

__device__ __forceinline__ uint64_t x7_computer(uint64_t x) {
    return x7_computer_pipelined(x);
}

__host__ inline uint64_t x7_computer_host(uint64_t x) {
    uint64_t x2 = field_mul_host(x, x);
    uint64_t x4 = field_mul_host(x2, x2);
    uint64_t x6 = field_mul_host(x4, x2);
    return field_mul_host(x6, x);
}

__device__ __forceinline__ uint64_t split_lookup_shared(uint64_t element_in, const uint8_t* __restrict__ shared_lut) {
    uint64_t lo1 = element_in * R2;
    uint64_t hi1 = __umul64hi(element_in, R2);
    uint64_t reduced_in = montyred_from_parts(hi1, lo1);

    uint8_t addr0 = (uint8_t)(reduced_in >> 0);
    uint8_t addr1 = (uint8_t)(reduced_in >> 8);
    uint8_t addr2 = (uint8_t)(reduced_in >> 16);
    uint8_t addr3 = (uint8_t)(reduced_in >> 24);
    uint8_t addr4 = (uint8_t)(reduced_in >> 32);
    uint8_t addr5 = (uint8_t)(reduced_in >> 40);
    uint8_t addr6 = (uint8_t)(reduced_in >> 48);
    uint8_t addr7 = (uint8_t)(reduced_in >> 56);

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

    return montyred_from_parts(0, sbox_out);
}

__host__ inline uint64_t split_lookup_host(uint64_t element_in) {
#ifdef _WIN32
    #ifdef _MSC_VER
        unsigned __int64 high;
        unsigned __int64 low = _umul128(element_in, R2, &high);
        host_uint128 stage1(low, high);
    #else
        __uint128_t stage1 = static_cast<__uint128_t>(element_in) * R2;
    #endif
#else
    __uint128_t stage1 = static_cast<__uint128_t>(element_in) * R2;
#endif
    uint64_t reduced_in = montyred_host(stage1);

    uint64_t sbox_out = 0;
    for (int i = 0; i < 8; ++i) {
        uint8_t byte = (reduced_in >> (i * 8)) & 0xFF;
        sbox_out |= static_cast<uint64_t>(LOOKUP_TABLE_HOST[byte]) << (i * 8);
    }

#ifdef _WIN32
    #ifdef _MSC_VER
        host_uint128 stage3(sbox_out, 0);
    #else
        __uint128_t stage3 = static_cast<__uint128_t>(sbox_out);
    #endif
#else
    __uint128_t stage3 = static_cast<__uint128_t>(sbox_out);
#endif
    return montyred_host(stage3);
}

__device__ __forceinline__ uint32_t get_mds_coeff(int i, int j) {
    // STATE_SIZE is 16, use bitmask for modulo 16
    int idx = (i - j) & (STATE_SIZE - 1);
    return MDS_COEFF[idx];
}

__host__ inline uint32_t get_mds_coeff_host(int i, int j) {
    int idx = (i - j) & (STATE_SIZE - 1);
    return MDS_COEFF_HOST[idx];
}

__device__ void sbox_layer(const uint64_t* __restrict__ state_in, uint64_t* __restrict__ state_out);
__device__ void mds_layer(const uint64_t* state_in, uint64_t* state_out);
__device__ void round_constants_layer(int round_index, const uint64_t* state_in, uint64_t* state_out);
__device__ void generated_function(const uint64_t* input, uint64_t* output);

__host__ void sbox_layer_host(const uint64_t* state_in, uint64_t* state_out);
__host__ void mds_layer_host(const uint64_t* state_in, uint64_t* state_out);
__host__ void round_constants_layer_host(int round_index, const uint64_t* state_in, uint64_t* state_out);

__device__ __noinline__ void tip5_permutation(uint64_t* state);
__host__ void tip5_permutation_host(uint64_t* state);

__device__ __forceinline__ void tip5_sponge_init(uint64_t* __restrict__ state, Domain domain) {
    // OPTIMIZATION #16: Combined initialization in single loop
    uint64_t capacity_val = (domain == Domain::FixedLength) ? static_cast<uint64_t>(domain) : BFE_ZERO;

    for (int i = 0; i < STATE_SIZE; ++i) {
        state[i] = (i < RATE) ? BFE_ZERO : capacity_val;
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
    for (int i = 0; i < DIGEST_LEN; ++i) {
        digest[i] = state[i];
    }
}

__host__ void tip5_sponge_init_host(uint64_t* state, Domain domain);
__host__ void tip5_sponge_permutation_host(uint64_t* state);
__host__ void tip5_sponge_absorb_chunk_host(uint64_t* state, const uint64_t* chunk, size_t len);
__host__ void tip5_sponge_squeeze_host(uint64_t* state, uint64_t* digest);

void tip5_init_constants();

#endif
