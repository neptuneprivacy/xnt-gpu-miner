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
// Uses 32-bit arithmetic with carry chain for better GPU efficiency
__device__ __forceinline__ uint64_t montyred_from_parts(uint64_t xh, uint64_t xl) {
    uint64_t result;
    asm("{\n\t"
        ".reg .u64 neg_m, u;\n\t"
        ".reg .u32 xl_lo, xl_hi, neg_m_lo, neg_m_hi, c;\n\t"
        ".reg .pred q;\n\t"
        // Split xl into 32-bit parts
        "mov.b64 {xl_lo, xl_hi}, %1;\n\t"
        // neg_m = xl + (xl << 32) using 32-bit arithmetic with carry
        "add.cc.u32 neg_m_lo, xl_lo, 0;\n\t"
        "addc.cc.u32 neg_m_hi, xl_hi, xl_lo;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        // Reconstruct neg_m
        "mov.b64 neg_m, {neg_m_lo, neg_m_hi};\n\t"
        // u = neg_m - neg_m_hi - c
        "cvt.u64.u32 u, neg_m_hi;\n\t"
        "sub.u64 u, neg_m, u;\n\t"
        "cvt.u64.u32 neg_m, c;\n\t"
        "sub.u64 u, u, neg_m;\n\t"
        // result = xh - u
        "sub.u64 %0, %2, u;\n\t"
        // Branchless underflow correction
        "setp.lt.u64 q, %2, u;\n\t"
        "selp.u64 u, 4294967295, 0, q;\n\t"
        "sub.u64 %0, %0, u;\n\t"
        "}" : "=l"(result) : "l"(xl), "l"(xh));
    return result;
}

// PTX-optimized field multiplication: computes (a * b) mod p in Montgomery form
// Fuses 64x64->128 multiplication with Montgomery reduction for better register usage
// Uses funnel shift (shf) for efficient 32-bit extraction from 64-bit values
__device__ __forceinline__ uint64_t field_mul_ptx(uint64_t a, uint64_t b) {
    uint64_t result;
    asm("{\n\t"
        ".reg .u64 lo, hi, neg_m, u;\n\t"
        ".reg .u32 lo_lo, lo_hi, neg_m_lo, neg_m_hi, t32, c;\n\t"
        ".reg .pred p, q;\n\t"
        // 64x64 -> 128-bit multiplication
        "mul.lo.u64 lo, %1, %2;\n\t"
        "mul.hi.u64 hi, %1, %2;\n\t"
        // Split lo into 32-bit parts for efficient computation
        "mov.b64 {lo_lo, lo_hi}, lo;\n\t"
        // neg_m = lo + (lo << 32) using 32-bit arithmetic with carry
        // neg_m_lo = lo_lo + 0 = lo_lo (low 32 bits of lo << 32 is 0)
        // neg_m_hi = lo_hi + lo_lo + carry
        "add.cc.u32 neg_m_lo, lo_lo, 0;\n\t"        // neg_m_lo = lo_lo, no carry in
        "addc.cc.u32 neg_m_hi, lo_hi, lo_lo;\n\t"   // neg_m_hi = lo_hi + lo_lo + 0
        "addc.u32 c, 0, 0;\n\t"                      // c = carry out
        // Reconstruct neg_m
        "mov.b64 neg_m, {neg_m_lo, neg_m_hi};\n\t"
        // u = neg_m - (neg_m >> 32) - carry = neg_m - neg_m_hi - c
        // Using 64-bit: u = neg_m - (uint64_t)neg_m_hi - c
        "cvt.u64.u32 u, neg_m_hi;\n\t"
        "sub.u64 u, neg_m, u;\n\t"
        "cvt.u64.u32 neg_m, c;\n\t"                  // reuse neg_m for carry
        "sub.u64 u, u, neg_m;\n\t"
        // result = hi - u
        "sub.u64 %0, hi, u;\n\t"
        // Branchless underflow correction
        "setp.lt.u64 q, hi, u;\n\t"
        "selp.u64 u, 4294967295, 0, q;\n\t"
        "sub.u64 %0, %0, u;\n\t"
        "}" : "=l"(result) : "l"(a), "l"(b));
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
    // PTX-optimized branchless field addition for Goldilocks prime p = 2^64 - 2^32 + 1
    // Uses add.cc to detect overflow via carry flag
    uint64_t result;
    asm("{\n\t"
        ".reg .u64 sum, correction;\n\t"
        ".reg .pred p, q, r;\n\t"
        "add.cc.u64 sum, %1, %2;\n\t"          // sum = a + b, set carry on overflow
        "setp.lt.u64 p, sum, %1;\n\t"          // p = overflow (sum < a)
        "setp.hs.u64 q, sum, %3;\n\t"          // q = (sum >= GOLDILOCKS_MODULUS)
        "or.pred r, p, q;\n\t"                  // r = needs_reduction
        "selp.u64 correction, %3, 0, r;\n\t"   // correction = r ? p : 0
        "sub.u64 %0, sum, correction;\n\t"     // result = sum - correction
        "}" : "=l"(result) : "l"(a), "l"(b), "l"(GOLDILOCKS_MODULUS));
    return result;
}

// Goldilocks opt: 2*x = x+x (one add vs full mul) - for constant 2
__device__ __forceinline__ uint64_t field_mul_by_2_ptx(uint64_t x) {
    return fast_field_add(x, x);
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
    // Use fused PTX multiply-reduce for better register efficiency
    uint64_t x2 = field_mul_ptx(x, x);
    uint64_t x4 = field_mul_ptx(x2, x2);
    uint64_t x6 = field_mul_ptx(x4, x2);
    return field_mul_ptx(x6, x);
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
    // Use fused PTX multiply-reduce for R2 multiplication
    uint64_t reduced_in = field_mul_ptx(element_in, R2);

    // PTX-optimized byte extraction using bfe (bit field extract)
    // Extract each byte directly into uint32_t for efficient LUT indexing
    uint32_t addr0, addr1, addr2, addr3, addr4, addr5, addr6, addr7;
    asm("bfe.u32 %0, %8, 0, 8;\n\t"
        "bfe.u32 %1, %8, 8, 8;\n\t"
        "bfe.u32 %2, %8, 16, 8;\n\t"
        "bfe.u32 %3, %8, 24, 8;\n\t"
        "bfe.u32 %4, %9, 0, 8;\n\t"
        "bfe.u32 %5, %9, 8, 8;\n\t"
        "bfe.u32 %6, %9, 16, 8;\n\t"
        "bfe.u32 %7, %9, 24, 8;"
        : "=r"(addr0), "=r"(addr1), "=r"(addr2), "=r"(addr3),
          "=r"(addr4), "=r"(addr5), "=r"(addr6), "=r"(addr7)
        : "r"((uint32_t)reduced_in), "r"((uint32_t)(reduced_in >> 32)));

    // LUT lookups
    uint64_t b0 = shared_lut[addr0];
    uint64_t b1 = shared_lut[addr1];
    uint64_t b2 = shared_lut[addr2];
    uint64_t b3 = shared_lut[addr3];
    uint64_t b4 = shared_lut[addr4];
    uint64_t b5 = shared_lut[addr5];
    uint64_t b6 = shared_lut[addr6];
    uint64_t b7 = shared_lut[addr7];

    // PTX-optimized byte packing using bfi (bit field insert)
    uint32_t lo_result, hi_result;
    asm("{\n\t"
        ".reg .u32 t0, t1;\n\t"
        // Pack low 4 bytes: b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        "bfi.b32 t0, %3, %2, 8, 8;\n\t"   // t0 = b0 | (b1 << 8)
        "bfi.b32 t1, %5, %4, 8, 8;\n\t"   // t1 = b2 | (b3 << 8)
        "bfi.b32 %0, t1, t0, 16, 16;\n\t" // lo = t0 | (t1 << 16)
        // Pack high 4 bytes: b4 | (b5 << 8) | (b6 << 16) | (b7 << 24)
        "bfi.b32 t0, %7, %6, 8, 8;\n\t"   // t0 = b4 | (b5 << 8)
        "bfi.b32 t1, %9, %8, 8, 8;\n\t"   // t1 = b6 | (b7 << 8)
        "bfi.b32 %1, t1, t0, 16, 16;\n\t" // hi = t0 | (t1 << 16)
        "}"
        : "=r"(lo_result), "=r"(hi_result)
        : "r"((uint32_t)b0), "r"((uint32_t)b1), "r"((uint32_t)b2), "r"((uint32_t)b3),
          "r"((uint32_t)b4), "r"((uint32_t)b5), "r"((uint32_t)b6), "r"((uint32_t)b7));

    uint64_t sbox_out = ((uint64_t)hi_result << 32) | lo_result;
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
__device__ __noinline__ void mds_layer(const uint64_t* state_in, uint64_t* state_out);
__device__ void round_constants_layer(int round_index, const uint64_t* state_in, uint64_t* state_out);
__device__ void generated_function(const uint64_t* input, uint64_t* output);

__host__ void sbox_layer_host(const uint64_t* state_in, uint64_t* state_out);
__host__ void mds_layer_host(const uint64_t* state_in, uint64_t* state_out);
__host__ void round_constants_layer_host(int round_index, const uint64_t* state_in, uint64_t* state_out);

__device__ __noinline__ void tip5_permutation(uint64_t* state);
__device__ __noinline__ void tip5_permutation_fixed_right_zero(uint64_t* state, uint64_t x7_one);
__host__ void tip5_permutation_host(uint64_t* state);

__device__ __forceinline__ void tip5_sponge_init(uint64_t* __restrict__ state, Domain domain) {
    // PTX-optimized state initialization using vectorized stores
    uint64_t capacity_val = (domain == Domain::FixedLength) ? static_cast<uint64_t>(domain) : BFE_ZERO;
    
    // Vectorized zero initialization for RATE elements (0-9)
    // Using ulonglong2 for 128-bit stores
    *reinterpret_cast<ulonglong2*>(&state[0]) = make_ulonglong2(0ULL, 0ULL);
    *reinterpret_cast<ulonglong2*>(&state[2]) = make_ulonglong2(0ULL, 0ULL);
    *reinterpret_cast<ulonglong2*>(&state[4]) = make_ulonglong2(0ULL, 0ULL);
    *reinterpret_cast<ulonglong2*>(&state[6]) = make_ulonglong2(0ULL, 0ULL);
    *reinterpret_cast<ulonglong2*>(&state[8]) = make_ulonglong2(0ULL, 0ULL);
    
    // Capacity elements (10-15) - set to domain value
    *reinterpret_cast<ulonglong2*>(&state[10]) = make_ulonglong2(capacity_val, capacity_val);
    *reinterpret_cast<ulonglong2*>(&state[12]) = make_ulonglong2(capacity_val, capacity_val);
    *reinterpret_cast<ulonglong2*>(&state[14]) = make_ulonglong2(capacity_val, capacity_val);
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
    // Vectorized copy for DIGEST_LEN (5) elements
    // Copy first 4 as two ulonglong2, then the 5th separately
    *reinterpret_cast<ulonglong2*>(&digest[0]) = *reinterpret_cast<ulonglong2*>(&state[0]);
    *reinterpret_cast<ulonglong2*>(&digest[2]) = *reinterpret_cast<ulonglong2*>(&state[2]);
    digest[4] = state[4];
}

__host__ void tip5_sponge_init_host(uint64_t* state, Domain domain);
__host__ void tip5_sponge_permutation_host(uint64_t* state);
__host__ void tip5_sponge_absorb_chunk_host(uint64_t* state, const uint64_t* chunk, size_t len);
__host__ void tip5_sponge_squeeze_host(uint64_t* state, uint64_t* digest);

void tip5_init_constants();

#endif
