#ifndef XNT_DIGEST_CUH
#define XNT_DIGEST_CUH

#include "tip5.cuh"

struct alignas(8) Digest {
    uint64_t values[DIGEST_LEN];
    
    __device__ __host__ Digest() {
        for (int i = 0; i < DIGEST_LEN; ++i) {
            values[i] = 0;
        }
    }
    
    __device__ __host__ static Digest from_u64(uint64_t val) {
        Digest d;
        d.values[0] = val;
        return d;
    }
    
    __device__ __host__ static Digest default_digest() {
        return Digest();
    }
    
    __device__ __host__ bool operator==(const Digest& other) const {
        for (int i = 0; i < DIGEST_LEN; ++i) {
            if (values[i] != other.values[i]) return false;
        }
        return true;
    }
    
    __device__ __host__ bool operator!=(const Digest& other) const {
        return !(*this == other);
    }
    
    __device__ __host__ bool is_zero() const {
        for (int i = 0; i < DIGEST_LEN; ++i) {
            if (values[i] != 0) return false;
        }
        return true;
    }
    
    __device__ __host__ void from_array(const uint64_t* arr) {
        for (int i = 0; i < DIGEST_LEN; ++i) {
            values[i] = arr[i];
        }
    }
    
    __device__ __host__ void to_array(uint64_t* arr) const {
        for (int i = 0; i < DIGEST_LEN; ++i) {
            arr[i] = values[i];
        }
    }
};

static_assert(sizeof(Digest) == DIGEST_LEN * sizeof(uint64_t), "Digest must be 5x u64");

__device__ __noinline__ Digest tip5_hash_fixed_device(const Digest& left, const Digest& right);
__device__ Digest tip5_hash_varlen_device(const uint64_t* input, size_t input_len);

__host__ Digest tip5_hash_fixed_host(const Digest& left, const Digest& right);
__host__ std::array<uint64_t, DIGEST_LEN> tip5_hash_fixed_host(
    const std::array<uint64_t, DIGEST_LEN>& left,
    const std::array<uint64_t, DIGEST_LEN>& right);
__host__ Digest tip5_hash_varlen_host(const std::vector<uint64_t>& input);

std::string digest_to_hex(const Digest& digest);
Digest hex_to_digest(const std::string& hex);
Digest parse_digest_string(const std::string& str);
bool digest_less_than_or_equal(const Digest& a, const Digest& b);
Digest make_target_easier(const Digest& target, uint64_t factor);

__host__ inline int digest_compare_host(const Digest& a, const Digest& b) {
    for (int i = DIGEST_LEN - 1; i >= 0; --i) {
        if (a.values[i] < b.values[i]) return -1;
        if (a.values[i] > b.values[i]) return 1;
    }
    return 0;
}

// PTX-optimized digest comparison for GPU - returns true if a <= b
// Uses early-exit on most significant limb which fails most often
__device__ __forceinline__ bool digest_less_equal_ptx(const Digest& a, const Digest& b) {
    // Most hashes fail on highest limb - check it first with PTX
    uint64_t a4 = a.values[4], b4 = b.values[4];
    uint64_t a3 = a.values[3], b3 = b.values[3];
    uint64_t a2 = a.values[2], b2 = b.values[2];
    uint64_t a1 = a.values[1], b1 = b.values[1];
    uint64_t a0 = a.values[0], b0 = b.values[0];
    
    int result;
    asm("{\n\t"
        ".reg .pred lt4, gt4, eq4, lt3, gt3, eq3, lt2, gt2, eq2, lt1, gt1, eq1, lt0;\n\t"
        // Compare limb 4 (most significant)
        "setp.lt.u64 lt4, %1, %2;\n\t"
        "setp.gt.u64 gt4, %1, %2;\n\t"
        "setp.eq.u64 eq4, %1, %2;\n\t"
        // Compare limb 3
        "setp.lt.u64 lt3, %3, %4;\n\t"
        "setp.gt.u64 gt3, %3, %4;\n\t"
        "setp.eq.u64 eq3, %3, %4;\n\t"
        // Compare limb 2
        "setp.lt.u64 lt2, %5, %6;\n\t"
        "setp.gt.u64 gt2, %5, %6;\n\t"
        "setp.eq.u64 eq2, %5, %6;\n\t"
        // Compare limb 1
        "setp.lt.u64 lt1, %7, %8;\n\t"
        "setp.gt.u64 gt1, %7, %8;\n\t"
        "setp.eq.u64 eq1, %7, %8;\n\t"
        // Compare limb 0
        "setp.le.u64 lt0, %9, %10;\n\t"  // lt0 is actually le for limb 0
        // Build result: a <= b if:
        //   lt4 OR (eq4 AND (lt3 OR (eq3 AND (lt2 OR (eq2 AND (lt1 OR (eq1 AND lt0)))))))
        // Simplify with selp chain
        "selp.s32 %0, 1, 0, lt0;\n\t"      // result = (a0 <= b0) ? 1 : 0
        "selp.s32 %0, 1, %0, lt1;\n\t"     // if a1 < b1, result = 1
        "selp.s32 %0, 0, %0, gt1;\n\t"     // if a1 > b1, result = 0
        "selp.s32 %0, 1, %0, lt2;\n\t"     // if a2 < b2, result = 1
        "selp.s32 %0, 0, %0, gt2;\n\t"     // if a2 > b2, result = 0
        "selp.s32 %0, 1, %0, lt3;\n\t"     // if a3 < b3, result = 1
        "selp.s32 %0, 0, %0, gt3;\n\t"     // if a3 > b3, result = 0
        "selp.s32 %0, 1, %0, lt4;\n\t"     // if a4 < b4, result = 1
        "selp.s32 %0, 0, %0, gt4;\n\t"     // if a4 > b4, result = 0
        "}"
        : "=r"(result)
        : "l"(a4), "l"(b4), "l"(a3), "l"(b3), "l"(a2), "l"(b2), 
          "l"(a1), "l"(b1), "l"(a0), "l"(b0));
    return result != 0;
}

__device__ __host__ __inline__ int digest_compare(const Digest& a, const Digest& b) {
    for (int i = DIGEST_LEN - 1; i >= 0; --i) {
        if (a.values[i] < b.values[i]) return -1;
        if (a.values[i] > b.values[i]) return 1;
    }
    return 0;
}

__host__ __device__ __inline__ Digest digest_right_shift(const Digest& d, int bits) {
    Digest result;
    if (bits >= 64 * DIGEST_LEN) return result;
    
    int word_shift = bits / 64;
    int bit_shift = bits % 64;
    
    for (int i = 0; i < DIGEST_LEN; ++i) {
        int src = i + word_shift;
        if (src < DIGEST_LEN) {
            result.values[i] = d.values[src] >> bit_shift;
            if (bit_shift > 0 && src + 1 < DIGEST_LEN) {
                result.values[i] |= d.values[src + 1] << (64 - bit_shift);
            }
        }
    }
    return result;
}

__device__ __inline__ Digest tip5_hash_fixed_right_zero_device(const Digest& left) {
    uint64_t state[STATE_SIZE];
    tip5_sponge_init(state, Domain::FixedLength);
    
    for (int i = 0; i < DIGEST_LEN; ++i) {
        state[i] = left.values[i];
    }
    for (int i = DIGEST_LEN; i < RATE; ++i) {
        state[i] = 0;
    }
    
    tip5_permutation(state);
    
    Digest result;
    for (int i = 0; i < DIGEST_LEN; ++i) {
        result.values[i] = state[i];
    }
    return result;
}

__device__ __inline__ Digest tip5_hash_varlen_len5_device(const Digest& in) {
    uint64_t state[STATE_SIZE];
    tip5_sponge_init(state, Domain::VariableLength);
    // absorb 5 words
    for (int i = 0; i < DIGEST_LEN; ++i) {
        state[i] = in.values[i];
    }
    // padding marker
    state[DIGEST_LEN] = BFE_ONE;
    // zero the rest of the rate slots
    for (int i = DIGEST_LEN + 1; i < RATE; ++i) {
        state[i] = 0ULL;
    }
    tip5_permutation(state);
    Digest out;
    for (int i = 0; i < DIGEST_LEN; ++i) out.values[i] = state[i];
    return out;
}

#endif
