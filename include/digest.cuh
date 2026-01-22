#ifndef XNT_DIGEST_CUH
#define XNT_DIGEST_CUH

#include "tip5.cuh"

// ===== OPTIMIZED DIGEST STRUCTURE =====
// 40-byte aligned structure for efficient GPU memory access
struct alignas(8) Digest {
    uint64_t values[DIGEST_LEN];
    
    // Default constructor - zero initialize
    __device__ __host__ __forceinline__ Digest() {
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) {
            values[i] = 0;
        }
    }
    
    // Create digest with single value (rest zeros)
    __device__ __host__ __forceinline__ static Digest new_from_u64(uint64_t val) {
        Digest d;
        d.values[0] = val;
        #pragma unroll
        for (size_t i = 1; i < DIGEST_LEN; ++i) {
            d.values[i] = 0;
        }
        return d;
    }
    
    // Default (zero) digest
    __device__ __host__ __forceinline__ static Digest default_digest() {
        return Digest();
    }
    
    // Equality comparison
    __device__ __host__ __forceinline__ bool operator==(const Digest& other) const {
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) {
            if (values[i] != other.values[i]) return false;
        }
        return true;
    }
    
    __device__ __host__ __forceinline__ bool operator!=(const Digest& other) const {
        return !(*this == other);
    }
    
    // Check if all values are zero
    __device__ __host__ __forceinline__ bool is_zero() const {
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) {
            if (values[i] != 0) return false;
        }
        return true;
    }
    
    // Copy from array
    __device__ __host__ __forceinline__ void from_array(const uint64_t* arr) {
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) {
            values[i] = arr[i];
        }
    }
    
    // Copy to array
    __device__ __host__ __forceinline__ void to_array(uint64_t* arr) const {
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) {
            arr[i] = values[i];
        }
    }
};

// Verify layout is exactly 5 x uint64_t
static_assert(sizeof(Digest) == DIGEST_LEN * sizeof(uint64_t), "Digest layout must be 5x u64");

// ===== TIP5 HASH FUNCTIONS (DEVICE) =====

// Hash two digests together (fixed length, 10 elements input)
__device__ Digest tip5_hash_fixed_device(const Digest& left, const Digest& right);

// Hash digest with right side being all zeros (optimization)
__device__ __forceinline__ Digest tip5_hash_fixed_right_zero_device(const Digest& left);

// Hash variable length input
__device__ Digest tip5_hash_varlen_device(const uint64_t* input, size_t input_len);

// Hash exactly 5 elements (single digest)
__device__ __forceinline__ Digest tip5_hash_varlen_len5_device(const Digest& in);

// ===== TIP5 HASH FUNCTIONS (HOST) =====

// Hash two digests together
__host__ Digest tip5_hash_fixed_host(const Digest& left, const Digest& right);

// Hash two arrays of 5 elements each
__host__ std::array<uint64_t, DIGEST_LEN> tip5_hash_fixed_host(
    const std::array<uint64_t, DIGEST_LEN>& left,
    const std::array<uint64_t, DIGEST_LEN>& right);

// Hash variable length input (vector)
__host__ Digest tip5_hash_varlen_host(const std::vector<uint64_t>& input);

// ===== DIGEST CONVERSION UTILITIES =====

// Convert digest to hexadecimal string
std::string digest_to_hex(const Digest& digest);

// Parse hexadecimal string to digest
Digest hex_to_digest(const std::string& hex);

// Parse JSON array string format to digest
// Format: "[val0, val1, val2, val3, val4]"
Digest arrayStringToDigest(const std::string& str);

// Parse digest from various string formats (auto-detect)
Digest parseDigestString(const std::string& str);

// ===== DIGEST COMPARISON =====

// Compare two digests (for threshold checking)
// Returns true if a <= b (treating as big integers, MSB at index 4)
bool digest_less_than_or_equal(const Digest& a, const Digest& b);

// Compare two digests for ordering
// Returns -1 if a < b, 0 if a == b, 1 if a > b
__device__ __host__ __forceinline__ int digest_compare(const Digest& a, const Digest& b) {
    // Compare from most significant limb to least significant
    for (int i = DIGEST_LEN - 1; i >= 0; --i) {
        if (a.values[i] < b.values[i]) return -1;
        if (a.values[i] > b.values[i]) return 1;
    }
    return 0;
}

// ===== DIGEST ARITHMETIC (for target calculations) =====

// Right shift digest by bits (divide by 2^bits)
__host__ __device__ __forceinline__ Digest digest_right_shift(const Digest& d, int bits) {
    Digest result;
    if (bits >= 64 * DIGEST_LEN) {
        return result; // All zeros
    }
    
    int word_shift = bits / 64;
    int bit_shift = bits % 64;
    
    for (int i = 0; i < DIGEST_LEN; ++i) {
        int src_idx = i + word_shift;
        if (src_idx < DIGEST_LEN) {
            result.values[i] = d.values[src_idx] >> bit_shift;
            if (bit_shift > 0 && src_idx + 1 < DIGEST_LEN) {
                result.values[i] |= d.values[src_idx + 1] << (64 - bit_shift);
            }
        }
    }
    return result;
}

// ===== INLINE IMPLEMENTATIONS =====

// Optimized hash with right side all zeros
__device__ __forceinline__ Digest tip5_hash_fixed_right_zero_device(const Digest& left) {
    uint64_t state[STATE_SIZE];
    
    // Initialize sponge with fixed length domain
    tip5_sponge_init(state, Domain::FixedLength);
    
    // Absorb left digest (5 elements) + 5 zeros
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        state[i] = left.values[i];
    }
    #pragma unroll
    for (int i = DIGEST_LEN; i < RATE; ++i) {
        state[i] = 0;
    }
    
    tip5_permutation(state);
    
    // Squeeze output
    Digest result;
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        result.values[i] = state[i];
    }
    return result;
}

// Hash exactly 5 elements
__device__ __forceinline__ Digest tip5_hash_varlen_len5_device(const Digest& in) {
    uint64_t state[STATE_SIZE];
    
    // Initialize sponge with variable length domain
    tip5_sponge_init(state, Domain::VariableLength);
    
    // Absorb 5 elements + length (5) + padding
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        state[i] = in.values[i];
    }
    state[DIGEST_LEN] = 5;  // Length
    #pragma unroll
    for (int i = DIGEST_LEN + 1; i < RATE; ++i) {
        state[i] = 0;
    }
    
    tip5_permutation(state);
    
    // Squeeze output
    Digest result;
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        result.values[i] = state[i];
    }
    return result;
}

#endif // XNT_DIGEST_CUH