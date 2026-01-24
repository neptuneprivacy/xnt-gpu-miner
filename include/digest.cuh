#ifndef XNT_DIGEST_CUH
#define XNT_DIGEST_CUH

#include "tip5.cuh"

struct alignas(8) Digest {
    uint64_t values[DIGEST_LEN];
    
    __device__ __host__ __forceinline__ Digest() {
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) {
            values[i] = 0;
        }
    }
    
    __device__ __host__ __forceinline__ static Digest from_u64(uint64_t val) {
        Digest d;
        d.values[0] = val;
        return d;
    }
    
    __device__ __host__ __forceinline__ static Digest default_digest() {
        return Digest();
    }
    
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
    
    __device__ __host__ __forceinline__ bool is_zero() const {
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) {
            if (values[i] != 0) return false;
        }
        return true;
    }
    
    __device__ __host__ __forceinline__ void from_array(const uint64_t* arr) {
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) {
            values[i] = arr[i];
        }
    }
    
    __device__ __host__ __forceinline__ void to_array(uint64_t* arr) const {
        #pragma unroll
        for (int i = 0; i < DIGEST_LEN; ++i) {
            arr[i] = values[i];
        }
    }
};

static_assert(sizeof(Digest) == DIGEST_LEN * sizeof(uint64_t), "Digest must be 5x u64");

__device__ Digest tip5_hash_fixed_device(const Digest& left, const Digest& right);
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

__device__ __host__ __forceinline__ int digest_compare(const Digest& a, const Digest& b) {
    for (int i = DIGEST_LEN - 1; i >= 0; --i) {
        if (a.values[i] < b.values[i]) return -1;
        if (a.values[i] > b.values[i]) return 1;
    }
    return 0;
}

__host__ __device__ __forceinline__ Digest digest_right_shift(const Digest& d, int bits) {
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

__device__ __forceinline__ Digest tip5_hash_fixed_right_zero_device(const Digest& left) {
    uint64_t state[STATE_SIZE];
    tip5_sponge_init(state, Domain::FixedLength);
    
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        state[i] = left.values[i];
    }
    #pragma unroll
    for (int i = DIGEST_LEN; i < RATE; ++i) {
        state[i] = 0;
    }
    
    tip5_permutation(state);
    
    Digest result;
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        result.values[i] = state[i];
    }
    return result;
}

__device__ __forceinline__ Digest tip5_hash_varlen_len5_device(const Digest& in) {
    uint64_t state[STATE_SIZE];
    tip5_sponge_init(state, Domain::VariableLength);
    
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        state[i] = in.values[i];
    }
    state[DIGEST_LEN] = 5;
    #pragma unroll
    for (int i = DIGEST_LEN + 1; i < RATE; ++i) {
        state[i] = 0;
    }
    
    tip5_permutation(state);
    
    Digest result;
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        result.values[i] = state[i];
    }
    return result;
}

#endif
