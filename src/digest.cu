#include "digest.cuh"

// ===== TIP5 HASH FUNCTIONS (DEVICE) =====

__device__ Digest tip5_hash_fixed_device(const Digest& left, const Digest& right) {
    uint64_t state[STATE_SIZE];
    
    // Initialize sponge with fixed length domain
    tip5_sponge_init(state, Domain::FixedLength);
    
    // Absorb left digest (5 elements)
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        state[i] = left.values[i];
    }
    
    // Absorb right digest (5 elements)
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        state[i + DIGEST_LEN] = right.values[i];
    }
    
    // Permute
    tip5_permutation(state);
    
    // Squeeze output
    Digest result;
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        result.values[i] = state[i];
    }
    return result;
}

__device__ Digest tip5_hash_varlen_device(const uint64_t* input, size_t input_len) {
    uint64_t state[STATE_SIZE];
    
    // Initialize sponge with variable length domain
    tip5_sponge_init(state, Domain::VariableLength);
    
    // Process full chunks
    size_t pos = 0;
    while (pos + RATE <= input_len) {
        for (size_t i = 0; i < RATE; ++i) {
            state[i] = input[pos + i];
        }
        tip5_permutation(state);
        pos += RATE;
    }
    
    // Process remaining elements with padding
    size_t remaining = input_len - pos;
    for (size_t i = 0; i < remaining; ++i) {
        state[i] = input[pos + i];
    }
    
    // Add length as padding
    state[remaining] = input_len;
    
    // Zero remaining state elements
    for (size_t i = remaining + 1; i < RATE; ++i) {
        state[i] = 0;
    }
    
    // Final permutation
    tip5_permutation(state);
    
    // Squeeze output
    Digest result;
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        result.values[i] = state[i];
    }
    return result;
}

// ===== TIP5 HASH FUNCTIONS (HOST) =====

__host__ Digest tip5_hash_fixed_host(const Digest& left, const Digest& right) {
    uint64_t state[STATE_SIZE];
    
    // Initialize sponge with fixed length domain
    tip5_sponge_init_host(state, Domain::FixedLength);
    
    // Absorb left digest (5 elements)
    for (int i = 0; i < DIGEST_LEN; ++i) {
        state[i] = left.values[i];
    }
    
    // Absorb right digest (5 elements)
    for (int i = 0; i < DIGEST_LEN; ++i) {
        state[i + DIGEST_LEN] = right.values[i];
    }
    
    // Permute
    tip5_permutation_host(state);
    
    // Squeeze output
    Digest result;
    for (int i = 0; i < DIGEST_LEN; ++i) {
        result.values[i] = state[i];
    }
    return result;
}

__host__ std::array<uint64_t, DIGEST_LEN> tip5_hash_fixed_host(
    const std::array<uint64_t, DIGEST_LEN>& left,
    const std::array<uint64_t, DIGEST_LEN>& right) {
    
    uint64_t state[STATE_SIZE];
    
    // Initialize sponge with fixed length domain
    tip5_sponge_init_host(state, Domain::FixedLength);
    
    // Absorb inputs
    for (int i = 0; i < DIGEST_LEN; ++i) {
        state[i] = left[i];
        state[i + DIGEST_LEN] = right[i];
    }
    
    // Permute
    tip5_permutation_host(state);
    
    // Squeeze output
    std::array<uint64_t, DIGEST_LEN> result;
    for (int i = 0; i < DIGEST_LEN; ++i) {
        result[i] = state[i];
    }
    return result;
}

__host__ Digest tip5_hash_varlen_host(const std::vector<uint64_t>& input) {
    uint64_t state[STATE_SIZE];
    
    // Initialize sponge with variable length domain
    tip5_sponge_init_host(state, Domain::VariableLength);
    
    // Process full chunks
    size_t pos = 0;
    while (pos + RATE <= input.size()) {
        for (size_t i = 0; i < RATE; ++i) {
            state[i] = input[pos + i];
        }
        tip5_permutation_host(state);
        pos += RATE;
    }
    
    // Process remaining elements with padding
    size_t remaining = input.size() - pos;
    for (size_t i = 0; i < remaining; ++i) {
        state[i] = input[pos + i];
    }
    
    // Add length as padding
    state[remaining] = input.size();
    
    // Zero remaining state elements
    for (size_t i = remaining + 1; i < RATE; ++i) {
        state[i] = 0;
    }
    
    // Final permutation
    tip5_permutation_host(state);
    
    // Squeeze output
    Digest result;
    for (int i = 0; i < DIGEST_LEN; ++i) {
        result.values[i] = state[i];
    }
    return result;
}

// ===== DIGEST CONVERSION UTILITIES =====

std::string digest_to_hex(const Digest& digest) {
    std::ostringstream oss;
    
    // Convert each limb to bytes in little-endian order
    // Then output as hex, skipping leading zeros
    std::vector<uint8_t> bytes;
    bytes.reserve(DIGEST_LEN * 8);
    
    for (int limb = 0; limb < DIGEST_LEN; ++limb) {
        uint64_t value = digest.values[limb];
        for (int byte = 0; byte < 8; ++byte) {
            uint8_t b = (value >> (byte * 8)) & 0xFF;
            bytes.push_back(b);
        }
    }
    
    // Find first non-zero byte (from the end for big-endian display)
    size_t first_non_zero = bytes.size();
    for (size_t i = bytes.size(); i > 0; --i) {
        if (bytes[i - 1] != 0) {
            first_non_zero = i - 1;
        }
    }
    
    // Output as hex
    if (first_non_zero == bytes.size()) {
        return "0";
    }
    
    oss << std::hex << std::setfill('0');
    for (size_t i = bytes.size(); i > 0; --i) {
        oss << std::setw(2) << static_cast<int>(bytes[i - 1]);
    }
    
    // Remove leading zeros from output
    std::string result = oss.str();
    size_t start = result.find_first_not_of('0');
    if (start == std::string::npos) {
        return "0";
    }
    return result.substr(start);
}

Digest hex_to_digest(const std::string& hex) {
    Digest result;
    
    // Initialize to zero
    for (int i = 0; i < DIGEST_LEN; ++i) {
        result.values[i] = 0;
    }
    
    // Skip "0x" prefix if present
    std::string hex_str = hex;
    if (hex_str.size() >= 2 && hex_str[0] == '0' && (hex_str[1] == 'x' || hex_str[1] == 'X')) {
        hex_str = hex_str.substr(2);
    }
    
    // Remove leading zeros
    size_t start = hex_str.find_first_not_of('0');
    if (start == std::string::npos) {
        return result; // All zeros
    }
    hex_str = hex_str.substr(start);
    
    // Pad to even length
    if (hex_str.length() % 2 != 0) {
        hex_str = "0" + hex_str;
    }
    
    // Convert hex pairs to bytes (little-endian into limbs)
    std::vector<uint8_t> bytes;
    for (size_t i = hex_str.length(); i >= 2; i -= 2) {
        std::string byteString = hex_str.substr(i - 2, 2);
        uint8_t byte = static_cast<uint8_t>(std::stoul(byteString, nullptr, 16));
        bytes.push_back(byte);
    }
    
    // Fill digest limbs from bytes
    const size_t max_bytes = DIGEST_LEN * 8;
    const size_t nbytes = std::min(bytes.size(), max_bytes);
    for (size_t i = 0; i < nbytes; ++i) {
        const int limb_index = i / 8;
        const int byte_offset = i % 8;
        result.values[limb_index] |= static_cast<uint64_t>(bytes[i]) << (byte_offset * 8);
    }
    
    return result;
}

Digest arrayStringToDigest(const std::string& str) {
    Digest result;
    
    // Initialize to zero
    for (int i = 0; i < DIGEST_LEN; ++i) {
        result.values[i] = 0;
    }
    
    // Parse format: "[val0, val1, val2, val3, val4]"
    std::string data = str;
    
    // Remove brackets and whitespace
    size_t start = data.find('[');
    size_t end = data.find(']');
    if (start != std::string::npos && end != std::string::npos && end > start) {
        data = data.substr(start + 1, end - start - 1);
    }
    
    // Parse comma-separated values
    std::istringstream ss(data);
    std::string token;
    int index = 0;
    
    while (std::getline(ss, token, ',') && index < DIGEST_LEN) {
        // Trim whitespace
        size_t first = token.find_first_not_of(" \t\n\r");
        size_t last = token.find_last_not_of(" \t\n\r");
        if (first != std::string::npos && last != std::string::npos) {
            token = token.substr(first, last - first + 1);
        }
        
        // Parse as uint64_t
        try {
            result.values[index] = std::stoull(token);
        } catch (...) {
            // Invalid number, leave as zero
        }
        index++;
    }
    
    return result;
}

Digest parseDigestString(const std::string& str) {
    // Detect format and parse accordingly
    
    // Check for array format "[...]"
    if (str.find('[') != std::string::npos) {
        return arrayStringToDigest(str);
    }
    
    // Otherwise treat as hex string
    return hex_to_digest(str);
}

// ===== DIGEST COMPARISON =====

bool digest_less_than_or_equal(const Digest& a, const Digest& b) {
    // Compare from most significant limb to least significant
    // Limb 4 is most significant
    for (int i = DIGEST_LEN - 1; i >= 0; --i) {
        if (a.values[i] < b.values[i]) {
            return true;  // a < b
        }
        if (a.values[i] > b.values[i]) {
            return false; // a > b
        }
    }
    return true; // a == b
}

// ===== DIGEST ARITHMETIC =====

__host__ Digest digest_multiply_scalar(const Digest& d, uint64_t scalar) {
    Digest result;
#ifdef _WIN32
    #ifdef _MSC_VER
        // MSVC: Use 64-bit arithmetic with manual carry handling
        uint64_t carry_low = 0;
        uint64_t carry_high = 0;
        for (int i = 0; i < DIGEST_LEN; ++i) {
            unsigned __int64 high;
            unsigned __int64 low = _umul128(d.values[i], scalar, &high);
            // Add carry
            uint64_t new_low = low + carry_low;
            bool carry_bit = (new_low < low);
            low = new_low;
            high += carry_high + (carry_bit ? 1 : 0);
            result.values[i] = low;
            carry_low = high;
            carry_high = 0; // High part of carry is always 0 for 64x64 multiplication
        }
        uint64_t carry = carry_low;
    #else
        __uint128_t carry = 0;
        for (int i = 0; i < DIGEST_LEN; ++i) {
            __uint128_t product = static_cast<__uint128_t>(d.values[i]) * scalar + carry;
            result.values[i] = static_cast<uint64_t>(product);
            carry = product >> 64;
        }
    #endif
#else
    __uint128_t carry = 0;
    for (int i = 0; i < DIGEST_LEN; ++i) {
        __uint128_t product = static_cast<__uint128_t>(d.values[i]) * scalar + carry;
        result.values[i] = static_cast<uint64_t>(product);
        carry = product >> 64;
    }
#endif
    
    // If there's overflow, cap at max value
    if (carry > 0) {
        for (int i = 0; i < DIGEST_LEN; ++i) {
            result.values[i] = UINT64_MAX;
        }
    }
    
    return result;
}