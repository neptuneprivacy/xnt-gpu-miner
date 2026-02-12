// xnt-miner-unity.cu — Single file: all source inlined
// Auto-generated from src/*.cu — do not edit

#define XNT_UNITY_BUILD 1

// --- common.cu ---
#include "common.cuh"

std::atomic<bool> stop_mining{false};

std::string g_miner_wallet_address = "";
std::string g_miner_worker_name = "";
std::atomic<int> g_total_gpu_count{1};
int g_gpu_device_id = -1;
bool g_test_mode = false;
bool g_benchmark_mode = false;
int g_fetch_interval_sec = 5;

std::mutex g_log_mutex;

// Tuning parameters (optimized defaults)
int g_block_size = 256;              // Threads per block (optimal for LUT loading)
uint64_t g_batch_size = DEFAULT_BATCH_SIZE;  // Mining + benchmark default (0 = auto)
int g_blocks_per_grid = 0;           // Blocks per grid (0 = auto; 680 = fast miner config)
bool g_blocks_sweep = false;         // If true, benchmark sweeps block counts

// --- mining_core.cu ---
// ============================================================================
// mining_core.cu — Unity build for all mining-critical CUDA code.
//
// Contains tip5, digest, kernels, and pow/merkle inlined below into a single TU.
// ============================================================================

#define MINING_CORE_UNITY

// ========== Inlined from tip5.cu (start) ==========
#include "common.cuh"

constexpr int TIP5_STATE_SIZE = 16;
constexpr int TIP5_NUM_ROUNDS = 5;

__constant__ uint32_t MDS_COEFF[TIP5_STATE_SIZE] = {
    61402, 1108, 28750, 33823, 7454, 43244, 53865, 12034,
    56951, 27521, 41351, 40901, 12021, 59689, 26798, 17845
};

__constant__ uint64_t ROUND_CONSTANTS[TIP5_NUM_ROUNDS][TIP5_STATE_SIZE] = {
    {
        0xBD2A3DEB61AB60DEULL, 0xEA7DF21AD9547ED2ULL, 0x900B3677A1DE063FULL, 0x1B46887E876C8677ULL,
        0xD364D977889CFB97ULL, 0xDC8DFAC843699F02ULL, 0x375C405D7190DB58ULL, 0x27924006D2B0D4B1ULL,
        0x78DD1172D483CD38ULL, 0x3346C66244882A56ULL, 0xB0249B279F498AA5ULL, 0x94CD51BE79338D4DULL,
        0xB0E0DC7052C5B218ULL, 0xF8DCC4D248ADAD95ULL, 0x68E3C635FEC868B7ULL, 0xD7D06B3FFB6B0D8CULL
    },
    {
        0xF3500DEA20EF032AULL, 0x4865BF175BBA5803ULL, 0xD5F7FE3027287A27ULL, 0xA57333F44E193412ULL,
        0x8726E153A977EAE2ULL, 0x3014A98463FC191BULL, 0xBA145461AF39B212ULL, 0x03AB70105933202FULL,
        0x3D90B7EEBFCF71E5ULL, 0x386322B1CC520BFDULL, 0x27C2C8DAF774F675ULL, 0x4FCB83F50309BC6AULL,
        0x5E6D5CE8275F3CB3ULL, 0xECC2F6592C8F905CULL, 0x837F532461E609B4ULL, 0xB2B1F6B95C92C93CULL
    },
    {
        0xC0027AF556411DC1ULL, 0x16E18C885FC2A26CULL, 0x8880EF183D9F2BF3ULL, 0xB2930BDB5CA88C45ULL,
        0x9C2EC8322E1C1553ULL, 0xE5B05EAF3220A674ULL, 0xA49CC6AE4B861C4EULL, 0x11708E0AEB86EBD7ULL,
        0xC09DE92BBC3902E0ULL, 0x929B3C79516BCBC1ULL, 0xE006E5BF738F27D1ULL, 0x2D9E1EC0EAC8EA38ULL,
        0x0984D8D94BF937C5ULL, 0x4959273C220E6747ULL, 0xFE1D934207E796FAULL, 0x2B9B9298F2F6DD73ULL
    },
    {
        0x07A1F5A67D6E3A41ULL, 0x4407593EE73743D9ULL, 0x9F054720EF802E59ULL, 0x78D4B711336E6AA6ULL,
        0xADC638AEF3C8B228ULL, 0xA4D6D3E86AFB2114ULL, 0x9D4808E725531968ULL, 0x369804DF3866D0EFULL,
        0xE6DBD9A9D2215024ULL, 0x8ED22CA212EE85B2ULL, 0x397BB882FCD23EB6ULL, 0xEB8F8786D7277531ULL,
        0x9999D4CDAFF543B5ULL, 0xF382A61217F192D6ULL, 0x49C37260B026ADC1ULL, 0x3FF8918CE35C1019ULL
    },
    {
        0x2E7DF8B76080BD07ULL, 0xF5DBAC250B8A28B9ULL, 0x853C3727AE9DA4CCULL, 0xB2F1F5F3D9E5A26DULL,
        0x3FCE22012D337847ULL, 0x6B5A3E6DB7EEE347ULL, 0x171582CD59DDE50DULL, 0xC0C0B3095EE62A8AULL,
        0x665B25C6F6A203D2ULL, 0x3099AED93B6AE69FULL, 0x801DF6092BE69C38ULL, 0x8066AD0CDFFF43CDULL,
        0x8AF9D44A5F4FDC6BULL, 0xD80219CD97C0D762ULL, 0x10C9CEBA14148EBBULL, 0x539BD4C3F2F24474ULL
    }
};

__constant__ uint8_t LOOKUP_TABLE[256] = {
      0,   7,  26,  63, 124, 215,  85, 254, 214, 228,  45, 185, 140, 173,  33, 240,
     29, 177, 176,  32,   8, 110,  87, 202, 204,  99, 150, 106, 230,  14, 235, 128,
    213, 239, 212, 138,  23, 130, 208,   6,  44,  71,  93, 116, 146, 189, 251,  81,
    199,  97,  38,  28,  73, 179,  95,  84, 152,  48,  35, 119,  49,  88, 242,   3,
    148, 169,  72, 120,  62, 161, 166,  83, 175, 191, 137,  19, 100, 129, 112,  55,
    221, 102, 218,  61, 151, 237,  68, 164,  17, 147,  46, 234, 203, 216,  22, 141,
     65,  57, 123,  12, 244,  54, 219, 231,  96,  77, 180, 154,   5, 253, 133, 165,
     98, 195, 205, 134, 245,  30,   9, 188,  59, 142, 186, 197, 181, 144,  92,  31,
    224, 163, 111,  74,  58,  69, 113, 196,  67, 246, 225,  10, 121,  50,  60, 157,
     90, 122,   2, 250, 101,  75, 178, 159,  24,  36, 201,  11, 243, 132, 198, 190,
    114, 233,  39,  52,  21, 209, 108, 238,  91, 187,  18, 104, 194,  37, 153,  34,
    200, 143, 126, 155, 236, 118,  64,  80, 172,  89,  94, 193, 135, 183,  86, 107,
    252,  13, 167, 206, 136, 220, 207, 103, 171, 160,  76, 182, 227, 217, 158,  56,
    174,   4,  66, 109, 139, 162, 184, 211, 249,  47, 125, 232, 117,  43,  16,  42,
    127,  20, 241,  25, 149, 105, 156,  51,  53, 168, 145, 247, 223,  79,  78, 226,
     15, 222,  82, 115,  70, 210,  27,  41,   1, 170,  40, 131, 192, 229, 248, 255
};

#define TIP5_DEFINING_CONSTANTS
#define KERNELS_DEFINING_RANGE_CONSTANTS  // Skip tip5 extern d_gpu_range_* (we define below)
#include "tip5.cuh"
#undef TIP5_DEFINING_CONSTANTS
#undef KERNELS_DEFINING_RANGE_CONSTANTS

const uint8_t LOOKUP_TABLE_HOST[256] = {
    0,   7,  26,  63, 124, 215,  85, 254, 214, 228,  45, 185, 140, 173,  33, 240,
    29, 177, 176,  32,   8, 110,  87, 202, 204,  99, 150, 106, 230,  14, 235, 128,
   213, 239, 212, 138,  23, 130, 208,   6,  44,  71,  93, 116, 146, 189, 251,  81,
   199,  97,  38,  28,  73, 179,  95,  84, 152,  48,  35, 119,  49,  88, 242,   3,
   148, 169,  72, 120,  62, 161, 166,  83, 175, 191, 137,  19, 100, 129, 112,  55,
   221, 102, 218,  61, 151, 237,  68, 164,  17, 147,  46, 234, 203, 216,  22, 141,
    65,  57, 123,  12, 244,  54, 219, 231,  96,  77, 180, 154,   5, 253, 133, 165,
    98, 195, 205, 134, 245,  30,   9, 188,  59, 142, 186, 197, 181, 144,  92,  31,
   224, 163, 111,  74,  58,  69, 113, 196,  67, 246, 225,  10, 121,  50,  60, 157,
    90, 122,   2, 250, 101,  75, 178, 159,  24,  36, 201,  11, 243, 132, 198, 190,
   114, 233,  39,  52,  21, 209, 108, 238,  91, 187,  18, 104, 194,  37, 153,  34,
   200, 143, 126, 155, 236, 118,  64,  80, 172,  89,  94, 193, 135, 183,  86, 107,
   252,  13, 167, 206, 136, 220, 207, 103, 171, 160,  76, 182, 227, 217, 158,  56,
   174,   4,  66, 109, 139, 162, 184, 211, 249,  47, 125, 232, 117,  43,  16,  42,
   127,  20, 241,  25, 149, 105, 156,  51,  53, 168, 145, 247, 223,  79,  78, 226,
    15, 222,  82, 115,  70, 210,  27,  41,   1, 170,  40, 131, 192, 229, 248, 255
};

const uint32_t MDS_COEFF_HOST[16] = {
    61402, 1108, 28750, 33823, 7454, 43244, 53865, 12034,
    56951, 27521, 41351, 40901, 12021, 59689, 26798, 17845
};

const uint64_t ROUND_CONSTANTS_HOST[5][16] = {
    {
        0xBD2A3DEB61AB60DEULL, 0xEA7DF21AD9547ED2ULL, 0x900B3677A1DE063FULL, 0x1B46887E876C8677ULL,
        0xD364D977889CFB97ULL, 0xDC8DFAC843699F02ULL, 0x375C405D7190DB58ULL, 0x27924006D2B0D4B1ULL,
        0x78DD1172D483CD38ULL, 0x3346C66244882A56ULL, 0xB0249B279F498AA5ULL, 0x94CD51BE79338D4DULL,
        0xB0E0DC7052C5B218ULL, 0xF8DCC4D248ADAD95ULL, 0x68E3C635FEC868B7ULL, 0xD7D06B3FFB6B0D8CULL
    },
    {
        0xF3500DEA20EF032AULL, 0x4865BF175BBA5803ULL, 0xD5F7FE3027287A27ULL, 0xA57333F44E193412ULL,
        0x8726E153A977EAE2ULL, 0x3014A98463FC191BULL, 0xBA145461AF39B212ULL, 0x03AB70105933202FULL,
        0x3D90B7EEBFCF71E5ULL, 0x386322B1CC520BFDULL, 0x27C2C8DAF774F675ULL, 0x4FCB83F50309BC6AULL,
        0x5E6D5CE8275F3CB3ULL, 0xECC2F6592C8F905CULL, 0x837F532461E609B4ULL, 0xB2B1F6B95C92C93CULL
    },
    {
        0xC0027AF556411DC1ULL, 0x16E18C885FC2A26CULL, 0x8880EF183D9F2BF3ULL, 0xB2930BDB5CA88C45ULL,
        0x9C2EC8322E1C1553ULL, 0xE5B05EAF3220A674ULL, 0xA49CC6AE4B861C4EULL, 0x11708E0AEB86EBD7ULL,
        0xC09DE92BBC3902E0ULL, 0x929B3C79516BCBC1ULL, 0xE006E5BF738F27D1ULL, 0x2D9E1EC0EAC8EA38ULL,
        0x0984D8D94BF937C5ULL, 0x4959273C220E6747ULL, 0xFE1D934207E796FAULL, 0x2B9B9298F2F6DD73ULL
    },
    {
        0x07A1F5A67D6E3A41ULL, 0x4407593EE73743D9ULL, 0x9F054720EF802E59ULL, 0x78D4B711336E6AA6ULL,
        0xADC638AEF3C8B228ULL, 0xA4D6D3E86AFB2114ULL, 0x9D4808E725531968ULL, 0x369804DF3866D0EFULL,
        0xE6DBD9A9D2215024ULL, 0x8ED22CA212EE85B2ULL, 0x397BB882FCD23EB6ULL, 0xEB8F8786D7277531ULL,
        0x9999D4CDAFF543B5ULL, 0xF382A61217F192D6ULL, 0x49C37260B026ADC1ULL, 0x3FF8918CE35C1019ULL
    },
    {
        0x2E7DF8B76080BD07ULL, 0xF5DBAC250B8A28B9ULL, 0x853C3727AE9DA4CCULL, 0xB2F1F5F3D9E5A26DULL,
        0x3FCE22012D337847ULL, 0x6B5A3E6DB7EEE347ULL, 0x171582CD59DDE50DULL, 0xC0C0B3095EE62A8AULL,
        0x665B25C6F6A203D2ULL, 0x3099AED93B6AE69FULL, 0x801DF6092BE69C38ULL, 0x8066AD0CDFFF43CDULL,
        0x8AF9D44A5F4FDC6BULL, 0xD80219CD97C0D762ULL, 0x10C9CEBA14148EBBULL, 0x539BD4C3F2F24474ULL
    }
};

__device__ __forceinline__ void generated_function(const uint64_t* __restrict__ input, uint64_t* __restrict__ output) {
    uint64_t node_34 = input[0] + input[8];
    uint64_t node_38 = input[4] + input[12];
    uint64_t node_36 = input[2] + input[10];
    uint64_t node_40 = input[6] + input[14];
    uint64_t node_35 = input[1] + input[9];
    uint64_t node_39 = input[5] + input[13];
    uint64_t node_37 = input[3] + input[11];
    uint64_t node_41 = input[7] + input[15];
    uint64_t node_50 = node_34 + node_38;
    uint64_t node_52 = node_36 + node_40;
    uint64_t node_51 = node_35 + node_39;
    uint64_t node_53 = node_37 + node_41;
    uint64_t node_160 = input[0] - input[8];
    uint64_t node_161 = input[1] - input[9];
    uint64_t node_165 = input[5] - input[13];
    uint64_t node_163 = input[3] - input[11];
    uint64_t node_167 = input[7] - input[15];
    uint64_t node_162 = input[2] - input[10];
    uint64_t node_166 = input[6] - input[14];
    uint64_t node_164 = input[4] - input[12];
    uint64_t node_58 = node_50 + node_52;
    uint64_t node_59 = node_51 + node_53;
    uint64_t node_90 = node_34 - node_38;
    uint64_t node_91 = node_35 - node_39;
    uint64_t node_93 = node_37 - node_41;
    uint64_t node_92 = node_36 - node_40;
    uint64_t node_64 = (node_58 + node_59) * 524757;
    uint64_t node_67 = (node_58 - node_59) * 52427;
    uint64_t node_71 = node_50 - node_52;
    uint64_t node_72 = node_51 - node_53;
    uint64_t node_177 = node_161 + node_165;
    uint64_t node_179 = node_163 + node_167;
    uint64_t node_178 = node_162 + node_166;
    uint64_t node_176 = node_160 + node_164;
    uint64_t node_69 = node_64 + node_67;
    uint64_t node_397 = node_71 * 18446744073709525744ULL - node_72 * 53918;
    uint64_t node_1857 = node_90 * 395512;
    uint64_t node_99 = node_91 + node_93;
    uint64_t node_1865 = node_91 * 18446744073709254400ULL;
    uint64_t node_1869 = node_93 * 179380;
    uint64_t node_1873 = node_92 * 18446744073709509368ULL;
    uint64_t node_1879 = node_160 * 35608;
    uint64_t node_185 = node_161 + node_163;
    uint64_t node_1915 = node_161 * 18446744073709340312ULL;
    uint64_t node_1921 = node_163 * 18446744073709494992ULL;
    uint64_t node_1927 = node_162 * 18446744073709450808ULL;
    uint64_t node_228 = node_165 + node_167;
    uint64_t node_1939 = node_165 * 18446744073709420056ULL;
    uint64_t node_1945 = node_167 * 18446744073709505128ULL;
    uint64_t node_1951 = node_166 * 216536;
    uint64_t node_1957 = node_164 * 18446744073709515080ULL;
    uint64_t node_70 = node_64 - node_67;
    uint64_t node_702 = node_71 * 53918 + node_72 * 18446744073709525744ULL;
    uint64_t node_1961 = node_90 * 18446744073709254400ULL;
    uint64_t node_1963 = node_91 * 395512;
    uint64_t node_1965 = node_92 * 179380;
    uint64_t node_1967 = node_93 * 18446744073709509368ULL;
    uint64_t node_1970 = node_160 * 18446744073709340312ULL;
    uint64_t node_1973 = node_161 * 35608;
    uint64_t node_1982 = node_162 * 18446744073709494992ULL;
    uint64_t node_1985 = node_163 * 18446744073709450808ULL;
    uint64_t node_1988 = node_166 * 18446744073709505128ULL;
    uint64_t node_1991 = node_167 * 216536;
    uint64_t node_1994 = node_164 * 18446744073709420056ULL;
    uint64_t node_1997 = node_165 * 18446744073709515080ULL;
    uint64_t node_98 = node_90 + node_92;
    uint64_t node_184 = node_160 + node_162;
    uint64_t node_227 = node_164 + node_166;
    uint64_t node_86 = node_69 + node_397;
    uint64_t tmp1 = node_99 * 18446744073709433780ULL;
    uint64_t node_403 = node_1857 - (tmp1 - node_1865 - node_1869 + node_1873);
    uint64_t node_271 = node_177 + node_179;
    uint64_t node_1891 = node_177 * 18446744073709208752ULL;
    uint64_t node_1897 = node_179 * 18446744073709448504ULL;
    uint64_t node_1903 = node_178 * 115728;
    uint64_t node_1909 = node_185 * 18446744073709283688ULL;
    uint64_t node_1933 = node_228 * 18446744073709373568ULL;
    uint64_t node_88 = node_70 + node_702;
    uint64_t node_708 = node_1961 + node_1963 - (node_1965 + node_1967);
    uint64_t node_1976 = node_178 * 18446744073709448504ULL;
    uint64_t node_1979 = node_179 * 115728;
    uint64_t node_87 = node_69 - node_397;
    uint64_t tmp2 = node_98 * 353264;
    uint64_t node_897 = node_1865 + tmp2 - node_1857 - node_1873 - node_1869;
    uint64_t node_2007 = node_184 * 18446744073709486416ULL;
    uint64_t node_2013 = node_227 * 180000;
    uint64_t node_89 = node_70 - node_702;
    uint64_t tmp3 = node_98 * 18446744073709433780ULL;
    uint64_t tmp4 = node_99 * 353264;
    uint64_t node_1077 = tmp3 + tmp4 - (node_1961 + node_1963) - (node_1965 + node_1967);
    uint64_t node_2020 = node_184 * 18446744073709283688ULL;
    uint64_t node_2023 = node_185 * 18446744073709486416ULL;
    uint64_t node_2026 = node_227 * 18446744073709373568ULL;
    uint64_t node_2029 = node_228 * 180000;
    uint64_t node_2035 = node_176 * 18446744073709550688ULL;
    uint64_t node_2038 = node_176 * 18446744073709208752ULL;
    uint64_t node_2041 = node_177 * 18446744073709550688ULL;
    uint64_t node_270 = node_176 + node_178;
    uint64_t node_152 = node_86 + node_403;
    uint64_t tmp5 = node_271 * 18446744073709105640ULL - node_1891 - node_1897 + node_1903;
    uint64_t tmp6 = node_1909 - node_1915 - node_1921 + node_1927;
    uint64_t tmp7 = node_1933 - node_1939 - node_1945 + node_1951;
    uint64_t node_412 = node_1879 - (tmp5 - tmp6 - tmp7 + node_1957);
    uint64_t node_154 = node_88 + node_708;
    uint64_t tmp8 = node_1976 + node_1979;
    uint64_t tmp9 = node_1982 + node_1985;
    uint64_t tmp10 = node_1988 + node_1991;
    uint64_t tmp11 = node_1994 + node_1997;
    uint64_t node_717 = node_1970 + node_1973 - (tmp8 - tmp9 - tmp10 + tmp11);
    uint64_t node_156 = node_87 + node_897;
    uint64_t tmp12 = node_1897 - node_1921 - node_1945;
    uint64_t tmp13 = node_1939 + node_2013 - node_1957 - node_1951;
    uint64_t node_906 = node_1915 + node_2007 - node_1879 - node_1927 - (tmp12 + tmp13);
    uint64_t node_158 = node_89 + node_1077;
    uint64_t tmp14 = node_1970 + node_1973;
    uint64_t tmp15 = node_1982 + node_1985;
    uint64_t tmp16 = node_2026 + node_2029;
    uint64_t tmp17 = node_1994 + node_1997;
    uint64_t tmp18 = node_1988 + node_1991;
    uint64_t node_1086 = node_2020 + node_2023 - tmp14 - tmp15 - (tmp16 - tmp17 - tmp18);
    uint64_t node_153 = node_86 - node_403;
    uint64_t tmp19 = node_1909 - node_1915 - node_1921 + node_1927;
    uint64_t tmp20 = node_1933 - node_1939 - node_1945 + node_1951;
    uint64_t node_1237 = tmp19 + node_2035 - node_1879 - node_1957 - tmp20;
    uint64_t node_155 = node_88 - node_708;
    uint64_t tmp21 = node_2038 + node_2041;
    uint64_t tmp22 = node_1970 + node_1973;
    uint64_t tmp23 = node_1994 + node_1997;
    uint64_t tmp24 = node_1988 + node_1991;
    uint64_t node_1375 = node_1982 + node_1985 + tmp21 - tmp22 - tmp23 - tmp24;
    uint64_t node_157 = node_87 - node_897;
    uint64_t tmp25 = node_270 * 114800;
    uint64_t tmp26 = node_1891 + tmp25 - node_2035 - node_1903;
    uint64_t tmp27 = node_1915 + node_2007 - node_1879 - node_1927;
    uint64_t tmp28 = node_1939 + node_2013 - node_1957 - node_1951;
    uint64_t node_1492 = node_1921 + tmp26 - tmp27 - tmp28 - node_1945;
    uint64_t node_159 = node_89 - node_1077;
    uint64_t tmp29 = node_270 * 18446744073709105640ULL;
    uint64_t tmp30 = node_271 * 114800;
    uint64_t tmp31 = node_2038 + node_2041;
    uint64_t tmp32 = node_1976 + node_1979;
    uint64_t tmp33 = node_2020 + node_2023;
    uint64_t tmp34 = node_1970 + node_1973;
    uint64_t tmp35 = node_1982 + node_1985;
    uint64_t tmp36 = node_2026 + node_2029;
    uint64_t tmp37 = node_1994 + node_1997;
    uint64_t tmp38 = node_1988 + node_1991;
    uint64_t node_1657 = tmp29 + tmp30 - tmp31 - tmp32 - (tmp33 - tmp34 - tmp35) - (tmp36 - tmp37 - tmp38);

    output[0] = node_152 + node_412;
    output[1] = node_154 + node_717;
    output[2] = node_156 + node_906;
    output[3] = node_158 + node_1086;
    output[4] = node_153 + node_1237;
    output[5] = node_155 + node_1375;
    output[6] = node_157 + node_1492;
    output[7] = node_159 + node_1657;
    output[8] = node_152 - node_412;
    output[9] = node_154 - node_717;
    output[10] = node_156 - node_906;
    output[11] = node_158 - node_1086;
    output[12] = node_153 - node_1237;
    output[13] = node_155 - node_1375;
    output[14] = node_157 - node_1492;
    output[15] = node_159 - node_1657;
}

// Shared memory S-box lookup table — loaded once per block in each mining kernel.
#ifdef MINING_CORE_UNITY
__shared__ uint8_t s_lookup_table[256];
#else
extern __shared__ uint8_t s_lookup_table[];
#endif

__device__ void sbox_layer(const uint64_t* __restrict__ state_in, uint64_t* __restrict__ state_out) {
    uint64_t e0  = state_in[0];
    uint64_t e1  = state_in[1];
    uint64_t e2  = state_in[2];
    uint64_t e3  = state_in[3];
    uint64_t e4  = state_in[4];
    uint64_t e5  = state_in[5];
    uint64_t e6  = state_in[6];
    uint64_t e7  = state_in[7];
    uint64_t e8  = state_in[8];
    uint64_t e9  = state_in[9];
    uint64_t e10 = state_in[10];
    uint64_t e11 = state_in[11];
    uint64_t e12 = state_in[12];
    uint64_t e13 = state_in[13];
    uint64_t e14 = state_in[14];
    uint64_t e15 = state_in[15];

    e0 = split_lookup_shared(e0, s_lookup_table);
    e1 = split_lookup_shared(e1, s_lookup_table);
    e2 = split_lookup_shared(e2, s_lookup_table);
    e3 = split_lookup_shared(e3, s_lookup_table);

    uint64_t x4_2  = field_mul_ptx(e4, e4);
    uint64_t x5_2  = field_mul_ptx(e5, e5);
    uint64_t x6_2  = field_mul_ptx(e6, e6);
    uint64_t x7_2  = field_mul_ptx(e7, e7);
    uint64_t x8_2  = field_mul_ptx(e8, e8);
    uint64_t x9_2  = field_mul_ptx(e9, e9);
    uint64_t x10_2 = field_mul_ptx(e10, e10);
    uint64_t x11_2 = field_mul_ptx(e11, e11);
    uint64_t x12_2 = field_mul_ptx(e12, e12);
    uint64_t x13_2 = field_mul_ptx(e13, e13);
    uint64_t x14_2 = field_mul_ptx(e14, e14);
    uint64_t x15_2 = field_mul_ptx(e15, e15);

    uint64_t x4_4  = field_mul_ptx(x4_2, x4_2);
    uint64_t x5_4  = field_mul_ptx(x5_2, x5_2);
    uint64_t x6_4  = field_mul_ptx(x6_2, x6_2);
    uint64_t x7_4  = field_mul_ptx(x7_2, x7_2);
    uint64_t x8_4  = field_mul_ptx(x8_2, x8_2);
    uint64_t x9_4  = field_mul_ptx(x9_2, x9_2);
    uint64_t x10_4 = field_mul_ptx(x10_2, x10_2);
    uint64_t x11_4 = field_mul_ptx(x11_2, x11_2);
    uint64_t x12_4 = field_mul_ptx(x12_2, x12_2);
    uint64_t x13_4 = field_mul_ptx(x13_2, x13_2);
    uint64_t x14_4 = field_mul_ptx(x14_2, x14_2);
    uint64_t x15_4 = field_mul_ptx(x15_2, x15_2);

    uint64_t x4_6  = field_mul_ptx(x4_4, x4_2);
    uint64_t x5_6  = field_mul_ptx(x5_4, x5_2);
    uint64_t x6_6  = field_mul_ptx(x6_4, x6_2);
    uint64_t x7_6  = field_mul_ptx(x7_4, x7_2);
    uint64_t x8_6  = field_mul_ptx(x8_4, x8_2);
    uint64_t x9_6  = field_mul_ptx(x9_4, x9_2);
    uint64_t x10_6 = field_mul_ptx(x10_4, x10_2);
    uint64_t x11_6 = field_mul_ptx(x11_4, x11_2);
    uint64_t x12_6 = field_mul_ptx(x12_4, x12_2);
    uint64_t x13_6 = field_mul_ptx(x13_4, x13_2);
    uint64_t x14_6 = field_mul_ptx(x14_4, x14_2);
    uint64_t x15_6 = field_mul_ptx(x15_4, x15_2);

    e4  = field_mul_ptx(x4_6, e4);
    e5  = field_mul_ptx(x5_6, e5);
    e6  = field_mul_ptx(x6_6, e6);
    e7  = field_mul_ptx(x7_6, e7);
    e8  = field_mul_ptx(x8_6, e8);
    e9  = field_mul_ptx(x9_6, e9);
    e10 = field_mul_ptx(x10_6, e10);
    e11 = field_mul_ptx(x11_6, e11);
    e12 = field_mul_ptx(x12_6, e12);
    e13 = field_mul_ptx(x13_6, e13);
    e14 = field_mul_ptx(x14_6, e14);
    e15 = field_mul_ptx(x15_6, e15);

    state_out[0]  = e0;
    state_out[1]  = e1;
    state_out[2]  = e2;
    state_out[3]  = e3;
    state_out[4]  = e4;
    state_out[5]  = e5;
    state_out[6]  = e6;
    state_out[7]  = e7;
    state_out[8]  = e8;
    state_out[9]  = e9;
    state_out[10] = e10;
    state_out[11] = e11;
    state_out[12] = e12;
    state_out[13] = e13;
    state_out[14] = e14;
    state_out[15] = e15;
}

__device__ void mds_layer(const uint64_t* state_in, uint64_t* state_out) {
    uint64_t lo[STATE_SIZE], hi[STATE_SIZE];

    #pragma unroll
    for (int i = 0; i < STATE_SIZE; i++) {
        uint32_t lo32, hi32;
        asm("mov.b64 {%0, %1}, %2;" : "=r"(lo32), "=r"(hi32) : "l"(state_in[i]));
        lo[i] = lo32;
        hi[i] = hi32;
    }

    uint64_t lo_out[STATE_SIZE], hi_out[STATE_SIZE];
    generated_function(lo, lo_out);
    generated_function(hi, hi_out);

    #pragma unroll
    for (int i = 0; i < STATE_SIZE; i++) {
        uint64_t lo_shifted = lo_out[i] >> 4;
        uint64_t hi_shifted_lo = hi_out[i] << 28;
        uint64_t hi_shifted_hi = hi_out[i] >> 36;

        uint64_t s_lo, s_hi;
        asm("{\n\t"
            ".reg .pred c;\n\t"
            "add.cc.u64 %0, %2, %3;\n\t"
            "addc.u64 %1, %4, 0;\n\t"
            "}" : "=l"(s_lo), "=l"(s_hi) : "l"(lo_shifted), "l"(hi_shifted_lo), "l"(hi_shifted_hi));

        uint64_t res;
        asm("{\n\t"
            ".reg .u64 prod;\n\t"
            ".reg .pred p;\n\t"
            "mul.lo.u64 prod, %2, 4294967295;\n\t"
            "add.cc.u64 %0, %1, prod;\n\t"
            "setp.lt.u64 p, %0, %1;\n\t"
            "selp.u64 prod, 4294967295, 0, p;\n\t"
            "add.u64 %0, %0, prod;\n\t"
            "}" : "=l"(res) : "l"(s_lo), "l"(s_hi));

        state_out[i] = res;
    }
}

__device__ __forceinline__ void mds_layer_round0_sparse(
    const uint64_t sbox_out_0,
    const uint64_t sbox_out_1,
    const uint64_t sbox_out_2,
    const uint64_t sbox_out_3,
    const uint64_t sbox_out_4,
    uint64_t x7_one,
    uint64_t* __restrict__ state_out
) {
    static constexpr uint32_t MDS_SUM_10_15[16] = {
        168244, 179170, 207371, 201069, 234966, 232623, 190779, 238434,
        208281, 198605, 218656, 178863, 195592, 169726, 150382, 175781
    };

    uint32_t s_lo[6], s_hi[6];
    asm("mov.b64 {%0, %1}, %2;" : "=r"(s_lo[0]), "=r"(s_hi[0]) : "l"(sbox_out_0));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(s_lo[1]), "=r"(s_hi[1]) : "l"(sbox_out_1));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(s_lo[2]), "=r"(s_hi[2]) : "l"(sbox_out_2));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(s_lo[3]), "=r"(s_hi[3]) : "l"(sbox_out_3));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(s_lo[4]), "=r"(s_hi[4]) : "l"(sbox_out_4));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(s_lo[5]), "=r"(s_hi[5]) : "l"(x7_one));

    #pragma unroll
    for (int i = 0; i < STATE_SIZE; i++) {
        uint32_t c0 = MDS_COEFF[i & 15];
        uint32_t c1 = MDS_COEFF[(i - 1) & 15];
        uint32_t c2 = MDS_COEFF[(i - 2) & 15];
        uint32_t c3 = MDS_COEFF[(i - 3) & 15];
        uint32_t c4 = MDS_COEFF[(i - 4) & 15];
        uint32_t cs = MDS_SUM_10_15[i];

        uint64_t lo_out = (uint64_t)s_lo[0] * c0 + (uint64_t)s_lo[1] * c1
                        + (uint64_t)s_lo[2] * c2 + (uint64_t)s_lo[3] * c3
                        + (uint64_t)s_lo[4] * c4 + (uint64_t)s_lo[5] * cs;

        uint64_t hi_out = (uint64_t)s_hi[0] * c0 + (uint64_t)s_hi[1] * c1
                        + (uint64_t)s_hi[2] * c2 + (uint64_t)s_hi[3] * c3
                        + (uint64_t)s_hi[4] * c4 + (uint64_t)s_hi[5] * cs;

        uint64_t lo_shifted = lo_out >> 4;
        uint64_t hi_shifted_lo = hi_out << 28;
        uint64_t hi_shifted_hi = hi_out >> 36;

        uint64_t r_lo, r_hi;
        asm("{\n\t"
            ".reg .pred c;\n\t"
            "add.cc.u64 %0, %2, %3;\n\t"
            "addc.u64 %1, %4, 0;\n\t"
            "}" : "=l"(r_lo), "=l"(r_hi) : "l"(lo_shifted), "l"(hi_shifted_lo), "l"(hi_shifted_hi));

        uint64_t res;
        asm("{\n\t"
            ".reg .u64 prod;\n\t"
            ".reg .pred p;\n\t"
            "mul.lo.u64 prod, %2, 4294967295;\n\t"
            "add.cc.u64 %0, %1, prod;\n\t"
            "setp.lt.u64 p, %0, %1;\n\t"
            "selp.u64 prod, 4294967295, 0, p;\n\t"
            "add.u64 %0, %0, prod;\n\t"
            "}" : "=l"(res) : "l"(r_lo), "l"(r_hi));

        state_out[i] = res;
    }
}

__device__ void round_constants_layer(int round_index, const uint64_t* state_in, uint64_t* state_out) {
    for (int i = 0; i < STATE_SIZE; ++i) {
        state_out[i] = fast_field_add(state_in[i], ROUND_CONSTANTS[round_index][i]);
    }
}

__host__ void sbox_layer_host(const uint64_t* state_in, uint64_t* state_out) {
    for (int i = 0; i < 4; ++i) {
        state_out[i] = split_lookup_host(state_in[i]);
    }
    for (int i = 4; i < STATE_SIZE; ++i) {
        state_out[i] = x7_computer_host(state_in[i]);
    }
}

__host__ void mds_layer_host(const uint64_t* state_in, uint64_t* state_out) {
    for (int i = 0; i < STATE_SIZE; ++i) {
#ifdef _WIN32
        #ifdef _MSC_VER
            uint64_t acc_low = 0;
            uint64_t acc_high = 0;
            for (int j = 0; j < STATE_SIZE; ++j) {
                uint32_t coeff = get_mds_coeff_host(i, j);
                unsigned __int64 high;
                unsigned __int64 low = _umul128(state_in[j], coeff, &high);
                uint64_t new_low = acc_low + low;
                bool carry = (new_low < acc_low);
                acc_low = new_low;
                acc_high += high + (carry ? 1 : 0);
            }
            uint64_t lower = acc_low;
            uint64_t upper = acc_high;
        #else
            __uint128_t acc = 0;
            for (int j = 0; j < STATE_SIZE; ++j) {
                uint32_t coeff = get_mds_coeff_host(i, j);
                acc += (__uint128_t)state_in[j] * coeff;
            }
            uint64_t lower = (uint64_t)acc;
            uint64_t upper = (uint64_t)(acc >> 64);
        #endif
#else
        __uint128_t acc = 0;
        for (int j = 0; j < STATE_SIZE; ++j) {
            uint32_t coeff = get_mds_coeff_host(i, j);
            acc += (__uint128_t)state_in[j] * coeff;
        }
        uint64_t lower = (uint64_t)acc;
        uint64_t upper = (uint64_t)(acc >> 64);
#endif

#ifdef _WIN32
        #ifdef _MSC_VER
            unsigned __int64 mul_high;
            unsigned __int64 mul_low = _umul128(upper, 0xFFFFFFFFULL, &mul_high);
            uint64_t new_low = lower + mul_low;
            bool carry = (new_low < lower);
            uint64_t temp = new_low + (carry ? 0xFFFFFFFFULL : 0);
        #else
            __uint128_t temp128 = (__uint128_t)lower + (__uint128_t)upper * 0xFFFFFFFFULL;
            uint64_t temp = (uint64_t)temp128;
        #endif
#else
        __uint128_t temp128 = (__uint128_t)lower + (__uint128_t)upper * 0xFFFFFFFFULL;
        uint64_t temp = (uint64_t)temp128;
#endif

#ifdef _WIN32
        #ifdef _MSC_VER
            if (temp >= 0xFFFFFFFF00000001ULL) {
                temp -= 0xFFFFFFFF00000001ULL;
            }
        #else
            if (temp128 > 0xFFFFFFFFFFFFFFFFULL) {
                temp += (uint64_t)(temp128 >> 64) * 0xFFFFFFFFULL;
            }
        #endif
#else
        if (temp128 > 0xFFFFFFFFFFFFFFFFULL) {
            temp += (uint64_t)(temp128 >> 64) * 0xFFFFFFFFULL;
        }
#endif

        while (temp >= GOLDILOCKS_MODULUS) {
            temp -= GOLDILOCKS_MODULUS;
        }

        state_out[i] = temp;
    }
}

__host__ void round_constants_layer_host(int round_index, const uint64_t* state_in, uint64_t* state_out) {
    for (int i = 0; i < STATE_SIZE; ++i) {
        state_out[i] = fast_field_add_host(state_in[i], ROUND_CONSTANTS_HOST[round_index][i]);
    }
}

__device__ __noinline__ void tip5_permutation(uint64_t* state) {
    uint64_t temp_state[STATE_SIZE];
    for (int round = 0; round < NUM_ROUNDS; ++round) {
        sbox_layer(state, temp_state);
        mds_layer(temp_state, state);
        for (int i = 0; i < STATE_SIZE; ++i) {
            state[i] = fast_field_add(state[i], ROUND_CONSTANTS[round][i]);
        }
    }
}

__device__ __noinline__ void tip5_permutation_fixed_right_zero(uint64_t* state, uint64_t x7_one) {
    uint64_t temp_state[STATE_SIZE];
    {
        uint64_t s0 = split_lookup_shared(state[0], s_lookup_table);
        uint64_t s1 = split_lookup_shared(state[1], s_lookup_table);
        uint64_t s2 = split_lookup_shared(state[2], s_lookup_table);
        uint64_t s3 = split_lookup_shared(state[3], s_lookup_table);
        uint64_t s4 = x7_computer_pipelined(state[4]);
        mds_layer_round0_sparse(s0, s1, s2, s3, s4, x7_one, state);
    }
    for (int i = 0; i < STATE_SIZE; ++i) {
        state[i] = fast_field_add(state[i], ROUND_CONSTANTS[0][i]);
    }
    for (int round = 1; round < NUM_ROUNDS; ++round) {
        sbox_layer(state, temp_state);
        mds_layer(temp_state, state);
        for (int i = 0; i < STATE_SIZE; ++i) {
            state[i] = fast_field_add(state[i], ROUND_CONSTANTS[round][i]);
        }
    }
}

__host__ void tip5_permutation_host(uint64_t* state) {
    uint64_t temp_state[STATE_SIZE];
    for (int round = 0; round < NUM_ROUNDS; ++round) {
        sbox_layer_host(state, temp_state);
        mds_layer_host(temp_state, state);
        round_constants_layer_host(round, state, temp_state);
        for (int i = 0; i < STATE_SIZE; ++i) {
            state[i] = temp_state[i];
        }
    }
}

__host__ void tip5_sponge_init_host(uint64_t* state, Domain domain) {
    for (int i = 0; i < STATE_SIZE; ++i) {
        state[i] = BFE_ZERO;
    }
    if (domain == Domain::FixedLength) {
        for (int i = RATE; i < STATE_SIZE; ++i) {
            state[i] = static_cast<uint64_t>(domain);
        }
    }
}

__host__ void tip5_sponge_permutation_host(uint64_t* state) {
    tip5_permutation_host(state);
}

__host__ void tip5_sponge_absorb_chunk_host(uint64_t* state, const uint64_t* chunk, size_t len) {
    for (size_t i = 0; i < len && i < RATE; ++i) {
        state[i] = chunk[i];
    }
    tip5_sponge_permutation_host(state);
}

__host__ void tip5_sponge_squeeze_host(uint64_t* state, uint64_t* digest) {
    for (int i = 0; i < DIGEST_LEN; ++i) {
        digest[i] = state[i];
    }
}
// ========== Inlined from tip5.cu (end) ==========

// ========== Inlined from digest.cu (start) ==========
#include "digest.cuh"

__device__ __noinline__ Digest tip5_hash_fixed_device(const Digest& left, const Digest& right) {
    uint64_t state[STATE_SIZE];
    tip5_sponge_init(state, Domain::FixedLength);
    
    // OPTIMIZATION #10: Directly write to state instead of using intermediate array
    for (int i = 0; i < DIGEST_LEN; ++i) {
        state[i] = left.values[i];
        state[i + DIGEST_LEN] = right.values[i];
    }
    tip5_permutation(state);
    
    Digest result;
    tip5_sponge_squeeze(state, result.values);
    return result;
}

__device__ Digest tip5_hash_varlen_device(const uint64_t* input, size_t input_len) {
    uint64_t state[STATE_SIZE];
    tip5_sponge_init(state, Domain::VariableLength);
    
    size_t pos = 0;
    while (pos + RATE <= input_len) {
        for (size_t i = 0; i < RATE; i++) {
            state[i] = input[pos + i];
        }
        tip5_permutation(state);
        pos += RATE;
    }
    
    size_t remaining = input_len - pos;
    
    for (size_t i = 0; i < RATE; i++) {
        state[i] = BFE_ZERO;
    }
    
    for (size_t i = 0; i < remaining; i++) {
        state[i] = input[pos + i];
    }
    
    state[remaining] = BFE_ONE;
    tip5_permutation(state);
    
    Digest result;
    tip5_sponge_squeeze(state, result.values);
    return result;
}

__host__ Digest tip5_hash_fixed_host(const Digest& left, const Digest& right) {
    uint64_t state[STATE_SIZE];
    tip5_sponge_init_host(state, Domain::FixedLength);
    
    uint64_t combined_input[RATE];
    for (int i = 0; i < DIGEST_LEN; ++i) {
        combined_input[i] = left.values[i];
        combined_input[i + DIGEST_LEN] = right.values[i];
    }
    
    tip5_sponge_absorb_chunk_host(state, combined_input, RATE);
    
    Digest result;
    tip5_sponge_squeeze_host(state, result.values);
    return result;
}

__host__ std::array<uint64_t, DIGEST_LEN> tip5_hash_fixed_host(
    const std::array<uint64_t, DIGEST_LEN>& left,
    const std::array<uint64_t, DIGEST_LEN>& right) {
    uint64_t state[STATE_SIZE];
    tip5_sponge_init_host(state, Domain::FixedLength);
    
    uint64_t combined_input[RATE];
    for (int i = 0; i < DIGEST_LEN; ++i) {
        combined_input[i] = left[i];
        combined_input[i + DIGEST_LEN] = right[i];
    }
    
    tip5_sponge_absorb_chunk_host(state, combined_input, RATE);
    
    std::array<uint64_t, DIGEST_LEN> digest;
    tip5_sponge_squeeze_host(state, digest.data());
    return digest;
}

__host__ Digest tip5_hash_varlen_host(const std::vector<uint64_t>& input) {
    uint64_t state[STATE_SIZE];
    tip5_sponge_init_host(state, Domain::VariableLength);
    
    size_t pos = 0;
    while (pos + RATE <= input.size()) {
        for (size_t i = 0; i < RATE; i++) {
            state[i] = input[pos + i];
        }
        tip5_permutation_host(state);
        pos += RATE;
    }
    
    size_t remaining = input.size() - pos;
    
    for (size_t i = 0; i < RATE; i++) {
        state[i] = BFE_ZERO;
    }
    
    for (size_t i = 0; i < remaining; i++) {
        state[i] = input[pos + i];
    }
    
    state[remaining] = BFE_ONE;
    
    tip5_permutation_host(state);
    
    Digest result;
    tip5_sponge_squeeze_host(state, result.values);
    return result;
}

std::string digest_to_hex(const Digest& digest) {
    std::ostringstream oss;
    
    // Match Rust: serialize bytes in order (little-endian per limb), fixed 40 bytes
    std::array<uint8_t, DIGEST_LEN * 8> bytes{};
    for (int limb = 0; limb < DIGEST_LEN; ++limb) {
        uint64_t value = digest.values[limb];
        for (int byte = 0; byte < 8; ++byte) {
            bytes[limb * 8 + byte] = static_cast<uint8_t>((value >> (byte * 8)) & 0xFF);
        }
    }
    
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < bytes.size(); ++i) {
        oss << std::setw(2) << static_cast<int>(bytes[i]);
    }
    
    return oss.str();
}

Digest hex_to_digest(const std::string& hex) {
    Digest result;
    
    for (int i = 0; i < DIGEST_LEN; ++i) {
        result.values[i] = 0;
    }
    
    std::string hex_str = hex;
    if (hex_str.size() >= 2 && hex_str[0] == '0' && (hex_str[1] == 'x' || hex_str[1] == 'X')) {
        hex_str = hex_str.substr(2);
    }
    
    if (hex_str.length() % 2 != 0) {
        hex_str = "0" + hex_str;
    }
    
    std::vector<uint8_t> bytes;
    bytes.reserve(hex_str.length() / 2);
    for (size_t i = 0; i + 1 < hex_str.length(); i += 2) {
        std::string byteString = hex_str.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::stoul(byteString, nullptr, 16));
        bytes.push_back(byte);
    }
    
    // Right-pad to 40 bytes if shorter
    const size_t max_bytes = DIGEST_LEN * 8;
    if (bytes.size() < max_bytes) {
        bytes.resize(max_bytes, 0);
    }
    
    const size_t nbytes = std::min(bytes.size(), max_bytes);
    for (size_t i = 0; i < nbytes; ++i) {
        const int limb_index = static_cast<int>(i / 8);
        const int byte_offset = static_cast<int>(i % 8);
        result.values[limb_index] |= static_cast<uint64_t>(bytes[i]) << (byte_offset * 8);
    }
    
    return result;
}

Digest arrayStringToDigest(const std::string& str) {
    Digest result;
    
    for (int i = 0; i < DIGEST_LEN; ++i) {
        result.values[i] = 0;
    }
    
    std::string data = str;
    
    size_t start = data.find('[');
    size_t end = data.find(']');
    if (start != std::string::npos && end != std::string::npos && end > start) {
        data = data.substr(start + 1, end - start - 1);
    }
    
    std::istringstream ss(data);
    std::string token;
    int index = 0;
    
    while (std::getline(ss, token, ',') && index < DIGEST_LEN) {
        size_t first = token.find_first_not_of(" \t\n\r");
        size_t last = token.find_last_not_of(" \t\n\r");
        if (first != std::string::npos && last != std::string::npos) {
            token = token.substr(first, last - first + 1);
        }
        
        try {
            result.values[index] = std::stoull(token);
        } catch (...) {
        }
        index++;
    }
    
    return result;
}

Digest parseDigestString(const std::string& str) {
    if (str.find('[') != std::string::npos) {
        return arrayStringToDigest(str);
    }
    
    return hex_to_digest(str);
}

bool digest_less_than_or_equal(const Digest& a, const Digest& b) {
    for (int i = DIGEST_LEN - 1; i >= 0; --i) {
        if (a.values[i] < b.values[i]) {
            return true;
        }
        if (a.values[i] > b.values[i]) {
            return false;
        }
    }
    return true;
}

__host__ Digest digest_multiply_scalar(const Digest& d, uint64_t scalar) {
    Digest result;
#ifdef _WIN32
    #ifdef _MSC_VER
        uint64_t carry_low = 0;
        uint64_t carry_high = 0;
        for (int i = 0; i < DIGEST_LEN; ++i) {
            unsigned __int64 high;
            unsigned __int64 low = _umul128(d.values[i], scalar, &high);
            uint64_t new_low = low + carry_low;
            bool carry_bit = (new_low < low);
            low = new_low;
            high += carry_high + (carry_bit ? 1 : 0);
            result.values[i] = low;
            carry_low = high;
            carry_high = 0;
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
    
    if (carry > 0) {
        for (int i = 0; i < DIGEST_LEN; ++i) {
            result.values[i] = UINT64_MAX;
        }
    }
    
    return result;
}

// Make target easier by multiplying (for testing - makes it easier to find solutions)
__host__ Digest make_target_easier(const Digest& target, uint64_t factor) {
    // Multiply target by factor to make it easier
    // This makes it 10000x easier to find solutions for testing
    return digest_multiply_scalar(target, factor);
}
// ========== Inlined from digest.cu (end) ==========

#include "merkle.cuh"

// ===== MTREE CLASS IMPLEMENTATION =====

MTree MTree::build_inplace(std::vector<Digest> leafs, 
                           std::vector<Digest> internal_nodes,
                           bool* cancel_flag) {
    // Calculate height
    size_t height = 0;
    size_t temp = leafs.size();
    while (temp > 1) {
        temp >>= 1;
        height++;
    }
    
    size_t num_sequential_layers = std::min(height, size_t(8));
    size_t seq_cutoff_height = std::max(size_t(1), height - num_sequential_layers);
    
    // First layer: connects leafs to internal nodes
    size_t range_layer_0_start = 1 << (height - 1);
    size_t range_layer_0_end = 1 << height;
    
    // EXACT CPU LOGIC: Use merkle_zip like CPU version
    merkle_zip(&internal_nodes[range_layer_0_start], leafs.data(), 
               range_layer_0_end - range_layer_0_start);
    
    if (cancel_flag && *cancel_flag) {
        return MTree();
    }
    
    // EXACT CPU LOGIC: Remaining layers connect internal nodes to internal nodes
    for (size_t layer = 1; layer < seq_cutoff_height; ++layer) {
        size_t parents_start = 1 << (height - 1 - layer);
        size_t mid_point = 1 << (height - layer);
        
        par_merkle_zip(&internal_nodes[parents_start], 
                      &internal_nodes[mid_point], 
                      mid_point - parents_start,
                      cancel_flag);
        
        if (cancel_flag && *cancel_flag) {
            return MTree();
        }
    }
    
    // EXACT CPU LOGIC: Do top of tree sequentially
    for (size_t layer = seq_cutoff_height; layer < height; ++layer) {
        size_t parents_start = 1 << (height - 1 - layer);
        size_t mid_point = 1 << (height - layer);
        
        merkle_zip(&internal_nodes[parents_start], 
                  &internal_nodes[mid_point], 
                  mid_point - parents_start);
    }
    
    return MTree(std::move(leafs), std::move(internal_nodes));
}

void MTree::merkle_zip(Digest* parents, const Digest* children, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        parents[i] = tip5_hash_fixed_host(children[2 * i], children[2 * i + 1]);
    }
}

void MTree::par_merkle_zip(Digest* parents, const Digest* children, 
                           size_t count, bool* cancel_flag) {
    // Simple parallel implementation using OpenMP if available
    // Otherwise falls back to sequential
    #pragma omp parallel for if(count > 1000)
    for (size_t i = 0; i < count; ++i) {
        if (cancel_flag && *cancel_flag) continue;
        parents[i] = tip5_hash_fixed_host(children[2 * i], children[2 * i + 1]);
    }
}

std::vector<Digest> MTree::path(size_t index) const {
    std::vector<Digest> path;
    size_t running_index = index + leafs_.size();
    path.push_back(leafs_[index ^ 1]);
    
    // Calculate tree height as log2 of number of leafs
    size_t tree_height = 0;
    size_t temp = leafs_.size();
    while (temp > 1) {
        temp >>= 1;
        tree_height++;
    }
    
    for (size_t i = 1; i < tree_height; ++i) {
        running_index >>= 1;
        path.push_back(internal_nodes_[running_index ^ 1]);
    }
    
    return path;
}

// ===== GPU PREPROCESSING KERNELS =====

__global__ void __launch_bounds__(256) bitreverse_swap_leafs_kernel(
    Digest* __restrict__ leafs,
    size_t num_leafs,
    uint32_t log2_n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_leafs) return;
    
    // Calculate bit-reversed index - match Rust bitreverse implementation exactly
    // Rust uses u32 and reverses only the lower log2_n bits
    uint32_t k = (uint32_t)idx;
    
    // Manual bit-reversal matching Rust implementation (for log2_n bits)
    // This is more accurate than __brev which reverses all 32 bits
    uint32_t rev_k = k;
    rev_k = ((rev_k & 0x55555555) << 1) | ((rev_k & 0xaaaaaaaa) >> 1);
    rev_k = ((rev_k & 0x33333333) << 2) | ((rev_k & 0xcccccccc) >> 2);
    rev_k = ((rev_k & 0x0f0f0f0f) << 4) | ((rev_k & 0xf0f0f0f0) >> 4);
    rev_k = ((rev_k & 0x00ff00ff) << 8) | ((rev_k & 0xff00ff00) >> 8);
    rev_k = __funnelshift_r(rev_k, rev_k, 16);  // rotate_right(16)
    rev_k = rev_k >> ((32 - log2_n) & 0x1f);
    
    // Only swap if k < rev_k (prevents double-swapping) - matches Rust swap_indices logic
    if (k < rev_k && rev_k < num_leafs) {
        Digest temp = leafs[k];
        leafs[k] = leafs[rev_k];
        leafs[rev_k] = temp;
    }
}

__global__ void __launch_bounds__(128) build_layer1_from_layer0_on_demand_kernel(
    Digest* __restrict__ internal_nodes,
    size_t height,
    size_t num_leafs,
    const Digest commitment
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t layer1_start = 1ULL << (height - 2); // Layer 1 start index (2^25)
    size_t layer1_count = 1ULL << (height - 2); // Number of nodes in layer 1  
    if (idx >= layer1_count) return;
    
    // Write index will be [layer1_start + idx]
    size_t write_idx = layer1_start + idx;
    
    // Layer 1 node at write_idx has children from layer 0
    size_t layer0_child_a = write_idx * 2;
    size_t layer0_child_b = layer0_child_a + 1;
    
    // Compute layer 0 children on-demand from leaves
    // Each layer 0 node is built from 2 leaves
    Digest layer0_node_a, layer0_node_b;
    
    // Compute layer 0 node a from its leaf children
    {
        uint64_t leaf_idx_a = (layer0_child_a - (1ULL << (height-1))) * 2;
        uint64_t leaf_idx_b = leaf_idx_a + 1;
        Digest leaf_a = compute_leaf_from_commitment_device_parallel(commitment, leaf_idx_a, num_leafs);
        Digest leaf_b = compute_leaf_from_commitment_device_parallel(commitment, leaf_idx_b, num_leafs);
        layer0_node_a = tip5_hash_fixed_device(leaf_a, leaf_b);
    }
    
    // Compute layer 0 node b from its leaf children
    {
        uint64_t leaf_idx_a = (layer0_child_b - (1ULL << (height-1))) * 2;
        uint64_t leaf_idx_b = leaf_idx_a + 1;
        Digest leaf_a = compute_leaf_from_commitment_device_parallel(commitment, leaf_idx_a, num_leafs);
        Digest leaf_b = compute_leaf_from_commitment_device_parallel(commitment, leaf_idx_b, num_leafs);
        layer0_node_b = tip5_hash_fixed_device(leaf_a, leaf_b);
    }
    
    // Hash layer 0 children to create layer 1 node
    internal_nodes[write_idx] = tip5_hash_fixed_device(layer0_node_a, layer0_node_b);
}

__global__ void __launch_bounds__(256) build_layer0_from_commitment_kernel(
    Digest* __restrict__ internal_nodes,
    size_t height,
    size_t num_leafs,
    const Digest commitment
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t range_layer_0_start = 1ULL << (height - 1);
    size_t layer_0_count = range_layer_0_start;
    if (idx >= layer_0_count) return;

    uint64_t leaf_a_index = (uint64_t)(idx * 2ULL);
    uint64_t leaf_b_index = leaf_a_index + 1ULL;

    // Compute leaves separately to reduce register pressure
    Digest leaf_a = compute_leaf_from_commitment_device(commitment, leaf_a_index, num_leafs);
    Digest leaf_b = compute_leaf_from_commitment_device(commitment, leaf_b_index, num_leafs);

    internal_nodes[range_layer_0_start + idx] = tip5_hash_fixed_device(leaf_a, leaf_b);
}

__global__ void __launch_bounds__(256) compute_buds_kernel(
    Digest* __restrict__ buds,
    const Digest commitment,
    size_t segment_len,
    size_t base_index) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < segment_len) {
        size_t global_idx = base_index + idx;
        // Bud computation: 32 rounds of hashing (BUDDING_ROUNDS)
        // hash = Tip5::hash_pair(hash, Digest::new([index, 0, 0, 0, round]))
        Digest hash = commitment;
        for (size_t round = 0; round < BUDDING_ROUNDS; ++round) {
            // Create Digest with values [global_idx, 0, 0, 0, round]
            // Minimize scope of round_digest to reduce register pressure
            Digest round_digest;
            round_digest.values[0] = global_idx;
            round_digest.values[1] = 0;
            round_digest.values[2] = 0;
            round_digest.values[3] = 0;
            round_digest.values[4] = round;
            hash = tip5_hash_fixed_device(hash, round_digest);
        }
        buds[global_idx] = hash;
    }
}

__global__ void __launch_bounds__(256) compute_leafs_from_buds_kernel(
    Digest* __restrict__ leafs, 
    const Digest* __restrict__ buds, 
    size_t num_leafs, size_t layer) {
    // Fast 32-bit path when safe
    if (num_leafs <= 0xFFFFFFFFu && layer < 32) {
        uint32_t idx32 = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t n32 = static_cast<uint32_t>(num_leafs);
        if (idx32 < n32) {
            uint32_t stride32 = (1u << static_cast<unsigned int>(layer));
            uint32_t buddy32 = (idx32 + stride32) & (n32 - 1u);
            leafs[idx32] = tip5_hash_fixed_device(buds[idx32], buds[buddy32]);
        }
    } else {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_leafs) {
            size_t stride = (1ULL << layer);
            size_t mask = num_leafs - 1ULL; // num_leafs is power-of-two
            size_t buddy_index = (idx + stride) & mask;
            leafs[idx] = tip5_hash_fixed_device(buds[idx], buds[buddy_index]);
        }
    }
}

__global__ void __launch_bounds__(256) merkle_zip_kernel(
    Digest* __restrict__ parents, 
    const Digest* __restrict__ children, 
    size_t count) {
    if (count <= 0xFFFFFFFFu) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < static_cast<uint32_t>(count)) {
            parents[idx] = tip5_hash_fixed_device(children[2*idx], children[2*idx+1]);
        }
    } else {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < count) {
            parents[idx] = tip5_hash_fixed_device(children[2*idx], children[2*idx+1]);
        }
    }
}

// ===== DEVICE HELPER FUNCTIONS =====

__device__ __noinline__ Digest compute_leaf_from_commitment_device(
    const Digest commitment, uint64_t base_index, size_t num_leafs) {
    const int kLevels = (int)NUM_BUD_LAYERS; // 5
    const int span = 1 << kLevels;           // 32
    Digest buf[1 << NUM_BUD_LAYERS];

    // Compute all buds for this leaf
    for (int i = 0; i < span; ++i) {
        uint64_t idx = (base_index + (uint64_t)i) % (uint64_t)num_leafs;
        // Bud computation: 32 rounds of hashing (BUDDING_ROUNDS)
        // hash = Tip5::hash_pair(hash, Digest::new([index, 0, 0, 0, round]))
        Digest hash = commitment;
        // Reduce variable scope to help register allocation
        {
            Digest round_digest;
            round_digest.values[0] = idx;
            round_digest.values[1] = 0;
            round_digest.values[2] = 0;
            round_digest.values[3] = 0;
            for (size_t round = 0; round < BUDDING_ROUNDS; ++round) {
                // Create Digest with values [idx, 0, 0, 0, round]
                round_digest.values[4] = round;
                hash = tip5_hash_fixed_device(hash, round_digest);
            }
        }
        buf[i] = hash;
    }

    // Pairwise fold with doubling stride each level
    for (int level = 0, step = 1; level < kLevels; ++level, step <<= 1) {
        for (int i = 0; i < span; i += (step << 1)) {
            buf[i] = tip5_hash_fixed_device(buf[i], buf[i + step]);
        }
    }
    return buf[0];
}

// Parallel version - same implementation but may use shared memory optimizations in future
__device__ __noinline__ Digest compute_leaf_from_commitment_device_parallel(
    const Digest commitment, uint64_t base_index, size_t num_leafs) {
    // Use the same implementation as the regular version to avoid memory access issues
    return compute_leaf_from_commitment_device(commitment, base_index, num_leafs);
}

__device__ __noinline__ Digest get_internal_node_safe(
    const Digest* d_internal_nodes,
    size_t node_index,
    size_t stored_nodes_count,
    const Digest& commitment,
    const Digest& leaf_prefix,  // Commitment (Reboot/Xnt) or prev_block_digest (HardforkAlpha)
    size_t num_leafs
) {
    // Fast path: If node is within stored buffer, return it directly
    if (node_index < stored_nodes_count) {
        return d_internal_nodes[node_index];
    }
    
    // Slow path: Compute layer 0 node on-demand if needed
    const size_t layer0_start = 1ULL << (MERKLE_TREE_HEIGHT_ - 1); // 2^26
    const size_t layer0_end = 1ULL << MERKLE_TREE_HEIGHT_;   // 2^27
    
    if (node_index >= layer0_start && node_index < layer0_end) {
        // Layer 0 node - compute from leaves
        size_t layer0_idx = node_index - layer0_start;
        uint64_t leaf_a_idx = layer0_idx * 2;
        uint64_t leaf_b_idx = leaf_a_idx + 1;
        
        // Use leaf_prefix (commitment for Reboot/Xnt, prev_block_digest for HardforkAlpha)
        Digest leaf_a = compute_leaf_from_commitment_device(leaf_prefix, leaf_a_idx, num_leafs);
        Digest leaf_b = compute_leaf_from_commitment_device(leaf_prefix, leaf_b_idx, num_leafs);
        
        // Hash and return
        return tip5_hash_fixed_device(leaf_a, leaf_b);
    }
    
    // Unknown index - return zero digest
    return Digest::default_digest();
}

__device__ void compute_merkle_path(
    Digest* path,
    size_t leaf_index,
    const Digest* d_internal_nodes,
    const Digest* d_leafs,
    size_t num_leafs,
    size_t merkle_height) {
    
    size_t running_index = leaf_index;
    
    // First level: sibling leaf
    size_t sibling_leaf_index = running_index ^ 1;
    if (d_leafs != nullptr && sibling_leaf_index < num_leafs) {
        path[0] = d_leafs[sibling_leaf_index];
    } else {
        path[0] = Digest::default_digest();
    }
    running_index >>= 1;
    
    // Subsequent levels: sibling internal nodes
    for (size_t i = 1; i < merkle_height; ++i) {
        size_t layer_start = layer_start_index_device(i - 1, num_leafs);
        size_t sibling_index = layer_start + (running_index ^ 1);
        
        path[i] = d_internal_nodes[sibling_index];
        running_index >>= 1;
    }
}

__device__ void compute_merkle_paths(
    Digest* path_a,
    Digest* path_b,
    size_t index_a,
    size_t index_b,
    const Digest* d_internal_nodes,
    const Digest* d_leafs,
    size_t num_leafs,
    size_t merkle_height) {
    
    compute_merkle_path(path_a, index_a, d_internal_nodes, d_leafs, num_leafs, merkle_height);
    compute_merkle_path(path_b, index_b, d_internal_nodes, d_leafs, num_leafs, merkle_height);
}
// ========== Inlined from kernels.cu (start) ==========
// Define before including headers to prevent extern declarations of d_gpu_range_*
#define KERNELS_DEFINING_RANGE_CONSTANTS

#include "kernels.cuh"
#include "gpu_resources.cuh"
#include "common.cuh"

// ===== GPU RANGE CONSTANTS =====
// Define the __constant__ variables here (declared extern in tip5.cuh for other TUs)
__constant__ uint64_t d_gpu_range_start;
__constant__ uint64_t d_gpu_range_size;

// ===== TOP MERKLE TREE CACHE =====
// Cache top internal nodes of Merkle tree in constant memory for fast access.
// These nodes are accessed by ALL threads, so caching eliminates global memory reads.
// TOP_TREE_CACHE_SIZE (defined in kernels.cuh) is sized to fill available constant memory.
// Store as raw uint64_t since Digest has constructor (not allowed in __constant__)
__constant__ uint64_t d_top_tree_cache[TOP_TREE_CACHE_SIZE * DIGEST_LEN];

// Flag to track if cache is initialized for current job
static bool g_top_tree_cache_initialized = false;

// Helper to read cached node as Digest - defined here where d_top_tree_cache is visible
__device__ Digest get_cached_node(size_t index) {
    Digest d;
    size_t base = index * DIGEST_LEN;
    d.values[0] = d_top_tree_cache[base];
    d.values[1] = d_top_tree_cache[base + 1];
    d.values[2] = d_top_tree_cache[base + 2];
    d.values[3] = d_top_tree_cache[base + 3];
    d.values[4] = d_top_tree_cache[base + 4];
    return d;
}

// ===== MINING OUTPUT BUFFERS =====

bool MiningOutputBuffers::allocate() {
    cudaError_t err;
    
    err = cudaMalloc(&d_solution_nonce, sizeof(uint64_t));
    if (err != cudaSuccess) return false;
    
    err = cudaMalloc(&d_solution_found, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_solution_nonce);
        d_solution_nonce = nullptr;
        return false;
    }
    
    err = cudaMalloc(&d_solution_path_a, MERKLE_TREE_HEIGHT_ * sizeof(Digest));
    if (err != cudaSuccess) {
        cudaFree(d_solution_nonce);
        cudaFree(d_solution_found);
        d_solution_nonce = nullptr;
        d_solution_found = nullptr;
        return false;
    }
    
    err = cudaMalloc(&d_solution_path_b, MERKLE_TREE_HEIGHT_ * sizeof(Digest));
    if (err != cudaSuccess) {
        cudaFree(d_solution_nonce);
        cudaFree(d_solution_found);
        cudaFree(d_solution_path_a);
        d_solution_nonce = nullptr;
        d_solution_found = nullptr;
        d_solution_path_a = nullptr;
        return false;
    }
    
    err = cudaMalloc(&d_solution_nonce_digest, sizeof(Digest));
    if (err != cudaSuccess) {
        cudaFree(d_solution_nonce);
        cudaFree(d_solution_found);
        cudaFree(d_solution_path_a);
        cudaFree(d_solution_path_b);
        d_solution_nonce = nullptr;
        d_solution_found = nullptr;
        d_solution_path_a = nullptr;
        d_solution_path_b = nullptr;
        return false;
    }
    
    err = cudaMalloc(&d_solution_final_hash, sizeof(Digest));
    if (err != cudaSuccess) {
        cudaFree(d_solution_nonce);
        cudaFree(d_solution_found);
        cudaFree(d_solution_path_a);
        cudaFree(d_solution_path_b);
        cudaFree(d_solution_nonce_digest);
        d_solution_nonce = nullptr;
        d_solution_found = nullptr;
        d_solution_path_a = nullptr;
        d_solution_path_b = nullptr;
        d_solution_nonce_digest = nullptr;
        return false;
    }
    
    return true;
}

void MiningOutputBuffers::free() {
    if (d_solution_nonce) {
        cudaFree(d_solution_nonce);
        d_solution_nonce = nullptr;
    }
    if (d_solution_found) {
        cudaFree(d_solution_found);
        d_solution_found = nullptr;
    }
    if (d_solution_path_a) {
        cudaFree(d_solution_path_a);
        d_solution_path_a = nullptr;
    }
    if (d_solution_path_b) {
        cudaFree(d_solution_path_b);
        d_solution_path_b = nullptr;
    }
    if (d_solution_nonce_digest) {
        cudaFree(d_solution_nonce_digest);
        d_solution_nonce_digest = nullptr;
    }
    if (d_solution_final_hash) {
        cudaFree(d_solution_final_hash);
        d_solution_final_hash = nullptr;
    }
}

bool MiningOutputBuffers::reset() {
    if (!d_solution_found) return false;
    
    cudaError_t err = cudaMemset(d_solution_found, 0, sizeof(int));
    return err == cudaSuccess;
}

// ===== GUESSER BUFFER MINING RESOURCES =====

bool GuesserBuffer::ensure_mining_resources() {
    // Create stream if not initialized
    if (!stream_initialized) {
        cudaError_t err = cudaStreamCreate(&mining_stream);
        if (err != cudaSuccess) {
            LOG_ERROR("cudaStreamCreate", err);
            return false;
        }
        stream_initialized = true;
    }
    
    // Allocate output buffers if not allocated
    if (!output_buffers_allocated) {
        cudaError_t err;
        
        err = cudaMalloc(&d_solution_nonce, sizeof(uint64_t));
        if (err != cudaSuccess) { LOG_ERROR("alloc d_solution_nonce", err); return false; }
        
        err = cudaMalloc(&d_solution_found, sizeof(int));
        if (err != cudaSuccess) { 
            cudaFree(d_solution_nonce); d_solution_nonce = nullptr;
            LOG_ERROR("alloc d_solution_found", err); 
            return false; 
        }
        
        err = cudaMalloc(&d_solution_path_a, MERKLE_TREE_HEIGHT_ * sizeof(Digest));
        if (err != cudaSuccess) {
            cudaFree(d_solution_nonce); d_solution_nonce = nullptr;
            cudaFree(d_solution_found); d_solution_found = nullptr;
            LOG_ERROR("alloc d_solution_path_a", err);
            return false;
        }
        
        err = cudaMalloc(&d_solution_path_b, MERKLE_TREE_HEIGHT_ * sizeof(Digest));
        if (err != cudaSuccess) {
            cudaFree(d_solution_nonce); d_solution_nonce = nullptr;
            cudaFree(d_solution_found); d_solution_found = nullptr;
            cudaFree(d_solution_path_a); d_solution_path_a = nullptr;
            LOG_ERROR("alloc d_solution_path_b", err);
            return false;
        }
        
        err = cudaMalloc(&d_solution_nonce_digest, sizeof(Digest));
        if (err != cudaSuccess) {
            cudaFree(d_solution_nonce); d_solution_nonce = nullptr;
            cudaFree(d_solution_found); d_solution_found = nullptr;
            cudaFree(d_solution_path_a); d_solution_path_a = nullptr;
            cudaFree(d_solution_path_b); d_solution_path_b = nullptr;
            LOG_ERROR("alloc d_solution_nonce_digest", err);
            return false;
        }
        
        err = cudaMalloc(&d_solution_final_hash, sizeof(Digest));
        if (err != cudaSuccess) {
            cudaFree(d_solution_nonce); d_solution_nonce = nullptr;
            cudaFree(d_solution_found); d_solution_found = nullptr;
            cudaFree(d_solution_path_a); d_solution_path_a = nullptr;
            cudaFree(d_solution_path_b); d_solution_path_b = nullptr;
            cudaFree(d_solution_nonce_digest); d_solution_nonce_digest = nullptr;
            LOG_ERROR("alloc d_solution_final_hash", err);
            return false;
        }
        
        output_buffers_allocated = true;
    }
    
    return true;
}

bool GuesserBuffer::reset_output_buffers() {
    if (!d_solution_found) return false;
    
    // Use async memset on the mining stream for better overlap
    cudaError_t err = cudaMemsetAsync(d_solution_found, 0, sizeof(int), mining_stream);
    return err == cudaSuccess;
}

// ===== TOP TREE CACHE FUNCTIONS =====

// Initialize the top tree cache from the merkle tree
// Call this once per job when the tree changes
bool initialize_top_tree_cache(const Digest* d_merkle_tree, size_t num_leafs) {
    if (g_top_tree_cache_initialized) return true;
    
    // Cache top TOP_TREE_CACHE_SIZE internal nodes in constant memory.
    // These are the first nodes (lowest indices) which correspond to the top
    // levels of the Merkle tree, accessed by all threads in the mining kernel.
    
    cudaError_t err = cudaMemcpyToSymbol(
        d_top_tree_cache, 
        d_merkle_tree,  // First TOP_TREE_CACHE_SIZE nodes (as raw bytes)
        TOP_TREE_CACHE_SIZE * DIGEST_LEN * sizeof(uint64_t),
        0,
        cudaMemcpyDeviceToDevice);
    
    if (err != cudaSuccess) {
        LOG_ERROR("initialize_top_tree_cache", err);
        return false;
    }
    
    g_top_tree_cache_initialized = true;
    return true;
}

// Reset cache flag when job changes
void reset_top_tree_cache() {
    g_top_tree_cache_initialized = false;
}

// Device function to get internal node - uses cache for small indices
__device__ __forceinline__ Digest get_internal_node_cached(
    const Digest* __restrict__ d_internal_nodes,
    size_t index,
    size_t num_leafs
) {
    // Check if this node is in our top-tree cache
    if (index < TOP_TREE_CACHE_SIZE) {
        return get_cached_node(index);
    }
    // Fall back to global memory
    return (index < num_leafs) ? d_internal_nodes[index] : Digest::default_digest();
}

// ===== HIGH-VRAM MINING KERNEL =====

__global__ void parallel_mining_kernel_high_vram(
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
    Digest* __restrict__ d_solution_final_hash) {
    
    // Load S-box lookup table into shared memory (one load per thread)
    if (threadIdx.x < 256) {
        s_lookup_table[threadIdx.x] = LOOKUP_TABLE[threadIdx.x];
    }
    __syncthreads();
    
    // Check if solution already found
    if (*d_solution_found) return;
    
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    
    // Cache constant values outside loop to prevent repeated memory accesses
    // Merkle root is constant - read once and reuse to maintain 40 MH/s performance
    const Digest* __restrict__ root_ptr = &d_internal_nodes[num_leafs - 2];
    const Digest merkle_root = *root_ptr;  // Cache constant value
    
    // Check solution flag periodically to exit early (one global read per 16K nonces)
    const uint64_t CHECK_INTERVAL = 16384ULL;
    
    // Process nonces
    for (uint64_t idx = tid; idx < num_nonces; idx += stride) {
        if ((idx & (CHECK_INTERVAL - 1)) == 0 && *d_solution_found)
            break;
        
        // Sequential nonce within GPU's range (using original working format)
        uint64_t nonce_value = d_gpu_range_start + start_nonce + idx;
        
        // Nonce digest - EXACT ORIGINAL FORMAT (this was working at 13-15 M/s!)
        Digest nonce_digest;
        nonce_digest.values[0] = nonce_value;
        nonce_digest.values[1] = 0;  // Upper bits in second limb
        nonce_digest.values[2] = 0;
        nonce_digest.values[3] = 0;
        nonce_digest.values[4] = 0;
        
        // Compute indices from index picker preimage and nonce
        uint64_t index_a, index_b;
        Pow_indices_device(hash, nonce_digest, index_a, index_b);
        
        // Compute POW hash directly from global memory - no Pow struct needed
        // This eliminates 2240 bytes of register pressure per thread
        // Uses top tree cache for fast access to upper Merkle levels
        Digest final_hash = fast_mast_hash_direct(
            mast_paths,
            nonce_digest,
            merkle_root,
            d_leafs,
            d_internal_nodes,
            index_a,  // path_index_a
            index_b,  // path_index_b
            num_leafs,
            merkle_height);
        
        // Check against target - PTX-optimized comparison (solution is rare)
        bool is_solution = digest_less_equal_ptx(final_hash, target);
        
        if (__builtin_expect(is_solution, 0)) {
            int was = atomicCAS(d_solution_found, 0, 1);
            if (was == 0) {
                // Store first limb of nonce for basic tracking
                atomicExch((unsigned long long*)d_solution_nonce, nonce_digest.values[0]);
                // Store full nonce digest so host can submit all 5 limbs
                *d_solution_nonce_digest = nonce_digest;
                
                // Store the final_hash that met the threshold (for debugging)
                *d_solution_final_hash = final_hash;
                
                // Recompute paths for solution storage (solutions are rare, so this is fine)
                // Path A
                {
                    size_t running_index = index_a + num_leafs;
                    d_solution_path_a[0] = d_leafs[index_a ^ 1];
                    for (size_t level = 1; level < merkle_height; ++level) {
                        running_index >>= 1;
                        size_t sibling_index = running_index ^ 1;
                        d_solution_path_a[level] = (sibling_index < num_leafs) 
                            ? d_internal_nodes[sibling_index] 
                            : Digest::default_digest();
                    }
                }
                // Path B
                {
                    size_t running_index = index_b + num_leafs;
                    d_solution_path_b[0] = d_leafs[index_b ^ 1];
                    for (size_t level = 1; level < merkle_height; ++level) {
                        running_index >>= 1;
                        size_t sibling_index = running_index ^ 1;
                        d_solution_path_b[level] = (sibling_index < num_leafs) 
                            ? d_internal_nodes[sibling_index] 
                            : Digest::default_digest();
                    }
                }
                
                return;
            }
        }
    }
}

// ===== MINING KERNEL FINALIZE =====
// Tiny kernel run after mining kernel (matches fast miner 3-phase structure).
// Ensures mining kernel completes before host readback; 2μs typical.

__global__ void mining_kernel_finalize(const int* __restrict__ d_solution_found) {
    (void)__ldg(d_solution_found);  // Force completion of mining kernel before this
}

// ===== LOW-VRAM MINING KERNEL =====

__global__ void parallel_mining_kernel_low_vram(
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
    Digest* __restrict__ d_solution_final_hash) {
    
    // Load S-box lookup table into shared memory
    if (threadIdx.x < 256) {
        s_lookup_table[threadIdx.x] = LOOKUP_TABLE[threadIdx.x];
    }
    __syncthreads();
    
    if (*d_solution_found) return;
    
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    
    // Check solution flag less frequently to reduce global memory traffic and cache pollution
    const uint64_t CHECK_INTERVAL = 8192ULL;
    for (uint64_t idx = tid; idx < num_nonces; idx += stride) {
        if ((idx & (CHECK_INTERVAL - 1)) == 0 && *d_solution_found) return;
        
        uint64_t nonce_value = d_gpu_range_start + start_nonce + idx;
        
        // Nonce digest - MUST match Rust: Digest(bfe_array![0, 0, 0, 0, i])
        // The nonce value goes in the LAST limb (index 4), not the first!
        Digest nonce_digest;
        nonce_digest.values[0] = 0;
        nonce_digest.values[1] = 0;
        nonce_digest.values[2] = 0;
        nonce_digest.values[3] = 0;
        nonce_digest.values[4] = nonce_value;
        
        uint64_t index_a, index_b;
        Pow_indices_device(hash, nonce_digest, index_a, index_b);
        
        // Paths are ALWAYS computed using original indices (matching Rust guess())
        // For HardforkAlpha, leaves are swapped during preprocessing, so paths use original indices
        // but the tree structure matches the swapped leaves
        uint64_t path_index_a = index_a;
        uint64_t path_index_b = index_b;
        
        // Commitment is needed for get_internal_node_safe (for computing missing nodes)
        Digest commitment = mast_paths.commit_device();
        // leaf_prefix is passed as parameter (commitment for Reboot/Xnt, prev_block_digest for HardforkAlpha)
        
        // Compute final hash directly from global memory - no Pow struct or local arrays needed
        // This eliminates 2240 bytes of register pressure per thread (same as high VRAM version)
        // Root is at last index in sequentially-built tree
        const Digest* __restrict__ root_ptr = &d_internal_nodes[stored_nodes_count - 1];
        Digest merkle_root = *root_ptr;
        
        // Use optimized low VRAM version that computes nodes on-demand
        Digest final_hash = fast_mast_hash_direct_low_vram(
            mast_paths,
            nonce_digest,
            merkle_root,
            d_internal_nodes,
            path_index_a,  // path_index_a
            path_index_b,  // path_index_b
            num_leafs,
            merkle_height,
            stored_nodes_count,
            commitment,
            leaf_prefix);
        
        // Check against target - PTX-optimized comparison (solution is rare)
        bool is_solution = digest_less_equal_ptx(final_hash, target);
        
        if (__builtin_expect(is_solution, 0)) {
            int was = atomicCAS(d_solution_found, 0, 1);
            if (was == 0) {
                atomicExch((unsigned long long*)d_solution_nonce, nonce_value);
                
                // Store the final_hash that met the threshold (for debugging)
                *d_solution_final_hash = final_hash;
                *d_solution_nonce_digest = nonce_digest;
                
                // Store paths - use get_internal_node_safe for both stored and computed nodes
                size_t running_index_a = path_index_a + num_leafs;
                size_t sibling_leaf_index_a = path_index_a ^ 1;
                // Use original index for computation (not bit-reversed)
                if (sibling_leaf_index_a < num_leafs) {
                    d_solution_path_a[0] = compute_leaf_from_commitment_device_parallel(leaf_prefix, sibling_leaf_index_a, num_leafs);
                } else {
                    d_solution_path_a[0] = Digest::default_digest();
                }
                for (size_t level = 1; level < merkle_height; ++level) {
                    running_index_a >>= 1;
                    size_t sibling_index_a = running_index_a ^ 1;
                    d_solution_path_a[level] = get_internal_node_safe(d_internal_nodes, sibling_index_a, stored_nodes_count, commitment, leaf_prefix, num_leafs);
                }
            
                size_t running_index_b = path_index_b + num_leafs;
                size_t sibling_leaf_index_b = path_index_b ^ 1;
                // Use original index for computation (not bit-reversed)
                if (sibling_leaf_index_b < num_leafs) {
                    d_solution_path_b[0] = compute_leaf_from_commitment_device_parallel(leaf_prefix, sibling_leaf_index_b, num_leafs);
                } else {
                    d_solution_path_b[0] = Digest::default_digest();
                }
                for (size_t level = 1; level < merkle_height; ++level) {
                    running_index_b >>= 1;
                    size_t sibling_index_b = running_index_b ^ 1;
                    d_solution_path_b[level] = get_internal_node_safe(d_internal_nodes, sibling_index_b, stored_nodes_count, commitment, leaf_prefix, num_leafs);
                }
            
                return;
            }
        }
    }
}

// ===== HOST MINING FUNCTIONS =====

void calculate_mining_launch_config(
    uint64_t num_nonces,
    int& threads_per_block,
    int& blocks_per_grid,
    int gpu_id) {
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    
    threads_per_block = (g_block_size > 0) ? g_block_size : MINING_THREADS_PER_BLOCK;
    
    // Calculate optimal number of blocks
    // Note: Temporarily disabled due to CUDA 13.0 compatibility issue
    // int min_grid_size, optimal_block_size;
    // cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &optimal_block_size,
    //                                     parallel_mining_kernel_high_vram, 0, 0);
    
    // Use multiple of SM count for good occupancy
    // Architecture-specific grid sizing based on compute capability
    // SM 100/120 = Blackwell (RTX 5090), SM 89 = Ada (RTX 4090), SM 90 = Hopper
    int num_sms = prop.multiProcessorCount;
    int blocks_per_sm;
    if (prop.major >= 10) {
        // Blackwell architecture (RTX 5090) - use more blocks for better occupancy
        blocks_per_sm = 128;
    } else if (prop.major == 9) {
        // Hopper architecture
        blocks_per_sm = 6;
    } else if (prop.major == 8 && prop.minor == 9) {
        // Ada Lovelace (RTX 4090) - more blocks in flight for latency hiding
        blocks_per_sm = 16;
    } else {
        // Ampere and others
        blocks_per_sm = 8;
    }
    int max_blocks = num_sms * blocks_per_sm;
    
    // Override via env or --blocks: 680 = fast miner config from profile
    if (g_blocks_per_grid > 0) {
        blocks_per_grid = std::min(g_blocks_per_grid, MAX_GRID_DIM_X);
        blocks_per_grid = std::max(blocks_per_grid, 1);
        return;
    }
    if (const char* env = std::getenv("XNT_BLOCKS_PER_GRID"); env && env[0] != '\0') {
        int v = std::atoi(env);
        if (v > 0 && v <= MAX_GRID_DIM_X) {
            blocks_per_grid = v;
            return;
        }
    }
    
    // Calculate blocks needed for nonces
    int needed_blocks = (num_nonces + threads_per_block - 1) / threads_per_block;
    
    // Cap at max blocks
    blocks_per_grid = std::min(needed_blocks, max_blocks);
    blocks_per_grid = std::min(blocks_per_grid, MAX_GRID_DIM_X);
    blocks_per_grid = 24*num_sms;
}

uint64_t get_optimal_batch_size(int gpu_id, int target_duration_ms) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    
    // Base batch size scaled by SM count and architecture
    // Target ~800-900ms kernel duration for good responsiveness
    uint64_t optimal;
    
    if (prop.major >= 10) {
        // Blackwell (RTX 5090) - 192 SMs, larger batches
        optimal = 80000000ULL;
    } else if (prop.major == 9) {
        // Hopper - medium batch
        optimal = 40000000ULL;
    } else if (prop.major == 8 && prop.minor == 9) {
        // Ada Lovelace (RTX 4090) - 64 SMs; sweet spot for throughput
        optimal = 40000000ULL;  // 40M nonces per batch
    } else {
        // Ampere and others
        optimal = 30000000ULL;
    }
    
    // Apply bounds
    const uint64_t min_batch = 1;
    const uint64_t max_batch = 100000000;  // Increased max for high-end GPUs
    
    optimal = std::max(optimal, min_batch);
    optimal = std::min(optimal, max_batch);
    
    return optimal;
}

MiningKernelType select_mining_kernel(int gpu_id) {
    VramMode mode = detect_vram_mode(gpu_id);
    
    switch (mode) {
        case VramMode::HIGH_VRAM:
            return MiningKernelType::HIGH_VRAM;
        case VramMode::LOW_VRAM:
            return MiningKernelType::LOW_VRAM;
        default:
            return MiningKernelType::HIGH_VRAM;
    }
}

bool check_kernel_launch_errors(const char* kernel_name) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR(kernel_name, err);
        return false;
    }
    return true;
}

bool sync_and_check_errors(const char* stage) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        LOG_ERROR(stage, err);
        return false;
    }
    return true;
}

// Forward declare phase-split kernels (defined after fast_mast_hash_direct)
extern __global__ void mining_kernel_phase1_high_vram(
    const Digest hash,
    const uint64_t start_nonce,
    const uint64_t num_nonces,
    uint64_t* __restrict__ d_phase_indices);
extern __global__ void mining_kernel_phase2_high_vram(
    const Digest* __restrict__ d_leafs,
    const Digest* __restrict__ d_internal_nodes,
    const Digest hash,
    const Digest target,
    const uint64_t start_nonce,
    const uint64_t num_nonces,
    const size_t num_leafs,
    const size_t merkle_height,
    const PowMastPaths mast_paths,
    const uint64_t* __restrict__ d_phase_indices,
    uint64_t* __restrict__ d_solution_nonce,
    int* __restrict__ d_solution_found,
    Digest* __restrict__ d_solution_path_a,
    Digest* __restrict__ d_solution_path_b,
    Digest* __restrict__ d_solution_nonce_digest,
    Digest* __restrict__ d_solution_final_hash);

// ===== MAIN MINING FUNCTION =====

std::optional<MiningSolution> mine_pow_with_buffer(
    GuesserBuffer& buffer,
    const Digest& target,
    const PowMastPaths& mast_paths,
    uint64_t start_nonce,
    uint64_t max_nonces,
    int consensus_rule_set,
    bool* cancel_flag) {
    
    if (!buffer.is_valid()) {
        LOG_DEBUG("mine_pow_with_buffer: invalid buffer");
        return std::nullopt;
    }
    
    // Ensure persistent mining resources are allocated (stream + output buffers)
    if (!buffer.ensure_mining_resources()) {
        LOG_DEBUG("mine_pow_with_buffer: failed to ensure mining resources");
        return std::nullopt;
    }
    
    // Reset the solution_found flag (async on stream)
    if (!buffer.reset_output_buffers()) {
        return std::nullopt;
    }
    
    // Get launch configuration
    int threads_per_block, blocks_per_grid;
    int gpu_id;
    cudaGetDevice(&gpu_id);
    calculate_mining_launch_config(max_nonces, threads_per_block, blocks_per_grid, gpu_id);
    
    // Calculate GPU's dedicated nonce range to avoid overlap with other GPUs
    // Only set once per buffer - the range doesn't change during mining
    if (!buffer.gpu_range_initialized) {
        int actual_gpu_count = g_total_gpu_count.load();
        GpuNonceRange gpu_range = calculate_gpu_range(gpu_id, actual_gpu_count);
        
        // Copy range to constant memory (avoids register pressure from extra parameters)
        cudaError_t range_err = cudaMemcpyToSymbol(d_gpu_range_start, &gpu_range.range_start, sizeof(uint64_t));
        if (range_err != cudaSuccess) {
            LOG_ERROR("copy range_start", range_err);
            return std::nullopt;
        }
        range_err = cudaMemcpyToSymbol(d_gpu_range_size, &gpu_range.range_size, sizeof(uint64_t));
        if (range_err != cudaSuccess) {
            LOG_ERROR("copy range_size", range_err);
            return std::nullopt;
        }
        
        // Initialize top tree cache (top 8 levels in constant memory)
        if (!initialize_top_tree_cache(buffer.d_merkle_tree, buffer.num_leafs)) {
            LOG_DEBUG("mine_pow_with_buffer: failed to initialize top tree cache");
            // Non-fatal - continue without cache
        }
        
        buffer.gpu_range_initialized = true;
    }
    
    // Select and launch appropriate kernel on the buffer's stream
    MiningKernelType kernel_type = select_mining_kernel(gpu_id);
    
    // Phase split default ON for HIGH_VRAM; set XNT_USE_PHASE_SPLIT=0 to disable
    bool use_phase_split = false;
    if (kernel_type == MiningKernelType::HIGH_VRAM) {
        const char* env = std::getenv("XNT_USE_PHASE_SPLIT");
        use_phase_split = (env == nullptr || env[0] != '0');
    }
    
    if (use_phase_split) {
        // Phase-split: d_phase_indices stores (index_a, index_b) = 16 bytes/nonce
        // RTX 4090: 40M nonces * 16 = 640 MB, so full batch fits
        constexpr size_t PHASE_CHUNK_MAX = 50ULL * 1024 * 1024;  // 50M nonces = 800 MB
        size_t need_cap = std::min(max_nonces, PHASE_CHUNK_MAX);
        if (buffer.d_phase_indices_capacity < need_cap) {
            if (buffer.d_phase_indices) {
                cudaFree(buffer.d_phase_indices);
                buffer.d_phase_indices = nullptr;
            }
            cudaError_t err = cudaMalloc(&buffer.d_phase_indices, need_cap * 2 * sizeof(uint64_t));
            if (err != cudaSuccess) {
                LOG_ERROR("alloc d_phase_indices", err);
                use_phase_split = false;
            } else {
                buffer.d_phase_indices_capacity = need_cap;
            }
        }
    }
    
    if (use_phase_split && buffer.d_phase_indices) {
        // Process in chunks to fit buffer
        uint64_t off = 0;
        while (off < max_nonces) {
            uint64_t chunk = std::min(max_nonces - off, buffer.d_phase_indices_capacity);
            mining_kernel_phase1_high_vram<<<blocks_per_grid, threads_per_block, 0, buffer.mining_stream>>>(
                buffer.index_picker_preimage,
                start_nonce + off,
                chunk,
                buffer.d_phase_indices);
            if (!check_kernel_launch_errors("mining_kernel_phase1")) return std::nullopt;
            
            mining_kernel_phase2_high_vram<<<blocks_per_grid, threads_per_block, 0, buffer.mining_stream>>>(
                buffer.d_leafs,
                buffer.d_merkle_tree,
                buffer.index_picker_preimage,
                target,
                start_nonce + off,
                chunk,
                buffer.num_leafs,
                MERKLE_TREE_HEIGHT_,
                mast_paths,
                buffer.d_phase_indices,
                buffer.d_solution_nonce,
                buffer.d_solution_found,
                buffer.d_solution_path_a,
                buffer.d_solution_path_b,
                buffer.d_solution_nonce_digest,
                buffer.d_solution_final_hash);
            if (!check_kernel_launch_errors("mining_kernel_phase2")) return std::nullopt;
            
            int sol = 0;
            cudaMemcpy(&sol, buffer.d_solution_found, sizeof(int), cudaMemcpyDeviceToHost);
            if (sol) break;
            off += chunk;
        }
        mining_kernel_finalize<<<1, 1, 0, buffer.mining_stream>>>(buffer.d_solution_found);
    } else switch (kernel_type) {
        case MiningKernelType::HIGH_VRAM:
            parallel_mining_kernel_high_vram<<<blocks_per_grid, threads_per_block, 0, buffer.mining_stream>>>(
                buffer.d_leafs,
                buffer.d_merkle_tree,
                buffer.index_picker_preimage,
                target,
                start_nonce,
                max_nonces,
                buffer.num_leafs,
                MERKLE_TREE_HEIGHT_,
                mast_paths,
                buffer.hash, // leaf_prefix = commitment
                consensus_rule_set,
                buffer.d_solution_nonce,
                buffer.d_solution_found,
                buffer.d_solution_path_a,
                buffer.d_solution_path_b,
                buffer.d_solution_nonce_digest,
                buffer.d_solution_final_hash);
            break;
            
        case MiningKernelType::LOW_VRAM:
            parallel_mining_kernel_low_vram<<<blocks_per_grid, threads_per_block, 0, buffer.mining_stream>>>(
                nullptr,
                buffer.d_merkle_tree,
                buffer.index_picker_preimage,
                target,
                start_nonce,
                max_nonces,
                buffer.num_leafs,
                MERKLE_TREE_HEIGHT_,
                buffer.tree_size, // stored_nodes_count
                mast_paths,
                buffer.hash,
                consensus_rule_set,
                buffer.d_solution_nonce,
                buffer.d_solution_found,
                buffer.d_solution_path_a,
                buffer.d_solution_path_b,
                buffer.d_solution_nonce_digest,
                buffer.d_solution_final_hash);
            break;
    }
    
    mining_kernel_finalize<<<1, 1, 0, buffer.mining_stream>>>(buffer.d_solution_found);
    
    // Check for launch errors
    if (!check_kernel_launch_errors("mining_kernel")) {
        return std::nullopt;
    }
    
    // Check if solution was found
    // Note: cudaMemcpy implicitly synchronizes with the device, so no need for explicit sync
    int solution_found = 0;
    cudaError_t sync_err = cudaMemcpy(&solution_found, buffer.d_solution_found, sizeof(int), cudaMemcpyDeviceToHost);
    if (sync_err != cudaSuccess) {
        LOG_ERROR("mining_kernel memcpy sync", sync_err);
        return std::nullopt;
    }
    
    if (solution_found) {
        // Copy solution data back to host
        Pow solution;
        
        uint64_t nonce_value;
        cudaMemcpy(&nonce_value, buffer.d_solution_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(&solution.nonce, buffer.d_solution_nonce_digest, sizeof(Digest), cudaMemcpyDeviceToHost);
        cudaMemcpy(solution.path_a, buffer.d_solution_path_a, 
                   MERKLE_TREE_HEIGHT_ * sizeof(Digest), cudaMemcpyDeviceToHost);
        cudaMemcpy(solution.path_b, buffer.d_solution_path_b,
                   MERKLE_TREE_HEIGHT_ * sizeof(Digest), cudaMemcpyDeviceToHost);
        
        // Copy the final_hash that kernel computed (for debugging)
        Digest kernel_final_hash;
        cudaMemcpy(&kernel_final_hash, buffer.d_solution_final_hash, sizeof(Digest), cudaMemcpyDeviceToHost);
        
        // Set root from buffer
        solution.root = buffer.merkle_root;
        
        // Return both solution and kernel_final_hash
        MiningSolution mining_solution;
        mining_solution.pow = solution;
        mining_solution.kernel_final_hash = kernel_final_hash;
        
        return mining_solution;
    }
    
    return std::nullopt;
}

// ===== MINING BATCH FUNCTION =====

MiningResult mine_batch(
    GuesserBuffer& buffer,
    const Digest& target,
    const PowMastPaths& mast_paths,
    uint64_t start_nonce,
    uint64_t batch_size,
    int consensus_rule_set,
    bool* cancel_flag) {
    
    MiningResult result;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto solution = mine_pow_with_buffer(
        buffer, target, mast_paths,
        start_nonce, batch_size, consensus_rule_set, cancel_flag);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    result.elapsed_seconds = duration.count() / 1e6;
    result.nonces_tested = batch_size;
    
    if (solution.has_value()) {
        result.solution_found = true;
        result.pow_solution = solution.value().pow;
        result.solution_hash = solution.value().kernel_final_hash;  // Use kernel's final_hash
    }
    
    return result;
}

// ============================================================================
// Double-Buffered Async Mining Implementation
// ============================================================================

bool AsyncMiningSlot::allocate() {
    if (allocated) return true;
    
    cudaError_t err;
    
    // Allocate device buffers
    err = cudaMalloc(&d_solution_nonce, sizeof(uint64_t));
    if (err != cudaSuccess) { LOG_ERROR("AsyncMiningSlot alloc d_solution_nonce", err); return false; }
    
    err = cudaMalloc(&d_solution_found, sizeof(int));
    if (err != cudaSuccess) { 
        cudaFree(d_solution_nonce); d_solution_nonce = nullptr;
        LOG_ERROR("AsyncMiningSlot alloc d_solution_found", err); 
        return false; 
    }
    
    err = cudaMalloc(&d_solution_path_a, MERKLE_TREE_HEIGHT_ * sizeof(Digest));
    if (err != cudaSuccess) {
        cudaFree(d_solution_nonce); d_solution_nonce = nullptr;
        cudaFree(d_solution_found); d_solution_found = nullptr;
        LOG_ERROR("AsyncMiningSlot alloc d_solution_path_a", err);
        return false;
    }
    
    err = cudaMalloc(&d_solution_path_b, MERKLE_TREE_HEIGHT_ * sizeof(Digest));
    if (err != cudaSuccess) {
        cudaFree(d_solution_nonce); d_solution_nonce = nullptr;
        cudaFree(d_solution_found); d_solution_found = nullptr;
        cudaFree(d_solution_path_a); d_solution_path_a = nullptr;
        LOG_ERROR("AsyncMiningSlot alloc d_solution_path_b", err);
        return false;
    }
    
    err = cudaMalloc(&d_solution_nonce_digest, sizeof(Digest));
    if (err != cudaSuccess) {
        cudaFree(d_solution_nonce); d_solution_nonce = nullptr;
        cudaFree(d_solution_found); d_solution_found = nullptr;
        cudaFree(d_solution_path_a); d_solution_path_a = nullptr;
        cudaFree(d_solution_path_b); d_solution_path_b = nullptr;
        LOG_ERROR("AsyncMiningSlot alloc d_solution_nonce_digest", err);
        return false;
    }
    
    err = cudaMalloc(&d_solution_final_hash, sizeof(Digest));
    if (err != cudaSuccess) {
        cudaFree(d_solution_nonce); d_solution_nonce = nullptr;
        cudaFree(d_solution_found); d_solution_found = nullptr;
        cudaFree(d_solution_path_a); d_solution_path_a = nullptr;
        cudaFree(d_solution_path_b); d_solution_path_b = nullptr;
        cudaFree(d_solution_nonce_digest); d_solution_nonce_digest = nullptr;
        LOG_ERROR("AsyncMiningSlot alloc d_solution_final_hash", err);
        return false;
    }
    
    // Allocate pinned host memory for async copy
    err = cudaMallocHost(&h_solution_found_pinned, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_solution_nonce); d_solution_nonce = nullptr;
        cudaFree(d_solution_found); d_solution_found = nullptr;
        cudaFree(d_solution_path_a); d_solution_path_a = nullptr;
        cudaFree(d_solution_path_b); d_solution_path_b = nullptr;
        cudaFree(d_solution_nonce_digest); d_solution_nonce_digest = nullptr;
        cudaFree(d_solution_final_hash); d_solution_final_hash = nullptr;
        LOG_ERROR("AsyncMiningSlot alloc h_solution_found_pinned", err);
        return false;
    }
    
    // Create stream with high priority for mining
    int least_priority, greatest_priority;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    err = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest_priority);
    if (err != cudaSuccess) {
        cudaFree(d_solution_nonce); d_solution_nonce = nullptr;
        cudaFree(d_solution_found); d_solution_found = nullptr;
        cudaFree(d_solution_path_a); d_solution_path_a = nullptr;
        cudaFree(d_solution_path_b); d_solution_path_b = nullptr;
        cudaFree(d_solution_nonce_digest); d_solution_nonce_digest = nullptr;
        cudaFree(d_solution_final_hash); d_solution_final_hash = nullptr;
        cudaFreeHost(h_solution_found_pinned); h_solution_found_pinned = nullptr;
        LOG_ERROR("AsyncMiningSlot create stream", err);
        return false;
    }
    
    // Create event for completion tracking
    err = cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        cudaFree(d_solution_nonce); d_solution_nonce = nullptr;
        cudaFree(d_solution_found); d_solution_found = nullptr;
        cudaFree(d_solution_path_a); d_solution_path_a = nullptr;
        cudaFree(d_solution_path_b); d_solution_path_b = nullptr;
        cudaFree(d_solution_nonce_digest); d_solution_nonce_digest = nullptr;
        cudaFree(d_solution_final_hash); d_solution_final_hash = nullptr;
        cudaFreeHost(h_solution_found_pinned); h_solution_found_pinned = nullptr;
        cudaStreamDestroy(stream); stream = nullptr;
        LOG_ERROR("AsyncMiningSlot create event", err);
        return false;
    }
    
    allocated = true;
    kernel_launched = false;
    return true;
}

void AsyncMiningSlot::free() {
    if (d_solution_nonce) { cudaFree(d_solution_nonce); d_solution_nonce = nullptr; }
    if (d_solution_found) { cudaFree(d_solution_found); d_solution_found = nullptr; }
    if (d_solution_path_a) { cudaFree(d_solution_path_a); d_solution_path_a = nullptr; }
    if (d_solution_path_b) { cudaFree(d_solution_path_b); d_solution_path_b = nullptr; }
    if (d_solution_nonce_digest) { cudaFree(d_solution_nonce_digest); d_solution_nonce_digest = nullptr; }
    if (d_solution_final_hash) { cudaFree(d_solution_final_hash); d_solution_final_hash = nullptr; }
    if (h_solution_found_pinned) { cudaFreeHost(h_solution_found_pinned); h_solution_found_pinned = nullptr; }
    if (completion_event) { cudaEventDestroy(completion_event); completion_event = nullptr; }
    if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
    allocated = false;
    kernel_launched = false;
}

bool AsyncMiningSlot::reset() {
    if (!d_solution_found || !stream) return false;
    *h_solution_found_pinned = 0;
    cudaError_t err = cudaMemsetAsync(d_solution_found, 0, sizeof(int), stream);
    kernel_launched = false;
    return err == cudaSuccess;
}

bool DoubleBufferedMiner::initialize() {
    if (initialized) return true;
    
    for (int i = 0; i < 2; ++i) {
        if (!slots[i].allocate()) {
            cleanup();
            return false;
        }
    }
    
    current_slot = 0;
    initialized = true;
    return true;
}

void DoubleBufferedMiner::cleanup() {
    for (int i = 0; i < 2; ++i) {
        slots[i].free();
    }
    initialized = false;
}

bool DoubleBufferedMiner::launch_async(
    GuesserBuffer& buffer,
    const Digest& target,
    const PowMastPaths& mast_paths,
    uint64_t start_nonce,
    uint64_t num_nonces,
    int consensus_rule_set) {
    
    return launch_mining_kernel_async(
        slots[current_slot], buffer, target, mast_paths,
        start_nonce, num_nonces, consensus_rule_set);
}

int DoubleBufferedMiner::check_previous_slot(MiningSolution* out_solution, GuesserBuffer& buffer) {
    int prev = previous_slot();
    if (!slots[prev].kernel_launched) {
        return 0;  // No previous kernel to check
    }
    return check_async_mining_result(slots[prev], out_solution, buffer);
}

bool DoubleBufferedMiner::wait_current_slot() {
    AsyncMiningSlot& slot = slots[current_slot];
    if (!slot.kernel_launched) return true;
    
    cudaError_t err = cudaStreamSynchronize(slot.stream);
    return err == cudaSuccess;
}

bool launch_mining_kernel_async(
    AsyncMiningSlot& slot,
    GuesserBuffer& buffer,
    const Digest& target,
    const PowMastPaths& mast_paths,
    uint64_t start_nonce,
    uint64_t num_nonces,
    int consensus_rule_set) {
    
    if (!buffer.is_valid() || !slot.allocated) {
        return false;
    }
    
    // Reset slot for new batch
    if (!slot.reset()) {
        return false;
    }
    
    // Store batch info for later retrieval
    slot.batch_start_nonce = start_nonce;
    slot.batch_size = num_nonces;
    
    // Ensure GPU range is initialized (one-time setup)
    if (!buffer.gpu_range_initialized) {
        int gpu_id;
        cudaGetDevice(&gpu_id);
        int actual_gpu_count = g_total_gpu_count.load();
        GpuNonceRange gpu_range = calculate_gpu_range(gpu_id, actual_gpu_count);
        
        cudaError_t range_err = cudaMemcpyToSymbol(d_gpu_range_start, &gpu_range.range_start, sizeof(uint64_t));
        if (range_err != cudaSuccess) {
            LOG_ERROR("async copy range_start", range_err);
            return false;
        }
        range_err = cudaMemcpyToSymbol(d_gpu_range_size, &gpu_range.range_size, sizeof(uint64_t));
        if (range_err != cudaSuccess) {
            LOG_ERROR("async copy range_size", range_err);
            return false;
        }
        
        if (!initialize_top_tree_cache(buffer.d_merkle_tree, buffer.num_leafs)) {
            // Non-fatal
        }
        
        buffer.gpu_range_initialized = true;
    }
    
    // Get launch configuration
    int threads_per_block, blocks_per_grid;
    int gpu_id;
    cudaGetDevice(&gpu_id);
    calculate_mining_launch_config(num_nonces, threads_per_block, blocks_per_grid, gpu_id);
    
    // Select and launch kernel on slot's stream
    MiningKernelType kernel_type = select_mining_kernel(gpu_id);
    
    switch (kernel_type) {
        case MiningKernelType::HIGH_VRAM:
            parallel_mining_kernel_high_vram<<<blocks_per_grid, threads_per_block, 0, slot.stream>>>(
                buffer.d_leafs,
                buffer.d_merkle_tree,
                buffer.index_picker_preimage,
                target,
                start_nonce,
                num_nonces,
                buffer.num_leafs,
                MERKLE_TREE_HEIGHT_,
                mast_paths,
                buffer.hash,
                consensus_rule_set,
                slot.d_solution_nonce,
                slot.d_solution_found,
                slot.d_solution_path_a,
                slot.d_solution_path_b,
                slot.d_solution_nonce_digest,
                slot.d_solution_final_hash);
            break;
            
        case MiningKernelType::LOW_VRAM:
            parallel_mining_kernel_low_vram<<<blocks_per_grid, threads_per_block, 0, slot.stream>>>(
                nullptr,
                buffer.d_merkle_tree,
                buffer.index_picker_preimage,
                target,
                start_nonce,
                num_nonces,
                buffer.num_leafs,
                MERKLE_TREE_HEIGHT_,
                buffer.tree_size,
                mast_paths,
                buffer.hash,
                consensus_rule_set,
                slot.d_solution_nonce,
                slot.d_solution_found,
                slot.d_solution_path_a,
                slot.d_solution_path_b,
                slot.d_solution_nonce_digest,
                slot.d_solution_final_hash);
            break;
    }
    
    mining_kernel_finalize<<<1, 1, 0, slot.stream>>>(slot.d_solution_found);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("async mining kernel launch", err);
        return false;
    }
    
    // Queue async copy of solution_found flag to pinned host memory
    cudaMemcpyAsync(slot.h_solution_found_pinned, slot.d_solution_found, 
                    sizeof(int), cudaMemcpyDeviceToHost, slot.stream);
    
    // Record completion event
    cudaEventRecord(slot.completion_event, slot.stream);
    
    slot.kernel_launched = true;
    return true;
}

int check_async_mining_result(
    AsyncMiningSlot& slot,
    MiningSolution* out_solution,
    GuesserBuffer& buffer) {
    
    if (!slot.kernel_launched) {
        return 0;  // No kernel was launched
    }
    
    // Check if kernel has completed (non-blocking)
    cudaError_t status = cudaEventQuery(slot.completion_event);
    
    if (status == cudaErrorNotReady) {
        return -1;  // Kernel still running
    }
    
    if (status != cudaSuccess) {
        LOG_ERROR("check_async_mining_result event query", status);
        slot.kernel_launched = false;
        return 0;
    }
    
    // Kernel completed - check if solution was found
    slot.kernel_launched = false;
    
    int solution_found = *slot.h_solution_found_pinned;
    
    if (solution_found && out_solution) {
        // Copy solution data back to host (blocking, but solutions are rare)
        Pow solution;
        
        uint64_t nonce_value;
        cudaMemcpy(&nonce_value, slot.d_solution_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(&solution.nonce, slot.d_solution_nonce_digest, sizeof(Digest), cudaMemcpyDeviceToHost);
        cudaMemcpy(solution.path_a, slot.d_solution_path_a, 
                   MERKLE_TREE_HEIGHT_ * sizeof(Digest), cudaMemcpyDeviceToHost);
        cudaMemcpy(solution.path_b, slot.d_solution_path_b,
                   MERKLE_TREE_HEIGHT_ * sizeof(Digest), cudaMemcpyDeviceToHost);
        
        Digest kernel_final_hash;
        cudaMemcpy(&kernel_final_hash, slot.d_solution_final_hash, sizeof(Digest), cudaMemcpyDeviceToHost);
        
        solution.root = buffer.merkle_root;
        
        out_solution->pow = solution;
        out_solution->kernel_final_hash = kernel_final_hash;
        
        return 1;  // Solution found
    }
    
    return 0;  // No solution
}
// ========== Inlined from kernels.cu (end) ==========

#include "pow.cuh"
#include "kernels.cuh"
#include <cstdlib>

// d_gpu_range_start and d_gpu_range_size are defined via extern in tip5.cuh
// CUDA 13+ treats extern __constant__ as static definition

__device__ Digest PowMastPaths::commit_device() const {
    // Match Rust: Tip5::hash_varlen over flattened pow, header, kernel digests
    uint64_t values[DIGEST_LEN * 6];
    size_t pos = 0;
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < DIGEST_LEN; ++j) {
            values[pos++] = pow[i].values[j];
        }
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < DIGEST_LEN; ++j) {
            values[pos++] = header[i].values[j];
        }
    }
    for (int j = 0; j < DIGEST_LEN; ++j) {
        values[pos++] = kernel[0].values[j];
    }
    
    return tip5_hash_varlen_device(values, pos);
}

namespace {
constexpr size_t kPrefetchChunkBytes = 256ULL * 1024ULL * 1024ULL;

void maybe_prefetch_managed(void* ptr, size_t bytes, int device_id) {
    const char* env = std::getenv("XNT_PREFETCH_MANAGED");
    if (!env || env[0] != '1') {
        return;
    }
    if (!ptr || bytes == 0) {
        return;
    }

    size_t offset = 0;
    while (offset < bytes) {
        size_t chunk = std::min(kPrefetchChunkBytes, bytes - offset);
        // CUDA 12.0+ uses cudaMemLocation struct, older versions use int
        // Use the struct version for CUDA 12.0+ compatibility
        cudaMemLocation location;
        location.type = cudaMemLocationTypeDevice;
        location.id = device_id;
        cudaError_t err = cudaMemPrefetchAsync(
            static_cast<char*>(ptr) + offset, chunk, location, 0);
        if (err != cudaSuccess) {
            LOG_ERROR("cudaMemPrefetchAsync (managed)", err);
            break;
        }
        offset += chunk;
    }
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        LOG_ERROR("cudaDeviceSynchronize (managed prefetch)", sync_err);
    }
}
} // namespace

__host__ Digest PowMastPaths::commit() const {
    // Match Rust: Tip5::hash_varlen over flattened pow, header, kernel digests
    std::vector<uint64_t> values;
    values.reserve(DIGEST_LEN * 6);
    
    auto append_digest = [&values](const Digest& d) {
        for (int i = 0; i < DIGEST_LEN; ++i) {
            values.push_back(d.values[i]);
        }
    };
    
    for (int i = 0; i < 3; ++i) {
        append_digest(pow[i]);
    }
    for (int i = 0; i < 2; ++i) {
        append_digest(header[i]);
    }
    append_digest(kernel[0]);
    
    return tip5_hash_varlen_host(values);
}

Digest PowMastPaths::fast_mast_hash(const Pow& pow_obj) const {
    // 39 permutations to get the block hash, when Merkle tree height is 20
    auto pow_encoding = pow_obj.encode();
    auto header_mast_hash = tip5_hash_fixed_host(tip5_hash_varlen_host(pow_encoding), this->pow[0]);
    header_mast_hash = tip5_hash_fixed_host(header_mast_hash, this->pow[1]);
    header_mast_hash = tip5_hash_fixed_host(this->pow[2], header_mast_hash);
    
    // Convert header_mast_hash to vector for varlen hash
    std::vector<uint64_t> header_encoding;
    for (const auto& val : header_mast_hash.values) {
        header_encoding.push_back(val);
    }
    auto kernel_mast_hash = tip5_hash_fixed_host(tip5_hash_varlen_host(header_encoding), this->header[0]);
    kernel_mast_hash = tip5_hash_fixed_host(kernel_mast_hash, this->header[1]);
    
    // Convert kernel_mast_hash to vector for varlen hash
    std::vector<uint64_t> kernel_encoding;
    for (const auto& val : kernel_mast_hash.values) {
        kernel_encoding.push_back(val);
    }
    return tip5_hash_fixed_host(tip5_hash_varlen_host(kernel_encoding), this->kernel[0]);
}

// Streaming hash that reads paths directly from global memory
// Avoids building large Pow struct in registers - reads on demand
__device__ __noinline__ Digest hash_pow_encoding_direct(
    const Digest& nonce,
    const Digest& root,
    const Digest* __restrict__ d_leafs,
    const Digest* __restrict__ d_internal_nodes,
    uint64_t path_index_a,
    uint64_t path_index_b,
    size_t num_leafs,
    size_t merkle_height
) {
    uint64_t state[STATE_SIZE];
    tip5_sponge_init(state, Domain::VariableLength);
    constexpr size_t kMerkleHeight = MERKLE_TREE_HEIGHT_;
    (void)merkle_height; // Mining uses fixed height; keep param to avoid API churn.
    
    // Total encoding: nonce(5) + path_b(135) + path_a(135) + root(5) = 280 words
    // RATE = 10, so 28 full chunks
    
    int chunk_pos = 0;
    
    // Inline absorb - directly write to state, permute when full
    #define ABSORB_VALUE(val) do { \
        state[chunk_pos++] = (val); \
        if (chunk_pos == RATE) { \
            tip5_permutation(state); \
            chunk_pos = 0; \
        } \
    } while(0)
    #define ABSORB_DIGEST(d) do { \
        ABSORB_VALUE((d).values[0]); \
        ABSORB_VALUE((d).values[1]); \
        ABSORB_VALUE((d).values[2]); \
        ABSORB_VALUE((d).values[3]); \
        ABSORB_VALUE((d).values[4]); \
    } while(0)
    #define ABSORB_DIGEST_PTR_LDG(p) do { \
        ABSORB_VALUE(__ldg(&(p)->values[0])); \
        ABSORB_VALUE(__ldg(&(p)->values[1])); \
        ABSORB_VALUE(__ldg(&(p)->values[2])); \
        ABSORB_VALUE(__ldg(&(p)->values[3])); \
        ABSORB_VALUE(__ldg(&(p)->values[4])); \
    } while(0)
    
    // 1. Add nonce (5 words)
    ABSORB_DIGEST(nonce);
    
    // 2. Add path_b (27 * 5 = 135 words) - optimized with read-only cache hints
    {
        size_t running_index = path_index_b + num_leafs;
        // Level 0: leaf sibling - use read-only cache hint for better caching
        size_t sibling_leaf = path_index_b ^ 1;
        // Use __ldg() for read-only global memory - enables read-only cache
        ABSORB_DIGEST_PTR_LDG(&d_leafs[sibling_leaf]);
        
        // Levels 1-26: internal nodes - use read-only cache and constant memory
        #pragma unroll
        for (size_t level = 1; level < kMerkleHeight; ++level) {
            running_index >>= 1;
            size_t sibling_index = running_index ^ 1;
                if (sibling_index < TOP_TREE_CACHE_SIZE) {
                    // Use constant memory cache for top tree levels (fastest)
                    Digest node = get_cached_node(sibling_index);
                    ABSORB_DIGEST(node);
                } else if (sibling_index < num_leafs) {
                    // Global memory - use read-only cache hint for better caching
                    ABSORB_DIGEST_PTR_LDG(&d_internal_nodes[sibling_index]);
                } else {
                    Digest node = Digest::default_digest();
                    ABSORB_DIGEST(node);
                }
        }
    }
    
    // 3. Add path_a (27 * 5 = 135 words) - optimized with read-only cache hints
    {
        size_t running_index = path_index_a + num_leafs;
        // Level 0: leaf sibling - use read-only cache hint for better caching
        size_t sibling_leaf = path_index_a ^ 1;
        ABSORB_DIGEST_PTR_LDG(&d_leafs[sibling_leaf]);
        
        // Levels 1-26: internal nodes - use read-only cache and constant memory
        #pragma unroll
        for (size_t level = 1; level < kMerkleHeight; ++level) {
            running_index >>= 1;
            size_t sibling_index = running_index ^ 1;
                if (sibling_index < TOP_TREE_CACHE_SIZE) {
                    // Use constant memory cache for top tree levels (fastest)
                    Digest node = get_cached_node(sibling_index);
                    ABSORB_DIGEST(node);
                } else if (sibling_index < num_leafs) {
                    // Global memory - use read-only cache hint for better caching
                    ABSORB_DIGEST_PTR_LDG(&d_internal_nodes[sibling_index]);
                } else {
                    Digest node = Digest::default_digest();
                    ABSORB_DIGEST(node);
                }
        }
    }
    
    // 4. Add root (5 words)
    ABSORB_DIGEST(root);
    
    #undef ABSORB_DIGEST_PTR_LDG
    #undef ABSORB_DIGEST
    #undef ABSORB_VALUE
    
    // Final padding (280 % 10 = 0, so chunk_pos should be 0)
    // Zero remaining slots and add padding marker
    #pragma unroll
    for (int i = chunk_pos; i < RATE; ++i) {
        state[i] = BFE_ZERO;
    }
    state[chunk_pos] = BFE_ONE;
    tip5_permutation(state);
    
    Digest result;
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        result.values[i] = state[i];
    }
    return result;
}

// Version that takes Pow struct (for compatibility with existing code)
__device__ __noinline__ Digest hash_pow_encoding_streaming(const Pow& pow_obj) {
    uint64_t state[STATE_SIZE];
    tip5_sponge_init(state, Domain::VariableLength);
    
    int chunk_pos = 0;
    
    #define ABSORB_VALUE(val) do { \
        state[chunk_pos++] = (val); \
        if (chunk_pos == RATE) { \
            tip5_permutation(state); \
            chunk_pos = 0; \
        } \
    } while(0)
    #define ABSORB_DIGEST(d) do { \
        ABSORB_VALUE((d).values[0]); \
        ABSORB_VALUE((d).values[1]); \
        ABSORB_VALUE((d).values[2]); \
        ABSORB_VALUE((d).values[3]); \
        ABSORB_VALUE((d).values[4]); \
    } while(0)
    
    // 1. Add nonce (5 words)
    for (int i = 0; i < DIGEST_LEN; ++i) {
        ABSORB_VALUE(pow_obj.nonce.values[i]);
    }
    
    // 2. Add path_b (27 * 5 = 135 words)
    for (int i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        for (int j = 0; j < DIGEST_LEN; ++j) {
            ABSORB_VALUE(pow_obj.path_b[i].values[j]);
        }
    }
    
    // 3. Add path_a (27 * 5 = 135 words)
    for (int i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        for (int j = 0; j < DIGEST_LEN; ++j) {
            ABSORB_VALUE(pow_obj.path_a[i].values[j]);
        }
    }
    
    // 4. Add root (5 words)
    for (int i = 0; i < DIGEST_LEN; ++i) {
        ABSORB_VALUE(pow_obj.root.values[i]);
    }
    
    #undef ABSORB_VALUE
    
    // Final padding
    for (int i = chunk_pos; i < RATE; ++i) {
        state[i] = BFE_ZERO;
    }
    state[chunk_pos] = BFE_ONE;
    tip5_permutation(state);
    
    Digest result;
    for (int i = 0; i < DIGEST_LEN; ++i) {
        result.values[i] = state[i];
    }
    return result;
}

__device__ __noinline__ Digest PowMastPaths::fast_mast_hash_device(const Pow& pow_obj) const {
    // Streaming hash - no large local array needed
    Digest pow_encoding_digest = hash_pow_encoding_streaming(pow_obj);
    
    // MAST hash chain
    Digest header_mast_hash = tip5_hash_fixed_device(pow_encoding_digest, pow[0]);
    header_mast_hash = tip5_hash_fixed_device(header_mast_hash, pow[1]);
    header_mast_hash = tip5_hash_fixed_device(pow[2], header_mast_hash);
    
    // Use specialized 5-word hash function instead of building array
    Digest kernel_mast_hash = tip5_hash_fixed_device(tip5_hash_varlen_len5_device(header_mast_hash), header[0]);
    kernel_mast_hash = tip5_hash_fixed_device(kernel_mast_hash, header[1]);
    
    return tip5_hash_fixed_device(tip5_hash_varlen_len5_device(kernel_mast_hash), kernel[0]);
}

// Low VRAM version - uses get_internal_node_safe for on-demand node computation
__device__ __noinline__ Digest hash_pow_encoding_direct_low_vram(
    const Digest& nonce,
    const Digest& root,
    const Digest* __restrict__ d_internal_nodes,
    uint64_t path_index_a,
    uint64_t path_index_b,
    size_t num_leafs,
    size_t merkle_height,
    size_t stored_nodes_count,
    const Digest& commitment,
    const Digest& leaf_prefix
) {
    uint64_t state[STATE_SIZE];
    tip5_sponge_init(state, Domain::VariableLength);
    constexpr size_t kMerkleHeight = MERKLE_TREE_HEIGHT_;
    (void)merkle_height; // Mining uses fixed height; keep param to avoid API churn.
    
    int chunk_pos = 0;
    
    #define ABSORB_VALUE(val) do { \
        state[chunk_pos++] = (val); \
        if (chunk_pos == RATE) { \
            tip5_permutation(state); \
            chunk_pos = 0; \
        } \
    } while(0)
    
    // 1. Add nonce (5 words)
    ABSORB_DIGEST(nonce);
    
    // 2. Add path_b (27 * 5 = 135 words) - compute on-demand
    {
        size_t running_index = path_index_b + num_leafs;
        size_t sibling_leaf_index = path_index_b ^ 1;
        // Compute leaf on-demand
        Digest sib = compute_leaf_from_commitment_device_parallel(leaf_prefix, sibling_leaf_index, num_leafs);
        ABSORB_DIGEST(sib);
        
        // Levels 1-26: use get_internal_node_safe
        #pragma unroll
        for (size_t level = 1; level < kMerkleHeight; ++level) {
            running_index >>= 1;
            size_t sibling_index = running_index ^ 1;
            Digest node = get_internal_node_safe(d_internal_nodes, sibling_index, stored_nodes_count, commitment, leaf_prefix, num_leafs);
            ABSORB_DIGEST(node);
        }
    }
    
    // 3. Add path_a (27 * 5 = 135 words) - compute on-demand
    {
        size_t running_index = path_index_a + num_leafs;
        size_t sibling_leaf_index = path_index_a ^ 1;
        // Compute leaf on-demand
        Digest sib = compute_leaf_from_commitment_device_parallel(leaf_prefix, sibling_leaf_index, num_leafs);
        ABSORB_DIGEST(sib);
        
        // Levels 1-26: use get_internal_node_safe
        #pragma unroll
        for (size_t level = 1; level < kMerkleHeight; ++level) {
            running_index >>= 1;
            size_t sibling_index = running_index ^ 1;
            Digest node = get_internal_node_safe(d_internal_nodes, sibling_index, stored_nodes_count, commitment, leaf_prefix, num_leafs);
            ABSORB_DIGEST(node);
        }
    }
    
    // 4. Add root (5 words)
    ABSORB_DIGEST(root);
    
    #undef ABSORB_DIGEST
    #undef ABSORB_VALUE
    
    // Final padding
    #pragma unroll
    for (int i = chunk_pos; i < RATE; ++i) {
        state[i] = BFE_ZERO;
    }
    state[chunk_pos] = BFE_ONE;
    tip5_permutation(state);
    
    Digest result;
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        result.values[i] = state[i];
    }
    return result;
}

// Helper: Fixed-length hash with state reuse (avoids reinitializing state array)
// OPTIMIZED: Fully unrolled loops
__device__ __forceinline__ void tip5_hash_fixed_inplace(uint64_t* state, const Digest& left, const Digest& right) {
    // Load left digest into state[0..4]
    state[0] = left.values[0];
    state[1] = left.values[1];
    state[2] = left.values[2];
    state[3] = left.values[3];
    state[4] = left.values[4];
    
    // Load right digest into state[5..9]
    state[5] = right.values[0];
    state[6] = right.values[1];
    state[7] = right.values[2];
    state[8] = right.values[3];
    state[9] = right.values[4];
    
    // Capacity region (state[10..15]) = FixedLength domain
    constexpr uint64_t FIXED_LEN_VAL = static_cast<uint64_t>(Domain::FixedLength);
    state[10] = FIXED_LEN_VAL;
    state[11] = FIXED_LEN_VAL;
    state[12] = FIXED_LEN_VAL;
    state[13] = FIXED_LEN_VAL;
    state[14] = FIXED_LEN_VAL;
    state[15] = FIXED_LEN_VAL;
    
    tip5_permutation(state);
}

// Helper: Varlen hash for 5 words with state reuse
// OPTIMIZED: Fully unrolled loops
__device__ __forceinline__ void tip5_hash_varlen5_inplace(uint64_t* state, const Digest& in) {
    // Load input digest into state[0..4]
    state[0] = in.values[0];
    state[1] = in.values[1];
    state[2] = in.values[2];
    state[3] = in.values[3];
    state[4] = in.values[4];
    
    // Padding: state[5] = 1, state[6..9] = 0
    state[5] = BFE_ONE;
    state[6] = 0ULL;
    state[7] = 0ULL;
    state[8] = 0ULL;
    state[9] = 0ULL;
    
    // Capacity region (state[10..15]) = VariableLength domain (0)
    state[10] = 0ULL;
    state[11] = 0ULL;
    state[12] = 0ULL;
    state[13] = 0ULL;
    state[14] = 0ULL;
    state[15] = 0ULL;
    
    tip5_permutation(state);
}

// ===== In-place helpers to avoid temporary Digest copies in the MAST chain =====

// Fixed-length hash where LEFT input is already in state[0..4]:
// state = Tip5::hash_pair(left=state[0..4], right)
__device__ __forceinline__ void tip5_hash_fixed_left_state(uint64_t* state, const Digest& right) {
    state[5] = right.values[0];
    state[6] = right.values[1];
    state[7] = right.values[2];
    state[8] = right.values[3];
    state[9] = right.values[4];

    constexpr uint64_t FIXED_LEN_VAL = static_cast<uint64_t>(Domain::FixedLength);
    state[10] = FIXED_LEN_VAL;
    state[11] = FIXED_LEN_VAL;
    state[12] = FIXED_LEN_VAL;
    state[13] = FIXED_LEN_VAL;
    state[14] = FIXED_LEN_VAL;
    state[15] = FIXED_LEN_VAL;

    tip5_permutation(state);
}

// Fixed-length hash where RIGHT input is already in state[0..4]:
// state = Tip5::hash_pair(left, right=prev_state_digest)
__device__ __forceinline__ void tip5_hash_fixed_right_state(uint64_t* state, const Digest& left) {
    // Save current digest (right input)
    uint64_t r0 = state[0];
    uint64_t r1 = state[1];
    uint64_t r2 = state[2];
    uint64_t r3 = state[3];
    uint64_t r4 = state[4];

    // Load left digest into state[0..4]
    state[0] = left.values[0];
    state[1] = left.values[1];
    state[2] = left.values[2];
    state[3] = left.values[3];
    state[4] = left.values[4];

    // Move saved right digest into state[5..9]
    state[5] = r0;
    state[6] = r1;
    state[7] = r2;
    state[8] = r3;
    state[9] = r4;

    constexpr uint64_t FIXED_LEN_VAL = static_cast<uint64_t>(Domain::FixedLength);
    state[10] = FIXED_LEN_VAL;
    state[11] = FIXED_LEN_VAL;
    state[12] = FIXED_LEN_VAL;
    state[13] = FIXED_LEN_VAL;
    state[14] = FIXED_LEN_VAL;
    state[15] = FIXED_LEN_VAL;

    tip5_permutation(state);
}

// Variable-length hash for 5 words where input is already in state[0..4]:
// state = Tip5::hash_varlen([state[0..4]]) in VariableLength domain
__device__ __forceinline__ void tip5_hash_varlen5_state(uint64_t* state) {
    // Padding: state[5] = 1, state[6..9] = 0
    state[5] = BFE_ONE;
    state[6] = 0ULL;
    state[7] = 0ULL;
    state[8] = 0ULL;
    state[9] = 0ULL;

    // Capacity region (state[10..15]) = VariableLength domain (0)
    state[10] = 0ULL;
    state[11] = 0ULL;
    state[12] = 0ULL;
    state[13] = 0ULL;
    state[14] = 0ULL;
    state[15] = 0ULL;

    tip5_permutation(state);
}

// Direct version - reads paths from global memory without building Pow struct
// OPTIMIZED: Reuses single state array for all hash operations
__device__ __noinline__ Digest fast_mast_hash_direct(
    const PowMastPaths& mast_paths,
    const Digest& nonce,
    const Digest& root,
    const Digest* __restrict__ d_leafs,
    const Digest* __restrict__ d_internal_nodes,
    uint64_t path_index_a,
    uint64_t path_index_b,
    size_t num_leafs,
    size_t merkle_height
) {
    // Streaming hash directly from global memory
    Digest pow_encoding_digest = hash_pow_encoding_direct(
        nonce, root, d_leafs, d_internal_nodes,
        path_index_a, path_index_b, num_leafs, merkle_height);
    
    // OPTIMIZED MAST hash chain - reuse single state array
    uint64_t state[STATE_SIZE];
    
    // Keep the rolling digest in state[0..4] to avoid temporary Digest copies.
    // Step 1: state = hash(pow_encoding_digest, pow[0])
    tip5_hash_fixed_inplace(state, pow_encoding_digest, mast_paths.pow[0]);
    // Step 2: state = hash(state, pow[1])
    tip5_hash_fixed_left_state(state, mast_paths.pow[1]);
    // Step 3: state = hash(pow[2], state)
    tip5_hash_fixed_right_state(state, mast_paths.pow[2]);
    // Step 4: state = varlen_hash_len5(state)
    tip5_hash_varlen5_state(state);
    // Step 5: state = hash(state, header[0])
    tip5_hash_fixed_left_state(state, mast_paths.header[0]);
    // Step 6: state = hash(state, header[1])
    tip5_hash_fixed_left_state(state, mast_paths.header[1]);
    // Step 7: state = varlen_hash_len5(state)
    tip5_hash_varlen5_state(state);
    // Step 8: state = hash(state, kernel[0])
    tip5_hash_fixed_left_state(state, mast_paths.kernel[0]);
    
    Digest result;
    result.values[0] = state[0];
    result.values[1] = state[1];
    result.values[2] = state[2];
    result.values[3] = state[3];
    result.values[4] = state[4];
    return result;
}

// ===== PHASE-SPLIT KERNELS (XNT_USE_PHASE_SPLIT=1) =====
// Phase 1: Pow_indices only → store (index_a, index_b) buffer (16 bytes/nonce)
// Phase 2: Read buffer, fast_mast_hash_direct (hash_pow_encoding + MAST) → target check
// Uses less VRAM than pow_encoding buffer; RTX 4090 can handle full batch.

__device__ __forceinline__ Digest mast_hash_from_pow_encoding(
    const Digest& pow_encoding_digest,
    const PowMastPaths& mast_paths) {
    uint64_t state[STATE_SIZE];
    tip5_hash_fixed_inplace(state, pow_encoding_digest, mast_paths.pow[0]);
    tip5_hash_fixed_left_state(state, mast_paths.pow[1]);
    tip5_hash_fixed_right_state(state, mast_paths.pow[2]);
    tip5_hash_varlen5_state(state);
    tip5_hash_fixed_left_state(state, mast_paths.header[0]);
    tip5_hash_fixed_left_state(state, mast_paths.header[1]);
    tip5_hash_varlen5_state(state);
    tip5_hash_fixed_left_state(state, mast_paths.kernel[0]);
    Digest result;
    result.values[0] = state[0];
    result.values[1] = state[1];
    result.values[2] = state[2];
    result.values[3] = state[3];
    result.values[4] = state[4];
    return result;
}

__global__ void __launch_bounds__(256, 2) mining_kernel_phase1_high_vram(
    const Digest hash,
    const uint64_t start_nonce,
    const uint64_t num_nonces,
    uint64_t* __restrict__ d_phase_indices) {
    
    if (threadIdx.x < 256) {
        s_lookup_table[threadIdx.x] = LOOKUP_TABLE[threadIdx.x];
    }
    __syncthreads();
    
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    
    for (uint64_t idx = tid; idx < num_nonces; idx += stride) {
        uint64_t nonce_value = d_gpu_range_start + start_nonce + idx;
        Digest nonce_digest;
        nonce_digest.values[0] = nonce_value;
        nonce_digest.values[1] = 0;
        nonce_digest.values[2] = 0;
        nonce_digest.values[3] = 0;
        nonce_digest.values[4] = 0;
        
        uint64_t index_a, index_b;
        Pow_indices_device(hash, nonce_digest, index_a, index_b);
        
        d_phase_indices[idx * 2] = index_a;
        d_phase_indices[idx * 2 + 1] = index_b;
    }
}

__global__ void __launch_bounds__(256, 2) mining_kernel_phase2_high_vram(
    const Digest* __restrict__ d_leafs,
    const Digest* __restrict__ d_internal_nodes,
    const Digest hash,
    const Digest target,
    const uint64_t start_nonce,
    const uint64_t num_nonces,
    const size_t num_leafs,
    const size_t merkle_height,
    const PowMastPaths mast_paths,
    const uint64_t* __restrict__ d_phase_indices,
    uint64_t* __restrict__ d_solution_nonce,
    int* __restrict__ d_solution_found,
    Digest* __restrict__ d_solution_path_a,
    Digest* __restrict__ d_solution_path_b,
    Digest* __restrict__ d_solution_nonce_digest,
    Digest* __restrict__ d_solution_final_hash) {
    
    if (threadIdx.x < 256) {
        s_lookup_table[threadIdx.x] = LOOKUP_TABLE[threadIdx.x];
    }
    __syncthreads();
    
    if (*d_solution_found) return;
    
    const Digest merkle_root = d_internal_nodes[num_leafs - 2];
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    const uint64_t CHECK_INTERVAL = 16384ULL;
    
    for (uint64_t idx = tid; idx < num_nonces; idx += stride) {
        if ((idx & (CHECK_INTERVAL - 1)) == 0 && *d_solution_found) break;
        
        uint64_t index_a = d_phase_indices[idx * 2];
        uint64_t index_b = d_phase_indices[idx * 2 + 1];
        
        uint64_t nonce_value = d_gpu_range_start + start_nonce + idx;
        Digest nonce_digest;
        nonce_digest.values[0] = nonce_value;
        nonce_digest.values[1] = 0;
        nonce_digest.values[2] = 0;
        nonce_digest.values[3] = 0;
        nonce_digest.values[4] = 0;
        
        Digest pow_enc = hash_pow_encoding_direct(
            nonce_digest, merkle_root, d_leafs, d_internal_nodes,
            index_a, index_b, num_leafs, merkle_height);
        Digest final_hash = mast_hash_from_pow_encoding(pow_enc, mast_paths);
        bool is_solution = digest_less_equal_ptx(final_hash, target);
        
        if (__builtin_expect(is_solution, 0)) {
            int was = atomicCAS(d_solution_found, 0, 1);
            if (was == 0) {
                atomicExch((unsigned long long*)d_solution_nonce, nonce_digest.values[0]);
                *d_solution_nonce_digest = nonce_digest;
                *d_solution_final_hash = final_hash;
                
                size_t running_index = index_a + num_leafs;
                d_solution_path_a[0] = d_leafs[index_a ^ 1];
                for (size_t level = 1; level < merkle_height; ++level) {
                    running_index >>= 1;
                    size_t sibling_index = running_index ^ 1;
                    d_solution_path_a[level] = (sibling_index < num_leafs)
                        ? d_internal_nodes[sibling_index]
                        : Digest::default_digest();
                }
                running_index = index_b + num_leafs;
                d_solution_path_b[0] = d_leafs[index_b ^ 1];
                for (size_t level = 1; level < merkle_height; ++level) {
                    running_index >>= 1;
                    size_t sibling_index = running_index ^ 1;
                    d_solution_path_b[level] = (sibling_index < num_leafs)
                        ? d_internal_nodes[sibling_index]
                        : Digest::default_digest();
                }
                return;
            }
        }
    }
}

// Low VRAM version - uses on-demand node computation
// OPTIMIZED: Reuses single state array for all hash operations
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
    const Digest& leaf_prefix
) {
    // Streaming hash with on-demand node computation
    Digest pow_encoding_digest = hash_pow_encoding_direct_low_vram(
        nonce, root, d_internal_nodes,
        path_index_a, path_index_b, num_leafs, merkle_height,
        stored_nodes_count, commitment, leaf_prefix);
    
    // OPTIMIZED MAST hash chain - reuse single state array
    uint64_t state[STATE_SIZE];
    
    // Keep the rolling digest in state[0..4] to avoid temporary Digest copies.
    // Step 1: state = hash(pow_encoding_digest, pow[0])
    tip5_hash_fixed_inplace(state, pow_encoding_digest, mast_paths.pow[0]);
    // Step 2: state = hash(state, pow[1])
    tip5_hash_fixed_left_state(state, mast_paths.pow[1]);
    // Step 3: state = hash(pow[2], state)
    tip5_hash_fixed_right_state(state, mast_paths.pow[2]);
    // Step 4: state = varlen_hash_len5(state)
    tip5_hash_varlen5_state(state);
    // Step 5: state = hash(state, header[0])
    tip5_hash_fixed_left_state(state, mast_paths.header[0]);
    // Step 6: state = hash(state, header[1])
    tip5_hash_fixed_left_state(state, mast_paths.header[1]);
    // Step 7: state = varlen_hash_len5(state)
    tip5_hash_varlen5_state(state);
    // Step 8: state = hash(state, kernel[0])
    tip5_hash_fixed_left_state(state, mast_paths.kernel[0]);
    
    Digest result;
    result.values[0] = state[0];
    result.values[1] = state[1];
    result.values[2] = state[2];
    result.values[3] = state[3];
    result.values[4] = state[4];
    return result;
}

VramMode detect_vram_mode(int gpu_id) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, gpu_id);
    
    if (err != cudaSuccess) {
        LOG_DEBUG("Failed to get device properties for GPU " << gpu_id);
        return VramMode::HIGH_VRAM;
    }
    
    size_t total_vram_gb = prop.totalGlobalMem / (1024ULL * 1024ULL * 1024ULL);
    LOG_DEBUG("GPU " << gpu_id << " has " << total_vram_gb << " GB VRAM");
    
    if (total_vram_gb >= 11) {
        return VramMode::HIGH_VRAM;
    } else {
        return VramMode::LOW_VRAM;
    }
}

__device__ Digest Pow::bud(const Digest& commitment, uint64_t index) {
    Digest hash = commitment;
    
    for (size_t round = 0; round < BUDDING_ROUNDS; ++round) {
        Digest round_digest;
        round_digest.values[0] = index;
        round_digest.values[1] = 0;
        round_digest.values[2] = 0;
        round_digest.values[3] = 0;
        round_digest.values[4] = round;
        hash = tip5_hash_fixed_device(hash, round_digest);
    }
    
    return hash;
}

__host__ Digest Pow::bud_host(const Digest& commitment, uint64_t index) {
    Digest hash = commitment;
    
    for (size_t round = 0; round < BUDDING_ROUNDS; ++round) {
        Digest round_digest;
        round_digest.values[0] = index;
        round_digest.values[1] = 0;
        round_digest.values[2] = 0;
        round_digest.values[3] = 0;
        round_digest.values[4] = round;
        hash = tip5_hash_fixed_host(hash, round_digest);
    }
    
    return hash;
}

__device__ void Pow::indices(const Digest& hash, const Digest& nonce, 
                              uint64_t& index_a, uint64_t& index_b) {
    Digest indexer = tip5_hash_fixed_device(hash, nonce);
    
    // Use fast right-zero hash variant for the bulk of repetitions
    for (uint32_t i = 1; i < NUM_INDEX_REPETITIONS; ++i) {
        indexer = tip5_hash_fixed_right_zero_device(indexer);
    }
    
    index_a = indexer.values[0] & MERKLE_INDEX_MASK;
    index_b = indexer.values[1] & MERKLE_INDEX_MASK;
}

__host__ std::pair<uint64_t, uint64_t> Pow::indices(const Digest& hash, const Digest& nonce) {
    Digest indexer = tip5_hash_fixed_host(hash, nonce);
    
    for (uint32_t i = 1; i < NUM_INDEX_REPETITIONS; ++i) {
        indexer = tip5_hash_fixed_host(indexer, Digest::default_digest());
    }
    
    uint64_t index_a = indexer.values[0] & MERKLE_INDEX_MASK;
    uint64_t index_b = indexer.values[1] & MERKLE_INDEX_MASK;
    
    return {index_a, index_b};
}

__host__ Digest Pow::compute_leaf_from_commitment_host(const Digest& commitment, 
                                                        uint64_t index, 
                                                        uint64_t num_leafs) {
    return bud_host(commitment, index);
}

__host__ bool Pow::verify_merkle_path_host(const Digest& root, uint64_t index, 
                                            const Digest* path, const Digest& element) {
    if (index >= (1ULL << MERKLE_TREE_HEIGHT_)) {
        return false;
    }
    
    uint64_t running_index = index;
    Digest running_digest = element;
    
    for (size_t i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        if (running_index & 1) {
            // Right child: hash(sibling, running_digest)
            running_digest = tip5_hash_fixed_host(path[i], running_digest);
        } else {
            // Left child: hash(running_digest, sibling)
            running_digest = tip5_hash_fixed_host(running_digest, path[i]);
        }
        running_index >>= 1;
    }
    
    // Compare with root
    for (int i = 0; i < DIGEST_LEN; ++i) {
        if (running_digest.values[i] != root.values[i]) {
            return false;
        }
    }
    return true;
}

__host__ uint64_t Pow::bitreverse_host(uint64_t n, uint32_t log2_n) {
    uint64_t rev = 0;
    for (uint32_t i = 0; i < log2_n; ++i) {
        if ((n >> i) & 1) {
            rev |= (1ULL << (log2_n - 1 - i));
        }
    }
    return rev;
}

std::vector<uint64_t> Pow::encode() const {
    std::vector<uint64_t> encoding;
    encoding.reserve(DIGEST_LEN * (1 + 2 * MERKLE_TREE_HEIGHT_ + 1));
    
    for (int i = 0; i < DIGEST_LEN; ++i) {
        encoding.push_back(nonce.values[i]);
    }
    
    for (int i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        for (int j = 0; j < DIGEST_LEN; ++j) {
            encoding.push_back(path_b[i].values[j]);
        }
    }
    
    for (int i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        for (int j = 0; j < DIGEST_LEN; ++j) {
            encoding.push_back(path_a[i].values[j]);
        }
    }
    
    for (int i = 0; i < DIGEST_LEN; ++i) {
        encoding.push_back(root.values[i]);
    }
    
    return encoding;
}

GuesserBuffer Pow::preprocess(const PowMastPaths& mast_auth_paths,
                              const Digest& prev_block_digest,
                              int consensus_rule_set,
                              bool* cancel_flag) {
    return preprocess_gpu(mast_auth_paths, prev_block_digest, consensus_rule_set, cancel_flag);
}

__host__ GuesserBuffer Pow::preprocess_gpu(const PowMastPaths& mast_auth_paths,
                                            const Digest& prev_block_digest,
                                            int consensus_rule_set,
                                            bool* cancel_flag) {
    // For mainnet blocks >= 15256, we always use CONSENSUS_XNT
    LOG_DEBUG("preprocess: starting (XNT consensus mode)");
    
    int current_gpu = 0;
    cudaError_t cuda_error = cudaGetDevice(&current_gpu);
    if (cuda_error != cudaSuccess) {
        LOG_ERROR("get device in preprocess", cuda_error);
        return GuesserBuffer();
    }
    
    VramMode vram_mode = detect_vram_mode(current_gpu);
    
    switch (vram_mode) {
        case VramMode::HIGH_VRAM:
            return preprocess_gpu_high_vram(mast_auth_paths, prev_block_digest, 
                                            consensus_rule_set, cancel_flag);
        case VramMode::LOW_VRAM:
            return preprocess_gpu_low_vram(mast_auth_paths, prev_block_digest,
                                            consensus_rule_set, cancel_flag);
        default:
            return preprocess_gpu_high_vram(mast_auth_paths, prev_block_digest,
                                            consensus_rule_set, cancel_flag);
    }
}

__host__ GuesserBuffer Pow::preprocess_gpu_high_vram(const PowMastPaths& mast_auth_paths,
                                                      const Digest& prev_block_digest,
                                                      int consensus_rule_set,
                                                      bool* cancel_flag) {
    GuesserBuffer buffer;
    
    // For mainnet blocks >= 15256, we use CONSENSUS_XNT
    // Always use mast_auth_paths.commit() for XNT consensus
    Digest commitment = mast_auth_paths.commit();
    
    buffer.hash = commitment;
    buffer.prev_block_digest = prev_block_digest;  // Still needed for validation
    buffer.consensus_rule_set = CONSENSUS_XNT;  // Always XNT for blocks >= 15256
    buffer.mast_paths = mast_auth_paths;
    buffer.num_leafs = MERKLE_NUM_LEAFS;
    
    LOG_DEBUG("preprocess_high_vram: commitment computed");
    
    if (cancel_flag && *cancel_flag) {
        return GuesserBuffer();
    }
    
    size_t leafs_size = MERKLE_NUM_LEAFS * sizeof(Digest);
    cudaError_t alloc_err = cudaMalloc(&buffer.d_leafs, leafs_size);
    if (alloc_err != cudaSuccess) {
        LOG_ERROR("cudaMalloc leafs", alloc_err);
        return GuesserBuffer();
    }
    
    LOG_DEBUG("preprocess_high_vram: allocated " << (leafs_size / (1024*1024)) << " MB for leafs");
    
    // NOTE: d_merkle_tree allocation is deferred until after the temp buffer is freed.
    // This reduces peak GPU memory by ~5 GB, allowing async preprocessing (P2.7) to
    // run concurrently with mining on the same 24 GB GPU:
    //   Peak with deferred alloc: old_buffer(10.7GB) + new_leafs(5.4GB) + temp(5.4GB) = 21.5 GB
    //   Peak without deferral:    old_buffer(10.7GB) + new_leafs(5.4GB) + temp(5.4GB) + new_tree(5.4GB) = 26.9 GB
    
    if (cancel_flag && *cancel_flag) {
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    // Use a dedicated stream for all preprocessing kernels.
    // Kernels on the same stream execute in-order, so we only need one sync at the end.
    // This eliminates ~33 cudaDeviceSynchronize() calls (full-device barriers) and lets
    // the driver pipeline kernel launches more efficiently.
    cudaStream_t pp_stream;
    cudaError_t stream_err = cudaStreamCreate(&pp_stream);
    if (stream_err != cudaSuccess) {
        LOG_ERROR("cudaStreamCreate (preprocess)", stream_err);
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    int threadsPerBlock = 256;
    int numBlocks = (MERKLE_NUM_LEAFS + threadsPerBlock - 1) / threadsPerBlock;
    
    // Step 1: Compute buds
    compute_buds_kernel<<<numBlocks, threadsPerBlock, 0, pp_stream>>>(
        buffer.d_leafs, commitment, MERKLE_NUM_LEAFS, 0);
    
    // Step 2: Convert buds → leafs through NUM_BUD_LAYERS iterations
    size_t temp_buffer_size = MERKLE_NUM_LEAFS * sizeof(Digest);
    Digest* d_temp_leafs = nullptr;
    cudaError_t temp_alloc_err = cudaMalloc(&d_temp_leafs, temp_buffer_size);
    if (temp_alloc_err != cudaSuccess) {
        LOG_ERROR("cudaMalloc temp_leafs", temp_alloc_err);
        cudaStreamDestroy(pp_stream);
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    Digest* current_buds = buffer.d_leafs;
    Digest* current_leafs = d_temp_leafs;
    
    for (size_t layer = 0; layer < NUM_BUD_LAYERS; ++layer) {
        int layerBlocks = (MERKLE_NUM_LEAFS + threadsPerBlock - 1) / threadsPerBlock;
        compute_leafs_from_buds_kernel<<<layerBlocks, threadsPerBlock, 0, pp_stream>>>(
            current_leafs, current_buds, MERKLE_NUM_LEAFS, layer);
        std::swap(current_buds, current_leafs);
    }
    
    // Copy final leafs to buffer.d_leafs if needed (async on same stream)
    if (current_buds != buffer.d_leafs) {
        cudaMemcpyAsync(buffer.d_leafs, current_buds, temp_buffer_size,
                        cudaMemcpyDeviceToDevice, pp_stream);
    }
    
    // Sync bud+leaf phase, then free temp buffer BEFORE allocating merkle tree.
    // This keeps peak memory within ~21.5 GB (fits 24 GB alongside old mining buffer).
    cudaError_t leaf_sync = cudaStreamSynchronize(pp_stream);
    cudaFree(d_temp_leafs);
    d_temp_leafs = nullptr;
    
    if (leaf_sync != cudaSuccess) {
        LOG_ERROR("preprocessing leaf sync", leaf_sync);
        cudaStreamDestroy(pp_stream);
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    if (cancel_flag && *cancel_flag) {
        cudaStreamDestroy(pp_stream);
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    LOG_DEBUG("preprocess_high_vram: bud+leaf phase complete, allocating merkle tree");
    
    // Now allocate d_merkle_tree (temp buffer is freed, so we have room)
    size_t internal_size = (MERKLE_NUM_LEAFS - 1) * sizeof(Digest);
    alloc_err = cudaMalloc(&buffer.d_merkle_tree, internal_size);
    if (alloc_err != cudaSuccess) {
        LOG_ERROR("cudaMalloc internal_nodes", alloc_err);
        cudaStreamDestroy(pp_stream);
        buffer.cleanup();
        return GuesserBuffer();
    }
    buffer.tree_size = MERKLE_NUM_LEAFS - 1;
    
    LOG_DEBUG("preprocess_high_vram: allocated " << (internal_size / (1024*1024)) << " MB for internal nodes");
    
    // Step 3: Build Merkle tree layer-by-layer (all on same stream)
    size_t current_count = MERKLE_NUM_LEAFS;
    const Digest* current_layer = buffer.d_leafs;
    size_t write_offset = 0;
    
    for (size_t layer = 0; layer < MERKLE_TREE_HEIGHT_; ++layer) {
        size_t parent_count = current_count / 2;
        Digest* parent_layer = buffer.d_merkle_tree + write_offset;
        
        int layerBlocks = (parent_count + threadsPerBlock - 1) / threadsPerBlock;
        merkle_zip_kernel<<<layerBlocks, threadsPerBlock, 0, pp_stream>>>(
            parent_layer, current_layer, parent_count);
        
        current_layer = parent_layer;
        write_offset += parent_count;
        current_count = parent_count;
    }
    
    // Single sync: wait for Merkle tree build to complete
    cudaError_t sync_err = cudaStreamSynchronize(pp_stream);
    cudaStreamDestroy(pp_stream);
    
    if (sync_err != cudaSuccess) {
        LOG_ERROR("preprocessing stream sync", sync_err);
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    // Check for cancellation after sync
    if (cancel_flag && *cancel_flag) {
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    LOG_DEBUG("preprocess_high_vram: Merkle tree built");
    
    // Copy root to host (tree is complete now)
    cudaMemcpy(&buffer.merkle_root, buffer.d_merkle_tree + buffer.tree_size - 1,
               sizeof(Digest), cudaMemcpyDeviceToHost);
    
    // Precompute index picker preimage: Tip5::hash_pair(root, mast_auth_paths.commit())
    buffer.index_picker_preimage = tip5_hash_fixed_host(buffer.merkle_root, commitment);
    
    LOG_DEBUG("preprocess_high_vram: complete");
    return buffer;
}

__host__ GuesserBuffer Pow::preprocess_gpu_low_vram(const PowMastPaths& mast_auth_paths,
                                                     const Digest& prev_block_digest,
                                                     int consensus_rule_set,
                                                     bool* cancel_flag) {
    GuesserBuffer buffer;
    
    // For mainnet blocks >= 15256, we use CONSENSUS_XNT
    Digest commitment = mast_auth_paths.commit();
    
    buffer.hash = commitment;
    buffer.prev_block_digest = prev_block_digest;
    buffer.consensus_rule_set = CONSENSUS_XNT;
    buffer.mast_paths = mast_auth_paths;
    buffer.num_leafs = MERKLE_NUM_LEAFS;
    
    LOG_DEBUG("preprocess_low_vram: commitment computed");
    
    if (cancel_flag && *cancel_flag) {
        return GuesserBuffer();
    }
    
    // For LOW_VRAM mode: Store top 8 layers (256 nodes including root)
    // Tree height is 27, so we need layers 20-26 (top 8 layers)
    const size_t TOP_LAYERS = 8;
    const size_t STORED_LAYER_START = MERKLE_TREE_HEIGHT_ - TOP_LAYERS;  // Layer 20
    const size_t STORED_NODES_COUNT = (1ULL << TOP_LAYERS);  // 256 nodes
    
    // Allocate buffer for top layers
    size_t internal_size = STORED_NODES_COUNT * sizeof(Digest);
    cudaError_t alloc_err = cudaMalloc(&buffer.d_merkle_tree, internal_size);
    if (alloc_err != cudaSuccess) {
        LOG_ERROR("cudaMalloc internal_nodes (low_vram)", alloc_err);
        return GuesserBuffer();
    }
    buffer.tree_size = STORED_NODES_COUNT;
    
    LOG_DEBUG("preprocess_low_vram: allocated " << (internal_size / 1024) << " KB for top " << TOP_LAYERS << " layers");
    LOG_DEBUG("preprocess_low_vram: using chunked tree construction (layers 0-19 computed and discarded, layers 20-26 stored)");
    
    // Chunked tree construction: build layer-by-layer, only keeping what we need
    // Strategy: Use a sliding window of 2 layers at a time
    // For layers 0-19: compute and discard immediately
    // For layers 20-26: compute and store in our buffer
    
    int threadsPerBlock = 256;
    
    // Step 1: Compute leafs from commitment (needed for layer 0)
    // We'll compute leafs on-demand in chunks to save memory
    // Actually, we need all leafs to compute layer 0, so we need to store them temporarily
    // But we can use a smaller buffer and compute in batches
    
    // For chunked approach: compute layer 0 in chunks, then build tree layer-by-layer
    // Layer 0 needs all leafs, so we need temporary storage for leafs
    // But we can compute them in smaller batches and build tree incrementally
    
    // Actually, the most memory-efficient approach:
    // 1. Compute leafs in chunks (e.g., 1M at a time)
    // 2. For each chunk, compute the corresponding layer 0 nodes
    // 3. Build tree from layer 0 up, keeping only what we need
    
    // Simpler approach for now: Compute all leafs, but use managed memory
    // Then build tree layer-by-layer, discarding lower layers as we go
    size_t leafs_size = MERKLE_NUM_LEAFS * sizeof(Digest);
    Digest* d_leafs = nullptr;
    alloc_err = cudaMallocManaged(&d_leafs, leafs_size);
    if (alloc_err != cudaSuccess) {
        LOG_ERROR("cudaMallocManaged leafs (low_vram)", alloc_err);
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    int device_id = 0;
    cudaError_t device_err = cudaGetDevice(&device_id);
    if (device_err == cudaSuccess) {
        maybe_prefetch_managed(d_leafs, leafs_size, device_id);
    }
    
    // Compute leafs from commitment
    int numBlocks = (MERKLE_NUM_LEAFS + threadsPerBlock - 1) / threadsPerBlock;
    compute_buds_kernel<<<numBlocks, threadsPerBlock>>>(
        d_leafs, commitment, MERKLE_NUM_LEAFS, 0);
    
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        LOG_ERROR("compute_buds_kernel sync", sync_err);
        cudaFree(d_leafs);
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    // Convert buds to leafs (need temp buffer for this)
    size_t temp_leafs_size = MERKLE_NUM_LEAFS * sizeof(Digest);
    Digest* d_temp_leafs = nullptr;
    alloc_err = cudaMallocManaged(&d_temp_leafs, temp_leafs_size);
    if (alloc_err != cudaSuccess) {
        LOG_ERROR("cudaMallocManaged temp_leafs (low_vram)", alloc_err);
        cudaFree(d_leafs);
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    if (device_err == cudaSuccess) {
        maybe_prefetch_managed(d_temp_leafs, temp_leafs_size, device_id);
    }
    
    Digest* current_buds = d_leafs;
    Digest* current_leafs = d_temp_leafs;
    
    for (size_t layer = 0; layer < NUM_BUD_LAYERS; ++layer) {
        if (cancel_flag && *cancel_flag) {
            cudaFree(d_leafs);
            cudaFree(d_temp_leafs);
            buffer.cleanup();
            return GuesserBuffer();
        }
        
        int layerBlocks = (MERKLE_NUM_LEAFS + threadsPerBlock - 1) / threadsPerBlock;
        compute_leafs_from_buds_kernel<<<layerBlocks, threadsPerBlock>>>(
            current_leafs, current_buds, MERKLE_NUM_LEAFS, layer);
        
        sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            LOG_ERROR("compute_leafs_from_buds_kernel sync", sync_err);
            cudaFree(d_leafs);
            cudaFree(d_temp_leafs);
            buffer.cleanup();
            return GuesserBuffer();
        }
        
        std::swap(current_buds, current_leafs);
    }
    
    // Now current_buds contains the final leafs
    // Ensure final leafs live in d_leafs (current_buds may be d_temp_leafs if NUM_BUD_LAYERS is odd)
    if (current_buds != d_leafs) {
        cudaMemcpy(d_leafs, current_buds, leafs_size, cudaMemcpyDeviceToDevice);
        current_buds = d_leafs;
    }
    // Free the temp buffer - we only need d_leafs now
    cudaFree(d_temp_leafs);
    d_temp_leafs = nullptr;
    
    // Step 2: Build tree layer-by-layer using chunked approach
    // We'll use a sliding window: keep current and next layer
    // For layers 0-19: compute and discard
    // For layers 20-26: compute and store
    
    // Allocate buffers for sliding window (2 layers max at a time)
    // The largest layer we need to keep is layer 20 with 2^7 = 128 nodes
    // But we need to compute from layer 0 (2^26 nodes) up
    // So we need a buffer that can hold at least one full layer
    
    // Strategy: Use d_leafs as layer 0, then allocate buffers for subsequent layers
    // We'll compute layer N from layer N-1, then discard layer N-1
    
    // For layers 0-19: we need buffers that can hold up to 2^25 nodes (layer 1 is largest)
    // For layers 20-26: we store in our final buffer
    // Layer 0 is in d_leafs (2^26 nodes), layer 1 needs 2^25 nodes
    // After layer 1, we can free d_leafs and reuse buffers
    
    const size_t MAX_INTERMEDIATE_LAYER_SIZE = 1ULL << 26;  // 2^26 nodes (layer 1 size)
    size_t intermediate_buffer_size = MAX_INTERMEDIATE_LAYER_SIZE * sizeof(Digest);
    Digest* d_layer_a = nullptr;
    Digest* d_layer_b = nullptr;
    
    alloc_err = cudaMallocManaged(&d_layer_a, intermediate_buffer_size);
    if (alloc_err != cudaSuccess) {
        LOG_ERROR("cudaMallocManaged layer_a (low_vram)", alloc_err);
        cudaFree(d_leafs);
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    alloc_err = cudaMallocManaged(&d_layer_b, intermediate_buffer_size);
    if (alloc_err != cudaSuccess) {
        LOG_ERROR("cudaMallocManaged layer_b (low_vram)", alloc_err);
        cudaFree(d_layer_a);
        cudaFree(d_leafs);
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    if (device_err == cudaSuccess) {
        maybe_prefetch_managed(d_layer_a, intermediate_buffer_size, device_id);
        maybe_prefetch_managed(d_layer_b, intermediate_buffer_size, device_id);
    }
    
    // Build tree from layer 0 (leafs) up to layer 26 (root)
    // Use sliding window: current_layer -> next_layer
    const Digest* current_layer = current_buds;  // Start with leafs (layer 0)
    size_t current_layer_size = MERKLE_NUM_LEAFS;
    Digest* next_layer = d_layer_a;
    bool use_layer_a = true;
    
    size_t stored_offset = 0;  // Offset in our final buffer
    
    for (size_t layer = 0; layer < MERKLE_TREE_HEIGHT_; ++layer) {
        if (cancel_flag && *cancel_flag) {
            cudaFree(d_leafs);
            cudaFree(d_layer_a);
            cudaFree(d_layer_b);
            buffer.cleanup();
            return GuesserBuffer();
        }
        
        size_t next_layer_size = current_layer_size / 2;
        
        // Compute next layer from current layer
        int layerBlocks = (next_layer_size + threadsPerBlock - 1) / threadsPerBlock;
        merkle_zip_kernel<<<layerBlocks, threadsPerBlock>>>(
            next_layer, current_layer, next_layer_size);
        
        sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            LOG_ERROR("merkle_zip_kernel sync (layer " << layer << ")", sync_err);
            cudaFree(d_leafs);
            cudaFree(d_layer_a);
            cudaFree(d_layer_b);
            buffer.cleanup();
            return GuesserBuffer();
        }
        
        // If this is one of the layers we need to store (20-26), copy it
        // Note: layer is the index of the PARENT layer we just computed
        // So layer 0 computes layer 1, layer 1 computes layer 2, etc.
        // We want to store layers 20-26, which are computed in iterations 19-25
        if (layer >= STORED_LAYER_START - 1) {
            size_t copy_size = next_layer_size * sizeof(Digest);
            Digest* dest_ptr = buffer.d_merkle_tree + stored_offset;
            cudaMemcpy(dest_ptr, next_layer, copy_size, cudaMemcpyDeviceToDevice);
            stored_offset += next_layer_size;
            
            // If this is the root (last layer), copy it separately
            if (layer == MERKLE_TREE_HEIGHT_ - 1) {
                cudaMemcpy(&buffer.merkle_root, next_layer, sizeof(Digest),
                          cudaMemcpyDeviceToHost);
            }
        }
        
        // Prepare for next iteration
        // Swap buffers for next iteration
        if (layer < MERKLE_TREE_HEIGHT_ - 1) {
            // After computing layer 1 (iteration 0), we can free d_leafs
            if (layer == 0 && current_layer == current_buds) {
                cudaFree(d_leafs);
                d_leafs = nullptr;  // Mark as freed
            }
            
            current_layer = next_layer;
            current_layer_size = next_layer_size;
            
            // Swap buffers for next iteration
            use_layer_a = !use_layer_a;
            next_layer = use_layer_a ? d_layer_a : d_layer_b;
        }
    }
    
    // Free d_leafs if not already freed
    if (d_leafs != nullptr) {
        cudaFree(d_leafs);
    }
    
    // Free temporary buffers
    cudaFree(d_leafs);
    cudaFree(d_layer_a);
    cudaFree(d_layer_b);
    
    // Precompute index picker preimage
    buffer.index_picker_preimage = tip5_hash_fixed_host(buffer.merkle_root, commitment);
    
    LOG_DEBUG("preprocess_low_vram: complete (stored " << buffer.tree_size << " nodes)");
    return buffer;
}

__device__ void Pow_indices_device(const Digest& hash, const Digest& nonce, uint64_t& index_a, uint64_t& index_b) {
    // OPTIMIZED: Reuse state array across all 63 iterations instead of recreating each time
    // This saves 62 state initializations and reduces register pressure
    uint64_t state[STATE_SIZE];
    
    // First hash: hash(hash, nonce) with FixedLength domain
    // OPTIMIZATION: Inline the state setup instead of calling tip5_sponge_init + loop
    // Load hash into state[0..4]
    state[0] = hash.values[0];
    state[1] = hash.values[1];
    state[2] = hash.values[2];
    state[3] = hash.values[3];
    state[4] = hash.values[4];
    
    // Load nonce into state[5..9]
    state[5] = nonce.values[0];
    state[6] = nonce.values[1];
    state[7] = nonce.values[2];
    state[8] = nonce.values[3];
    state[9] = nonce.values[4];
    
    // Capacity region for FixedLength domain
    constexpr uint64_t FIXED_LEN_VAL = static_cast<uint64_t>(Domain::FixedLength);
    state[10] = FIXED_LEN_VAL;
    state[11] = FIXED_LEN_VAL;
    state[12] = FIXED_LEN_VAL;
    state[13] = FIXED_LEN_VAL;
    state[14] = FIXED_LEN_VAL;
    state[15] = FIXED_LEN_VAL;
    
    tip5_permutation(state);
    
    // Remaining 62 iterations: hash(state, zeros) with FixedLength domain
    // OPTIMIZATION: Use specialized permutation that exploits known right side:
    //   state[5..9]=0, state[10..15]=FIXED_LEN_VAL(=1)
    // This saves 44 field_mul_ptx per iteration by skipping x^7 for 11 elements
    // Precompute x^7(FIXED_LEN_VAL) once for all 62 iterations
    uint64_t x7_fixed = x7_computer_pipelined(FIXED_LEN_VAL);
    
    #pragma unroll 2
    for (uint32_t i = 1; i < NUM_INDEX_REPETITIONS; ++i) {
        // Specialized permutation handles the known right-side values internally
        // — no need to write state[5..15] here (saves 11 writes per iteration)
        tip5_permutation_fixed_right_zero(state, x7_fixed);
    }

    index_a = state[0] & MERKLE_INDEX_MASK;
    index_b = state[1] & MERKLE_INDEX_MASK;
}

// Mining pool configuration
static int g_miner_id = 0;           // Unique miner ID in pool (0-65535)
static int g_total_miners = 1024;    // Total miners in pool (configurable)

// Generate cryptographically strong random start with multiple entropy sources
// Uses GPU UUID for hardware-unique identification instead of worker_id
uint64_t generate_secure_random_start(const std::string& puzzle_id, int gpu_id, const std::string& worker_id, const std::string& gpu_uuid) {
    // Entropy source 1: High-resolution timestamp (nanoseconds)
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    
    // Entropy source 2: Process ID (unique per miner instance)
    uint64_t process_id = static_cast<uint64_t>(getpid());
    
    // Entropy source 3: Thread ID (additional randomness)
    uint64_t thread_id = static_cast<uint64_t>(pthread_self());
    
    // Entropy source 4: Random device (hardware RNG if available)
    std::random_device rd;
    uint64_t hw_random = (static_cast<uint64_t>(rd()) << 32) | static_cast<uint64_t>(rd());
    
    // Entropy source 5: Memory address (ASLR provides randomness)
    uint64_t stack_addr = reinterpret_cast<uint64_t>(&now);
    
    // Combine all entropy sources with XOR and mixing
    uint64_t random_seed = timestamp_ns;
    random_seed ^= process_id * 0x9e3779b97f4a7c15ULL;  // Golden ratio
    random_seed ^= thread_id * 0x85ebca6b;
    random_seed ^= hw_random;
    random_seed ^= stack_addr;
    
    // Hash proposal_id (puzzle_id) to ensure different proposals get different nonce ranges
    // This ensures the same GPU mines different nonce ranges for different jobs/proposals
    uint64_t proposal_hash = 0;
    for (size_t i = 0; i < puzzle_id.length(); ++i) {
        proposal_hash = proposal_hash * 31 + (uint64_t)puzzle_id[i];
    }
    
    // NEW: Use GPU UUID AND proposal_id together for hardware-unique + job-unique identification
    // GPU UUID ensures different GPUs get different ranges
    // Proposal ID ensures the same GPU gets different ranges for different jobs/proposals
    uint64_t gpu_proposal_combined = 0;
    if (!gpu_uuid.empty()) {
        uint64_t uuid_hash = 0;
        for (size_t i = 0; i < gpu_uuid.length(); ++i) {
            uuid_hash = uuid_hash * 31 + (uint64_t)gpu_uuid[i];
        }
        // Combine GPU UUID and proposal_id hash together
        gpu_proposal_combined = uuid_hash ^ (proposal_hash * 0x9e3779b97f4a7c15ULL);
        random_seed ^= gpu_proposal_combined;  // GPU UUID + proposal_id provides uniqueness per GPU per job
    } else if (!worker_id.empty()) {
        // Fallback 1: Use worker_id from server if GPU UUID not available
        uint64_t worker_hash = 0;
        for (size_t i = 0; i < worker_id.length(); ++i) {
            worker_hash = worker_hash * 31 + (uint64_t)worker_id[i];
        }
        // Combine worker_id and proposal_id hash together
        gpu_proposal_combined = worker_hash ^ (proposal_hash * 0x9e3779b97f4a7c15ULL);
        random_seed ^= gpu_proposal_combined;
    } else {
        // Fallback 2: Use miner_id + gpu_id if neither UUID nor worker_id available
        uint64_t fallback_hash = (uint64_t)g_miner_id ^ ((uint64_t)gpu_id * 0x9e3779b97f4a7c15ULL);
        gpu_proposal_combined = fallback_hash ^ (proposal_hash * 0x9e3779b97f4a7c15ULL);
        random_seed ^= gpu_proposal_combined;
    }
    

    // Apply SplitMix64 hash mixing for better distribution
    random_seed ^= (random_seed >> 30);
    random_seed *= 0xbf58476d1ce4e5b9ULL;
    random_seed ^= (random_seed >> 27);
    random_seed *= 0x94d049bb133111ebULL;
    random_seed ^= (random_seed >> 31);
    
    return random_seed;
}

// GPU Range Calculation with Mining Pool Support
// Hierarchical partitioning: First by miner ID, then by GPU ID
// Supports miners with hundreds of GPUs and large pools

GpuNonceRange calculate_gpu_range(int gpu_id, int total_gpus) {
    // Get partition ID from environment (for nonce space partitioning)
    // Check both PARTITION_ID (new) and MINER_ID (legacy) for backward compatibility
    const char* env_partition_id = std::getenv("PARTITION_ID");
    if (env_partition_id) {
        int partition_val = std::atoi(env_partition_id);
        if (partition_val >= 0 && partition_val < 65536) {
            g_miner_id = partition_val;
        }
    } else {
        // Legacy support for MINER_ID env var
        const char* env_miner_id = std::getenv("MINER_ID");
        if (env_miner_id) {
            int miner_val = std::atoi(env_miner_id);
            if (miner_val >= 0 && miner_val < 65536) {
                g_miner_id = miner_val;
            }
        }
    }
    
    // Get total miners in pool
    const char* env_total_miners = std::getenv("POOL_SIZE");
    if (env_total_miners) {
        int pool_val = std::atoi(env_total_miners);
        if (pool_val > 0 && pool_val <= 65536) {  // Support up to 65K miners
            g_total_miners = pool_val;
        }
    }
    
    // Get GPUs per miner (support up to 36 GPUs per machine)
    // NOTE: This env var is OPTIONAL - if not set, uses the total_gpus parameter
    //       which should be the actual detected GPU count from the caller
    const char* env_gpus = std::getenv("NUM_GPUS");
    if (env_gpus) {
        int env_val = std::atoi(env_gpus);
        if (env_val > 0 && env_val <= 36) {  // Hardware limit: 36 GPUs per machine
            total_gpus = env_val;
        }
    }
    
    GpuNonceRange range;
    range.miner_id = g_miner_id;
    range.total_miners = g_total_miners;
    range.gpu_id = gpu_id;
    range.total_gpus = total_gpus;
    
    // Two-level partitioning:
    // Level 1: Divide by total miners in pool
    // Level 2: Divide miner's range by GPUs per miner
    
    const uint64_t usable_bits = 62;
    const uint64_t total_space = (1ULL << usable_bits);
    
    // Each miner gets: total_space / total_miners
    uint64_t miner_range_size = total_space / g_total_miners;
    uint64_t miner_range_start = miner_range_size * g_miner_id;
    
    // Each GPU within miner gets: miner_range / gpus_per_miner
    range.range_size = miner_range_size / total_gpus;
    range.range_start = miner_range_start + (range.range_size * gpu_id);
    
    return range;
}

// ===== SOLUTION VERIFICATION =====

bool verify_pow_solution(const Pow& pow,
                         const Digest& hash,
                         const Digest& target,
                         const Digest& commitment) {
    // Compute indices from nonce
    auto [index_a, index_b] = Pow::indices(hash, pow.nonce);
    
    // Compute expected leafs from commitment
    Digest leaf_a = Pow::compute_leaf_from_commitment_host(commitment, index_a, MERKLE_NUM_LEAFS);
    Digest leaf_b = Pow::compute_leaf_from_commitment_host(commitment, index_b, MERKLE_NUM_LEAFS);
    
    // Verify Merkle paths
    bool path_a_valid = Pow::verify_merkle_path_host(pow.root, index_a, pow.path_a, leaf_a);
    bool path_b_valid = Pow::verify_merkle_path_host(pow.root, index_b, pow.path_b, leaf_b);
    
    if (!path_a_valid || !path_b_valid) {
        return false;
    }
    
    // Compute final hash and check against target
    // (Full verification would include MAST hash computation)
    
    return true;
}

// ===== AUTH PATHS CONVERSION =====

PowMastPaths convertToPowMastPaths(const AuthPaths& auth_paths) {
    PowMastPaths result;
    
    // Convert pow paths
    for (size_t i = 0; i < std::min(auth_paths.pow.size(), size_t(3)); ++i) {
        result.pow[i] = hex_to_digest(auth_paths.pow[i]);
    }
    
    // Convert header paths
    for (size_t i = 0; i < std::min(auth_paths.header.size(), size_t(2)); ++i) {
        result.header[i] = hex_to_digest(auth_paths.header[i]);
    }
    
    // Convert kernel paths
    for (size_t i = 0; i < std::min(auth_paths.kernel.size(), size_t(1)); ++i) {
        result.kernel[i] = hex_to_digest(auth_paths.kernel[i]);
    }
    
    return result;
}

#include "mining.cuh"
#include "rpc_client.cuh"
#include "connection_multiplexer.cuh"
#include "stratum_client.cuh"
#include "common.cuh"
#include "kernels.cuh"

// ========== Inlined from gpu_resources.cu (start) ==========
// ===== GLOBAL VARIABLES =====
std::vector<GpuResources*> g_all_gpu_resources;
std::mutex g_all_gpu_resources_mutex;


std::string get_gpu_uuid(int device_id) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    
    if (err != cudaSuccess) {
        return "unknown";
    }
    
    // Convert UUID bytes to hex string
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    
    for (int i = 0; i < 16; ++i) {
        oss << std::setw(2) << static_cast<int>(static_cast<unsigned char>(prop.uuid.bytes[i]));
        if (i == 3 || i == 5 || i == 7 || i == 9) {
            oss << "-";
        }
    }
    
    return oss.str();
}

void broadcast_stop_to_all_gpus(int source_gpu_id, const char* reason) {
    std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
    
    int stopped_count = 0;
    for (GpuResources* gpu : g_all_gpu_resources) {
        if (gpu) {
            try {
                bool was_stopped = gpu->gpu_stop_flag.load();
                if (!was_stopped) {
                    gpu->gpu_stop_flag = true;
                    stopped_count++;
                    if (gpu->event_handler) {
                        gpu->event_handler->postEvent(EventType::STOP_MINING);
                    }
                }
            } catch (...) {
                continue;
            }
        }
    }
    
    if (stopped_count > 0) {
        LOG_DEBUG("[GPU " << source_gpu_id << "] " << reason
                  << " - Broadcasting stop to " << stopped_count << " GPU(s)");
    }
}

GpuResources* find_gpu_resources(int gpu_id) {
    std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
    
    for (GpuResources* gpu : g_all_gpu_resources) {
        if (gpu && gpu->gpu_id == gpu_id) {
            return gpu;
        }
    }
    
    return nullptr;
}

double get_total_hashrate() {
    std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
    
    double total = 0.0;
    for (GpuResources* gpu : g_all_gpu_resources) {
        if (gpu && gpu->is_mining()) {
            total += gpu->hash_tracker.get_average();
        }
    }
    
    return total;
}

uint64_t get_total_solutions_found() {
    std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
    
    uint64_t total = 0;
    for (GpuResources* gpu : g_all_gpu_resources) {
        if (gpu) {
            total += gpu->solutions_found.load();
        }
    }
    
    return total;
}

void cleanup_gpu_memory() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        return;
    }
    
    for (int gpu_id = 0; gpu_id < device_count; ++gpu_id) {
        cudaSetDevice(gpu_id);
        cudaDeviceSynchronize();
        cudaGetLastError();
        cudaDeviceReset();
    }
    
    {
        std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
        g_all_gpu_resources.clear();
    }
}
// ========== Inlined from gpu_resources.cu (end) ==========

// ============================================================================
// GpuWorker Implementation
// ============================================================================

GpuWorker::GpuWorker(int gpu_id, GpuResources* resources)
    : gpu_id(gpu_id)
    , gpu_resources(resources)
    , worker_handle(nullptr) {
    mining_mode = gpu_resources ? gpu_resources->mining_mode : MiningMode::Solo;
}

GpuWorker::~GpuWorker() {
    stop();
}

void GpuWorker::start() {
    if (!gpu_resources->event_handler) {
        gpu_resources->event_handler = std::make_unique<EventHandler>();
    }
    // GpuWorker doesn't manage clients directly - ConnectionMultiplexer does
    // Just register with the multiplexer
    ConnectionMultiplexer& mux = ConnectionMultiplexer::getInstance();
    worker_handle = mux.registerWorker(gpu_id, gpu_resources);
    
    if (!worker_handle) {
        std::cerr << "[GPU " << gpu_id << "] Failed to register with multiplexer" << std::endl;
        return;
    }
    
    // The multiplexer will handle connection and job fetching
    // GpuWorker just needs to wait for events
    worker_running = true;
    
    // Start mining thread
    mining_thread = std::thread(&GpuWorker::miningLoop, this);
}

// Legacy UnifiedMiningController code removed - using GpuWorker architecture now.

void GpuWorker::stop() {
    worker_running = false;
    gpu_resources->gpu_stop_flag = true;
    
    if (gpu_resources->event_handler) {
        gpu_resources->event_handler->shutdown();
    }

    if (mining_thread.joinable()) {
        mining_thread.join();
    }
}

bool GpuWorker::initializeCuda() {
    cudaError_t err = cudaSetDevice(gpu_id);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaSetDevice", err);
        return false;
    }
    
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, gpu_id);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaGetDeviceProperties", err);
        return false;
    }
    
    gpu_resources->gpu_name = prop.name;
    gpu_resources->gpu_vram_total = prop.totalGlobalMem;
    gpu_resources->gpu_uuid = get_gpu_uuid(gpu_id);
    
    // Batch size: use global override if set, otherwise auto-detect for this GPU
    const uint64_t min_batch = 1;
    gpu_resources->optimal_max_nonces = (g_batch_size > 0)
        ? std::max(g_batch_size, min_batch)
        : get_optimal_batch_size(gpu_id);
    
    return true;
}

void GpuWorker::miningLoop() {
    while (worker_running && !stop_mining && !gpu_resources->gpu_stop_flag) {
        // Wait for mining events from EventHandler
        MiningEvent event;
        bool has_event = gpu_resources->event_handler->waitForEvent(
            event, std::chrono::milliseconds(100));
        
        if (has_event) {
            switch (event.type) {
                case EventType::NEW_PUZZLE:
                    handleNewPuzzle(event);
                    break;
                case EventType::STOP_MINING:
                    worker_running = false;
                    break;
                default:
                    break;
            }
        }
    }
}

void GpuWorker::handleNewPuzzle(const MiningEvent& event) {
    if (event.data.empty()) {
        std::cout << "[GPU " << gpu_id << "] " << Color::RED << "Empty puzzle data" << Color::RESET << std::endl;
        return;
    }
    
    PowPuzzle puzzle = parsePowPuzzle(event.data);
    if (!puzzle.is_valid()) {
        std::cout << "[GPU " << gpu_id << "] " << Color::RED << "Invalid puzzle from parsePowPuzzle" << Color::RESET << std::endl;
        return;
    }
    
    // Parse and store the template for this proposal
    json template_response = json::parse(event.data);
    json template_obj;
    if (template_response.contains("result") && template_response["result"].contains("template")) {
        template_obj = template_response["result"]["template"];
    } else if (template_response.contains("template")) {
        template_obj = template_response["template"];
    }
    
    Digest prev_block = hex_to_digest(puzzle.prev_block);
    PowMastPaths mast_paths = convertToPowMastPaths(puzzle.auth_paths);
    Digest commitment = mast_paths.commit();
    Digest original_target = hex_to_digest(puzzle.threshold);
    Digest effective_target = g_test_mode ? make_target_easier(original_target, 100000) : original_target;
    
    // Check for duplicate (solo mode only)
    {
        std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
        bool is_duplicate = (puzzle.id == gpu_resources->current_proposal_id);
        if (is_duplicate && mining_mode == MiningMode::Solo) {
            return;
        }
    }
    
    // Check if we can reuse existing buffer (XNT uses commitment only)
    bool need_preprocess = true;
    bool have_existing_buffer = false;
    {
        std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
        if (gpu_resources->buffer && gpu_resources->buffer->is_valid()) {
            have_existing_buffer = true;
            bool commitment_match = true;
            for (int i = 0; i < DIGEST_LEN; ++i) {
                if (gpu_resources->cached_commitment.values[i] != commitment.values[i]) {
                    commitment_match = false;
                    break;
                }
            }
            
            bool prev_block_match = true;
            if (puzzle.consensus_rule_set != CONSENSUS_XNT) {
                for (int i = 0; i < DIGEST_LEN; ++i) {
                    if (gpu_resources->cached_prev_block.values[i] != prev_block.values[i]) {
                        prev_block_match = false;
                        break;
                    }
                }
            }

            if (commitment_match && prev_block_match) {
                need_preprocess = false;
                gpu_resources->buffer->mast_paths = mast_paths;
                gpu_resources->buffer->prev_block_digest = prev_block;
                gpu_resources->cached_mast_paths = mast_paths;
                gpu_resources->cached_commitment = commitment;
                std::cout << "[GPU " << gpu_id << "] " << Color::GREEN 
                          << "Reusing cached buffer" << Color::RESET << std::endl;
            }
        }
    }
    
    if (need_preprocess) {
        if (have_existing_buffer) {
            // ---- P2.7: Async preprocessing ----
            // We have a valid old buffer — launch preprocessing in background and
            // continue mining the OLD puzzle. The mining loop will swap buffers when done.
            
            // Wait for any prior async preprocessing to finish first
            if (gpu_resources->async_preprocess_thread.joinable()) {
                gpu_resources->async_preprocess_thread.join();
            }
            gpu_resources->async_preprocess_done = false;
            gpu_resources->async_preprocess_running = true;
            
            // Store new puzzle metadata (will be applied on buffer swap)
            {
                std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
                gpu_resources->pending_proposal_id = puzzle.id;
                gpu_resources->pending_template = template_obj;
                gpu_resources->pending_target = effective_target;
                gpu_resources->pending_real_target = original_target;
                gpu_resources->pending_prev_block = prev_block;
                gpu_resources->pending_mast_paths = mast_paths;
                gpu_resources->pending_commitment = commitment;
            }
            
            // Launch background preprocessing thread
            int bg_gpu_id = gpu_id;
            int bg_consensus = puzzle.consensus_rule_set;
            gpu_resources->async_preprocess_thread = std::thread(
                [this, bg_gpu_id, mast_paths, prev_block, bg_consensus]() {
                    cudaError_t err = cudaSetDevice(bg_gpu_id);
                    if (err != cudaSuccess) {
                        LOG_ERROR("cudaSetDevice in async preprocess", err);
                        gpu_resources->async_preprocess_running = false;
                        return;
                    }
                    
                    auto start = std::chrono::steady_clock::now();
                    auto new_buffer = Pow::preprocess(mast_paths, prev_block, bg_consensus, nullptr);
                    auto end = std::chrono::steady_clock::now();
                    double elapsed_s = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
                    
                    if (new_buffer.is_valid()) {
                        gpu_resources->pending_buffer = std::make_unique<GuesserBuffer>(std::move(new_buffer));
                        gpu_resources->async_preprocess_done = true;
                        std::cout << "[GPU " << bg_gpu_id << "] " << Color::GREEN 
                                  << "Async preprocessing complete" << Color::RESET 
                                  << " (" << std::fixed << std::setprecision(2) << elapsed_s << "s, mining continued)" << std::endl;
                    } else {
                        std::cout << "[GPU " << bg_gpu_id << "] " << Color::RED 
                                  << "Async preprocessing failed" << Color::RESET << std::endl;
                    }
                    gpu_resources->async_preprocess_running = false;
                }
            );
            
            std::cout << "[GPU " << gpu_id << "] " << Color::YELLOW 
                      << "Async preprocessing started (mining continues with old buffer)" << Color::RESET << std::endl;
            
            // DON'T update current_* metadata yet — keep mining old puzzle.
            // The mining loop will apply pending metadata on buffer swap.
            // But we DO need to resume the mining loop, so fall through to it.
        } else {
            // No existing buffer (first puzzle) — must block on preprocessing
            {
                std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
                gpu_resources->current_proposal_id = puzzle.id;
                gpu_resources->current_template = template_obj;
                gpu_resources->current_target = effective_target;
                gpu_resources->current_real_target = original_target;
            }
            
            if (g_test_mode) {
                std::cout << "[GPU " << gpu_id << "] " << Color::YELLOW 
                          << "TEST MODE: Target made 100000x easier" << Color::RESET << std::endl;
            }
            
            resetNonceCounter(gpu_resources, puzzle.id);
            
            auto preprocess_start = std::chrono::steady_clock::now();
            
            if (!preprocessPuzzle(puzzle, gpu_resources)) {
                std::cout << "[GPU " << gpu_id << "] " << Color::RED << "Preprocessing failed" << Color::RESET << std::endl;
                return;
            }
            
            auto preprocess_end = std::chrono::steady_clock::now();
            auto preprocess_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - preprocess_start).count();
            double preprocess_seconds = preprocess_duration / 1000.0;
            
            std::cout << "[GPU " << gpu_id << "] " << Color::GREEN << "Finished preprocessing" << Color::RESET 
                      << " (" << std::fixed << std::setprecision(2) << preprocess_seconds << "s)" << std::endl;
            
            {
                std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
                gpu_resources->cached_prev_block = prev_block;
                gpu_resources->cached_mast_paths = mast_paths;
                gpu_resources->cached_commitment = commitment;
            }
        }
    } else {
        // Buffer reused — update current metadata to new puzzle
        {
            std::lock_guard<std::mutex> lock(gpu_resources->state_mutex);
            gpu_resources->current_proposal_id = puzzle.id;
            gpu_resources->current_template = template_obj;
            gpu_resources->current_target = effective_target;
            gpu_resources->current_real_target = original_target;
        }
        resetNonceCounter(gpu_resources, puzzle.id);
    }
    
    gpu_resources->update_job_received();
    gpu_resources->set_paused(false);
    
    std::string short_id = puzzle.id.length() > 20 ? puzzle.id.substr(0, 12) + "..." + puzzle.id.substr(puzzle.id.length() - 8) : puzzle.id;
    std::cout << "[GPU " << gpu_id << "] " << Color::CYAN << Color::BOLD 
              << "New job received" << Color::RESET 
              << " | Template ID: " << Color::CYAN << short_id << Color::RESET << std::endl;
    
    // Run the continuous mining loop
    continuousMiningLoop(gpu_resources, this);
}

std::future<bool> GpuWorker::submitSolution(
    const std::string& proposal_id,
    const Pow& pow_solution,
    const Digest& solution_hash,
    const json& template_obj) {
    
    ConnectionMultiplexer& mux = ConnectionMultiplexer::getInstance();
    return mux.submitSolution(gpu_id, proposal_id, pow_solution, solution_hash, template_obj);
}

// ============================================================================
// MultiGpuManager Implementation
// ============================================================================

// Legacy UnifiedMiningController methods removed.

// ===== MULTI-GPU MANAGER IMPLEMENTATION =====

MultiGpuManager::MultiGpuManager(
    const std::string& endpoint,
    int specific_gpu,
    MiningMode mode,
    const std::string& stratum_pass)
    : endpoint(endpoint)
    , single_gpu_id(specific_gpu)
    , mining_mode(mode)
    , stratum_password(stratum_pass) {
}

MultiGpuManager::~MultiGpuManager() {
    stopAll();
}

bool MultiGpuManager::detectAndInitGpus() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return false;
    }
    
    // Determine which GPUs to use
    if (single_gpu_id >= 0) {
        // Single GPU mode
        if (single_gpu_id >= device_count) {
            std::cerr << "GPU " << single_gpu_id << " not found (max: " << device_count - 1 << ")" << std::endl;
            return false;
        }
        gpu_ids.push_back(single_gpu_id);
    } else {
        // All GPUs mode
        for (int i = 0; i < device_count; ++i) {
            gpu_ids.push_back(i);
        }
    }
    
    for (int gpu_id : gpu_ids) {
        if (!initializeGpu(gpu_id)) {
            std::cerr << "Failed to initialize GPU " << gpu_id << std::endl;
            continue;
        }
    }
    
    g_total_gpu_count = static_cast<int>(gpu_resources.size());
    
    {
        std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
        for (auto& res : gpu_resources) {
            g_all_gpu_resources.push_back(res.get());
        }
    }
    
    return !gpu_resources.empty();
}

bool MultiGpuManager::initializeGpu(int device_id) {
    auto gpu_res = std::make_unique<GpuResources>(device_id);
    
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return false;
    }
    
    size_t vram_gb = prop.totalGlobalMem / (1024ULL * 1024ULL * 1024ULL);
    if (vram_gb < 6) {
        LOG_DEBUG("GPU " << device_id << " has insufficient VRAM (" << vram_gb << " GB)");
        return false;
    }
    
    gpu_res->gpu_name = prop.name;
    gpu_res->gpu_vram_total = prop.totalGlobalMem;
    gpu_res->gpu_uuid = get_gpu_uuid(device_id);
    
    // Batch size: use global override if set, otherwise auto-detect for this GPU
    const uint64_t min_batch = 1;
    gpu_res->optimal_max_nonces = (g_batch_size > 0)
        ? std::max(g_batch_size, min_batch)
        : get_optimal_batch_size(device_id);
    gpu_res->mining_mode = mining_mode;
    
    // Create GpuWorker (uses shared connection through multiplexer)
    auto worker = std::make_unique<GpuWorker>(device_id, gpu_res.get());
    
    gpu_resources.push_back(std::move(gpu_res));
    workers.push_back(std::move(worker));
    
    std::cout << "Initialized GPU " << device_id << ": " << prop.name 
              << " (" << vram_gb << " GB)" << std::endl;
    
    return true;
}

void MultiGpuManager::startAll() {
    if (gpu_resources.empty()) {
        std::cerr << "No GPUs initialized" << std::endl;
        return;
    }
    
    // Initialize the connection multiplexer first
    ConnectionMultiplexer& mux = ConnectionMultiplexer::getInstance();
    if (!mux.initialize(endpoint, g_miner_wallet_address, stratum_password)) {
        std::cerr << "Failed to initialize connection multiplexer" << std::endl;
        return;
    }
    
    // Start GPU worker threads
    for (size_t i = 0; i < workers.size(); ++i) {
        gpu_threads.emplace_back(&MultiGpuManager::gpuWorkerThread, this, i);
        if (i < workers.size() - 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
    
    for (auto& thread : gpu_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void MultiGpuManager::stopAll() {
    stop_mining = true;
    
    // Stop workers first
    for (auto& worker : workers) {
        worker->stop();
    }
    
    // Wait for threads
    for (auto& thread : gpu_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Shutdown multiplexer after workers are done
    ConnectionMultiplexer::destroyInstance();
    
    for (auto& res : gpu_resources) {
        // Join async preprocessing thread before cleaning up buffers
        if (res->async_preprocess_thread.joinable()) {
            res->async_preprocess_thread.join();
        }
        if (res->pending_buffer) {
            res->pending_buffer->cleanup();
            res->pending_buffer.reset();
        }
        if (res->buffer) {
            res->buffer->cleanup();
            res->buffer.reset();
        }
    }
    
    {
        std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
        g_all_gpu_resources.clear();
    }
}

GpuResources* MultiGpuManager::getGpuResources(size_t index) {
    if (index >= gpu_resources.size()) return nullptr;
    return gpu_resources[index].get();
}

void MultiGpuManager::gpuWorkerThread(size_t index) {
    if (index >= workers.size()) return;
    workers[index]->start();
    
    // Wait for the worker to finish (blocks until mining stops)
    while (!stop_mining && workers[index]->isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void startUnifiedMining(
    const std::string& endpoint,
    int specific_gpu,
    MiningMode mode,
    const std::string& stratum_pass) {
    
    MultiGpuManager manager(endpoint, specific_gpu, mode, stratum_pass);
    
    if (!manager.detectAndInitGpus()) {
        std::cerr << "Failed to initialize GPUs" << std::endl;
        return;
    }
    
    try {
        manager.startAll();
    } catch (const std::exception& e) {
        std::cerr << "Mining error: " << e.what() << std::endl;
    }
    
    cleanup_gpu_memory();
}

// ============================================================================
// Continuous Mining Loop - Double-Buffered Async Version
// ============================================================================
// TRUE PIPELINING: We use 2 independent CUDA streams. Each stream has its own
// output buffers. The pipeline works as follows:
//
// Time:    |----Batch 0----|----Batch 2----|----Batch 4----|
// Stream A: [==KERNEL 0===][==KERNEL 2===][==KERNEL 4===]
// Stream B:      [==KERNEL 1===][==KERNEL 3===][==KERNEL 5===]
// CPU:      [Launch0][Chk-1][Launch2][Chk0][Launch4][Chk2]...
//
// Key: We NEVER block. After launching batch N, we check if batch N-2 (same
// stream, previous kernel) is done. If not done, we continue - the kernel
// will complete eventually and we'll catch it next time around.

bool continuousMiningLoop(GpuResources* gpu_res, GpuWorker* worker) {
    if (!gpu_res || !worker) return false;
    
    auto last_status_time = std::chrono::steady_clock::now();
    const auto STATUS_UPDATE_INTERVAL = std::chrono::seconds(5);
    
    // Default: sync mode. Set XNT_ASYNC_MINING=1 to use async double-buffered mode.
    bool use_async = false;
    if (const char* env = std::getenv("XNT_ASYNC_MINING"); env && env[0] == '1') {
        use_async = true;
    }
    if (!use_async) {
        return continuousMiningLoopSync(gpu_res, worker);
    }
    
    // Initialize double-buffered miner (async x2 mode)
    if (!gpu_res->async_miner) {
        gpu_res->async_miner = std::make_unique<DoubleBufferedMiner>();
    }
    
    DoubleBufferedMiner& miner = *gpu_res->async_miner;
    if (!miner.initialize()) {
        std::cerr << "[GPU " << gpu_res->gpu_id << "] Failed to initialize async miner, falling back to sync" << std::endl;
        return continuousMiningLoopSync(gpu_res, worker);
    }
    
    // Track batch info for each slot
    struct BatchInfo {
        std::string proposal_id;
        json template_obj;
        uint64_t start_nonce;
        uint64_t batch_size;
        std::chrono::high_resolution_clock::time_point start_time;
        bool pending_result;  // True if kernel launched and result not yet processed
    };
    BatchInfo batch_info[2] = {};
    
    // Pipeline state
    int active_slot = 0;  // Which slot to use for NEXT launch
    int batches_in_flight = 0;
    
    while (!stop_mining && !gpu_res->gpu_stop_flag) {
        // Check for new events
        if (gpu_res->event_handler && gpu_res->event_handler->hasEvents()) {
            // Drain pipeline before breaking
            for (int i = 0; i < 2; i++) {
                if (batch_info[i].pending_result) {
                    cudaStreamSynchronize(miner.slots[i].stream);
                }
            }
            break;
        }
        
        // ---- P2.7: Check if async preprocessing completed → swap buffers ----
        if (gpu_res->async_preprocess_done.load(std::memory_order_acquire)) {
            // Drain in-flight mining batches before swapping (they reference old buffer)
            for (int i = 0; i < 2; i++) {
                if (batch_info[i].pending_result) {
                    cudaStreamSynchronize(miner.slots[i].stream);
                    // Count nonces from the batch we're draining
                    gpu_res->total_nonces_tested.fetch_add(batch_info[i].batch_size);
                    batch_info[i].pending_result = false;
                    miner.slots[i].kernel_launched = false;
                }
            }
            batches_in_flight = 0;
            
            // Swap buffer and apply pending puzzle metadata
            gpu_res->buffer = std::move(gpu_res->pending_buffer);
            {
                std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
                gpu_res->current_proposal_id = gpu_res->pending_proposal_id;
                gpu_res->current_template = gpu_res->pending_template;
                gpu_res->current_target = gpu_res->pending_target;
                gpu_res->current_real_target = gpu_res->pending_real_target;
                gpu_res->cached_prev_block = gpu_res->pending_prev_block;
                gpu_res->cached_mast_paths = gpu_res->pending_mast_paths;
                gpu_res->cached_commitment = gpu_res->pending_commitment;
            }
            
            // Reset nonce counter for new puzzle and force re-init of GPU constants
            resetNonceCounter(gpu_res, gpu_res->pending_proposal_id);
            gpu_res->buffer->gpu_range_initialized = false;
            
            gpu_res->async_preprocess_done.store(false, std::memory_order_release);
            
            std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::GREEN << Color::BOLD
                      << "Buffer swapped — now mining new template" << Color::RESET << std::endl;
        }
        
        if (gpu_res->gpu_pause_flag) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        if (!gpu_res->buffer || !gpu_res->buffer->is_valid()) {
            gpu_res->set_paused(true);
            continue;
        }
        
        std::string proposal_id;
        json template_for_this_batch;
        {
            std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
            proposal_id = gpu_res->current_proposal_id;
            template_for_this_batch = gpu_res->current_template;
        }
        
        if (proposal_id.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        // Check template staleness
        if (!template_for_this_batch.is_null() && !template_for_this_batch.empty()) {
            auto& multiplexer = ConnectionMultiplexer::getInstance();
            if (multiplexer.isTemplateStale(template_for_this_batch)) {
                // Wait for in-flight batches then pause
                for (int i = 0; i < 2; i++) {
                    if (batch_info[i].pending_result) {
                        cudaStreamSynchronize(miner.slots[i].stream);
                        batch_info[i].pending_result = false;
                    }
                }
                batches_in_flight = 0;
                
                std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::YELLOW
                          << "Template stale, pausing..." << Color::RESET << std::endl;
                gpu_res->set_paused(true);
                
                while (!stop_mining && !gpu_res->gpu_stop_flag && gpu_res->gpu_pause_flag) {
                    if (gpu_res->event_handler && gpu_res->event_handler->hasEvents()) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                
                if (!stop_mining && !gpu_res->gpu_stop_flag) {
                    gpu_res->set_paused(false);
                }
                continue;
            }
        }
        
        // =====================================================================
        // STEP 1: Check if CURRENT slot's previous batch is complete (non-blocking)
        // =====================================================================
        AsyncMiningSlot* pslot = &miner.slots[active_slot];
        BatchInfo& current_batch_info = batch_info[active_slot];
        
        if (current_batch_info.pending_result) {
            // Check if this slot's kernel is done (NON-BLOCKING)
            cudaError_t status = cudaEventQuery(pslot->completion_event);
            
            if (status == cudaSuccess) {
                // Kernel completed! Process result
                auto now = std::chrono::high_resolution_clock::now();
                auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - current_batch_info.start_time).count();
                
                gpu_res->total_nonces_tested.fetch_add(current_batch_info.batch_size);
                batches_in_flight--;
                
                if (duration_ms > 0) {
                    double hashrate_ms = static_cast<double>(current_batch_info.batch_size) / duration_ms;
                    gpu_res->hash_tracker.add(hashrate_ms);
                }
                
                // Check for solution (pinned memory was async copied)
                int solution_found = *pslot->h_solution_found_pinned;
                
                if (solution_found) {
                    gpu_res->solutions_found++;
                    
                    // Retrieve solution data
                    MiningSolution solution;
                    cudaMemcpy(&solution.pow.nonce, pslot->d_solution_nonce_digest, 
                               sizeof(Digest), cudaMemcpyDeviceToHost);
                    cudaMemcpy(solution.pow.path_a, pslot->d_solution_path_a,
                               MERKLE_TREE_HEIGHT_ * sizeof(Digest), cudaMemcpyDeviceToHost);
                    cudaMemcpy(solution.pow.path_b, pslot->d_solution_path_b,
                               MERKLE_TREE_HEIGHT_ * sizeof(Digest), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&solution.kernel_final_hash, pslot->d_solution_final_hash,
                               sizeof(Digest), cudaMemcpyDeviceToHost);
                    solution.pow.root = gpu_res->buffer->merkle_root;
                    
                    // Check if proposal still current
                    std::string current_proposal_id;
                    {
                        std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
                        current_proposal_id = gpu_res->current_proposal_id;
                    }
                    
                    if (current_proposal_id != current_batch_info.proposal_id) {
                        std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::YELLOW 
                                  << "Solution discarded - template changed" << Color::RESET << std::endl;
                    } else {
                        const char* target = gpu_res->is_stratum_mode() ? "pool" : "node";
                        std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::YELLOW << Color::BOLD
                                  << "*** SOLUTION FOUND! ***" << Color::RESET 
                                  << " Submitting to " << target << "..." << std::endl;
                        
                        PowMastPaths mast_paths = gpu_res->buffer->mast_paths;
                        Digest solution_hash = mast_paths.fast_mast_hash(solution.pow);
                        
                        // Submit asynchronously
                        auto future = worker->submitSolution(
                            current_batch_info.proposal_id,
                            solution.pow,
                            solution_hash,
                            current_batch_info.template_obj);
                        
                        // Non-blocking check with short timeout
                        if (future.wait_for(std::chrono::seconds(30)) == std::future_status::ready) {
                            bool accepted = future.get();
                            if (accepted) {
                                gpu_res->solutions_accepted++;
                                const char* msg = gpu_res->is_stratum_mode() ? "SHARE ACCEPTED" : "BLOCK ACCEPTED";
                                std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::GREEN << Color::BOLD 
                                          << "*** " << msg << "! ***" << Color::RESET << std::endl;
                            } else {
                                gpu_res->solutions_rejected++;
                            }
                        } else {
                            gpu_res->solutions_rejected++;
                        }
                    }
                }
                
                current_batch_info.pending_result = false;
                pslot->kernel_launched = false;
                
            } else if (status == cudaErrorNotReady) {
                // Kernel still running - switch to other slot and continue
                // DON'T BLOCK! Just use the other slot
                active_slot = 1 - active_slot;
                
                // If other slot also busy, we need to wait for one
                if (batch_info[active_slot].pending_result) {
                    cudaError_t other_status = cudaEventQuery(miner.slots[active_slot].completion_event);
                    if (other_status == cudaErrorNotReady) {
                        // Both slots busy - wait for current one (this is expected with 2 slots)
                        cudaStreamSynchronize(pslot->stream);
                        // Will process on next iteration
                        continue;
                    }
                }
                continue;
            }
        }
        
        // =====================================================================
        // STEP 2: Launch new batch on current slot (non-blocking)
        // =====================================================================
        Digest target;
        {
            std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
            target = gpu_res->current_target;
        }
        
        uint64_t start_nonce = getNextNonceRange(gpu_res, gpu_res->optimal_max_nonces);
        
        do { *miner.slots[active_slot].h_solution_found_pinned = 0; } while (0);
        cudaMemsetAsync(miner.slots[active_slot].d_solution_found, 0, sizeof(int), miner.slots[active_slot].stream);
        
        int threads_per_block, blocks_per_grid;
        int gpu_id;
        cudaGetDevice(&gpu_id);
        calculate_mining_launch_config(gpu_res->optimal_max_nonces, threads_per_block, blocks_per_grid, gpu_id);
        
        if (!gpu_res->buffer->gpu_range_initialized) {
            int actual_gpu_count = g_total_gpu_count.load();
            GpuNonceRange gpu_range = calculate_gpu_range(gpu_id, actual_gpu_count);
            cudaMemcpyToSymbol(d_gpu_range_start, &gpu_range.range_start, sizeof(uint64_t));
            cudaMemcpyToSymbol(d_gpu_range_size, &gpu_range.range_size, sizeof(uint64_t));
            initialize_top_tree_cache(gpu_res->buffer->d_merkle_tree, gpu_res->buffer->num_leafs);
            gpu_res->buffer->gpu_range_initialized = true;
        }
        
        MiningKernelType kernel_type = select_mining_kernel(gpu_id);
        
        if (kernel_type == MiningKernelType::HIGH_VRAM) {
            parallel_mining_kernel_high_vram<<<blocks_per_grid, threads_per_block, 0, miner.slots[active_slot].stream>>>(
                gpu_res->buffer->d_leafs,
                gpu_res->buffer->d_merkle_tree,
                gpu_res->buffer->index_picker_preimage,
                target,
                start_nonce,
                gpu_res->optimal_max_nonces,
                gpu_res->buffer->num_leafs,
                MERKLE_TREE_HEIGHT_,
                gpu_res->buffer->mast_paths,
                gpu_res->buffer->hash,
                gpu_res->buffer->consensus_rule_set,
                miner.slots[active_slot].d_solution_nonce,
                miner.slots[active_slot].d_solution_found,
                miner.slots[active_slot].d_solution_path_a,
                miner.slots[active_slot].d_solution_path_b,
                miner.slots[active_slot].d_solution_nonce_digest,
                miner.slots[active_slot].d_solution_final_hash);
        } else {
            parallel_mining_kernel_low_vram<<<blocks_per_grid, threads_per_block, 0, miner.slots[active_slot].stream>>>(
                nullptr,
                gpu_res->buffer->d_merkle_tree,
                gpu_res->buffer->index_picker_preimage,
                target,
                start_nonce,
                gpu_res->optimal_max_nonces,
                gpu_res->buffer->num_leafs,
                MERKLE_TREE_HEIGHT_,
                gpu_res->buffer->tree_size,
                gpu_res->buffer->mast_paths,
                gpu_res->buffer->hash,
                gpu_res->buffer->consensus_rule_set,
                miner.slots[active_slot].d_solution_nonce,
                miner.slots[active_slot].d_solution_found,
                miner.slots[active_slot].d_solution_path_a,
                miner.slots[active_slot].d_solution_path_b,
                miner.slots[active_slot].d_solution_nonce_digest,
                miner.slots[active_slot].d_solution_final_hash);
        }
        
        mining_kernel_finalize<<<1, 1, 0, miner.slots[active_slot].stream>>>(miner.slots[active_slot].d_solution_found);
        
        cudaMemcpyAsync(miner.slots[active_slot].h_solution_found_pinned, miner.slots[active_slot].d_solution_found,
                        sizeof(int), cudaMemcpyDeviceToHost, miner.slots[active_slot].stream);
        cudaEventRecord(miner.slots[active_slot].completion_event, miner.slots[active_slot].stream);
        
        current_batch_info.proposal_id = proposal_id;
        current_batch_info.template_obj = template_for_this_batch;
        current_batch_info.start_nonce = start_nonce;
        current_batch_info.batch_size = gpu_res->optimal_max_nonces;
        current_batch_info.start_time = std::chrono::high_resolution_clock::now();
        current_batch_info.pending_result = true;
        miner.slots[active_slot].kernel_launched = true;
        batches_in_flight++;
        active_slot = 1 - active_slot;
        
        // =====================================================================
        // STEP 3: Status update (non-blocking)
        // =====================================================================
        auto now = std::chrono::steady_clock::now();
        if (now - last_status_time >= STATUS_UPDATE_INTERVAL) {
            double avg_hashrate = gpu_res->hash_tracker.get_average();
            double hashrate_hps = avg_hashrate * 1000.0;
            std::string hashrate_str = format_hashrate(hashrate_hps);
            
            std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::GREEN 
                      << "Mining (async x2)..." << Color::RESET 
                      << " | Rate: " << Color::YELLOW << hashrate_str << Color::RESET
                      << " | Nonces: " << gpu_res->total_nonces_tested.load()
                      << " | " << Color::GREEN << gpu_res->solutions_accepted.load() << Color::RESET 
                      << "/" << Color::RED << gpu_res->solutions_rejected.load() << Color::RESET 
                      << std::endl;
            
            last_status_time = now;
        }
    }
    
    // Cleanup: drain pipeline
    for (int i = 0; i < 2; i++) {
        if (batch_info[i].pending_result) {
            cudaStreamSynchronize(miner.slots[i].stream);
        }
    }
    
    return true;
}

// ============================================================================
// Synchronous Mining Loop (Fallback)
// ============================================================================
// Original synchronous implementation used as fallback if async init fails

bool continuousMiningLoopSync(GpuResources* gpu_res, GpuWorker* worker) {
    if (!gpu_res || !worker) return false;
    
    auto last_status_time = std::chrono::steady_clock::now();
    const auto STATUS_UPDATE_INTERVAL = std::chrono::seconds(5);
    
    while (!stop_mining && !gpu_res->gpu_stop_flag) {
        if (gpu_res->event_handler && gpu_res->event_handler->hasEvents()) {
            break;
        }
        
        // ---- P2.7: Check if async preprocessing completed → swap buffers (sync loop) ----
        if (gpu_res->async_preprocess_done.load(std::memory_order_acquire)) {
            gpu_res->buffer = std::move(gpu_res->pending_buffer);
            {
                std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
                gpu_res->current_proposal_id = gpu_res->pending_proposal_id;
                gpu_res->current_template = gpu_res->pending_template;
                gpu_res->current_target = gpu_res->pending_target;
                gpu_res->current_real_target = gpu_res->pending_real_target;
                gpu_res->cached_prev_block = gpu_res->pending_prev_block;
                gpu_res->cached_mast_paths = gpu_res->pending_mast_paths;
                gpu_res->cached_commitment = gpu_res->pending_commitment;
            }
            resetNonceCounter(gpu_res, gpu_res->pending_proposal_id);
            gpu_res->buffer->gpu_range_initialized = false;
            gpu_res->async_preprocess_done.store(false, std::memory_order_release);
            
            std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::GREEN << Color::BOLD
                      << "Buffer swapped — now mining new template" << Color::RESET << std::endl;
        }
        
        if (gpu_res->gpu_pause_flag) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        if (!gpu_res->buffer || !gpu_res->buffer->is_valid()) {
            gpu_res->set_paused(true);
            continue;
        }
        
        std::string proposal_id;
        json template_for_this_batch;
        {
            std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
            proposal_id = gpu_res->current_proposal_id;
            template_for_this_batch = gpu_res->current_template;
        }
        
        if (proposal_id.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        uint64_t start_nonce = getNextNonceRange(gpu_res, gpu_res->optimal_max_nonces);
        Digest target;
        {
            std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
            target = gpu_res->current_target;
        }
        
        auto result = mine_pow_with_buffer(
            *gpu_res->buffer,
            target,
            gpu_res->buffer->mast_paths,
            start_nonce,
            gpu_res->optimal_max_nonces,
            gpu_res->buffer->consensus_rule_set,
            nullptr
        );
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        gpu_res->total_nonces_tested.fetch_add(gpu_res->optimal_max_nonces);
        
        if (duration_ms > 0) {
            double hashrate_ms = static_cast<double>(gpu_res->optimal_max_nonces) / duration_ms;
            gpu_res->hash_tracker.add(hashrate_ms);
        }
        
        auto now = std::chrono::steady_clock::now();
        if (now - last_status_time >= STATUS_UPDATE_INTERVAL) {
            double avg_hashrate = gpu_res->hash_tracker.get_average();
            double hashrate_hps = avg_hashrate * 1000.0;
            std::string hashrate_str = format_hashrate(hashrate_hps);
            
            uint64_t total_nonces = gpu_res->total_nonces_tested.load();
            uint64_t accepted = gpu_res->solutions_accepted.load();
            uint64_t rejected = gpu_res->solutions_rejected.load();
            
            std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::GREEN 
                      << "Mining (sync)..." << Color::RESET 
                      << " | Hash Rate: " << Color::YELLOW << hashrate_str << Color::RESET
                      << " | Nonces: " << total_nonces
                      << " | " << Color::GREEN << accepted << Color::RESET 
                      << " / " << Color::RED << rejected << Color::RESET 
                      << " (success / reject)" << std::endl;
            
            last_status_time = now;
        }
        
        if (result.has_value()) {
            gpu_res->solutions_found++;
            
            std::string current_proposal_id;
            {
                std::lock_guard<std::mutex> lock(gpu_res->state_mutex);
                current_proposal_id = gpu_res->current_proposal_id;
            }
            
            if (current_proposal_id != proposal_id) {
                std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::YELLOW 
                          << "Solution discarded - template changed" << Color::RESET << std::endl;
                continue;
            }
            
            const char* submit_target = gpu_res->is_stratum_mode() ? "pool" : "node";
            std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::YELLOW << Color::BOLD
                      << "*** SOLUTION FOUND! ***" << Color::RESET 
                      << " Submitting to " << submit_target << "..." << std::endl;
            
            PowMastPaths mast_paths = gpu_res->buffer->mast_paths;
            Digest solution_hash = mast_paths.fast_mast_hash(result.value().pow);
            
            auto future = worker->submitSolution(
                proposal_id,
                result.value().pow,
                solution_hash,
                template_for_this_batch);
            
            if (future.wait_for(std::chrono::seconds(30)) == std::future_status::ready) {
                bool accepted = future.get();
                if (accepted) {
                    gpu_res->solutions_accepted++;
                    const char* msg = gpu_res->is_stratum_mode() ? "SHARE ACCEPTED" : "BLOCK ACCEPTED";
                    std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::GREEN << Color::BOLD 
                              << "*** " << msg << "! ***" << Color::RESET 
                              << " | Total: " << Color::GREEN << gpu_res->solutions_accepted.load() 
                              << Color::RESET << std::endl;
                } else {
                    gpu_res->solutions_rejected++;
                }
            } else {
                gpu_res->solutions_rejected++;
                std::cout << "[GPU " << gpu_res->gpu_id << "] " << Color::RED 
                          << "Submission timed out" << Color::RESET << std::endl;
            }
        }
    }
    
    return true;
}

// ============================================================================
// Helper Functions
// ============================================================================

bool preprocessPuzzle(const PowPuzzle& puzzle, GpuResources* gpu_res) {
    if (!gpu_res) return false;
    
    cudaError_t err = cudaSetDevice(gpu_res->gpu_id);
    if (err != cudaSuccess) {
        LOG_ERROR("cudaSetDevice in preprocess", err);
        return false;
    }
    
    PowMastPaths mast_paths = convertToPowMastPaths(puzzle.auth_paths);
    Digest prev_block = hex_to_digest(puzzle.prev_block);
    auto buffer = Pow::preprocess(mast_paths, prev_block, puzzle.consensus_rule_set, nullptr);
    
    if (!buffer.is_valid()) {
        LOG_DEBUG("[GPU " << gpu_res->gpu_id << "] Preprocessing failed");
        return false;
    }
    
    gpu_res->buffer = std::make_unique<GuesserBuffer>(std::move(buffer));
    return true;
}

bool minePuzzleWithCuda(const PowPuzzle& puzzle, GpuResources* gpu_res) {
    if (!gpu_res || !gpu_res->buffer || !gpu_res->buffer->is_valid()) {
        return false;
    }
    
    cudaError_t err = cudaSetDevice(gpu_res->gpu_id);
    if (err != cudaSuccess) {
        return false;
    }
    
    Digest original_target = hex_to_digest(puzzle.threshold);
    Digest target = g_test_mode ? make_target_easier(original_target, 100000) : original_target;
    PowMastPaths mast_paths = convertToPowMastPaths(puzzle.auth_paths);
    uint64_t start_nonce = getNextNonceRange(gpu_res, gpu_res->optimal_max_nonces);
    
    auto result = mine_pow_with_buffer(
        *gpu_res->buffer,
        target,
        mast_paths,
        start_nonce,
        gpu_res->optimal_max_nonces,
        puzzle.consensus_rule_set,
        nullptr
    );
    
    // Update nonce counter
    gpu_res->total_nonces_tested.fetch_add(gpu_res->optimal_max_nonces);
    
    return result.has_value();
}

// ============================================================================
// Nonce Management
// ============================================================================

uint64_t getNextNonceRange(GpuResources* gpu_res, uint64_t batch_size) {
    if (!gpu_res) return 0;
    
    uint64_t current = gpu_res->gpu_puzzle_nonce_counter.fetch_add(batch_size);
    return gpu_res->gpu_puzzle_random_start + current;
}

void resetNonceCounter(GpuResources* gpu_res, const std::string& puzzle_id) {
    if (!gpu_res) return;
    
    // Generate new random start for this puzzle
    uint64_t random_start = generate_secure_random_start(
        puzzle_id,
        gpu_res->gpu_id,
        gpu_res->gpu_uuid
    );
    
    gpu_res->gpu_puzzle_random_start = random_start;
    gpu_res->gpu_puzzle_nonce_counter = 0;
}

// ============================================================================
// Signal Handling
// ============================================================================

void signal_handler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\n\nReceived interrupt signal, shutting down..." << std::endl;
        stop_mining = true;
        
        std::lock_guard<std::mutex> lock(g_all_gpu_resources_mutex);
        for (GpuResources* gpu : g_all_gpu_resources) {
            if (gpu && gpu->event_handler) {
                gpu->event_handler->postEvent(EventType::STOP_MINING);
            }
        }
    }
}

void install_signal_handlers() {
    signal(SIGINT, signal_handler);
#ifndef _WIN32
    signal(SIGTERM, signal_handler);
#endif
}

// Legacy puzzleFetcher removed; ConnectionMultiplexer handles polling.

bool verifySolution(
    const Pow& pow,
    const PowPuzzle& puzzle,
    const Digest& commitment,
    const Digest& target) {
    
    auto [index_a, index_b] = Pow::indices(commitment, pow.nonce);
    Digest leaf_a = Pow::compute_leaf_from_commitment_host(commitment, index_a, MERKLE_NUM_LEAFS);
    Digest leaf_b = Pow::compute_leaf_from_commitment_host(commitment, index_b, MERKLE_NUM_LEAFS);
    
    bool path_a_valid = Pow::verify_merkle_path_host(pow.root, index_a, pow.path_a, leaf_a);
    bool path_b_valid = Pow::verify_merkle_path_host(pow.root, index_b, pow.path_b, leaf_b);
    
    if (!path_a_valid || !path_b_valid) {
        LOG_DEBUG("Merkle path verification failed");
        return false;
    }
    
    PowMastPaths mast_paths = convertToPowMastPaths(puzzle.auth_paths);
    Digest final_hash = mast_paths.fast_mast_hash(pow);
    
    if (!digest_less_than_or_equal(final_hash, target)) {
        LOG_DEBUG("Hash does not meet target");
        return false;
    }
    
    return true;
}

// --- rpc_client.cu ---
#include "rpc_client.cuh"
#include "network.cuh"
#include "pow.cuh"
#include "digest.cuh"
#include <sstream>
#include <fstream>
#include <algorithm>

static bool http_post(const std::string& url,
                     const std::string& body,
                     const std::string& auth_header,
                     std::string& response,
                     int timeout_sec) {
    std::string protocol, host, path;
    int port = 80;
    
    size_t protocol_end = url.find("://");
    if (protocol_end == std::string::npos) {
        return false;
    }
    protocol = url.substr(0, protocol_end);
    std::string rest = url.substr(protocol_end + 3);
    
    if (protocol == "https") {
        return false;
    }
    
    size_t path_start = rest.find('/');
    if (path_start != std::string::npos) {
        host = rest.substr(0, path_start);
        path = rest.substr(path_start);
    } else {
        host = rest;
        path = "/";
    }
    
    size_t port_start = host.find(':');
    if (port_start != std::string::npos) {
        try {
            port = std::stoi(host.substr(port_start + 1));
            host = host.substr(0, port_start);
        } catch (...) {
            return false;
        }
    }
    
    socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET_VALUE) {
        return false;
    }
    
    struct hostent* host_entry = gethostbyname(host.c_str());
    if (!host_entry) {
        close(sock);
        return false;
    }
    
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    memcpy(&server_addr.sin_addr, host_entry->h_addr, host_entry->h_length);
    
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR_VALUE) {
        close(sock);
        return false;
    }
    
    #ifdef _WIN32
        DWORD timeout = timeout_sec * 1000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));
    #else
        struct timeval tv;
        tv.tv_sec = timeout_sec;
        tv.tv_usec = 0;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    #endif
    
    std::ostringstream request;
    request << "POST " << path << " HTTP/1.1\r\n";
    request << "Host: " << host << ":" << port << "\r\n";
    request << "Content-Type: application/json\r\n";
    request << "Content-Length: " << body.length() << "\r\n";
    if (!auth_header.empty()) {
        request << "Authorization: " << auth_header << "\r\n";
    }
    request << "Connection: close\r\n";
    request << "\r\n";
    request << body;
    
    std::string request_str = request.str();
    
    if (send(sock, request_str.c_str(), request_str.length(), 0) == SOCKET_ERROR_VALUE) {
        close(sock);
        return false;
    }
    
    response.clear();
    char buffer[4096];
    ssize_t received;
    while ((received = recv(sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
        buffer[received] = '\0';
        response += buffer;
        
        // Check if we've received the complete response
        size_t header_end = response.find("\r\n\r\n");
        if (header_end != std::string::npos) {
            // Headers received, check Content-Length
            std::string headers = response.substr(0, header_end);
            size_t content_length_pos = headers.find("Content-Length:");
            if (content_length_pos != std::string::npos) {
                size_t len_start = content_length_pos + 15;
                while (len_start < headers.length() && headers[len_start] == ' ') len_start++;
                size_t len_end = len_start;
                while (len_end < headers.length() && headers[len_end] >= '0' && headers[len_end] <= '9') len_end++;
                if (len_end > len_start) {
                    try {
                        int content_length = std::stoi(headers.substr(len_start, len_end - len_start));
                        size_t body_start = header_end + 4;
                        if (response.length() - body_start >= static_cast<size_t>(content_length)) {
                            break; // Received complete response
                        }
                    } catch (...) {
                        // If parsing fails, continue reading
                    }
                }
            }
        }
    }
    
    // recv returns 0 when connection is closed (normal)
    // recv returns -1 on error (timeout or other error)
    if (received < 0) {
        close(sock);
        return false;
    }
    
    close(sock);
    
    size_t header_end = response.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        return false;
    }
    
    size_t status_line_end = response.find("\r\n");
    if (status_line_end == std::string::npos) {
        return false;
    }
    
    std::string status_line = response.substr(0, status_line_end);
    
    if (status_line.find("HTTP/1.1 200") == std::string::npos &&
        status_line.find("HTTP/1.0 200") == std::string::npos) {
        return false;
    }
    
    response = response.substr(header_end + 4);
    
    return true;
}

// Base64 encode (simple implementation)
static std::string base64_encode_simple(const std::string& input) {
    const char base64_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    int val = 0, valb = -6;
    
    for (unsigned char c : input) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            encoded.push_back(base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    
    if (valb > -6) {
        encoded.push_back(base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    }
    
    while (encoded.size() % 4) {
        encoded.push_back('=');
    }
    
    return encoded;
}

XntRpcClient::XntRpcClient(const RpcConfig& cfg)
    : config(cfg)
    , request_id_counter(1)
    , last_error(RpcError::None)
    , last_error_message("") {
}

XntRpcClient::XntRpcClient(const std::string& url,
                           const std::string& user,
                           const std::string& password)
    : config()
    , request_id_counter(1)
    , last_error(RpcError::None)
    , last_error_message("") {
    config.url = url;
    config.user = user;
    config.password = password;
}

bool XntRpcClient::make_rpc_request(const std::string& method,
                                   const json& params,
                                   json& response) {
    std::lock_guard<std::mutex> lock(rpc_mutex);
    
    json request;
    request["jsonrpc"] = "2.0";
    request["method"] = method;
    request["params"] = params;
    request["id"] = request_id_counter.fetch_add(1);
    
    std::string request_body = request.dump();
    std::string auth_header = build_auth_header();
    std::string http_response;
    
    if (!http_post(config.url, request_body, auth_header, http_response, config.timeout_sec)) {
        last_error = RpcError::ConnectionFailed;
        last_error_message = "HTTP request failed";
        return false;
    }
    
    try {
        response = json::parse(http_response);
        
        if (response.contains("error")) {
            last_error = parse_rpc_error(response);
            last_error_message = get_error_message(response);
            return false;
        }
        
        last_error = RpcError::None;
        last_error_message = "";
        return true;
    } catch (const std::exception& e) {
        last_error = RpcError::InvalidResponse;
        last_error_message = std::string("JSON parse error: ") + e.what();
        return false;
    }
}

std::string XntRpcClient::build_auth_header() const {
    if (!config.user.empty() && !config.password.empty()) {
        std::string credentials = config.user + ":" + config.password;
        return "Basic " + base64_encode_simple(credentials);
    }
    
    if (!config.cookie_file.empty()) {
        std::string cookie = load_cookie();
        if (!cookie.empty()) {
            return "Cookie: " + cookie;
        }
    }
    
    return "";
}

std::string XntRpcClient::load_cookie() const {
    if (config.cookie_file.empty()) {
        return "";
    }
    
    std::ifstream file(config.cookie_file);
    if (!file.is_open()) {
        return "";
    }
    
    std::string cookie;
    std::getline(file, cookie);
    return cookie;
}

RpcError XntRpcClient::parse_rpc_error(const json& response) const {
    if (!response.contains("error")) {
        return RpcError::None;
    }
    
    json error = response["error"];
    std::string message = (error.contains("message") && !error["message"].is_null()) 
        ? error.value("message", "") : "";
    
    if (message.find("InvalidBlock") != std::string::npos) {
        return RpcError::InvalidBlock;
    } else if (message.find("InsufficientWork") != std::string::npos) {
        return RpcError::InsufficientWork;
    } else if (message.find("Authentication") != std::string::npos ||
               message.find("Unauthorized") != std::string::npos) {
        return RpcError::AuthenticationFailed;
    }
    
    return RpcError::Unknown;
}

std::string XntRpcClient::get_error_message(const json& response) const {
    if (!response.contains("error")) {
        return "";
    }
    
    json error = response["error"];
    return (error.contains("message") && !error["message"].is_null())
        ? error.value("message", "Unknown error") : "Unknown error";
}

json XntRpcClient::getBlockTemplate(const std::string& guesser_address) {
    json params = json::array();
    params.push_back(guesser_address);
    
    json response;
    if (make_rpc_request("mining_getBlockTemplate", params, response)) {
        return response;
    }
    
    return json();
}

json XntRpcClient::submitBlock(const json& template_obj, const json& pow) {
    // JSON-RPC server expects params as an array: [template, pow]
    json params = json::array();
    params.push_back(template_obj);
    params.push_back(pow);
    
    json response;
    make_rpc_request("mining_submitBlock", params, response);
    
    // Return response even if it contains an error, so caller can extract error details
    return response;
}

uint64_t XntRpcClient::getChainHeight() {
    json params = json::array();
    json response;
    
    if (make_rpc_request("chain_height", params, response)) {
        if (response.contains("result") && response["result"].contains("height")) {
            return response["result"]["height"].get<uint64_t>();
        }
    }
    
    return 0;
}

std::string XntRpcClient::getTipDigest() {
    json params = json::array();
    json response;
    
    if (make_rpc_request("chain_tipDigest", params, response)) {
        if (response.contains("result") && response["result"].contains("digest")) {
            return response["result"]["digest"].get<std::string>();
        }
    }
    
    return "";
}

json XntRpcClient::getTipHeader() {
    json params = json::array();
    json response;
    
    if (make_rpc_request("chain_tipHeader", params, response)) {
        return response;
    }
    
    return json();
}

bool XntRpcClient::testConnection() {
    json params = json::array();
    json response;
    
    return make_rpc_request("chain_height", params, response);
}

// Track last logged proposal ID to avoid duplicate logs
static std::string g_last_logged_proposal_id;
static bool g_null_template_logged = false;

PowPuzzle parseRpcTemplate(const json& template_response) {
    PowPuzzle puzzle;
    
    try {
        if (!template_response.contains("result")) {
            return puzzle;
        }
        
        json result = template_response["result"];
        if (!result.contains("template") || result["template"].is_null()) {
            // Only log null template once to avoid spam
            if (!g_null_template_logged) {
                std::cout << "[RPC] " << Color::YELLOW << "Template is null (node may be syncing)" << Color::RESET << std::endl;
                g_null_template_logged = true;
            }
            return puzzle;
        }
        // Reset null template flag when we get a valid template
        if (g_null_template_logged) {
            g_null_template_logged = false;
        }
        
        json template_obj = result["template"];
        if (!template_obj.contains("metadata") || template_obj["metadata"].is_null()) {
            return puzzle;
        }
        
        // Match Rust structure: RpcBlockTemplateMetadata
        // Fields: digest, prev_block, threshold, total_guesser_reward, pow_mast_paths
        json metadata = template_obj["metadata"];
        
        // Handle both snake_case (pow_mast_paths) and camelCase (powMastPaths)
        json pow_mast_paths;
        if (metadata.contains("pow_mast_paths") && !metadata["pow_mast_paths"].is_null()) {
            pow_mast_paths = metadata["pow_mast_paths"];
        } else if (metadata.contains("powMastPaths") && !metadata["powMastPaths"].is_null()) {
            pow_mast_paths = metadata["powMastPaths"];
        } else {
            return puzzle;
        }
        
        // Extract proposal/template ID (digest field)
        // This is the unique identifier for this block template
        if (metadata.contains("digest") && !metadata["digest"].is_null()) {
            puzzle.id = metadata.value("digest", "");
        } else {
            // If digest is missing, template is invalid
            return puzzle;
        }
        // Extract threshold (target digest for PoW solution)
        if (metadata.contains("threshold") && !metadata["threshold"].is_null()) {
            puzzle.threshold = metadata.value("threshold", "");
        } else {
            // Threshold is required for mining
            return puzzle;
        }
        // Handle both snake_case (total_guesser_reward) and camelCase (totalGuesserReward)
        if (metadata.contains("total_guesser_reward") && !metadata["total_guesser_reward"].is_null()) {
            puzzle.total_guesser_reward = metadata.value("total_guesser_reward", "");
        } else if (metadata.contains("totalGuesserReward") && !metadata["totalGuesserReward"].is_null()) {
            puzzle.total_guesser_reward = metadata.value("totalGuesserReward", "");
        }
        // Handle both snake_case (prev_block) and camelCase (prevBlock)
        if (metadata.contains("prev_block") && !metadata["prev_block"].is_null()) {
            puzzle.prev_block = metadata.value("prev_block", "");
        } else if (metadata.contains("prevBlock") && !metadata["prevBlock"].is_null()) {
            puzzle.prev_block = metadata.value("prevBlock", "");
        }
        
        if (pow_mast_paths.contains("pow") && pow_mast_paths["pow"].is_array()) {
            for (const auto& path : pow_mast_paths["pow"]) {
                if (path.is_string()) {
                    puzzle.auth_paths.pow.push_back(path.get<std::string>());
                } else if (path.is_array()) {
                    std::ostringstream hex;
                    for (const auto& limb : path) {
                        if (limb.is_number()) {
                            hex << std::hex << limb.get<uint64_t>();
                        }
                    }
                    puzzle.auth_paths.pow.push_back(hex.str());
                }
            }
        }
        
        if (pow_mast_paths.contains("header") && pow_mast_paths["header"].is_array()) {
            for (const auto& path : pow_mast_paths["header"]) {
                if (path.is_string()) {
                    puzzle.auth_paths.header.push_back(path.get<std::string>());
                } else if (path.is_array()) {
                    std::ostringstream hex;
                    for (const auto& limb : path) {
                        if (limb.is_number()) {
                            hex << std::hex << limb.get<uint64_t>();
                        }
                    }
                    puzzle.auth_paths.header.push_back(hex.str());
                }
            }
        }
        
        if (pow_mast_paths.contains("kernel") && pow_mast_paths["kernel"].is_array()) {
            for (const auto& path : pow_mast_paths["kernel"]) {
                if (path.is_string()) {
                    puzzle.auth_paths.kernel.push_back(path.get<std::string>());
                } else if (path.is_array()) {
                    std::ostringstream hex;
                    for (const auto& limb : path) {
                        if (limb.is_number()) {
                            hex << std::hex << limb.get<uint64_t>();
                        }
                    }
                    puzzle.auth_paths.kernel.push_back(hex.str());
                }
            }
        }
        
        // For mainnet blocks >= 15256, we use CONSENSUS_XNT
        // Since current block height is > 15256, always use XNT consensus
        puzzle.consensus_rule_set = CONSENSUS_XNT;
        
        // Only log when we get a NEW block proposal (different proposal ID)
        // This matches the Rust RpcBlockTemplateMetadata structure
        if (puzzle.id != g_last_logged_proposal_id && !puzzle.id.empty()) {
            g_last_logged_proposal_id = puzzle.id;
            
            std::string short_id = puzzle.id.length() > 20 
                ? puzzle.id.substr(0, 12) + "..." + puzzle.id.substr(puzzle.id.length() - 8) 
                : puzzle.id;
            
            std::cout << "[RPC] " << Color::GREEN << Color::BOLD << "✓ Block proposal received" << Color::RESET << std::endl
                      << "  Proposal ID: " << Color::CYAN << short_id << Color::RESET << std::endl
                      << "  Threshold: " << puzzle.threshold.substr(0, 16) << "..." << std::endl
                      << "  Prev Block: " << (puzzle.prev_block.length() > 16 ? puzzle.prev_block.substr(0, 16) + "..." : puzzle.prev_block) << std::endl;
            // Only show reward in solo mode (pool mode doesn't include reward)
            if (!puzzle.total_guesser_reward.empty()) {
                std::cout << "  Reward: " << format_reward_xnt(puzzle.total_guesser_reward) << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "[RPC] " << Color::RED << "✗ Failed to parse block proposal: " << e.what() << Color::RESET << std::endl;
        LOG_DEBUG("Failed to parse RPC template: " << e.what());
    }
    
    return puzzle;
}

json powToRpcFormat(const Pow& pow_solution, const Digest& solution_hash) {
    json pow_json;

    // Digests are serialized as hex strings in the JSON-RPC API
    // Use pow_solution.root (the Merkle root) not solution_hash (the final block hash)
    pow_json["root"] = digest_to_hex(pow_solution.root);

    pow_json["pathA"] = json::array();
    for (size_t i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        pow_json["pathA"].push_back(digest_to_hex(pow_solution.path_a[i]));
    }

    pow_json["pathB"] = json::array();
    for (size_t i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        pow_json["pathB"].push_back(digest_to_hex(pow_solution.path_b[i]));
    }

    pow_json["nonce"] = digest_to_hex(pow_solution.nonce);

    return pow_json;
}

// --- stratum_client.cu ---
#include "stratum_client.cuh"
#include "pow.cuh"
#include "digest.cuh"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <thread>

// OpenSSL includes
#ifndef _WIN32
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/x509v3.h>
#else
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/x509v3.h>
#endif

// Parse stratum URL into host and port
bool parse_stratum_url(const std::string& url, std::string& host, int& port, bool& use_ssl) {
    std::string work_url = url;
    use_ssl = false;
    
    // Remove protocol prefix
    const std::vector<std::string> prefixes = {
        "stratum+ssl://", "stratum+tcp://", "stratum://", "tcp://"
    };
    
    for (const auto& prefix : prefixes) {
        if (work_url.find(prefix) == 0) {
            work_url = work_url.substr(prefix.length());
            if (prefix == "stratum+ssl://") {
                use_ssl = true;
            }
            break;
        }
    }
    
    // Parse host:port
    size_t port_pos = work_url.rfind(':');
    if (port_pos != std::string::npos) {
        host = work_url.substr(0, port_pos);
        try {
            port = std::stoi(work_url.substr(port_pos + 1));
        } catch (...) {
            return false;
        }
    } else {
        host = work_url;
        port = 3333;  // Default stratum port
    }
    
    return !host.empty() && port > 0 && port <= 65535;
}

StratumClient::StratumClient(const StratumConfig& cfg)
    : config(cfg)
    , sock(INVALID_SOCKET_VALUE)
    , use_ssl(false)
    , ssl_ctx(nullptr)
    , ssl(nullptr) {
    platform_socket_init();
    // Initialize OpenSSL
    SSL_library_init();
    SSL_load_error_strings();
    OpenSSL_add_all_algorithms();

    use_ssl = config.use_ssl;
}

StratumClient::StratumClient(const std::string& url, const std::string& address,
                             const std::string& worker_name,
                             const std::string& password)
    : sock(INVALID_SOCKET_VALUE)
    , use_ssl(false)
    , ssl_ctx(nullptr)
    , ssl(nullptr) {
    platform_socket_init();
    // Initialize OpenSSL
    SSL_library_init();
    SSL_load_error_strings();
    OpenSSL_add_all_algorithms();
    
    bool url_use_ssl = false;
    if (!parse_stratum_url(url, config.host, config.port, url_use_ssl)) {
        config.host = "127.0.0.1";
        config.port = 3333;
    }
    use_ssl = url_use_ssl;
    config.use_ssl = url_use_ssl;
    
    config.address = address;
    config.name = worker_name;
    config.password = password;
}

StratumClient::~StratumClient() {
    disconnect();
    cleanup_ssl();
    platform_socket_cleanup();
}

bool StratumClient::connect_tcp() {
    if (sock != INVALID_SOCKET_VALUE) {
        disconnect_tcp();
    }
    
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET_VALUE) {
        last_error = StratumError::ConnectionFailed;
        last_error_message = "Failed to create socket";
        return false;
    }
    
    // Resolve hostname
    struct hostent* host_entry = gethostbyname(config.host.c_str());
    if (!host_entry) {
        close(sock);
        sock = INVALID_SOCKET_VALUE;
        last_error = StratumError::ConnectionFailed;
        last_error_message = "Failed to resolve hostname: " + config.host;
        return false;
    }
    
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(config.port);
    memcpy(&server_addr.sin_addr, host_entry->h_addr, host_entry->h_length);
    
    // Set connection timeout
    #ifdef _WIN32
        DWORD timeout = config.connect_timeout_sec * 1000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));
    #else
        struct timeval tv;
        tv.tv_sec = config.connect_timeout_sec;
        tv.tv_usec = 0;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    #endif
    
    if (::connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR_VALUE) {
        close(sock);
        sock = INVALID_SOCKET_VALUE;
        last_error = StratumError::ConnectionFailed;
        last_error_message = "Failed to connect to " + config.host + ":" + std::to_string(config.port);
        return false;
    }
    
    // Set read timeout for normal operation
    #ifdef _WIN32
        timeout = config.read_timeout_sec * 1000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
    #else
        tv.tv_sec = config.read_timeout_sec;
        tv.tv_usec = 0;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    #endif
    
    // Initialize SSL if needed
    if (use_ssl) {
        if (!init_ssl()) {
            disconnect_tcp();
            return false;
        }
    }
    
    return true;
}

bool StratumClient::init_ssl() {
    if (!use_ssl) return true;
    
    // Create SSL context
    const SSL_METHOD* method = TLS_client_method();
    ssl_ctx = SSL_CTX_new(method);
    if (!ssl_ctx) {
        last_error = StratumError::ConnectionFailed;
        last_error_message = "Failed to create SSL context";
        return false;
    }
    
    // Set options
    SSL_CTX_set_options(ssl_ctx, SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3 | SSL_OP_NO_TLSv1 | SSL_OP_NO_TLSv1_1);
    SSL_CTX_set_min_proto_version(ssl_ctx, TLS1_2_VERSION);
    
    // Create SSL object
    ssl = SSL_new(ssl_ctx);
    if (!ssl) {
        cleanup_ssl();
        last_error = StratumError::ConnectionFailed;
        last_error_message = "Failed to create SSL object";
        return false;
    }
    
    // Attach socket to SSL
    if (SSL_set_fd(ssl, sock) != 1) {
        cleanup_ssl();
        last_error = StratumError::ConnectionFailed;
        last_error_message = "Failed to set SSL file descriptor";
        return false;
    }
    
    // Set SNI for virtual-hosted TLS endpoints
    SSL_set_tlsext_host_name(ssl, config.host.c_str());
    SSL_set_mode(ssl, SSL_MODE_AUTO_RETRY);
    
    // Perform SSL handshake
    int ssl_result = SSL_connect(ssl);
    if (ssl_result != 1) {
        int ssl_error = SSL_get_error(ssl, ssl_result);
        std::string error_msg = "SSL handshake failed: ";
        char error_buf[256];
        ERR_error_string_n(ERR_get_error(), error_buf, sizeof(error_buf));
        error_msg += error_buf;
        
        cleanup_ssl();
        last_error = StratumError::ConnectionFailed;
        last_error_message = error_msg;
        return false;
    }
    
    std::cout << "[Pool] " << Color::GREEN << "SSL/TLS connection established" << Color::RESET << std::endl;
    return true;
}

void StratumClient::cleanup_ssl() {
    if (ssl) {
        SSL_shutdown(ssl);
        SSL_free(ssl);
        ssl = nullptr;
    }
    if (ssl_ctx) {
        SSL_CTX_free(ssl_ctx);
        ssl_ctx = nullptr;
    }
}

void StratumClient::disconnect_tcp() {
    if (use_ssl && ssl) {
        SSL_shutdown(ssl);
    }
    
    if (sock != INVALID_SOCKET_VALUE) {
        #ifdef _WIN32
            shutdown(sock, SD_BOTH);
        #else
            shutdown(sock, SHUT_RDWR);
        #endif
        close(sock);
        sock = INVALID_SOCKET_VALUE;
    }
}

bool StratumClient::send_line(const std::string& line) {
    if (sock == INVALID_SOCKET_VALUE) {
        return false;
    }
    
    std::string msg = line;
    if (msg.empty() || msg.back() != '\n') {
        msg += '\n';
    }
    
    size_t total_sent = 0;
    while (total_sent < msg.length()) {
        ssize_t sent;
        if (use_ssl && ssl) {
            sent = SSL_write(ssl, msg.c_str() + total_sent, msg.length() - total_sent);
            if (sent <= 0) {
                int ssl_error = SSL_get_error(ssl, sent);
                if (ssl_error == SSL_ERROR_WANT_READ || ssl_error == SSL_ERROR_WANT_WRITE) {
                    continue;  // Retry
                }
                last_error = StratumError::ConnectionLost;
                last_error_message = "SSL write failed";
                return false;
            }
        } else {
            sent = send(sock, msg.c_str() + total_sent, msg.length() - total_sent, 0);
            if (sent <= 0) {
                last_error = StratumError::ConnectionLost;
                last_error_message = "Failed to send data";
                return false;
            }
        }
        total_sent += sent;
    }
    
    return true;
}

bool StratumClient::send_json(const json& message) {
    return send_line(message.dump());
}

std::string StratumClient::read_line(int timeout_ms) {
    if (sock == INVALID_SOCKET_VALUE) {
        std::cerr << "[Pool] read_line: invalid socket" << std::endl;
        return "";
    }
    
    // Set timeout on the underlying socket (for non-SSL recv)
    #ifdef _WIN32
        DWORD timeout = timeout_ms;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
    #else
        struct timeval tv;
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    #endif
    
    std::string line;
    char c;
    int bytes_received = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    while (true) {
        // Check timeout
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count();
        if (elapsed > timeout_ms) {
            std::cerr << "[Pool] read_line: timeout after " << elapsed << "ms, received " << bytes_received << " bytes" << std::endl;
            break;
        }
        
        ssize_t received;
        if (use_ssl && ssl) {
            // For SSL, we need to handle non-blocking differently
            received = SSL_read(ssl, &c, 1);
            if (received <= 0) {
                int ssl_error = SSL_get_error(ssl, received);
                if (ssl_error == SSL_ERROR_WANT_READ || ssl_error == SSL_ERROR_WANT_WRITE) {
                    // Would block, sleep briefly and retry
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
                if (ssl_error == SSL_ERROR_ZERO_RETURN) {
                    // Connection closed cleanly
                    std::cerr << "[Pool] read_line: SSL connection closed" << std::endl;
                    connected = false;
                    break;
                }
                // Other SSL error
                char err_buf[256];
                ERR_error_string_n(ERR_get_error(), err_buf, sizeof(err_buf));
                std::cerr << "[Pool] read_line: SSL error " << ssl_error << ": " << err_buf << std::endl;
                break;
            }
        } else {
            received = recv(sock, &c, 1, 0);
            if (received <= 0) {
                if (received == 0) {
                    // Connection closed by peer
                    std::cerr << "[Pool] read_line: connection closed by peer" << std::endl;
                    connected = false;
                }
                #ifndef _WIN32
                else if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    // Timeout - return what we have
                    break;
                }
                #endif
                break;
            }
        }
        
        bytes_received++;
        
        if (c == '\n') {
            break;
        }
        
        if (c != '\r') {
            line += c;
        }
    }
    
    return line;
}

bool StratumClient::connect() {
    if (connected.load()) {
        return true;
    }
    
    std::cout << "[Pool] Connecting to " << config.host << ":" << config.port << "..." << std::endl;
    
    if (!connect_tcp()) {
        std::cout << "[Pool] " << Color::RED << "Connection failed: " << last_error_message << Color::RESET << std::endl;
        return false;
    }
    
    connected = true;
    std::cout << "[Pool] " << Color::GREEN << "TCP connection established" << Color::RESET << std::endl;
    
    // Perform login
    if (!do_login()) {
        std::cout << "[Pool] " << Color::RED << "Login failed: " << last_error_message << Color::RESET << std::endl;
        disconnect();
        return false;
    }
    
    std::cout << "[Pool] " << Color::GREEN << "Successfully logged in as worker " << worker_id << Color::RESET << std::endl;
    
    // Start receive thread
    running = true;
    receive_thread = std::thread(&StratumClient::receive_loop, this);
    
    // Start keepalive thread
    keepalive_thread = std::thread(&StratumClient::keepalive_loop, this);
    
    return true;
}

void StratumClient::disconnect() {
    running = false;
    connected = false;
    logged_in = false;
    
    disconnect_tcp();
    
    // Wake up any waiting threads
    job_cv.notify_all();
    
    if (receive_thread.joinable()) {
        receive_thread.join();
    }
    
    if (keepalive_thread.joinable()) {
        keepalive_thread.join();
    }
    
    // Clear job queue
    {
        std::lock_guard<std::mutex> lock(job_mutex);
        while (!job_queue.empty()) {
            job_queue.pop();
        }
    }
    
    // Clear pending requests
    {
        std::lock_guard<std::mutex> lock(pending_mutex);
        pending_requests.clear();
    }
    
    worker_id = 0;
}

bool StratumClient::is_connected() const {
    return connected.load() && logged_in.load();
}

bool StratumClient::do_login() {
    // Build login request matching pool protocol:
    // JsonRequest { id: Option<u64>, request: Request::Login { name, address, password, agent } }
    // Serializes to: { "id": N, "method": "login", "params": { "name": "...", "address": "...", "password": "...", "agent": "..." } }
    // Note: No "jsonrpc" field - this is NOT standard JSON-RPC 2.0
    
    json params;
    params["name"] = config.name;
    params["address"] = config.address;
    if (!config.password.empty()) {
        params["password"] = config.password;
    }
    params["agent"] = config.agent;
    
    json request;
    request["id"] = request_id_counter++;
    request["method"] = "login";
    request["params"] = params;
    
    std::string request_str = request.dump();
    std::cout << "[Pool] Sending login: " << request_str << std::endl;
    
    if (!send_json(request)) {
        last_error = StratumError::AuthenticationFailed;
        last_error_message = "Failed to send login request";
        return false;
    }
    
    // Wait for response with longer timeout (60 seconds)
    std::string line = read_line(60000);
    if (line.empty()) {
        last_error = StratumError::Timeout;
        last_error_message = "Timeout waiting for login response";
        return false;
    }
    
    std::cout << "[Pool] Received: " << (line.length() > 500 ? line.substr(0, 500) + "..." : line) << std::endl;
    
    try {
        json response = json::parse(line);
        
        // Check for JSON-RPC error
        if (response.contains("error") && !response["error"].is_null()) {
            last_error = StratumError::AuthenticationFailed;
            auto& err = response["error"];
            if (err.is_object() && err.contains("message")) {
                last_error_message = err["message"].get<std::string>();
            } else if (err.is_string()) {
                last_error_message = err.get<std::string>();
            } else {
                last_error_message = response.dump();
            }
            return false;
        }
        
        // Parse login response: { "id": N, "jsonrpc": "2.0", "result": { "id": worker_id, "job": {...} } }
        if (response.contains("result")) {
            json result = response["result"];
            
            // Handle case where result might be nested or direct
            if (result.is_object()) {
                // Get worker ID
                if (result.contains("id") && result["id"].is_number()) {
                    worker_id = result["id"].get<size_t>();
                } else {
                    // Some pools might not return worker id, use 0
                    worker_id = 0;
                }
                
                logged_in = true;
                std::cout << "[Pool] Logged in as: " << config.name << " (address: " << config.address.substr(0, 20) << "...)" << std::endl;
                std::cout << "[Pool] Worker ID: " << worker_id << std::endl;
                
                // Check for initial job in login response
                if (result.contains("job") && !result["job"].is_null()) {
                    StratumJob job = parse_job_notification(result["job"]);
                    if (job.is_valid()) {
                        std::lock_guard<std::mutex> lock(current_job_mutex);
                        current_job = job;
                        
                        {
                            std::lock_guard<std::mutex> job_lock(job_mutex);
                            job_queue.push(job);
                        }
                        job_cv.notify_one();
                        
                        std::cout << "[Pool] Received initial job: " << job.job_id.substr(0, 16) << "..." << std::endl;
                    }
                }
                
                return true;
            }
        }
        
        last_error = StratumError::InvalidResponse;
        last_error_message = "Invalid login response: " + response.dump().substr(0, 200);
        return false;
        
    } catch (const std::exception& e) {
        last_error = StratumError::InvalidResponse;
        last_error_message = std::string("Failed to parse response: ") + e.what() + " - Raw: " + line.substr(0, 200);
        return false;
    }
}

bool StratumClient::send_keepalive() {
    // Build keepalive request matching pool protocol:
    // Request::Keepalived {} -> { "method": "keepalived", "params": {} }
    // Note: keepalived is a notification, no id needed
    json request;
    request["method"] = "keepalived";
    request["params"] = json::object();
    
    return send_json(request);
}

void StratumClient::keepalive_loop() {
    while (running.load() && connected.load()) {
        // Sleep for keepalive interval
        for (int i = 0; i < config.keepalive_interval_sec && running.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        if (!running.load() || !connected.load()) {
            break;
        }
        
        if (!send_keepalive()) {
            std::cerr << "[Pool] Failed to send keepalive" << std::endl;
        }
    }
}

void StratumClient::receive_loop() {
    while (running.load() && connected.load()) {
        std::string line = read_line(5000);  // 5 second timeout for polling
        
        if (line.empty()) {
            if (!running.load()) break;
            continue;
        }
        
        try {
            json message = json::parse(line);
            handle_message(message);
        } catch (const std::exception& e) {
            std::cerr << "[Pool] Failed to parse message: " << e.what() << std::endl;
        }
    }
}

void StratumClient::handle_message(const json& message) {
    // Check if this is a notification (no id) or a response (has id)
    if (!message.contains("id") || message["id"].is_null()) {
        // Notification from pool (job, pause, etc.)
        if (message.contains("method")) {
            std::string method = message["method"].get<std::string>();
            json params = message.value("params", json::object());
            handle_notification(method, params);
        }
    } else {
        // Response to a request
        uint64_t id = message["id"].get<uint64_t>();
        json result = message.value("result", json());
        json error = message.value("error", json());
        handle_response(id, result, error);
    }
}

void StratumClient::handle_notification(const std::string& method, const json& params) {
    // Pool schema notifications: job, pause
    if (method == "job") {
        // Job notification: { "method": "job", "params": { "id": "...", "paths": {...}, "difficulty": "..." } }
        StratumJob job = parse_job_notification(params);
        if (job.is_valid()) {
            // Update current job
            {
                std::lock_guard<std::mutex> lock(current_job_mutex);
                current_job = job;
            }
            
            // Add to queue (new jobs always replace old ones)
            {
                std::lock_guard<std::mutex> lock(job_mutex);
                // Clear old jobs on new job
                while (!job_queue.empty()) {
                    job_queue.pop();
                }
                job_queue.push(job);
            }
            job_cv.notify_one();
            
            std::string short_id = job.job_id.length() > 20 
                ? job.job_id.substr(0, 12) + "..." + job.job_id.substr(job.job_id.length() - 8)
                : job.job_id;
            std::cout << "[Pool] " << Color::CYAN << Color::BOLD 
                      << "New job received" << Color::RESET 
                      << " | Job ID: " << short_id 
                      << " | Difficulty: " << job.difficulty << std::endl;
        }
    } else if (method == "pause") {
        // Pause notification: stop mining temporarily
        std::cout << "[Pool] " << Color::YELLOW << "Mining paused by pool" << Color::RESET << std::endl;
        // Clear job queue to stop mining
        {
            std::lock_guard<std::mutex> lock(job_mutex);
            while (!job_queue.empty()) {
                job_queue.pop();
            }
        }
        {
            std::lock_guard<std::mutex> lock(current_job_mutex);
            current_job = StratumJob();  // Invalidate current job
        }
    } else {
        std::cout << "[Pool] Unknown notification: " << method << std::endl;
    }
}

void StratumClient::handle_response(uint64_t id, const json& result, const json& error) {
    std::lock_guard<std::mutex> lock(pending_mutex);
    auto it = pending_requests.find(id);
    if (it != pending_requests.end()) {
        json response;
        response["result"] = result;
        response["error"] = error;
        it->second.set_value(response);
        pending_requests.erase(it);
    }
}

StratumJob StratumClient::parse_job_notification(const json& params) {
    StratumJob job;
    
    // Pool protocol Job format:
    // {
    //   "id": "digest_hex_string",
    //   "paths": {
    //     "pow": { "kernel_body": [...], "type_scripts": [...], "kernel": [...] },
    //     "header": { "body": [...], "appendix": [...] },
    //     "kernel": [...]
    //   },
    //   "difficulty": "string"
    // }
    
    try {
        if (!params.is_object()) {
            std::cerr << "[Pool] Job params is not an object" << std::endl;
            return job;
        }
        
        // Job ID (digest as hex string)
        if (params.contains("id")) {
            if (params["id"].is_string()) {
                job.job_id = params["id"].get<std::string>();
            } else if (params["id"].is_array()) {
                // Handle Digest as array of u64 values - convert to hex
                std::stringstream ss;
                for (const auto& val : params["id"]) {
                    if (val.is_number()) {
                        ss << std::hex << std::setfill('0') << std::setw(16) << val.get<uint64_t>();
                    }
                }
                job.job_id = ss.str();
            }
        }
        
        // Difficulty (pool difficulty - easier than RPC threshold for share validation)
        if (params.contains("difficulty") && params["difficulty"].is_string()) {
            job.difficulty = params["difficulty"].get<std::string>();
        }
        
        // PowMastPaths structure
        if (params.contains("paths") && params["paths"].is_object()) {
            const auto& paths = params["paths"];
            
            // Parse pow paths: { "kernel_body": [...], "type_scripts": [...], "kernel": [...] }
            if (paths.contains("pow") && paths["pow"].is_object()) {
                const auto& pow = paths["pow"];
                
                if (pow.contains("kernel_body") && pow["kernel_body"].is_array()) {
                    for (const auto& item : pow["kernel_body"]) {
                        if (item.is_string()) {
                            job.paths.pow_kernel_body.push_back(item.get<std::string>());
                        }
                    }
                }
                if (pow.contains("type_scripts") && pow["type_scripts"].is_array()) {
                    for (const auto& item : pow["type_scripts"]) {
                        if (item.is_string()) {
                            job.paths.pow_type_scripts.push_back(item.get<std::string>());
                        }
                    }
                }
                if (pow.contains("kernel") && pow["kernel"].is_array()) {
                    for (const auto& item : pow["kernel"]) {
                        if (item.is_string()) {
                            job.paths.pow_kernel.push_back(item.get<std::string>());
                        }
                    }
                }
            }
            
            // Parse header paths: { "body": [...], "appendix": [...] }
            if (paths.contains("header") && paths["header"].is_object()) {
                const auto& header = paths["header"];
                
                if (header.contains("body") && header["body"].is_array()) {
                    for (const auto& item : header["body"]) {
                        if (item.is_string()) {
                            job.paths.header_body.push_back(item.get<std::string>());
                        }
                    }
                }
                if (header.contains("appendix") && header["appendix"].is_array()) {
                    for (const auto& item : header["appendix"]) {
                        if (item.is_string()) {
                            job.paths.header_appendix.push_back(item.get<std::string>());
                        }
                    }
                }
            }
            
            // Parse kernel paths (direct array)
            if (paths.contains("kernel") && paths["kernel"].is_array()) {
                for (const auto& item : paths["kernel"]) {
                    if (item.is_string()) {
                        job.paths.kernel.push_back(item.get<std::string>());
                    }
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[Pool] Failed to parse job notification: " << e.what() << std::endl;
        return StratumJob();
    }
    
    return job;
}

json StratumClient::getBlockTemplate() {
    // Return the current job as a JSON template compatible with mining code
    std::lock_guard<std::mutex> lock(current_job_mutex);
    
    if (!current_job.is_valid()) {
        return json();
    }
    
    // Build a template response that matches the expected format
    json template_obj;
    json metadata;
    
            metadata["digest"] = current_job.job_id;
            metadata["threshold"] = current_job.difficulty;  // Pool difficulty (for share validation)
            // Note: No total_guesser_reward in pool mode
            
            // Build pow_mast_paths from pool paths structure
    json pow_mast_paths;
    
    // Combine pow paths into single array for legacy format
    json pow_paths = json::array();
    for (const auto& p : current_job.paths.pow_kernel_body) pow_paths.push_back(p);
    for (const auto& p : current_job.paths.pow_type_scripts) pow_paths.push_back(p);
    for (const auto& p : current_job.paths.pow_kernel) pow_paths.push_back(p);
    pow_mast_paths["pow"] = pow_paths;
    
    // Combine header paths
    json header_paths = json::array();
    for (const auto& p : current_job.paths.header_body) header_paths.push_back(p);
    for (const auto& p : current_job.paths.header_appendix) header_paths.push_back(p);
    pow_mast_paths["header"] = header_paths;
    
    // Kernel paths
    pow_mast_paths["kernel"] = current_job.paths.kernel;
    
    metadata["pow_mast_paths"] = pow_mast_paths;
    
    template_obj["metadata"] = metadata;
    template_obj["block"] = json::object();
    
    json result;
    result["template"] = template_obj;
    
    json response;
    response["result"] = result;
    
    return response;
}

json StratumClient::getBlockTemplate(const std::string& wallet_address) {
    (void)wallet_address;  // Stratum uses address from login, ignore parameter
    return getBlockTemplate();
}

bool StratumClient::wait_for_job(json& job, int timeout_ms) {
    std::unique_lock<std::mutex> lock(job_mutex);
    
    if (job_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
        return !job_queue.empty() || !running.load();
    })) {
        if (!job_queue.empty()) {
            StratumJob stratum_job = job_queue.front();
            job_queue.pop();
            
            // Convert to JSON format compatible with mining controller
            PowPuzzle puzzle = stratum_job.to_pow_puzzle();
            
            // Build response matching expected format
            json template_obj;
            json metadata;
            
            metadata["digest"] = puzzle.id;
            metadata["threshold"] = puzzle.threshold;  // Pool difficulty (for share validation)
            // Note: No total_guesser_reward in pool mode
            
            json pow_mast_paths;
            pow_mast_paths["pow"] = puzzle.auth_paths.pow;
            pow_mast_paths["header"] = puzzle.auth_paths.header;
            pow_mast_paths["kernel"] = puzzle.auth_paths.kernel;
            metadata["pow_mast_paths"] = pow_mast_paths;
            
            template_obj["metadata"] = metadata;
            template_obj["block"] = json::object();
            
            json result;
            result["template"] = template_obj;
            
            job["result"] = result;
            
            return true;
        }
    }
    
    return false;
}

bool StratumClient::submit_solution(
    const std::string& proposal_id,
    const Pow& pow_solution,
    const Digest& solution_hash,
    const json& template_obj) {
    
    (void)template_obj;  // Not used in pool submission
    (void)solution_hash; // Not needed in pool schema
    
    if (!is_connected()) {
        std::cout << "[Pool] " << Color::RED << "Cannot submit - not connected" << Color::RESET << std::endl;
        return false;
    }
    
    shares_submitted++;
    
    // Build submit request matching pool schema:
    // { "id": N, "method": "submit", "params": { "worker": worker_id, "id": job_id, "pow": BlockPow } }
    //
    // BlockPow structure (from neptune-cash):
    // {
    //   "nonce": [u64; 5],           // Digest as array
    //   "root": [u64; 5],            // Digest as array  
    //   "authentication_path_a": [[u64; 5]; 25],  // Array of Digests
    //   "authentication_path_b": [[u64; 5]; 25]   // Array of Digests
    // }
    
    // Build BlockPow object
    json pow_obj;
    
    // Nonce as array of 5 u64 values
    json nonce_arr = json::array();
    for (int i = 0; i < 5; ++i) {
        nonce_arr.push_back(pow_solution.nonce.values[i]);
    }
    pow_obj["nonce"] = nonce_arr;
    
    // Root as array of 5 u64 values
    json root_arr = json::array();
    for (int i = 0; i < 5; ++i) {
        root_arr.push_back(pow_solution.root.values[i]);
    }
    pow_obj["root"] = root_arr;
    
    // Path A - array of MERKLE_TREE_HEIGHT_ digests
    // Match BlockPow serialization (authentication_path_a / authentication_path_b)
    json path_a = json::array();
    for (size_t i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        json digest_arr = json::array();
        for (int j = 0; j < 5; ++j) {
            digest_arr.push_back(pow_solution.path_a[i].values[j]);
        }
        path_a.push_back(digest_arr);
    }
    pow_obj["authentication_path_a"] = path_a;
    
    // Path B - array of MERKLE_TREE_HEIGHT_ digests
    json path_b = json::array();
    for (size_t i = 0; i < MERKLE_TREE_HEIGHT_; ++i) {
        json digest_arr = json::array();
        for (int j = 0; j < 5; ++j) {
            digest_arr.push_back(pow_solution.path_b[i].values[j]);
        }
        path_b.push_back(digest_arr);
    }
    pow_obj["authentication_path_b"] = path_b;
    
    // Build params object
    json params;
    params["worker"] = worker_id;
    params["id"] = proposal_id;
    params["pow"] = pow_obj;
    
    json request;
    uint64_t req_id = request_id_counter++;
    request["id"] = req_id;
    request["method"] = "submit";
    request["params"] = params;
    
    // Create promise for response
    std::promise<json> response_promise;
    std::future<json> response_future = response_promise.get_future();
    
    {
        std::lock_guard<std::mutex> lock(pending_mutex);
        pending_requests[req_id] = std::move(response_promise);
    }
    
    if (!send_json(request)) {
        std::lock_guard<std::mutex> lock(pending_mutex);
        pending_requests.erase(req_id);
        std::cout << "[Pool] " << Color::RED << "Failed to send submit request" << Color::RESET << std::endl;
        shares_rejected++;
        return false;
    }
    
    // Wait for response with timeout
    if (response_future.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
        std::lock_guard<std::mutex> lock(pending_mutex);
        pending_requests.erase(req_id);
        std::cout << "[Pool] " << Color::RED << "Timeout waiting for submit response" << Color::RESET << std::endl;
        shares_rejected++;
        return false;
    }
    
    json response = response_future.get();
    
    // Check for JSON-RPC error
    if (response.contains("error") && !response["error"].is_null()) {
        shares_rejected++;
        std::string error_msg = "Unknown error";
        auto& err = response["error"];
        if (err.is_object() && err.contains("message")) {
            error_msg = err["message"].get<std::string>();
        } else if (err.is_string()) {
            error_msg = err.get<std::string>();
        }
        std::cout << "[Pool] " << Color::RED << "Share rejected: " << error_msg << Color::RESET << std::endl;
        return false;
    }
    
    // Pool schema submit response: { "result": { "success": true } }
    if (response.contains("result") && response["result"].is_object()) {
        auto& result = response["result"];
        if (result.contains("success") && result["success"].is_boolean() && result["success"].get<bool>()) {
            shares_accepted++;
            std::cout << "[Pool] " << Color::GREEN << Color::BOLD 
                      << "Share accepted!" << Color::RESET 
                      << " (" << shares_accepted.load() << "/" << shares_submitted.load() << ")" << std::endl;
            return true;
        }
    }
    
    shares_rejected++;
    std::cout << "[Pool] " << Color::RED << "Share rejected (unknown reason)" << Color::RESET << std::endl;
    return false;
}

// --- network.cu ---
#include "network.cuh"
#include "rpc_client.cuh"
#include "stratum_client.cuh"
#include "pow.cuh"
#include "digest.cuh"
#include "common.cuh"

NeptuneCudaMinerClient::NeptuneCudaMinerClient(
    const std::string& rpc_url,
    const std::string& wallet_addr)
    : rpc_url(rpc_url)
    , has_last_template(false)
    , wallet_address(wallet_addr) {
    RpcConfig config;
    config.url = rpc_url;
    config.timeout_sec = 30;
    config.poll_interval_sec = g_fetch_interval_sec;
    rpc_client = std::make_unique<XntRpcClient>(config);
}

NeptuneCudaMinerClient::~NeptuneCudaMinerClient() {
}

bool NeptuneCudaMinerClient::connect_to_node() {
    if (!rpc_client) {
        return false;
    }
    return rpc_client->testConnection();
}

bool NeptuneCudaMinerClient::is_connected() const {
    if (!rpc_client) {
        return false;
    }
    
    return rpc_client->testConnection();
}

json NeptuneCudaMinerClient::getBlockTemplate() {
    if (!rpc_client || wallet_address.empty()) {
        return json();
    }
    
    return rpc_client->getBlockTemplate(wallet_address);
}

bool NeptuneCudaMinerClient::submit_solution(
    const std::string& proposal_id,
    const Pow& pow_solution,
    const Digest& solution_hash,
    const json& template_obj) {
    
    if (!rpc_client) {
        return false;
    }
    
    if (template_obj.is_null() || template_obj.empty()) {
        std::cout << "[SUBMIT] ERROR: No template provided for proposal " << proposal_id << std::endl;
        return false;
    }
    std::string tip_digest = rpc_client->getTipDigest();
    
    // Check if template is stale
    if (template_obj.contains("metadata") && !template_obj["metadata"].is_null()) {
        json metadata = template_obj["metadata"];
        std::string prev_block;
        if (metadata.contains("prevBlock") && !metadata["prevBlock"].is_null()) {
            prev_block = metadata.value("prevBlock", "");
        } else if (metadata.contains("prev_block") && !metadata["prev_block"].is_null()) {
            prev_block = metadata.value("prev_block", "");
        }

        if (!prev_block.empty() && tip_digest != prev_block) {
            std::string short_prev = prev_block.length() > 20 ? prev_block.substr(0, 12) + "..." + prev_block.substr(prev_block.length() - 8) : prev_block;
            std::string short_tip = tip_digest.length() > 20 ? tip_digest.substr(0, 12) + "..." + tip_digest.substr(tip_digest.length() - 8) : tip_digest;
            std::cout << "[SUBMIT] " << Color::RED << "✗ REJECTED: Template stale (prev_block=" << short_prev << " != tip=" << short_tip << ")" << Color::RESET << std::endl;
            return false;
        }
    }
    
    // Extract the block from template (RpcBlockTemplate has "block" and "metadata" fields)
    // SubmitBlockRequest expects: { template: RpcBlock, pow: RpcBlockPow }
    if (!template_obj.contains("block") || template_obj["block"].is_null()) {
        std::cout << "[SUBMIT] ERROR: Template missing 'block' field" << std::endl;
        return false;
    }
    
    json block_obj = template_obj["block"];
    
    // Ensure block has kernel.appendix with required structure
    // RpcBlockAppendix is serialized as an array of RpcClaim objects
    if (!block_obj.contains("kernel") || block_obj["kernel"].is_null()) {
        std::cout << "[SUBMIT] ERROR: Block missing 'kernel' field" << std::endl;
        return false;
    }
    
    json kernel_obj = block_obj["kernel"];
    
    // Debug: Log the kernel structure
    bool has_appendix = kernel_obj.contains("appendix") && !kernel_obj["appendix"].is_null();
    if (has_appendix && kernel_obj["appendix"].is_array()) {
        LOG_DEBUG("[SUBMIT] kernel.appendix has " << kernel_obj["appendix"].size() << " claims");
    } else {
        std::cout << "[SUBMIT] " << Color::YELLOW << "WARNING: Block kernel missing valid 'appendix' - template may be incomplete" << Color::RESET << std::endl;
        // Don't override with empty array - the node requires the proper appendix claims
    }
    
    json pow_json = powToRpcFormat(pow_solution, solution_hash);
    
    json response = rpc_client->submitBlock(block_obj, pow_json);
    
    if (!response.empty() && response.contains("result")) {
        if (response["result"].is_boolean()) {
            bool success = response["result"].get<bool>();
            if (!success) {
                std::cout << "[SUBMIT] " << Color::RED << "✗ REJECTED: Block rejected" << Color::RESET << std::endl;
            }
            return success;
        }
        if (response["result"].contains("success")) {
            bool success = response["result"]["success"].get<bool>();
            if (!success) {
                std::cout << "[SUBMIT] " << Color::RED << "✗ REJECTED: Block rejected" << Color::RESET << std::endl;
            }
            return success;
        }
        return true;
    }
    
    if (response.contains("error") && !response["error"].is_null()) {
        json error = response["error"];
        std::string error_reason = "Unknown error";
        
        // Extract exact error reason from error data
        if (error.contains("data") && !error["data"].is_null()) {
            json error_data = error["data"];
            if (error_data.is_object() && !error_data.empty()) {
                // Get the first value from the error data object
                auto it = error_data.begin();
                if (it != error_data.end() && it.value().is_string()) {
                    error_reason = it.value().get<std::string>();
                } else if (it != error_data.end()) {
                    error_reason = it.value().dump();
                }
            } else if (error_data.is_string()) {
                error_reason = error_data.get<std::string>();
            }
        } else if (error.contains("message") && !error["message"].is_null()) {
            error_reason = error.value("message", "Unknown error");
        }
        
        std::cout << "[SUBMIT] " << Color::RED << "✗ REJECTED: " << error_reason << Color::RESET << std::endl;
    } else {
        std::cout << "[SUBMIT] " << Color::RED << "✗ REJECTED: No response from server" << Color::RESET << std::endl;
    }
    
    return false;
}

void NeptuneCudaMinerClient::cache_puzzle(const json& template_obj) {
    last_template = template_obj;
    has_last_template = true;
}

PowPuzzle parsePowPuzzle(const std::string& jsonStr) {
    try {
        json j = json::parse(jsonStr);
        if (j.contains("result") && j["result"].contains("template")) {
            return parseRpcTemplate(j);
        }
        return parseRpcTemplate(j);
    } catch (const std::exception& e) {
        LOG_DEBUG("Failed to parse puzzle JSON: " << e.what());
        return PowPuzzle();
    }
}

// ===== UnifiedMinerClient Implementation =====

UnifiedMinerClient::UnifiedMinerClient(
    const std::string& endpoint,
    const std::string& wallet_addr,
    const std::string& stratum_pass)
    : endpoint(endpoint)
    , wallet_address(wallet_addr)
    , stratum_password(stratum_pass) {
    mode = detect_mining_mode(endpoint);
}

bool UnifiedMinerClient::initialize() {
    if (client) {
        return true;  // Already initialized
    }
    
    if (mode == MiningMode::Stratum) {
        // Create stratum client
        std::string host;
        int port;
        bool use_ssl = false;
        if (!parse_stratum_url(endpoint, host, port, use_ssl)) {
            std::cerr << Color::RED << "Invalid stratum URL: " << endpoint << Color::RESET << std::endl;
            return false;
        }
        
        StratumConfig config;
        config.host = host;
        config.port = port;
        config.use_ssl = use_ssl;
        config.address = wallet_address;
        config.name = g_miner_worker_name.empty() ? "xnt-miner" : g_miner_worker_name;
        config.password = stratum_password;
        
        client = std::make_unique<StratumClient>(config);
        
        std::cout << "Initialized stratum client for " << host << ":" << port << std::endl;
    } else {
        // Create solo (HTTP RPC) client
        client = std::make_unique<NeptuneCudaMinerClient>(endpoint, wallet_address);
        std::cout << "Initialized solo mining client for " << endpoint << std::endl;
    }
    
    return client != nullptr;
}



// --- connection_multiplexer.cu ---
#include "connection_multiplexer.cuh"
#include "rpc_client.cuh"
#include "stratum_client.cuh"
#include "network.cuh"
#include "mining.cuh"
#include "pow.cuh"
#include "digest.cuh"
#include "common.cuh"
#include <algorithm>

// ============================================================================
// Singleton Instance
// ============================================================================

std::unique_ptr<ConnectionMultiplexer> ConnectionMultiplexer::instance = nullptr;
std::mutex ConnectionMultiplexer::instance_mutex;

ConnectionMultiplexer& ConnectionMultiplexer::getInstance() {
    std::lock_guard<std::mutex> lock(instance_mutex);
    if (!instance) {
        instance = std::unique_ptr<ConnectionMultiplexer>(new ConnectionMultiplexer());
    }
    return *instance;
}

void ConnectionMultiplexer::destroyInstance() {
    std::lock_guard<std::mutex> lock(instance_mutex);
    if (instance) {
        instance->shutdown();
        instance.reset();
    }
}

// ============================================================================
// SoloMiningClient Implementation
// ============================================================================

SoloMiningClient::SoloMiningClient(const std::string& url)
    : rpc_url(url) {
    RpcConfig config;
    config.url = url;
    config.timeout_sec = 30;
    config.poll_interval_sec = g_fetch_interval_sec;
    rpc_client = std::make_unique<XntRpcClient>(config);
}

SoloMiningClient::~SoloMiningClient() {
    disconnect();
}

bool SoloMiningClient::connect() {
    std::lock_guard<std::mutex> lock(client_mutex);
    if (!rpc_client) {
        return false;
    }
    bool result = rpc_client->testConnection();
    connected.store(result);
    return result;
}

void SoloMiningClient::disconnect() {
    connected.store(false);
}

bool SoloMiningClient::is_connected() const {
    return connected.load();
}

// reconnect() is now implemented inline in the header

json SoloMiningClient::getBlockTemplate() {
    std::lock_guard<std::mutex> lock(client_mutex);
    if (!rpc_client) {
        return json();
    }
    return rpc_client->getBlockTemplate("");
}

json SoloMiningClient::getBlockTemplate(const std::string& wallet_address) {
    std::lock_guard<std::mutex> lock(client_mutex);
    if (!rpc_client || wallet_address.empty()) {
        return json();
    }
    return rpc_client->getBlockTemplate(wallet_address);
}

bool SoloMiningClient::submitSolution(
    const std::string& proposal_id,
    const Pow& pow_solution,
    const Digest& solution_hash,
    const json& template_obj) {
    
    std::lock_guard<std::mutex> lock(client_mutex);
    
    if (!rpc_client) {
        return false;
    }
    
    if (template_obj.is_null() || template_obj.empty()) {
        std::cout << "[SUBMIT] ERROR: No template provided for proposal " << proposal_id << std::endl;
        return false;
    }
    
    // Check if template is stale by comparing prev_block with current tip
    std::string tip_digest = rpc_client->getTipDigest();
    
    std::string prev_block;
    if (template_obj.contains("metadata") && !template_obj["metadata"].is_null()) {
        json metadata = template_obj["metadata"];
        if (metadata.contains("prevBlock") && !metadata["prevBlock"].is_null()) {
            prev_block = metadata.value("prevBlock", "");
        } else if (metadata.contains("prev_block") && !metadata["prev_block"].is_null()) {
            prev_block = metadata.value("prev_block", "");
        }
    }

    if (!prev_block.empty() && tip_digest != prev_block) {
        std::cout << "[SUBMIT] " << Color::RED << "STALE: Template stale (new block arrived)" << Color::RESET << std::endl;
        return false;
    }
    
    // Extract the block from template
    if (!template_obj.contains("block") || template_obj["block"].is_null()) {
        std::cout << "[SUBMIT] ERROR: Template missing 'block' field" << std::endl;
        return false;
    }
    
    json block_obj = template_obj["block"];
    
    if (!block_obj.contains("kernel") || block_obj["kernel"].is_null()) {
        std::cout << "[SUBMIT] ERROR: Block missing 'kernel' field" << std::endl;
        return false;
    }

    json kernel_obj = block_obj["kernel"];
    if (!kernel_obj.contains("appendix") || kernel_obj["appendix"].is_null()) {
        std::cout << "[SUBMIT] ERROR: Block kernel missing 'appendix' field" << std::endl;
        return false;
    }
    if (!kernel_obj["appendix"].is_array()) {
        std::cout << "[SUBMIT] ERROR: Block kernel 'appendix' is not an array" << std::endl;
        return false;
    }
    if (kernel_obj["appendix"].empty()) {
        std::cout << "[SUBMIT] ERROR: Block kernel 'appendix' is empty" << std::endl;
        return false;
    }
    
    json pow_json = powToRpcFormat(pow_solution, solution_hash);
    
    json response = rpc_client->submitBlock(block_obj, pow_json);
    
    if (!response.empty() && response.contains("result")) {
        if (response["result"].is_boolean()) {
            bool success = response["result"].get<bool>();
            if (!success) {
                std::cout << "[SUBMIT] " << Color::RED << "REJECTED: Block rejected" << Color::RESET << std::endl;
            }
            return success;
        }
        if (response["result"].contains("success")) {
            bool success = response["result"]["success"].get<bool>();
            if (!success) {
                std::cout << "[SUBMIT] " << Color::RED << "REJECTED: Block rejected" << Color::RESET << std::endl;
            }
            return success;
        }
        return true;
    }
    
    if (response.contains("error") && !response["error"].is_null()) {
        json error = response["error"];
        std::string error_reason = "Unknown error";
        
        if (error.contains("data") && !error["data"].is_null()) {
            json error_data = error["data"];
            if (error_data.is_object() && !error_data.empty()) {
                auto it = error_data.begin();
                if (it != error_data.end() && it.value().is_string()) {
                    error_reason = it.value().get<std::string>();
                } else if (it != error_data.end()) {
                    error_reason = it.value().dump();
                }
            } else if (error_data.is_string()) {
                error_reason = error_data.get<std::string>();
            }
        } else if (error.contains("message") && !error["message"].is_null()) {
            error_reason = error.value("message", "Unknown error");
        }
        
        std::cout << "[SUBMIT] " << Color::RED << "REJECTED: " << error_reason << Color::RESET << std::endl;
        
        // Store error reason for InvalidBlock detection
        {
            std::lock_guard<std::mutex> lock(error_mutex);
            last_error_reason = error_reason;
        }
    } else {
        std::cout << "[SUBMIT] " << Color::RED << "REJECTED: No response from server" << Color::RESET << std::endl;
    }
    
    return false;
}

std::string SoloMiningClient::getTipDigest() {
    std::lock_guard<std::mutex> lock(client_mutex);
    if (!rpc_client) {
        return "";
    }
    return rpc_client->getTipDigest();
}

std::string SoloMiningClient::getLastError() const {
    std::lock_guard<std::mutex> lock(error_mutex);
    return last_error_reason;
}

uint64_t SoloMiningClient::getChainHeight() {
    std::lock_guard<std::mutex> lock(client_mutex);
    if (!rpc_client) {
        return 0;
    }
    return rpc_client->getChainHeight();
}

// ============================================================================
// ConnectionMultiplexer Implementation
// ============================================================================

ConnectionMultiplexer::ConnectionMultiplexer() {
    stats.last_job_time = std::chrono::steady_clock::now();
    stats.last_submission_time = std::chrono::steady_clock::now();
}

ConnectionMultiplexer::~ConnectionMultiplexer() {
    shutdown();
}

bool ConnectionMultiplexer::initialize(const std::string& ep, const std::string& wallet, 
                                       const std::string& stratum_password) {
    if (initialized.load()) {
        return true;
    }
    
    endpoint = ep;
    wallet_address = wallet;
    
    // Detect mining mode from endpoint URL
    MiningMode mode = detect_mining_mode(endpoint);
    mining_mode = mode;
    
    // Create the appropriate mining client based on mode
    if (mode == MiningMode::Stratum) {
        std::string host;
        int port;
        bool use_ssl = false;
        if (parse_stratum_url(endpoint, host, port, use_ssl)) {
            StratumConfig config;
            config.host = host;
            config.port = port;
            config.use_ssl = use_ssl;
            config.address = wallet_address;
            config.name = g_miner_worker_name.empty() ? "xnt-miner" : g_miner_worker_name;
            config.password = stratum_password;
            config.agent = "xnt-gpu-miner/1.0";
            client = std::make_unique<StratumClient>(config);
            std::cout << "Attempting to connect to stratum server at " << endpoint << "..." << std::endl;
        } else {
            std::cerr << Color::RED << "Invalid stratum URL: " << endpoint << Color::RESET << std::endl;
            return false;
        }
    } else {
        client = std::make_unique<SoloMiningClient>(endpoint);
        std::cout << "Attempting to connect to RPC server at " << endpoint << "..." << std::endl;
    }
    
    // Initial connection attempt
    int retry_count = 0;
    bool conn_success = false;
    
    while (!conn_success && !stop_mining && retry_count < 10) {
        if (client->connect()) {
            conn_success = true;
            const char* server_type = (mining_mode == MiningMode::Stratum) ? "stratum server" : "RPC server";
            std::cout << Color::GREEN << "Successfully connected to " << server_type << "!" << Color::RESET << std::endl;
            break;
        }
        
        retry_count++;
        std::cout << "Connection attempt " << retry_count << " failed, retrying in 10 seconds..." << std::endl;
        
        for (int i = 0; i < 10 && !stop_mining; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    
    if (!conn_success) {
        std::cout << Color::RED << "Failed to connect to RPC server after " << retry_count << " attempts" << Color::RESET << std::endl;
        return false;
    }
    
    connected.store(true);
    running.store(true);
    initialized.store(true);
    
    // Start worker threads
    job_broadcaster_thread = std::thread(&ConnectionMultiplexer::jobBroadcasterLoop, this);
    solution_submitter_thread = std::thread(&ConnectionMultiplexer::solutionSubmitterLoop, this);
    health_monitor_thread = std::thread(&ConnectionMultiplexer::healthMonitorLoop, this);
    
    // Start tip monitor for solo mode (fast stale detection)
    if (mining_mode == MiningMode::Solo) {
        tip_monitor_thread = std::thread(&ConnectionMultiplexer::tipMonitorLoop, this);
    }
    
    std::cout << Color::GREEN << "Connection multiplexer started" << Color::RESET << std::endl;
    
    return true;
}

void ConnectionMultiplexer::shutdown() {
    if (!initialized.load()) {
        return;
    }
    
    running.store(false);
    connected.store(false);
    
    // Wake up the solution submitter
    submission_cv.notify_all();
    
    // Wait for threads to finish
    if (job_broadcaster_thread.joinable()) {
        job_broadcaster_thread.join();
    }
    if (solution_submitter_thread.joinable()) {
        solution_submitter_thread.join();
    }
    if (health_monitor_thread.joinable()) {
        health_monitor_thread.join();
    }
    if (tip_monitor_thread.joinable()) {
        tip_monitor_thread.join();
    }
    
    // Clear workers
    {
        std::unique_lock<std::shared_mutex> lock(workers_mutex);
        workers.clear();
    }
    
    // Clear submission queue
    {
        std::lock_guard<std::mutex> lock(submission_mutex);
        while (!submission_queue.empty()) {
            auto& submission = submission_queue.front();
            submission->result_promise->set_value(false);
            submission_queue.pop();
        }
    }
    
    client.reset();
    initialized.store(false);
    
    std::cout << "[Multiplexer] Shutdown complete" << std::endl;
}

GpuWorkerHandle* ConnectionMultiplexer::registerWorker(int gpu_id, GpuResources* resources) {
    std::unique_lock<std::shared_mutex> lock(workers_mutex);
    
    if (!resources->event_handler) {
        resources->event_handler = std::make_unique<EventHandler>();
    }
    auto handle = std::make_unique<GpuWorkerHandle>(gpu_id, resources);
    GpuWorkerHandle* ptr = handle.get();
    workers.push_back(std::move(handle));
    
    std::cout << "[Multiplexer] Registered GPU " << gpu_id << " worker" << std::endl;
    
    return ptr;
}

void ConnectionMultiplexer::unregisterWorker(GpuWorkerHandle* handle) {
    if (!handle) return;
    
    // Save GPU ID before erasing (handle will be deleted by unique_ptr)
    int gpu_id = handle->gpu_id;
    
    std::unique_lock<std::shared_mutex> lock(workers_mutex);
    
    handle->active = false;
    
    workers.erase(
        std::remove_if(workers.begin(), workers.end(),
            [handle](const std::unique_ptr<GpuWorkerHandle>& h) {
                return h.get() == handle;
            }),
        workers.end());
    
    std::cout << "[Multiplexer] Unregistered GPU " << gpu_id << " worker" << std::endl;
}

size_t ConnectionMultiplexer::getActiveWorkerCount() const {
    std::shared_lock<std::shared_mutex> lock(workers_mutex);
    size_t count = 0;
    for (const auto& w : workers) {
        if (w && w->active) {
            count++;
        }
    }
    return count;
}

std::future<bool> ConnectionMultiplexer::submitSolution(
    int gpu_id,
    const std::string& proposal_id,
    const Pow& pow_solution,
    const Digest& solution_hash,
    const json& template_obj) {
    
    auto submission = std::make_unique<SolutionSubmission>(
        gpu_id, proposal_id, pow_solution, solution_hash, template_obj);
    
    auto future = submission->result_promise->get_future();
    
    {
        std::lock_guard<std::mutex> lock(submission_mutex);
        submission_queue.push(std::move(submission));
    }
    submission_cv.notify_one();
    
    // Update worker stats
    {
        std::shared_lock<std::shared_mutex> lock(workers_mutex);
        for (const auto& worker : workers) {
            if (worker && worker->gpu_id == gpu_id) {
                worker->submissions_queued++;
                break;
            }
        }
    }
    
    return future;
}

bool ConnectionMultiplexer::isConnected() const {
    return connected.load() && client && client->is_connected();
}

bool ConnectionMultiplexer::isTemplateStale(const json& template_obj) const {
    if (!client || !client->is_connected()) {
        return false; // Can't check if not connected
    }
    
    if (template_obj.is_null() || template_obj.empty()) {
        return false; // Empty template, can't determine staleness
    }
    
    if (!template_obj.contains("metadata") || template_obj["metadata"].is_null()) {
        return false; // No metadata, can't check
    }
    
    json metadata = template_obj["metadata"];
    std::string prev_block;
    if (metadata.contains("prevBlock") && !metadata["prevBlock"].is_null()) {
        prev_block = metadata.value("prevBlock", "");
    } else if (metadata.contains("prev_block") && !metadata["prev_block"].is_null()) {
        prev_block = metadata.value("prev_block", "");
    }
    
    if (prev_block.empty()) {
        return false; // No prev_block, can't determine staleness
    }
    
    std::string tip_digest = client->getTipDigest();
    if (tip_digest.empty()) {
        return false; // Can't get tip, assume not stale
    }
    
    return prev_block != tip_digest;
}

void ConnectionMultiplexer::updateAllWorkersConnectionState(bool is_connected) {
    std::shared_lock<std::shared_mutex> lock(workers_mutex);
    for (const auto& worker : workers) {
        if (worker && worker->resources) {
            worker->resources->gpu_node_connected = is_connected;
        }
    }
}

// ============================================================================
// Thread Loops
// ============================================================================

void ConnectionMultiplexer::jobBroadcasterLoop() {
    try {
        LOG_DEBUG("[JobBroadcaster] Started");

        auto last_poll_time = std::chrono::steady_clock::now();

        // Wait for at least one worker to register
        while (running.load()) {
            if (getActiveWorkerCount() > 0) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::cout << "[JobBroadcaster] Block proposal fetcher started" << std::endl;

        while (running.load() && !stop_mining) {
            // Check connection
            if (!client || !client->is_connected()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }

            auto now = std::chrono::steady_clock::now();
            auto time_since_poll = std::chrono::duration_cast<std::chrono::seconds>(
                now - last_poll_time).count();

            bool should_poll = false;

            // Check if it's time to poll
            if (time_since_poll >= g_fetch_interval_sec) {
                should_poll = true;
            }

            // Also poll if we have no template yet or if we're composing (waiting for new proposal)
            {
                std::lock_guard<std::mutex> lock(job_mutex);
                if (last_template_id.empty() || composing_new_block.load()) {
                    should_poll = true;
                }
            }

            if (should_poll) {
                json template_response = client->getBlockTemplate(wallet_address);
                last_poll_time = std::chrono::steady_clock::now();

                if (!template_response.empty() && template_response.contains("result")) {
                    json result = template_response["result"];

                    if (!result.contains("template") || result["template"].is_null()) {
                        // Template null - node is still composing, keep waiting
                        if (composing_new_block.load()) {
                            // Poll more frequently during composition
                            std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        } else {
                            std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        }
                        continue;
                    }

                    json template_obj = result["template"];
                    if (!template_obj.contains("metadata") || template_obj["metadata"].is_null()) {
                        // Invalid template - node still composing
                        if (composing_new_block.load()) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        } else {
                            std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        }
                        continue;
                    }

                    json metadata = template_obj["metadata"];
                    std::string template_id = metadata.contains("digest")
                        ? metadata.value("digest", "") : "";

                    bool is_new_template = false;
                    {
                        std::lock_guard<std::mutex> lock(job_mutex);
                        // When composing, also process same template_id to allow second-proposal count
                        bool template_changed = !template_id.empty() && template_id != last_template_id;
                        bool composing_same_template = composing_new_block.load() && !template_id.empty() 
                            && template_id == last_template_id && proposals_seen_for_tip.load() == 1;
                        if (template_changed || composing_same_template) {
                            // Verify prev_block matches tip
                            std::string prev_block;
                            if (metadata.contains("prevBlock")) {
                                prev_block = metadata.value("prevBlock", "");
                            } else if (metadata.contains("prev_block")) {
                                prev_block = metadata.value("prev_block", "");
                            }

                            std::string tip_digest = client->getTipDigest();

                            if (prev_block.empty() || prev_block == tip_digest) {
                                // After tip change, wait for second proposal with same prev_block
                                // First proposal often causes InvalidBlock errors (fixed in 503af36, dfffca9)
                                bool should_resume = true;
                                bool is_composing = composing_new_block.load();
                                
                                if (is_composing) {
                                    int count = proposals_seen_for_tip.fetch_add(1) + 1;
                                    
                                    if (count == 1) {
                                        // First proposal - record prev_block but don't resume yet
                                        first_proposal_prev_block = prev_block;
                                        should_resume = false;
                                        bool in_recovery = recovery_mode.load();
                                        std::cout << "[JobBroadcaster] " << Color::YELLOW 
                                                  << "First proposal received (prev_block: " 
                                                  << prev_block.substr(0, 16) << "...), waiting for second..."
                                                  << (in_recovery ? " (recovery mode)" : "")
                                                  << Color::RESET << std::endl;
                                    } else if (count == 2 && prev_block == first_proposal_prev_block) {
                                        // Second proposal with same prev_block - safe to resume
                                        composing_new_block.store(false);
                                        recovery_mode.store(false);
                                        std::cout << "[JobBroadcaster] " << Color::GREEN 
                                                  << "Stable proposal confirmed (2nd with same prev_block), resuming mining" 
                                                  << Color::RESET << std::endl;
                                        should_resume = true;
                                    } else if (count >= 2 && prev_block != first_proposal_prev_block) {
                                        // Prev_block changed, reset counter
                                        proposals_seen_for_tip.store(1);
                                        first_proposal_prev_block = prev_block;
                                        should_resume = false;
                                        std::cout << "[JobBroadcaster] " << Color::YELLOW 
                                                  << "Prev_block changed, resetting proposal count" 
                                                  << Color::RESET << std::endl;
                                    } else {
                                        should_resume = false;
                                    }
                                }

                                // Always update template_id and tip tracking
                                last_template_id = template_id;
                                current_tip_digest = tip_digest;
                                
                                if (should_resume) {
                                    is_new_template = true;
                                    stats.total_jobs_fetched++;
                                    {
                                        std::lock_guard<std::mutex> time_lock(stats.time_mutex);
                                        stats.last_job_time = std::chrono::steady_clock::now();
                                    }
                                }
                            }
                        }
                    }

                    if (is_new_template) {
                        // Resume workers when we have a stable new proposal
                        {
                            std::shared_lock<std::shared_mutex> lock(workers_mutex);
                            for (const auto& worker : workers) {
                                if (worker && worker->active && worker->resources) {
                                    worker->resources->set_paused(false);
                                }
                            }
                        }
                        broadcastJobToWorkers(template_response);
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        LOG_DEBUG("[JobBroadcaster] Stopped");
    } catch (const std::exception& e) {
        std::cerr << "[JobBroadcaster] Exception: " << e.what() << std::endl;
        running.store(false);
        connected.store(false);
    } catch (...) {
        std::cerr << "[JobBroadcaster] Unknown exception" << std::endl;
        running.store(false);
        connected.store(false);
    }
}

void ConnectionMultiplexer::broadcastJobToWorkers(const json& job) {
    std::shared_lock<std::shared_mutex> lock(workers_mutex);
    
    std::string job_data = job.dump();
    
    for (const auto& worker : workers) {
        if (worker && worker->active && worker->resources && worker->resources->event_handler) {
            worker->resources->event_handler->postEvent(
                EventType::NEW_PUZZLE, "", job_data);
            worker->jobs_received++;
        }
    }
    
    stats.total_jobs_broadcast++;
}

void ConnectionMultiplexer::solutionSubmitterLoop() {
    try {
        LOG_DEBUG("[SolutionSubmitter] Started");

        while (running.load() && !stop_mining) {
            std::unique_ptr<SolutionSubmission> submission;

            {
                std::unique_lock<std::mutex> lock(submission_mutex);
                submission_cv.wait_for(lock, std::chrono::milliseconds(100), [this] {
                    return !submission_queue.empty() || !running.load() || stop_mining;
                });

                if ((!running.load() || stop_mining) && submission_queue.empty()) {
                    break;
                }

                if (!submission_queue.empty()) {
                    submission = std::move(submission_queue.front());
                    submission_queue.pop();
                }
            }

            if (submission) {
                bool result = processSubmission(*submission);
                submission->result_promise->set_value(result);
            }
        }

        LOG_DEBUG("[SolutionSubmitter] Stopped");
    } catch (const std::exception& e) {
        std::cerr << "[SolutionSubmitter] Exception: " << e.what() << std::endl;
        running.store(false);
        connected.store(false);
    } catch (...) {
        std::cerr << "[SolutionSubmitter] Unknown exception" << std::endl;
        running.store(false);
        connected.store(false);
    }
}

bool ConnectionMultiplexer::processSubmission(SolutionSubmission& submission) {
    if (!client || !client->is_connected()) {
        std::cout << "[GPU " << submission.gpu_id << "] " << Color::RED 
                  << "Cannot submit - not connected to node" << Color::RESET << std::endl;
        return false;
    }
    
    // Reject submissions during composition gap (no valid proposal exists yet)
    if (composing_new_block.load()) {
        std::cout << "[SUBMIT] " << Color::YELLOW
                  << "STALE: Node composing new block, no valid proposal yet" 
                  << Color::RESET << std::endl;
        return false;
    }
    
    stats.total_solutions_submitted++;
    bool accepted = false;
    bool is_invalid_block = false;  // Declare outside try block for use after catch
    try {
        // For solo mode, refresh the template on submission to ensure
        // the appendix claims are current for this proposal.
        if (mining_mode == MiningMode::Solo) {
            json template_response = client->getBlockTemplate(wallet_address);
            if (template_response.empty() || !template_response.contains("result")) {
                std::cout << "[SUBMIT] " << Color::YELLOW
                          << "No template response on refresh, dropping submission" 
                          << Color::RESET << std::endl;
                return false;
            }
            json result = template_response["result"];
            if (!result.contains("template") || result["template"].is_null()) {
                std::cout << "[SUBMIT] " << Color::YELLOW
                          << "Template missing on refresh, dropping submission" 
                          << Color::RESET << std::endl;
                return false;
            }
            json template_obj = result["template"];
            if (!template_obj.contains("metadata") || template_obj["metadata"].is_null()) {
                std::cout << "[SUBMIT] " << Color::YELLOW
                          << "Template metadata missing on refresh, dropping submission" 
                          << Color::RESET << std::endl;
                return false;
            }
            json metadata = template_obj["metadata"];
            std::string template_id = metadata.contains("digest")
                ? metadata.value("digest", "") : "";

            auto normalize_digest = [](std::string value) {
                if (value.size() >= 2 && value[0] == '0' && (value[1] == 'x' || value[1] == 'X')) {
                    value = value.substr(2);
                }
                for (auto& ch : value) {
                    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                }
                return value;
            };

            if (template_id.empty()) {
                std::cout << "[SUBMIT] " << Color::YELLOW
                          << "Template digest missing on refresh, dropping submission" 
                          << Color::RESET << std::endl;
                return false;
            }

            if (normalize_digest(template_id) != normalize_digest(submission.proposal_id)) {
                std::cout << "[SUBMIT] " << Color::YELLOW
                          << "Template digest mismatch on refresh, dropping submission" 
                          << Color::RESET << std::endl;
                return false;
            }

            // Additional validation: Ensure the block body (transaction kernel) hasn't changed
            // by comparing a hash of the transaction kernel structure
            json fresh_block = template_obj["block"];
            json original_block = submission.template_obj["block"];
            
            if (fresh_block.contains("kernel") && original_block.contains("kernel")) {
                json fresh_kernel = fresh_block["kernel"];
                json original_kernel = original_block["kernel"];
                
                // Compare transaction kernel body fields that affect the MAST hash
                // If these differ, the appendix claims will be invalid
                auto kernel_body_hash = [](const json& kernel) -> std::string {
                    if (!kernel.contains("body") || kernel["body"].is_null()) {
                        return "";
                    }
                    json body = kernel["body"];
                    // Create a simple hash from key transaction kernel fields
                    std::ostringstream oss;
                    if (body.contains("transactionKernel")) {
                        json tk = body["transactionKernel"];
                        if (tk.contains("fee")) oss << tk["fee"].dump();
                        if (tk.contains("coinbase")) oss << tk["coinbase"].dump();
                        if (tk.contains("timestamp")) oss << tk["timestamp"].dump();
                        if (tk.contains("mutatorSetHash")) oss << tk["mutatorSetHash"].dump();
                    }
                    return oss.str();
                };
                
                std::string fresh_hash = kernel_body_hash(fresh_kernel);
                std::string original_hash = kernel_body_hash(original_kernel);
                
                if (fresh_hash != original_hash) {
                    std::cout << "[SUBMIT] " << Color::YELLOW
                              << "Transaction kernel body changed on refresh, dropping submission" 
                              << Color::RESET << std::endl;
                    return false;
                }
            }

            // Always use the freshly fetched template for submission
            submission.template_obj = template_obj;
        }

        accepted = client->submitSolution(
            submission.proposal_id,
            submission.pow_solution,
            submission.solution_hash,
            submission.template_obj);
            
        // Check if the rejection was due to InvalidBlock
        if (!accepted && mining_mode == MiningMode::Solo) {
            SoloMiningClient* solo_client = dynamic_cast<SoloMiningClient*>(client.get());
            if (solo_client) {
                std::string error = solo_client->getLastError();
                std::string error_lower = error;
                std::transform(error_lower.begin(), error_lower.end(), error_lower.begin(), ::tolower);
                if (error_lower.find("invalidblock") != std::string::npos || 
                    error_lower.find("invalid block") != std::string::npos) {
                    is_invalid_block = true;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[Submit] Exception: " << e.what() << std::endl;
        accepted = false;
    } catch (...) {
        std::cerr << "[Submit] Unknown exception" << std::endl;
        accepted = false;
    }
    
    if (accepted) {
        stats.total_solutions_accepted++;
        // Reset rejection counter on success
        consecutive_rejections.store(0);
        recovery_mode.store(false);
    } else {
        stats.total_solutions_rejected++;
        
        // Only invalidate on actual InvalidBlock errors, not InsufficientWork
        if (is_invalid_block && !composing_new_block.load()) {
            std::cout << "[SUBMIT] " << Color::YELLOW
                      << "InvalidBlock error detected, invalidating template and forcing refresh" 
                      << Color::RESET << std::endl;
            
            // Clear template to force immediate refresh
            {
                std::lock_guard<std::mutex> lock(job_mutex);
                last_template_id.clear();
            }
            
            // Set recovery mode - wait for second proposal (same as normal mode)
            // First proposal after InvalidBlock is still unstable
            recovery_mode.store(true);
            proposals_seen_for_tip.store(0);
            first_proposal_prev_block.clear();
            composing_new_block.store(true);
            
            // Pause workers temporarily until new template arrives
            {
                std::shared_lock<std::shared_mutex> lock(workers_mutex);
                for (const auto& worker : workers) {
                    if (worker && worker->active && worker->resources) {
                        worker->resources->set_paused(true);
                    }
                }
            }
            
            // Reset counter
            consecutive_rejections.store(0);
        } else {
            // Reset counter on success or non-InvalidBlock errors
            consecutive_rejections.store(0);
        }
    }
    
    {
        std::lock_guard<std::mutex> time_lock(stats.time_mutex);
        stats.last_submission_time = std::chrono::steady_clock::now();
    }
    
    return accepted;
}

void ConnectionMultiplexer::healthMonitorLoop() {
    try {
        LOG_DEBUG("[HealthMonitor] Started");

        while (running.load() && !stop_mining) {
            std::this_thread::sleep_for(std::chrono::seconds(5));

            if (!running.load() || stop_mining) break;

            // Check connection health
            if (client) {
                bool was_connected = connected.load();
                bool is_now_connected = client->is_connected();

                if (was_connected && !is_now_connected) {
                    // Lost connection
                    std::cout << "[HealthMonitor] " << Color::YELLOW
                              << "Connection lost, attempting reconnection..." << Color::RESET << std::endl;

                    connected.store(false);
                    updateAllWorkersConnectionState(false);

                    if (attemptReconnection()) {
                        connected.store(true);
                        updateAllWorkersConnectionState(true);
                    }
                } else if (!was_connected && is_now_connected) {
                    // Connection restored (shouldn't happen normally)
                    connected.store(true);
                    updateAllWorkersConnectionState(true);
                }
            }

            // Check for stale jobs
            {
                std::lock_guard<std::mutex> time_lock(stats.time_mutex);
                auto now = std::chrono::steady_clock::now();
                auto time_since_job = std::chrono::duration_cast<std::chrono::seconds>(
                    now - stats.last_job_time).count();

                if (time_since_job > JOB_STALENESS_THRESHOLD_SEC && connected.load()) {
                    LOG_DEBUG("[HealthMonitor] Jobs stale (" << time_since_job << "s), will refresh on next poll");
                }
            }
        }

        LOG_DEBUG("[HealthMonitor] Stopped");
    } catch (const std::exception& e) {
        std::cerr << "[HealthMonitor] Exception: " << e.what() << std::endl;
        running.store(false);
        connected.store(false);
    } catch (...) {
        std::cerr << "[HealthMonitor] Unknown exception" << std::endl;
        running.store(false);
        connected.store(false);
    }
}

bool ConnectionMultiplexer::attemptReconnection() {
    int attempt = 0;
    
    while (running.load() && !stop_mining && attempt < MAX_CONSECUTIVE_RECONNECTIONS) {
        attempt++;
        
        int delay = std::min(MIN_RECONNECT_DELAY_SEC * (1 << (attempt - 1)), 
                            MAX_RECONNECT_DELAY_SEC);
        
        std::cout << "[Multiplexer] Reconnection attempt " << attempt 
                  << " in " << delay << "s..." << std::endl;
        
        for (int i = 0; i < delay && running.load() && !stop_mining; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        if (!running.load() || stop_mining) break;
        
        if (client && client->reconnect()) {
            std::cout << Color::GREEN << "[Multiplexer] Reconnected successfully!" 
                      << Color::RESET << std::endl;
            
            stats.reconnection_count++;
            return true;
        }
    }
    
    std::cout << Color::RED << "[Multiplexer] Failed to reconnect after " 
              << attempt << " attempts" << Color::RESET << std::endl;
    return false;
}

void ConnectionMultiplexer::tipMonitorLoop() {
    try {
        LOG_DEBUG("[TipMonitor] Started");
        
        // Check tip every 500ms for fast stale detection
        const auto TIP_CHECK_INTERVAL = std::chrono::milliseconds(500);
        
        while (running.load() && !stop_mining) {
            std::this_thread::sleep_for(TIP_CHECK_INTERVAL);
            
            if (!client || !client->is_connected()) {
                continue;
            }
            
            // Get current tip from node
            std::string new_tip = client->getTipDigest();
            if (new_tip.empty()) {
                continue;
            }
            
            bool tip_changed = false;
            {
                std::lock_guard<std::mutex> lock(job_mutex);
                if (!current_tip_digest.empty() && current_tip_digest != new_tip) {
                    tip_changed = true;
                    std::string short_old = current_tip_digest.length() > 16 ? 
                        current_tip_digest.substr(0, 8) + "..." + current_tip_digest.substr(current_tip_digest.length() - 8) : 
                        current_tip_digest;
                    std::string short_new = new_tip.length() > 16 ? 
                        new_tip.substr(0, 8) + "..." + new_tip.substr(new_tip.length() - 8) : 
                        new_tip;
                    std::cout << "[TipMonitor] " << Color::YELLOW << "New block detected!" << Color::RESET
                              << " Tip: " << short_old << " -> " << short_new << std::endl;
                }
                current_tip_digest = new_tip;
            }
            
            if (tip_changed) {
                // Check if we're waiting for second proposal and new tip matches prev_block (dfffca9)
                bool should_reset = true;
                {
                    std::lock_guard<std::mutex> lock(job_mutex);
                    int current_count = proposals_seen_for_tip.load();
                    if (current_count == 1 && !first_proposal_prev_block.empty()) {
                        if (new_tip == first_proposal_prev_block) {
                            // New tip matches the prev_block we're waiting for - keep waiting for second
                            should_reset = false;
                            std::cout << "[TipMonitor] " << Color::YELLOW 
                                      << "New block matches waiting prev_block, continuing to wait for second proposal..." 
                                      << Color::RESET << std::endl;
                        } else {
                            std::cout << "[TipMonitor] " << Color::YELLOW 
                                      << "Prev_block changed, resetting proposal wait..." 
                                      << Color::RESET << std::endl;
                        }
                    }
                }
                
                if (should_reset) {
                    composing_new_block.store(true);
                    proposals_seen_for_tip.store(0);
                    first_proposal_prev_block.clear();
                }
                
                // Immediately pause all workers - their templates are now stale
                {
                    std::shared_lock<std::shared_mutex> lock(workers_mutex);
                    for (const auto& worker : workers) {
                        if (worker && worker->active && worker->resources) {
                            worker->resources->set_paused(true);
                            if (worker->resources->event_handler) {
                                worker->resources->event_handler->postEvent(EventType::TIP_CHANGED);
                            }
                        }
                    }
                }
                
                if (should_reset) {
                    std::lock_guard<std::mutex> lock(job_mutex);
                    last_template_id.clear();
                }
                
                if (should_reset) {
                    std::cout << "[TipMonitor] " << Color::YELLOW 
                              << "Node composing new block, waiting for stable proposal..." 
                              << Color::RESET << std::endl;
                }
                
                // Don't fetch here - let jobBroadcasterLoop handle it with retries
                // This ensures we wait until a valid proposal is ready
            }
        }
        
        LOG_DEBUG("[TipMonitor] Stopped");
    } catch (const std::exception& e) {
        std::cerr << "[TipMonitor] Exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[TipMonitor] Unknown exception" << std::endl;
    }
}

// --- main.cu ---
#include "mining.cuh"
#include "connection_multiplexer.cuh"
#include "mining_client.h"
#include "network.cuh"
#include "rpc_client.cuh"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

void print_usage(const char* program_name) {
    std::cerr << "\n" << Color::BOLD << "Usage:" << Color::RESET << std::endl;
    std::cerr << "  " << program_name << " [OPTIONS] -w, --wallet ADDRESS\n" << std::endl;
    
    std::cerr << Color::BOLD << "Required:" << Color::RESET << std::endl;
    std::cerr << "  -w, --wallet ADDRESS  Your Neptune wallet address\n" << std::endl;
    
    std::cerr << Color::BOLD << "Solo Mining (default):" << Color::RESET << std::endl;
    std::cerr << "  --rpc-url URL         RPC endpoint URL (default: http://127.0.0.1:9897)" << std::endl;
    std::cerr << "  -H, --host HOST       RPC host" << std::endl;
    std::cerr << "  -p, --port PORT       RPC port\n" << std::endl;
    
    std::cerr << Color::BOLD << "Pool Mining (Stratum):" << Color::RESET << std::endl;
    std::cerr << "  --stratum URL         Stratum pool URL (e.g., stratum://pool.example.com:3333)" << std::endl;
    std::cerr << "  --stratum-pass PASS   Stratum password (default: x)" << std::endl;
    std::cerr << "  --stratum-worker NAME Stratum worker name (default: xnt-miner)\n" << std::endl;
    
    std::cerr << Color::BOLD << "General Options:" << Color::RESET << std::endl;
    std::cerr << "  -d, --device ID       Use specific GPU device ID (default: all GPUs)" << std::endl;
    std::cerr << "  --rpc-url URL         RPC endpoint URL (default: http://127.0.0.1:9897)" << std::endl;
    std::cerr << "  --test-mode           Enable test mode (100,000x easier target)" << std::endl;
    std::cerr << "  --benchmark           Run mining benchmark (saves/loads test template)" << std::endl;
    std::cerr << "  --fetch-interval SEC  Job fetch interval in seconds (default: 5)" << std::endl;
    std::cerr << "  --batch N             Nonces per kernel (0=auto, e.g. 20000000 for 20M)" << std::endl;
    std::cerr << "                        Or set XNT_BATCH_SIZE env for quick tuning" << std::endl;
    std::cerr << "  --blocks N            Blocks per grid (0=auto; 680=fast miner config)" << std::endl;
    std::cerr << "  --blocks-sweep        Benchmark sweep 512-8192 blocks (with --benchmark)" << std::endl;
    std::cerr << "  XNT_USE_PHASE_SPLIT=0 Disable phase split (default ON for HIGH_VRAM)" << std::endl;
    std::cerr << "  -h, --help            Show this help message\n" << std::endl;
    
    std::cerr << Color::BOLD << "Examples:" << Color::RESET << std::endl;
    std::cerr << "  " << Color::DIM << "# Solo mining to local node" << Color::RESET << std::endl;
    std::cerr << "  " << program_name << " -w nolgam..." << std::endl;
    std::cerr << std::endl;
    std::cerr << "  " << Color::DIM << "# Solo mining to remote node" << Color::RESET << std::endl;
    std::cerr << "  " << program_name << " -w nolgam... --rpc-url http://192.168.1.100:9897" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  " << Color::DIM << "# Pool mining via stratum" << Color::RESET << std::endl;
    std::cerr << "  " << program_name << " -w nolgam... --stratum stratum://pool.example.com:3333" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  " << Color::DIM << "# Benchmark mining speed" << Color::RESET << std::endl;
    std::cerr << "  " << program_name << " -w nolgam... --benchmark" << std::endl;
    std::cerr << std::endl;
}

void print_system_info() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess) {
        std::cerr << Color::RED << "CUDA Error: " << cudaGetErrorString(error) << Color::RESET << std::endl;
        return;
    }
    
    std::cout << Color::BOLD << "System Information:" << Color::RESET << std::endl;
    std::cout << "  CUDA Devices: " << device_count << std::endl;
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        size_t vram_gb = prop.totalGlobalMem / (1024ULL * 1024ULL * 1024ULL);
        
        std::cout << "  GPU " << i << ": " << prop.name 
                  << " (" << vram_gb << " GB)" << std::endl;
    }
    std::cout << std::endl;
}

// ============================================================================
// Benchmark Functions
// ============================================================================
// Benchmark mode allows testing mining performance without requiring an active
// RPC connection. It saves a block template to disk and reuses it for
// consistent performance measurements.

const std::string BENCHMARK_FILE = "benchmark_template.json";

/// Save a block template to disk for offline benchmark testing.
/// 
/// @param template_response JSON response containing the block template
/// @return true if template was successfully saved, false otherwise
bool saveBenchmarkTemplate(const json& template_response) {
    std::ofstream file(BENCHMARK_FILE);
    if (!file.is_open()) {
        std::cerr << Color::RED << "Error: Cannot create benchmark file: " << BENCHMARK_FILE << Color::RESET << std::endl;
        return false;
    }
    // Write formatted JSON (indented for readability)
    file << std::setw(2) << template_response << std::endl;
    file.close();
    std::cout << Color::GREEN << "Benchmark template saved to " << BENCHMARK_FILE << Color::RESET << std::endl;
    return true;
}

/// Load a previously saved block template from disk.
/// 
/// @param template_response Output parameter to receive the loaded template
/// @return true if template was successfully loaded, false if file doesn't exist or is invalid
bool loadBenchmarkTemplate(json& template_response) {
    std::ifstream file(BENCHMARK_FILE);
    if (!file.is_open()) {
        return false;
    }
    try {
        file >> template_response;
        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << Color::RED << "Error: Invalid benchmark file: " << e.what() << Color::RESET << std::endl;
        return false;
    }
}

/// Run mining benchmark to measure hash rate and validate solutions.
/// 
/// Benchmark mode operates in two phases:
/// 1. Template acquisition: Fetches a block template from the node (if not cached)
/// 2. Mining loop: Mines using an easier target (100000x) to find solutions quickly
///    and validates them using the same logic as the Rust node.
/// 
/// @param endpoint RPC endpoint URL (used only if template needs to be fetched)
/// @param gpu_id GPU device ID to use for mining (-1 for auto-select)
void runBenchmark(const std::string& endpoint, int gpu_id) {
    std::cout << Color::BOLD << "\n=== Mining Benchmark ===" << Color::RESET << std::endl;
    
    json template_response;
    bool template_exists = loadBenchmarkTemplate(template_response);
    
    if (!template_exists) {
        std::cout << "Benchmark template not found. Fetching from node..." << std::endl;
        
        // Wallet address is required only when fetching a new template from the node.
        // Once the template is saved, the benchmark can run offline.
        if (g_miner_wallet_address.empty()) {
            std::cerr << Color::RED << "Error: Wallet address is required to fetch template from node" << std::endl;
            std::cerr << "Please provide wallet address with -w/--wallet or ensure " << BENCHMARK_FILE << " exists" << Color::RESET << std::endl;
            return;
        }
        
        // Try to fetch a template from the node
        try {
            XntRpcClient rpc_client(endpoint);
            if (!rpc_client.testConnection()) {
                std::cerr << Color::RED << "Error: Cannot connect to node at " << endpoint << std::endl;
                std::cerr << "Please ensure the node is running or provide a valid RPC URL." << Color::RESET << std::endl;
                return;
            }
            
            template_response = rpc_client.getBlockTemplate(g_miner_wallet_address);
            if (template_response.empty() || !template_response.contains("result")) {
                std::cerr << Color::RED << "Error: Failed to fetch block template from node" << Color::RESET << std::endl;
                return;
            }
            
            if (!saveBenchmarkTemplate(template_response)) {
                return;
            }
        } catch (const std::exception& e) {
            std::cerr << Color::RED << "Error fetching template: " << e.what() << Color::RESET << std::endl;
            return;
        }
    } else {
        std::cout << "Using existing benchmark template from " << BENCHMARK_FILE << std::endl;
        std::cout << Color::GREEN << "No RPC connection required - running offline benchmark" << Color::RESET << std::endl;
    }
    
    // Extract template object from JSON response (handles both RPC and direct formats)
    json template_obj;
    if (template_response.contains("result") && template_response["result"].contains("template")) {
        template_obj = template_response["result"]["template"];
    } else if (template_response.contains("template")) {
        template_obj = template_response["template"];
    } else {
        std::cerr << Color::RED << "Error: Invalid template format" << Color::RESET << std::endl;
        return;
    }
    
    // Parse the template into a PowPuzzle structure for mining
    std::string template_json_str = template_response.dump();
    PowPuzzle puzzle = parsePowPuzzle(template_json_str);
    if (!puzzle.is_valid()) {
        std::cerr << Color::RED << "Error: Invalid puzzle in template" << Color::RESET << std::endl;
        return;
    }
    
    // Initialize GPU
    int device_id = (gpu_id >= 0) ? gpu_id : 0;
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << Color::RED << "CUDA Error: " << cudaGetErrorString(err) << Color::RESET << std::endl;
        return;
    }
    
    // Initialize GPU resources
    auto gpu_res = std::make_unique<GpuResources>(device_id);
    gpu_res->mining_mode = MiningMode::Solo;
    
    if (!preprocessPuzzle(puzzle, gpu_res.get())) {
        std::cerr << Color::RED << "Failed to initialize GPU resources" << Color::RESET << std::endl;
        return;
    }
    
    // Batch size: use global override if set, otherwise auto-detect for this GPU
    const uint64_t min_batch = 1;
    if (g_batch_size > 0) {
        gpu_res->optimal_max_nonces = std::max(g_batch_size, min_batch);
    } else {
        gpu_res->optimal_max_nonces = get_optimal_batch_size(device_id);
    }
    
    std::cout << "\n" << Color::BOLD << "Starting benchmark..." << Color::RESET << std::endl;
    std::cout << "  Batch size: " << gpu_res->optimal_max_nonces << " (" << (gpu_res->optimal_max_nonces / 1000000.0) << "M)" << std::endl;
    std::cout << "Press Ctrl+C to stop\n" << std::endl;
    
    // Benchmark configuration
    int BENCHMARK_DURATION_SEC = 10;        // Total benchmark duration
    if (const char* bench_env = std::getenv("XNT_BENCHMARK_SEC"); bench_env && bench_env[0] != '\0') {
        int v = std::atoi(bench_env);
        // Keep bounds sane so a typo doesn't run forever
        if (v > 0 && v <= 600) {
            BENCHMARK_DURATION_SEC = v;
        }
    }
    // Use GPU's optimal batch size for maximum performance
    // For RTX 5090 (256 SMs), this will be ~20M nonces, reducing kernel launch overhead
    uint64_t NONCES_PER_BATCH = gpu_res->optimal_max_nonces;
    
    // Extract the original difficulty threshold from the puzzle
    Digest original_threshold = hex_to_digest(puzzle.threshold);
    
    // Use a significantly easier target for benchmark mode to ensure we always
    // find solutions for validation testing. 1M× gives ~20-30 solutions per
    // 10-second run while keeping enough full batches for accurate sustained rate.
    constexpr uint64_t BENCHMARK_EASY_FACTOR = 1000000ULL;  // 1 million
    Digest test_target = make_target_easier(original_threshold, BENCHMARK_EASY_FACTOR);
    std::string test_target_hex = digest_to_hex(test_target);
    std::string original_threshold_hex = digest_to_hex(original_threshold);
    
    
    uint64_t total_nonces = 0;        // Always increases to avoid nonce reuse
    uint64_t measured_nonces = 0;     // Excludes warmup batches for steady-state rate
    uint64_t solutions_found = 0;
    uint64_t valid_solutions = 0;
    uint64_t invalid_threshold = 0;
    
    const char* warmup_env = std::getenv("XNT_WARMUP_BATCHES");
    int warmup_batches = 5;
    if (warmup_env && warmup_env[0] != '\0') {
        warmup_batches = std::max(0, std::atoi(warmup_env));
    }
    
    auto start_time = std::chrono::steady_clock::now();
    auto last_update = start_time;
    uint64_t last_nonces = 0;  // Track nonces at last update for instantaneous rate calculation
    int iteration = 0;
    
    install_signal_handlers();
    
    std::cout << "\n" << Color::BOLD << "Validation:" << Color::RESET << std::endl;
    std::cout << "  Original Threshold: " << original_threshold_hex << std::endl;
    std::cout << "  Test Threshold:     " << test_target_hex << " (" << BENCHMARK_EASY_FACTOR << "x easier)" << std::endl;
    std::cout << "  Validating against:  " << Color::YELLOW << "Test Threshold (" << BENCHMARK_EASY_FACTOR << "x easier)" << Color::RESET << std::endl;
    std::cout << "  Checking trailing zeros and threshold comparison" << std::endl;
    std::cout << std::endl;
    
    // Debug: per-batch timing (disabled by default for steady performance)
    bool debug_batch_timing = false;
    if (const char* dbg_env = std::getenv("XNT_DEBUG_BATCH"); dbg_env && dbg_env[0] == '1') {
        debug_batch_timing = true;
    }
    
    // Async mode: use double-buffered pipelining (set XNT_ASYNC_BENCH=1)
    bool use_async = false;
    if (const char* async_env = std::getenv("XNT_ASYNC_BENCH"); async_env && async_env[0] == '1') {
        use_async = true;
        std::cout << Color::CYAN << "  Mode: ASYNC (double-buffered, 2 streams)" << Color::RESET << std::endl;
    } else {
        std::cout << Color::CYAN << "  Mode: SYNC (single stream)" << Color::RESET << std::endl;
    }
    std::cout << std::endl;
    
    int batch_count = 0;
    int measured_batch_count = 0;
    int full_batch_count = 0;
    double full_batch_total_ms = 0.0;
    
    // Initialize async miner if using async mode
    std::unique_ptr<DoubleBufferedMiner> async_miner;
    int active_slot = 0;
    struct AsyncBatchInfo {
        uint64_t start_nonce;
        std::chrono::steady_clock::time_point start_time;
        bool pending;
    };
    AsyncBatchInfo async_batch_info[2] = {};
    
    if (use_async) {
        async_miner = std::make_unique<DoubleBufferedMiner>();
        if (!async_miner->initialize()) {
            std::cerr << Color::RED << "Failed to initialize async miner" << Color::RESET << std::endl;
            return;
        }
    }
    
    while (!stop_mining) {
        std::optional<MiningSolution> result;
        auto batch_start = std::chrono::steady_clock::now();
        Digest target = test_target;
        
        if (use_async) {
            // ============ ASYNC DOUBLE-BUFFERED MODE ============
            AsyncMiningSlot& slot = async_miner->slots[active_slot];
            AsyncBatchInfo& info = async_batch_info[active_slot];
            
            // Check if current slot has a pending result
            if (info.pending) {
                cudaError_t status = cudaEventQuery(slot.completion_event);
                if (status == cudaSuccess) {
                    // Kernel done - process result
                    auto now = std::chrono::steady_clock::now();
                    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(now - info.start_time).count();
                    double batch_duration_ms = duration_us / 1000.0;
                    
                    batch_count++;
                    total_nonces += NONCES_PER_BATCH;
                    
                    const bool in_warmup = batch_count <= warmup_batches;
                    if (!in_warmup) {
                        measured_batch_count++;
                        measured_nonces += NONCES_PER_BATCH;
                        
                        int sol_found = *slot.h_solution_found_pinned;
                        if (!sol_found) {
                            full_batch_count++;
                            full_batch_total_ms += batch_duration_ms;
                        } else {
                            // Retrieve solution
                            MiningSolution sol;
                            cudaMemcpy(&sol.pow.nonce, slot.d_solution_nonce_digest, sizeof(Digest), cudaMemcpyDeviceToHost);
                            cudaMemcpy(sol.pow.path_a, slot.d_solution_path_a, MERKLE_TREE_HEIGHT_ * sizeof(Digest), cudaMemcpyDeviceToHost);
                            cudaMemcpy(sol.pow.path_b, slot.d_solution_path_b, MERKLE_TREE_HEIGHT_ * sizeof(Digest), cudaMemcpyDeviceToHost);
                            cudaMemcpy(&sol.kernel_final_hash, slot.d_solution_final_hash, sizeof(Digest), cudaMemcpyDeviceToHost);
                            sol.pow.root = gpu_res->buffer->merkle_root;
                            solutions_found++;
                            
                            // Validate solution
                            bool meets_threshold = digest_less_than_or_equal(sol.kernel_final_hash, test_target);
                            if (meets_threshold) {
                                valid_solutions++;
                                if (solutions_found <= 3) {
                                    std::string hash_hex = digest_to_hex(sol.kernel_final_hash);
                                    std::cout << "\n" << Color::GREEN << "[ASYNC VALIDATION] Valid solution found!" << Color::RESET << std::endl;
                                    std::cout << "  Kernel final_hash: " << hash_hex << std::endl;
                                }
                            } else {
                                invalid_threshold++;
                            }
                        }
                    } else if (batch_count == warmup_batches) {
                        // Reset after warmup
                        measured_nonces = 0;
                        measured_batch_count = 0;
                        full_batch_count = 0;
                        full_batch_total_ms = 0.0;
                        solutions_found = 0;
                        valid_solutions = 0;
                        invalid_threshold = 0;
                        start_time = std::chrono::steady_clock::now();
                        last_update = start_time;
                        last_nonces = 0;
                    }
                    
                    info.pending = false;
                } else if (status == cudaErrorNotReady) {
                    // Switch to other slot
                    active_slot = 1 - active_slot;
                    if (async_batch_info[active_slot].pending) {
                        // Both busy, wait
                        cudaStreamSynchronize(slot.stream);
                    }
                    continue;
                }
            }
            
            // Launch new batch on current slot
            *slot.h_solution_found_pinned = 0;
            cudaMemsetAsync(slot.d_solution_found, 0, sizeof(int), slot.stream);
            
            int threads_per_block, blocks_per_grid;
            calculate_mining_launch_config(NONCES_PER_BATCH, threads_per_block, blocks_per_grid, device_id);
            
            if (!gpu_res->buffer->gpu_range_initialized) {
                GpuNonceRange gpu_range = calculate_gpu_range(device_id, 1);
                cudaMemcpyToSymbol(d_gpu_range_start, &gpu_range.range_start, sizeof(uint64_t));
                cudaMemcpyToSymbol(d_gpu_range_size, &gpu_range.range_size, sizeof(uint64_t));
                initialize_top_tree_cache(gpu_res->buffer->d_merkle_tree, gpu_res->buffer->num_leafs);
                gpu_res->buffer->gpu_range_initialized = true;
            }
            
            parallel_mining_kernel_high_vram<<<blocks_per_grid, threads_per_block, 0, slot.stream>>>(
                gpu_res->buffer->d_leafs,
                gpu_res->buffer->d_merkle_tree,
                gpu_res->buffer->index_picker_preimage,
                target,
                total_nonces,
                NONCES_PER_BATCH,
                gpu_res->buffer->num_leafs,
                MERKLE_TREE_HEIGHT_,
                gpu_res->buffer->mast_paths,
                gpu_res->buffer->hash,
                gpu_res->buffer->consensus_rule_set,
                slot.d_solution_nonce,
                slot.d_solution_found,
                slot.d_solution_path_a,
                slot.d_solution_path_b,
                slot.d_solution_nonce_digest,
                slot.d_solution_final_hash);
            
            cudaMemcpyAsync(slot.h_solution_found_pinned, slot.d_solution_found, sizeof(int), cudaMemcpyDeviceToHost, slot.stream);
            cudaEventRecord(slot.completion_event, slot.stream);
            
            info.start_nonce = total_nonces;
            info.start_time = std::chrono::steady_clock::now();
            info.pending = true;
            
            active_slot = 1 - active_slot;
            
            // Update progress display
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            auto since_update = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count();
            if (since_update >= 1000 && measured_nonces > 0) {
                uint64_t nonces_since_update = measured_nonces - last_nonces;
                double time_since_update_sec = since_update / 1000.0;
                double hash_rate = (nonces_since_update / 1000000.0) / std::max(0.001, time_since_update_sec);
                std::cout << "\r[Benchmark ASYNC] Hash Rate: " << Color::CYAN << std::fixed << std::setprecision(2) 
                          << hash_rate << " MH/s" << Color::RESET 
                          << " | Nonces: " << measured_nonces 
                          << " | Time: " << elapsed << "s" << std::flush;
                last_update = now;
                last_nonces = measured_nonces;
            }
            if (elapsed >= BENCHMARK_DURATION_SEC) break;
        } else {
            // ============ SYNC MODE (original) ============
            result = mine_pow_with_buffer(
                *gpu_res->buffer,
                target,
                gpu_res->buffer->mast_paths,
                total_nonces,
                NONCES_PER_BATCH,
                gpu_res->buffer->consensus_rule_set,
                nullptr
            );
        
            auto batch_end = std::chrono::steady_clock::now();
            auto batch_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count();
            double batch_duration_ms = batch_duration_us / 1000.0;
            double batch_hashrate = (NONCES_PER_BATCH / 1000000.0) / (batch_duration_ms / 1000.0);
            
            batch_count++;
            total_nonces += NONCES_PER_BATCH;
            
            const bool in_warmup = batch_count <= warmup_batches;
            if (in_warmup) {
                if (debug_batch_timing) {
                    std::cout << "\n[DEBUG] Warmup Batch #" << batch_count 
                              << " | Nonces: " << NONCES_PER_BATCH 
                              << " | Duration: " << std::fixed << std::setprecision(2) << batch_duration_ms << " ms"
                              << " | Rate: " << std::setprecision(2) << batch_hashrate << " MH/s"
                              << " | Solution: " << (result.has_value() ? "YES" : "NO")
                              << std::endl;
                }
                
                if (batch_count == warmup_batches) {
                    // Reset measurement after warmup to avoid boost/thermal ramp affecting results
                    measured_nonces = 0;
                    measured_batch_count = 0;
                    full_batch_count = 0;
                    full_batch_total_ms = 0.0;
                    solutions_found = 0;
                    valid_solutions = 0;
                    invalid_threshold = 0;
                    start_time = std::chrono::steady_clock::now();
                    last_update = start_time;
                    last_nonces = 0;
                }
                continue;
            }
            
            measured_batch_count++;
            measured_nonces += NONCES_PER_BATCH;
            
            // Track full batches (no solution found = processed all nonces)
            if (!result.has_value()) {
                full_batch_count++;
                full_batch_total_ms += batch_duration_ms;
            }
            
            if (debug_batch_timing) {
                std::cout << "\n[DEBUG] Batch #" << measured_batch_count 
                          << " | Nonces: " << NONCES_PER_BATCH 
                          << " | Duration: " << std::fixed << std::setprecision(2) << batch_duration_ms << " ms"
                          << " | Rate: " << std::setprecision(2) << batch_hashrate << " MH/s"
                          << " | Solution: " << (result.has_value() ? "YES" : "NO")
                          << std::endl;
            }
            
            // Validate any solution found by the kernel
            if (result.has_value()) {
                solutions_found++;
                
                // Extract solution components
                const MiningSolution& mining_solution = result.value();
                const Digest& kernel_final_hash = mining_solution.kernel_final_hash;
                
                // Convert to hex for display
                std::string kernel_final_hash_hex = digest_to_hex(kernel_final_hash);
                
                // Validate using the kernel's final_hash (the value the kernel actually checked)
                // This matches the Rust node's validation: final_hash <= threshold
                bool meets_threshold = digest_less_than_or_equal(kernel_final_hash, test_target);
                
                // Extract trailing hex digits for display (matches the format shown in node logs)
                std::string pow_trailing = "";
                std::string threshold_trailing = "";
                if (kernel_final_hash_hex.length() >= 16) {
                    pow_trailing = kernel_final_hash_hex.substr(kernel_final_hash_hex.length() - 16);
                }
                if (test_target_hex.length() >= 16) {
                    threshold_trailing = test_target_hex.substr(test_target_hex.length() - 16);
                }
                
                // Validate solution: kernel_final_hash must be <= threshold
                if (meets_threshold) {
                    valid_solutions++;
                    
                    // Log first few valid solutions for verification (limit output to avoid spam)
                    if (solutions_found <= 3) {
                        std::cout << "\n" << Color::GREEN << "[VALIDATION] ✓ Valid solution found! (Kernel final_hash <= threshold)" << Color::RESET << std::endl;
                        std::cout << "  Kernel final_hash: " << kernel_final_hash_hex << std::endl;
                        std::cout << "  Test Threshold:    " << test_target_hex << std::endl;
                        std::cout << "  Meets threshold:    " << Color::GREEN << "YES" << Color::RESET << std::endl;
                        if (!pow_trailing.empty()) {
                            std::cout << "  Final Hash Trailing: " << pow_trailing << std::endl;
                        }
                        if (!threshold_trailing.empty()) {
                            std::cout << "  Threshold Trailing:  " << threshold_trailing << std::endl;
                        }
                    }
                } else {
                    invalid_threshold++;
                    // Log first few invalid solutions for debugging
                    if (solutions_found <= 3) {
                        std::cout << "\n" << Color::YELLOW << "[VALIDATION] ✗ Solution does not meet threshold" << Color::RESET << std::endl;
                        std::cout << "  " << Color::DIM << "Note: kernel_final_hash > threshold" << Color::RESET << std::endl;
                        std::cout << "  Kernel final_hash: " << kernel_final_hash_hex << std::endl;
                        std::cout << "  Test Threshold:    " << test_target_hex << std::endl;
                        std::cout << "  Meets threshold:   " << Color::RED << "NO" << Color::RESET << std::endl;
                        if (!pow_trailing.empty()) {
                            std::cout << "  Final Hash Trailing: " << pow_trailing << std::endl;
                            // Check for trailing zeros (valid solutions typically have trailing zeros,
                            // matching the pattern shown in the Rust node's validation logs)
                            bool has_trailing_zeros = (pow_trailing.find_first_not_of('0') == std::string::npos) || 
                                                      (pow_trailing.back() == '0');
                            std::cout << "  Has trailing zeros:  " << (has_trailing_zeros ? Color::GREEN : Color::RED) 
                                      << (has_trailing_zeros ? "YES" : "NO") << Color::RESET << std::endl;
                        }
                        if (!threshold_trailing.empty()) {
                            std::cout << "  Threshold Trailing:  " << threshold_trailing << std::endl;
                        }
                    }
                }
            }
            
            iteration++;
            
            // Update progress display and check if benchmark duration has elapsed
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            auto since_update = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count();
            
            // Update hash rate display every second
            // Calculate instantaneous rate (nonces processed in last second) instead of cumulative average
            if (since_update >= 1000) {
                uint64_t nonces_since_update = measured_nonces - last_nonces;
                double time_since_update_sec = since_update / 1000.0;
                // Calculate instantaneous hash rate: nonces in last second / time elapsed
                double hash_rate = (nonces_since_update / 1000000.0) / std::max(0.001, time_since_update_sec);
                std::cout << "\r[Benchmark] Hash Rate: " << Color::CYAN << std::fixed << std::setprecision(2) 
                          << hash_rate << " MH/s" << Color::RESET 
                          << " | Nonces: " << measured_nonces 
                          << " | Time: " << elapsed << "s" << std::flush;
                last_update = now;
                last_nonces = measured_nonces;  // Update tracked nonces for next calculation
            }
            
            // Stop benchmark after the configured duration
            if (elapsed >= BENCHMARK_DURATION_SEC) {
                break;
            }
        } // end else (sync mode)
    }
    
    // Drain any pending async batches
    if (use_async && async_miner) {
        for (int i = 0; i < 2; i++) {
            if (async_batch_info[i].pending) {
                cudaStreamSynchronize(async_miner->slots[i].stream);
            }
        }
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    double total_hash_rate = (measured_nonces / 1000000.0) / (total_elapsed / 1000.0);
    
    std::cout << "\n\n" << Color::BOLD << "=== Benchmark Results ===" << Color::RESET << std::endl;
    std::cout << "  Total Nonces:  " << measured_nonces << std::endl;
    std::cout << "  Total Batches: " << measured_batch_count << std::endl;
    if (warmup_batches > 0) {
        std::cout << "  Warmup Batches: " << warmup_batches << std::endl;
    }
    std::cout << "  Batch Size:    " << NONCES_PER_BATCH << " (" << (NONCES_PER_BATCH / 1000000.0) << "M)" << std::endl;
    std::cout << "  Duration:      " << (total_elapsed / 1000.0) << " seconds" << std::endl;
    if (measured_batch_count > 0) {
        std::cout << "  Avg ms/batch:  " << std::fixed << std::setprecision(2) << (total_elapsed / (double)measured_batch_count) << " ms" << std::endl;
    }
    std::cout << "  Average Rate:  " << Color::GREEN << std::fixed << std::setprecision(2) 
              << total_hash_rate << " MH/s" << Color::RESET << " (includes early exits)" << std::endl;
    
    // Show sustained rate from full batches only (only meaningful for sync mode)
    if (full_batch_count > 0 && !use_async) {
        double full_batch_avg_ms = full_batch_total_ms / full_batch_count;
        double sustained_rate = (NONCES_PER_BATCH / 1000000.0) / (full_batch_avg_ms / 1000.0);
        std::cout << "  " << Color::BOLD << "Sustained Rate: " << Color::YELLOW << std::setprecision(2) 
                  << sustained_rate << " MH/s" << Color::RESET << " (full batches only, n=" << full_batch_count << ")" << std::endl;
    } else if (use_async) {
        // For async mode, the average rate IS the sustained rate due to double-buffering
        std::cout << "  " << Color::BOLD << "Sustained Rate: " << Color::YELLOW << std::setprecision(2) 
                  << total_hash_rate << " MH/s" << Color::RESET << " (async double-buffered)" << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << Color::BOLD << "=== Validation Results ===" << Color::RESET << std::endl;
    std::cout << "  Solutions Found:     " << solutions_found << std::endl;
    std::cout << "  Valid (meets threshold): " << Color::GREEN << valid_solutions << Color::RESET << std::endl;
    if (invalid_threshold > 0) {
        std::cout << "  Invalid (threshold):  " << Color::YELLOW << invalid_threshold << Color::RESET << std::endl;
        std::cout << "    (kernel_final_hash > threshold, rejected like Rust node)" << std::endl;
    }
    if (solutions_found > 0) {
        double valid_rate = (valid_solutions * 100.0) / solutions_found;
        std::cout << "  Validation Rate:     " << std::fixed << std::setprecision(1) 
                  << valid_rate << "%" << std::endl;
        std::cout << "\n  " << Color::DIM << "Note: Trailing zeros in hex representation indicate" << std::endl;
        std::cout << "  the exact format used by the node for validation." << Color::RESET << std::endl;
    }
    std::cout << std::endl;
    
    // Cleanup GPU resources
    if (gpu_res->buffer) {
        gpu_res->buffer->cleanup();
        gpu_res->buffer.reset();
    }
}

/// Run benchmark with different block counts and report MH/s.
void runBenchmarkBlocksSweep(const std::string& endpoint, int gpu_id) {
    constexpr int SWEEP_BLOCKS[] = {512, 680, 1024, 1536, 2048, 2560, 3072, 4096, 6144, 8192};
    constexpr int SWEEP_DURATION_SEC = 5;
    constexpr int WARMUP_BATCHES = 2;
    
    std::cout << Color::BOLD << "\n=== Blocks Sweep Benchmark ===" << Color::RESET << std::endl;
    std::cout << "  Duration per config: " << SWEEP_DURATION_SEC << " s  |  Warmup: " << WARMUP_BATCHES << " batches\n" << std::endl;
    
    json template_response;
    if (!loadBenchmarkTemplate(template_response)) {
        std::cerr << Color::RED << "Error: " << BENCHMARK_FILE << " not found. Run --benchmark first to save template." << Color::RESET << std::endl;
        return;
    }
    
    json template_obj;
    if (template_response.contains("result") && template_response["result"].contains("template")) {
        template_obj = template_response["result"]["template"];
    } else if (template_response.contains("template")) {
        template_obj = template_response["template"];
    } else {
        std::cerr << Color::RED << "Error: Invalid template format" << Color::RESET << std::endl;
        return;
    }
    
    std::string template_json_str = template_response.dump();
    PowPuzzle puzzle = parsePowPuzzle(template_json_str);
    if (!puzzle.is_valid()) {
        std::cerr << Color::RED << "Error: Invalid puzzle" << Color::RESET << std::endl;
        return;
    }
    
    int device_id = (gpu_id >= 0) ? gpu_id : 0;
    if (cudaSetDevice(device_id) != cudaSuccess) {
        std::cerr << Color::RED << "CUDA Error setting device" << Color::RESET << std::endl;
        return;
    }
    
    auto gpu_res = std::make_unique<GpuResources>(device_id);
    gpu_res->mining_mode = MiningMode::Solo;
    if (!preprocessPuzzle(puzzle, gpu_res.get())) {
        std::cerr << Color::RED << "Failed to initialize GPU" << Color::RESET << std::endl;
        return;
    }
    
    if (g_batch_size > 0) {
        gpu_res->optimal_max_nonces = std::max(g_batch_size, uint64_t(1));
    } else {
        gpu_res->optimal_max_nonces = get_optimal_batch_size(device_id);
    }
    
    Digest original_threshold = hex_to_digest(puzzle.threshold);
    constexpr uint64_t BENCHMARK_EASY_FACTOR = 1000000ULL;
    Digest test_target = make_target_easier(original_threshold, BENCHMARK_EASY_FACTOR);
    uint64_t NONCES_PER_BATCH = gpu_res->optimal_max_nonces;
    
    std::cout << "  Blocks  |  MH/s (sustained)" << std::endl;
    std::cout << "  ------- |  -----------------" << std::endl;
    
    double best_mhps = 0;
    int best_blocks = 0;
    
    for (int blocks : SWEEP_BLOCKS) {
        g_blocks_per_grid = blocks;
        uint64_t total_nonces = 0;
        int batch_count = 0;
        uint64_t measured_nonces = 0;
        int full_batch_count = 0;
        double full_batch_total_ms = 0.0;
        
        auto start_time = std::chrono::steady_clock::now();
        
        while (!stop_mining) {
            auto batch_start = std::chrono::steady_clock::now();
            auto result = mine_pow_with_buffer(
                *gpu_res->buffer,
                test_target,
                gpu_res->buffer->mast_paths,
                total_nonces,
                NONCES_PER_BATCH,
                gpu_res->buffer->consensus_rule_set,
                nullptr);
            auto batch_end = std::chrono::steady_clock::now();
            double batch_ms = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();
            
            batch_count++;
            total_nonces += NONCES_PER_BATCH;
            
            if (batch_count > WARMUP_BATCHES) {
                measured_nonces += NONCES_PER_BATCH;
                if (!result.has_value()) {
                    full_batch_count++;
                    full_batch_total_ms += batch_ms;
                }
            }
            
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(batch_end - start_time).count();
            if (elapsed >= SWEEP_DURATION_SEC) break;
        }
        
        double sustained_mhps = 0;
        if (full_batch_count > 0) {
            sustained_mhps = (NONCES_PER_BATCH / 1000000.0) / (full_batch_total_ms / full_batch_count / 1000.0);
        } else if (measured_nonces > 0) {
            auto end_time = std::chrono::steady_clock::now();
            double sec = std::chrono::duration<double>(end_time - start_time).count();
            sustained_mhps = (measured_nonces / 1000000.0) / std::max(0.001, sec);
        }
        
        std::cout << "  " << std::setw(6) << blocks << "  |  " << std::fixed << std::setprecision(2) << sustained_mhps << std::endl;
        
        if (sustained_mhps > best_mhps) {
            best_mhps = sustained_mhps;
            best_blocks = blocks;
        }
    }
    
    std::cout << std::endl;
    std::cout << Color::BOLD << "Best: " << best_blocks << " blocks @ " << std::fixed << std::setprecision(2) 
              << best_mhps << " MH/s" << Color::RESET << std::endl;
    std::cout << "  Use: --blocks " << best_blocks << " or XNT_BLOCKS_PER_GRID=" << best_blocks << std::endl;
    std::cout << std::endl;
    
    g_blocks_per_grid = 0;  // Reset
    if (gpu_res->buffer) {
        gpu_res->buffer->cleanup();
        gpu_res->buffer.reset();
    }
}

int main(int argc, char* argv[]) {
    enable_ansi_colors();
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    
    // Batch size: env XNT_BATCH_SIZE as default (0=auto), --batch overrides
    if (const char* env = std::getenv("XNT_BATCH_SIZE"); env && env[0] != '\0') {
        try { g_batch_size = std::stoull(env); } catch (...) { /* keep default */ }
    }
    // Blocks per grid: env XNT_BLOCKS_PER_GRID (0=auto; 680=fast miner)
    if (const char* env = std::getenv("XNT_BLOCKS_PER_GRID"); env && env[0] != '\0') {
        int v = std::atoi(env);
        if (v > 0 && v <= 65535) g_blocks_per_grid = v;
    }
    
    std::string endpoint = "http://127.0.0.1:9897";
    std::string stratum_password = "x";
    std::string stratum_worker_name = "xnt-miner";
    MiningMode mining_mode = MiningMode::Solo;
    bool show_help = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            show_help = true;
        } else if ((arg == "--wallet" || arg == "-w") && i + 1 < argc) {
            g_miner_wallet_address = argv[++i];
        } else if ((arg == "--device" || arg == "-d") && i + 1 < argc) {
            try {
                g_gpu_device_id = std::stoi(argv[++i]);
                if (g_gpu_device_id < 0) {
                    std::cerr << Color::RED << "Error: Device ID must be non-negative" << Color::RESET << std::endl;
                    return 1;
                }
            } catch (const std::exception& e) {
                std::cerr << Color::RED << "Error: Invalid device ID: " << argv[i] << Color::RESET << std::endl;
                return 1;
            }
        } else if ((arg == "--host" || arg == "-H") && i + 1 < argc) {
            std::string host = argv[++i];
            // Host option only applies to solo mining mode (not stratum pools)
            if (mining_mode == MiningMode::Solo) {
                size_t port_start = endpoint.find(":", 7);
                if (port_start != std::string::npos) {
                    std::string port = endpoint.substr(port_start + 1);
                    endpoint = "http://" + host + ":" + port;
                } else {
                    endpoint = "http://" + host + ":9897";
                }
            }
        } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            try {
                int port = std::stoi(argv[++i]);
                if (port < 1 || port > 65535) {
                    std::cerr << Color::RED << "Error: Port must be between 1 and 65535" << Color::RESET << std::endl;
                    return 1;
                }
                // Port option only applies to solo mining mode (not stratum pools)
                if (mining_mode == MiningMode::Solo) {
                    size_t host_start = endpoint.find("://") + 3;
                    size_t port_start = endpoint.find(":", host_start);
                    if (port_start != std::string::npos) {
                        endpoint = endpoint.substr(0, port_start + 1) + std::to_string(port);
                    } else {
                        endpoint = endpoint + ":" + std::to_string(port);
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << Color::RED << "Error: Invalid port number: " << argv[i] << Color::RESET << std::endl;
                return 1;
            }
        } else if ((arg == "--rpc-url") && i + 1 < argc) {
            endpoint = argv[++i];
            mining_mode = MiningMode::Solo;  // RPC URL explicitly sets solo mining mode
        } else if ((arg == "--stratum") && i + 1 < argc) {
            endpoint = argv[++i];
            mining_mode = MiningMode::Stratum;
        } else if ((arg == "--stratum-pass") && i + 1 < argc) {
            stratum_password = argv[++i];
        } else if ((arg == "--stratum-worker") && i + 1 < argc) {
            stratum_worker_name = argv[++i];
        } else if (arg == "--test-mode") {
            g_test_mode = true;
        } else if (arg == "--benchmark") {
            g_benchmark_mode = true;
        } else if ((arg == "--fetch-interval") && i + 1 < argc) {
            try {
                int interval = std::stoi(argv[++i]);
                if (interval < 1) {
                    std::cerr << Color::RED << "Error: Fetch interval must be at least 1 second" << Color::RESET << std::endl;
                    return 1;
                }
                g_fetch_interval_sec = interval;
            } catch (const std::exception& e) {
                std::cerr << Color::RED << "Error: Invalid fetch interval: " << argv[i] << Color::RESET << std::endl;
                return 1;
            }
        } else if ((arg == "--batch") && i + 1 < argc) {
            try {
                g_batch_size = std::stoull(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << Color::RED << "Error: Invalid batch size: " << argv[i] << Color::RESET << std::endl;
                return 1;
            }
        } else if ((arg == "--blocks") && i + 1 < argc) {
            try {
                g_blocks_per_grid = std::stoi(argv[++i]);
                if (g_blocks_per_grid < 0 || g_blocks_per_grid > 65535) {
                    std::cerr << Color::RED << "Error: Blocks must be 1–65535" << Color::RESET << std::endl;
                    return 1;
                }
            } catch (const std::exception& e) {
                std::cerr << Color::RED << "Error: Invalid blocks value: " << argv[i] << Color::RESET << std::endl;
                return 1;
            }
        } else if (arg == "--blocks-sweep") {
            g_blocks_sweep = true;
        } else if (arg[0] == '-') {
            std::cerr << Color::RED << "Error: Unknown option: " << arg << Color::RESET << std::endl;
            std::cerr << "Use --help or -h for usage information" << std::endl;
            return 1;
        } else {
            std::cerr << Color::RED << "Error: Unexpected argument: " << arg << Color::RESET << std::endl;
            std::cerr << "Use --help or -h for usage information" << std::endl;
            return 1;
        }
    }
    
    if (show_help) {
        print_usage(argv[0]);
        return 0;
    }
    
    // Handle benchmark mode early (before normal mining setup)
    // Wallet address is only required if the benchmark template doesn't exist yet
    if (g_benchmark_mode) {
        // Check if a cached template exists - if so, wallet is not required
        json test_template;
        bool template_exists = loadBenchmarkTemplate(test_template);
        
        if (!template_exists && g_miner_wallet_address.empty()) {
            std::cerr << "\n" << Color::RED << Color::BOLD << "Error: Wallet address is required to fetch template" << Color::RESET << std::endl;
            std::cerr << "\nFirst-time benchmark requires wallet address to fetch template from node." << std::endl;
            std::cerr << "After template is saved, wallet is not required.\n" << std::endl;
            print_usage(argv[0]);
            return 1;
        }
        
        print_system_info();
        cudaDeviceReset();
        install_signal_handlers();
        if (g_blocks_sweep) {
            runBenchmarkBlocksSweep(endpoint, g_gpu_device_id);
        } else {
            runBenchmark(endpoint, g_gpu_device_id);
        }
        cudaDeviceReset();
        return 0;
    }
    
    // Normal mining mode: wallet address is required for block rewards
    if (g_miner_wallet_address.empty()) {
        std::cerr << "\n" << Color::RED << Color::BOLD << "Error: Wallet address is required" << Color::RESET << std::endl;
        std::cerr << "\nMining requires your Neptune wallet address.\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    // Auto-detect mining mode from URL format if not explicitly set
    // (stratum:// URLs indicate pool mining, http:// indicates solo mining)
    if (mining_mode == MiningMode::Solo) {
        mining_mode = detect_mining_mode(endpoint);
    }
    
    g_miner_worker_name = stratum_worker_name;
    print_system_info();
    
    std::cout << Color::BOLD << "Configuration:" << Color::RESET << std::endl;
    std::cout << "  Mining Mode:   " << mining_mode_name(mining_mode) << std::endl;
    if (mining_mode == MiningMode::Stratum) {
        std::cout << "  Pool:          " << endpoint << std::endl;
        std::cout << "  Worker:        " << g_miner_worker_name << std::endl;
    } else {
        std::cout << "  RPC Endpoint:  " << endpoint << std::endl;
    }
    std::cout << "  Wallet:        " << shorten_address(g_miner_wallet_address) << std::endl;
    if (g_gpu_device_id >= 0) {
        std::cout << "  Device:        GPU " << g_gpu_device_id << std::endl;
    } else {
        std::cout << "  Device:        All available GPUs" << std::endl;
    }
    if (g_test_mode) {
        std::cout << "  Test Mode:     " << Color::YELLOW << "ENABLED" << Color::RESET << std::endl;
    }
    std::cout << "  Batch size:    " << (g_batch_size == 0 ? "auto" : std::to_string(g_batch_size) + " nonces") << std::endl;
    std::cout << std::endl;
    
    cudaDeviceReset();
    install_signal_handlers();
    
    try {
        startUnifiedMining(endpoint, g_gpu_device_id, mining_mode, stratum_password);
    } catch (const std::exception& e) {
        std::string error_msg = e.what();
        // Sanitize error messages: replace full wallet address with shortened version
        // to avoid exposing sensitive information in logs
        if (!g_miner_wallet_address.empty() && error_msg.find(g_miner_wallet_address) != std::string::npos) {
            size_t pos = 0;
            while ((pos = error_msg.find(g_miner_wallet_address, pos)) != std::string::npos) {
                error_msg.replace(pos, g_miner_wallet_address.length(), shorten_address(g_miner_wallet_address));
                pos += shorten_address(g_miner_wallet_address).length();
            }
        }
        std::cerr << Color::RED << "Error: " << error_msg << Color::RESET << std::endl;
        return 1;
    }
    
    // Cleanup CUDA resources before exit
    cudaDeviceReset();
    std::cout << "\n" << Color::GREEN << "Mining stopped" << Color::RESET << std::endl;
    
    return 0;
}
