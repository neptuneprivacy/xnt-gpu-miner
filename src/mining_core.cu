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

__global__ void __launch_bounds__(512, 2) bitreverse_swap_leafs_kernel(
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

__global__ void __launch_bounds__(256) build_layer1_from_layer0_on_demand_kernel(
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

__global__ void __launch_bounds__(512, 2) build_layer0_from_commitment_kernel(
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

__global__ void __launch_bounds__(512, 2) compute_buds_kernel(
    Digest* __restrict__ buds,
    const Digest commitment,
    size_t segment_len,
    size_t base_index) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < segment_len) {
        size_t global_idx = base_index + idx;
        // Bud computation: 32 rounds of hashing (BUDDING_ROUNDS)
        // OPTIMIZATION: Reuse state array and reduce register pressure
        uint64_t state[STATE_SIZE];
        
        // Initialize state once with commitment (left side)
        state[0] = commitment.values[0];
        state[1] = commitment.values[1];
        state[2] = commitment.values[2];
        state[3] = commitment.values[3];
        state[4] = commitment.values[4];
        
        // Right side: [global_idx, 0, 0, 0, round] - varies only in round
        state[5] = global_idx;
        state[6] = 0;
        state[7] = 0;
        state[8] = 0;
        // state[9] will be set to round in the loop
        
        // Capacity region for FixedLength domain
        constexpr uint64_t FIXED_LEN_VAL = static_cast<uint64_t>(Domain::FixedLength);
        state[10] = FIXED_LEN_VAL;
        state[11] = FIXED_LEN_VAL;
        state[12] = FIXED_LEN_VAL;
        state[13] = FIXED_LEN_VAL;
        state[14] = FIXED_LEN_VAL;
        state[15] = FIXED_LEN_VAL;
        
        // First round (round=0)
        state[9] = 0;
        tip5_permutation(state);
        
        // Extract result as new left side for next round
        Digest hash;
        hash.values[0] = state[0];
        hash.values[1] = state[1];
        hash.values[2] = state[2];
        hash.values[3] = state[3];
        hash.values[4] = state[4];
        
        // Remaining 31 rounds - no pragma unroll to avoid register spills
        for (size_t round = 1; round < BUDDING_ROUNDS; ++round) {
            state[0] = hash.values[0];
            state[1] = hash.values[1];
            state[2] = hash.values[2];
            state[3] = hash.values[3];
            state[4] = hash.values[4];
            state[5] = global_idx;
            state[6] = 0;
            state[7] = 0;
            state[8] = 0;
            state[9] = round;
            
            tip5_permutation(state);
            
            hash.values[0] = state[0];
            hash.values[1] = state[1];
            hash.values[2] = state[2];
            hash.values[3] = state[3];
            hash.values[4] = state[4];
        }
        
        buds[global_idx] = hash;
    }
}

__global__ void __launch_bounds__(512, 2) compute_leafs_from_buds_kernel(
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
            size_t mask = num_leafs - 1ULL;
            size_t buddy_index = (idx + stride) & mask;
            leafs[idx] = tip5_hash_fixed_device(buds[idx], buds[buddy_index]);
        }
    }
}

__global__ void __launch_bounds__(512, 2) merkle_zip_kernel(
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
// Removed unused maybe_prefetch_managed function - no longer needed since we use
// regular device memory instead of managed memory for better performance
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
    
    int threadsPerBlock = 512;
    int numBlocks = (MERKLE_NUM_LEAFS + threadsPerBlock - 1) / threadsPerBlock;
    
    // Use cudaMallocAsync for leafs if available to eliminate allocation overhead
    bool used_async_alloc_leafs = false;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    int mem_pools_supported = 0;
    int device_id = 0;
    cudaGetDevice(&device_id);
    if (cudaDeviceGetAttribute(&mem_pools_supported, cudaDevAttrMemoryPoolsSupported, device_id) == cudaSuccess
        && mem_pools_supported) {
        cudaError_t async_err = cudaMallocAsync(&buffer.d_leafs, leafs_size, pp_stream);
        if (async_err == cudaSuccess) {
            used_async_alloc_leafs = true;
        }
    }
#endif
    if (!used_async_alloc_leafs) {
        alloc_err = cudaMalloc(&buffer.d_leafs, leafs_size);
        if (alloc_err != cudaSuccess) {
            LOG_ERROR("cudaMalloc leafs", alloc_err);
            cudaStreamDestroy(pp_stream);
            return GuesserBuffer();
        }
    }
    
    LOG_DEBUG("preprocess_high_vram: allocated " << (leafs_size / (1024*1024)) << " MB for leafs");
    
    // Step 1: Compute buds
    compute_buds_kernel<<<numBlocks, threadsPerBlock, 0, pp_stream>>>(
        buffer.d_leafs, commitment, MERKLE_NUM_LEAFS, 0);
    
    // Step 2: Convert buds → leafs through NUM_BUD_LAYERS iterations
    size_t temp_buffer_size = MERKLE_NUM_LEAFS * sizeof(Digest);
    Digest* d_temp_leafs = nullptr;
    
    // Use async allocation for temp buffer as well
    bool used_async_temp = false;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    if (mem_pools_supported) {
        cudaError_t async_err = cudaMallocAsync(&d_temp_leafs, temp_buffer_size, pp_stream);
        if (async_err == cudaSuccess) {
            used_async_temp = true;
        }
    }
#endif
    if (!used_async_temp) {
        cudaError_t temp_alloc_err = cudaMalloc(&d_temp_leafs, temp_buffer_size);
        if (temp_alloc_err != cudaSuccess) {
            LOG_ERROR("cudaMalloc temp_leafs", temp_alloc_err);
            cudaStreamDestroy(pp_stream);
            buffer.cleanup();
            return GuesserBuffer();
        }
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
    
    size_t internal_size = (MERKLE_NUM_LEAFS - 1) * sizeof(Digest);
    bool used_async_alloc = false;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    // Stream-ordered alloc: free temp and alloc merkle on same stream, no host sync.
    // Saves ~50-100 ms by eliminating cudaStreamSynchronize + cudaFree + cudaMalloc blocking.
    if (mem_pools_supported) {
        if (used_async_temp) {
            cudaFreeAsync(d_temp_leafs, pp_stream);
        } else {
            cudaFree(d_temp_leafs);
        }
        d_temp_leafs = nullptr;
        alloc_err = cudaMallocAsync(&buffer.d_merkle_tree, internal_size, pp_stream);
        if (alloc_err == cudaSuccess) {
            used_async_alloc = true;
        }
    }
#endif
    if (!used_async_alloc) {
        // Fallback: sync, free temp, then alloc (required for peak memory < 24 GB)
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
        alloc_err = cudaMalloc(&buffer.d_merkle_tree, internal_size);
    }
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
    
    // Use a dedicated stream for all preprocessing kernels (like HIGH_VRAM mode)
    cudaStream_t pp_stream;
    cudaError_t stream_err = cudaStreamCreate(&pp_stream);
    if (stream_err != cudaSuccess) {
        LOG_ERROR("cudaStreamCreate (preprocess low_vram)", stream_err);
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    int threadsPerBlock = 512;
    
    // Step 1: Allocate leafs buffer - use async if available
    size_t leafs_size = MERKLE_NUM_LEAFS * sizeof(Digest);
    Digest* d_leafs = nullptr;
    
    bool used_async_leafs = false;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    int mem_pools_supported = 0;
    int device_id = 0;
    cudaGetDevice(&device_id);
    if (cudaDeviceGetAttribute(&mem_pools_supported, cudaDevAttrMemoryPoolsSupported, device_id) == cudaSuccess
        && mem_pools_supported) {
        cudaError_t async_err = cudaMallocAsync(&d_leafs, leafs_size, pp_stream);
        if (async_err == cudaSuccess) {
            used_async_leafs = true;
        }
    }
#endif
    if (!used_async_leafs) {
        alloc_err = cudaMalloc(&d_leafs, leafs_size);
        if (alloc_err != cudaSuccess) {
            LOG_ERROR("cudaMalloc leafs (low_vram)", alloc_err);
            cudaStreamDestroy(pp_stream);
            buffer.cleanup();
            return GuesserBuffer();
        }
    }
    
    // Compute buds (all kernels use the same stream for in-order execution)
    int numBlocks = (MERKLE_NUM_LEAFS + threadsPerBlock - 1) / threadsPerBlock;
    compute_buds_kernel<<<numBlocks, threadsPerBlock, 0, pp_stream>>>(
        d_leafs, commitment, MERKLE_NUM_LEAFS, 0);
    
    // Allocate temp buffer for bud->leaf conversion - use async if available
    size_t temp_leafs_size = MERKLE_NUM_LEAFS * sizeof(Digest);
    Digest* d_temp_leafs = nullptr;
    
    bool used_async_temp = false;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    if (mem_pools_supported) {
        cudaError_t async_err = cudaMallocAsync(&d_temp_leafs, temp_leafs_size, pp_stream);
        if (async_err == cudaSuccess) {
            used_async_temp = true;
        }
    }
#endif
    if (!used_async_temp) {
        alloc_err = cudaMalloc(&d_temp_leafs, temp_leafs_size);
        if (alloc_err != cudaSuccess) {
            LOG_ERROR("cudaMalloc temp_leafs (low_vram)", alloc_err);
            if (used_async_leafs) {
                cudaFreeAsync(d_leafs, pp_stream);
            } else {
                cudaFree(d_leafs);
            }
            cudaStreamDestroy(pp_stream);
            buffer.cleanup();
            return GuesserBuffer();
        }
    }
    
    Digest* current_buds = d_leafs;
    Digest* current_leafs = d_temp_leafs;
    
    // Convert buds to leafs (all on same stream, no sync needed between kernels)
    for (size_t layer = 0; layer < NUM_BUD_LAYERS; ++layer) {
        int layerBlocks = (MERKLE_NUM_LEAFS + threadsPerBlock - 1) / threadsPerBlock;
        compute_leafs_from_buds_kernel<<<layerBlocks, threadsPerBlock, 0, pp_stream>>>(
            current_leafs, current_buds, MERKLE_NUM_LEAFS, layer);
        std::swap(current_buds, current_leafs);
    }
    
    // Ensure final leafs are in d_leafs (async copy on same stream)
    if (current_buds != d_leafs) {
        cudaMemcpyAsync(d_leafs, current_buds, leafs_size, cudaMemcpyDeviceToDevice, pp_stream);
        current_buds = d_leafs;
    }
    
    // Step 2: Build tree layer-by-layer using sliding window approach
    // Allocate two buffers for ping-pong between layers - use async if available
    // Max size needed is for layer 1 (2^25 nodes = half of leafs)
    const size_t MAX_INTERMEDIATE_LAYER_SIZE = MERKLE_NUM_LEAFS / 2;
    size_t intermediate_buffer_size = MAX_INTERMEDIATE_LAYER_SIZE * sizeof(Digest);
    Digest* d_layer_a = nullptr;
    Digest* d_layer_b = nullptr;
    
    bool used_async_layer_a = false;
    bool used_async_layer_b = false;
    
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    if (mem_pools_supported) {
        cudaError_t async_err = cudaMallocAsync(&d_layer_a, intermediate_buffer_size, pp_stream);
        if (async_err == cudaSuccess) {
            used_async_layer_a = true;
            async_err = cudaMallocAsync(&d_layer_b, intermediate_buffer_size, pp_stream);
            if (async_err == cudaSuccess) {
                used_async_layer_b = true;
            }
        }
    }
#endif
    
    if (!used_async_layer_a) {
        alloc_err = cudaMalloc(&d_layer_a, intermediate_buffer_size);
        if (alloc_err != cudaSuccess) {
            LOG_ERROR("cudaMalloc layer_a (low_vram)", alloc_err);
            if (used_async_temp) cudaFreeAsync(d_temp_leafs, pp_stream);
            else cudaFree(d_temp_leafs);
            if (used_async_leafs) cudaFreeAsync(d_leafs, pp_stream);
            else cudaFree(d_leafs);
            cudaStreamDestroy(pp_stream);
            buffer.cleanup();
            return GuesserBuffer();
        }
    }
    
    if (!used_async_layer_b) {
        alloc_err = cudaMalloc(&d_layer_b, intermediate_buffer_size);
        if (alloc_err != cudaSuccess) {
            LOG_ERROR("cudaMalloc layer_b (low_vram)", alloc_err);
            if (used_async_layer_a) cudaFreeAsync(d_layer_a, pp_stream);
            else cudaFree(d_layer_a);
            if (used_async_temp) cudaFreeAsync(d_temp_leafs, pp_stream);
            else cudaFree(d_temp_leafs);
            if (used_async_leafs) cudaFreeAsync(d_leafs, pp_stream);
            else cudaFree(d_leafs);
            cudaStreamDestroy(pp_stream);
            buffer.cleanup();
            return GuesserBuffer();
        }
    }
    
    
    // Free temp buffer now (we only need d_leafs and the two layer buffers)
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    if (used_async_temp) {
        cudaFreeAsync(d_temp_leafs, pp_stream);
    } else {
        cudaFree(d_temp_leafs);
    }
#else
    cudaFree(d_temp_leafs);
#endif
    d_temp_leafs = nullptr;
    
    // Build tree from layer 0 (leafs) up to layer 26 (root) using ping-pong buffers
    const Digest* current_layer = d_leafs;
    size_t current_layer_size = MERKLE_NUM_LEAFS;
    Digest* next_layer = d_layer_a;
    bool use_layer_a = true;
    
    size_t stored_offset = 0;
    
    for (size_t layer = 0; layer < MERKLE_TREE_HEIGHT_; ++layer) {
        size_t next_layer_size = current_layer_size / 2;
        
        // Compute next layer (all kernels on same stream, no sync needed)
        int layerBlocks = (next_layer_size + threadsPerBlock - 1) / threadsPerBlock;
        merkle_zip_kernel<<<layerBlocks, threadsPerBlock, 0, pp_stream>>>(
            next_layer, current_layer, next_layer_size);
        
        // If this is one of the layers we need to store (20-26), copy it asynchronously
        // Layer index here is the source layer (0-25), we're computing layer+1
        // We want to store layers 20-26, computed in iterations 19-25
        if (layer >= STORED_LAYER_START - 1) {
            size_t copy_size = next_layer_size * sizeof(Digest);
            Digest* dest_ptr = buffer.d_merkle_tree + stored_offset;
            cudaMemcpyAsync(dest_ptr, next_layer, copy_size, cudaMemcpyDeviceToDevice, pp_stream);
            stored_offset += next_layer_size;
        }
        
        // Prepare for next iteration - ping-pong between buffers
        current_layer = next_layer;
        current_layer_size = next_layer_size;
        
        // After computing layer 1 (iteration 0), we can free d_leafs to save memory
        if (layer == 0) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
            if (used_async_leafs) {
                cudaFreeAsync(d_leafs, pp_stream);
            } else {
                cudaFree(d_leafs);
            }
#else
            cudaFree(d_leafs);
#endif
            d_leafs = nullptr;
        }
        
        // Swap buffers for next iteration
        use_layer_a = !use_layer_a;
        next_layer = use_layer_a ? d_layer_a : d_layer_b;
    }
    
    // Single sync at the end: wait for all tree construction and copies to complete
    cudaError_t sync_err = cudaStreamSynchronize(pp_stream);
    cudaStreamDestroy(pp_stream);
    
    if (sync_err != cudaSuccess) {
        LOG_ERROR("preprocessing stream sync (low_vram)", sync_err);
        if (d_leafs != nullptr) cudaFree(d_leafs);
        cudaFree(d_layer_a);
        cudaFree(d_layer_b);
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    // Check for cancellation after sync
    if (cancel_flag && *cancel_flag) {
        if (d_leafs != nullptr) cudaFree(d_leafs);
        cudaFree(d_layer_a);
        cudaFree(d_layer_b);
        buffer.cleanup();
        return GuesserBuffer();
    }
    
    // Free temporary buffers
    if (d_leafs != nullptr) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
        if (used_async_leafs) {
            cudaFreeAsync(d_leafs, pp_stream);
        } else {
            cudaFree(d_leafs);
        }
#else
        cudaFree(d_leafs);
#endif
    }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    if (used_async_layer_a) cudaFreeAsync(d_layer_a, pp_stream);
    else cudaFree(d_layer_a);
    if (used_async_layer_b) cudaFreeAsync(d_layer_b, pp_stream);
    else cudaFree(d_layer_b);
#else
    cudaFree(d_layer_a);
    cudaFree(d_layer_b);
#endif
    
    // Copy root to host (tree is complete now)
    cudaMemcpy(&buffer.merkle_root, buffer.d_merkle_tree + buffer.tree_size - 1,
               sizeof(Digest), cudaMemcpyDeviceToHost);
    
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
