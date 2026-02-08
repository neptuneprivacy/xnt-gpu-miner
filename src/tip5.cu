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
#include "tip5.cuh"
#undef TIP5_DEFINING_CONSTANTS

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

// Extern declaration for the shared lookup table loaded by mining kernels
// This is defined in kernels.cu and loaded ONCE at kernel start with __syncthreads
extern __shared__ uint8_t s_lookup_table[257];  // 256 + 1 padding for bank conflict reduction

__device__ void sbox_layer(const uint64_t* __restrict__ state_in, uint64_t* __restrict__ state_out) {
    // Load all 16 elements into registers
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

    // S-box lookup for elements 0-3
    e0 = split_lookup_shared(e0, s_lookup_table);
    e1 = split_lookup_shared(e1, s_lookup_table);
    e2 = split_lookup_shared(e2, s_lookup_table);
    e3 = split_lookup_shared(e3, s_lookup_table);

    // ILP-OPTIMIZED x^7 computation for elements 4-15 (12 elements)
    // Using fused PTX field_mul_ptx for better register efficiency
    // x^7 = x * x^2 * x^4 where x^2 = x*x, x^4 = x^2*x^2, x^6 = x^4*x^2

    // Stage 1: Compute x^2 for all 12 elements
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

    // Stage 2: Compute x^4 for all 12 elements
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

    // Stage 3: Compute x^6 = x^4 * x^2 for all 12 elements
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

    // Stage 4: Compute x^7 = x^6 * x for all 12 elements
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

    // Store results
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

    // Split each element into lo and hi 32-bit limbs using PTX
    #pragma unroll
    for (int i = 0; i < STATE_SIZE; i++) {
        uint32_t lo32, hi32;
        asm("mov.b64 {%0, %1}, %2;" : "=r"(lo32), "=r"(hi32) : "l"(state_in[i]));
        lo[i] = lo32;
        hi[i] = hi32;
    }

    // Process each limb with FFT-based generated_function
    uint64_t lo_out[STATE_SIZE], hi_out[STATE_SIZE];
    generated_function(lo, lo_out);
    generated_function(hi, hi_out);

    // Combine elementwise: (lo >> 4) + (hi << 28), then reduce
    // Fully PTX-optimized using shf (funnel shift) and mad with carry
    #pragma unroll
    for (int i = 0; i < STATE_SIZE; i++) {
        // Compute s = (lo_out[i] >> 4) + (hi_out[i] << 28)
        // This is a 128-bit value, we compute s_lo and s_hi directly
        uint64_t lo_shifted = lo_out[i] >> 4;           // (lo >> 4), high 4 bits lost
        uint64_t hi_shifted_lo = hi_out[i] << 28;       // low 64 bits of (hi << 28)
        uint64_t hi_shifted_hi = hi_out[i] >> 36;       // high bits of (hi << 28)

        // s_lo = lo_shifted + hi_shifted_lo (with carry out)
        // s_hi = hi_shifted_hi + carry
        uint64_t s_lo, s_hi;
        asm("{\n\t"
            ".reg .pred c;\n\t"
            "add.cc.u64 %0, %2, %3;\n\t"      // s_lo = lo_shifted + hi_shifted_lo, set carry
            "addc.u64 %1, %4, 0;\n\t"         // s_hi = hi_shifted_hi + carry
            "}" : "=l"(s_lo), "=l"(s_hi) : "l"(lo_shifted), "l"(hi_shifted_lo), "l"(hi_shifted_hi));

        // Goldilocks reduction: res = s_lo + s_hi * 0xFFFFFFFF
        // Using PTX mad.lo for multiply-add with branchless overflow
        uint64_t res;
        asm("{\n\t"
            ".reg .u64 prod;\n\t"
            ".reg .pred p;\n\t"
            "mul.lo.u64 prod, %2, 4294967295;\n\t"  // prod = s_hi * 0xFFFFFFFF
            "add.cc.u64 %0, %1, prod;\n\t"          // res = s_lo + prod, set carry
            // Branchless: if overflow, add 0xFFFFFFFF
            "setp.lt.u64 p, %0, %1;\n\t"            // p = (res < s_lo)
            "selp.u64 prod, 4294967295, 0, p;\n\t"  // correction = p ? 0xFFFFFFFF : 0
            "add.u64 %0, %0, prod;\n\t"             // res += correction
            "}" : "=l"(res) : "l"(s_lo), "l"(s_hi));

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

        // Fused round constants addition
        for (int i = 0; i < STATE_SIZE; ++i) {
            state[i] = fast_field_add(state[i], ROUND_CONSTANTS[round][i]);
        }
    }
}

// Specialized permutation for Pow_indices hot loop (62 iterations).
// PRECONDITION: state[5..9]=0, state[10..15]=FIXED_LEN_VAL (=1).
// First round skips x^7 for 11 elements (saves 44 field_mul_ptx per call).
__device__ __noinline__ void tip5_permutation_fixed_right_zero(uint64_t* state, uint64_t x7_one) {
    uint64_t temp_state[STATE_SIZE];

    // === ROUND 0: exploit known right side ===
    // Elements 0-3: normal S-box lookup (data-dependent)
    temp_state[0] = split_lookup_shared(state[0], s_lookup_table);
    temp_state[1] = split_lookup_shared(state[1], s_lookup_table);
    temp_state[2] = split_lookup_shared(state[2], s_lookup_table);
    temp_state[3] = split_lookup_shared(state[3], s_lookup_table);

    // Element 4: x^7 of unknown value (must compute)
    temp_state[4] = x7_computer_pipelined(state[4]);

    // Elements 5-9: x^7(0) = 0 (trivially)
    temp_state[5]  = 0;
    temp_state[6]  = 0;
    temp_state[7]  = 0;
    temp_state[8]  = 0;
    temp_state[9]  = 0;

    // Elements 10-15: x^7(1) = precomputed constant (passed as arg)
    temp_state[10] = x7_one;
    temp_state[11] = x7_one;
    temp_state[12] = x7_one;
    temp_state[13] = x7_one;
    temp_state[14] = x7_one;
    temp_state[15] = x7_one;

    mds_layer(temp_state, state);
    for (int i = 0; i < STATE_SIZE; ++i) {
        state[i] = fast_field_add(state[i], ROUND_CONSTANTS[0][i]);
    }

    // === ROUNDS 1-4: normal ===
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
