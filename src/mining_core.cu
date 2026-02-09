// ============================================================================
// mining_core.cu — Unity build for all mining-critical CUDA code.
//
// Combines tip5.cu, digest.cu, merkle.cu, pow.cu, kernels.cu into a single
// translation unit (TU). This gives the compiler maximum visibility for:
//   - Cross-function inlining without DLTO overhead
//   - Global register allocation across the entire call tree
//   - Better instruction scheduling and dead code elimination
//   - Eliminates all cross-TU function call overhead
//
// The individual .cu files remain for code navigation and are still
// compilable standalone (without MINING_CORE_UNITY defined).
// ============================================================================

// Signal to individual .cu files that they're part of the unity build.
#define MINING_CORE_UNITY

// Order matters: base → consumers. Headers have include guards so
// each is processed exactly once regardless of include chains.
//
// 1. tip5.cu   — defines __constant__ MDS_COEFF, ROUND_CONSTANTS, LOOKUP_TABLE
//                and all Tip5 hash functions (sbox, mds, permutation)
// 2. digest.cu — tip5_hash_fixed_device, tip5_hash_varlen_device
// 3. merkle.cu — Merkle tree build kernels, path computation
// 4. kernels.cu — defines __constant__ d_gpu_range_*, d_top_tree_cache,
//                 mining kernels
// 5. pow.cu    — Pow_indices_device, hash_pow_encoding, fast_mast_hash,
//                GuesserBuffer, launch helpers

#include "tip5.cu"
#include "digest.cu"
#include "merkle.cu"
#include "kernels.cu"
#include "pow.cu"
