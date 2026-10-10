// Rewrite of the "count" family (flash_serve, gfx1151): k_count_diff_u32.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o count.co count.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git); the 0xd4.... / 0xd7.... ".long" lines are VOP3
// compares (v_cmpx_*) and 64-bit add/shift that llvm-objdump will not decode.
#include <hip/hip_runtime.h>

// halogen::(anonymous namespace)::k_count_diff_u32(unsigned int const*, unsigned int const*, long, unsigned int*)
// _ZN7halogen12_GLOBAL__N_116k_count_diff_u32EPKjS2_lPj, kernarg: a@0, b@8, n@16, out@24
// Grid-stride loop with a 64-bit index: i = blockIdx.x*blockDim.x + tid (v_mad_u64_u32, codeobj_53000.s:57587),
// stride = block_count_x * blockDim (s_load of block_count_x at kernarg+32, :57592; s_mul_hi/s_mul_i32 :57595-57596),
// the loop exits when n <= i (v_cmp_le_i64 s[8:9], :57603).
// Per element a[i] != b[i] (two global_load_b32, :57619-57620; v_cmpx_ne_u32, :57622).
//   The wave leader does global_atomic_add_u32 out[0] += popcount(differing lanes) (:57630-57635).
//   The lanes then take the minimum of their low 32 index bits (v_readlane_b32 loop, :57640-57647) and one
//   global_atomic_min_u32 on out[1] (offset:4, :57656). The final values equal per-lane atomics.
extern "C" __global__ void k_count_diff_u32(const unsigned* a, const unsigned* b, long n, unsigned* out) {
  long stride = (long)gridDim.x * blockDim.x;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    if (a[i] != b[i]) {
      atomicAdd(&out[0], 1u);
      atomicMin(&out[1], (unsigned)i);
    }
  }
}
