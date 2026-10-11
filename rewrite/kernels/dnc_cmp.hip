// Rewrite of k_dnc_cmp_words from the vendor code object (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dnc_cmp.co dnc_cmp.hip
// Listing: kernels-disasm/codeobj_1146000.s (not in git), kernel at lines 1768-1866. The stock listing prints some
// 64-bit VOP3 words as ".long"; cites below are raw-listing lines of the instruction (re-decoded with gfx1151 llvm-mc).
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)  // each a*b+c in this file is an explicit __builtin_fmaf where the listing has v_fma/v_fmac

// halogen::(anonymous namespace)::k_dnc_cmp_words(unsigned int const*, unsigned int const*, long, unsigned long long*)
// _ZN7halogen12_GLOBAL__N_115k_dnc_cmp_wordsEPKjS2_lPy, kernarg: a@0, b@8, n@16, out@24
// Grid-stride count of differing words, added into *out:
//   i = blockIdx.x*blockDim.x + threadIdx.x as a 64-bit product (v_mad_u64_u32, :1780)
//   stride = gridDim.x*blockDim.x (s_mul_hi_u32/s_mul_i32, :1799-1801); loop while i < n, signed 64-bit (:1782, :1818)
//   count += (a[i] != b[i])  (global_load_b32 :1809-1810, v_cmp_ne_u32 :1822, v_cndmask 0/1, 64-bit add :1824-1826)
// Then the wave sums the per-lane counts with readlane (lanes with a nonzero count) and the first such lane does one
// global_atomic_add_u64 on out (:1833-1857). Integer sums are order independent, so one atomic per lane gives the same bits.
extern "C" __global__ void k_dnc_cmp_words(const unsigned* a, const unsigned* b, long n, unsigned long long* out) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  unsigned long long c = 0;
  for (; i < n; i += stride) c += (a[i] != b[i]) ? 1ull : 0ull;
  if (c) atomicAdd(out, c);
}
