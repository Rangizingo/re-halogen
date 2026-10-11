// Rewrite of the "sel" family from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o sel.co sel.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

// Keep float multiply and add separate unless an explicit __builtin_fmaf is written (hipcc fuses a*b+c by default).
#pragma clang fp contract(off)

// halogen::(anonymous namespace)::k_sel_tilemask(int const*, int const*, int, int, int, unsigned int*, int, int)
// _ZN7halogen12_GLOBAL__N_114k_sel_tilemaskEPKiS2_iiiPjii
// kernarg: tab@0 (int*), cnt@8 (int*), rows_per_block@16, total_rows@20, off@24, out@32 (uint*), W@40, stride@44
// Dynamic LDS holds W words (group_segment is 0 in the metadata; every LDS address is 4*word, :60299, :60387).
// 1. t = tid; t < W: lds[t] = 0 (:60301-60320).  barrier (:60324).
// 2. Rows i in [bx*rpb, min(total, bx*rpb+rpb)) (s14 = s4*s2 :60326; s_min_i32 :60329; loop :60331-60346), in order:
//    c = cnt[i] (:60352). for t = tid; t < c; t += blockDim.x (v_cmpx_gt_i32 :60355, v_cmp_le_i32 :60375):
//      v = tab[i*stride + t] (:60358-60377);  q = v + ((v >> 31) >>> 30)  (:60380-60383, signed bias for /4);
//      lds[(q >> 7)] |= 1 << ((q >>> 2) & 31)   (ds_or_b32 at word (q>>7)*4 :60385-60391; v_lshlrev_b32 uses the
//      low 5 bits of the shift count)
//    Then thread 0 only (:60395): x = i + off; lo = (x + ((x>>31)>>>28)) >> 4, hi = ((x+1) + (((x+1)>>31)>>>28)) >> 4
//      (:60397-60408). If lo < hi nothing more (:60409-60410). Otherwise for k = hi..lo (count lo-hi+1, :60411-60413):
//      lds[k >> 5] |= 1 << (k & 31)  (:60437-60442 and the unrolled copies up to :60553).
// 3. barrier (:60555). For t = tid; t < W: out[bx*W + t] = lds[t] (:60558-60596; guarded by tid < W from :60301).
extern "C" __global__ void k_sel_tilemask(const int* tab, const int* cnt, int rpb, int total, int off, unsigned* out,
                                          int W, int stride) {
  extern __shared__ unsigned lds[];
  int tid = threadIdx.x, bs = blockDim.x, bx = blockIdx.x;
  for (int t = tid; t < W; t += bs) lds[t] = 0;
  __syncthreads();
  int i0 = rpb * bx;
  int i1 = min(total, i0 + rpb);
  for (int i = i0; i < i1; i++) {
    int c = cnt[i];
    for (int t = tid; t < c; t += bs) {
      int v = tab[(long)i * stride + t];
      int q = v + (int)((unsigned)(v >> 31) >> 30);
      atomicOr(&lds[q >> 7], 1u << (((unsigned)q >> 2) & 31));
    }
    if (tid == 0) {
      int x = i + off;
      int x1 = x + 1;
      int lo = (x + (int)((unsigned)(x >> 31) >> 28)) >> 4;
      int hi = (x1 + (int)((unsigned)(x1 >> 31) >> 28)) >> 4;
      if (!(lo < hi))
        for (int k = hi; k <= lo; k++) atomicOr(&lds[k >> 5], 1u << (k & 31));
    }
  }
  __syncthreads();
  for (int t = tid; t < W; t += bs) out[(long)bx * W + t] = lds[t];
}
