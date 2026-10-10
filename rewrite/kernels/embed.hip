// Rewrites of the "embed" family from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o embed.co embed.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

// k_embed_gather(unsigned short const* tab, int const* ids, unsigned short* out)
// _ZN7halogen12_GLOBAL__N_114k_embed_gatherEPKtPKiPt, kernarg: tab@0, ids@8, out@16
// One block per token: row = blockIdx.x (no count argument, no bound check). id = ids[row] (s_load_b32, codeobj_53000.s:50344).
// Copy 2560 bf16 (0xa00, :50354; bound 0x9ff, :50434) from tab[id*2560] (row stride 0x1400 bytes, :50350-50351)
// to out[row*2560] (:50346-50348). Vendor has a 4-element unrolled path for blockDim.x == 1 (:50347); same result.
extern "C" __global__ void k_embed_gather(const unsigned short* tab, const int* ids, unsigned short* out) {
  int row = blockIdx.x;
  int id = ids[row];
  const unsigned short* src = tab + (long)id * 2560;
  unsigned short* dst = out + (long)row * 2560;
  for (int j = threadIdx.x; j < 2560; j += blockDim.x) dst[j] = src[j];
}

// k_embed_image(unsigned short* out, unsigned short const* tab, int const* ids, int a, int b, int v)
// _ZN7halogen12_GLOBAL__N_113k_embed_imageEPtPKtPKiiii, kernarg: out@0, tab@8, ids@16, a@24, b@28, v@32
// One block per row: row = blockIdx.x. Exit if min(a, b) <= row (s_min_i32 s3,s4,s5 ; s_cmp_le_i32, codeobj_53000.s:65059-65062).
// id = ids[row] (:65070); exit if id < 0 or id >= v (:65072-65079). Otherwise copy 2560 bf16 from tab[id*2560] to out[row*2560]
// (strides 0x1400 bytes, :65083-65089; trip count 0xa00 :65092, bound 0x9ff :65172; unrolled path for blockDim.x == 1 :65088).
extern "C" __global__ void k_embed_image(unsigned short* out, const unsigned short* tab, const int* ids, int a, int b, int v) {
  int row = blockIdx.x;
  if (min(a, b) <= row) return;
  int id = ids[row];
  if (id < 0 || id >= v) return;
  const unsigned short* src = tab + (long)id * 2560;
  unsigned short* dst = out + (long)row * 2560;
  for (int j = threadIdx.x; j < 2560; j += blockDim.x) dst[j] = src[j];
}
