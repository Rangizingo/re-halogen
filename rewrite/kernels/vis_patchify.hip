// Rewrite of k_vis_patchify (vision patch extraction), codeobj_53000.s:64900-65054 (kernel at 0x26c900).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3
// _ZN7halogen12_GLOBAL__N_114k_vis_patchifyEPKhPtiiii, kernarg: img@0 (u8 RGB, HxWx3), out@8 (bf16), H@16, W@20, mw@24, nrows@28
#include <hip/hip_runtime.h>

// grid.x = patch row s2, 1D block. Row guard s2 >= nrows exits (:64904-64905).
// Row s2 splits into (:64906-64983, signed trunc div/mod):
//   a = s2 % 2, b = (s2 / 2) % 2, c = (s2 / 4) % mw, d = s2 / (4 * mw)
//   x0 = c*32 + a*16, y0 = d*32 + b*16        (s_lshl4_add, :64982-64983)
// Thread loop over e = tid; e < 0x300 (768), e += blockDim (:64918, v_cmp 0x2ff region near :65010).
//   dx = e & 15, dy = (e >> 4) & 15, ch = e >> 8   (:64992-64993, :65003)
//   x = min(x0+dx, W-1), y = min(y0+dy, H-1)       (v_min_i32 :64998-64999)
//   b = img[(y*W + x)*3 + ch]                       (v_mad_i64_i32 :65000, v_mad_u64_u32 x2 :65008-65010)
//   v = float(b) / 255.0f (IEEE div, 0x437f0000; :65022-65040), v = (v - 0.5f); v = v + v   (:65046-65047)
//   store hi 16 bits of v (truncation, global_store_d16_hi_b16) twice: +0 and +512 bytes (:65049-65050)
//   Output element index in the row: ((ch*2 + t)*16 + dy)*16 + dx, t = 0,1 (INFERRED from the +512 offset and the
//   0x300 x 2 = 1536 row width, row stride 0xc00 bytes :64984; confirmed only by MATCH).
extern "C" __global__ void k_vis_patchify(const unsigned char* img, unsigned short* out, int H, int W, int mw, int nrows) {
  int s2 = blockIdx.x;
  if (s2 >= nrows) return;
  int a = s2 % 2, b = (s2 / 2) % 2, c = (s2 / 4) % mw, d = s2 / (4 * mw);
  int x0 = c * 32 + a * 16, y0 = d * 32 + b * 16;
  unsigned short* row = out + (long)s2 * 1536;
  for (int e = threadIdx.x; e < 0x300; e += blockDim.x) {
    int dx = e & 15, dy = (e >> 4) & 15, ch = e >> 8;
    int x = min(x0 + dx, W - 1), y = min(y0 + dy, H - 1);
    float v = (float)img[((long)y * W + x) * 3 + ch] / 255.0f;
    v = v - 0.5f;
    v = v + v;
    unsigned short h = (unsigned short)(__float_as_uint(v) >> 16);
    int o = ((ch * 2) * 16 + dy) * 16 + dx;
    row[o] = h;
    row[o + 256] = h;
  }
}
