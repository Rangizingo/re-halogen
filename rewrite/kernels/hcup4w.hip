// Rewrite of k_hcup4w (bf16 tensor-core version of the hcup4 mix, batched over tokens) from flash_serve, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o hcup4w.co hcup4w.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). Line cites below are in that file.
#include <hip/hip_runtime.h>

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
__device__ static inline unsigned rne_bits(float f) {
  unsigned u = __float_as_uint(f);
  return u + ((u >> 16) & 1u) + 0x7fffu;  // bfe 16,1 ; add3 0x7fff (:34031-34033, :34049-34056)
}
__device__ static inline float widen_rne(float f) { return __uint_as_float(rne_bits(f) & 0xffff0000u); }
__device__ static inline float sigmoid_widen(float r) {  // :34059-34145: and 0xffff0000, *-1.442695, v_exp_f32, +1.0, IEEE 1/x, bfe/add3, and 0xffff0000
  float e = __builtin_amdgcn_exp2f(r * -1.44269502162933349609375f);
  return widen_rne(1.0f / (1.0f + e));
}

// halogen::(anonymous namespace)::k_hcup4w(unsigned short const*, unsigned short const*, unsigned short const*, unsigned short*,
//                                          float*, unsigned short*, int)
// _ZN7halogen12_GLOBAL__N_18k_hcup4wEPKtS2_S2_PtPfS3_i, kernarg: W@0, X@8, wm@16, y@24, f@32, lo@40, ntok@48 (s_load :33742-33744).
// Grid (ceil(ntok/128), 80, 1), block 256 (= 8 wave32). Wave w of block (bx, by) owns tokens bx*128 + w*16 .. +16 (:33750).
//
// W is 4 planes x 2560 rows x 320 bf16 (row stride 0x280 bytes :33854, plane stride 0xa00 rows :33850); X is ntok x 320 bf16 (row stride 0x280 :33770).
// Block by covers n = by*32 .. by*32+31 in every plane: LDS holds 4 planes x 32 rows (:33836-33862, 0x1400 bytes per plane, 0xa0 bytes per row),
// read back as 8 tiles of 16 rows each (ds offsets 0, 2560, ..., 17920 :33880-33910). Tile t = plane t/2, n-block t%2 (acc regs: v33, v1, v41, v9, v49, v17, v57, v25).
// The rewrite reads W straight from global; the K values per fragment are the same.
// K loop: 5 outer steps of 64 (s0 += 64 until > 0xff :33884-33885), 4 WMMA k-steps of 16 each (:33888-34008).
//   A fragment (16 bf16 of one token row, row = min(ntok-1, bx*128 + w*16 + (lane&15)) :33750-33764): X[row][K .. K+16) (:33872-33879).
//   B fragment (16 bf16 of one W row, column = lane&15 of the tile): W[plane][n][K .. K+16) (LDS rows 0xa0*(lane&15) :33765, :33880-33881).
//   v_wmma_f32_16x16x16_bf16 D, A, B, D with the same tile order per k-step (:33888-33916); lanes 16..31 repeat lanes 0..15 in A and B.
// Epilogue: acc reg i of lane l is token bx*128 + w*16 + 2*i + (l >> 4) (:34016, :34215), column n = by*32 + (l & 15) (+16 for the odd tiles) (:33755, :34010).
// Guard token < ntok (v_cmp :34022). Per output element it is the k_hcup4 set-0 epilogue (:34031-34213):
//   s_k = bf16 widened (1 / (1 + exp2(-1.442695 * widen_rne(P_k)))), P_k the plane-k accumulator,
//   m = fma(wm[t][0][n], s_0, 0) then fmac planes 1..3 (:34147-34172), wm token stride 0x5000 bytes = 10240 bf16, plane stride 2560 (:34030-34048),
//   out = m * 0.25 (:34176); y[t*2560 + n] = bf16_rne(out) (:34183-34185), f[t*2560 + n] = out when f != 0 (:34187-34193),
//   lo[t*5120 + n] = bf16(out), lo[t*5120 + 2560 + n] = bf16_rne(out - widen(bf16(out))) when lo != 0 (:34196-34213).
extern "C" __global__ void k_hcup4w(const unsigned short* W, const unsigned short* X, const unsigned short* wm, unsigned short* y, float* f,
                                    unsigned short* lo, int ntok) {
  const unsigned tid = threadIdx.x;
  const unsigned lane = tid & 31u, wave = tid >> 5;
  const unsigned bx = blockIdx.x, by = blockIdx.y;
  const int arow = min(ntok - 1, (int)(bx * 128 + wave * 16 + (lane & 15)));
  const unsigned short* xr = X + (size_t)arow * 320;
  v8f acc[8];
  for (int t = 0; t < 8; t++) acc[t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
  const unsigned ncol = by * 32 + (lane & 15);
  for (int k0 = 0; k0 < 320; k0 += 16) {
    v16bf a = *(const v16bf*)(xr + k0);
    for (int t = 0; t < 8; t++) {
      const unsigned short* wr = W + ((size_t)(t >> 1) * 2560 + ncol + (t & 1) * 16) * 320;
      v16bf b = *(const v16bf*)(wr + k0);
      acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[t]);
    }
  }
  for (int nb = 0; nb < 2; nb++) {
    const unsigned n = ncol + nb * 16;
    for (int i = 0; i < 8; i++) {
      const int tok = (int)(bx * 128 + wave * 16 + 2 * i + (lane >> 4));
      if (tok >= ntok) continue;
      const unsigned short* wt = wm + (size_t)tok * 10240;
      float s[4];
      for (int k = 0; k < 4; k++) s[k] = sigmoid_widen(widen_rne(acc[2 * k + nb][i]));
      float m = __builtin_fmaf(bf2f(wt[0 * 2560 + n]), s[0], 0.0f);
      m = __builtin_fmaf(bf2f(wt[1 * 2560 + n]), s[1], m);
      m = __builtin_fmaf(bf2f(wt[2 * 2560 + n]), s[2], m);
      m = __builtin_fmaf(bf2f(wt[3 * 2560 + n]), s[3], m);
      float out = m * 0.25f;
      unsigned ob = rne_bits(out);
      y[(size_t)tok * 2560 + n] = (unsigned short)(ob >> 16);
      if (f) f[(size_t)tok * 2560 + n] = out;
      if (lo) {
        lo[(size_t)tok * 5120 + n] = (unsigned short)(ob >> 16);
        float rem = out - __uint_as_float(ob & 0xffff0000u);
        lo[(size_t)tok * 5120 + 2560 + n] = (unsigned short)(rne_bits(rem) >> 16);
      }
    }
  }
}
