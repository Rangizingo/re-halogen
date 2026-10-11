// Rewrites of the "conv1d" family from the vendor code object codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o conv1d.co conv1d.hip
// Listing cites are addresses in codeobj_53000 (ROCm llvm-objdump 23 --mcpu=gfx1151 decode; the repo listing
// kernels-disasm/codeobj_53000.s shows some 8-byte VOP3 words as ".long", the address is the same in both).
//
// Shared model (all four kernels, 10240 channels, rows of 0x5000 B = 10240 bf16):
//   X(j)  = j >= 0 ? xin[j*10240 + c] : st[(j+3)*10240 + c]      (history rows j = -3..-1 live in st rows 0..2)
//   acc   = fma(X(t-3),w0,0); acc = fma(X(t-2),w1,acc); acc = fma(X(t-1),w2,acc); acc = fma(X(t),w3,acc)
//   y     = bf16_rne(silu(acc))   silu: e = exp2(acc * -1.4426950216), y = acc / (1 + e) IEEE division
//   w[c*4 + k] is bf16, tap k multiplies X(t-3+k).
#include <hip/hip_runtime.h>

#define NCH 10240          // v_cmpx_gt_i32 0x2800, v0 (e.g. @8AA024)
#define ROW 10240

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
// v_bfe_u32 v,x,16,1 ; v_add3_u32 y,x,v,0x7fff ; store high half (@8AA2A4-8AA2B0, @8AA2CC)
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
// v_mul_f32 0xbfb8aa3b ; v_exp_f32 ; v_add_f32 1.0 ; v_div_scale/rcp/fma/div_fmas/div_fixup (@8AA234-8AA29C)
__device__ static inline float silu_f(float v) {
  float e = __builtin_amdgcn_exp2f(v * -1.44269502162933349609375f);
  return v / (1.0f + e);
}

__device__ static inline float X(const unsigned short* xin, const unsigned short* st, int j, int c) {
  return bf2f(j >= 0 ? xin[(long)j * ROW + c] : st[(long)(j + 3) * ROW + c]);
}

// k_conv1d_silu<false>: _ZN7halogen12_GLOBAL__N_113k_conv1d_siluILb0EEEvPKtS3_S3_Pti
// kernarg: xin@0 (s[4:5]), st@8 (s[6:7]), w@16 (s[8:9]), y@24 (s[10:11]), n@32 (never loaded, unused).
// grid: x = token t (s2), y = channel block (s3); c = blockIdx.y*blockDim.x + tid, guard c < 10240 (@8AA018-8AA02C).
// Tap source choices @8AA044/8AA0A0/8AA108/8AA158: t>2 ? xin row t-3 : st row t; t>1 ? t-2 : st row t+1; t>0 ? t-1 : st row t+2; xin row t.
// Weights: global_load_b64 at w + 8c (@8AA11C). y at y + t*0x5000 + 2c (@8AA1C0-8AA2CC).
template <bool CH_X>
__device__ static inline void conv1d_silu_body(const unsigned short* xin, const unsigned short* st,
                                               const unsigned short* w, unsigned short* y) {
  int t = CH_X ? blockIdx.y : blockIdx.x;
  int c = (CH_X ? blockIdx.x : blockIdx.y) * blockDim.x + threadIdx.x;
  if (c >= NCH) return;
  float w0 = bf2f(w[c * 4 + 0]), w1 = bf2f(w[c * 4 + 1]), w2 = bf2f(w[c * 4 + 2]), w3 = bf2f(w[c * 4 + 3]);
  float a = X(xin, st, t - 3, c);
  float b = X(xin, st, t - 2, c);
  float d = X(xin, st, t - 1, c);
  float e = X(xin, st, t, c);
  float acc = __builtin_fmaf(a, w0, 0.0f);     // v_fma_f32 v1,v4,v1,0 (@8AA204)
  acc = __builtin_fmaf(b, w1, acc);            // v_fmac_f32 (@8AA218)
  acc = __builtin_fmaf(d, w2, acc);            // (@8AA228)
  acc = __builtin_fmaf(e, w3, acc);            // (@8AA22C)
  y[(long)t * ROW + c] = f2bf_rne(silu_f(acc));
}

// k_conv1d_silu<true>: _ZN7halogen12_GLOBAL__N_113k_conv1d_siluILb1EEEvPKtS3_S3_Pti (@8A9D00)
// Same math; the grid axes are swapped: channel block = blockIdx.x (s2), token = blockIdx.y (s3).
extern "C" __global__ void k_conv1d_silu_b0(const unsigned short* xin, const unsigned short* st,
                                            const unsigned short* w, unsigned short* y, int n) {
  conv1d_silu_body<false>(xin, st, w, y);
}
extern "C" __global__ void k_conv1d_silu_b1(const unsigned short* xin, const unsigned short* st,
                                            const unsigned short* w, unsigned short* y, int n) {
  conv1d_silu_body<true>(xin, st, w, y);
}

// k_conv1d_silu_tb<16>: _ZN7halogen12_GLOBAL__N_116k_conv1d_silu_tbILi16EEEvPKtS3_S3_Pti (@8A9500)
// kernarg as above but n@32 is the token count. c = blockIdx.x*blockDim.x + tid (@8A9518); t0 = blockIdx.y*16 (s_lshl_b32 s2,s3,4 @8A953C).
// Each thread walks t = t0 .. min(t0+16, n)-1 (count = min(max(n,t0)-t0, 15)+1 with a t < n guard per step: s_cmp_ge_i32 s6,s0 @8A9740),
// keeping the 3-deep tap window in registers (v_mov shifts @8A9868-8A9870). Math per step is the same as k_conv1d_silu.
extern "C" __global__ void k_conv1d_silu_tb16(const unsigned short* xin, const unsigned short* st,
                                              const unsigned short* w, unsigned short* y, int n) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= NCH) return;
  int t0 = blockIdx.y * 16;
  float w0 = bf2f(w[c * 4 + 0]), w1 = bf2f(w[c * 4 + 1]), w2 = bf2f(w[c * 4 + 2]), w3 = bf2f(w[c * 4 + 3]);
  float a = X(xin, st, t0 - 3, c);
  float b = X(xin, st, t0 - 2, c);
  float d = X(xin, st, t0 - 1, c);
  for (int t = t0; t < t0 + 16 && t < n; ++t) {
    float e = bf2f(xin[(long)t * ROW + c]);
    float acc = __builtin_fmaf(a, w0, 0.0f);
    acc = __builtin_fmaf(b, w1, acc);
    acc = __builtin_fmaf(d, w2, acc);
    acc = __builtin_fmaf(e, w3, acc);
    y[(long)t * ROW + c] = f2bf_rne(silu_f(acc));
    a = b; b = d; d = e;
  }
}

// k_conv1d_step: _ZN7halogen12_GLOBAL__N_113k_conv1d_stepEPKtS2_PtS3_iS2_ (@23BC00 in the repo listing, 0x23bc00)
// kernarg: x@0 (s[4:5]), w@8 (s[6:7]), state@16 (s[8:9]), y@24 (s[10:11]), flag@32 (s0), state2@40 (s[2:3]).
// 1D grid, c = blockIdx.x*blockDim.x + tid, guard c < 10240 (@23BC18-23BC2C).
// state is 4 bf16 per channel (8 B). Read base = state2 ? state2 : state (s_cmp_eq_u64 s[2:3],0 @23BC7C);
// read three slots starting at slot (flag != 1 ? 1 : 0) (v_cndmask 1/0 << 1 @23BCA0-23BCBC; b32 + d16 at +4 @23BCD8-23BCF8).
// acc = fma(s_a,w0,0); fma(s_b,w1); fma(s_c,w2); fma(x,w3) (@23BD4C-23BD90); y = bf16_rne(silu(acc)) at y+2c (@23BE3C-23BE80).
// New state {s_a, s_b, s_c, x} is stored to state (not state2) at +8c (global_store_b64 @23BE78). The bf16 round trips
// (and 0x10001 / or / add 0x7fff) are the identity on bf16 bit patterns.
extern "C" __global__ void k_conv1d_step(const unsigned short* x, const unsigned short* w, unsigned short* state,
                                         unsigned short* y, int flag, const unsigned short* state2) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= NCH) return;
  const unsigned short* rd = state2 ? state2 : state;
  int off = (flag != 1) ? 1 : 0;
  unsigned short sa = rd[c * 4 + off], sb = rd[c * 4 + off + 1], sc = rd[c * 4 + off + 2];
  unsigned short xv = x[c];
  float acc = __builtin_fmaf(bf2f(sa), bf2f(w[c * 4 + 0]), 0.0f);
  acc = __builtin_fmaf(bf2f(sb), bf2f(w[c * 4 + 1]), acc);
  acc = __builtin_fmaf(bf2f(sc), bf2f(w[c * 4 + 2]), acc);
  acc = __builtin_fmaf(bf2f(xv), bf2f(w[c * 4 + 3]), acc);
  state[c * 4 + 0] = sa; state[c * 4 + 1] = sb; state[c * 4 + 2] = sc; state[c * 4 + 3] = xv;
  y[c] = f2bf_rne(silu_f(acc));
}
