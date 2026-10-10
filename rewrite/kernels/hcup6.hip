// Rewrites of k_hcup6<N> (N = 1..4) and k_hcup6w<N> (N = 5..8): the standalone 6-bit hybrid-code projection of the four hyper-connection streams,
// from flash_serve, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o hcup6.co hcup6.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). Cites are lines in that file; "u1" = k_hcup6<1> (:252490-253917), "w5" = k_hcup6w<5> (:259816).
//
// k_hcup6<N>(unsigned char const* W, ushort const* x, ushort const* wm, ushort* y, float* f): kernarg W@0, x@8, wm@16, y@24, f@32
// (s_load_b256 s[4:11] :252492, s_load_b64 s[12:13] at 0x20 :252493). Same shape as k_hcup8 / k_hc8 (hc8.hip, g11), with a 6-bit code row:
//   row layout (u1 :252508 v_mad_i64_i32 0x110 * n; plane stride 0xaa000 = 2560 * 272 :252518-252566): 272 bytes = 160 bytes of low nibbles (32 weights
//   per lane in 16 bytes, lane * 16: global_load_b128 :252529, :252533, :252545), then 80 bytes of 2-bit high parts (8 bytes per lane at +160 + lane * 8:
//   global_load_b64 offset:160 :252530, :252534, :252546), then 5 group dwords at +240 + 4 * (lane >> 1) (global_load_b32 offset:240 :252552,
//   :252564-252566; v_and 28 of tid*2 :252510). Only lanes 0..9 of a 16-lane group load (v_cmpx_gt_u32 10 :252505), 10 * 32 = 320 weights per row.
//   plane k = 0..3 at W + k * 0xaa000; row n = blockIdx.x * 16 + g, g = tid >> 4 (v_or3 :252500).
//   code of weight w (0..31) in a lane: q = nibble_w | ((hi >> 2w) & 3) << 4 (v_bfe_u32 4,4 / v_and 15 for the nibble, v_and_or_b32 (hi shifted) & 48
//   for the high bits: :252558-252700). weight = bf16_rne(fma(float(q), scale, bias)), scale = low f16 and bias = high f16 of the group dword
//   (v_cvt_f32_ubyte0 then v_fma_mix_f32 v, v, g, g op_sel [0,0,1] op_sel_hi [0,1,1]: :252767-252832), same as hc8.
//   dot: chain of 16 v_dot2_f32_bf16 from 0 over the 16 weight pairs (pair m = weights 2m (low), 2m+1) against x[lane * 32 + 2m] (:252596+ dot2 v87,v5,v87,0).
// x set j is at x + 640 * j bytes (as hc8). After the dots: xor-butterfly 8, 4, 2, 1 over the 16 lanes, then lane 0 runs the hc8 epilogue
// (sigmoid at :253786-253885 [v_mul 0xbfb8aa3b, v_exp_f32, v_div_scale / div_fmas / div_fixup], fma chain with wm, v_mul 0x3e800000 :253898,
// y store d16_hi :253914, f store :253906).
// k_hcup6w<N> (N >= 5): 4 rows per block, plane = (tid >> 4) & 3, bf16-rounded N sums through LDS, then a combine step (as hc8_big).
#include <hip/hip_runtime.h>

typedef __bf16 v2bf __attribute__((ext_vector_type(2)));
__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
__device__ static inline unsigned rne_bits(float f) {
  unsigned u = __float_as_uint(f);
  return u + ((u >> 16) & 1u) + 0x7fffu;  // v_bfe_u32 16,1 ; v_add3_u32 0x7fff (f1 :167470-167475)
}
__device__ static inline float widen_rne(float f) { return __uint_as_float(rne_bits(f) & 0xffff0000u); }
__device__ static inline float dot2(unsigned a, unsigned b, float c) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(v2bf, a), __builtin_bit_cast(v2bf, b), c, false);
}
__device__ static inline float sigmoid_widen(float r) {  // f1 :168302-168396
  float e = __builtin_amdgcn_exp2f(r * -1.44269502162933349609375f);
  return widen_rne(1.0f / (1.0f + e));
}

// Decode one lane's 32 weights: nibbles nib[4] (16 bytes), high parts hi (8 bytes), group dword grp -> 16 packed bf16 pairs.
__device__ static inline void dequant6(uint4 nib, uint2 hi, unsigned grp, unsigned pk[16]) {
  const float s = (float)__builtin_bit_cast(_Float16, (unsigned short)(grp & 0xffffu));
  const float b = (float)__builtin_bit_cast(_Float16, (unsigned short)(grp >> 16));
  const unsigned nd[4] = {nib.x, nib.y, nib.z, nib.w};
  const unsigned long long h = ((unsigned long long)hi.y << 32) | hi.x;
  for (int m = 0; m < 16; m++) {
    const int w0 = 2 * m, w1 = 2 * m + 1;
    unsigned q0 = ((nd[w0 >> 3] >> (4 * (w0 & 7))) & 15u) | ((unsigned)((h >> (2 * w0)) & 3u) << 4);
    unsigned q1 = ((nd[w1 >> 3] >> (4 * (w1 & 7))) & 15u) | ((unsigned)((h >> (2 * w1)) & 3u) << 4);
    float v0 = __builtin_fmaf((float)q0, s, b);
    float v1 = __builtin_fmaf((float)q1, s, b);
    pk[m] = (rne_bits(v1) & 0xffff0000u) | (rne_bits(v0) >> 16);
  }
}

// Row n of plane k: 272 bytes (u1 :252508), plane stride 0xaa000 = 696320 (:252518+). Returns the 16 weight pairs of lane `lane` (< 10).
__device__ static inline void load6(const unsigned char* W, int k, long n, unsigned lane, unsigned pk[16]) {
  const unsigned char* row = W + (long)k * 696320 + n * 272;
  const uint4 nib = *(const uint4*)(row + lane * 16);
  const uint2 hi = *(const uint2*)(row + 160 + lane * 8);
  const unsigned grp = *(const unsigned*)(row + 240 + (lane >> 1) * 4);
  dequant6(nib, hi, grp, pk);
}

__device__ static inline float chain16(const unsigned* xp, const unsigned pk[16]) {
  float a = 0.f;
  for (int m = 0; m < 16; m++) a = dot2(xp[m], pk[m], a);
  return a;
}

// N = 1..4 (k_hcup6): 16 rows per block, the 4 planes in-lane.
template <int N>
__device__ static inline void hcup6_small(const unsigned short* wm, const unsigned char* W, const unsigned short* x, unsigned short* y, float* f) {
  const unsigned tid = threadIdx.x;
  const unsigned lane = tid & 15u, g = tid >> 4;
  const long n = (long)blockIdx.x * 16 + g;
  float p[N][4];
  for (int j = 0; j < N; j++) p[j][0] = p[j][1] = p[j][2] = p[j][3] = 0.f;
  if (lane < 10) {
    for (int k = 0; k < 4; k++) {
      unsigned pk[16];
      load6(W, k, n, lane, pk);
      for (int j = 0; j < N; j++) p[j][k] = chain16((const unsigned*)(x + 320 * j + 32 * lane), pk);
    }
  }
  for (int off = 8; off >= 1; off >>= 1)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < 4; k++) p[j][k] = p[j][k] + __shfl_xor(p[j][k], off);
  if (lane == 0) for (int j = 0; j < N; j++) {
    const unsigned short* wj = wm + (long)j * 10240;
    float s[4];
    for (int k = 0; k < 4; k++) s[k] = sigmoid_widen(widen_rne(p[j][k]));
    float m = __builtin_fmaf(s[0], bf2f(wj[0 * 2560 + n]), 0.0f);
    m = __builtin_fmaf(s[1], bf2f(wj[1 * 2560 + n]), m);
    m = __builtin_fmaf(s[2], bf2f(wj[2 * 2560 + n]), m);
    m = __builtin_fmaf(s[3], bf2f(wj[3 * 2560 + n]), m);
    float o = m * 0.25f;
    if (f) f[(long)j * 2560 + n] = o;
    y[(long)j * 2560 + n] = (unsigned short)(rne_bits(o) >> 16);
  }
}

// N = 5..8 (k_hcup6w): 4 rows per block, group gi = tid >> 4, plane gi & 3, row gi >> 2; N sums as bf16 RNE through LDS, combine on tid < 4N.
template <int N>
__device__ static inline void hcup6_big(const unsigned short* wm, const unsigned char* W, const unsigned short* x, unsigned short* y, float* f) {
  __shared__ unsigned short lds[16 * N];
  const unsigned tid = threadIdx.x;
  const unsigned lane = tid & 15u, gi = tid >> 4;
  const unsigned k = gi & 3u, rr = gi >> 2;
  const long b = (long)blockIdx.x * 4;
  float p[N];
  for (int j = 0; j < N; j++) p[j] = 0.f;
  if (lane < 10) {
    unsigned pk[16];
    load6(W, k, b + rr, lane, pk);
    for (int j = 0; j < N; j++) p[j] = chain16((const unsigned*)(x + 320 * j + 32 * lane), pk);
  }
  for (int off = 8; off >= 1; off >>= 1)
    for (int j = 0; j < N; j++) p[j] = p[j] + __shfl_xor(p[j], off);
  if (lane == 0)
    for (int j = 0; j < N; j++) lds[gi * N + j] = (unsigned short)(rne_bits(p[j]) >> 16);
  __syncthreads();
  if (tid < 4u * N) {
    const unsigned j = tid % N, r = tid / N;
    const long n = b + r;
    const unsigned short* wj = wm + (long)j * 10240;
    float s[4];
    for (int kk = 0; kk < 4; kk++) s[kk] = sigmoid_widen(bf2f(lds[(r * 4 + kk) * N + j]));
    float m = __builtin_fmaf(bf2f(wj[0 * 2560 + n]), s[0], 0.0f);
    m = __builtin_fmaf(bf2f(wj[1 * 2560 + n]), s[1], m);
    m = __builtin_fmaf(bf2f(wj[2 * 2560 + n]), s[2], m);
    m = __builtin_fmaf(bf2f(wj[3 * 2560 + n]), s[3], m);
    float o = m * 0.25f;
    if (f) f[(long)j * 2560 + n] = o;
    y[(long)j * 2560 + n] = (unsigned short)(rne_bits(o) >> 16);
  }
}

#define HCUP6(N)                                                                                                    \
  extern "C" __global__ void k_hcup6_##N(const unsigned char* W, const unsigned short* x, const unsigned short* wm, \
                                         unsigned short* y, float* f) { hcup6_small<N>(wm, W, x, y, f); }
HCUP6(1) HCUP6(2) HCUP6(3) HCUP6(4)
#define HCUP6W(N)                                                                                                    \
  extern "C" __global__ void k_hcup6w_##N(const unsigned char* W, const unsigned short* x, const unsigned short* wm, \
                                          unsigned short* y, float* f) { hcup6_big<N>(wm, W, x, y, f); }
HCUP6W(5) HCUP6W(6) HCUP6W(7) HCUP6W(8)
