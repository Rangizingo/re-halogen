// Rewrites of k_hcup8<N> (N = 1..4) and k_hcup8w<N> (N = 5..8): the standalone hc8 path of the hybrid-code projection, from flash_serve, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o hcup8.co hcup8.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). "h1" = k_hcup8<1> (starts :305930), "w5" = k_hcup8w<5> (starts :311848).
//
// k_hcup8<N>(unsigned char const* W, ushort const* x, ushort const* wm, ushort* y, float* f): kernarg W@0, x@8, wm@16, y@24, f@32
// (s_load_b256 s[4:11] h1 :305932, s_load_b64 s[4:5] at 0x20 h1 :305950). x is read at lane*32 bytes (global_load_b128 v44, s[6:7]: h1 :306075, :306083).
// It is the hc8 path of k_hc8_fused<N> with no LqGrp role, no counter wait and block offset 0: row n = blockIdx.x*16 + g (v_or3 v36, v1, s2<<4, v2:
// h1 :305939), plane stride 0xd4000 (870400, h1 :305958+), row stride 0x154 (340, h1 :305945). The decode / dot / sigmoid math below is the
// hc8.hip (g11) body, whose cites are in codeobj_53000.s under k_hc8_fused<1> (starts :167332); only the entry differs.
// k_hcup8w<N> is the N >= 5 form (4 rows per block, plane = (tid >> 4) & 3, LDS combine), as in k_hc8_fused<N> N >= 5.
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

// One 16-byte chunk of u8 codes to eight packed bf16 pairs. w = bf16_rne(fma(float(byte), scale, bias)), scale = low f16, bias = high f16 of the
// group dword (v_cvt_f32_ubyte0..3 then v_fma_mix_f32 v, v, s, s with op_sel [0,0,1], op_sel_hi [0,1,1]: f1 :167448-167468, :167476-167493).
// Pair m holds bytes 2m (low half) and 2m+1 (high half): v_and_or_b32 0xffff0000 merge (f1 :167499).
__device__ static inline void dequant_chunk(uint4 c, unsigned grp, unsigned pk[8]) {
  const float s = (float)__builtin_bit_cast(_Float16, (unsigned short)(grp & 0xffffu));
  const float b = (float)__builtin_bit_cast(_Float16, (unsigned short)(grp >> 16));
  const unsigned dw[4] = {c.x, c.y, c.z, c.w};
  for (int m = 0; m < 8; m++) {
    const unsigned d = dw[m >> 1];
    const int sh = (m & 1) * 16;
    float w0 = __builtin_fmaf((float)((d >> sh) & 0xffu), s, b);
    float w1 = __builtin_fmaf((float)((d >> (sh + 8)) & 0xffu), s, b);
    pk[m] = (rne_bits(w1) & 0xffff0000u) | (rne_bits(w0) >> 16);
  }
}

// Row layout of the code tensor: [plane 0..3][row 0..2559][340 bytes] (plane stride 0xd4000 = 2560*340 :167357-167381; row stride 0x154 :167357).
// A row is 20 chunks of 16 code bytes (320 weights) followed by 5 group dwords at +320 + 4*group (group = chunk >> 2) (:167384, :167407).

// hc8 path of k_hc8_fused<N>, N = 1..4. Block (after nblk) b' = blockIdx.x - nblk; lane = tid & 15, g = tid >> 4; row n = b'*16 + g
// (v_or3 s3<<4, g & 14, g & 1 :167353). Lane l reads chunk l, lanes 0..3 also chunk 16 + l (v_cmpx 20 > v45 :167397).
// x set j = x + 320*j bf16 (offset 640 bytes per set, <2> :169183-169186); lane reads 16 bf16 at 16*chunk.
// Per plane k and set j: acc1 = chain of 8 dot2 over the chunk-l pairs starting from 0; total = 0 + acc1 (:167822-167823), and for lanes < 4
// total += chain2 over chunk 16 + l (:167826-168206). Then xor-butterfly 8,4,2,1 over the 16 lanes (:168212-168263), lane 0 of the group finishes.
// Epilogue per set j (same as k_hcup4<A,0>): s_k = sigmoid_widen(widen_rne(P_jk)); m = fma chain over planes 0..3 with wm[j*10240 + k*2560 + n]
// (wm = kernarg 0x128, :168271-168295, :168398-168409); out = m * 0.25 (:168411); f[j*2560 + n] = out if f != 0 (:168412-168419),
// y[j*2560 + n] = bf16_rne(out) (:168421-168427) (set 1 at +2560 elements: <2> :170427-170440).
template <int N>
__device__ static inline void hc8_small(const unsigned short* wm, const unsigned char* W, const unsigned short* x, unsigned short* y, float* f, int nblk) {
  const unsigned tid = threadIdx.x;
  const unsigned lane = tid & 15u, g = tid >> 4;
  const long n = (long)(blockIdx.x - nblk) * 16 + g;
  float p[N][4];
  for (int j = 0; j < N; j++) p[j][0] = p[j][1] = p[j][2] = p[j][3] = 0.f;
  for (int k = 0; k < 4; k++) {
    const unsigned char* row = W + (long)k * 870400 + n * 340;
    const uint4 c1 = *(const uint4*)(row + lane * 16);
    const unsigned g1 = *(const unsigned*)(row + 320 + (lane >> 2) * 4);
    unsigned pk1[8];
    dequant_chunk(c1, g1, pk1);
    unsigned pk2[8];
    if (lane < 4) {
      const uint4 c2 = *(const uint4*)(row + (16 + lane) * 16);
      const unsigned g2 = *(const unsigned*)(row + 336);
      dequant_chunk(c2, g2, pk2);
    }
    for (int j = 0; j < N; j++) {
      const unsigned* x1 = (const unsigned*)(x + 320 * j + 16 * lane);
      float a1 = 0.f;
      for (int m = 0; m < 8; m++) a1 = dot2(x1[m], pk1[m], a1);
      float tot = 0.0f + a1;
      if (lane < 4) {
        const unsigned* x2 = (const unsigned*)(x + 320 * j + 16 * (16 + lane));
        float a2 = 0.f;
        for (int m = 0; m < 8; m++) a2 = dot2(x2[m], pk2[m], a2);
        tot = tot + a2;
      }
      p[j][k] = tot;
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

// hc8 path of k_hc8_fused<N>, N = 5..8 (f5 :176258). One block covers 4 rows and the 4 planes: group gi = tid >> 4, plane k = gi & 3, row rr = gi >> 2,
// n = (blockIdx.x - nblk)*4 + rr (v_bfe_u32 v0,4,2 * 0xa00 + (tid >> 6) + s3<<2: f5 :176269-176281; code row (plane*2560 + n)*340 :176283).
// Each group dots its single code row with all N x sets (x offsets 0, 640, 1280, ... :176328-176337) with the same chunk/dequant/chain order as hc8_small
// (total = 0 + chain1 [+ chain2 for lanes < 4], :176475-176478), xor-butterfly 8,4,2,1 (:176650-176698), then lane 0 stores the N totals as bf16 RNE
// into LDS at gi*2N bytes (ds_store_b64 + b16 for N = 5: :176705-176730). After a barrier threads tid < 4N take set j = tid % N, row rr = tid / N (:176737-176757),
// read the four plane values at (rr*4 + k)*2N + 2j (:176779-176785), and run the hc8_small epilogue with n = b'*4 + rr (:176787-176912):
// s_k = sigmoid_widen(bf16 value), m = fma chain of wm[j*10240 + k*2560 + n] * s_k, out = m * 0.25, f[j*2560 + n] = out if f != 0, y[j*2560 + n] = bf16_rne(out).
template <int N>
__device__ static inline void hc8_big(const unsigned short* wm, const unsigned char* W, const unsigned short* x, unsigned short* y, float* f, int nblk) {
  __shared__ unsigned short lds[16 * N];
  const unsigned tid = threadIdx.x;
  const unsigned lane = tid & 15u, gi = tid >> 4;
  const unsigned k = gi & 3u, rr = gi >> 2;
  const long b = (long)(blockIdx.x - nblk) * 4;
  const unsigned char* row = W + (long)k * 870400 + (b + rr) * 340;
  const uint4 c1 = *(const uint4*)(row + lane * 16);
  const unsigned g1 = *(const unsigned*)(row + 320 + (lane >> 2) * 4);
  unsigned pk1[8];
  dequant_chunk(c1, g1, pk1);
  unsigned pk2[8];
  if (lane < 4) {
    const uint4 c2 = *(const uint4*)(row + (16 + lane) * 16);
    const unsigned g2 = *(const unsigned*)(row + 336);
    dequant_chunk(c2, g2, pk2);
  }
  float p[N];
  for (int j = 0; j < N; j++) {
    const unsigned* x1 = (const unsigned*)(x + 320 * j + 16 * lane);
    float a1 = 0.f;
    for (int m = 0; m < 8; m++) a1 = dot2(x1[m], pk1[m], a1);
    float tot = 0.0f + a1;
    if (lane < 4) {
      const unsigned* x2 = (const unsigned*)(x + 320 * j + 16 * (16 + lane));
      float a2 = 0.f;
      for (int m = 0; m < 8; m++) a2 = dot2(x2[m], pk2[m], a2);
      tot = tot + a2;
    }
    p[j] = tot;
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

#define HCUP8(N)                                                                                                    \
  extern "C" __global__ void k_hcup8_##N(const unsigned char* W, const unsigned short* x, const unsigned short* wm, \
                                         unsigned short* y, float* f) { hc8_small<N>(wm, W, x, y, f, 0); }
HCUP8(1) HCUP8(2) HCUP8(3) HCUP8(4)
#define HCUP8W(N)                                                                                                    \
  extern "C" __global__ void k_hcup8w_##N(const unsigned char* W, const unsigned short* x, const unsigned short* wm, \
                                          unsigned short* y, float* f) { hc8_big<N>(wm, W, x, y, f, 0); }
HCUP8W(5) HCUP8W(6) HCUP8W(7) HCUP8W(8)
