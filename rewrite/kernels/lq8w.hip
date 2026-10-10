// Rewrite of the k_lq8w<M, R, W> family (flash_serve, gfx1151): q8g64 GEMV, bf16 activations, bf16 output.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o lq8w.co lq8w.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). Reference instance k_lq8w<1,16,1> (symbol 0x2dfc00, :147311); cites are
// codeobj_53000.s:<line>. The other shapes (R=32, W=1/2/8, M=2..8) have the same structure as the lq8s32 instances read in
// lq8s32.hip (same prologue/loop/reduction shapes, checked by the bitwise match of every instance, not instruction by instruction).
//
// Signature: (unsigned char const* W, unsigned short const* x, unsigned short* y, long K, long N)
//   kernarg W@0 x@8 y@16 K@24 N@32 (s_load_b256 s[4:11] :147315, s_load_b64 N :147314).
// This family is the q8s32 GEMV (lq8s32.hip) with the q8g64 weight decode; everything else is the same structure:
//   M tokens, R lanes per row, W waves per row, 256 threads, rows per block = 256/(R*W), split of the K/16 groups
//   across waves by chunk = ceil(ng/W), xor-butterfly over the R lanes, LDS sum over waves (W>1) in wave order from 0.
// Differences from k_lq8s32:
//   - row stride is K + K/16 bytes with no 16-byte padding (s_add_u32 s2,s10,s0 :147330 adds K to K/16, row stride in s0 :147335,
//     no s_and_b32 -16 in the prologue);
//   - the (scale, offset) f16 pair is one 32-bit word per 64 weights at row + K + 4*(j>>2) for group j
//     (global_load_b32 :147369; address masks the lane with 12 for R=16: v_and_b32 v7,12,v0 :147357);
//   - w[i] = bf16_rne( fma((float)u8[i], scale, offset) ): v_cvt_f32_ubyte0..3 (:147394 first) then
//     v_fma_mix_f32 v,v,v19,v19 op_sel:[0,0,1] op_sel_hi:[0,1,1] (:147417 first) = v*lo(v19) + hi(v19), one rounding;
//   - bf16 pack, x loads (:147407-147408), dot2 chain and reduction (:147560-147584) as in lq8s32; final bf16 store :147594;
//   - output is always bf16 (no f32 variant).
#include <hip/hip_runtime.h>

// f16 -> f32 is exact (the f16 halves of the fma_mix operand).
__device__ static inline float h2f(unsigned short u) { return (float)__builtin_bit_cast(_Float16, u); }

typedef __bf16 v2bf __attribute__((ext_vector_type(2)));

__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
__device__ static inline float dot2(unsigned x, unsigned w, float acc) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(v2bf, x), __builtin_bit_cast(v2bf, w), acc, false);
}

template <int M, int R, int W>
__device__ static inline void lq8w_body(const unsigned char* Wt, const unsigned short* x, unsigned short* y, long K, long N) {
  constexpr int RPB = 256 / (R * W);
  __shared__ float lds[8 * M];
  const int tid = threadIdx.x;
  const int lane = tid % R;
  const int wv = (tid / R) % W;
  const int rin = tid / (R * W);
  const long row = (long)blockIdx.x * RPB + rin;
  const bool valid = row < N;
  const long lr = valid ? row : 0;
  const long ng = K / 16;
  const long rowbytes = K + K / 16;
  const unsigned char* rp = Wt + lr * rowbytes;
  const unsigned char* sc = rp + K;
  long begin = 0, end = ng;
  if (W > 1) {
    long chunk = (ng + W - 1) / W;
    begin = wv * chunk;
    end = begin + chunk < ng ? begin + chunk : ng;
  }
  float acc[M];
  for (int m = 0; m < M; m++) acc[m] = 0.0f;
  for (long j = begin + lane; j < end; j += R) {
    uint4 q4 = *(const uint4*)(rp + 16 * j);
    unsigned so = *(const unsigned*)(sc + 4 * (j >> 2));
    float s = h2f((unsigned short)(so & 0xffffu));
    float o = h2f((unsigned short)(so >> 16));
    unsigned qw[4] = {q4.x, q4.y, q4.z, q4.w};
    unsigned wp[8];
    for (int i = 0; i < 8; i++) {
      unsigned b0 = (qw[i / 2] >> (16 * (i & 1))) & 0xffu;
      unsigned b1 = (qw[i / 2] >> (16 * (i & 1) + 8)) & 0xffu;
      float f0 = __builtin_fmaf((float)b0, s, o);
      float f1 = __builtin_fmaf((float)b1, s, o);
      wp[i] = (unsigned)f2bf_rne(f0) | ((unsigned)f2bf_rne(f1) << 16);
    }
    for (int m = 0; m < M; m++) {
      const uint4* xp = (const uint4*)(x + (long)m * K + 16 * j);
      uint4 x0 = xp[0], x1 = xp[1];
      unsigned xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
      float t = 0.0f;
      for (int p = 0; p < 8; p++) t = dot2(xw[p], wp[p], t);
      acc[m] += t;
    }
  }
  float r[M];
  for (int m = 0; m < M; m++) {
    float a = acc[m];
    for (int off = R / 2; off >= 1; off >>= 1) a = a + __shfl_xor(a, off, R);
    r[m] = a;
  }
  if (W == 1) {
    if (lane == 0 && valid)
      for (int m = 0; m < M; m++) y[(long)m * N + row] = f2bf_rne(r[m]);
  } else {
    for (int m = 0; m < M; m++) {
      if (lane == 0) lds[(rin * W + wv) * M + m] = r[m];
    }
    __syncthreads();
    if (tid < RPB) {
      long orow = (long)blockIdx.x * RPB + tid;
      if (orow < N)
        for (int m = 0; m < M; m++) {
          float s = 0.0f;
          for (int w = 0; w < W; w++) s = s + lds[(tid * W + w) * M + m];
          y[(long)m * N + orow] = f2bf_rne(s);
        }
    }
  }
}

#define LQ8W(M, R, W)                                                                                              \
  extern "C" __global__ void __launch_bounds__(256) lq8w_m##M##_r##R##_w##W(                                       \
      const unsigned char* Wt, const unsigned short* x, unsigned short* y, long K, long N) {                       \
    lq8w_body<M, R, W>(Wt, x, y, K, N);                                                                            \
  }
#define LQ8W_ALL(M) LQ8W(M, 16, 1) LQ8W(M, 32, 1) LQ8W(M, 32, 2) LQ8W(M, 32, 8)
LQ8W_ALL(1) LQ8W_ALL(2) LQ8W_ALL(3) LQ8W_ALL(4) LQ8W_ALL(5) LQ8W_ALL(6) LQ8W_ALL(7) LQ8W_ALL(8)
