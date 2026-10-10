// Rewrites of k_linear_mt<8, T, 256> (T = 2..8 tokens) from codeobj_53000, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o linear_mt.co linear_mt.hip
// Listing: kernels-disasm/codeobj_53000.s. Starts: T=2 834978 (718 lines), 3 835696, 4 836684, 5 837938, 6 839477, 7 841297, 8 843393.
// Undecoded .long words are read as the i64 address/compare ops around them (inferred); the MATCH rows settle every such reading.
#include <hip/hip_runtime.h>

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
// v_bfe_u32 v13,v12,16,1 ; v_add3_u32 v12,v12,v13,0x7fff ; global_store_d16_hi_b16 (:835407-835410).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// k_linear_mt<8,T,256>(a, x, c, K, ldc), kernarg a@0, x@8, c@16, K@24, ldc@32 (no hidden args; blockDim.x is the template's 256).
//   Block x owns weight rows x*8 .. x*8+7 of `a` (a += x*8*K elements, :835007-835025). No row guard, so the grid must be exact.
//   Token tk's vector is x + tk*K (second token via +K*2 bytes, :835040-835044).
//   Thread t: for i = t; i < K; i += 256 (do-while, :835078-835082): acc[r][tk] = fma(bf(a_r[i]), bf(x_tk[i]), acc[r][tk]) for the 8*T
//   accumulators (v_fmac_f32 :835090-835118), start 0.
//   Reduction runs T passes of 8 slots (group_segment 8192 = 8 slots x 256 floats): pass tk stores acc[0..7][tk] into slots r*256 + t
//   (ds_store_2addr_stride64 :835129-835132), barrier, tree with st = 128,64,32,16,8,4,2,1: slot[t] = slot[t+st] + slot[t] for t < st
//   (:835136-835321, steps 128/64 use ds_load_2addr_stride64, 32/16/8/4 ds_load_2addr_b32, last two inlined :835326-835390).
//   Threads t < 8 write c[tk*ldc + x*8 + t] = bf16_rne(slot t element 0) (v_cmp_gt_u32 8 :835257; s[12:13] = ldc scaled <<1 for token 1 at
//   :835684; token 0 at :835392-835410).
template <int T>
__device__ static inline void linear_mt_body(const unsigned short* a, const unsigned short* xv, unsigned short* c, long K, long ldc, float* sm) {
  long x = blockIdx.x;
  const unsigned short* ar = a + x * 8 * K;
  unsigned t = threadIdx.x;
  float acc[8][T];
#pragma unroll
  for (int r = 0; r < 8; r++)
#pragma unroll
    for (int k = 0; k < T; k++) acc[r][k] = 0.f;
  for (long i = t; i < K; i += 256) {
    float xb[T];
#pragma unroll
    for (int k = 0; k < T; k++) xb[k] = bf2f(xv[k * K + i]);
#pragma unroll
    for (int r = 0; r < 8; r++) {
      float av = bf2f(ar[r * K + i]);
#pragma unroll
      for (int k = 0; k < T; k++) acc[r][k] = __builtin_fmaf(av, xb[k], acc[r][k]);
    }
  }
#pragma unroll
  for (int k = 0; k < T; k++) {
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 8; r++) sm[r * 256 + t] = acc[r][k];
    __syncthreads();
    for (unsigned st = 128; st > 0; st >>= 1) {
      if (t < st) {
#pragma unroll
        for (int r = 0; r < 8; r++) sm[r * 256 + t] = sm[r * 256 + t + st] + sm[r * 256 + t];
      }
      __syncthreads();
    }
    if (t < 8) c[k * ldc + x * 8 + t] = f2bf_rne(sm[t * 256]);
  }
}

#define LINEAR_MT(T)                                                                                                      \
  extern "C" __global__ void k_linear_mt_##T(const unsigned short* a, const unsigned short* xv, unsigned short* c, long K, long ldc) { \
    __shared__ float sm[8 * 256];                                                                                         \
    linear_mt_body<T>(a, xv, c, K, ldc, sm);                                                                              \
  }
LINEAR_MT(2) LINEAR_MT(3) LINEAR_MT(4) LINEAR_MT(5) LINEAR_MT(6) LINEAR_MT(7) LINEAR_MT(8)
