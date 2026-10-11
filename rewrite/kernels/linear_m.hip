// Rewrites of k_linear_m<M> (M = 4, 8, 16, 32) from codeobj_53000, gfx1151: one block, M dot products against one shared vector.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o linear_m.co linear_m.hip
// Listing: kernels-disasm/codeobj_53000.s. Starts: <4> 853917, <16> 854123, <32> 854425, <8> 854910.
// Undecoded .long words (VOP3 forms llvm-objdump does not know) are read as the i64 address/compare ops around them (inferred);
// the MATCH rows settle every such reading.
#include <hip/hip_runtime.h>

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
// v_bfe_u32 v1,v0,16,1 ; v_add3_u32 v0,v0,v1,0x7fff ; global_store_d16_hi_b16 (:854116-854119).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// k_linear_m<M>(a, b, c, K, ldc), kernarg a@0, b@8, c@16, K@24, ldc@32.  blockDim.x = 256 (group_segment = M*1024 B in kernargs.json).
//   x = blockIdx.x, y = blockIdx.y. The M rows of `a` are x*M .. x*M+M-1, each K elements (row base s16 = x<<2 for M=4, :853936, 853944-853962,
//   then rows 1..3 at (x*8|2),(x*8|4),(x*8|6) * K bytes, :853963-853992). b is row y of length K (b += y*K, :853940-853947).
//   Thread t: for i = t; i < K; i += blockDim.x (do-while, :854025): bv = bf(b[i]); acc[j] = fma(bv, bf(a_j[i]), acc[j]) (v_fmac_f32 :854035-854039),
//   acc starts at 0.
//   Each acc[j] goes to LDS slot j (256 floats apart: ds_store_2addr_stride64 offsets 0,4 / 8,12 = 1024 B apart, :854051-854052).
//   Block tree per slot: for (st = blockDim/2; st > 0; st >>= 1) if (t < st) slot[t] = slot[t+st] + slot[t]; barrier (:854060-854095).
//   Threads t < M (v_cmpx_gt_u32 4, :854100) read slot t element 0 (v_mad_u32_u24 0x3fc*t + 4t = 1024*t, :854102) and write
//   c[y*ldc + x*M + t] = bf16_rne(sum) (:854103-854119).
template <typename OutT, bool F32, int M>
__device__ static inline void linear_m_body(const unsigned short* a, const unsigned short* b, OutT* c, long K, long ldc, float* sm) {
  long x = blockIdx.x, y = blockIdx.y;
  const unsigned short* ar = a + x * M * K;
  const unsigned short* br = b + y * K;
  unsigned t = threadIdx.x;
  float acc[M];
#pragma unroll
  for (int j = 0; j < M; j++) acc[j] = 0.f;
  for (long i = t; i < K; i += blockDim.x) {
    float bv = bf2f(br[i]);
#pragma unroll
    for (int j = 0; j < M; j++) acc[j] = __builtin_fmaf(bv, bf2f(ar[j * K + i]), acc[j]);
  }
#pragma unroll
  for (int j = 0; j < M; j++) sm[j * 256 + t] = acc[j];
  __syncthreads();
  for (unsigned st = blockDim.x >> 1; st > 0; st >>= 1) {
    if (t < st) {
#pragma unroll
      for (int j = 0; j < M; j++) sm[j * 256 + t] = sm[j * 256 + t + st] + sm[j * 256 + t];
    }
    __syncthreads();
  }
  if (t < M) {
    float r = sm[t * 256];
    if (F32) c[y * ldc + x * M + t] = r;
    else c[y * ldc + x * M + t] = f2bf_rne(r);
  }
}

#define LINEAR_M(M)                                                                                                        \
  extern "C" __global__ void k_linear_m_##M(const unsigned short* a, const unsigned short* b, unsigned short* c, long K, long ldc) { \
    __shared__ float sm[M * 256];                                                                                          \
    linear_m_body<unsigned short, false, M>(a, b, c, K, ldc, sm);                                                          \
  }
LINEAR_M(4) LINEAR_M(8) LINEAR_M(16) LINEAR_M(32)
