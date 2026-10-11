// Rewrite of k_linear (codeobj_53000, gfx1151): one block reduces one dot product of length K, bf16 out.
// k_linear_f32 is the same code with an f32 store; it lives in linear_f32.hip.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o linear.co linear.hip
// Listing: kernels-disasm/codeobj_53000.s
// Undecoded .long words in the listing (llvm-objdump lacks the encodings) are read as the i64 address/compare ops
// that surround them; every such reading is "inferred" and is settled by the MATCH rows.
#include <hip/hip_runtime.h>

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
// v_bfe_u32 v3,v2,16,1 ; v_add_nc_u32 ; v_add_nc_u32 0x7fff ; global_store_d16_hi_b16 (codeobj_53000.s:18721-18724, 18738).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// Shared body. kernarg: a@0, b@8, c@16, K@24 (k_linear_f32 :18372, k_linear :18592).
//   a row = blockIdx.x (s2: a += x*K elements, :18615-18627 area), b row = blockIdx.y (s3->s4: b += y*K).
//   Thread t strides i = t, t+blockDim.x, ... < K (do-while compare v_cmp_le_i64 K,i at :18650), acc = fma(bf(b[i]), bf(a[i]), acc)
//   with v_fmac_f32 starting from 0 (:18657, :18437).
//   LDS tree: s[t] = acc; barrier; for (st = blockDim/2; st > 0; st >>= 1) { if (t < st) s[t] += s[t+st]; barrier }
//   (ds_store :18670, loop :18690-18710; group_segment 1024 = 256 floats in kernargs.json).
//   Thread 0 writes c[y*gridDim.x + x] (hidden block_count_x loaded from kernarg+32 at :18716, mad :18726).
//   k_linear: bf16 RNE (:18721-18724) and a 2-byte store; k_linear_f32: 4-byte store (:18514).
__device__ static inline float dot_block(const unsigned short* a, const unsigned short* b, long K, float* sm) {
  long x = blockIdx.x, y = blockIdx.y;
  const unsigned short* ar = a + x * K;
  const unsigned short* br = b + y * K;
  float acc = 0.f;
  for (long i = threadIdx.x; i < K; i += blockDim.x) acc = __builtin_fmaf(bf2f(br[i]), bf2f(ar[i]), acc);
  unsigned t = threadIdx.x;
  sm[t] = acc;
  __syncthreads();
  for (unsigned st = blockDim.x >> 1; st > 0; st >>= 1) {
    if (t < st) sm[t] = sm[t + st] + sm[t];
    __syncthreads();
  }
  return sm[0];
}

// halogen::(anonymous namespace)::k_linear(unsigned short const*, unsigned short const*, unsigned short*, long)
extern "C" __global__ void k_linear(const unsigned short* a, const unsigned short* b, unsigned short* c, long K) {
  __shared__ float sm[256];
  float r = dot_block(a, b, K, sm);
  if (threadIdx.x == 0) c[(long)blockIdx.y * gridDim.x + blockIdx.x] = f2bf_rne(r);
}
