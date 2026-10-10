// Rewrites of the "lbfw" family (bf16 weights x bf16 activations, f32 accumulate, bf16 out) from codeobj_53000, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o lbfw.co lbfw.hip
// Listing: kernels-disasm/codeobj_53000.s. Instance ranges (start line): k_lbfw<1,4> 141666, <1,2> 141821, <2,4> 141963,
// <2,2> 142180, <3,4> 142363, <3,2> 142635, ... <8,2> 146267; k_lbfw_lds<1,4> 141223, <2,4> 141416.
#include <hip/hip_runtime.h>

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));

// v_dot2_f32_bf16 acc, a, b (codeobj_53000.s:141888-141900): acc + a.lo*b.lo + a.hi*b.hi, one instruction.
__device__ static inline float dot2(unsigned a, unsigned b, float acc) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2, a), __builtin_bit_cast(bf16x2, b), acc, false);
}
// f32 -> bf16 RNE without NaN case: v_bfe_u32 v4,v3,16,1 ; v_add3_u32 v2,v3,v4,0x7fff ; global_store_d16_hi_b16 (:141953,141958-141959).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// Wave32 butterfly over lanes xor 16,8,4,2,1 with v_add_f32 (ds_bpermute_b32 pairs, :141910-141949). Lane 0 ends with the sum.
__device__ static inline float wave_sum(float v) {
  for (int o = 16; o >= 1; o >>= 1) v = v + __shfl_xor(v, o, 32);
  return v;
}

// k_lbfw<M,U>(W, X, Y, K, N), kernarg: W@0 (N x K bf16), X@8 (M x K bf16), Y@16 (M x N bf16), K@24, N@32.
//   Block = 256 threads = 8 waves; wave w of block b owns row r = b*8 + (tid>>5) (v_lshrrev 5, s2<<3: :141824-141829).
//   Guard r < N (undecoded compare .long at :141834 against kernarg+0x20, inferred; the whole wave exits).
//   Lane l loads 16 bytes (8 bf16) per vector at element l*8 (v_lshlrev 4: :141854); a loop step covers U*256 elements:
//   vectors at element offsets 0, 256, ..., (U-1)*256 (global_load_b128 offsets 0/512/1024/1536 bytes, :141874-141876, :141711-141721).
//   Loop is a do-while entered when K > 0 and advances U*256 elements (s0 += 0x200 for U=2 :141877, 0x400 for U=4 :141722) while k < K;
//   K must be a multiple of U*256 because the vendor has no tail (loads run past K).
//   Per token m the single accumulator is chained: for each vector u, for j in 0..3: acc = dot2(x_word, w_word, acc)  (:141888-141900;
//   M=2 interleaves the two tokens' chains, :142252-142277, which are independent).
//   Then wave_sum, lane 0 writes Y[m*N + r] as bf16 RNE (second token at +N elements via s[8:9]<<1, :142339-142359).
template <int M, int U>
__device__ static inline void lbfw_body(const uint4* __restrict__ W, const uint4* __restrict__ X, unsigned short* Y, long K, long N) {
  long r = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
  if (r >= N) return;
  int lane = threadIdx.x & 31;
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; m++) acc[m] = 0.f;
  long kv = K / 8;  // uint4 per row
  const uint4* w = W + r * kv + lane;
  long steps = K > 0 ? (K + U * 256 - 1) / (U * 256) : 0;
  for (long s = 0; s < steps; s++) {
    uint4 wv[U], xv[M][U];
#pragma unroll
    for (int u = 0; u < U; u++) wv[u] = w[s * U * 32 + u * 32];
#pragma unroll
    for (int m = 0; m < M; m++)
#pragma unroll
      for (int u = 0; u < U; u++) xv[m][u] = X[m * kv + lane + s * U * 32 + u * 32];
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int m = 0; m < M; m++) {
        acc[m] = dot2(xv[m][u].x, wv[u].x, acc[m]);
        acc[m] = dot2(xv[m][u].y, wv[u].y, acc[m]);
        acc[m] = dot2(xv[m][u].z, wv[u].z, acc[m]);
        acc[m] = dot2(xv[m][u].w, wv[u].w, acc[m]);
      }
  }
#pragma unroll
  for (int m = 0; m < M; m++) {
    float s = wave_sum(acc[m]);
    if (lane == 0) Y[m * N + r] = f2bf_rne(s);
  }
}

// k_lbfw_lds<M,4>: same math, but X (M*K bf16, row m at element m*K) is first copied to LDS by the whole block:
//   thread t copies 16 bytes at element t*8, stride 2048 elements, while e < M*K (:141228-141267, M=2 uses K*2 at :141424);
//   barrier (:141273); then X comes from ds_load_b128 at byte lane*16 + u*512 (+ m*K*2) (:141315-141316, 141332, 141339).
//   LDS size 12288 B * M (kernargs.json group_segment) fixes K <= 6144. Dot operand order is (X from LDS, W) with the same chain.
template <int M, int U>
__device__ static inline void lbfw_lds_body(const uint4* __restrict__ W, const uint4* __restrict__ X, unsigned short* Y, long K, long N,
                                            uint4* sx) {
  long tot = K * M;
  for (long e = (long)threadIdx.x * 8; e < tot; e += 2048) sx[e >> 3] = X[e >> 3];
  __syncthreads();
  long r = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
  if (r >= N) return;
  int lane = threadIdx.x & 31;
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; m++) acc[m] = 0.f;
  long kv = K / 8;
  const uint4* w = W + r * kv + lane;
  long steps = K > 0 ? (K + U * 256 - 1) / (U * 256) : 0;
  for (long s = 0; s < steps; s++) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      uint4 wv = w[s * U * 32 + u * 32];
#pragma unroll
      for (int m = 0; m < M; m++) {
        uint4 xv = sx[m * kv + lane + s * U * 32 + u * 32];
        acc[m] = dot2(xv.x, wv.x, acc[m]);
        acc[m] = dot2(xv.y, wv.y, acc[m]);
        acc[m] = dot2(xv.z, wv.z, acc[m]);
        acc[m] = dot2(xv.w, wv.w, acc[m]);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < M; m++) {
    float s = wave_sum(acc[m]);
    if (lane == 0) Y[m * N + r] = f2bf_rne(s);
  }
}

#define LBFW(M, U)                                                                                                                   \
  extern "C" __global__ void k_lbfw_##M##_##U(const unsigned short* W, const unsigned short* X, unsigned short* Y, long K, long N) { \
    lbfw_body<M, U>((const uint4*)W, (const uint4*)X, Y, K, N);                                                                      \
  }
#define LBFW_LDS(M, U)                                                                                                                   \
  extern "C" __global__ void k_lbfw_lds_##M##_##U(const unsigned short* W, const unsigned short* X, unsigned short* Y, long K, long N) { \
    __shared__ uint4 sx[M * 6144 / 8];                                                                                                   \
    lbfw_lds_body<M, U>((const uint4*)W, (const uint4*)X, Y, K, N, sx);                                                                  \
  }
LBFW(1, 2) LBFW(1, 4) LBFW(2, 2) LBFW(2, 4) LBFW(3, 2) LBFW(3, 4) LBFW(4, 2) LBFW(4, 4)
LBFW(5, 2) LBFW(5, 4) LBFW(6, 2) LBFW(6, 4) LBFW(7, 2) LBFW(7, 4) LBFW(8, 2) LBFW(8, 4)
LBFW_LDS(1, 4) LBFW_LDS(2, 4)
