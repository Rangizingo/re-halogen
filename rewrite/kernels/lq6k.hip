// Rewrites of k_lq6k (Q6_K GEMV, M tokens) from codeobj_53000 (flash_serve, gfx1151).
// Listing: kernels-disasm/codeobj_53000.s (not in git). Slices: work/kernels/*k_lq6k*.s (line numbers below are slice-relative
// for k_lq6kILi1ELi16ELi1ELb0E, slice start = codeobj address 0x597100).
#include <hip/hip_runtime.h>
// hipcc fuses a*b+c by default; the listing uses separate v_mul/v_add (explicit fmaf where fused).
#pragma clang fp contract(off)

typedef __bf16 v2bf __attribute__((ext_vector_type(2)));

// f32 -> bf16 RNE, no NaN special case: v_bfe_u32 ; v_add3_u32 0x7fff (slice:225-252). Returns the bf16 in the low half.
__device__ static inline unsigned f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (u + ((u >> 16) & 1u) + 0x7fffu) >> 16;
}
__device__ static inline float half_to_f(unsigned short h) { return (float)__builtin_bit_cast(_Float16, h); }

// One 16-element Q6_K scale group (lane l of 16 within a 256-element block), dotted with x[0..15] (bf16, 8 dwords).
// Q6_K block = ql[128] qh[64] scales[16] (int8) d (f16), 210 bytes (stride 0xd2, slice:36-44, :197).
// Lane l: half = l>>3, quad = (l>>1)&3.
//   ql dwords  at  half*64 + (l&3)*16   (16 bytes, b128, offset 0)
//   qh dwords  at  128 + half*32 + (l&1)*16   (b128 offset 128)
//   scale      at  192 + l  (global_load_i8 offset 192), d at 208 (global_load_d16_b16)
// Element 4k+i: ql nibble = (qlw[k] >> (8i + (l&4))) & 15 (v_bfe_u32 ..,4 with shifts v9|{0,8,16,24}),
//   qh bits = (qhw[k] >> (8i + (l&6))) & 3, <<4 (v_lshlrev 4 ; v_and_or 48), q = (qh<<4|ql) - 32 (v_subrev 32).
// w = bf16_rne( (d*scale) * (float)q ), packed in element pairs; acc = dot2_bf16 chain over the 8 pairs starting from 0.
__device__ static inline float group_dot(const unsigned char* blk, int l, const unsigned* xw) {
  int half = l >> 3;
  const unsigned char* qlp = blk + half * 64 + (l & 3) * 16;
  const unsigned char* qhp = blk + 128 + half * 32 + (l & 1) * 16;
  float sc = (float)(int)(signed char)blk[192 + l];
  unsigned short dh;
  __builtin_memcpy(&dh, blk + 208, 2);
  float ds = half_to_f(dh) * sc;
  int qs = l & 4, hs = l & 6;
  float acc = 0.0f;
  #pragma unroll
  for (int k = 0; k < 4; k++) {
    unsigned qlw, qhw;
    __builtin_memcpy(&qlw, qlp + 4 * k, 4);
    __builtin_memcpy(&qhw, qhp + 4 * k, 4);
    unsigned wb[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int q = (int)((((qhw >> (8 * i + hs)) << 4) & 48u) | ((qlw >> (8 * i + qs)) & 15u)) - 32;
      wb[i] = f2bf_rne(ds * (float)q);
    }
    #pragma unroll
    for (int p = 0; p < 2; p++) {
      unsigned wp = (wb[2 * p] & 0xffffu) | (wb[2 * p + 1] << 16);
      unsigned xp = xw[2 * k + p];
      acc = __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(v2bf, xp), __builtin_bit_cast(v2bf, wp), acc, false);
    }
  }
  return acc;
}

// Generic k_lq6k<M, R, W, F32OUT>(w, x, y, K, N)   kernarg: w@0 x@8 y@16 K@24 N@32, block 256 threads, wave32.
//   M tokens, R lanes per row (16 or 32), W waves per row (R=32 only), F32OUT = the Lb1 variants (f32 store, else bf16 RNE).
//   rows per block = 256/(R*W): row = blockIdx.x*(256/(R*W)) + tid/(R*W)   (slice R16: s_lshl_b64 s[2:3],4 ; R32: ,3 ; W2: ,2 ; W8: s2 only)
//   G = K/16 groups (s_lshr 4); wave w of W covers groups [w*c, min(G, w*c+c)), c = ceil(G/W) (R32 W2 listing: s_add 1, ashr 1;
//   W8: s_add 7, ashr 3), lane steps R groups at a time; group g -> block g>>4, lane-in-block g&15 (l), x offset g*16 elements.
//   For W=1 the range is [0, G). Per iteration: acc += group_dot(...) per token, x of token m at x + m*K (s_lshl_b64 s[10:11],1),
//   y of token m at y + m*N (second store at s[12:13]*2 bytes, f32: *4).
//   Reduction inside a wave: v_add + ds_bpermute xor 16 (R=32 only), 8, 4, 2, 1. W>1: lane 0 of each wave -> LDS, barrier, one thread
//   per row sums 0 + p0 + p1 + ... in wave order (W8 listing: v_add_f32 v0,0,v0 then v1..v7).
template <int M, int R, int W, bool F32OUT>
__device__ static inline void lq6k_body(const unsigned char* w, const unsigned short* x, void* y, long K, long N) {
  constexpr int RPB = 256 / (R * W);
  __shared__ float part[M * RPB * W];
  int tid = threadIdx.x;
  int rib = tid / (R * W);
  long row = (long)blockIdx.x * RPB + rib;
  int wave = (tid / 32) % W;
  int lane = tid % R;
  long G = K / 16;
  long nblk = K / 256;
  long chunk = (G + W - 1) / W;
  long beg = (W == 1) ? 0 : wave * chunk;
  long end = (W == 1) ? G : (beg + chunk < G ? beg + chunk : G);
  float acc[M];
  #pragma unroll
  for (int m = 0; m < M; m++) acc[m] = 0.0f;
  if (row < N) {
    const unsigned char* wr = w + row * nblk * 210;
    for (long g = beg + lane; g < end; g += R) {
      long b = g >> 4;
      int l = (int)(g & 15);
      #pragma unroll
      for (int m = 0; m < M; m++) {
        unsigned xw[8];
        __builtin_memcpy(xw, x + (long)m * K + g * 16, 32);
        acc[m] += group_dot(wr + b * 210, l, xw);
      }
    }
  }
  #pragma unroll
  for (int m = 0; m < M; m++) {
    if (R == 32) acc[m] += __shfl_xor(acc[m], 16);
    acc[m] += __shfl_xor(acc[m], 8);
    acc[m] += __shfl_xor(acc[m], 4);
    acc[m] += __shfl_xor(acc[m], 2);
    acc[m] += __shfl_xor(acc[m], 1);
  }
  float out[M];
  if (W == 1) {
    #pragma unroll
    for (int m = 0; m < M; m++) out[m] = acc[m];
  } else {
    if ((tid & 31) == 0) {
      #pragma unroll
      for (int m = 0; m < M; m++) part[(m * RPB + rib) * W + wave] = acc[m];
    }
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < M; m++) {
      float s = 0.0f;
      #pragma unroll
      for (int k = 0; k < W; k++) s += part[(m * RPB + rib) * W + k];
      out[m] = s;
    }
  }
  if (row < N && (tid % (R * W)) == 0) {
    #pragma unroll
    for (int m = 0; m < M; m++) {
      if (F32OUT) ((float*)y)[(long)m * N + row] = out[m];
      else ((unsigned short*)y)[(long)m * N + row] = (unsigned short)f2bf_rne(out[m]);
    }
  }
}

#define LQ6K_ARGS const unsigned char* w, const unsigned short* x, void* y, long K, long N
#define LQ6K_ONE(M, R, W) \
  extern "C" __global__ void lq6k_##M##_##R##_##W##_0(LQ6K_ARGS) { lq6k_body<M, R, W, false>(w, x, y, K, N); } \
  extern "C" __global__ void lq6k_##M##_##R##_##W##_1(LQ6K_ARGS) { lq6k_body<M, R, W, true>(w, x, y, K, N); }
#define LQ6K_M(M) LQ6K_ONE(M, 16, 1) LQ6K_ONE(M, 32, 1) LQ6K_ONE(M, 32, 2) LQ6K_ONE(M, 32, 8)
LQ6K_M(1) LQ6K_M(2) LQ6K_M(3) LQ6K_M(4) LQ6K_M(5) LQ6K_M(6) LQ6K_M(7) LQ6K_M(8)
