// Rewrite of the k_lq8s32<M, R, W, F32> family (flash_serve, gfx1151): q8s32 GEMV, bf16 activations.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o lq8s32.co lq8s32.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). Cites are codeobj_53000.s:<line> of these reference instances:
//   A = k_lq8s32<1,16,1,false> (symbol 0x580900, :594313)     B = <1,32,2,false> (0x580100, :593988)
//   C = <1,32,8,false> (0x57f900, :593663)                    D = <1,32,1,false> (0x581000, :594617)
//   E = <2,16,1,false> (0x582a00, :595672)                    F = <1,16,1,true> (0x723a00, :871651)
// The listing prints a few VOP3 instructions as .long pairs (compares with 64-bit operands); those are read from a fresh
// ROCm-23 llvm-objdump of the same symbol and cited as ".long pair".
//
// Signature (all instances): (unsigned char const* W, unsigned short const* x, void* y, long K, long N)
//   kernarg W@0 x@8 y@16 K@24 N@32 (s_load_b256 s[4:11] and s_load_b64 N: A:594316-594317).
// Template: M = tokens (1..8), R = lanes per row (16 or 32), W = waves per row (1, 2 or 8), F32 = f32 output.
//   Block is 256 threads, rows per block = 256/(R*W).  R=16,W=1: 16 rows, row = tid>>4 (A:594318 v_lshrrev 4);
//   R=32,W=2: 4 rows, row = tid>>6, wave-in-row = (tid>>5)&1 (B:593994 v_lshrrev 6, :593996 v_bfe_u32 5,1);
//   R=32,W=8: 1 row, wave = tid>>5 (C:593669); R=32,W=1: 8 rows, row = tid>>5 (D:594622).
//   row = blockIdx.x*rows_per_block + row_in_block (A:594321 s_lshl_b64 blockIdx,4; D:594625 ,3). Rows >= N are clamped to row 0
//   for the loads (v_cmp_gt_i64 N,row then v_cndmask 0: A:594329) and never stored.
//
// Weight layout per row: K int8, then K/32 f16 scales at byte K, row stride pad16(K + K/16) bytes, K/16 rounded toward zero
//   (A:594334 s_ashr_i64 .,4 with the sign fix before it, A:594341 s_and_b32 -16).
// Work split: ng = K/16 groups of 16 weights. Lane l of a row, wave w, handles groups j = begin + l, begin + l + R, ... < end:
//   W=1: begin=0, end=ng (A:594536 steps j by 16 = R; D steps by 32).
//   W>1: chunk = ceil(ng/W): B:594017-594024 ((ng+1) then /2), C:593691-593698 ((ng+7) then >>3);
//        begin = w*chunk, end = min(ng, begin+chunk) (B:594039, C:593710 v_min_i32).
// Per group j: 16 weight bytes at row + 16*j (one global_load_b128, A:594373), f16 scale at row + K + 2*(j>>1)
//   (global_load_d16_b16, A:594374; address masks the lane with 14 for R=16, A:594362; B and D mask with -2 and 30 on j):
//   w[i] = bf16_rne( (float)(int8)q[i] * (float)scale ), i = 0..15: v_bfe_i32 / v_ashrrev 24 (A:594398), v_cvt_f32_i32 (A:594407),
//   v_fma_mix_f32 scale*v + (-0.0), op_sel_hi:[1,0,0] (A:594414; exact product), v_bfe_u32 16,1 (A:594430) and
//   v_add3_u32 0x7fff (A:594446), high half packed in pairs (low element first).
//   x: two b128 loads, 32 bytes at x + 2*16*j bytes (A:594411-594412); token m adds m*K elements (E:595755 s_lshl_b64 K,1 as the stride).
//   t = 0; for p in 0..7: t = v_dot2_f32_bf16(x[2p..2p+1], w[2p..2p+1], t)  (A:594499-594529, first src operand is x, accumulator starts at 0);
//   acc += t (A:594532 v_add_f32 v16,v16,v20). Lanes with no group keep acc = 0.
// Reduction over the R lanes of a row, ds_bpermute xor partner then v_add_f32 acc + partner:
//   R=16: xor 8,4,2,1 (A:594579-594603); R=32: xor 16,8,4,2,1 (B:594236..; D:594883-594917).
// W=1: lane 0 stores bf16_rne(sum) to y[m*N + row] (A:594603 final v_add_f32, :594607 v_bfe_u32, :594612 v_add3_u32, :594613 store_d16_hi_b16;
//   second token at y + N*2 bytes: E:596015-596016).
// W>1: lane 0 of each wave writes its sum to LDS[(row_in_block*W + w)*4 bytes] (ds_store_b32 B:594279, C:593948; 32 bytes of LDS), barrier
//   (B:594287, C:593954); then threads with tid < rows_per_block (a .long-pair v_cmp_gt_u32 4,tid at B:594284; C: v_cmp_eq_u32 0,tid :593952)
//   and row < N sum s = ((0 + l[0]) + l[1]) + ... in wave order (B: ds_load_b64 :594295, adds :594297,:594299; C: two ds_load_b128 :593962-593963,
//   adds :593967-593979) and store bf16_rne(s) (B:594309, C:593984).  For M>1 the per-token LDS layout is not read from a listing; the
//   rewrite uses [(row*W + w)*M + m] and the bitwise match on M=2..8 is what supports it.
// F32 (F): the same sum stored as f32 without the bf16 rounding (F:871948 global_store_b32; the bf16 instance has global_store_d16_hi_b16).
#include <hip/hip_runtime.h>

// f16 -> f32 is exact (v_cvt_f32_f16 / the f16 operand of v_fma_mix_f32).
__device__ static inline float h2f(unsigned short u) { return (float)__builtin_bit_cast(_Float16, u); }

typedef __bf16 v2bf __attribute__((ext_vector_type(2)));

// bf16 RNE without a NaN special case: v_bfe_u32 16,1 ; v_add3_u32 0x7fff ; high half.
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
__device__ static inline float dot2(unsigned x, unsigned w, float acc) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(v2bf, x), __builtin_bit_cast(v2bf, w), acc, false);
}

template <int M, int R, int W, bool F32>
__device__ static inline void lq8s32_body(const unsigned char* Wt, const unsigned short* x, void* y, long K, long N) {
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
  const long rowbytes = (K + K / 16 + 15) & -16L;
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
    float s = h2f(*(const unsigned short*)(sc + 2 * (j >> 1)));
    unsigned qw[4] = {q4.x, q4.y, q4.z, q4.w};
    unsigned wp[8];
    for (int i = 0; i < 8; i++) {
      unsigned b0 = (qw[i / 2] >> (16 * (i & 1))) & 0xffu;
      unsigned b1 = (qw[i / 2] >> (16 * (i & 1) + 8)) & 0xffu;
      float f0 = (float)(int)(signed char)b0 * s;
      float f1 = (float)(int)(signed char)b1 * s;
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
      for (int m = 0; m < M; m++) {
        if (F32) ((float*)y)[(long)m * N + row] = r[m];
        else ((unsigned short*)y)[(long)m * N + row] = f2bf_rne(r[m]);
      }
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
          if (F32) ((float*)y)[(long)m * N + orow] = s;
          else ((unsigned short*)y)[(long)m * N + orow] = f2bf_rne(s);
        }
    }
  }
}

#define LQ8S32(M, R, W, F)                                                                                         \
  extern "C" __global__ void __launch_bounds__(256) lq8s32_m##M##_r##R##_w##W##_f##F(                              \
      const unsigned char* Wt, const unsigned short* x, void* y, long K, long N) {                                 \
    lq8s32_body<M, R, W, (bool)F>(Wt, x, y, K, N);                                                                 \
  }
#define LQ8S32_ALL(M, F) LQ8S32(M, 16, 1, F) LQ8S32(M, 32, 1, F) LQ8S32(M, 32, 2, F) LQ8S32(M, 32, 8, F)
#define LQ8S32_MS(F) LQ8S32_ALL(1, F) LQ8S32_ALL(2, F) LQ8S32_ALL(3, F) LQ8S32_ALL(4, F) \
                     LQ8S32_ALL(5, F) LQ8S32_ALL(6, F) LQ8S32_ALL(7, F) LQ8S32_ALL(8, F)
LQ8S32_MS(0)
LQ8S32_MS(1)
