// Rewrites of the quantized linear families from codeobj_53000 (gfx1151), bf16 output (g15):
//   linear_q4: k_linear_q4, k_linear_q4_m<R,U,B> (8), k_linear_q4_mt<R,T,U,B> (63)
//   linear_q8: k_linear_q8     linear_q8s32: k_linear_q8s32
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o linear_q4.co linear_q4.hip
// Listing: kernels-disasm/codeobj_53000.s. Starts: k_linear_q4 18742, k_linear_q8 19418, k_linear_q8s32 20207,
//   k_linear_q4_mt<16,2,1,128> 625103 (first of 63; <32,2,1,64> 698415, <8,2,1,256> 788783), k_linear_q4_m<32,1,64> 845766,
//   <4,1,256> 846736, <16,1,256> 847121, <32,1,256> 848091, <8,2,256> 849062, <8,4,256> 850000, <8,8,256> 851345, <8,1,256> 853350.
// Undecoded .long words are VOP3 forms llvm-objdump does not know (i64 shift/add/compare, v_cmp feeding v_cndmask); they are read as the
// i64 address math around them (inferred). The MATCH rows settle every such reading.
//
// Same q4 layout as g04 (rewrite/reports/g04.md, kernels/linear_f32.hip) with a bf16 instead of an f32 result.
// Template args of k_linear_q4_mt<R,T,U,B>: R weight rows per block (rows x*R..), T tokens, U unroll, B block size. LDS = R*B floats
// (slot r at r*B) plus the 16-float table at byte 8192 (group_segment 8256 when R*B = 2048). k_linear_q4_m<R,U,B> is T = 1 with the token
// in blockIdx.y. Arg order (R,T,U,B) is read from the group_segment sizes in rewrite/harness/kernargs.json: R*B*4 + 64.
#include <hip/hip_runtime.h>

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// bf16 round to nearest even with no NaN special case: v_bfe_u32 v,16,1 ; v_add3_u32 v,v,bit,0x7fff (:18851-18853). Result is the upper
// 16 bits after >> 16.
__device__ static inline unsigned bf16_rne_bits(float v) {
  unsigned u = __float_as_uint(v);
  return (u + ((u >> 16) & 1u) + 0x7fffu);
}
// Same rounding kept as an f32 with the low half cleared (v_and_b32 0xffff0000).
__device__ static inline float bf16_rne_f32(float v) { return __uint_as_float(bf16_rne_bits(v) & 0xffff0000u); }

// ---------------------------------------------------------------------------------------------------------------------------
// q4 weight buffer q (kernarg 0): 16 f32 table at q[0..63]; 4-bit codes at q+off_d, row r at r*(K>>1) bytes, low nibble for even i;
//   f16 scales at q+off_s, row r at r*sstride bytes, one per 32 elements at byte (i>>4)&~1.
//   Table copy by threads t < 16: v_cmpx_gt_u32 16 (:18751), global_load_b32 + ds_store_b32 offset:1024 (single :18756) or offset:8192
//   (m<32,1,64> :845777, mt<16,2,1,128> :625114). Row bases: s_lshr_b64 K,1 (single :18786, m :845813), row multiplies follow.
//   weight = bf16_rne(f16(scale) * lut[code]): v_fma_mix_f32 (scale f16, lut f32, neg(0)) (single :18851), RNE (:18852-18854), then
//   acc = v_fmac_f32(x, w) (single :18857, loop test v_cmp_le_i64 K,i :18845).
__device__ static inline float q4_weight(const unsigned char* drow, const unsigned char* srow, const float* lut, long i) {
  unsigned char byte = drow[i >> 1];
  unsigned code = (i & 1) ? (byte >> 4) : (byte & 15);
  unsigned short sh = *(const unsigned short*)(srow + ((i >> 4) & ~1L));
  float s = (float)__builtin_bit_cast(_Float16, sh);
  float w = __builtin_fmaf(s, lut[code], -0.0f);
  return bf16_rne_f32(w);
}

// k_linear_q4(q, x, c, K, off_d, off_s, sstride): kernarg q@0 x@8 c@16 K@24 off_d@32 off_s@40 sstride@48 (s_load_b256 s[8:15] and s[16:23],
//   :18744-18745). blockDim 256, LDS 1024 + 64. Weight row blockIdx.x, token blockIdx.y (x += y*K elements, s_mul y*K :18772ff).
//   Strided loop i = t, t+256.., LDS tree 128..1 (s_lshr 1 per round), thread 0 stores bf16 RNE at c[y*gridDim.x + x]: hidden
//   block_count_x (kernarg+56 loaded :18766) times y (v_mad_u64_u32 :18914) then global_store_d16_hi_b16 (:18926).
extern "C" __global__ __launch_bounds__(256) void k_linear_q4(const unsigned char* q, const unsigned short* xb, unsigned short* c, long K,
                                                              long off_d, long off_s, long sstride) {
  __shared__ float slots[256];
  __shared__ float lut[16];
  long x = blockIdx.x, y = blockIdx.y;
  unsigned t = threadIdx.x;
  if (t < 16) lut[t] = ((const float*)q)[t];
  __syncthreads();
  const unsigned char* drow = q + off_d + x * (K >> 1);
  const unsigned char* srow = q + off_s + x * sstride;
  const unsigned short* xr = xb + y * K;
  float acc = 0.f;
  for (long i = t; i < K; i += blockDim.x) acc = __builtin_fmaf(bf2f(xr[i]), q4_weight(drow, srow, lut, i), acc);
  slots[t] = acc;
  __syncthreads();
  for (unsigned st = blockDim.x >> 1; st > 0; st >>= 1) {
    if (t < st) slots[t] = slots[t + st] + slots[t];
    __syncthreads();
  }
  if (t == 0) c[y * gridDim.x + x] = (unsigned short)(bf16_rne_bits(slots[0]) >> 16);
}

// R rows x T tokens. Thread t strides i = t, t+B, ... with R*T single accumulator chains in increasing i; the vendor unrolls U times with
// the same chains (inferred from R*T accumulators in every U, settled by the MATCH rows). Per token one pass of an R-slot LDS tree starting
// at B/2 (m<32,1,64>: s_mov_b32 s0, 32 :846599). Threads t < R store bf16 RNE at ctok + rg*R + t (v_cmpx_gt_u32 R, v0: m<32,1,64> :846713;
// RNE then global_store_d16_hi_b16 :846732).
template <int R, int T, int U, int B>
__device__ static inline void q4_body(const unsigned char* q, const unsigned short* xb, long xtok_stride, unsigned short* c, long ctok_stride,
                                      long K, long off_d, long off_s, long sstride, float* slots, float* lut) {
  long rg = blockIdx.x;
  unsigned t = threadIdx.x;
  if (t < 16) lut[t] = ((const float*)q)[t];
  __syncthreads();
  const unsigned char* dbase = q + off_d + rg * R * (K >> 1);
  const unsigned char* sbase = q + off_s + rg * R * sstride;
  float acc[R][T];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int k = 0; k < T; k++) acc[r][k] = 0.f;
#pragma unroll U
  for (long i = t; i < K; i += B) {
    float xv[T];
#pragma unroll
    for (int k = 0; k < T; k++) xv[k] = bf2f(xb[k * xtok_stride + i]);
#pragma unroll
    for (int r = 0; r < R; r++) {
      float w = q4_weight(dbase + r * (K >> 1), sbase + r * sstride, lut, i);
#pragma unroll
      for (int k = 0; k < T; k++) acc[r][k] = __builtin_fmaf(xv[k], w, acc[r][k]);
    }
  }
#pragma unroll
  for (int k = 0; k < T; k++) {
    __syncthreads();
#pragma unroll
    for (int r = 0; r < R; r++) slots[r * B + t] = acc[r][k];
    __syncthreads();
    for (unsigned st = B >> 1; st > 0; st >>= 1) {
      if (t < st) {
#pragma unroll
        for (int r = 0; r < R; r++) slots[r * B + t] = slots[r * B + t + st] + slots[r * B + t];
      }
      __syncthreads();
    }
    if (t < R) c[k * ctok_stride + rg * R + t] = (unsigned short)(bf16_rne_bits(slots[t * B]) >> 16);
  }
}

// k_linear_q4_mt<R,T,U,B>(q, x, c, K, ldc, off_d, off_s, sstride): kernarg q@0 x@8 c@16 K@24 ldc@32 off_d@40 off_s@48 sstride@56
//   (s_load_b512 s[4:19], :625105). Rows x*R.. (R=16: s_lshl_b64 x,4 :625137), T tokens at x + k*K, token k stored at c[k*ldc + x*R + t].
//   No hidden args, no row guard (grid.x must be exactly ceil(rows/R)).
#define Q4_MT(R, T, U, B)                                                                                                         \
  extern "C" __global__ __launch_bounds__(B) void k_linear_q4_mt_##R##_##T##_##U##_##B(                                           \
      const unsigned char* q, const unsigned short* xb, unsigned short* c, long K, long ldc, long off_d, long off_s, long sstride) { \
    __shared__ float slots[R * B];                                                                                                \
    __shared__ float lut[16];                                                                                                     \
    q4_body<R, T, U, B>(q, xb, K, c, ldc, K, off_d, off_s, sstride, slots, lut);                                                  \
  }
#define Q4_MT_T(R, U, B) \
  Q4_MT(R, 2, U, B) Q4_MT(R, 3, U, B) Q4_MT(R, 4, U, B) Q4_MT(R, 5, U, B) Q4_MT(R, 6, U, B) Q4_MT(R, 7, U, B) Q4_MT(R, 8, U, B)
#define Q4_MT_U(R, B) Q4_MT_T(R, 1, B) Q4_MT_T(R, 2, B) Q4_MT_T(R, 4, B)
Q4_MT_U(16, 128)
Q4_MT_U(32, 64)
Q4_MT_U(8, 256)

// k_linear_q4_m<R,U,B>(q, x, c, K, ldc, off_d, off_s, sstride): same kernargs; token = blockIdx.y (x += y*K: s_mul_i32 K.hi*y :845804),
//   stored at c[y*ldc + x*R + t] (ldc*y: s_mul_i32 s16,s4 :846718).
#define Q4_M(R, U, B)                                                                                                             \
  extern "C" __global__ __launch_bounds__(B) void k_linear_q4_m_##R##_##U##_##B(                                                  \
      const unsigned char* q, const unsigned short* xb, unsigned short* c, long K, long ldc, long off_d, long off_s, long sstride) { \
    __shared__ float slots[R * B];                                                                                                \
    __shared__ float lut[16];                                                                                                     \
    q4_body<R, 1, U, B>(q, xb + (long)blockIdx.y * K, 0, c + (long)blockIdx.y * ldc, 0, K, off_d, off_s, sstride, slots, lut);    \
  }
Q4_M(32, 1, 64)
Q4_M(4, 1, 256)
Q4_M(16, 1, 256)
Q4_M(32, 1, 256)
Q4_M(8, 2, 256)
Q4_M(8, 4, 256)
Q4_M(8, 8, 256)
Q4_M(8, 1, 256)

// ---------------------------------------------------------------------------------------------------------------------------
// k_linear_q8(q, x, c, K, ldc): kernarg q@0 x@8 c@16 K@24 ldc@32 (s_load_b256 s[8:15] :19421, s_load_b64 s[6:7] kernarg+32 :19422).
//   blockDim 256 (hidden group_size_x at kernarg+0x34 read :19434; LDS 1024; tree 128..1 unrolled, first test v_cmpx_gt_u32 0x80 :19529).
//   Row x at q + x*(K + K/16) (signed K/16 via s_ashr_i64 4 :19440, row multiply follows): K bytes of u8 codes, then 4-byte entries, one
//   per 64 elements, at byte ((i>>4) & ~3) (v_lshrrev_b64 4 then v_and -4 :19485). An entry is two f16: low = scale, high = offset.
//   w = bf16_rne(fma(float(u8), scale, offset)): v_cvt_f32_ubyte0 (:19505), v_fma_mix_f32 v1, v1, v7, v7 op_sel:[0,0,1]
//   op_sel_hi:[0,1,1] (src1 = f16 low half, src2 = f16 high half, :19508), RNE (:19509-19511), acc = fmac(x, w) (:19516).
//   Thread 0 stores bf16 RNE at c[y*ldc + x] (kernarg+32 times y; global_store_d16_hi_b16 :19642).
extern "C" __global__ __launch_bounds__(256) void k_linear_q8(const unsigned char* q, const unsigned short* xb, unsigned short* c, long K, long ldc) {
  __shared__ float slots[256];
  long x = blockIdx.x, y = blockIdx.y;
  unsigned t = threadIdx.x;
  const unsigned char* drow = q + x * (K + K / 16);
  const unsigned char* srow = drow + K;
  const unsigned short* xr = xb + y * K;
  float acc = 0.f;
  for (long i = t; i < K; i += blockDim.x) {
    float code = (float)drow[i];
    unsigned pair = *(const unsigned*)(srow + ((i >> 4) & ~3L));
    float sc = (float)__builtin_bit_cast(_Float16, (unsigned short)(pair & 0xffffu));
    float off = (float)__builtin_bit_cast(_Float16, (unsigned short)(pair >> 16));
    float w = bf16_rne_f32(__builtin_fmaf(code, sc, off));
    acc = __builtin_fmaf(bf2f(xr[i]), w, acc);
  }
  slots[t] = acc;
  __syncthreads();
  for (unsigned st = blockDim.x >> 1; st > 0; st >>= 1) {
    if (t < st) slots[t] = slots[t + st] + slots[t];
    __syncthreads();
  }
  if (t == 0) c[y * ldc + x] = (unsigned short)(bf16_rne_bits(slots[0]) >> 16);
}

// k_linear_q8s32(q, x, c, K, ldc): same kernargs and skeleton as k_linear_q8 (:20207ff vs :19418ff).
//   Row stride align16(K + K/16) (s_add 15 then s_and -16 :20235); K int8 codes (global_load_i8 :20283), then one f16 scale per 32
//   elements at byte ((i>>4) & ~1) (v_and -2 :20274, global_load_d16_b16 :20284); w = bf16_rne(fma(f16 scale, float(i8), -0)):
//   v_cvt_f32_i32 (:20294), v_fma_mix_f32 v1, v7, v1, neg(0) op_sel_hi:[1,0,0] (:20297), RNE, acc = fmac(x, w) (:20305).
//   Tree and store as k_linear_q8 (first test :20318, global_store_d16_hi_b16 :20431).
extern "C" __global__ __launch_bounds__(256) void k_linear_q8s32(const unsigned char* q, const unsigned short* xb, unsigned short* c, long K,
                                                                 long ldc) {
  __shared__ float slots[256];
  long x = blockIdx.x, y = blockIdx.y;
  unsigned t = threadIdx.x;
  long stride = (K + K / 16 + 15) & ~15L;
  const signed char* drow = (const signed char*)(q + x * stride);
  const unsigned char* srow = (const unsigned char*)drow + K;
  const unsigned short* xr = xb + y * K;
  float acc = 0.f;
  for (long i = t; i < K; i += blockDim.x) {
    float code = (float)(int)drow[i];
    unsigned short sh = *(const unsigned short*)(srow + ((i >> 4) & ~1L));
    float sc = (float)__builtin_bit_cast(_Float16, sh);
    float w = bf16_rne_f32(__builtin_fmaf(sc, code, -0.0f));
    acc = __builtin_fmaf(bf2f(xr[i]), w, acc);
  }
  slots[t] = acc;
  __syncthreads();
  for (unsigned st = blockDim.x >> 1; st > 0; st >>= 1) {
    if (t < st) slots[t] = slots[t + st] + slots[t];
    __syncthreads();
  }
  if (t == 0) c[y * ldc + x] = (unsigned short)(bf16_rne_bits(slots[0]) >> 16);
}
