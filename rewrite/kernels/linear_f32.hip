// Rewrites of the "linear_f32" family from codeobj_53000 (gfx1151): bf16/q4 weights x bf16 vector(s), f32 output.
//   k_linear_f32, k_linear_f32_m<8>, k_linear_f32_q4, k_linear_f32_q4_m<8,{1,2,4,8}>, k_linear_f32_q4_mt<8,{2..8},4,256>.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o linear_f32.co linear_f32.hip
// Listing: kernels-disasm/codeobj_53000.s. Starts: k_linear_f32 18370, k_linear_f32_q4 18930, k_linear_f32_q4_mt<8,2,4,256> 923764
// (T=3..8: 925492 927495 929894 932604 935643 938976), k_linear_f32_q4_m<8,2> 942526, <8,4> 943309, <8,8> 944451, <8,1> 946330,
// k_linear_f32_m<8> 946674.
// Undecoded .long words (VOP3 forms llvm-objdump does not know) are read as the i64 address/compare/shift ops around them (inferred);
// the MATCH rows settle every such reading.
#include <hip/hip_runtime.h>

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// ---------------------------------------------------------------------------------------------------------------------------
// k_linear_f32(a, b, c, K), kernarg a@0, b@8, c@16, K@24; blockDim.x = 256 (group_segment 1024).
//   a row = blockIdx.x (a += x*K elements), b row = blockIdx.y (b += y*K): :18390-18413.
//   Thread t: for i = t; i < K; i += blockDim.x (do-while, v_cmp_le_i64 K,i :18430): acc = fma(bf(b[i]), bf(a[i]), acc) (v_fmac_f32 :18437), acc0 = 0.
//   LDS tree: sm[t] = acc; barrier; for (st = blockDim/2; st > 0; st >>= 1) { if (t < st) sm[t] = sm[t+st] + sm[t]; barrier }
//   (ds_store :18450, loop :18470-18490). Thread 0 stores f32 c[y*gridDim.x + x] (hidden block_count_x from kernarg+32, :18496-18514).
extern "C" __global__ void k_linear_f32(const unsigned short* a, const unsigned short* b, float* c, long K) {
  __shared__ float sm[256];
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
  if (t == 0) c[y * gridDim.x + x] = sm[0];
}

// ---------------------------------------------------------------------------------------------------------------------------
// k_linear_f32_m<8>(a, b, c, K, ldc), kernarg a@0, b@8, c@16, K@24, ldc@32; blockDim.x = 256 (group_segment 8192 = 8 slots x 256 floats).
//   Same body as k_linear_m<8> (see linear_m.hip) with f32 output: 8 rows x*8.. of a, b row y, 8 accumulators, per-slot tree
//   (ds_store_2addr_stride64 :946810-946813, tree :946844-946863), threads t < 8 (v_cmpx_gt_u32 8, :946869) write
//   c[y*ldc + x*8 + t] as a 4-byte store.
extern "C" __global__ void k_linear_f32_m_8(const unsigned short* a, const unsigned short* b, float* c, long K, long ldc) {
  __shared__ float sm[8 * 256];
  long x = blockIdx.x, y = blockIdx.y;
  const unsigned short* ar = a + x * 8 * K;
  const unsigned short* br = b + y * K;
  unsigned t = threadIdx.x;
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; j++) acc[j] = 0.f;
  for (long i = t; i < K; i += blockDim.x) {
    float bv = bf2f(br[i]);
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = __builtin_fmaf(bv, bf2f(ar[j * K + i]), acc[j]);
  }
#pragma unroll
  for (int j = 0; j < 8; j++) sm[j * 256 + t] = acc[j];
  __syncthreads();
  for (unsigned st = blockDim.x >> 1; st > 0; st >>= 1) {
    if (t < st) {
#pragma unroll
      for (int j = 0; j < 8; j++) sm[j * 256 + t] = sm[j * 256 + t + st] + sm[j * 256 + t];
    }
    __syncthreads();
  }
  if (t < 8) c[y * ldc + x * 8 + t] = sm[t * 256];
}

// ---------------------------------------------------------------------------------------------------------------------------
// q4 family. Weight buffer q (kernarg a0) layout, read from the kernel code:
//   q[0..63]   16 f32 lookup table, copied to LDS at +8192 (or +1024 for the single-row kernel) by threads 0..15
//              (global_load_b32 v2,v8,s[8:9] / ds_store_b32 offset:8192, :946339-946341; q4 single: :18942-18944).
//   q + off_d  4-bit codes, row r at r*(K>>1) bytes (s_lshr_b64 K,1 :946358, row multiply :946363-946377); element i is in byte i>>1,
//              low nibble for even i, high nibble for odd i (v_and 15 / v_lshrrev 4 selected by i&1, :946486-946512).
//   q + off_s  f16 scales, row r at r*scale_stride bytes, one scale per 32 elements: byte offset ((i>>4) & ~1) (v_lshrrev_b64 4 + v_and -2,
//              :946402, 946406).
//   weight w = bf16_rne(f16(scale) * lut[code]) widened back to f32: v_fma_mix_f32 (scale f16, lut f32, -0.0) then
//   v_bfe_u32/v_add3_u32 0x7fff/v_and 0xffff0000 (:946537-946548).
//   acc = fma(bf(x[i]), w, acc) (v_fmac_f32 v15,v19,v2 :946553).
__device__ static inline float q4_weight(const unsigned char* drow, const unsigned char* srow, const float* lut, long i) {
  unsigned char byte = drow[i >> 1];
  unsigned code = (i & 1) ? (byte >> 4) : (byte & 15);
  unsigned short sh = *(const unsigned short*)(srow + ((i >> 4) & ~1L));
  float s = (float)__builtin_bit_cast(_Float16, sh);
  float w = __builtin_fmaf(s, lut[code], -0.0f);
  unsigned u = __float_as_uint(w);
  u = (u + ((u >> 16) & 1u) + 0x7fffu) & 0xffff0000u;
  return __uint_as_float(u);
}

// R weight rows x T tokens. Layout of one block: row group rg = blockIdx.x (rows rg*R .. rg*R+R-1).
// Thread t strides i = t, t+blockDim, ...; every accumulator is a single chain in increasing i (the vendor unrolls the loop by U = 1/2/4/8
// with the same 8*T accumulators, e.g. k_linear_f32_q4_m<8,2> main loop + remainder loop :942553-943230, 943102; the order per chain is
// unchanged, inferred and settled by the MATCH rows).
template <int R, int T>
__device__ static inline void q4_body(const unsigned char* q, const unsigned short* xb, long xtok_stride, float* c, long K,
                                      long off_d, long off_s, long sstride, long ldc, int ctok_from_y, float* slots, float* lut) {
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
  for (long i = t; i < K; i += blockDim.x) {
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
    for (int r = 0; r < R; r++) slots[r * 256 + t] = acc[r][k];
    __syncthreads();
    for (unsigned st = blockDim.x >> 1; st > 0; st >>= 1) {
      if (t < st) {
#pragma unroll
        for (int r = 0; r < R; r++) slots[r * 256 + t] = slots[r * 256 + t + st] + slots[r * 256 + t];
      }
      __syncthreads();
    }
    if (t < R) c[(ctok_from_y ? (long)blockIdx.y : (long)k) * ldc + rg * R + t] = slots[t * 256];
  }
}

// k_linear_f32_q4(q, x, c, K, off_d, off_s, sstride), kernarg q@0, x@8, c@16, K@24, off_d@32, off_s@40, sstride@48; blockDim 256,
//   group_segment 1088 = 256 floats + the 16-float table at +1024. One row (blockIdx.x), token row y (x += y*K, :18964-18972),
//   thread 0 stores c[y*gridDim.x + x] (:19092-19110). Row offsets: data x*(K>>1) (:18973-18984), scales x*sstride (:18985-18992).
extern "C" __global__ void k_linear_f32_q4(const unsigned char* q, const unsigned short* xb, float* c, long K, long off_d, long off_s,
                                           long sstride) {
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
  if (t == 0) c[y * gridDim.x + x] = slots[0];
}

// k_linear_f32_q4_m<8,U>(q, x, c, K, ldc, off_d, off_s, sstride): kernarg q@0, x@8, c@16, K@24, ldc@32, off_d@40, off_s@48, sstride@56.
//   Rows x*8.. (s[6:7] = x<<3 :946350), token row y (x += y*K :946379-946388), blockDim 256 (hidden group_size at kernarg+0x4c, tree starts at
//   blockDim/2 :946619), threads t < 8 store c[y*ldc + x*8 + t] (:946653-946670). U (1,2,4,8) only changes the unroll.
#define Q4_M(U)                                                                                                              \
  extern "C" __global__ void k_linear_f32_q4_m_8_##U(const unsigned char* q, const unsigned short* xb, float* c, long K, long ldc, \
                                                     long off_d, long off_s, long sstride) {                                 \
    __shared__ float slots[8 * 256];                                                                                         \
    __shared__ float lut[16];                                                                                                \
    q4_body<8, 1>(q, xb + (long)blockIdx.y * K, 0, c + (long)blockIdx.y * ldc, K, off_d, off_s, sstride, ldc, 0, slots, lut); \
  }
Q4_M(1) Q4_M(2) Q4_M(4) Q4_M(8)

// k_linear_f32_q4_mt<8,T,4,256>(same kernarg layout as _m): T tokens at x + k*K, rows x*8.., T passes of the 8-slot tree, token k stored at
//   c[k*ldc + x*8 + t] (second token at + ldc*4 bytes, s[12:13]<<2 :925482). No hidden args; blockDim is the template's 256.
#define Q4_MT(T)                                                                                                             \
  extern "C" __global__ void k_linear_f32_q4_mt_8_##T(const unsigned char* q, const unsigned short* xb, float* c, long K, long ldc, \
                                                      long off_d, long off_s, long sstride) {                                \
    __shared__ float slots[8 * 256];                                                                                         \
    __shared__ float lut[16];                                                                                                \
    q4_body<8, T>(q, xb, K, c, K, off_d, off_s, sstride, ldc, 0, slots, lut);                                                \
  }
Q4_MT(2) Q4_MT(3) Q4_MT(4) Q4_MT(5) Q4_MT(6) Q4_MT(7) Q4_MT(8)
