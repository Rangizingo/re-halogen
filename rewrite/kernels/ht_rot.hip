// Rewrites of the "ht_rot" family (5 kernels) from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o ht_rot.co ht_rot.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). All five are the same 128-point Walsh-Hadamard
// rotation of a row block, one wave32 per 128 values (4 per lane), 8 waves per 256-thread block.
//
// Block index: g = blockIdx.x*8 + (tid>>5), valid when g < nb*a (64-bit compare), nb = b/128 (signed, rounds toward 0),
//   row = g / nb, blk = g % nb. Element base = row*b + blk*128 + lane*4 (b64 load = 4 values).
//   rot_in : codeobj_53000.s:20435-20600  (nb fixed 20: row stride 2560, 0x1400 bytes)
//   rot_rows: :20604-20985, rot_out: :20986-21356, rot_out_f32: :21480-21728, rot_rows_h: :22148-22505
//
// Butterfly over the 4 values of a lane (a0..a3) then over the 32 lanes (xor 1,2,4,8 via DPP row_xmask, xor 16 via
// permlanex16; partner added after the lane with the bit set negates itself: `v_cndmask -v,v ; v_add`), Sylvester order:
//   p0=a0+a1  m0=a0-a1  p1=a2+a3  m1=a2-a3 ;  r0=p0+p1  r1=m0+m1  r2=p0-p1  r3=m0-m1     (e.g. :20510-20517, :21238-21244)
// scale 1/sqrt(128) = 0x3db504f3 (:20464, :21317).
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)

typedef unsigned short u16;
typedef _Float16 half_t;

__device__ static inline float bf2f(u16 b) { return __uint_as_float((unsigned)b << 16); }
// bf16 RNE without NaN special case: v_and 1 ; v_add3 0x7fff (:21348-21351)
__device__ static inline u16 f2bf(float f) {
  unsigned u = __float_as_uint(f);
  return (u16)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
__device__ static inline float xs(float v, int lane, int bit) {
  float p = __shfl_xor(v, bit);
  return ((lane & bit) ? -v : v) + p;
}
// lane butterflies: rows r[0..3] of every lane combined over 32 lanes
__device__ static inline void lane_wht(float* r, int lane) {
#pragma unroll
  for (int k = 0; k < 4; k++) {
    float v = r[k];
    v = xs(v, lane, 1);
    v = xs(v, lane, 2);
    v = xs(v, lane, 4);
    v = xs(v, lane, 8);
    v = xs(v, lane, 16);
    r[k] = v;
  }
}
__device__ static inline void reg_wht(const float* a, float* r) {
  float p0 = a[0] + a[1], m0 = a[0] - a[1], p1 = a[2] + a[3], m1 = a[2] - a[3];
  r[0] = p0 + p1;
  r[1] = m0 + m1;
  r[2] = p0 - p1;
  r[3] = m0 - m1;
}
// first stage with the f16 column scale folded in by fma_mix: p = h0*x0 + (h1*x1), m = h0*x0 - (h1*x1) (:20502-20508)
__device__ static inline void reg_wht_scaled(const float* x, const half_t* h, float* r) {
  float t1 = __builtin_fmaf((float)h[1], x[1], -0.0f);
  float t3 = __builtin_fmaf((float)h[3], x[3], -0.0f);
  float p0 = __builtin_fmaf((float)h[0], x[0], t1);
  float m0 = __builtin_fmaf((float)h[0], x[0], -t1);
  float p1 = __builtin_fmaf((float)h[2], x[2], t3);
  float m1 = __builtin_fmaf((float)h[2], x[2], -t3);
  r[0] = p0 + p1;
  r[1] = m0 + m1;
  r[2] = p0 - p1;
  r[3] = m0 - m1;
}

#define S3 __uint_as_float(0x3db504f3u)

// Group decode shared by rows/out kernels: returns false when g >= nb*a.
__device__ static inline bool decode(int a, int b, int& row, int& blk, int& lane) {
  int nb = b / 128;
  long g = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
  lane = threadIdx.x & 31;
  if (g >= (long)nb * a) return false;
  row = (int)(g / nb);
  blk = (int)(g - (long)row * nb);
  return true;
}

// k_ht_rot_in(u16 const* x, half const* sv, half* out, int n): bf16 -> sv-scaled WHT -> f16. nb = 20, row stride 2560.
// kernarg x@0 sv@8 out@16 n@24. Output: half = fma_mix(r, s3, +0) (:20595-20598), lo=r0 hi=r1, then r2 r3.
extern "C" __global__ void k_ht_rot_in(const u16* x, const half_t* sv, half_t* out, int n) {
  long g = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (g >= (long)n * 20) return;
  int row = (int)(g / 20), blk = (int)(g - (long)row * 20);
  long xo = (long)row * 2560 + blk * 128 + lane * 4;
  int so = blk * 128 + lane * 4;
  float xv[4];
  half_t hv[4];
  for (int i = 0; i < 4; i++) { xv[i] = bf2f(x[xo + i]); hv[i] = sv[so + i]; }
  float r[4];
  reg_wht_scaled(xv, hv, r);
  lane_wht(r, lane);
  for (int i = 0; i < 4; i++) out[xo + i] = (half_t)__builtin_fmaf(r[i], S3, 0.0f);
}

// k_ht_rot_rows_h(u16 const* x, half const* sv, half* out, int a, int b): as rot_in with row stride b (:22148..).
extern "C" __global__ void k_ht_rot_rows_h(const u16* x, const half_t* sv, half_t* out, int a, int b) {
  int row, blk, lane;
  if (!decode(a, b, row, blk, lane)) return;
  long xo = (long)row * b + blk * 128 + lane * 4;
  int so = blk * 128 + lane * 4;
  float xv[4];
  half_t hv[4];
  for (int i = 0; i < 4; i++) { xv[i] = bf2f(x[xo + i]); hv[i] = sv[so + i]; }
  float r[4];
  reg_wht_scaled(xv, hv, r);
  lane_wht(r, lane);
  for (int i = 0; i < 4; i++) out[xo + i] = (half_t)__builtin_fmaf(r[i], S3, 0.0f);
}

// k_ht_rot_rows(u16 const* x, half const* sv, u16* out, int a, int b): sv-scaled WHT, v_mul by s3, bf16 RNE (:20604..)
extern "C" __global__ void k_ht_rot_rows(const u16* x, const half_t* sv, u16* out, int a, int b) {
  int row, blk, lane;
  if (!decode(a, b, row, blk, lane)) return;
  long xo = (long)row * b + blk * 128 + lane * 4;
  int so = blk * 128 + lane * 4;
  float xv[4];
  half_t hv[4];
  for (int i = 0; i < 4; i++) { xv[i] = bf2f(x[xo + i]); hv[i] = sv[so + i]; }
  float r[4];
  reg_wht_scaled(xv, hv, r);
  lane_wht(r, lane);
  for (int i = 0; i < 4; i++) out[xo + i] = f2bf(r[i] * S3);
}

// k_ht_rot_out(u16* x, half const* sv, int a, int b): in place. WHT of bf16 x, v_mul s3, then sv*value (fma_mix, -0), bf16 RNE.
// kernarg x@0 sv@8 a@16 b@20 (:20986..)
extern "C" __global__ void k_ht_rot_out(u16* x, const half_t* sv, int a, int b) {
  int row, blk, lane;
  if (!decode(a, b, row, blk, lane)) return;
  long xo = (long)row * b + blk * 128 + lane * 4;
  int so = blk * 128 + lane * 4;
  float xv[4], r[4];
  for (int i = 0; i < 4; i++) xv[i] = bf2f(x[xo + i]);
  reg_wht(xv, r);
  lane_wht(r, lane);
  for (int i = 0; i < 4; i++) {
    float y = r[i] * S3;
    x[xo + i] = f2bf(__builtin_fmaf((float)sv[so + i], y, -0.0f));
  }
}

// k_ht_rot_out_f32(float const* x, half const* sv, u16* out, int a, int b): same with f32 input (:21480..)
extern "C" __global__ void k_ht_rot_out_f32(const float* x, const half_t* sv, u16* out, int a, int b) {
  int row, blk, lane;
  if (!decode(a, b, row, blk, lane)) return;
  long xo = (long)row * b + blk * 128 + lane * 4;
  int so = blk * 128 + lane * 4;
  float xv[4], r[4];
  for (int i = 0; i < 4; i++) xv[i] = x[xo + i];
  reg_wht(xv, r);
  lane_wht(r, lane);
  for (int i = 0; i < 4; i++) {
    float y = r[i] * S3;
    out[xo + i] = f2bf(__builtin_fmaf((float)sv[so + i], y, -0.0f));
  }
}
