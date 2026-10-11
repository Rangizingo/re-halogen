// Rewrites of the "rope" family from the vendor code object codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o rope.co rope.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). Line numbers below are lines of that file.
#include <hip/hip_runtime.h>
// No mul+add fusion: the vendor fuses only where it emits v_fma/v_fmac, and every fma below is written out explicitly.
// (Without this, clang turned p + n into an extra fma in the logf expansion and 1 ulp of ln(theta) leaked into outputs.)
#pragma clang fp contract(off)

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
// bf16 RNE without NaN special case: v_and 1 ; v_add3_u32 +0x7fff ; store high half (k_rope :52479-52483).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// ln(x) the way the vendor expands logf (k_rope :52387-52415, same sequence in every rope kernel and in k_qk_prep):
//   dn = x < 2^-126 ; l = v_log_f32(dn ? x*2^32 : x) (log2) ; p = l*0x3f317217 ; n = fma(l,0x3f317217,-p) ;
//   n = fma(l,0x3377d1cf,n) ; r = p + n ; r = |l| < inf ? r : l ; result = r - (dn ? 0x41b17218 : 0).
// Plain `logf` in hipcc differs from this by 1 ulp for some inputs (theta = 1e6 mismatched), so it is spelled out.
__device__ static inline float vlog(float x) {
  bool dn = x < 1.17549435e-38f;
  float l = __builtin_amdgcn_logf(dn ? x * 4294967296.0f : x);
  float p = l * __uint_as_float(0x3f317217u);
  float n = __builtin_fmaf(l, __uint_as_float(0x3f317217u), -p);
  n = __builtin_fmaf(l, __uint_as_float(0x3377d1cfu), n);
  float r = p + n;
  r = (__builtin_fabsf(l) < __uint_as_float(0x7f800000u)) ? r : l;
  return r - (dn ? __uint_as_float(0x41b17218u) : 0.0f);
}

// ---- shared math. All ten kernels rotate the pair (x[i], x[i+c/2]) (NeoX layout, second pointer = first + c/2,
// k_rope :52397-52421), loop/guard on i < c/2 with c/2 = (c + (c>>>31)) >> 1 (:52358-52359).

// Plain angle: inv = exp2(-1.4426950 * (ln(theta) * ((float)(2*i) / (float)c))), IEEE division (div_scale/rcp/fma/
// div_fmas/div_fixup, k_rope :52430-52451). ln(theta) is the ocml logf expansion (v_log_f32 + 0x3f317217/0x3377d1cf
// fixup, :52391-52409); `logf` in HIP compiles to the same expansion.
__device__ static inline float plain_inv(float lnt, int i, int c) {
  float q = (float)(2 * i) / (float)c;
  return __builtin_amdgcn_exp2f(-1.44269502162933349609375f * (lnt * q));
}
// Angle in revolutions: (p * inv) * 0.15915494 (k_rope :52456-52460). Hardware sin/cos take revolutions (:52461-52462).
// Plain rotation (k_rope :52463-52483): t = s*x1 ; y0 = fma(co,x0,-t) ; y1 = fma(co,x1,s*x0).
__device__ static inline void rot_plain(unsigned short* p0, unsigned short* p1, float ang) {
  float s = __builtin_amdgcn_sinf(ang), co = __builtin_amdgcn_cosf(ang);
  float x0 = bf2f(*p0), x1 = bf2f(*p1);
  float t = s * x1;
  float y0 = __builtin_fmaf(co, x0, -t);
  float y1 = __builtin_fmaf(co, x1, s * x0);
  *p0 = f2bf_rne(y0);
  *p1 = f2bf_rne(y1);
}
// YaRN rotation (k_rope_yarn :52550-52590): sin and cos are each multiplied by mscale first, then
// t = s'*x1 ; y0 = fma(co',x0,-t) ; y1 = fma(s',x0,co'*x1).
__device__ static inline void rot_yarn(unsigned short* p0, unsigned short* p1, float ang, float ms) {
  float s = __builtin_amdgcn_sinf(ang), co = __builtin_amdgcn_cosf(ang);
  s = ms * s;
  co = ms * co;
  float x0 = bf2f(*p0), x1 = bf2f(*p1);
  float t = s * x1;
  float y0 = __builtin_fmaf(co, x0, -t);
  float y1 = __builtin_fmaf(s, x0, co * x1);
  *p0 = f2bf_rne(y0);
  *p1 = f2bf_rne(y1);
}
// mRoPE section pick for pair i (k_rope_blocks_mrope :54476-54500; the w variants do the same at 53307+):
//   r = i%3 (mulhi 0xaaaaaaab / 0x55555556) ; sel = (i<30 && r==2) ? 2 : ((i<33 && r==1) ? 1 : 0)
__device__ static inline int mrope_sel(int i) {
  int r = i - 3 * (i / 3);
  int sel = (i < 30 && r == 2) ? 2 : 0;
  return (i < 33 && r == 1) ? 1 : sel;
}

// k_rope(x@0, a@8, b@12, c@16, theta@20, off@24, pos@32): grid (token, head). Row base = (a*bx+by)*b elements, 64-bit
// (:52362-52373). p = pos ? pos[bx] : off + bx (:52350-52358, null test :52352). Listing :52343-52487.
extern "C" __global__ void k_rope(unsigned short* x, int a, int b, int c, float theta, int off, const int* pos) {
  int bx = blockIdx.x, by = blockIdx.y;
  int p = pos ? pos[bx] : off + bx;
  int half = c / 2;
  long base = ((long)a * bx + by) * b;
  float lnt = vlog(theta);
  for (int i = threadIdx.x; i < half; i += blockDim.x)
    rot_plain(x + base + i, x + base + i + half, ((float)p * plain_inv(lnt, i, c)) * 0.15915494f);
}

// k_rope_yarn(x@0, a@8, b@12, c@16, off@20, pos@24, freqs@32, mscale@40): as k_rope but inv = freqs[i]
// (global_load_b32 at 4*i, :52530-52545), ang = (p*freqs[i])*0.15915494 (:52555-52557), yarn rotation. Listing :52490-52603.
extern "C" __global__ void k_rope_yarn(unsigned short* x, int a, int b, int c, int off, const int* pos,
                                       const float* freqs, float mscale) {
  int bx = blockIdx.x, by = blockIdx.y;
  int p = pos ? pos[bx] : off + bx;
  int half = c / 2;
  long base = ((long)a * bx + by) * b;
  for (int i = threadIdx.x; i < half; i += blockDim.x)
    rot_yarn(x + base + i, x + base + i + half, ((float)p * freqs[i]) * 0.15915494f, mscale);
}

// k_rope_blocks(x@0, c@8, theta@12, off@16): grid (block). Row = 128 bf16 (s_lshl_b64 8 bytes, :54246). Position is
// 4*(off+bx) (s_lshl_b32 2, :54235) converted to float. Listing :54205-54323.
extern "C" __global__ void k_rope_blocks(unsigned short* x, int c, float theta, int off) {
  int bx = blockIdx.x;
  int half = c / 2;
  float p = (float)((off + bx) << 2);
  unsigned short* row = x + (long)bx * 128;
  float lnt = vlog(theta);
  for (int i = threadIdx.x; i < half; i += blockDim.x)
    rot_plain(row + i, row + i + half, (p * plain_inv(lnt, i, c)) * 0.15915494f);
}

// k_rope_blocks_yarn(x@0, c@8, off@12, freqs@16, mscale@24). Listing :54325-54414.
extern "C" __global__ void k_rope_blocks_yarn(unsigned short* x, int c, int off, const float* freqs, float mscale) {
  int bx = blockIdx.x;
  int half = c / 2;
  float p = (float)((off + bx) << 2);
  unsigned short* row = x + (long)bx * 128;
  for (int i = threadIdx.x; i < half; i += blockDim.x)
    rot_yarn(row + i, row + i + half, (freqs[i] * p) * 0.15915494f, mscale);
}

// k_rope_blocks_mrope(x@0, c@8, off@12, pos@16, theta@24): the position is pos[(off+bx)*12 + sel(i)] (stride 48 bytes,
// s_mul_i32 48, :54437; global_load_b32 v3 off s[6:7], :54480). Listing :54416-54564.
extern "C" __global__ void k_rope_blocks_mrope(unsigned short* x, int c, int off, const int* pos, float theta) {
  int bx = blockIdx.x;
  int half = c / 2;
  const int* prow = pos + (long)(off + bx) * 12;
  unsigned short* row = x + (long)bx * 128;
  float lnt = vlog(theta);
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    float p = (float)prow[mrope_sel(i)];
    rot_plain(row + i, row + i + half, (plain_inv(lnt, i, c) * p) * 0.15915494f);
  }
}

// k_rope_blocks_mrope_yarn(x@0, c@8, off@12, pos@16, freqs@24, mscale@32). Listing :54566-54699.
extern "C" __global__ void k_rope_blocks_mrope_yarn(unsigned short* x, int c, int off, const int* pos,
                                                    const float* freqs, float mscale) {
  int bx = blockIdx.x;
  int half = c / 2;
  const int* prow = pos + (long)(off + bx) * 12;
  unsigned short* row = x + (long)bx * 128;
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    float p = (float)prow[mrope_sel(i)];
    rot_yarn(row + i, row + i + half, (freqs[i] * p) * 0.15915494f, mscale);
  }
}

// ---- "_w" kernels: flat 1D launch over n rows * (c/2) pairs. t = blockIdx.x*blockDim.x + threadIdx.x (64-bit
// v_mad_u64_u32, k_rope_w :52613-52615), guard t < (c/2)*n (:52618-52620). row r = t / half, i = t - r*half
// (64-bit division expansion :52625-52775), token = r / heads (signed 32-bit division, :52780-52815).
// Element address x + r*stride + i (v_mad_i64_i32 v2,s5, :52829).

// k_rope_w(x@0, n@8, stride@12, c@16, theta@20, off@24, heads@28, pos@32). p = pos ? pos[token] : off + token
// (:52814-52826). Listing :52605-52969.
extern "C" __global__ void k_rope_w(unsigned short* x, int n, int stride, int c, float theta, int off, int heads,
                                    const int* pos) {
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  int half = c / 2;
  if (t >= (long)half * n) return;
  long r = t / half;
  int i = (int)(t - r * half);
  int token = (int)r / heads;
  int p = pos ? pos[token] : off + token;
  unsigned short* row = x + r * stride;
  float lnt = vlog(theta);
  rot_plain(row + i, row + i + half, ((float)p * plain_inv(lnt, i, c)) * 0.15915494f);
}

// k_rope_w_yarn(x@0, n@8, stride@12, c@16, off@20, heads@24, pos@32, freqs@40, mscale@48). Listing :52971-53305.
extern "C" __global__ void k_rope_w_yarn(unsigned short* x, int n, int stride, int c, int off, int heads,
                                         const int* pos, const float* freqs, float mscale) {
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  int half = c / 2;
  if (t >= (long)half * n) return;
  long r = t / half;
  int i = (int)(t - r * half);
  int token = (int)r / heads;
  int p = pos ? pos[token] : off + token;
  unsigned short* row = x + r * stride;
  rot_yarn(row + i, row + i + half, (freqs[i] * (float)p) * 0.15915494f, mscale);
}

// k_rope_w_mrope(x@0, n@8, stride@12, c@16, theta@20, off@24, heads@28, pos@32): the position is
// pos[(off+token)*3 + sel(i)] (v_mad_i64_i32 *12 + pos). Listing :53307-53682.
extern "C" __global__ void k_rope_w_mrope(unsigned short* x, int n, int stride, int c, float theta, int off,
                                          int heads, const int* pos) {
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  int half = c / 2;
  if (t >= (long)half * n) return;
  long r = t / half;
  int i = (int)(t - r * half);
  int token = (int)r / heads;
  float p = (float)pos[(long)(off + token) * 3 + mrope_sel(i)];
  unsigned short* row = x + r * stride;
  float lnt = vlog(theta);
  rot_plain(row + i, row + i + half, (plain_inv(lnt, i, c) * p) * 0.15915494f);
}

// k_rope_w_mrope_yarn(x@0, n@8, stride@12, c@16, off@20, heads@24, pos@32, freqs@40, mscale@48). Listing :53684-53976.
extern "C" __global__ void k_rope_w_mrope_yarn(unsigned short* x, int n, int stride, int c, int off, int heads,
                                               const int* pos, const float* freqs, float mscale) {
  long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  int half = c / 2;
  if (t >= (long)half * n) return;
  long r = t / half;
  int i = (int)(t - r * half);
  int token = (int)r / heads;
  float p = (float)pos[(long)(off + token) * 3 + mrope_sel(i)];
  unsigned short* row = x + r * stride;
  rot_yarn(row + i, row + i + half, (freqs[i] * p) * 0.15915494f, mscale);
}
