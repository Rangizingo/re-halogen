// Rewrites of the "i4r_rows" family from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o i4r_rows.co i4r_rows.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). VOP3 instructions appear as ".long <dword1>" plus a
// mis-decoded second dword; they are decoded by hand (for example ".long 0xcc0f40NN" + "v_mul_i32_i24 v1, 0xe408, vM" is
// v_pk_add_f16 vNN, vM, 0xe408).
// Cite note: kernel start/end lines in codeobj_53000.s are exact (see rewrite/reports/g22.md). Line numbers inside a kernel that are written
// as :NNNNN were read off a filtered re-listing; for the long unrolled kernels they can be off by tens of lines. Instruction names are exact.
#include <hip/hip_runtime.h>

// Keep float multiply and add separate unless an explicit __builtin_fmaf is written (hipcc fuses a*b+c by default).
#pragma clang fp contract(off)

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));

// i4r weights: int4, low nibble first, one f16 scale per 128 weights along K; value = (nibble - 8) * scale.
// The vendor builds f16 pairs (1024 + n0, 1024 + n1) with v_and 0x0f0f0f0f / v_lshrrev 4 / v_perm_b32 / v_or 0x64006400 and subtracts 1032.0
// (v_pk_add_f16 with 0xe408), which gives (n - 8) exactly; then v_dot2acc_f32_f16 against the f16 activation pairs. The builtin below emits the
// same instruction, so the accumulation matches bit for bit.

// halogen::(anonymous namespace)::k_i4r_rows<K>(unsigned char const*, long, int, _Float16 const*, int, int const*, float*)
// _ZN7halogen12_GLOBAL__N_110k_i4r_rowsILi640EEEvPKhliPKDF16_iPKiPf (and ILi2560E)
// kernarg: W@0, soff@8 (scale base offset from W), N@16 (rows per expert), x@24 (f16 rows of K), xdiv@32, rowmap@40, y@48.
// Grid = (N/64) * nbatch blocks of 256 threads (64 rows per block). b = blockIdx.x / (N/64) (unsigned divide, listing :1994300-1994330),
// blk = blockIdx.x - b*(N/64). The activation row is x + (b / xdiv)*K halves (signed divide, :1994348-1994385), staged to LDS by the
// first K/8 threads, 16 B each (global_load_b128 + ds_store_b128, :1994390-1994392). barrier.
// Lane mapping: warp = tid>>5 (8 rows per warp), row r = blk*64 + 8*warp + ((tid>>2)&7) (v_or3_b32 :1994425), lane quad q = tid&3.
// Expert row = rowmap[b]*N + r; weight row at W + row*(K/2) bytes, scale row at W + soff + row*(K/128)*2 (v_mad_u64_u32 0x140 / *10, :1994447-1994455).
// Lane q handles, for each 128-weight block m, weights 128m + 32q .. +31: 16 B at W row + 64m + 16q (global_load_b128 at offsets 0,64,128,..).
// acc_m = 0; for pairs p = 0..15 in ascending order (word p/4... elements 2p, 2p+1): acc_m = dot2(f16(n-8) pair, x pair, acc_m).
// Epilogue (:2015-2030 of the K=640 kernel, listing :1994843-1994857): total = fma(scale_0, acc_0, +0.0); total = fma(scale_m, acc_m, total) for
// m = 1..; (v_fma_mix_f32 with f16 scale) then total += total[lane^1], total += total[lane^2] (DPP row_xmask 1, 2);
// lane q == 0 stores y[b*N + r] = total.
template <int K>
__device__ static inline void i4r_rows_body(const unsigned char* W, long soff, int N, const _Float16* x, int xdiv, const int* rowmap,
                                            float* y) {
  constexpr int NB = K / 128;
  __shared__ _Float16 xs[K];
  int tid = threadIdx.x;
  int nbl = (N + (int)((unsigned)(N >> 31) >> 26)) >> 6;
  int b = (int)((unsigned)blockIdx.x / (unsigned)nbl);
  int blk = blockIdx.x - b * nbl;
  const _Float16* xr = x + (long)(b / xdiv) * K;
  for (int c = tid; c < K / 8; c += blockDim.x) *(float4*)(xs + 8 * c) = *(const float4*)(xr + 8 * c);
  __syncthreads();
  int q = tid & 3;
  int r = blk * 64 + 8 * (tid >> 5) + ((tid >> 2) & 7);
  long row = (long)rowmap[b] * N + r;
  const unsigned char* wp = W + row * (K / 2) + 16 * q;
  const _Float16* sp = (const _Float16*)(W + soff + row * (NB * 2));
  float acc[NB];
  for (int m = 0; m < NB; m++) {
    uint4 wv = *(const uint4*)(wp + 64 * m);
    unsigned wd[4] = {wv.x, wv.y, wv.z, wv.w};
    float a = 0.0f;
    for (int w = 0; w < 4; w++)
      for (int k = 0; k < 4; k++) {
        unsigned byte = (wd[w] >> (8 * k)) & 0xffu;
        half2_t wh = {(_Float16)(int)((byte & 15u) - 8u), (_Float16)(int)((byte >> 4) - 8u)};
        half2_t xh = *(const half2_t*)(xs + 128 * m + 32 * q + 8 * w + 2 * k);
        a = __builtin_amdgcn_fdot2(wh, xh, a, false);
      }
    acc[m] = a;
  }
  float tot = __builtin_fmaf((float)sp[0], acc[0], 0.0f);
  for (int m = 1; m < NB; m++) tot = __builtin_fmaf((float)sp[m], acc[m], tot);
  tot = tot + __shfl_xor(tot, 1);
  tot = tot + __shfl_xor(tot, 2);
  if (q == 0) y[(long)b * N + r] = tot;
}

extern "C" __global__ void k_i4r_rows_640(const unsigned char* W, long soff, int N, const _Float16* x, int xdiv, const int* rowmap,
                                          float* y) {
  i4r_rows_body<640>(W, soff, N, x, xdiv, rowmap, y);
}
extern "C" __global__ void k_i4r_rows_2560(const unsigned char* W, long soff, int N, const _Float16* x, int xdiv, const int* rowmap,
                                           float* y) {
  i4r_rows_body<2560>(W, soff, N, x, xdiv, rowmap, y);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Helpers shared by the fused variants (the fold tail is the same as in i4r_fold.hip).
__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
__device__ static inline float xstep(float v, int m) {
  float partner = __shfl_xor(v, m);
  return ((threadIdx.x & m) ? -v : v) + partner;
}
// 4-point transform in the lane, xor butterflies 1,2,4,8,16, * 0x3db504f3, then y_k = bf16_rne(half(h[k]) * e_k + (-0.0)),
// out_k = bf16_rne(fma(z, bf16(r[k]), y_k)) (see i4r_fold.hip).
__device__ static inline void fold_tail(const float acc[4], const unsigned short* h, const unsigned short* r, float zt, unsigned short* o) {
  float p0 = acc[0] + acc[1], p1 = acc[0] - acc[1], p2 = acc[2] + acc[3], p3 = acc[2] - acc[3];
  float e[4] = {p0 + p2, p1 + p3, p0 - p2, p1 - p3};
  for (int m = 1; m <= 16; m <<= 1)
    for (int k = 0; k < 4; k++) e[k] = xstep(e[k], m);
  float C = __uint_as_float(0x3db504f3u);
  for (int k = 0; k < 4; k++) {
    float v = e[k] * C;
    float y = bf2f(f2bf_rne(__builtin_fmaf((float)__builtin_bit_cast(_Float16, h[k]), v, -0.0f)));
    o[k] = f2bf_rne(__builtin_fmaf(zt, bf2f(r[k]), y));
  }
}

// halogen::(anonymous namespace)::k_i4r_rows_fl(unsigned char const*, long, _Float16 const*, int const*, float*, float const*, _Float16 const*,
//                                              unsigned short const*, float const*, unsigned short*, unsigned int*)
// _ZN7halogen12_GLOBAL__N_113k_i4r_rows_flEPKhlPKDF16_PKiPfPKfS4_PKtS9_PtPj
// kernarg: W@0, soff@8, x@16, rowmap@24, y@32, wt@40, h@48, r@56, z@64, out@72, ctr@80.
// Down-projection GEMV (N = 2560 rows, K = 640) of 10 expert slots per token, fused with the fold of the 10 slot outputs.
// Block blockIdx.x: b = blockIdx.x / 40 (slot index, magic 0xcccccccd >> 5, codeobj_53000.s:27091-27094), blk = blockIdx.x - 40*b (64 rows each).
// x row = x + b*640 halves (:27100-27104); expert = rowmap[b] (:27111); row = expert*2560 + r (v_mad_i64_i32 0xa00, :27124). GEMV as k_i4r_rows<640>
// except the partial y[b*2560 + r] is written for lane quad 0 (:27181-27187; stride 0x2800 B per b).
// After the GEMV: tok = blockIdx.x / 400 (magic 0x51eb851f >> 7, :27196), counter index c = tok*20 + blk/2 (:27197-27202); thread 0 does
// old = atomicAdd(ctr + c, 1) (global_atomic_add_u32 glc :27217) after the block's y stores are visible (buffer_gl1_inv/gl0_inv, s_barrier);
// the block that sees old == 19 (the 20th of 10 slots x 2 blocks) is the last and performs the fold for columns 128*(blk/2) .. +127, one float4 per
// thread for tid < 32 (:27231-27238): acc[k] = ((0 + wt[10 tok + 0]*y0[k]) + wt[10 tok + 1]*y1[k]) + ... for the ten slots, y row of slot m at
// y + (10 tok + m)*2560 (glc loads), then fold_tail with h@48 (f16 x4 at the column), r@56 (+ tok*2560 bf16), z[tok], out@72 (+ tok*2560).
// Thread 0 then resets ctr[c] = 0 (global_store_b32 :27355).
template <bool SEL>
__device__ static inline void rows_fl_body(const unsigned char* W, long soff, const _Float16* x, const int* rowmap, float* y, const float* wt,
                                         const unsigned short* h, const unsigned short* r, const float* z, unsigned short* out, unsigned* ctr,
                                         const int* selp) {
  constexpr int K = 640, NB = 5, N = 2560;
  __shared__ _Float16 xs[K];
  __shared__ int last;
  int tid = threadIdx.x;
  int b0 = blockIdx.x / 40;
  int blk = blockIdx.x - 40 * b0;
  int b = SEL ? selp[b0] : b0;
  const _Float16* xr = x + (long)b * K;
  for (int c = tid; c < K / 8; c += blockDim.x) *(float4*)(xs + 8 * c) = *(const float4*)(xr + 8 * c);
  __syncthreads();
  int q = tid & 3;
  int rr = blk * 64 + 8 * (tid >> 5) + ((tid >> 2) & 7);
  long row = (long)rowmap[b] * N + rr;
  const unsigned char* wp = W + row * (K / 2) + 16 * q;
  const _Float16* sp = (const _Float16*)(W + soff + row * (NB * 2));
  float acc[NB];
  for (int m = 0; m < NB; m++) {
    uint4 wv = *(const uint4*)(wp + 64 * m);
    unsigned wd[4] = {wv.x, wv.y, wv.z, wv.w};
    float a = 0.0f;
    for (int w = 0; w < 4; w++)
      for (int k = 0; k < 4; k++) {
        unsigned byte = (wd[w] >> (8 * k)) & 0xffu;
        half2_t wh = {(_Float16)(int)((byte & 15u) - 8u), (_Float16)(int)((byte >> 4) - 8u)};
        half2_t xh = *(const half2_t*)(xs + 128 * m + 32 * q + 8 * w + 2 * k);
        a = __builtin_amdgcn_fdot2(wh, xh, a, false);
      }
    acc[m] = a;
  }
  float tot = __builtin_fmaf((float)sp[0], acc[0], 0.0f);
  for (int m = 1; m < NB; m++) tot = __builtin_fmaf((float)sp[m], acc[m], tot);
  tot = tot + __shfl_xor(tot, 1);
  tot = tot + __shfl_xor(tot, 2);
  if (q == 0) y[(long)b * N + rr] = tot;
  int tok = b / 10;
  int c = tok * 20 + (blk >> 1);
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    unsigned old = atomicAdd(&ctr[c], 1u);
    last = (old == 19u);
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  if (tid < 32) {
    int col = (blk >> 1) * 128 + 4 * tid;
    float f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int m = 0; m < 10; m++) {
      float wm = wt[tok * 10 + m];
      const volatile float* yp = (const volatile float*)(y + (long)(tok * 10 + m) * N + col);
      for (int k = 0; k < 4; k++) f[k] = f[k] + wm * yp[k];
    }
    fold_tail(f, h + col, r + (long)tok * N + col, z[tok], out + (long)tok * N + col);
  }
  if (tid == 0) ctr[c] = 0u;
}

extern "C" __global__ void k_i4r_rows_fl(const unsigned char* W, long soff, const _Float16* x, const int* rowmap, float* y, const float* wt,
                                         const unsigned short* h, const unsigned short* r, const float* z, unsigned short* out, unsigned* ctr) {
  rows_fl_body<false>(W, soff, x, rowmap, y, wt, h, r, z, out, ctr, nullptr);
}

// k_i4r_rows_fl_o: same as k_i4r_rows_fl with one more kernarg, sel@88 (int const*). Block group b0 = blockIdx.x / 40 works on slot
// b = sel[b0] (s_load_b32 s26, [sel + 4*b0], listing codeobj_53000.s:28160-28162): x row b (s_mul_i32 s1, s26, 0x500), rowmap[b], y row b
// (0x2800 per b), token = b / 10 (magic 0x66666667 >> 2) and counter c = token*20 + blk/2. The counter still completes after 20 arrivals.
extern "C" __global__ void k_i4r_rows_fl_o(const unsigned char* W, long soff, const _Float16* x, const int* rowmap, float* y, const float* wt,
                                           const unsigned short* h, const unsigned short* r, const float* z, unsigned short* out, unsigned* ctr,
                                           const int* sel) {
  rows_fl_body<true>(W, soff, x, rowmap, y, wt, h, r, z, out, ctr, sel);
}

// halogen::(anonymous namespace)::k_i4r_rows_rot(unsigned char const*, long, int, unsigned short const*, _Float16 const*, int const*, float*)
// _ZN7halogen12_GLOBAL__N_114k_i4r_rows_rotEPKhliPKtPKDF16_PKiPf
// kernarg: W@0, soff@8, N@16, x@24 (bf16 [tok][2560]), su@32 (f16 [2560]), rowmap@40, y@48. Same GEMV as k_i4r_rows<2560> (b = blockIdx.x / (N/64),
// rowmap[b], y[b*N + r]) but the activation is built in the kernel from the bf16 row of token b/10 (b / 10 via magic 0xcccccccd >> 3,
// s_mulk_i32 0x1400, codeobj_53000.s:23062-23066):
//   for each 128-block j (warp w = tid>>5 handles j = w, w+8, w+16 while j < 20) and lane l, elements 4l..4l+3 (global_load_b64 of x and of su):
//     t1 = su1*x1 (rounded), t3 = su3*x3 (rounded)       (v_fma_mix_f32 ..., neg(0), :23078-23083)
//     p0 = fma(su0,x0,t1), p1 = fma(su0,x0,-t1), p2 = fma(su2,x2,t3), p3 = fma(su2,x2,-t3)   (:23080-23085)
//     e0 = p0+p2, e1 = p1+p3, e2 = p0-p2, e3 = p1-p3; xor butterflies 1,2,4,8,16; xs[128j + 4l + k] = f16(fma(e_k, 0x3db504f3, +0.0))
//     (v_fma_mixlo/hi_f16 ..., s22, 0 and ds_store_b64, :23280-23286).
extern "C" __global__ void k_i4r_rows_rot(const unsigned char* W, long soff, int N, const unsigned short* x, const _Float16* su, const int* rowmap,
                                          float* y) {
  constexpr int K = 2560, NB = 20;
  __shared__ _Float16 xs[K];
  int tid = threadIdx.x;
  int nbl = (N + (int)((unsigned)(N >> 31) >> 26)) >> 6;
  int b = (int)((unsigned)blockIdx.x / (unsigned)nbl);
  int blk = blockIdx.x - b * nbl;
  const unsigned short* xr = x + (long)(b / 10) * K;
  int lane = tid & 31;
  float C = __uint_as_float(0x3db504f3u);
  for (int j = tid >> 5; j < NB; j += 8) {
    int base = 128 * j + 4 * lane;
    float x0 = bf2f(xr[base]), x1 = bf2f(xr[base + 1]), x2 = bf2f(xr[base + 2]), x3 = bf2f(xr[base + 3]);
    float s0 = (float)su[base], s1 = (float)su[base + 1], s2 = (float)su[base + 2], s3 = (float)su[base + 3];
    float t1 = __builtin_fmaf(s1, x1, -0.0f), t3 = __builtin_fmaf(s3, x3, -0.0f);
    float p0 = __builtin_fmaf(s0, x0, t1), p1 = __builtin_fmaf(s0, x0, -t1);
    float p2 = __builtin_fmaf(s2, x2, t3), p3 = __builtin_fmaf(s2, x2, -t3);
    float e[4] = {p0 + p2, p1 + p3, p0 - p2, p1 - p3};
    for (int m = 1; m <= 16; m <<= 1)
      for (int k = 0; k < 4; k++) e[k] = xstep(e[k], m);
    for (int k = 0; k < 4; k++) xs[base + k] = (_Float16)__builtin_fmaf(e[k], C, 0.0f);
  }
  __syncthreads();
  int q = tid & 3;
  int rr = blk * 64 + 8 * (tid >> 5) + ((tid >> 2) & 7);
  long row = (long)rowmap[b] * N + rr;
  const unsigned char* wp = W + row * (K / 2) + 16 * q;
  const _Float16* sp = (const _Float16*)(W + soff + row * (NB * 2));
  float acc[NB];
  for (int m = 0; m < NB; m++) {
    uint4 wv = *(const uint4*)(wp + 64 * m);
    unsigned wd[4] = {wv.x, wv.y, wv.z, wv.w};
    float a = 0.0f;
    for (int w = 0; w < 4; w++)
      for (int k = 0; k < 4; k++) {
        unsigned byte = (wd[w] >> (8 * k)) & 0xffu;
        half2_t wh = {(_Float16)(int)((byte & 15u) - 8u), (_Float16)(int)((byte >> 4) - 8u)};
        half2_t xh = *(const half2_t*)(xs + 128 * m + 32 * q + 8 * w + 2 * k);
        a = __builtin_amdgcn_fdot2(wh, xh, a, false);
      }
    acc[m] = a;
  }
  float tot = __builtin_fmaf((float)sp[0], acc[0], 0.0f);
  for (int m = 1; m < NB; m++) tot = __builtin_fmaf((float)sp[m], acc[m], tot);
  tot = tot + __shfl_xor(tot, 1);
  tot = tot + __shfl_xor(tot, 2);
  if (q == 0) y[(long)b * N + rr] = tot;
}

// silu(g) = g / (1 + exp2(g * -1.442695)) as in k_i4r_gu_epi (v_mul 0xbfb8aa3b, v_exp_f32, v_add 1.0, IEEE division).
__device__ static inline float silu_f(float g) {
  float e = __builtin_amdgcn_exp2f(g * -1.44269502162933349609375f);
  return g / (1.0f + e);
}

// halogen::(anonymous namespace)::k_i4r_rows_epi(unsigned char const*, long, int, float const*, _Float16 const*, _Float16 const*, int const*, float*)
// _ZN7halogen12_GLOBAL__N_114k_i4r_rows_epiEPKhliPKfPKDF16_S6_PKiPf
// kernarg: W@0, soff@8, N@16, g@24 (f32 rows of 1280: gate 640 then up 640), a@32 (f16 [1280]), bq@40 (f16 [640]), rowmap@48, y@56.
// Same GEMV as k_i4r_rows<640> (b = blockIdx.x / (N/64), rowmap[b], y[b*N + r]), but the activation row is produced in the kernel from g row b
// (0x1400 B per row, codeobj_53000.s:25391-25400) by exactly the per-item pipeline of k_i4r_gu_epi: for 128-block j = warp (tid < 160, five warps):
// gate and up transformed (4-point in lane + xor butterflies + * 0x3db504f3), scaled by a (gate) and a+640 (up) with fma(a, x, -0.0), silu(gate) * up,
// pair products with bq, second transform, xs[128j + 4l + k] = f16(fma(e_k, 0x3db504f3, +0.0)) (stored with ds_store_b64).
extern "C" __global__ void k_i4r_rows_epi(const unsigned char* W, long soff, int N, const float* g, const _Float16* a, const _Float16* bq,
                                          const int* rowmap, float* y) {
  constexpr int K = 640, NB = 5;
  __shared__ _Float16 xs[K];
  int tid = threadIdx.x;
  int nbl = (N + (int)((unsigned)(N >> 31) >> 26)) >> 6;
  int b = (int)((unsigned)blockIdx.x / (unsigned)nbl);
  int blk = blockIdx.x - b * nbl;
  int lane = tid & 31;
  if (tid < 160) {
    int j = tid >> 5;
    int ix = j * 128 + lane * 4;
    const float* gr = g + (long)b * 1280 + ix;
    float4 gv = *(const float4*)gr;
    float4 uv = *(const float4*)(gr + 640);
    float xg[4] = {gv.x, gv.y, gv.z, gv.w};
    float xu[4] = {uv.x, uv.y, uv.z, uv.w};
    float C = __uint_as_float(0x3db504f3u);
    float e[4], f[4];
    for (int pass = 0; pass < 2; pass++) {
      float* v = pass ? xu : xg;
      float* r = pass ? f : e;
      float p0 = v[0] + v[1], p1 = v[0] - v[1], p2 = v[2] + v[3], p3 = v[2] - v[3];
      r[0] = p0 + p2; r[2] = p0 - p2; r[1] = p1 + p3; r[3] = p1 - p3;
      for (int m = 1; m <= 16; m <<= 1)
        for (int k = 0; k < 4; k++) r[k] = xstep(r[k], m);
      for (int k = 0; k < 4; k++) r[k] = r[k] * C;
    }
    float prod[4];
    for (int k = 0; k < 4; k++) {
      float gs = __builtin_fmaf((float)a[ix + k], e[k], -0.0f);
      float us = __builtin_fmaf((float)a[640 + ix + k], f[k], -0.0f);
      prod[k] = us * silu_f(gs);
    }
    float bb[4];
    for (int k = 0; k < 4; k++) bb[k] = (float)bq[ix + k];
    float t1 = __builtin_fmaf(bb[1], prod[1], -0.0f), t3 = __builtin_fmaf(bb[3], prod[3], -0.0f);
    float s01 = __builtin_fmaf(prod[0], bb[0], t1), d01 = __builtin_fmaf(prod[0], bb[0], -t1);
    float s23 = __builtin_fmaf(prod[2], bb[2], t3), d23 = __builtin_fmaf(prod[2], bb[2], -t3);
    float h[4] = {s01 + s23, d01 + d23, s01 - s23, d01 - d23};
    for (int m = 1; m <= 16; m <<= 1)
      for (int k = 0; k < 4; k++) h[k] = xstep(h[k], m);
    for (int k = 0; k < 4; k++) xs[ix + k] = (_Float16)__builtin_fmaf(h[k], C, 0.0f);
  }
  __syncthreads();
  int q = tid & 3;
  int rr = blk * 64 + 8 * (tid >> 5) + ((tid >> 2) & 7);
  long row = (long)rowmap[b] * N + rr;
  const unsigned char* wp = W + row * (K / 2) + 16 * q;
  const _Float16* sp = (const _Float16*)(W + soff + row * (NB * 2));
  float acc[NB];
  for (int m = 0; m < NB; m++) {
    uint4 wv = *(const uint4*)(wp + 64 * m);
    unsigned wd[4] = {wv.x, wv.y, wv.z, wv.w};
    float aa = 0.0f;
    for (int w = 0; w < 4; w++)
      for (int k = 0; k < 4; k++) {
        unsigned byte = (wd[w] >> (8 * k)) & 0xffu;
        half2_t wh = {(_Float16)(int)((byte & 15u) - 8u), (_Float16)(int)((byte >> 4) - 8u)};
        half2_t xh = *(const half2_t*)(xs + 128 * m + 32 * q + 8 * w + 2 * k);
        aa = __builtin_amdgcn_fdot2(wh, xh, aa, false);
      }
    acc[m] = aa;
  }
  float tot = __builtin_fmaf((float)sp[0], acc[0], 0.0f);
  for (int m = 1; m < NB; m++) tot = __builtin_fmaf((float)sp[m], acc[m], tot);
  tot = tot + __shfl_xor(tot, 1);
  tot = tot + __shfl_xor(tot, 2);
  if (q == 0) y[(long)b * N + rr] = tot;
}
