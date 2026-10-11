// Rewrite of the "dnc_ut4" family (halogen::(anonymous namespace)::k_dnc_ut4<N, M, b1, b2, b3>) from codeobj_1146000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dnc_ut4.co dnc_ut4.hip
// Listing: kernels-disasm/codeobj_1146000.s (not in git). Line cites below are for k_dnc_ut4<4,1,false,false,false>
// (listing lines 544252-544912) unless another instance is named.
//
// Signature: (unsigned short const* qkv, float const* beta, float const* K, float const* A, float* U, float* W, int n)
// kernarg: qkv@0 beta@8 K@16 A@24 U@32 W@40 n@48.
// Grid (ceil(n/64), 48), block 256, static LDS 17408 B (group_segment). One block = one 64-token chunk of one head.
// Names are inferred: this is the DeltaNet chunk "UT" step. M[j][i] is the strictly lower triangular matrix
// -beta_j * (K_j . K_i) * exp(A_j - A_i); T = its forward-substituted inverse; U = T * (beta * V), W = T * (beta * exp(A) * K).
//
// Layout (inferred from the address arithmetic):
//   qkv   : bf16 [token][10240]; V for head h at element 4096 + h*128 (0x5000 B token stride, +0x2000 B, 544701/544743).
//   beta  : f32 [token][48]   (544265: idx = token*48 + head)
//   K     : f32 [token][48][128] (0x6000 B token stride, 544326)
//   A     : f32 [token][48]   (cumulative log decay)
//   U, W  : f32 [token][48][128]
#include <hip/hip_runtime.h>
// The vendor never fuses a multiply with the following add unless the listing shows a v_fma; keep the compiler from doing it.
#pragma clang fp contract(off)

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// expf as the vendor inlines it: Cody-Waite split of x*log2(e), v_exp_f32, v_ldexp_f32, then range clamps.
// listing 544287-544304: v_mul 0x3fb8aa3b ; v_fma ; v_rndne ; v_fmamk 0x32a5705f ; v_sub ; v_add ; v_cvt_i32 ; v_exp ; v_ldexp ;
// v_cmp_ngt 0xc2ce8ed0 (below it: 0) ; v_cmp_nlt 0x42b17218 (above it: +inf).
__device__ static inline float vexpf(float x) {
  const float l2e = __uint_as_float(0x3fb8aa3b);
  const float l2e_lo = __uint_as_float(0x32a5705f);
  float p = x * l2e;
  float r = __builtin_fmaf(l2e, x, -p);
  float rn = __builtin_rintf(p);
  r = __builtin_fmaf(x, l2e_lo, r);
  float f = (p - rn) + r;
  int i = (int)rn;
  float e = __builtin_amdgcn_exp2f(f);
  e = __builtin_amdgcn_ldexpf(e, i);
  if (__uint_as_float(0xc2ce8ed0) > x) e = 0.0f;
  if (__uint_as_float(0x42b17218) < x) e = __uint_as_float(0x7f800000);
  return e;
}

// LDS word offsets (floats): matrix M/T at 0 (row pitch 65 words = 0x104 B), beta at 16640 B, exp(A) at 16896 B, scratch row at 17152 B.
#define LDS_BETA 4160
#define LDS_EXPA 4224
#define LDS_SCR 4288

// SWAP: b2 instances read the head from ctaid.x and the chunk from ctaid.y (listing 539628 vs 544252: s_lshl_b32 s19,s3,6 ; no s_mov_b32 s16,s3).
// MP: b3 instance; Gram dot operands go through f16: x -> f16(x * 16) -> f32 * 0.0625 (v_fma_mixlo_f16 / v_fma_mix_f32, s_mov 0x41800000 / 0x3d800000,
// listing 539628-540556 first loop), then the same ascending f32 fma chain.
template <bool SWAP, bool MP>
__device__ static inline void ut4_body(const unsigned short* qkv, const float* beta, const float* K, const float* A, float* U,
                                       float* W, int n) {
  __shared__ float lds[4352];
  const int tid = threadIdx.x;
  const int h = SWAP ? blockIdx.x : blockIdx.y;
  const int base = (SWAP ? blockIdx.y : blockIdx.x) * 64;

  // Phase A (544263-544305): threads 0..63. beta is loaded only for token < n (else 0); A is loaded unguarded.
  if (tid < 64) {
    int t = base + tid;
    long idx = (long)(unsigned)t * 48 + h;
    float b = (t < n) ? beta[idx] : 0.0f;
    float a = A[idx];
    lds[LDS_BETA + tid] = b;
    lds[LDS_EXPA + tid] = vexpf(a);
  }
  __syncthreads();

  // Phase B (544307-544459): 64x64 entries, entry e = tid + 256*k, row-in-LDS j = e >> 6, column i = e & 63.
  // For i < j: M[j][i] = ((K_i . K_j) * -beta_j) * exp(A_j - A_i), dot is a sequential fma chain over 128 floats (544389-544415).
  for (int e = tid; e < 4096; e += 256) {
    int i = e & 63, j = e >> 6;
    float v = 0.0f;
    if (i < j) {
      const float* Ki = K + ((long)(unsigned)(base + i) * 48 + h) * 128;
      const float* Kj = K + ((long)(unsigned)(base + j) * 48 + h) * 128;
      float acc = 0.0f;
      for (int k = 0; k < 128; k++) {
        float a = Kj[k], b = Ki[k];
        if (MP) {
          a = (float)(_Float16)(a * 16.0f) * 0.0625f;
          b = (float)(_Float16)(b * 16.0f) * 0.0625f;
        }
        acc = __builtin_fmaf(a, b, acc);
      }
      float x = A[(long)(unsigned)(base + j) * 48 + h] - A[(long)(unsigned)(base + i) * 48 + h];
      float ex = vexpf(x);
      acc = acc * -lds[LDS_BETA + j];
      v = acc * ex;
    }
    lds[j * 65 + i] = v;
  }
  __syncthreads();

  // Phase C (544461-544591): forward substitution, rows r = 1..63, thread = column.
  //   T[r][c] = M[r][c] + sum_{k=c+1}^{r-1} M[r][k] * T[k][c], ascending k, fma chain.
  for (int r = 1; r < 64; r++) {
    if (tid < r) lds[LDS_SCR + tid] = lds[r * 65 + tid];
    __syncthreads();
    if (tid < r) {
      float acc = lds[LDS_SCR + tid];
      for (int k = tid + 1; k < r; k++) acc = __builtin_fmaf(lds[LDS_SCR + k], lds[k * 65 + tid], acc);
      lds[r * 65 + tid] = acc;
    }
    __syncthreads();
  }

  // Phase D (544592-544911): thread = (lane = tid & 127, half = tid >> 7), 128 lanes x 64 rows. Loads of V are guarded by
  // token < n (544701), K is loaded unguarded, stores are unguarded.
  //   bv_r = V_r * beta_r ; kk_r = (K_r * beta_r) * exp(A_r)
  //   U_r = bv_r + sum_{j<r} T[r][j] * bv_j ; W_r = kk_r + sum_{j<r} T[r][j] * kk_j   (ascending j, fma chain, accumulator starts at the diagonal term)
  const int lane = tid & 127;
  const int half = tid >> 7;
  for (int r = half; r < 64; r += 2) {
    int tok = base + r;
    float Vr = (tok < n) ? bf2f(qkv[(long)tok * 10240 + 4096 + h * 128 + lane]) : 0.0f;
    float Kr = K[((long)(unsigned)tok * 48 + h) * 128 + lane];
    float br = lds[LDS_BETA + r], er = lds[LDS_EXPA + r];
    float u = Vr * br;
    float w = (Kr * br) * er;
    for (int j = 0; j < r; j++) {
      int tj = base + j;
      float Vj = (tj < n) ? bf2f(qkv[(long)tj * 10240 + 4096 + h * 128 + lane]) : 0.0f;
      float Kj = K[((long)(unsigned)tj * 48 + h) * 128 + lane];
      float bj = lds[LDS_BETA + j], ej = lds[LDS_EXPA + j];
      float bv = Vj * bj;
      float kj = (Kj * bj) * ej;
      float t = lds[r * 65 + j];
      if (MP) {
        // b3: bv, kk and T each pass through f16(x * 16) / 16 before the f32 fma (listing 539628+: 556-599, 632-633, 652-658, 686-751
        // of the kernel body; the diagonal terms u, w above are not rounded).
        bv = (float)(_Float16)(bv * 16.0f) * 0.0625f;
        kj = (float)(_Float16)(kj * 16.0f) * 0.0625f;
        t = (float)(_Float16)(t * 16.0f) * 0.0625f;
      }
      u = __builtin_fmaf(bv, t, u);
      w = __builtin_fmaf(kj, t, w);
    }
    long o = ((long)(unsigned)tok * 48 + h) * 128 + lane;
    U[o] = u;
    W[o] = w;
  }
}

// <N,M,b1,b2,b3> with b2 = b3 = false: <4,1>, <4,4>, <8,1,F,F,F>, <8,1,T,F,F>. N, M and b1 only change the vendor's schedule
// (row grouping, full-tile fast path); the results are bitwise equal to this kernel.
extern "C" __global__ __launch_bounds__(256) void dnc_ut4(const unsigned short* qkv, const float* beta, const float* K,
                                                          const float* A, float* U, float* W, int n) {
  ut4_body<false, false>(qkv, beta, K, A, U, W, n);
}
// <8,1,T,T,F>
extern "C" __global__ __launch_bounds__(256) void dnc_ut4_sw(const unsigned short* qkv, const float* beta, const float* K,
                                                             const float* A, float* U, float* W, int n) {
  ut4_body<true, false>(qkv, beta, K, A, U, W, n);
}
// <8,1,T,T,T>
extern "C" __global__ __launch_bounds__(256) void dnc_ut4_sw_mp(const unsigned short* qkv, const float* beta, const float* K,
                                                                const float* A, float* U, float* W, int n) {
  ut4_body<true, true>(qkv, beta, K, A, U, W, n);
}
