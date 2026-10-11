// Rewrite of the "dnc_ut5" family (halogen::(anonymous namespace)::k_dnc_ut5<T, norm, rs, b3, b4, b5>) from codeobj_1146000
// (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dnc_ut5.co dnc_ut5.hip
// Listing: kernels-disasm/codeobj_1146000.s (not in git). Cites below are listing line numbers of k_dnc_ut5<true,false,0,false,false,false>
// (lines 7821-11206) unless another instance is named.
//
// Signature: (u16 const* qkv, f32 const* beta, f32 const* K, f32 const* A, f32* U, f32* W, int n, u16* Wh,
//             f32 const* a8, u16 const* a9, u16* a10, f32* a11)
// kernarg: qkv@0 beta@8 K@16 A@24 U@32 W@40 n@48 Wh@56 a8@64 a9@72 a10@80 a11@88.
// Grid (48, ceil(n/64)) = (head, chunk), block 256 (8 waves of 32), static LDS 19456 B for rs = 0.
// This is the WMMA form of dnc_ut4 (see dnc_ut4.hip): same math, but Gram matrix and the T * RHS products go through
// v_wmma_f32_16x16x16_f16 with f16 operands scaled by 16 (the 2^-8 factors undo the scale).
//
// LDS (bytes): K16 [64][144 halves] at 0 (row pitch 0x120) ; later the f32 matrix M/T [64][68 floats] (pitch 0x110) overlays it;
//              beta[64] at 18432, exp(A)[64] at 18688, A[64] at 18944, scratch row [64] at 19200.
// Layouts as in dnc_ut4: qkv bf16 [token][10240], V at 4096 + h*128 ; K f32 [token][48][128] ; beta, A f32 [token][48] ; U, W f32 [token][48][128].
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// expf as the vendor inlines it (7821+: 7841-7871 region of the listing, same sequence as dnc_ut4.hip).
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

// f16(16 * x): v_fma_mixlo_f16 d, x, 16.0, 0 (s_mov 0x41800000).
__device__ static inline _Float16 h16(float x) { return (_Float16)(x * 16.0f); }

#define M_PITCH 68
#define O_BETA (18432 / 4)
#define O_EXPA (18688 / 4)
#define O_A (18944 / 4)
#define O_SCR (19200 / 4)

// NORM (second template bool): K comes from the raw bf16 key columns of qkv (2048 + (h/3)*128 + col) times the per-token scale a8[token*48+h]
// (the q/k prep scale), zero for token >= n; otherwise from the f32 array K (unguarded in phase 4).
// B3: attention tile output (ATT_UT): S[i][j] = (Q16_i . K16_j * 2^-8) * exp(A_i - A_j) for j <= i, stored as f16(16*S) to a10 and (if a11 != null) as f32 to a11,
// one 64x64 matrix per (chunk, head), lower 16x16 tiles only. Q16 = a9 = f16(16*q) [token][48][128] (the prep qh output).
// KL: the K operand of phase 4 is the dequantised f16 copy f32(K16) * 2^-4 kept from phase 1 instead of K (b5).
template <bool NORM, int RS, bool B3, bool KL>
__device__ static inline void ut5_body(const unsigned short* qkv, const float* beta, const float* K, const float* A, float* U,
                                       float* W, int n, unsigned short* Wh, const float* a8, const unsigned short* a9,
                                       unsigned short* a10, float* a11) {
  __shared__ __attribute__((aligned(16))) unsigned char smem[19456];
  float* fl = (float*)smem;
  _Float16* k16 = (_Float16*)smem;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, ln = lane & 15, hl = lane >> 4;
  const int h = blockIdx.x;
  const int base = blockIdx.y * 64;
  auto getK = [&](int tok, int col) -> float {
    if (NORM) {
      if (tok >= n) return 0.0f;
      return bf2f(qkv[(long)tok * 10240 + 2048 + (h / 3) * 128 + col]) * a8[(long)tok * 48 + h];
    }
    return K[((long)(unsigned)tok * 48 + h) * 128 + col];
  };

  // Phase 0 (7821-7870): tid < 64. beta guarded by token < n, A unguarded.
  if (tid < 64) {
    int t = base + tid;
    float b = (t < n) ? beta[(long)(unsigned)t * 48 + h] : 0.0f;
    float a = A[(long)(unsigned)t * 48 + h];
    fl[O_BETA + tid] = b;
    fl[O_EXPA + tid] = vexpf(a);
    fl[O_A + tid] = a;
  }
  // Phase 1 (7893-7937): K16[row][col] = f16(16 * K) for token < n, else 0.
  for (int e = tid; e < 8192; e += 256) {
    int row = e >> 7, col = e & 127;
    int tok = base + row;
    _Float16 v = (_Float16)0.0f;
    if (tok < n) v = h16(getK(tok, col));
    k16[row * 144 + col] = v;
  }
  __syncthreads();

  // Phase 2: Gram tiles by WMMA. Tile (I,J), J <= I; wave w gets tile T1[w]; waves 0 and 1 also get (3,2) and (3,3).
  const int tI[8] = {0, 1, 1, 2, 2, 2, 3, 3}, tJ[8] = {0, 0, 1, 0, 1, 2, 0, 1};
  float8 g1, g2;
  {
    int I = tI[w], J = tJ[w];
    float8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int c = 0; c < 8; c++) {
      half16 a, b;
      __builtin_memcpy(&a, &k16[(16 * I + ln) * 144 + 16 * c], 32);
      __builtin_memcpy(&b, &k16[(16 * J + ln) * 144 + 16 * c], 32);
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    }
    g1 = acc;
  }
  if (tid < 64) {
    int I = 3, J = 2 + w;
    float8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int c = 0; c < 8; c++) {
      half16 a, b;
      __builtin_memcpy(&a, &k16[(16 * I + ln) * 144 + 16 * c], 32);
      __builtin_memcpy(&b, &k16[(16 * J + ln) * 144 + 16 * c], 32);
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    }
    g2 = acc;
  }
  // b3: attention scores of the same tiles with A = Q16 (global, 8 chained wmma over the 128 dims), B = K16 rows (F3100 dump 241-440).
  float8 s1 = {0, 0, 0, 0, 0, 0, 0, 0}, s2 = {0, 0, 0, 0, 0, 0, 0, 0};
  if (B3) {
    auto att = [&](int I, int J) -> float8 {
      float8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
      const unsigned short* qrow = a9 + ((long)(unsigned)(base + 16 * I + ln) * 48 + h) * 128;
      for (int c = 0; c < 8; c++) {
        half16 a, b;
        __builtin_memcpy(&a, qrow + 16 * c, 32);
        __builtin_memcpy(&b, &k16[(16 * J + ln) * 144 + 16 * c], 32);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
      }
      return acc;
    };
    s1 = att(tI[w], tJ[w]);
    if (tid < 64) s2 = att(3, 2 + w);
  }
  __syncthreads();
  // KL: keep this lane's K16 column (64 tokens) in registers; the matrix M overwrites K16 below.
  float kq16[64];
  if (KL) {
    for (int t = 0; t < 64; t++) kq16[t] = (float)k16[t * 144 + 16 * w + ln];
    __syncthreads();
  }

  // M[row][col] = ((G * -2^-8) * beta[row]) * exp(A[row] - A[col]) for col < row, else 0 (7960-8200: mul 0xbb800000, ds beta, v_sub, expf, mul).
  auto fill = [&](int I, int J, float8 g) {
    for (int r = 0; r < 8; r++) {
      int row = 16 * I + 2 * r + hl, col = 16 * J + ln;
      float m = 0.0f;
      if (col < row) {
        float x = g[r] * __uint_as_float(0xbb800000);
        x = x * fl[O_BETA + row];
        float ex = vexpf(fl[O_A + row] - fl[O_A + col]);
        m = x * ex;
      }
      fl[row * M_PITCH + col] = m;
    }
  };
  fill(tI[w], tJ[w], g1);
  if (tid < 64) fill(3, 2 + w, g2);
  if (B3) {
    // Stores are unconditional per element (0 where col > row inside a tile); a11 only if non-null (F3100 dump 829-1373).
    auto attstore = [&](int I, int J, float8 s) {
      const long cidx = (long)blockIdx.y * 48 + h;
      for (int r = 0; r < 8; r++) {
        int row = 16 * I + 2 * r + hl, col = 16 * J + ln;
        float v = 0.0f;
        if (col <= row) {
          float x = s[r] * __uint_as_float(0x3b800000);
          float ex = vexpf(fl[O_A + row] - fl[O_A + col]);
          v = x * ex;
        }
        a10[cidx * 4096 + row * 64 + col] = __builtin_bit_cast(unsigned short, h16(v));
        if (a11) a11[cidx * 4096 + row * 64 + col] = v;
      }
    };
    // Tiles above the diagonal are written as zeros (observed in the vendor output).
    for (int e = tid; e < 4096; e += 256) {
      int row = e >> 6, col = e & 63;
      if ((col >> 4) > (row >> 4)) {
        long o = ((long)blockIdx.y * 48 + h) * 4096 + e;
        a10[o] = 0;
        if (a11) a11[o] = 0.0f;
      }
    }
    attstore(tI[w], tJ[w], s1);
    if (tid < 64) attstore(3, 2 + w, s2);
  }
  // Zero the blocks above the diagonal (8310-8340): column block > row block.
  for (int e = tid; e < 4096; e += 256) {
    int row = (tid >> 6) + 4 * (e >> 8), col = tid & 63;
    if (((tid >> 4) & 3) > (e >> 10)) fl[row * M_PITCH + col] = 0.0f;
  }
  __syncthreads();

  // Phase 3 (8340-8460): forward substitution, same as dnc_ut4 with row pitch 68.
  // rs = 3 (k_dnc_ut5<true,false,3,...> listing 480391-486468): blocked solve with 16-row blocks a = 0..3. Same recurrence
  //   T = M + M T, but the chain for entries left of the diagonal block is split:
  //   A: diagonal block, in-block recurrence, ascending k (wave 0, lane = column; lines 700-1109 of the dump).
  //   B: for rows r of block a and columns c < 16a: acc = M[r][c]; for k = c+1 .. 16a-1: acc = fma(M[r][k], T[k][c], acc) (dump 1115-1905),
  //      stored to LDS after a barrier.
  //   C: T[r][c] = fma chain starting at B[r][c]: for j = 0 .. rho-1: acc = fma(Taa[rho][j], B[16a+j][c], acc) (dump 1943-2085), stored after a barrier.
  if (RS == 3) {
    for (int a = 0; a < 4; a++) {
      const int r0 = 16 * a;
      if (tid < 16) {
        float col[16];
        for (int rho = 1; rho < 16; rho++) {
          if (tid < rho) {
            float acc = fl[(r0 + rho) * M_PITCH + r0 + tid];
            for (int k = tid + 1; k < rho; k++) acc = __builtin_fmaf(fl[(r0 + rho) * M_PITCH + r0 + k], col[k], acc);
            col[rho] = acc;
          }
        }
        for (int rho = tid + 1; rho < 16; rho++) fl[(r0 + rho) * M_PITCH + r0 + tid] = col[rho];
      }
      __syncthreads();
      if (a > 0) {
        const int nc = 16 * a;
        float bacc[3];
        for (int i = 0; i < a; i++) {
          int idx = tid + 256 * i, rho = idx / nc, c = idx - rho * nc;
          float acc = fl[(r0 + rho) * M_PITCH + c];
          for (int k = c + 1; k < nc; k++) acc = __builtin_fmaf(fl[(r0 + rho) * M_PITCH + k], fl[k * M_PITCH + c], acc);
          bacc[i] = acc;
        }
        __syncthreads();
        for (int i = 0; i < a; i++) {
          int idx = tid + 256 * i, rho = idx / nc, c = idx - rho * nc;
          fl[(r0 + rho) * M_PITCH + c] = bacc[i];
        }
        __syncthreads();
        for (int i = 0; i < a; i++) {
          int idx = tid + 256 * i, rho = idx / nc, c = idx - rho * nc;
          float acc = bacc[i];
          for (int j = 0; j < rho; j++) acc = __builtin_fmaf(fl[(r0 + rho) * M_PITCH + r0 + j], fl[(r0 + j) * M_PITCH + c], acc);
          bacc[i] = acc;
        }
        __syncthreads();
        for (int i = 0; i < a; i++) {
          int idx = tid + 256 * i, rho = idx / nc, c = idx - rho * nc;
          fl[(r0 + rho) * M_PITCH + c] = bacc[i];
        }
        __syncthreads();
      }
    }
  }
  // rs = 2: the instruction stream has no substitution at all (listing 436739-440007 vs 7821-11206): T is the raw M matrix.
  // rs = 0 and 1 (rs = 1 is a fully unrolled schedule of the same chain, bitwise equal): plain forward substitution.
  for (int r = 1; (RS == 0 || RS == 1) && r < 64; r++) {
    if (tid < r) fl[O_SCR + tid] = fl[r * M_PITCH + tid];
    __syncthreads();
    if (tid < r) {
      float acc = fl[O_SCR + tid];
      for (int k = tid + 1; k < r; k++) acc = __builtin_fmaf(fl[O_SCR + k], fl[k * M_PITCH + tid], acc);
      fl[r * M_PITCH + tid] = acc;
    }
    __syncthreads();
  }

  // Phase 4 (8460-9090): per wave a 16-column tile c = 16*w + ln. Accumulate over source block b = 0..3, target block a = b..3:
  //   accU[a] = wmma(f16(16*T[a][b]), f16(16*bv[b]), accU[a]);  accW[a] likewise with kk.   bv = V*beta, kk = (K*beta)*exp(A).
  const int c = 16 * w + ln;
  float8 accU[4], accW[4];
  for (int a = 0; a < 4; a++)
    for (int i = 0; i < 8; i++) accU[a][i] = accW[a][i] = 0.0f;
  for (int b = 0; b < 4; b++) {
    half16 rv, rk;
    for (int k = 0; k < 16; k++) {
      int tl = 16 * b + k, tok = base + tl;
      float V = (tok < n) ? bf2f(qkv[(long)tok * 10240 + 4096 + h * 128 + c]) : 0.0f;
      float bt = fl[O_BETA + tl], et = fl[O_EXPA + tl];
      float bv = V * bt;
      rv[k] = h16(bv);
      if (KL) {
        // F3011 dump 4037-4050: fma_mix f32 (f16 K16 * beta), then v_fma_mixlo_f16 (that * exp(A)) -> f16; K16 already carries the factor 16.
        float p = kq16[tl] * bt;
        rk[k] = (_Float16)(p * et);
      } else {
        float Kv = getK(tok, c);
        float kk = (Kv * bt) * et;
        rk[k] = h16(kk);
      }
    }
    for (int a = b; a < 4; a++) {
      half16 at;
      for (int k = 0; k < 16; k++) at[k] = h16(fl[(16 * a + ln) * M_PITCH + 16 * b + k]);
      accU[a] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(at, rv, accU[a]);
      accW[a] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(at, rk, accW[a]);
    }
  }

  // Epilogue (9090-11200): lane holds rows 2r + hl of each 16-row block a.
  //   U = fma(acc, 2^-8, V * beta) ; W = fma(acc, 2^-8, (K * beta) * exp(A)) ; stores unguarded; W and Wh = f16(16 * W) only if non-null.
  for (int a = 0; a < 4; a++)
    for (int r = 0; r < 8; r++) {
      int tl = 16 * a + 2 * r + hl, tok = base + tl;
      float V = (tok < n) ? bf2f(qkv[(long)tok * 10240 + 4096 + h * 128 + c]) : 0.0f;
      float Kv = KL ? kq16[tl] * 0.0625f : getK(tok, c);  // KL: f32(f16(16K)) * 1/16 (F3011 dump 5855: fma_mix with 0x3d800000)
      float bt = fl[O_BETA + tl], et = fl[O_EXPA + tl];
      float u = V * bt;
      float wv = (Kv * bt) * et;
      u = __builtin_fmaf(accU[a][r], __uint_as_float(0x3b800000), u);
      wv = __builtin_fmaf(accW[a][r], __uint_as_float(0x3b800000), wv);
      long o = ((long)(unsigned)tok * 48 + h) * 128 + c;
      U[o] = u;
      if (W) W[o] = wv;
      if (Wh) Wh[o] = __builtin_bit_cast(unsigned short, h16(wv));
    }
}

// One extern "C" kernel per (norm, rs, b3, kl). The b4 (burst) template bool only changes the load schedule and is not modelled.
// Names: dnc_ut5_n<norm>_r<rs>_a<b3>_k<kl>.
#define UT5_KERNEL(NAME, NORM, RS, B3, KL)                                                                   \
  extern "C" __global__ __launch_bounds__(256) void NAME(                                                    \
      const unsigned short* qkv, const float* beta, const float* K, const float* A, float* U, float* W,    \
      int n, unsigned short* Wh, const float* a8, const unsigned short* a9, unsigned short* a10,           \
      float* a11) {                                                                                          \
    ut5_body<NORM, RS, B3, KL>(qkv, beta, K, A, U, W, n, Wh, a8, a9, a10, a11);                              \
  }
UT5_KERNEL(dnc_ut5_n0_r0_a0_k0, false, 0, false, false)
UT5_KERNEL(dnc_ut5_n1_r0_a0_k0, true, 0, false, false)
UT5_KERNEL(dnc_ut5_n0_r1_a0_k0, false, 1, false, false)
UT5_KERNEL(dnc_ut5_n1_r1_a0_k0, true, 1, false, false)
UT5_KERNEL(dnc_ut5_n0_r1_a1_k0, false, 1, true, false)
UT5_KERNEL(dnc_ut5_n1_r1_a1_k0, true, 1, true, false)
UT5_KERNEL(dnc_ut5_n0_r2_a0_k0, false, 2, false, false)
UT5_KERNEL(dnc_ut5_n1_r2_a0_k0, true, 2, false, false)
UT5_KERNEL(dnc_ut5_n0_r3_a0_k0, false, 3, false, false)
UT5_KERNEL(dnc_ut5_n1_r3_a0_k0, true, 3, false, false)
UT5_KERNEL(dnc_ut5_n0_r3_a0_k1, false, 3, false, true)
UT5_KERNEL(dnc_ut5_n1_r3_a0_k1, true, 3, false, true)
UT5_KERNEL(dnc_ut5_n0_r3_a1_k0, false, 3, true, false)
UT5_KERNEL(dnc_ut5_n1_r3_a1_k0, true, 3, true, false)
UT5_KERNEL(dnc_ut5_n0_r3_a1_k1, false, 3, true, true)
UT5_KERNEL(dnc_ut5_n1_r3_a1_k1, true, 3, true, true)
