// Rewrite of the "dn_fused16" family from codeobj_14aa000 (flash_serve 0.17.3, gfx1151):
// k_dn_fused16<P,b1,b2,b3>(rows, g, beta, state, idx, n, out, nrows, Py, DnfTaps, DnfNorm).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dn_fused16.co dn_fused16.hip
// Listing: ROCm llvm-objdump --mcpu=gfx1151 of work/fatbin/codeobj_14aa000.hsaco, no .long defects. Cites "@20xxx" are code-object
// addresses inside k_dn_fused16<0,false,false,false> (starts @20300).
//
// What the kernel is: a gated delta rule, 16 rows per group, one workgroup (256 threads = 8 waves) per head h (grid 48),
// key head kh = h/3. Rows are bf16 [row][10240] = q(2048) | k(2048) | v(6144); q,k at kh*128, v at h*128. Output bf16 [row][6144].
// The f32 state S[128 key][128 value] of head h lives in registers as S*256: lane (wave w, lr = lane&15, lh = lane>>4) holds
// column c = 16w+lr and rows lh+2m, m = 0..63 (@20314-@2086C loads, mul 256.0 @20954-@20ADC; stores x 1/256 @25464-@258B0).
// Per group (rows r0..r0+15, rows >= nrows read as zero):
//  1. q,k sums of squares, tree ((x0^2+x1^2)+(x2^2+x3^2))+((x4^2+x5^2)+(x6^2+x7^2)) per 8 bf16, xor-add over 16 lanes
//     (@20F90-@21118). +1e-6, guarded rsq (x < 2^-126: x*2^24, rsq, *2^12). ks = rsq_k; qs = rsq(128) * rsq_q (@2112C-@211C4).
//  2. gcum = sequential f32 prefix sum of g (first = g0 + 0.0), wave 0 (@21268-@213AC). eg = expf(gcum), ed = expf(gcum15-gcum),
//     bk = 16 * (eg * (ks*beta)) (@21510-@2161C).
//  3. bf16 WMMA Gram over 8 dim blocks. Q.K -> Hq[i][j] (j<=i) = f16(16*(ex*(ks_j*(acc*qs_i)))), ex = expf(gcum_i-gcum_j) (@218A8-@21E24).
//     K.K -> L[i][j] (j<i) = (((acc*ks_i)*ks_j)*beta_i) * -ex, else 0 (@21E34-@223E0).
//  4. T = (I-L)^-1 rows by forward substitution, lane j: x_m = e_j[m]; for i=15..1: x_m = fma(x_i, L[i][m], x_m), m<i (@227E0-@22B4C).
//     Split y = coef[m]*x_m into f16 hi = f16(fma(coef,x,0)) and lo = f16(y - hi) for coef = bk and beta (@22B44-@22F04).
//  5. Kd16[d][r] = f16(16*(ed_r*(ks_r*K[r][d]))), Q16[r][d] = f16(16*(qs_r*Q[r][d])) (@2243x-@226B0).
//  6. per wave column block: M[j][d] = f16(WMMA(BL,Kcol)+WMMA(BH,Kcol)) (@230D0-@23428), P2 = WMMA(AL,V)+WMMA(AH,V) f32 (@23414-@23420).
//  7. MS = sum_c WMMA(M[:,16c..], S16_c), QS = sum_c WMMA(Q16[:,16c..], S16_c), S16 = f16(S256 * 1/16) (@23680-@23CC0).
//     vnew16 = f16(fma(P2 - MS*(1/256), 16, 0)), zero for rows >= nrows (@23CD8-@23EAC).
//  8. out = bf16RNE((WMMA(Hq, vnew16) + QS*eg) * (1/256)) (@23EE0-@2423C), plain mul then add.
//  9. S256 = fma(S256, expf(gcum15), WMMA(Kd16 block, vnew16)) (@24318-@24558). Taps: if idx[k] == rows done, store S256/256 to p[k].
#pragma clang fp contract(off)
#include <hip/hip_runtime.h>

typedef _Float16 f16;
typedef f16 v16h __attribute__((ext_vector_type(16)));
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

struct DnfTaps { int idx[3]; int pad; const float* p[3]; };   // 40 B at kernarg 72
struct DnfNorm { const void* p[3]; };                         // 24 B at kernarg 112, unused

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float(((unsigned)b) << 16); }
// v_fma_mixlo_f16 d, x, y, 0 : f32 fma then RNE to f16
__device__ static inline f16 fm16(float x, float y) { return (f16)__builtin_fmaf(x, y, 0.0f); }
__device__ static inline unsigned short f2bf_rne(float f) {   // @23F24-@23F48: bfe 16,1 ; add3 0x7fff
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
__device__ static inline v8f wmma16(v16h a, v16h b, v8f c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c); }
__device__ static inline float rsq_guard(float x) {           // @21138-@211A8
  bool small = x < 1.17549435e-38f;
  float xs = small ? x * 16777216.0f : x;
  float r = __builtin_amdgcn_rsqf(xs);
  return small ? r * 4096.0f : r;
}

template <bool B1, bool B2, bool B3>
__device__ static inline void dnf_body(const unsigned short* rows, const float* gp, const float* bp, float* st, const int* idx, long n,
                                       unsigned short* out, int nrows, DnfTaps taps, DnfNorm norm) {
  __shared__ unsigned short sOut[16 * 136];
  __shared__ unsigned short sK[16 * 136], sQb[16 * 136];  // raw bf16 k and q rows
  __shared__ f16 sQ16[16 * 136];                           // Q16
  __shared__ f16 sM[16 * 136];                             // M[j][d]
  __shared__ f16 sKd[128 * 16];                            // Kd16[d][r]
  __shared__ f16 sHq[16 * 16];
  __shared__ f16 sBH[16 * 16], sBL[16 * 16], sAH[16 * 16], sAL[16 * 16];
  __shared__ float sL[16 * 16];
  __shared__ float sBeta[16], sG[16], sEd[16], sBk[16], sEg[16], sKs[16], sQs[16];

  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, lr = lane & 15, lh = lane >> 4;
  const int h = blockIdx.x, kh = h / 3;
  const int col = wave * 16 + lr;
  const long sbase = n + (long)idx[0] * 37748736L + (long)h * 16384;

  float s[64];
#pragma unroll
  for (int m = 0; m < 64; m++) s[m] = 256.0f * st[sbase + (long)(lh + 2 * m) * 128 + col];

  float c128 = 128.0f;
  asm volatile("" : "+v"(c128));
  const float rsq128 = __builtin_amdgcn_rsqf(c128);

  const int ngroups = (nrows + 15) >> 4;
  for (int grp = 0; grp < ngroups; grp++) {
    const int r0 = grp * 16;
    __syncthreads();
    // 1. load q,k rows; sums of squares
    {
      const int row = tid >> 4, c8 = tid & 15;
      uint4 qv = make_uint4(0, 0, 0, 0), kv = make_uint4(0, 0, 0, 0);
      if (r0 + row < nrows) {
        const unsigned short* base = rows + (long)(r0 + row) * 10240;
        qv = *(const uint4*)(base + kh * 128 + c8 * 8);
        kv = *(const uint4*)(base + 2048 + kh * 128 + c8 * 8);
      }
      *(uint4*)(&sQb[row * 136 + c8 * 8]) = qv;
      *(uint4*)(&sK[row * 136 + c8 * 8]) = kv;
      float qq[8], kk[8];
      const unsigned qw[4] = {qv.x, qv.y, qv.z, qv.w}, kw[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
      for (int i = 0; i < 4; i++) {
        qq[2 * i] = __uint_as_float(qw[i] << 16); qq[2 * i + 1] = __uint_as_float(qw[i] & 0xffff0000u);
        kk[2 * i] = __uint_as_float(kw[i] << 16); kk[2 * i + 1] = __uint_as_float(kw[i] & 0xffff0000u);
      }
      float qs2[8], ks2[8];
#pragma unroll
      for (int i = 0; i < 8; i++) { qs2[i] = qq[i] * qq[i]; ks2[i] = kk[i] * kk[i]; }
      float sq = ((qs2[0] + qs2[1]) + (qs2[2] + qs2[3])) + ((qs2[4] + qs2[5]) + (qs2[6] + qs2[7]));
      float sk = ((ks2[0] + ks2[1]) + (ks2[2] + ks2[3])) + ((ks2[4] + ks2[5]) + (ks2[6] + ks2[7]));
#pragma unroll
      for (int o = 1; o < 16; o <<= 1) { sq = sq + __shfl_xor(sq, o); sk = sk + __shfl_xor(sk, o); }
      if (c8 == 0) {
        sKs[row] = rsq_guard(sk + 1e-6f);
        sQs[row] = rsq128 * rsq_guard(sq + 1e-6f);
      }
    }
    if (tid < 16) {
      float gv = 0.0f, bv = 0.0f;
      if (r0 + tid < nrows) { gv = gp[(long)(r0 + tid) * 48 + h]; bv = bp[(long)(r0 + tid) * 48 + h]; }
      sBeta[tid] = bv;
      sG[tid] = gv;  // raw g, replaced by the prefix sum below
    }
    __syncthreads();
    if (tid == 0) {
      float c = sG[0] + 0.0f;
      float prev[16];
      prev[0] = c;
      for (int i = 1; i < 16; i++) { c = c + sG[i]; prev[i] = c; }
      for (int i = 0; i < 16; i++) sG[i] = prev[i];
    }
    __syncthreads();
    if (tid < 16) {
      const float gc = sG[tid];
      const float eg = expf(gc);
      sEg[tid] = eg;
      sEd[tid] = expf(sG[15] - gc);
      sBk[tid] = 16.0f * (eg * (sKs[tid] * sBeta[tid]));
    }
    __syncthreads();
    // 3. Gram. wave 0: K.K -> L ; wave 1: Q.K -> Hq
    if (B1) {
      // b1: f32 sequential sums, thread (i = tid>>4, j = tid&15) (@163A8-@16794), then L / Hq (@1679C-@16884)
      const int i = tid >> 4, j = tid & 15;
      float aL = 0.0f, aH = 0.0f;
      for (int d = 0; d < 128; d++) {
        float kd_i = sKs[i] * bf2f(sK[i * 136 + d]);
        float ta = sBeta[i] * kd_i;
        float u = sKs[j] * bf2f(sK[j * 136 + d]);
        aL = aL + u * ta;
        float tq = sQs[i] * bf2f(sQb[i * 136 + d]);
        aH = aH + tq * u;
      }
      float ex = expf(sG[i] - sG[j]);
      sL[i * 16 + j] = (j < i) ? aL * (-ex) : 0.0f;
      sHq[i * 16 + j] = (j <= i) ? fm16(aH * ex, 16.0f) : (f16)0.0f;
    } else if (wave < 2) {
      const unsigned short* X = (wave == 0) ? sK : sQb;
      v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
      for (int sblk = 0; sblk < 8; sblk++) {
        v16s a, b;
        for (int k = 0; k < 16; k++) { a[k] = (short)X[lr * 136 + 16 * sblk + k]; b[k] = (short)sK[lr * 136 + 16 * sblk + k]; }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
      }
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int i = 2 * e + lh, j = lr;
        if (wave == 1) {
          f16 v = (f16)0.0f;
          if (j <= i) {
            float ex = expf(sG[i] - sG[j]);
            float t = acc[e] * sQs[i];
            t = sKs[j] * t;
            v = fm16(ex * t, 16.0f);
          }
          sHq[i * 16 + j] = v;
        } else {
          float v = 0.0f;
          if (j < i) {
            float t = acc[e] * sKs[i];
            t = sKs[j] * t;
            t = sBeta[i] * t;
            float ex = expf(sG[i] - sG[j]);
            v = t * (-ex);
          }
          sL[i * 16 + j] = v;
        }
      }
    }
    __syncthreads();
    // 4. forward substitution (wave 0, lane j < 16) ; 5. Kd16 / Q16 (all other threads)
    if (wave == 0 && lane < 16) {
      float x[16];
#pragma unroll
      for (int m = 0; m < 16; m++) x[m] = (m == lane) ? 1.0f : 0.0f;
#pragma unroll
      for (int i = 15; i >= 1; i--)
#pragma unroll
        for (int m = 0; m < i; m++) x[m] = __builtin_fmaf(x[i], sL[i * 16 + m], x[m]);
#pragma unroll
      for (int m = 0; m < 16; m++) {
        float y = sBk[m] * x[m];
        f16 hi = fm16(sBk[m], x[m]);
        sBH[lane * 16 + m] = hi;
        sBL[lane * 16 + m] = (f16)__builtin_fmaf((float)hi, -1.0f, y);
        float y2 = sBeta[m] * x[m];
        f16 hi2 = fm16(sBeta[m], x[m]);
        sAH[lane * 16 + m] = hi2;
        sAL[lane * 16 + m] = (f16)__builtin_fmaf((float)hi2, -1.0f, y2);
      }
    } else if (wave != 0) {
      for (int e = tid - 32; e < 16 * 128; e += 224) {
        const int r = e >> 7, d = e & 127;
        float kd = sEd[r] * (sKs[r] * bf2f(sK[r * 136 + d]));
        sKd[d * 16 + r] = (f16)__builtin_fmaf(kd, 16.0f, 0.0f);
        float qv = sQs[r] * bf2f(sQb[r * 136 + d]);
        sQ16[r * 136 + d] = (f16)__builtin_fmaf(qv, 16.0f, 0.0f);
      }
    }
    __syncthreads();
    // 6. per-wave column block
    v8f p2;
    {
      v16h kc, vc;
      for (int m = 0; m < 16; m++) {
        kc[m] = (f16)bf2f(sK[m * 136 + col]);
        f16 vv = (f16)0.0f;
        if (r0 + m < nrows) vv = (f16)bf2f(rows[(long)(r0 + m) * 10240 + 4096 + h * 128 + col]);
        vc[m] = vv;
      }
      v16h a;
      v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
      for (int m = 0; m < 16; m++) a[m] = sBL[lr * 16 + m];
      acc = wmma16(a, kc, acc);
      for (int m = 0; m < 16; m++) a[m] = sBH[lr * 16 + m];
      acc = wmma16(a, kc, acc);
#pragma unroll
      for (int e = 0; e < 8; e++) sM[(2 * e + lh) * 136 + col] = (f16)acc[e];
      v8f z = {0, 0, 0, 0, 0, 0, 0, 0};
      for (int m = 0; m < 16; m++) a[m] = sAL[lr * 16 + m];
      p2 = wmma16(a, vc, z);
      for (int m = 0; m < 16; m++) a[m] = sAH[lr * 16 + m];
      p2 = wmma16(a, vc, p2);
    }
    __syncthreads();
    // 7. MS, QS
    v8f ms = {0, 0, 0, 0, 0, 0, 0, 0}, qs = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int c = 0; c < 8; c++) {
      v16h b, bl;
#pragma unroll
      for (int p = 0; p < 8; p++) {
        f16 own, ownl = (f16)0.0f;
        if (B2) {   // b2: S = S256*(1/256); hi = f16(16 S); lo = f16(fma(S,16,-hi)) (@1D15C-@1D310)
          float S = s[8 * c + p] * 0.00390625f;
          own = fm16(S, 16.0f);
          ownl = (f16)__builtin_fmaf(S, 16.0f, -(float)own);
        } else {
          own = fm16(s[8 * c + p], 0.0625f);
        }
        f16 oth = __builtin_bit_cast(f16, (unsigned short)__shfl_xor((int)__builtin_bit_cast(unsigned short, own), 16));
        b[2 * p + lh] = own;
        b[2 * p + 1 - lh] = oth;
        if (B2) {
          f16 othl = __builtin_bit_cast(f16, (unsigned short)__shfl_xor((int)__builtin_bit_cast(unsigned short, ownl), 16));
          bl[2 * p + lh] = ownl;
          bl[2 * p + 1 - lh] = othl;
        }
      }
      v16h am, aq;
      for (int k = 0; k < 16; k++) { am[k] = sM[lr * 136 + 16 * c + k]; aq[k] = sQ16[lr * 136 + 16 * c + k]; }
      if (B2) { ms = wmma16(am, bl, ms); qs = wmma16(aq, bl, qs); }  // lo first (@1D31C-@1D33C: B = v[58:65] is the residual)
      ms = wmma16(am, b, ms);
      qs = wmma16(aq, b, qs);
    }
    // vnew16 as B operand (rows natural)
    v16h vb;
    {
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int j = 2 * e + lh;
        float t = ms[e] * 0.00390625f;
        float un = p2[e] - t;
        f16 v = fm16(un, 16.0f);
        if (r0 + j >= nrows) v = (f16)0.0f;
        f16 oth = __builtin_bit_cast(f16, (unsigned short)__shfl_xor((int)__builtin_bit_cast(unsigned short, v), 16));
        vb[2 * e + lh] = v;
        vb[2 * e + 1 - lh] = oth;
      }
    }
    // 8. output
    {
      v16h ah;
      for (int k = 0; k < 16; k++) ah[k] = sHq[lr * 16 + k];
      v8f z = {0, 0, 0, 0, 0, 0, 0, 0};
      v8f oa = wmma16(ah, vb, z);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int j = 2 * e + lh;
        if (r0 + j < nrows) {
          float t = qs[e] * sEg[j];
          t = oa[e] + t;
          t = t * 0.00390625f;
          if (B3) sOut[j * 136 + col] = f2bf_rne(t);   // b3: o is kept in LDS (@CE20-@D0C8); global `out` is not written
          else out[(long)(r0 + j) * 6144 + h * 128 + col] = f2bf_rne(t);
        }
      }
    }
    if (B3) {
      // b3 gated RMS norm of this group's rows. The vendor pipelines it one group behind; per-row numerics are the same.
      __syncthreads();
      const unsigned short* nw = (const unsigned short*)norm.p[0];
      const unsigned short* nz = (const unsigned short*)norm.p[1];
      unsigned short* no = (unsigned short*)norm.p[2];
      for (int row = wave; row < 16; row += 8) {
        if (r0 + row >= nrows) continue;
        const unsigned short* orow = &sOut[row * 136];
        float x0 = bf2f(orow[lane]), x1 = bf2f(orow[lane + 32]), x2 = bf2f(orow[lane + 64]), x3 = bf2f(orow[lane + 96]);
        float P = (x0 * x0 + x2 * x2) + (x1 * x1 + x3 * x3);   // @AA3C-@AAE0
        float Q = P + __shfl_xor(P, 16);
        float q[16];
#pragma unroll
        for (int i = 0; i < 16; i++) q[i] = __shfl(Q, i);
        float A0 = q[8] + q[0], A1 = q[9] + q[1], A2 = q[10] + q[2], A3 = q[11] + q[3];
        float A4 = q[12] + q[4], A5 = q[13] + q[5], A6 = q[14] + q[6], A7 = q[15] + q[7];
        float D = ((A4 + A0) + (A6 + A2)) + ((A5 + A1) + (A7 + A3));   // dpp row_shl 8,4,2,1 (@AB4C-@ABB8)
        float m = D * 0.0078125f;
        m = m + 1e-6f;
        float r = rsq_guard(m);
        const unsigned long long wv = *(const unsigned long long*)(nw + 4 * lane);
        const unsigned long long zv = *(const unsigned long long*)(nz + (long)(r0 + row) * 6144 + h * 128 + 4 * lane);
        const unsigned long long ov = *(const unsigned long long*)(orow + 4 * lane);
        unsigned long long res = 0;
#pragma unroll
        for (int k = 0; k < 4; k++) {
          float z = bf2f((unsigned short)(zv >> (16 * k)));
          float w = bf2f((unsigned short)(wv >> (16 * k)));
          float o = bf2f((unsigned short)(ov >> (16 * k)));
          float e = __builtin_amdgcn_exp2f(z * -1.44269502f);
          float sig = 1.0f / (1.0f + e);
          float y = (r * o) * w;
          float v = sig * y;
          res |= (unsigned long long)f2bf_rne(v) << (16 * k);
        }
        *(unsigned long long*)(no + (long)(r0 + row) * 6144 + h * 128 + 4 * lane) = res;
      }
    }
    // 9. state update
    {
      const float G = expf(sG[15]);
#pragma unroll
      for (int t = 0; t < 8; t++) {
        v16h a;
        for (int k = 0; k < 16; k++) a[k] = sKd[(16 * t + lr) * 16 + k];
        v8f z = {0, 0, 0, 0, 0, 0, 0, 0};
        v8f acc = wmma16(a, vb, z);
#pragma unroll
        for (int e = 0; e < 8; e++) s[8 * t + e] = __builtin_fmaf(s[8 * t + e], G, acc[e]);
      }
    }
    // taps
    const unsigned done = (unsigned)(r0 + 16);
#pragma unroll
    for (int k = 0; k < 3; k++) {
      if ((unsigned)taps.idx[k] == done) {
        float* dst = (float*)taps.p[k] + (long)h * 16384;
#pragma unroll
        for (int m = 0; m < 64; m++) dst[(long)(lh + 2 * m) * 128 + col] = s[m] * 0.00390625f;
      }
    }
  }
#pragma unroll
  for (int m = 0; m < 64; m++) st[sbase + (long)(lh + 2 * m) * 128 + col] = s[m] * 0.00390625f;
}

#define DNF_ARGS const unsigned short* rows, const float* g, const float* beta, float* state, const int* idx, long n, \
                 unsigned short* out, int nrows, unsigned long long* Py, DnfTaps taps, DnfNorm norm

// k_dn_fused16<0,false,false,false> (P=0 production). Probe variants P>0 add realtime-clock reads and Py stores only.
extern "C" __global__ __launch_bounds__(256) void k_dnf_p0(DNF_ARGS) { dnf_body<false, false, false>(rows, g, beta, state, idx, n, out, nrows, taps, norm); }
// k_dn_fused16<0,true,false,false>: Gram by f32 sequential sums instead of bf16 WMMA
extern "C" __global__ __launch_bounds__(256) void k_dnf_b1(DNF_ARGS) { dnf_body<true, false, false>(rows, g, beta, state, idx, n, out, nrows, taps, norm); }
// k_dn_fused16<0,false,true,false>: state split into f16 hi + lo for the MS / QS products
extern "C" __global__ __launch_bounds__(256) void k_dnf_b2(DNF_ARGS) { dnf_body<false, true, false>(rows, g, beta, state, idx, n, out, nrows, taps, norm); }
// k_dn_fused16<0,true,true,false>: both
extern "C" __global__ __launch_bounds__(256) void k_dnf_b1b2(DNF_ARGS) { dnf_body<true, true, false>(rows, g, beta, state, idx, n, out, nrows, taps, norm); }
// k_dn_fused16<0,false,false,true>: o goes through the gated RMS norm (DnfNorm: weight, z, result); `out` is not written
extern "C" __global__ __launch_bounds__(256) void k_dnf_b3(DNF_ARGS) { dnf_body<false, false, true>(rows, g, beta, state, idx, n, out, nrows, taps, norm); }
