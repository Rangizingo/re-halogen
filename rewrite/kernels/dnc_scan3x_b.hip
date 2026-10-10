// Rewrite of the dnc_scan3x family (k_dnc_scan3x, codeobj_1146000, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dnc_scan3x.co dnc_scan3x.hip
// Listing: kernels-disasm/codeobj_1146000.s (not in git). Every cite below is codeobj_1146000.s:<line> in the
// default-path instance _ZN7halogen12_GLOBAL__N_112k_dnc_scan3xILi2ELi4ELb1ELb0ELb1ELi1ELi0ELi1ELb0EEE... (listing
// lines 230448-239033); the other instances differ only in the template-argument effects documented in
// rewrite/reports/g09.md.
//
// What the kernel does (gated delta rule, one 64-token chunk per loop trip, state kept in LDS):
//   workgroup = (head h = blockIdx.x of 48, column slab y = blockIdx.y of A), 128 threads = 4 wave32.
//   S^T[c][d] f32 (64 columns of the 128x128 state of head h) lives in LDS for the whole call.
//   per chunk:
//     E1[t] = exp(G - g[t]), E2[t] = exp(g[t]), G = g[63]                                   (230716-230764)
//     Kt[d][t] = f16(16 * ((bf16 K[t][d] * kinv[t]) * E1[t]))                               (231675-232268)
//     X1 = mir1_rows . f16(16 S)       (WMMA chain over 8 d-blocks)                          (232398-233118)
//     U[t][c] = V[t][c] - X1 * 2^-8 ;  U16 = f16(16 U)  (masked to the valid tokens)         (233115-233364)
//     X2 = mir0_rows . f16(16 S)                                                             (233533-234295)
//     intra = A_mat rows . U16         (WMMA chain over the j blocks 0..wave)                (234288-235038)
//     o[t][c] = bf16( fma(X2, E2[t], intra) * 2^-8 )  -> sel                                 (234331-234506)
//     S = fma(2^-8, Kt^T . U16, exp(G) * S)  (WMMA chain over 4 token blocks)                (234628-234986)
//     NormFold (i3 = 1): the second workgroup to finish the chunk (n3 flag) computes n2      (234620-239022)
//     Tap: if chunk == slot[k] dump the slab of S to p[k]                                    (236167-236238)
//   at the end S is written back to `out`.                                                   (238979-239027)
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)

typedef unsigned short u16;
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));

// bf16 -> f32: v_lshlrev_b32 16 (231669)
__device__ static inline float bf2f(u16 b) { return __uint_as_float((unsigned)b << 16); }
// f32 -> bf16 RNE without NaN special case: v_bfe_u32 x,16,1 ; v_add3_u32 x,x,0x7fff, store high half (234349-234354)
__device__ static inline u16 f2bf(float f) {
  unsigned u = __float_as_uint(f);
  return (u16)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
// v_fma_mixlo_f16 d, x, 16.0, 0.0 : fma in f32 then round to f16 (231675)
__device__ static inline _Float16 to_h16x16(float x) { return (_Float16)__builtin_fmaf(x, 16.0f, 0.0f); }

struct DncTapArgs { int slot[3]; int pad; float* p[3]; };                                  // 40 B, kernarg +0x80
struct DncNormFold { const u16* n0; const u16* n1; u16* n2; unsigned* n3; int w; int pad; };  // 40 B, kernarg +0xa8

template <int A>
struct Lds {
  static constexpr int NC = 128 / A;   // columns of the state slab
  float St[NC][132];                   // S^T[c][d], pitch 0x210 B in the vendor (230466)
  _Float16 Kt[128][72];                // Kt[d][t], pitch 0x90 B (231943)
  _Float16 Ut[NC][72];                 // U^T[c][t], pitch 0x90 B (233278)
  float E1[64];
  float E2[64];
  unsigned flag;
};

// Default path (A = 2, b1 = T, b2 = F, b3 = T, i1 = 1, i2 = 0, i3 = 1, b4 = F).
template <int A, bool B3, int I3>
__device__ static inline void scan3x_body(
    const u16* qkv, const float* kf, const float* g, const float* vv, float* out, const int* idx, long n, u16* sel, int k,
    const u16* m0, const u16* m1, const u16* m2, const float* kinv, DncTapArgs tap, DncNormFold nf) {
  constexpr int NC = 128 / A;     // slab width
  constexpr int NG = NC / 16;     // 16-column groups
  __shared__ Lds<A> L;
  const int head = blockIdx.x, y = blockIdx.y, kh = head / 3;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
  const int col0 = y * NC;
  float* outp = (float*)((char*)out + (long)idx[0] * 0x9000000L + n * 4) + (long)head * 16384 + col0;

  // state load: thread idx -> (r = idx>>6, c = idx&63), S^T[c][r]   (230489-230506)
  for (int it = 0; it < NC * 128 / 128; ++it) {
    int i2 = tid + it * 128, r = i2 / NC, c = i2 % NC;
    L.St[c][r] = outp[(long)r * 128 + c];
  }
  __syncthreads();

  const int nch = (k + 63) >> 6;
  for (int ch = 0; ch < nch; ++ch) {
    const int base = ch * 64;
    const int valid = min(64, k - base);
    // decay arrays (tid < 64)                                              (230698-230764)
    const float Gl = g[(long)(base + 63) * 48 + head];
    if (tid < 64) {
      float gt = g[(long)(base + tid) * 48 + head];
      L.E1[tid] = expf(Gl - gt);
      L.E2[tid] = expf(gt);
    }
    __syncthreads();
    // Kt[d][t]: thread = d (128 threads)                                     (231675-232268)
    {
      const int d = tid;
      for (int t = 0; t < 64; ++t) {
        int tok = min(base + t, k - 1);
        float x;
        if (B3) {  // b3 = T: bf16 K from qkv times kinv (231669-231675)
          float kk = bf2f(qkv[(long)tok * 10240 + 2048 + kh * 128 + d]);
          float ki = kinv[(long)tok * 48 + head];
          x = (kk * ki) * L.E1[t];
        } else {   // b3 = F: f32 K per head from kernarg +0x10, no kinv (listing of ILi2ELi4ELb1ELb0ELb0ELi1ELi0ELi0ELb0: 259783)
          x = kf[((long)tok * 48 + head) * 128 + d] * L.E1[t];
        }
        L.Kt[d][t] = (t < valid) ? to_h16x16(x) : (_Float16)0.0f;
      }
    }
    // X1 = mir1 rows . f16(16 S)
    const int trow = base + 16 * w + l16;
    f8 acc1[NG];
    {
      const u16* rp = m1 + ((long)min(trow, k - 1) * 48 + head) * 128;
      for (int gI = 0; gI < NG; ++gI) acc1[gI] = (f8){0, 0, 0, 0, 0, 0, 0, 0};
      for (int i = 0; i < 8; ++i) {
        h16 a;
        for (int e = 0; e < 16; ++e) a[e] = (trow < k) ? ((const _Float16*)rp)[16 * i + e] : (_Float16)0.0f;
        for (int gI = 0; gI < NG; ++gI) {
          h16 b;
          for (int e = 0; e < 16; ++e) b[e] = to_h16x16(L.St[16 * gI + l16][16 * i + e]);
          acc1[gI] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc1[gI]);
        }
      }
    }
    // U = V - X1 * 2^-8 ; U16 = f16(16 U), transposed into Ut[c][t], zero for t >= valid     (233115-233364)
    for (int gI = 0; gI < NG; ++gI)
      for (int r = 0; r < 8; ++r) {
        int t = 16 * w + 2 * r + hi, cc = 16 * gI + l16;
        _Float16 u16v = (_Float16)0.0f;
        if (t < valid) {
          int tok = min(base + t, k - 1);
          float V = vv[((long)tok * 48 + head) * 128 + col0 + cc];
          float U = __builtin_fmaf(-0.00390625f, acc1[gI][r], V);
          u16v = to_h16x16(U);
        }
        L.Ut[cc][t] = u16v;
      }
    __syncthreads();
    // X2 = mir0 rows . f16(16 S)                                                       (233533-234295)
    f8 acc2[NG], acc3[NG];
    {
      const u16* rp = m0 + ((long)min(trow, k - 1) * 48 + head) * 128;
      for (int gI = 0; gI < NG; ++gI) acc2[gI] = (f8){0, 0, 0, 0, 0, 0, 0, 0};
      for (int i = 0; i < 8; ++i) {
        h16 a;
        for (int e = 0; e < 16; ++e) a[e] = (trow < k) ? ((const _Float16*)rp)[16 * i + e] : (_Float16)0.0f;
        for (int gI = 0; gI < NG; ++gI) {
          h16 b;
          for (int e = 0; e < 16; ++e) b[e] = to_h16x16(L.St[16 * gI + l16][16 * i + e]);
          acc2[gI] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc2[gI]);
        }
      }
    }
    // intra = A_mat rows . U16, j blocks 0..w                                           (234288-235038)
    {
      const _Float16* rp = (const _Float16*)m2 + (((long)ch * 48 + head) * 64 + 16 * w + l16) * 64;
      for (int gI = 0; gI < NG; ++gI) acc3[gI] = (f8){0, 0, 0, 0, 0, 0, 0, 0};
      for (int j = 0; j <= w; ++j) {
        h16 a;
        for (int e = 0; e < 16; ++e) a[e] = rp[16 * j + e];
        for (int gI = 0; gI < NG; ++gI) {
          h16 b;
          for (int e = 0; e < 16; ++e) b[e] = L.Ut[16 * gI + l16][16 * j + e];
          acc3[gI] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc3[gI]);
        }
      }
    }
    // o -> sel                                                                          (234331-234506)
    for (int gI = 0; gI < NG; ++gI)
      for (int r = 0; r < 8; ++r) {
        int t = 16 * w + 2 * r + hi, cc = 16 * gI + l16;
        if (base + t < k) {
          float v = __builtin_fmaf(acc2[gI][r], L.E2[t], acc3[gI][r]) * 0.00390625f;
          sel[(long)(base + t) * 6144 + head * 128 + col0 + cc] = f2bf(v);
        }
      }
    __syncthreads();   // 234618
    // state update                                                                      (234628-234986)
    {
      const float El = expf(Gl);
      const int cc = 16 * w + l16;
      for (int i = 0; i < 8; ++i) {
        f8 acc = (f8){0, 0, 0, 0, 0, 0, 0, 0};
        for (int tb = 0; tb < 4; ++tb) {
          h16 a, b;
          for (int e = 0; e < 16; ++e) {
            a[e] = L.Kt[16 * i + l16][16 * tb + e];
            b[e] = L.Ut[cc][16 * tb + e];
          }
          acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
        }
        for (int r = 0; r < 8; ++r) {
          int d = 16 * i + 2 * r + hi;
          float old = L.St[cc][d];
          L.St[cc][d] = __builtin_fmaf(0.00390625f, acc[r], El * old);
        }
      }
    }
    __threadfence();
    __syncthreads();
    // NormFold: the second workgroup to arrive per (chunk, head) does the gated norm       (234620-239022)
    if (I3 != 0 && tid == 0) {
      L.flag = atomicInc(&nf.n3[ch * 48 + head], 1u);
    }
    __syncthreads();
    if (I3 == 1 && L.flag == 1u) {
      __threadfence();
      const float wf = (float)nf.w;
      for (int q = 0; q < 16; ++q) {
        int t = 16 * w + q, tok = base + t;
        if (tok < k) {
          long ro = (long)tok * 6144 + head * 128;
          float x[4], z[4], wt[4], p[32];
          for (int m = 0; m < 4; ++m) {
            x[m] = bf2f(sel[ro + lane + 32 * m]);
            z[m] = bf2f(nf.n1[ro + lane + 32 * m]);
            wt[m] = bf2f(nf.n0[lane + 32 * m]);
          }
          float sq0 = x[0] * x[0], sq1 = x[1] * x[1], sq2 = x[2] * x[2], sq3 = x[3] * x[3];
          float s = (sq0 + sq2) + (sq1 + sq3);
          for (int l = 0; l < 32; ++l) p[l] = __shfl(s, l, 32);
          // lane-0 butterfly: offsets 16, 8, 4, 2, 1                                      (235189-235222)
          for (int l = 0; l < 16; ++l) p[l] = p[l] + p[l + 16];
          for (int l = 0; l < 8; ++l) p[l] = p[l] + p[l + 8];
          for (int l = 0; l < 4; ++l) p[l] = p[l] + p[l + 4];
          for (int l = 0; l < 2; ++l) p[l] = p[l] + p[l + 2];
          float ss = p[0] + p[1];
          float ms = ss / wf + 1e-6f;
          float rs = rsqrtf(ms);
          for (int m = 0; m < 4; ++m) {
            float sg = 1.0f / (1.0f + __builtin_amdgcn_exp2f(z[m] * -1.44269502162933349609375f));
            float yv = sg * ((x[m] * rs) * wt[m]);
            nf.n2[ro + lane + 32 * m] = f2bf(yv);
          }
        }
      }
    }
    // taps: dump the slab of S after this chunk                                         (236167-236238)
    for (int kk = 0; kk < 3; ++kk) {
      if (ch == tap.slot[kk] && tap.p[kk] != nullptr) {
        float* tp = tap.p[kk] + (long)head * 16384 + col0;
        __syncthreads();
        for (int it = 0; it < NC * 128 / 128; ++it) {
          int i2 = tid + it * 128, r = i2 / NC, c = i2 % NC;
          tp[(long)r * 128 + c] = L.St[c][r];
        }
      }
    }
    __syncthreads();
  }
  // write the state back                                                                (238979-239027)
  for (int it = 0; it < NC * 128 / 128; ++it) {
    int i2 = tid + it * 128, r = i2 / NC, c = i2 % NC;
    outp[(long)r * 128 + c] = L.St[c][r];
  }
}

#define S3X(NAME, B3, I3)                                                                                   \
  extern "C" __global__ __launch_bounds__(128) void NAME(                                                    \
      const u16* qkv, const float* a1, const float* kf, const float* g, const float* vv, const float* a5,   \
      const float* a6, float* out, const int* idx, long n, u16* sel, int k, const u16* m0, const u16* m1,   \
      const u16* m2, const float* kinv, DncTapArgs tap, DncNormFold nf) {                                   \
    scan3x_body<2, B3, I3>(qkv, kf, g, vv, out, idx, n, sel, k, m0, m1, m2, kinv, tap, nf);                  \
  }
// i1 (0/1/2) and b4 change only code shape in the emulated listings (same sel/state/n2), see report.
S3X(k_dnc_scan3x_def, true, 1)     // <2,4,T,F,T,1,0,1,F>
S3X(k_s3x_b1_i30, true, 0)         // b3 = T, i3 = 0 (also b4 = T, i1 = 0/1)
S3X(k_s3x_b0_i30, false, 0)        // b3 = F, i3 = 0 (i1 = 0/1/2)
S3X(k_s3x_b1_i32, true, 2)         // i3 = 2: n3 flag only
