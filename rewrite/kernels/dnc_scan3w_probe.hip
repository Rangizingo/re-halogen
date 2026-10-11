// Rewrites of the timing-probe instances k_dnc_scan3w<2,4,true,false,P>, P = 1..8, from codeobj_1146000 (flash_serve 0.17.3, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dnc_scan3w_probe.co dnc_scan3w_probe.hip
// Base kernel (P = 0) and the meaning of every stage: rewrite/kernels/dnc_scan3w.hip (g19). This file repeats that body with one
// stage removed or altered per P. The vendor probes are the same code with a compile-time switch (HALOGEN_DN_PROBE, timing
// decomposition); each P was identified by diffing the instruction stream against P0. Listing: work/kernels/*scan3w*Li<P>E*.s
// (line numbers below are lines of those files; P0 = ...Li0E, start 0x170c00).
//
// Stages of the P0 chunk body:  (1) vnew = u - (1/256) WMMA(w, S)      [P0 lines 353-2020: 32 WMMA, then u fmac + ds store]
//                               (2) o = fma(oi, expf(g_t), oa)/256, oi = WMMA(q, S), oa = WMMA(amat, vnew)  [2128-3419]
//                               (3) S = fma(WMMA(k*e, vnew), 1/256, G*S) with a k*e f16 table in LDS at 0x2800  [3451-3640]
// Probe effects (all verified by MATCH rows in rewrite/tables/g19b.md):
//   P1  stage 1 WMMA chain and the w loads removed: vnew = f16(16*u) (p1 lines 232-770: global_load_b32 u, v_fma_mixlo/hi 16.0, ds store;
//       no v_fmac -1/256, 32 b128 loads fewer).
//   P2  stage 2 q*S chain and q loads removed: the output is fma(0, expf(g_t), oa)/256 (p2 line 2218 "v_fmac_f32 v25, 0, v3"), so
//       expf(g_t) = inf gives NaN.
//   P3  stage 2 amat chain (4 WMMA) and amat loads removed: o = fma(oi, expf(g_t), 0)/256 (p3 line 3334 "v_fma_f32 v3, v25, v3, 0").
//   P4  stage 3 WMMA chain and the k*e table removed: S = fma(G, S, 0) (p4 lines 167-177, "v_fma_f32 v9, v17, v9, 0"; the -0 -> +0 matters).
//   P5  stage 3 kept but the k*e table is never stored (p5 has no ds_store_b16 offset:10240); the WMMA reads the table region
//       0x2800.. of LDS as it was left. See report for what the hardware returns.
//   P6  stage 3 WMMA chain and table removed, decay kept: S = G * S with a plain multiply (p6 lines 3418-3421, "v_mul_f32 v12, v9, v12"
//       over ds_load_b32/ds_store_b32 of every element; no +0, so -0 stays -0).
//   P7  the B operands built from S (stage 1 and stage 2 oi chain) convert only 8 floats and use them twice:
//       b[8..15] = b[0..7] (p7 lines 301-333: 2 ds_load_b128 per half, "v_dual_mov_b32 v22, v18 :: v_dual_mov_b32 v23, v19", v24=v20, v25=v21).
//   P8  the A operands of stage 1 and stage 2 oi chain (w and q) are never loaded: WMMA(v[10:17], ...) with v10..v13 never written
//       (p8 lines 121-158 copy v10..v13 to v14..v17, nothing writes v10..v13). See report for what the hardware returns.
#pragma clang fp contract(off)
#include <hip/hip_runtime.h>

typedef _Float16 f16;
typedef f16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

__device__ static inline f16 c16(float x) { return (f16)__builtin_fmaf(x, 16.0f, 0.0f); }

__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

__device__ static inline v8f wmma(v16h a, v16h b, v8f c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c); }

__device__ static inline v16h ld16(const float* p, bool valid) {
  v16h r;
  for (int j = 0; j < 16; j++) r[j] = valid ? c16(p[j]) : (f16)0.0f;
  return r;
}

template <int SW, int P>
__device__ static inline void probe_body(const float* q, const float* kk, const float* gc, const float* u, const float* w,
                                         const float* am, float* st, const int* idx, long n, unsigned short* o, int ktok) {
  constexpr int TC = SW / 16;
  __shared__ float S[128 * SW];
  __shared__ f16 vn[SW * 72];
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, lr = lane & 15, lh = lane >> 4;
  const int h = blockIdx.x, by = blockIdx.y;
  const long sbase = n + (long)idx[0] * 37748736L + (long)h * 16384 + by * SW;

  for (int e = tid; e < 128 * SW; e += 128) { int r = e / SW, c = e % SW; S[e] = st[sbase + (long)r * 128 + c]; }
  __syncthreads();

  const int nch = (ktok + 63) >> 6;
  for (int ch = 0; ch < nch; ch++) {
    const int t0 = ch * 64;
    const float gend = gc[(long)(t0 + 63) * 48 + h];
    const float G = expf(gend);
    // 1. vnew
    {
      const int tok = t0 + 16 * wave + lr;
      const bool val = tok < ktok;
      const float* wrow = w + ((long)tok * 48 + h) * 128;
      for (int tc = 0; tc < TC; tc++) {
        v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
        if (P != 1) {
          for (int ks = 0; ks < 8; ks++) {
            v16h a, b;
            if (P == 8) { for (int j = 0; j < 16; j++) a[j] = (f16)((float)((lr + (j & 7)) & 7) * 0.125f); }  // P8: measured pattern, see report
            else a = ld16(wrow + 16 * ks, val);
            for (int j = 0; j < 16; j++) b[j] = c16(S[(16 * ks + (P == 7 ? (j & 7) : j)) * SW + 16 * tc + lr]);
            acc = wmma(a, b, acc);
          }
        }
        for (int i = 0; i < 8; i++) {
          int row = 16 * wave + 2 * i + lh, t = t0 + row;
          f16 r = (f16)0.0f;
          if (t < ktok) {
            float uu = u[((long)t * 48 + h) * 128 + by * SW + 16 * tc + lr];
            r = (P == 1) ? c16(uu) : c16(__builtin_fmaf(acc[i], -0.00390625f, uu));
          }
          vn[(16 * tc + lr) * 72 + row] = r;
        }
      }
    }
    __syncthreads();
    // 2. outputs
    {
      const int tok = t0 + 16 * wave + lr;
      const bool val = tok < ktok;
      const float* qrow = q + ((long)tok * 48 + h) * 128;
      const float* arow = am + (((long)ch * 48 + h) * 64 + 16 * wave + lr) * 64;
      for (int tc = 0; tc < TC; tc++) {
        v8f oi = {0, 0, 0, 0, 0, 0, 0, 0}, oa = {0, 0, 0, 0, 0, 0, 0, 0};
        if (P != 2) {
          for (int ks = 0; ks < 8; ks++) {
            v16h a, b;
            if (P == 8) { for (int j = 0; j < 16; j++) a[j] = (f16)((float)((lr + (j & 7)) & 7) * 0.125f); }  // P8: measured pattern, see report
            else a = ld16(qrow + 16 * ks, val);
            for (int j = 0; j < 16; j++) b[j] = c16(S[(16 * ks + (P == 7 ? (j & 7) : j)) * SW + 16 * tc + lr]);
            oi = wmma(a, b, oi);
          }
        }
        if (P != 3) {
          for (int s = 0; s <= wave; s++) {
            v16h a = ld16(arow + 16 * s, true), b;
            for (int j = 0; j < 16; j++) b[j] = vn[(16 * tc + lr) * 72 + 16 * s + j];
            oa = wmma(a, b, oa);
          }
        }
        for (int i = 0; i < 8; i++) {
          int row = 16 * wave + 2 * i + lh, t = t0 + row;
          if (t < ktok) {
            float ee = expf(gc[(long)t * 48 + h]);
            float oiv = (P == 2) ? 0.0f : oi[i];
            float r = __builtin_fmaf(oiv, ee, oa[i]);
            r = r * 0.00390625f;
            o[(long)t * 6144 + h * 128 + by * SW + 16 * tc + lr] = f2bf_rne(r);
          }
        }
      }
    }
    __syncthreads();
    // 3. state update
    if (P == 4) {
      for (int e = tid; e < 128 * SW; e += 128) S[e] = __builtin_fmaf(G, S[e], 0.0f);
    } else if (P == 6) {
      for (int e = tid; e < 128 * SW; e += 128) S[e] = G * S[e];  // p6 lines 3418-3421: v_mul_f32 G, S, ds store
    } else {
      for (int tile = wave; tile < 8 * TC; tile += 4) {
        const int rt = tile / TC, tc = tile % TC;
        v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int qq = 0; qq < 4; qq++) {
          v16h a, b;
          for (int j = 0; j < 16; j++) {
            int tl = 16 * qq + j, t = t0 + tl;
            float kv = 0.0f;
            if (t < ktok) kv = kk[((long)t * 48 + h) * 128 + 16 * rt + lr];
            float eb = expf(gend - gc[(long)t * 48 + h]);
            a[j] = (t < ktok) ? c16(kv * eb) : (f16)0.0f;
            if (P == 5) a[j] = (f16)0.0f;  // P5: unwritten LDS table, see report
            b[j] = vn[(16 * tc + lr) * 72 + tl];
          }
          acc = wmma(a, b, acc);
        }
        for (int i = 0; i < 8; i++) {
          int r = 16 * rt + 2 * i + lh, c = 16 * tc + lr;
          float sv = S[r * SW + c];
          S[r * SW + c] = __builtin_fmaf(acc[i], 0.00390625f, G * sv);
        }
      }
    }
    __syncthreads();
  }
  for (int e = tid; e < 128 * SW; e += 128) { int r = e / SW, c = e % SW; st[sbase + (long)r * 128 + c] = S[e]; }
}

#define PROBE_ARGS const unsigned short* keys, const float* q, const float* kk, const float* gc, const float* u, const float* w, \
                   const float* am, float* st, const int* idx, long n, unsigned short* o, int ktok
#define PROBE_KERNEL(PP) \
  extern "C" __global__ __launch_bounds__(128) void k_probe_p##PP(PROBE_ARGS) { probe_body<64, PP>(q, kk, gc, u, w, am, st, idx, n, o, ktok); }

PROBE_KERNEL(1) PROBE_KERNEL(2) PROBE_KERNEL(3) PROBE_KERNEL(4) PROBE_KERNEL(5) PROBE_KERNEL(6) PROBE_KERNEL(7) PROBE_KERNEL(8)
