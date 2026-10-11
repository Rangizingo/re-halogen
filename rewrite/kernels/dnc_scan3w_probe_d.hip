// g19d: corrected rewrites of probes P5 and P8 of k_dnc_scan3w<2,4,true,false,P> (codeobj_1146000, flash_serve 0.17.3, gfx1151).
// Replaces the "unwritten state" explanations in dnc_scan3w_probe.hip (g19b, left untouched). Built from the full listing
// kernels-disasm-v2/codeobj_1146000.s (no .long); line numbers below are lines of that file. P5 = kernel at 176736, P8 at 188227.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dnc_scan3w_probe_d.co dnc_scan3w_probe_d.hip
// Everything except the two items below is the g19b body (rewrite/kernels/dnc_scan3w_probe.hip), which MATCHed.
//
//   P5  The k*e table at LDS 0x2800 IS written, in the prologue, with zeros. 10 rows x 128 threads of `ds_store_b16 ... 0`:
//       176816 `ds_store_b16 v3, v2 offset:10240` (v2.l = 0 from 176815 `v_mov_b16 v2.l, 0`) and 176822 `ds_store_b16 v2, v1 offset:10496` (zero from 176821),
//       in the loop 176798-176823 (s8 = 0,2,..,8, address 2*(s8*128+tid) + 0x2800 and +0x2900, i.e. halfword rows s8 and s8+1).
//       That covers [0x2800, 0x3200) = 16 rows x 160 B, the whole table (row stride 0xa0, v84 = 0xa0*lr + 0x2800 at 176877). The
//       loop ends before the first s_barrier (176828), all 128 threads store, and the region is read only afterwards by
//       ds_load_b128 v84 (176901 ff), feeding v_wmma at 176937. The vendor never stores the k*e values (no other ds_store
//       with offset >= 10240, and the 32 ds_store_b16 on v88 stay under 10240); P0 has the same zero fill and then overwrites it.
//       So A = 0 is the defined value of an LDS table that was written (zeros), not stale LDS. The rewrite below does it the same
//       way: zero-fill a table in the prologue, barrier, read A operands from it in stage 3.
//   P8  v10..v13 are written: 188337 `v_pk_mul_f16 v11, 0x3000, v5`, 188341 v13 (v4), 188359 v10 (v2), 188372 v12 (v3), 0x3000 = 0.125 (f16),
//       v2..v5 = packed f16 of small integers built at 188315-188336 (`v_and_b32 7`, `v_cvt_f32_ubyte0`, `v_xor_b32 4`, `v_cvt_f16_f32`):
//       v2 = [t&7, (t+1)&7], v5 = [(t+2)&7, (t+3)&7], v3 = [(t+4)&7, (t+5)&7], v4 = [(t+6)&7, (t+7)&7] (t = tid). v14..v17 copy
//       v10..v13 (188342 v15,v11; 188346 v17,v13; 188364 v14,v10; v16,v12 by dual mov). So A[m][k] = 0.125 * ((m + (k & 7)) & 7) as an f16
//       multiply, the second half of k repeating the first. The rewrite computes it that way (f16 integer times f16 0.125).
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

// P8 A operand element (188315-188372): f16(n) * f16(0.125) with n = (m + (k & 7)) & 7, m = lane & 7 (= lr & 7).
__device__ static inline f16 p8_a(int lr, int j) {
  f16 n = (f16)(float)((lr + (j & 7)) & 7);
  return n * (f16)0.125f;
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
  __shared__ f16 ke[16 * 80];  // k*e table, LDS 0x2800..0x3200, row stride 160 B (v84, 176877)
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, lr = lane & 15, lh = lane >> 4;
  const int h = blockIdx.x, by = blockIdx.y;
  const long sbase = n + (long)idx[0] * 37748736L + (long)h * 16384 + by * SW;

  for (int e = tid; e < 128 * SW; e += 128) { int r = e / SW, c = e % SW; S[e] = st[sbase + (long)r * 128 + c]; }
  if (P == 5) { for (int e = tid; e < 16 * 80; e += 128) ke[e] = (f16)0.0f; }  // 176798-176823: prologue zero fill, before the barrier at 176828
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
            if (P == 8) { for (int j = 0; j < 16; j++) a[j] = p8_a(lr, j); }  // 188337-188372
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
            if (P == 8) { for (int j = 0; j < 16; j++) a[j] = p8_a(lr, j); }  // 188337-188372
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
            if (P == 5) a[j] = ke[lr * 80 + tl];  // P5: no store of k*e (no ds_store offset >= 10240 after the prologue); A comes from the zeroed table (ds_load_b128 v84 at 176901)
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
  extern "C" __global__ __launch_bounds__(128) void k_probe_d_p##PP(PROBE_ARGS) { probe_body<64, PP>(q, kk, gc, u, w, am, st, idx, n, o, ktok); }

PROBE_KERNEL(5) PROBE_KERNEL(8)
