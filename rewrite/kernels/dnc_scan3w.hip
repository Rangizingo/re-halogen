// Rewrites of the "dnc_scan3w" family from codeobj_1146000 (flash_serve 0.17.3, gfx1151): k_dnc_scan3w<A,4,b1,b2,P>.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dnc_scan3w.co dnc_scan3w.hip
// Listing: llvm-objdump --mcpu=gfx1151 of work/fatbin/codeobj_1146000.hsaco (ROCm build); cites are code-object addresses,
// for example "@13F684" is k_dnc_scan3w<4,4,true,false,0> (0x13f300 start). Same bytes as work/kernels/*scan3w*.s.
//
// What the kernel is (all proven by the MATCH rows in rewrite/tables/g19.md): the chunk-state scan of a gated delta rule, one
// workgroup per (head h = blockIdx.x in 0..47, column slab by = blockIdx.y of width SW = 128/A), 128 threads = 4 waves.
// The 128 x SW slab of the f32 state S[h] lives in LDS; tokens go in chunks of 64.
// Arguments (kernarg offsets): 0 unused u16*, 8 q, 16 k, 24 g, 32 u, 40 w, 48 amat, 56 state, 64 idx, 72 n, 80 o (bf16), 88 ktok.
//   q, k, u, w: f32 [T][48][128]; g: f32 [T][48] (cumulative log gate); amat: f32 [chunk][48][64][64]; o: bf16 [T][48*128].
// State base = state + n floats + idx[0]*0x9000000 bytes + h*16384 floats + by*SW (@13F344 s_load idx[0], @13F34C-@13F390 address),
// row stride 128 floats (@13F3C8 shl 9).
// Per chunk c (t0 = 64c), all f16 operands are f16(x*16) (v_fma_mixlo/hi_f16 x, 16.0, 0 : fma in f32 then RNE to f16, @13F774..),
// accumulators are zero-initialised WMMA f32_16x16x16_f16 chains (K step 16, ascending), products carry a factor 256 removed by 1/256:
//   gend = g[t0+63]; G = expf(gend)                                        (@13F8B4 s_load s23, @142CCC..@142D64 scalar expf)
//   1. vnew[t][c] = u[t][c] - (1/256) * sum_i w[t][i] S[i][c]  (8 WMMA steps over i; fma(acc,-1/256,u), stored as f16(16*vnew), 0 for t>=ktok)
//                                                                          (@13F794-@140F00 WMMA, @140F50 fmac -1/256, @140F58 fma_mixhi)
//   2. o[t][c] = bf16( (fma(oi, expf(g_t), oa)) / 256 ),  oi = 8-step WMMA chain of f16(16q) x f16(16S), oa = chain over 16-token
//      blocks s = 0..wave of f16(16*amat[t][16s..]) x vnew16                (@1417EC-@142A28 oi, @142BA4/@142BB0 oa, @142C80-@143A98 store)
//   3. S[i][c] = fma(acc, 1/256, G*S[i][c]), acc = 4-step chain over t of f16(16*(k[t][i]*expf(gend-g_t))) x vnew16   (@142F0C.., @142F8C)
// After the last chunk the slab is stored back (@143AA4-@143B88).
#pragma clang fp contract(off)
#include <hip/hip_runtime.h>

typedef _Float16 f16;
typedef f16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

// v_fma_mixlo_f16 d, x, 16.0, 0 : single f32 fma then RNE to f16 (x*16 is exact, the +0 turns -0 into +0).
__device__ static inline f16 c16(float x) { return (f16)__builtin_fmaf(x, 16.0f, 0.0f); }

// f32 -> bf16 RNE without NaN special case: v_bfe_u32 v,16,1 ; v_add3_u32 0x7fff (@142CA8-@142CB4).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

__device__ static inline v8f wmma(v16h a, v16h b, v8f c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c); }

// 16 consecutive f32 -> 16 f16 (x16), zero when !valid (the vendor skips the load and leaves 0, @13FA50..).
__device__ static inline v16h ld16(const float* p, bool valid) {
  v16h r;
  for (int j = 0; j < 16; j++) r[j] = valid ? c16(p[j]) : (f16)0.0f;
  return r;
}

template <int SW, bool OUT>
__device__ static inline void scan3w_body(const float* q, const float* kk, const float* gc, const float* u, const float* w,
                                          const float* am, float* st, const int* idx, long n, unsigned short* o, int ktok) {
  constexpr int TC = SW / 16;     // 16-column tiles per slab
  __shared__ float S[128 * SW];   // S[i*SW + c]
  __shared__ f16 vn[SW * 72];     // vnew16[c][t], 64 tokens, row stride 72
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
        for (int ks = 0; ks < 8; ks++) {
          v16h a = ld16(wrow + 16 * ks, val), b;
          for (int j = 0; j < 16; j++) b[j] = c16(S[(16 * ks + j) * SW + 16 * tc + lr]);
          acc = wmma(a, b, acc);
        }
        for (int i = 0; i < 8; i++) {
          int row = 16 * wave + 2 * i + lh, t = t0 + row;
          f16 r = (f16)0.0f;
          if (t < ktok) {
            float uu = u[((long)t * 48 + h) * 128 + by * SW + 16 * tc + lr];
            r = c16(__builtin_fmaf(acc[i], -0.00390625f, uu));
          }
          vn[(16 * tc + lr) * 72 + row] = r;
        }
      }
    }
    __syncthreads();
    // 2. outputs
    if (OUT) {
      const int tok = t0 + 16 * wave + lr;
      const bool val = tok < ktok;
      const float* qrow = q + ((long)tok * 48 + h) * 128;
      const float* arow = am + (((long)ch * 48 + h) * 64 + 16 * wave + lr) * 64;
      for (int tc = 0; tc < TC; tc++) {
        v8f oi = {0, 0, 0, 0, 0, 0, 0, 0}, oa = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int ks = 0; ks < 8; ks++) {
          v16h a = ld16(qrow + 16 * ks, val), b;
          for (int j = 0; j < 16; j++) b[j] = c16(S[(16 * ks + j) * SW + 16 * tc + lr]);
          oi = wmma(a, b, oi);
        }
        for (int s = 0; s <= wave; s++) {
          v16h a = ld16(arow + 16 * s, true), b;
          for (int j = 0; j < 16; j++) b[j] = vn[(16 * tc + lr) * 72 + 16 * s + j];
          oa = wmma(a, b, oa);
        }
        for (int i = 0; i < 8; i++) {
          int row = 16 * wave + 2 * i + lh, t = t0 + row;
          if (t < ktok) {
            float ee = expf(gc[(long)t * 48 + h]);
            float r = __builtin_fmaf(oi[i], ee, oa[i]);
            r = r * 0.00390625f;
            o[(long)t * 6144 + h * 128 + by * SW + 16 * tc + lr] = f2bf_rne(r);
          }
        }
      }
    }
    __syncthreads();
    // 3. state update
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
    __syncthreads();
  }
  for (int e = tid; e < 128 * SW; e += 128) { int r = e / SW, c = e % SW; st[sbase + (long)r * 128 + c] = S[e]; }
}

#define SCAN3W_ARGS const unsigned short* keys, const float* q, const float* kk, const float* gc, const float* u, const float* w, \
                    const float* am, float* st, const int* idx, long n, unsigned short* o, int ktok

// k_dnc_scan3w<4,4,true,false,0> (grid 48 x 4)
extern "C" __global__ __launch_bounds__(128) void k_scan3w_a4(SCAN3W_ARGS) { scan3w_body<32, true>(q, kk, gc, u, w, am, st, idx, n, o, ktok); }
// k_dnc_scan3w<2,4,true,false,0> (grid 48 x 2)
extern "C" __global__ __launch_bounds__(128) void k_scan3w_a2(SCAN3W_ARGS) { scan3w_body<64, true>(q, kk, gc, u, w, am, st, idx, n, o, ktok); }
// k_dnc_scan3w<8,4,false,false,0> (grid 48 x 8): b1 = false has no q/amat loads and no output stores (state only)
extern "C" __global__ __launch_bounds__(128) void k_scan3w_a8(SCAN3W_ARGS) { scan3w_body<16, false>(q, kk, gc, u, w, am, st, idx, n, o, ktok); }
