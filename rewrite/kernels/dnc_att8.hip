// Rewrite of k_dnc_att8 (attention matrix on the matrix unit: f16 WMMA, scale 16), vendor code object gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dnc_att8.co dnc_att8.hip
// Listing: kernels-disasm/codeobj_1146000.s (not in git), kernel at lines 5003-5530. The stock listing prints some
// 64-bit VOP3 words as ".long"; cites are raw-listing lines (those words were re-decoded with gfx1151 llvm-mc).
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)  // each a*b+c in this file is an explicit __builtin_fmaf where the listing has v_fma/v_fmac

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

// halogen::(anonymous namespace)::k_dnc_att8(float const*, float const*, float const*, float*, int)
// _ZN7halogen12_GLOBAL__N_110k_dnc_att8EPKfS2_S2_Pfi, kernarg: q@0, k@8, g@16, out@24, n@32 (n IS used here, :5007 loads it)
// Block 256 threads (8 waves of 32), LDS 37120 B = 2 tiles of 64 rows x 144 halves (288 B stride, :5030) + 64 floats.
// blockIdx.x = chunk c, blockIdx.y = head h. q, k: [token][48][128] f32, g: [token][48] f32, out: [c][48][64][64] f32 (:5105-5123).
//   tile load, e = tid, tid+256, ... < 8192 (:5029-5036): row = e>>7, col = e&127, token = 64c+row.
//     token >= n (signed, :5042): both halves 0. Else Kh = f16(fma(k, 16.0, 0)) at LDS 0, Qh = f16(fma(q, 16.0, 0)) at LDS 0x4800
//     (v_fma_mixlo_f16 / v_fma_mixhi_f16 with s4 = 0x41800000, :5067-5070). fma(x,16,0) turns -0 into +0.
//   g tile: threads < 64 load g[(64c+tid)*48+h] with no bounds check, at LDS 36864 (:5075-5088).
//   wave w = tid>>5: column tile tj = w&3 (:5098); row tiles ti = (tid>>7) + 2*it, it = 0,1 (:5096, :5146, loop runs twice).
//   tiles with ti < tj store zeros; ti >= tj: acc = 0, then 8 x v_wmma_f32_16x16x16_f16, k slice 16*kc, A = Q rows (16ti + lane%16),
//   B = K rows (16tj + lane%16), sequential over kc (:5195-5227).
//   lane result r (0..7): row = 16ti + 2r + (lane>>4), col j = 16tj + lane%16; value = 0 unless j <= row (:5232-5235), then
//   out = acc * (expf(g[row] - g[j]) * 2^-8)  (0x3b800000, :5241-5263; expf = full-precision llvm.exp.f32 expansion).
extern "C" __global__ void k_dnc_att8(const float* q, const float* k, const float* g, float* out, int n) {
  __shared__ _Float16 kt[64 * 144];
  __shared__ _Float16 qt[64 * 144];
  __shared__ float gt[64];
  const int tid = threadIdx.x, c = blockIdx.x, h = blockIdx.y;
  for (int e = tid; e < 8192; e += 256) {
    int row = e >> 7, col = e & 127, tok = c * 64 + row;
    _Float16 kh = (_Float16)0.0f, qh = (_Float16)0.0f;
    if (tok < n) {
      size_t at = ((size_t)tok * 48 + h) * 128 + col;
      qh = (_Float16)__builtin_fmaf(q[at], 16.0f, 0.0f);
      kh = (_Float16)__builtin_fmaf(k[at], 16.0f, 0.0f);
    }
    qt[row * 144 + col] = qh;
    kt[row * 144 + col] = kh;
  }
  if (tid < 64) gt[tid] = g[(size_t)(c * 64 + tid) * 48 + h];
  __syncthreads();
  const int w = tid >> 5, l = tid & 31, tj = w & 3, half = l >> 4, ln = l & 15;
  float* o_base = out + (size_t)(c * 48 + h) * 4096;
  for (int it = 0; it < 2; it++) {
    int ti = (tid >> 7) + 2 * it;
    float8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
    if (ti >= tj) {
      for (int kc = 0; kc < 8; kc++) {
        half16 a, b;
        for (int x = 0; x < 16; x++) {
          a[x] = qt[(16 * ti + ln) * 144 + 16 * kc + x];
          b[x] = kt[(16 * tj + ln) * 144 + 16 * kc + x];
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
      }
    }
    int j = 16 * tj + ln;
    for (int r = 0; r < 8; r++) {
      int row = 16 * ti + 2 * r + half;
      float v = 0.0f;
      if (ti >= tj && j <= row) {
        float e = expf(gt[row] - gt[j]);
        e = e * 0.00390625f;
        v = acc[r] * e;
      }
      o_base[row * 64 + j] = v;
    }
  }
}
