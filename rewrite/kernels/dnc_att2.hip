// Rewrite of k_dnc_att2 (attention-matrix kernel, 4 passes of 16 query rows), vendor code object gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dnc_att2.co dnc_att2.hip
// Listing: kernels-disasm/codeobj_1146000.s (not in git), kernel at lines 4312-5003. The stock listing prints some
// 64-bit VOP3 words as ".long"; cites are raw-listing lines (those words were re-decoded with gfx1151 llvm-mc).
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)  // each a*b+c in this file is an explicit __builtin_fmaf where the listing has v_fma/v_fmac

// halogen::(anonymous namespace)::k_dnc_att2(float const*, float const*, float const*, float*, int)
// _ZN7halogen12_GLOBAL__N_110k_dnc_att2EPKfS2_S2_Pfi, kernarg: q@0, k@8, g@16, out@24, n@32 (n is never loaded, :4314 loads the four pointers only)
// Same result as k_dnc_att (dnc_att.hip), computed differently. Block 256 threads, LDS 41280 B = (64 + 16) rows x 129 words.
// blockIdx.x = chunk c, blockIdx.y = head h; q, k: [token][48][128], g: [token][48], out: [c][48][64][64] (base (c*48+h)*16384 B, :4404-4409).
//   K tile: 64 rows at LDS 0, row stride 129 words (0x204 B, :4325, :4338-4352, 32 iterations of 256 words)
//   4 passes p = 0..3 (the bodies at :4421, :4571, :4735 and one more): Q tile = q rows 16p..16p+15 at LDS byte 33024 (:4389, :4560, :4724; 8 iterations, :4385 bound 0x6ff)
//   then o = tid, tid+256, ... < 1024: i = 16p + (o>>6), j = o&63; out[i*64+j] = 0 when j > i (:4440-4443, :4593-4594);
//   otherwise acc = sum_{d=0..127} Qt[i][d]*Kt[j][d] as one fma chain, ascending d, starting at 0 (:4452-4500, v_fmac_f32),
//   out = acc * expf(g[i] - g[j]) with the full-precision llvm.exp.f32 expansion (:4513-4535).
extern "C" __global__ void k_dnc_att2(const float* q, const float* k, const float* g, float* out, int n) {
  __shared__ float kt[64 * 129];
  __shared__ float qt[16 * 129];
  const int tid = threadIdx.x, c = blockIdx.x, h = blockIdx.y;
  for (int e = tid; e < 64 * 128; e += 256) {
    int j = e >> 7, d = e & 127;
    kt[j * 129 + d] = k[((size_t)(c * 64 + j) * 48 + h) * 128 + d];
  }
  float* o_base = out + (size_t)(c * 48 + h) * 4096;
  for (int p = 0; p < 4; p++) {
    __syncthreads();
    for (int e = tid; e < 16 * 128; e += 256) {
      int r = e >> 7, d = e & 127;
      qt[r * 129 + d] = q[((size_t)(c * 64 + 16 * p + r) * 48 + h) * 128 + d];
    }
    __syncthreads();
    for (int o = tid; o < 1024; o += 256) {
      int il = o >> 6, i = 16 * p + il, j = o & 63;
      float v = 0.0f;
      if (j <= i) {
        float acc = 0.0f;
        for (int d = 0; d < 128; d++) acc = __builtin_fmaf(qt[il * 129 + d], kt[j * 129 + d], acc);
        float dg = g[(size_t)(c * 64 + i) * 48 + h] - g[(size_t)(c * 64 + j) * 48 + h];
        v = acc * expf(dg);
      }
      o_base[i * 64 + j] = v;
    }
  }
}
