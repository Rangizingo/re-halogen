// Rewrite of k_dnc_att (the original attention-matrix kernel of the dnc family), vendor code object gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dnc_att.co dnc_att.hip
// Listing: kernels-disasm/codeobj_1146000.s (not in git), kernel at lines 2455-2651. The stock listing prints some
// 64-bit VOP3 words as ".long"; cites are raw-listing lines (those words were re-decoded with gfx1151 llvm-mc).
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)  // each a*b+c in this file is an explicit __builtin_fmaf where the listing has v_fma/v_fmac

// halogen::(anonymous namespace)::k_dnc_att(float const*, float const*, float const*, float*, int)
// _ZN7halogen12_GLOBAL__N_19k_dnc_attEPKfS2_S2_Pfi, kernarg: q@0, k@8, g@16, out@24, n@32 (n is never loaded, :2457 loads only the four pointers)
// Block 256 threads, LDS 33024 B = 64 rows x 129 words. blockIdx.x = chunk c (64 tokens), blockIdx.y = head h (48 heads).
// q, k: [token][48][128] f32. g: [token][48] f32 (per-token log gate). out: [c][48][64][64] f32 (out base (c*48+h)*16384 B, :2503-2514).
//   phase 1: LDS tile Kt[j][0..127] = k[(c*64+j)*48+h][0..127] for j=0..63, row stride 129 words (:2466, :2483-2497, 32 iterations of 256 words)
//   phase 2: for each o = tid, tid+256, ... < 4096: i = o>>6, j = o&63 (:2502, :2546); j > i stores 0 (:2547-2550, :2543)
//            j <= i: acc = sum_{d=0..127} q[(c*64+i)*48+h][d] * Kt[j][d], one fma chain in ascending d starting at 0 (:2553-2609, v_fmac_f32)
//            out[o] = acc * expf(g[i] - g[j])  (:2623-2646; expf is the full-precision llvm.exp.f32 expansion:
//            mul by log2e 0x3fb8aa3b, fma error, rndne, v_exp_f32, v_ldexp_f32, 0 below -103.278 (0xc2ce8ed0), +inf above 88.7228 (0x42b17218))
extern "C" __global__ void k_dnc_att(const float* q, const float* k, const float* g, float* out, int n) {
  __shared__ float kt[64 * 129];
  const int tid = threadIdx.x, c = blockIdx.x, h = blockIdx.y;
  for (int e = tid; e < 64 * 128; e += 256) {
    int j = e >> 7, d = e & 127;
    kt[j * 129 + d] = k[((size_t)(c * 64 + j) * 48 + h) * 128 + d];
  }
  __syncthreads();
  float* o_base = out + (size_t)(c * 48 + h) * 4096;
  for (int o = tid; o < 4096; o += 256) {
    int i = o >> 6, j = o & 63;
    float v = 0.0f;
    if (j <= i) {
      const float* qr = q + ((size_t)(c * 64 + i) * 48 + h) * 128;
      float acc = 0.0f;
      for (int d = 0; d < 128; d++) acc = __builtin_fmaf(qr[d], kt[j * 129 + d], acc);
      float dg = g[(size_t)(c * 64 + i) * 48 + h] - g[(size_t)(c * 64 + j) * 48 + h];
      v = acc * expf(dg);
    }
    o_base[o] = v;
  }
}
