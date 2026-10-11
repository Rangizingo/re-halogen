// Rewrite of k_dnc_att7<32, R> (attention matrix, two 32-column blocks per chunk), vendor code object gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dnc_att7.co dnc_att7.hip
// Listing: kernels-disasm/codeobj_1146000.s (not in git): R=2 lines 320292-320583, R=4 320583-320999, R=8 320999-321571.
// The stock listing prints some 64-bit VOP3 words as ".long"; cites are raw-listing lines (re-decoded with gfx1151 llvm-mc).
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)  // each a*b+c in this file is an explicit __builtin_fmaf where the listing has v_fma/v_fmac

// halogen::(anonymous namespace)::k_dnc_att7<32, R>(float const*, float const*, float const*, float*, int), R = 2, 4, 8
// _ZN7halogen12_GLOBAL__N_110k_dnc_att7ILi32ELi{2,4,8}EEEvPKfS3_S3_Pfi, kernarg: q@0, k@8, g@16, out@24, n@32 (n unused, :320294 loads the four pointers only)
// Block 256 threads, LDS 16768 B = 32 rows x 129 words (K tile) + 64 floats (g tile at byte 16512).
// Grid x = 2 * chunks, y = 48 heads. s2 = blockIdx.x: c = s2>>1, b0 = 32*(s2&1) (:320297-320302, R=2 listing).
// q, k: [token][48][128] f32, g: [token][48], out: [c][48][64][64] (base (c*48+h)*16384 B, :320371-320381).
//   K tile rows j' = 0..31 = k tokens 64c+b0+j' (:320321-320339, 16 iterations of 256 words, row stride 0x204 B).
//   g tile: threads < 64 load g[(64c+tid)*48+h] (:320345-320357).
//   This block computes columns b0..b0+31 for rows i = b0 + i', i' in [0, 64-b0): wave w (tid>>5) takes rows i' = R*w + 8R*m, R rows per step,
//   while i' < 64-b0 (:320360-320367, :320389-320393). Lane j' = tid&31.
//   value = 0 when j' > i'; else acc = sum_{d=0..127} q[64c+i][d]*Kt[j'][d], one fma chain per row in ascending d from 0 (s_load of 8 q floats
//   per step, v_fmac_f32 :320444-320458), out[i][b0+j'] = acc * expf(g[i] - g[b0+j']) (:320470-320493, exp expansion as in dnc_att.hip).
//   b0 = 32 blocks also zero the upper-right region rows 0..31, cols 32..63 (:320533-320579). The b0 = 0 block writes columns 0..31 only.
//   R changes only how many rows a wave handles per step; the arithmetic is the same.
template <int R>
__device__ static inline void att7_body(const float* q, const float* k, const float* g, float* out) {
  __shared__ float kt[32 * 129];
  __shared__ float gt[64];
  const int tid = threadIdx.x, c = blockIdx.x >> 1, b0 = 32 * (blockIdx.x & 1), h = blockIdx.y;
  for (int e = tid; e < 4096; e += 256) {
    int j = e >> 7, d = e & 127;
    kt[j * 129 + d] = k[((size_t)(c * 64 + b0 + j) * 48 + h) * 128 + d];
  }
  if (tid < 64) gt[tid] = g[(size_t)(c * 64 + tid) * 48 + h];
  __syncthreads();
  float* ob = out + (size_t)(c * 48 + h) * 4096;
  const int w = tid >> 5, jr = tid & 31, nrows = 64 - b0;
  for (int ib = R * w; ib < nrows; ib += 8 * R) {
    for (int r = 0; r < R; r++) {
      int ip = ib + r, i = b0 + ip, j = b0 + jr;
      float v = 0.0f;
      if (jr <= ip) {
        const float* qr = q + ((size_t)(c * 64 + i) * 48 + h) * 128;
        float acc = 0.0f;
        for (int d = 0; d < 128; d++) acc = __builtin_fmaf(qr[d], kt[jr * 129 + d], acc);
        v = acc * expf(gt[i] - gt[j]);
      }
      ob[i * 64 + j] = v;
    }
  }
  if (b0 == 32)
    for (int e = tid; e < 1024; e += 256) ob[(e >> 5) * 64 + 32 + (e & 31)] = 0.0f;
}

extern "C" __global__ void k_dnc_att7_2(const float* q, const float* k, const float* g, float* out, int n) { att7_body<2>(q, k, g, out); }
extern "C" __global__ void k_dnc_att7_4(const float* q, const float* k, const float* g, float* out, int n) { att7_body<4>(q, k, g, out); }
extern "C" __global__ void k_dnc_att7_8(const float* q, const float* k, const float* g, float* out, int n) { att7_body<8>(q, k, g, out); }
