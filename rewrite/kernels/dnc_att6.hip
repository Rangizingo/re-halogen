// Rewrite of k_dnc_att6<W, b1, b2, b3> (attention matrix in W-column blocks, f32 + f16 outputs), vendor code object gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dnc_att6.co dnc_att6.hip
// Listing: kernels-disasm/codeobj_1146000.s (not in git). Raw-listing line ranges of the 7 instances:
//   <64,F,F,F> 318813-319045   <64,T,F,F> 319045-319274   <32,F,F,F> 319274-319621   <32,T,F,F> 319621-319959
//   <16,F,F,F> 319959-320292   <32,T,T,F> 7483-7821       <32,T,T,T> 308094-308466
// The stock listing prints some 64-bit VOP3 words as ".long"; cites are raw-listing lines (those words re-decoded with gfx1151 llvm-mc).
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)  // each a*b+c in this file is an explicit __builtin_fmaf where the listing has v_fma/v_fmac

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// halogen::(anonymous namespace)::k_dnc_att6<W,b1,b2,b3>(float const*, float const*, float const*, float*, int,
//     unsigned short*, unsigned short const*, float const*, float const*)
// kernarg: q@0, k@8, g@16, out@24, n@32, out2@40, mirror@48, qscale@56, kscale@64 (size 72)
//
// Grid: B = block index along tokens, head h. b2 = false: x = B, y = h (:319276-319279 for <32,F,F,F>: s2 = B, s3 = h).
// b2 = true: x = h, y = B (:7487-7498, s2 = h, s3 = B); this is the launcher's (48, 2*nc) grid. Nothing else depends on b2.
// b1 only changes how q rows are fetched (wave-uniform row via s_load_b512 instead of per-lane global loads); same arithmetic.
// Block 256 threads. Block B covers tokens 64c+b0 .. +W-1 with c = B/(64/W), b0 = W*(B%(64/W)) (:319371 s9 = B*W - c*64 = b0).
// q, k: [token][48][128] f32 (b3 = false only). g: [token][48]. out: [c][48][64][64] f32 (base (c*48+h)*16384 B, :319355-319360),
// out2: [c][48][64][64] f16 (base (c*48+h)*8192 B, :319365-319370). Either output pointer may be null (:319347, :319363): then it is skipped.
//   K tile: W rows x 128, row stride 129 words, rows = tokens B*W + r (:319303-319321, e < W*128; the 0xeff/0x6ff bound).
//   g tile: threads < 64 load g[(64c+tid)*48+h] at LDS byte W*516 (:319331-319345, 16512 for W=32, 8256 for W=16, 33024 for W=64).
//   elements e = tid, tid+256, ... < (64-b0)*W (:319375, s10 = 0x800 - 32*b0 for W=32): i = b0 + e/W, j = b0 + e%W (:319402-319407).
//   j > i: value 0. Else acc = sum_{d=0..127} q[64c+i][d]*Kt[j-b0][d], one fma chain, ascending d, from 0;
//   value = acc * expf(g[i] - g[j])  (:319470-319494; expf = full-precision llvm.exp.f32 expansion as in dnc_att.hip).
//   out[i*64+j] = value; out2[i*64+j] = f16(fma(value, 16.0, 0)) (v_fma_mixlo_f16 with 0x41800000, :319520; fma turns -0 into +0).
//   Rows 0..b0-1 of the block's columns b0..b0+W-1 are written as 0 to both outputs (:319533-319614).
// b3 = true (<32,T,T,T>, :308094-308466) reads neither q nor k (:308097 loads kernarg 0x28.., :308164 loads 0x10..; 0x00 and 0x08 are never read).
//   gq = h/3 (:308099-308106). mirror row t holds 10240 u16: q section gq*128 + d at +0, k section at +2048 + gq*128 + d (:308140-308153, 0x5000 B per token).
//   K tile element: t = B*W + r; t >= n (signed) gives 0, else bf16(mirror[t*10240 + 2048 + gq*128 + d]) * kscale[t*48+h]
//   (:308136-308160: shift left 16, v_mul_f32 with the per-(token,head) scale).
//   q element: t = B*W + (e/W) - the query row token; t >= n makes acc 0 (:308254-308255), else q'_d = qscale[t*48+h] * bf16(mirror[t*10240 + gq*128 + d])
//   (s_mul_f32, :308283-308312), then acc = fma(q'_d, Kt[j-b0][d], acc), low half before high half of each u32, ascending d.
template <int W, bool B2, bool B3>
__device__ static inline void att6_body(const float* q, const float* k, const float* g, float* out, int n,
                                        unsigned short* out2, const unsigned short* mirror, const float* qsc,
                                        const float* ksc) {
  constexpr int NB = 64 / W;
  __shared__ float kt[W * 129];
  __shared__ float gt[64];
  const int tid = threadIdx.x;
  const int B = B2 ? blockIdx.y : blockIdx.x;
  const int h = B2 ? blockIdx.x : blockIdx.y;
  const int c = B / NB, b0 = W * (B % NB), gq = h / 3;
  const int tok0 = B * W;
  for (int e = tid; e < W * 128; e += 256) {
    int r = e >> 7, d = e & 127, t = tok0 + r;
    float v;
    if (B3) {
      v = 0.0f;
      if (t < n) v = ksc[(size_t)t * 48 + h] * bf2f(mirror[(size_t)t * 10240 + 2048 + gq * 128 + d]);
    } else {
      v = k[((size_t)t * 48 + h) * 128 + d];
    }
    kt[r * 129 + d] = v;
  }
  if (tid < 64) gt[tid] = g[(size_t)(c * 64 + tid) * 48 + h];
  __syncthreads();
  float* ob = out ? out + (size_t)(c * 48 + h) * 4096 : nullptr;
  unsigned short* ob2 = out2 ? out2 + (size_t)(c * 48 + h) * 4096 : nullptr;
  const int nel = (64 - b0) * W;
  for (int e = tid; e < nel; e += 256) {
    int ip = e / W, jp = e % W, i = b0 + ip, j = b0 + jp;
    float v = 0.0f;
    if (jp <= ip) {
      int t = tok0 + ip;
      float acc = 0.0f;
      if (B3) {
        if (t < n) {
          float s = qsc[(size_t)t * 48 + h];
          const unsigned short* mr = mirror + (size_t)t * 10240 + gq * 128;
          for (int d = 0; d < 128; d++) acc = __builtin_fmaf(s * bf2f(mr[d]), kt[jp * 129 + d], acc);
        }
      } else {
        const float* qr = q + ((size_t)t * 48 + h) * 128;
        for (int d = 0; d < 128; d++) acc = __builtin_fmaf(qr[d], kt[jp * 129 + d], acc);
      }
      v = acc * expf(gt[i] - gt[j]);
    }
    if (ob) ob[i * 64 + j] = v;
    if (ob2) { _Float16 hv = (_Float16)__builtin_fmaf(v, 16.0f, 0.0f); ob2[i * 64 + j] = __builtin_bit_cast(unsigned short, hv); }
  }
  const int nz = b0 * W;
  for (int e = tid; e < nz; e += 256) {
    int r = e / W, j = b0 + e % W;
    if (ob) ob[r * 64 + j] = 0.0f;
    if (ob2) ob2[r * 64 + j] = 0;
  }
}

#define ATT6(NAME, W, B2, B3) \
  extern "C" __global__ void NAME(const float* q, const float* k, const float* g, float* out, int n, \
                                  unsigned short* out2, const unsigned short* mirror, const float* qsc, const float* ksc) { \
    att6_body<W, B2, B3>(q, k, g, out, n, out2, mirror, qsc, ksc); }
ATT6(k_dnc_att6_64_0_0_0, 64, false, false)
ATT6(k_dnc_att6_64_1_0_0, 64, false, false)
ATT6(k_dnc_att6_32_0_0_0, 32, false, false)
ATT6(k_dnc_att6_32_1_0_0, 32, false, false)
ATT6(k_dnc_att6_16_0_0_0, 16, false, false)
ATT6(k_dnc_att6_32_1_1_0, 32, true, false)
ATT6(k_dnc_att6_32_1_1_1, 32, true, true)
