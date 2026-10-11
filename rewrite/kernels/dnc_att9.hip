// Rewrite of k_dnc_att9<S> (attention matrix on the matrix unit with a power-of-two scale 2^S), vendor code object gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dnc_att9.co dnc_att9.hip
// Listing: kernels-disasm/codeobj_1146000.s (not in git). Instances (raw-listing line ranges):
//   S=-8 308466-309946, -4 309946-311426, 0 311426-312893, 4 317333-318813, 8 312893-314373, 12 314373-315853, 16 315853-317333.
// The stock listing prints some 64-bit VOP3 words and the f16 moves as ".long"; cites are raw-listing lines (re-decoded with llvm-mc).
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)  // each a*b+c in this file is an explicit __builtin_fmaf where the listing has v_fma/v_fmac

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

// halogen::(anonymous namespace)::k_dnc_att9<S>(float const*, float const*, float const*, float*, int)
// kernarg: q@0, k@8, g@16, out@24, n@32 (n is used: 317338 s_load_b32 s3, 0x20)
// Same math as k_dnc_att8 (dnc_att8.hip) with these differences, shown for S = 4 (att9_4 at 317333):
//   - Only K goes through LDS (one tile of 64 rows x 144 halves, g tile at byte 18432; LDS 18688 B). Q rows are read from global memory
//     per lane straight into the WMMA A operand: token = 64c + 16*ti + lane%16, 16 floats per WMMA (kc*64 byte offset, :317480-317594),
//     each pair of floats converted separately; token >= n (signed, :317468) gives 0.
//   - scale: the f16 value is f16(fma(x, 2^S, 0)) (v_fma_mixlo_f16 with s8 = 0x41800000 for S = 4, :317482); for S = 0 the vendor emits a
//     plain v_cvt_f16_f32 (att9_0, no fma, keeps -0).
//   - output: acc * (expf(g[row] - g[j]) * 2^(-2S)) (v_mul_f32 with 0x3b800000 = 2^-8 for S = 4, :318638-318640; 0x37800000 for S = 8,
//     0x3d800000 / 0x43800000 for S = -4 / 8, ...). For S = 0 the extra multiply is absent: acc * expf(...).
//   - K tile load, token guard (token >= n gives 0) and g tile load are the same as att8 (:317371-317408; LDS K at 0, g at 18432).
//   - wave w = tid>>5: tj = w&3, ti = (w>>2) + 2*it for it = 0,1 (:317437-317461), tiles with ti < tj store zeros, 8 sequential WMMAs over k (:317616, :318482).
template <int S>
__device__ static inline void att9_body(const float* q, const float* k, const float* g, float* out, int n) {
  constexpr float SC = S >= 0 ? (float)(1u << S) : 1.0f / (float)(1u << -S);
  constexpr float OM = S > 0 ? 1.0f / (float)(1ull << (2 * S)) : (float)(1ull << (-2 * S));
  __shared__ _Float16 kt[64 * 144];
  __shared__ float gt[64];
  const int tid = threadIdx.x, c = blockIdx.x, h = blockIdx.y;
  auto cvt = [](float x) -> _Float16 {
    if (S == 0) return (_Float16)x;
    return (_Float16)__builtin_fmaf(x, SC, 0.0f);
  };
  for (int e = tid; e < 8192; e += 256) {
    int row = e >> 7, col = e & 127, tok = c * 64 + row;
    _Float16 kh = (_Float16)0.0f;
    if (tok < n) kh = cvt(k[((size_t)tok * 48 + h) * 128 + col]);
    kt[row * 144 + col] = kh;
  }
  if (tid < 64) gt[tid] = g[(size_t)(c * 64 + tid) * 48 + h];
  __syncthreads();
  const int w = tid >> 5, l = tid & 31, tj = w & 3, half = l >> 4, ln = l & 15;
  float* o_base = out + (size_t)(c * 48 + h) * 4096;
  for (int it = 0; it < 2; it++) {
    int ti = (w >> 2) + 2 * it;
    float8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
    if (ti >= tj) {
      int qt = c * 64 + 16 * ti + ln;
      for (int kc = 0; kc < 8; kc++) {
        half16 a, b;
        for (int x = 0; x < 16; x++) {
          a[x] = (_Float16)0.0f;
          if (qt < n) a[x] = cvt(q[((size_t)qt * 48 + h) * 128 + 16 * kc + x]);
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
        if (S != 0) e = e * OM;
        v = acc[r] * e;
      }
      o_base[row * 64 + j] = v;
    }
  }
}

#define ATT9(NAME, S) \
  extern "C" __global__ void NAME(const float* q, const float* k, const float* g, float* out, int n) { att9_body<S>(q, k, g, out, n); }
ATT9(k_dnc_att9_m8, -8)
ATT9(k_dnc_att9_m4, -4)
ATT9(k_dnc_att9_0, 0)
ATT9(k_dnc_att9_4, 4)
ATT9(k_dnc_att9_8, 8)
ATT9(k_dnc_att9_12, 12)
ATT9(k_dnc_att9_16, 16)
