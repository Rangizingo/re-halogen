// Rewrites of k_hc_mixed and k_hc_mixed_sig (4-stream weighted mix, quarter-scaled) from flash_serve, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o hc_mixed.co hc_mixed.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
__device__ static inline unsigned f2bf_rne_bits(float f) {  // rounded value in the high half (:33483-33485, :33711-33713)
  unsigned u = __float_as_uint(f);
  return u + ((u >> 16) & 1u) + 0x7fffu;
}

// Common body. Block r = blockIdx.x; loop i = tid; i < 0x2800/4 = 2560 (v_cmp 0x9ff :33412); stride blockDim (group_size_x at 0x34).
// Row offsets: a and b rows = 0x5000 bytes = 10240 bf16 (:33377-33378, :33399-33402); four chunks of 2560 bf16 each,
// byte offsets 0, 0x1400, 0x2800, 0x3c00 (:33430, :33437, :33445, :33447-33457).
// y row = 0x1400 bytes = 2560 bf16 (:33375-33376, :33394); f row = 0x2800 bytes = 2560 f32 (:33373-33374, :33391); lo row = 5120 bf16 (:33388).
// acc = fma(a0,b0,0), then fmac with chunks 1,2,3 in order (:33464, :33474-33477). s = acc * 0.25 (0x3e800000 :33478).
// yb = bf16_rne(s) stored to y (:33483-33486). If f != 0 (kernarg 3), f32 s stored to f (:33458, :33487-33488).
// If lo != 0 (kernarg 4): lo[i] = yb (:33505), lo[i + 2560] = bf16_rne(s - float(yb)) (:33491-33506).
template <bool SIG>
__device__ static inline void hc_mixed_body(const unsigned short* a, const unsigned short* b, unsigned short* y, float* f,
                                            unsigned short* lo) {
  const unsigned short* ar = a + (size_t)blockIdx.x * 10240;
  const unsigned short* br = b + (size_t)blockIdx.x * 10240;
  for (unsigned i = threadIdx.x; i < 2560u; i += blockDim.x) {
    float av[4], bv[4];
    for (int k = 0; k < 4; k++) {
      float x = bf2f(ar[i + 2560 * k]);
      if (SIG) {
        // k_hc_mixed_sig: a is passed through sigmoid and rounded to bf16 first:
        // v_mul 0xbfb8aa3b ; v_exp_f32 ; +1.0 ; IEEE 1/x ; bfe/add3 ; and 0xffff0000 (:33605-33617, :33662-33699).
        float e = __builtin_amdgcn_exp2f(x * -1.44269502162933349609375f);
        float d = 1.0f / (1.0f + e);
        x = __uint_as_float(f2bf_rne_bits(d) & 0xffff0000u);
      }
      av[k] = x;
      bv[k] = bf2f(br[i + 2560 * k]);
    }
    float acc = __builtin_fmaf(av[0], bv[0], 0.0f);
    acc = __builtin_fmaf(av[1], bv[1], acc);
    acc = __builtin_fmaf(av[2], bv[2], acc);
    acc = __builtin_fmaf(av[3], bv[3], acc);
    float s = acc * 0.25f;
    unsigned yb = f2bf_rne_bits(s);
    y[(size_t)blockIdx.x * 2560 + i] = (unsigned short)(yb >> 16);
    if (f) f[(size_t)blockIdx.x * 2560 + i] = s;
    if (lo) {
      unsigned short* l = lo + (size_t)blockIdx.x * 5120;
      l[i] = (unsigned short)(yb >> 16);
      float rem = s - __uint_as_float(yb & 0xffff0000u);
      l[i + 2560] = (unsigned short)(f2bf_rne_bits(rem) >> 16);
    }
  }
}

// halogen::(anonymous namespace)::k_hc_mixed(unsigned short const*, unsigned short const*, unsigned short*, float*, unsigned short*)
// _ZN7halogen12_GLOBAL__N_110k_hc_mixedEPKtS2_PtPfS3_, kernarg: a@0, b@8, y@16, f@24 (may be null), lo@32 (may be null).
extern "C" __global__ void k_hc_mixed(const unsigned short* a, const unsigned short* b, unsigned short* y, float* f, unsigned short* lo) {
  hc_mixed_body<false>(a, b, y, f, lo);
}

// halogen::(anonymous namespace)::k_hc_mixed_sig(...) same signature: _ZN7halogen12_GLOBAL__N_114k_hc_mixed_sigEPKtS2_PtPfS3_.
extern "C" __global__ void k_hc_mixed_sig(const unsigned short* a, const unsigned short* b, unsigned short* y, float* f, unsigned short* lo) {
  hc_mixed_body<true>(a, b, y, f, lo);
}
