// Rewrite of k_hc_fold (sum of the four hyper-connection streams, scaled) from flash_serve, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o hc_fold.co hc_fold.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

__device__ static inline unsigned short f2bf_rne(float f) {  // :60658-60663 (bfe 16,1 ; add3 0x7fff ; store high half)
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// halogen::(anonymous namespace)::k_hc_fold(unsigned short const*, unsigned short*, float)
// _ZN7halogen12_GLOBAL__N_19k_hc_foldEPKtPtf, kernarg: x@0, y@8, scale@16.
// i = blockIdx.x*blockDim.x + tid (v_mad_u64_u32 :60606), guard i < 0xa00 = 2560 (:60608).
// Row b = blockIdx.y (s3): x row = 0x5000 bytes at x + b*0x5000 (:60613-60614, :60619-60620); y row = 0x1400 bytes (:60654-60656).
// Four bf16 loads at element i + k*2560 (byte offsets 0, 0x1400, 0x2800, 0x3c00: :60627-60640).
// Sum order: ((0 + x0) + x1) + x2) + x3 in f32 (:60647-60651), then * scale (:60653), bf16 RNE store (:60658-60664).
extern "C" __global__ void k_hc_fold(const unsigned short* x, unsigned short* y, float scale) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 2560u) return;
  const unsigned short* xr = x + (size_t)blockIdx.y * 10240;
  float s = 0.0f + bf2f(xr[i]);
  s = s + bf2f(xr[i + 2560]);
  s = s + bf2f(xr[i + 5120]);
  s = s + bf2f(xr[i + 7680]);
  y[(size_t)blockIdx.y * 2560 + i] = f2bf_rne(scale * s);
}
