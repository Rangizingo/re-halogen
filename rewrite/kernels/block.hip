// Rewrite of k_block_pool (mean of four consecutive 128-wide rows) from flash_serve, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o block.co block.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

__device__ static inline unsigned short f2bf_rne(float f) {  // :54192-54194
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// halogen::(anonymous namespace)::k_block_pool(unsigned short const*, unsigned short*, int)
// _ZN7halogen12_GLOBAL__N_112k_block_poolEPKtPti, kernarg: x@0, y@8, n@16 (the kernel never reads n).
// Block b = blockIdx.x (s2); threads d = tid < 0x80 = 128 only (:54039). Loop d += blockDim (:54177, :54183).
// Row width 128 bf16 = 256 bytes: input row group at x + (4*b)*256 bytes (:54044, :54049), output row at y + b*256 bytes (:54048).
// Four loads at row offsets 0, 256, 512, 768 bytes (:54167-54170, same pattern in the packed 32-bit path :54079-54082).
// Sum order: ((0 + r0) + r1) + r2) + r3 in f32 (:54181-54189), * 0.25 (0x3e800000 :54190), bf16 RNE store (:54192-54194).
extern "C" __global__ void k_block_pool(const unsigned short* x, unsigned short* y, int n) {
  (void)n;
  const unsigned short* xr = x + (size_t)blockIdx.x * 4 * 128;
  for (unsigned d = threadIdx.x; d < 128u; d += blockDim.x) {
    float s = 0.0f + bf2f(xr[d]);
    s = s + bf2f(xr[128 + d]);
    s = s + bf2f(xr[256 + d]);
    s = s + bf2f(xr[384 + d]);
    y[(size_t)blockIdx.x * 128 + d] = f2bf_rne(s * 0.25f);
  }
}
