// Rewrites of the "hyper" family (hyper-connection stream seeding) from flash_serve, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o hyper.co hyper.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

// f32 -> bf16 RNE without NaN special case: v_bfe_u32 ; v_add3_u32 0x7fff, keep the high half (codeobj_53000.s:50692-50693).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// halogen::(anonymous namespace)::k_hyper_seed(unsigned short const*, unsigned short*)
// _ZN7halogen12_GLOBAL__N_112k_hyper_seedEPKtPt, kernarg: x@0, y@8.
// Block b = blockIdx.x. Input row = 0x1400 bytes (2560 bf16) at x + b*0x1400 (:50451-50454);
// output row = 0x5000 bytes (4*2560 bf16) at y + b*0x5000 (:50456-50459, :50463-50464).
// Thread loop i = tid; i < 0x2800 (10240); i += blockDim (:50622-50623 bound, stride in group_size_x at 0x1c :50449).
// i % 2560 via v_mul_u32_u24 0xcccd ; lshr 27 (= i/2560) and a multiply-subtract (:50482-50500, :50627-50637).
// y[b*10240 + i] = x[b*2560 + i % 2560]: the row is copied four times, bits unchanged (global_load_u16 / store_b16 :50610-50616).
extern "C" __global__ void k_hyper_seed(const unsigned short* x, unsigned short* y) {
  const unsigned short* xr = x + (size_t)blockIdx.x * 2560;
  unsigned short* yr = y + (size_t)blockIdx.x * 10240;
  for (unsigned i = threadIdx.x; i < 10240u; i += blockDim.x) yr[i] = xr[i % 2560u];
}

// halogen::(anonymous namespace)::k_hyper_seed_add(unsigned short const*, unsigned short const*, unsigned short*)
// _ZN7halogen12_GLOBAL__N_116k_hyper_seed_addEPKtS2_Pt, kernarg: x@0, r@8, y@16.
// x row = 2560 bf16 at x + b*0x1400 (:50612-50616); r and y rows = 10240 bf16 at +b*0x5000 (:50618, :50627-50641).
// Same i loop and i % 2560 as k_hyper_seed (:50643-50665, :50735-50746).
// y[i] = bf16_rne(f32(r[i]) + f32(x[i % 2560])): v_add_f32 then bfe/add3 0x7fff and store the high half (:50683-50693, :50755-50761).
extern "C" __global__ void k_hyper_seed_add(const unsigned short* x, const unsigned short* r, unsigned short* y) {
  const unsigned short* xr = x + (size_t)blockIdx.x * 2560;
  const unsigned short* rr = r + (size_t)blockIdx.x * 10240;
  unsigned short* yr = y + (size_t)blockIdx.x * 10240;
  for (unsigned i = threadIdx.x; i < 10240u; i += blockDim.x) yr[i] = f2bf_rne(bf2f(rr[i]) + bf2f(xr[i % 2560u]));
}
