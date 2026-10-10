// Rewrite of k_hc_di_epi (SiLU on 320 columns, 2*sigmoid on 4 columns) from flash_serve, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o hc_di.co hc_di.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
__device__ static inline unsigned short f2bf_rne(float f) {  // :33354-33358
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// halogen::(anonymous namespace)::k_hc_di_epi(unsigned short const*, unsigned short*, float*, long, float)
// _ZN7halogen12_GLOBAL__N_111k_hc_di_epiEPKtPtPflf, kernarg: x@0, y@8, z@16, rows@24, scale@32.
// idx = blockIdx.x*blockDim.x + tid as 64-bit (:33245), guard idx < rows*0x144 = rows*324 (:33246-33251).
// r = idx / 324, c = idx % 324: mulhi by 0xadd3c0cb/0x329161f9 and shift (:33254-33271), remainder via multiply-add of -324 = 0xfffffebc (:33273).
// Input row stride = 0x2a0 bytes = 336 bf16 (:33274); element x[r*336 + c] (:33283). v = bf16(x) * scale (:33285-33287).
// e = exp2(v * -1.442695) (0xbfb8aa3b, v_exp_f32 :33290-33291), d = 1 + e (:33296). Branch on c > 0x13f = 319 (:33297).
//   c >= 320: z[r*4 + (c-320)] = 2 * (1/d), IEEE division by 1.0 then v_add d,d (:33301-33331; z offset (c-0x140) = v4 :33315, row term r<<4 bytes :33303).
//   c <  320: y[r*320 + c] = bf16_rne(v / d), IEEE division (:33334-33363); y row stride 0x280 bytes = 320 bf16 (:33348).
extern "C" __global__ void k_hc_di_epi(const unsigned short* x, unsigned short* y, float* z, long rows, float scale) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * 324) return;
  long r = idx / 324;
  int c = (int)(idx - r * 324);
  float v = bf2f(x[r * 336 + c]) * scale;
  float e = __builtin_amdgcn_exp2f(v * -1.44269502162933349609375f);
  float d = 1.0f + e;
  if (c >= 320) {
    float s = 1.0f / d;
    z[r * 4 + (c - 320)] = s + s;
  } else {
    y[r * 320 + c] = f2bf_rne(v / d);
  }
}
