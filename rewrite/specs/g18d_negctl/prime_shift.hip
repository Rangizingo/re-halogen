// Priming kernel for attn_qs<2,0> and attn_qs<3,0> (g18d).
// The vendor kernel's PV product reads 48 KiB of LDS it never writes (codeobj_1513000.s ds_load_b128 at
// w*0x3000 + 0x800 + 48*col + 768*c + {0,16}, see rewrite/reports/g18c.md). The result is only defined after
// this kernel has filled that LDS. Launched by run_kernel ("prime" in the spec) right before the measured kernel,
// 1000 blocks x 256 threads so every CU's LDS gets the pattern, with the vendor's LDS size.
// Pattern: halfword i of LDS = bf16 0x3f80 + 8*(i&15). The rewrite models it in qs_full (attn_qs_c.hip):
// vb[k] = pattern[(8*(col&1)+k)&15].
#include <hip/hip_runtime.h>
extern "C" __global__ void prime() {
  extern __shared__ unsigned short L[];
  volatile unsigned short* v = L;
  for (int i = threadIdx.x; i < 32170; i += blockDim.x) v[i] = (unsigned short)(0x3f80 + 8 * ((i + 1) & 15));
  __syncthreads();
}
