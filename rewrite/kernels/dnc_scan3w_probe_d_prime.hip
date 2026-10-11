// Priming kernel for the g19d check of k_dnc_scan3w<2,4,true,false,5> (codeobj_1146000, group_segment 46848 B).
// Fills the vendor kernel's whole LDS size with f16 1.0 (0x3c00). If P5 still equals the rewrite (A = 0) after this, the
// vendor kernel writes its own k*e table region (zero fill, codeobj_1146000.s:176798-176823) instead of reading leftover LDS.
// Launched by run_kernel ("prime" in the spec) right before the measured kernel: 1000 blocks x 256 threads, dynamic LDS 46848.
#include <hip/hip_runtime.h>
extern "C" __global__ void prime() {
  extern __shared__ unsigned short L[];
  volatile unsigned short* v = L;
  for (int i = threadIdx.x; i < 23424; i += blockDim.x) v[i] = 0x3c00;
  __syncthreads();
}
