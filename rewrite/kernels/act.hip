// Rewrites of the "act" family (activation statistics: sum of squares and sum of |x| per column) from codeobj_53000
// (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o act.co act.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

// The vendor accumulates the squares with v_fmac_f32 and the absolute values with v_add_f32; nothing else is fused.
#pragma clang fp contract(off)

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// k_act_dense(unsigned short const* x, int k, int d, float* sq, float* l1)
// _ZN7halogen12_GLOBAL__N_111k_act_denseEPKtiiPfS3_, kernarg: x@0, k@8, d@12, sq@16, l1@24
// Column i = blockIdx.x*blockDim.x + threadIdx.x (v_mad_u64_u32, codeobj_53000.s:19214), guard i < d (:19216).
// For r = 0 .. k-1 (loop count s4, :19220-19252): v = bf16(x[r*d + i]) (row stride d*2 bytes, :19238-19239);
//   sqacc = fma(v, v, sqacc) (v_fmac_f32, :19249); l1acc = l1acc + |v| (v_add_f32 with abs modifier, .long :19250, inferred).
// If k <= 0 both sums are 0. Then sq[i] = sqacc + sq[i]; l1[i] = l1acc + l1[i] (:19265-19270, load/store both buffers).
extern "C" __global__ void k_act_dense(const unsigned short* x, int k, int d, float* sq, float* l1) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d) return;
  float s2 = 0.0f, s1 = 0.0f;
  for (int r = 0; r < k; r++) {
    float v = bf2f(x[(long)r * d + i]);
    s2 = __builtin_fmaf(v, v, s2);
    s1 = s1 + fabsf(v);
  }
  sq[i] = s2 + sq[i];
  l1[i] = s1 + l1[i];
}

// k_act_rows(unsigned short const* x, int d, int const* ptr, int const* base, int const* list, int mode, float* sq, float* l1)
// _ZN7halogen12_GLOBAL__N_110k_act_rowsEPKtiPKiS4_S4_iPfS5_, kernarg: x@0, d@8, ptr@16, base@24, list@32, mode@40, sq@48, l1@56
// One block per output row (s2 = blockIdx.x). (s, e) = ptr[row], ptr[row+1] (s_load_b64, :19291); b = base[row] (:19292).
// Columns i = tid, tid+blockDim, ... < d (:19332). For k = s .. e-1 (:19302, :19370):
//   source row sr = mode == 0 ? k : mode == 1 ? list[k] : list[k] / 10   (mode tests :19381-19410; /10 is the 0x66666667
//   multiply with ashr 2, :19406-19410; list pointer advanced from list + s*4, :19307-19312)
//   v = bf16(x[sr*d + i]) (:19371-19373); sqacc = fma(v, v, sqacc) (:19375); l1acc += |v| (.long :19376).
// Output index b*d + i (s15*s12, :19311): sq[o] = sqacc + sq[o]; l1[o] = l1acc + l1[o] (:19328-19336).
extern "C" __global__ void k_act_rows(const unsigned short* x, int d, const int* ptr, const int* base, const int* list,
                                      int mode, float* sq, float* l1) {
  int row = blockIdx.x;
  int s = ptr[row], e = ptr[row + 1];
  long b = base[row];
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    float s2 = 0.0f, s1 = 0.0f;
    for (int k = s; k < e; k++) {
      int sr = mode == 0 ? k : (mode == 1 ? list[k] : list[k] / 10);
      float v = bf2f(x[(long)sr * d + i]);
      s2 = __builtin_fmaf(v, v, s2);
      s1 = s1 + fabsf(v);
    }
    long o = b * d + i;
    sq[o] = s2 + sq[o];
    l1[o] = s1 + l1[o];
  }
}
