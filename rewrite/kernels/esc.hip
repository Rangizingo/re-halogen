// Rewrite of the "esc" family from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o esc.co esc.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

// Keep float multiply and add separate unless an explicit __builtin_fmaf is written (hipcc fuses a*b+c by default).
#pragma clang fp contract(off)

// halogen::AmxMasks, passed by value (64 bytes at kernarg 24..87). The kernel reads the low dword of m[row]
// with s_load_b32 at kernarg + 0x18 + row*8 (codeobj_53000.s:51884-51887), so m is eight 8-byte masks.
struct AmxMasks { unsigned long long m[8]; };

// halogen::(anonymous namespace)::k_esc_arbiter(float const*, int, int*, halogen::AmxMasks, int const*)
// _ZN7halogen12_GLOBAL__N_113k_esc_arbiterEPKfiPiNS_8AmxMasksEPKi
// kernarg: x@0, n@8, out@16, masks@24 (64 B), list@88. Block (256), LDS 2048 B, one block per row.
// row = blockIdx.x (s2). Return unless bit 0 of masks.m[row] is set (s_bitcmp0_b32 s4,0 :51889-51890).
// Per thread (:51897-51942): best = -inf (0xff800000, :51892), bi = INT_MAX (v_bfrev_b32 -2, :51893);
//   for j = tid; j < n; j += 256 (v_add 0x100 :51936): v = x[row*n + j];
//   take it when (v == best && j < bi) || v > best  (v_cmp_eq_f32 :51926, v_cmp_gt_f32 :51928, v_cmp_lt_i32 :51924,
//   s_and/s_or :51930-51932, v_cndmask :51935). NaN never wins; an all -inf row picks its first index through the
//   eq path.
// LDS: idx at tid*4, val at 0x400 + tid*4 (ds_store_2addr_stride64 offset1:4 :51948).
// Tree over strides 128,64,32,16,8,4,2,1 (:51952-52141): thread t < s reads (val, idx) of t+s and replaces its own
//   when (vo == v && io < i) || vo > v (:51957-51965 for stride 128, same shape each step), barrier between steps.
// Thread 0 (:52145-52185): idx = lds idx[0]; if idx equals any of list[0..3] (:52151-52176) then out[row] = list[4].
extern "C" __global__ void k_esc_arbiter(const float* x, int n, int* out, AmxMasks masks, const int* list) {
  __shared__ int sidx[256];
  __shared__ float sval[256];
  int row = blockIdx.x, tid = threadIdx.x;
  if (!(masks.m[row] & 1ull)) return;
  float best = __uint_as_float(0xff800000u);
  int bi = 0x7fffffff;
  for (int j = tid; j < n; j += 256) {
    float v = x[(long)row * n + j];
    if ((v == best && j < bi) || v > best) { best = v; bi = j; }
  }
  sidx[tid] = bi;
  sval[tid] = best;
  __syncthreads();
  for (int s = 128; s >= 1; s >>= 1) {
    if (tid < s) {
      float vo = sval[tid + s], v = sval[tid];
      int io = sidx[tid + s], i = sidx[tid];
      if ((vo == v && io < i) || vo > v) { sval[tid] = vo; sidx[tid] = io; }
    }
    __syncthreads();
  }
  if (tid == 0) {
    int idx = sidx[0];
    if (idx == list[0] || idx == list[1] || idx == list[2] || idx == list[3]) out[row] = list[4];
  }
}
