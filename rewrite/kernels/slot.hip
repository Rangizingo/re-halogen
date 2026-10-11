// Rewrites of the "slot" family (sum of the 10 expert/slot rows) from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o slot.co slot.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

// The vendor does v_mul_f32 then v_add_f32 (no fma) except in the last step of k_slot_moe_sum_rows.
#pragma clang fp contract(off)

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// f32 -> bf16 RNE, no NaN special case: v_bfe_u32 ; v_add3_u32 0x7fff (codeobj_53000.s:49527-49529).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// k_slot_sum(float const*, unsigned short*)
// _ZN7halogen12_GLOBAL__N_110k_slot_sumEPKfPt, kernarg: in@0 (f32), out@8 (bf16)
// i = blockIdx.x*blockDim.x + threadIdx.x (:49434), exit if i >= 2560 (0xa00, :49436); row = blockIdx.y (s3, :49440).
// in is f32 [row*10 + k][2560]: slot stride 0x2800 bytes (:49453). Sum k = 0..9 in order starting from +0.0
// (v_add_f32 v4,0,v12 at :49503, then :49506-49526), bf16 RNE, out[row*2560 + i] (row stride 0x1400 bytes, :49456-49458).
extern "C" __global__ void k_slot_sum(const float* in, unsigned short* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 2560) return;
  int row = blockIdx.y;
  float s = 0.0f;
  for (int k = 0; k < 10; k++) s = s + in[((long)row * 10 + k) * 2560 + i];
  out[(long)row * 2560 + i] = f2bf_rne(s);
}

// k_slot_sum_rows(unsigned short const*, int const*, float const*, unsigned short*)
// _ZN7halogen12_GLOBAL__N_115k_slot_sum_rowsEPKtPKiPKfPt, kernarg: tab@0 (bf16), ids@8, w@16, out@24
// Same index and guard (:49540-49542). For k = 0..9: id = ids[row*10+k], w = w[row*10+k] (s_load at :49558, :49564 ...),
// t_k = w * bf16(tab[id*2560 + i]) (row stride 0x1400 bytes, :49563-49568), sum from +0.0 in order (:49639-49657:
// terms 0..8 chained, term 9 is multiplied last then added), bf16 RNE, store out[row*2560 + i] (:49662-49666).
extern "C" __global__ void k_slot_sum_rows(const unsigned short* tab, const int* ids, const float* w, unsigned short* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 2560) return;
  int row = blockIdx.y;
  float s = 0.0f;
  for (int k = 0; k < 10; k++) {
    int id = ids[row * 10 + k];
    float t = w[row * 10 + k] * bf2f(tab[(long)id * 2560 + i]);
    s = s + t;
  }
  out[(long)row * 2560 + i] = f2bf_rne(s);
}

// k_slot_moe_sum_rows(tab, ids, w, shared, gate, out)
// _ZN7halogen12_GLOBAL__N_119k_slot_moe_sum_rowsEPKtPKiPKfS2_S6_Pt, kernarg: tab@0 (bf16), ids@8, w@16 (f32),
// shared@24 (bf16), gate@32 (f32, one per row), out@40 (bf16)
// Each thread owns 8 consecutive elements: e = (blockIdx.x*blockDim.x + threadIdx.x)*8 (v_add_lshl_u32 ..., 3, :50181),
// exit if e >= 2560 (:50184). row = blockIdx.y (s12 = s3, :50175).
// acc[j] = sum over k = 0..9 of w[row*10+k] * bf16(tab[ids[row*10+k]*2560 + e + j]) with mul then add, no fma
// (:50222-50241; 16-byte load :50220, row stride 0x1400 bytes :50213).
// acc is rounded to bf16 and widened again (v_add3_u32 ... :50262-50271, v_and 0xffff0000 :50274-50283), then
// r = fma(gate[row], bf16(shared[row*2560 + e + j]), acc) (v_fmac_f32 with the scalar gate, :50286-50301),
// rounded to bf16 RNE (:50292-50320) and stored as one 16-byte write at out[row*2560 + e] (:50328).
extern "C" __global__ void k_slot_moe_sum_rows(const unsigned short* tab, const int* ids, const float* w,
                                               const unsigned short* shared, const float* gate, unsigned short* out) {
  int e = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (e >= 2560) return;
  int row = blockIdx.y;
  float acc[8];
  for (int j = 0; j < 8; j++) acc[j] = 0.0f;
  for (int k = 0; k < 10; k++) {
    int id = ids[row * 10 + k];
    float wk = w[row * 10 + k];
    for (int j = 0; j < 8; j++) {
      float t = wk * bf2f(tab[(long)id * 2560 + e + j]);
      acc[j] = acc[j] + t;
    }
  }
  float g = gate[row];
  for (int j = 0; j < 8; j++) {
    float a = bf2f(f2bf_rne(acc[j]));
    float r = __builtin_fmaf(g, bf2f(shared[(long)row * 2560 + e + j]), a);
    out[(long)row * 2560 + e + j] = f2bf_rne(r);
  }
}
