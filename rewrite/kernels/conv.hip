// Rewrite of the "conv" family (k_conv_state_from_prefill) from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o conv.co conv.hip
// Cites are addresses in codeobj_53000 (repo listing kernels-disasm/codeobj_53000.s, same addresses).
#include <hip/hip_runtime.h>

// k_conv_state_from_prefill: _ZN7halogen12_GLOBAL__N_125k_conv_state_from_prefillEPKtS2_Pti
// kernarg: xin@0 (s[4:5]), prev@8 (s[6:7], may be null: s_cmp_lg_u64 s[6:7],0 @23B754), out@16 (s[8:9]), n@24 (s2 @23B73C).
// 1D grid, c = blockIdx.x*blockDim.x + tid, guard c < 0x2800 (@23B718-23B72C); blockDim from kernarg 0x2c.
// out[c*4 + k] = X(n-4+k), k = 0..3, one global_store_b16 each at byte offsets 0,2,4,6 of out + 8c (@23B82C, 23B8D0, 23B988, 23B9F4).
// X(j) = j >= 0 ? xin[j*10240 + c] : (prev ? prev[(j+3)*10240 + c] : 0):
//   slot0: n >= 4 ? xin row n-4 : prev row n-1       (@23B7A8 s10 = n-4 ; @23B774 s0 = n-1)
//   slot1: n >= 3 ? xin row n-3 : prev row n         (@23B874 ; @23B844)
//   slot2: n >= 2 ? xin row n-2 : prev row n+1       (@23B92C ; @23B900)
//   slot3: n >= 1 ? xin row n-1 : prev row n+2       (@23B9B8 s0 = n-1 ; @23BA1C s2 = n+2)
// row stride 0x5000 bytes (v_mad_u64_u32 0x5000, @23B7E8). A missing prev gives bf16 0 (v_mov_b16 v0.l, 0 @23B7C4).
#define NCH 10240
extern "C" __global__ void k_conv_state_from_prefill(const unsigned short* xin, const unsigned short* prev,
                                                     unsigned short* out, int n) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= NCH) return;
  for (int k = 0; k < 4; ++k) {
    int j = n - 4 + k;
    unsigned short v = 0;
    if (j >= 0) v = xin[(long)j * NCH + c];
    else if (prev) v = prev[(long)(j + 3) * NCH + c];
    out[c * 4 + k] = v;
  }
}
