// Rewrite of the "attn_fa" family: halogen::(anonymous)::k_attn_fa (codeobj_1513000, flash_serve, gfx1151), the FA1-style
// predecessor of k_attn_fa2 (listing kernels-disasm/codeobj_1513000.s:6-8276, not in git).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o attn_fa.co attn_fa.hip
// The wave-level algorithm is the same as the k_attn_fa2 instances, so this file reuses fa_core from attn_fa2.hip (same
// directory); see the comments there for the listing cites of each step.
#include "attn_fa2.hip"

// E for this kernel (1 = exp2 path with the log2e-folded scale, 0 = expf path); chosen by test, see rewrite/reports/g07.md.
#ifndef FA_E
#define FA_E 0
#endif
// FA_MM: 1 masks keys >= m (as the fa2 kernels do), 0 leaves them to the clamped row loads (key tile rows duplicate row m-1).
#ifndef FA_MM
#define FA_MM 0  /* the vendor does not mask keys >= m (confirmed on k_attn_fa with sel[0] = 13, 26; same assumed for fa2, tested below) */
#endif

// k_attn_fa(Q@0, K@8, V@16, stride@24, gate@32, O@40, sel@48, n@56, m@60): grid (tile, head 0..23), block 32 = one wave, no FaSel.
extern "C" __global__ void k_attn_fa(const unsigned short* Q, const unsigned short* K, const unsigned short* V, long stride,
                                     const unsigned short* gate, unsigned short* O, const int* sel, int n, int m) {
  fa_core<FA_E, FA_MM>(Q, K, V, stride, gate, O, sel[0], n, m, blockIdx.y / 12, blockIdx.y, blockIdx.x * 16, 0);
}
