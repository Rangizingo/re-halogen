// Rewrite of the "hist" family (flash_serve, gfx1151): k_hist_from_cstate.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o hist.co hist.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). The listing prints 64-bit address math as
// ".long 0xd73c...." / ".long 0xd700...." lines that llvm-objdump rejects (src2 is inline 0 on a
// two-source op). 0xd73c.... is v_lshlrev_b64 and 0xd700.... is v_add_co_u32; the meaning below
// follows the loads and stores.
#include <hip/hip_runtime.h>

// halogen::(anonymous namespace)::k_hist_from_cstate(unsigned short const*, unsigned short*)
// _ZN7halogen12_GLOBAL__N_118k_hist_from_cstateEPKtPt, kernarg: in@0, out@8
// i = blockIdx.x*blockDim.x + threadIdx.x (v_mad_u64_u32, codeobj_53000.s:30213), guard i < 0x2800 signed (v_cmpx_gt_i32, :30215).
//   in+8i+2 : global_load_b32 gives in[4i+1] (lo) and in[4i+2] (hi)   (:30234)
//   in+8i+6 : global_load_d16_b16 gives in[4i+3]                      (:30235)
//   out[i] = in[4i+1] (:30245), out[0x2800+i] = in[4i+2] (hi half, :30246; +0x5000 bytes at :30237),
//   out[0x5000+i] = in[4i+3] (:30248; +0xa000 bytes at :30240)
extern "C" __global__ void k_hist_from_cstate(const unsigned short* in, unsigned short* out) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if ((int)i >= 0x2800) return;  // the vendor compares the low 32 bits as a signed int
  out[i] = in[4 * i + 1];
  out[0x2800 + i] = in[4 * i + 2];
  out[0x5000 + i] = in[4 * i + 3];
}
