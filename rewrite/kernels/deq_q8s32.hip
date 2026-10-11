// Rewrite of k_deq_q8s32 (flash_serve, gfx1151): q8s32 weights -> bf16.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o deq_q8s32.co deq_q8s32.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git), symbol 0x228200 at :17234. That listing prints some VOP3
// instructions as .long pairs (the 64-bit compares and shifts of the index math); those were read from a fresh
// ROCm-23 llvm-objdump of the same symbol, cited by address. Everything else is cited as codeobj_53000.s:<line>.
#include <hip/hip_runtime.h>

// f16 -> f32 is exact (the f16 operand of v_fma_mix_f32).
__device__ static inline float h2f(unsigned short u) { return (float)__builtin_bit_cast(_Float16, u); }

// f32 -> bf16 RNE without a NaN special case: v_bfe_u32 x,v,16,1 ; v_add3_u32 v,v,x,0x7fff ; take the high half
// (codeobj_53000.s:17411 bfe_u32, :17437 add3; 32 of each in the kernel).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// halogen::(anonymous namespace)::k_deq_q8s32(unsigned char const*, unsigned short*, unsigned int, unsigned int, long)
// kernarg: src@0 and dst@8 (s_load_b128 s[4:7], :17250), a@16 and b@20 (s_load_b64 s[8:9],0x10, :17236),
// c@24 (s_load_b64 s[2:3],0x18, :17244). Hidden block_count_x@0x20 (s_load_b32 s12,0x20, :17246).
// Unit = one 32-byte block of int8 plus one f16 scale. a = blocks per row, b = total blocks, c = K (elements per row).
//   i = blockIdx.x*256 + tid (v_lshl_or_b32 v2,s2,8,v0, :17237), guard i < b (v_cmpx_gt_u32 s9,v2, :17241),
//   grid-stride loop, stride = block_count_x<<8 (s_lshl_b32 s12,s12,8 :17270; v_add_nc_u32 v2,s12,v2 :17304; back edge :17544).
//   row = i / a, blk = i % a: v_rcp_iflag + v_mul_hi_u32 (:17275) + two compare-and-add correction steps, i.e. exact unsigned division.
//   Row stride = pad16(K + K/16) bytes: s_ashr_i64 .,4 on K plus a sign fix (:17261), +15 (:17265), and -16 (:17268).
//   int8 at src + row*stride + blk*32 (v_lshlrev_b64 .,5 :17312 then two global_load_b128, offsets 0 and 16: :17325-17326);
//   f16 scale at src + row*stride + K + blk*2 (global_load_d16_b16, :17324).
//   out[j] = bf16_rne( (float)(int8)q[j] * (float)scale ), j = 0..31 in byte order: v_bfe_i32 .,0/8/16,8 and v_ashrrev_i32 24
//   (:17345 first), v_cvt_f32_i32 (:17365 first), v_fma_mix_f32 v, h, v, neg(0), op_sel_hi:[1,0,0] (:17386 first), i.e. scale*v + (-0.0),
//   which equals the exact product; 32 of them. Stored as 4 x global_store_b128 at dst + (row*K + blk*32)*2 bytes
//   (v_lshlrev_b64 .,6 :17332; stores :17539-17542).
extern "C" __global__ void __launch_bounds__(256)
k_deq_q8s32(const unsigned char* src, unsigned short* dst, unsigned a, unsigned b, long c) {
  long rowbytes = (c + c / 16 + 15) & -16L;
  for (unsigned i = blockIdx.x * 256u + threadIdx.x; i < b; i += gridDim.x * 256u) {
    unsigned row = i / a, blk = i % a;
    const unsigned char* rp = src + (long)row * rowbytes;
    float s = h2f(*(const unsigned short*)(rp + c + 2L * blk));
    const signed char* q = (const signed char*)(rp + 32L * blk);
    unsigned short* o = dst + (long)row * c + 32L * blk;
    for (int j = 0; j < 32; j++) o[j] = f2bf_rne((float)q[j] * s);
  }
}
