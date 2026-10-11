// Rewrite of k_deq_q8g64 (flash_serve, gfx1151): q8g64 weights -> bf16.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o deq_q8g64.co deq_q8g64.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git), symbol 0x227500 at :16778. That listing prints some VOP3
// instructions as .long pairs (the 64-bit index math); those were read from a fresh ROCm-23 llvm-objdump of the same
// symbol. Everything else is cited as codeobj_53000.s:<line>.
#include <hip/hip_runtime.h>

// f16 -> f32 is exact (the f16 halves of v41 in v_fma_mix_f32).
__device__ static inline float h2f(unsigned short u) { return (float)__builtin_bit_cast(_Float16, u); }

// f32 -> bf16 RNE without a NaN special case (v_bfe_u32 x,v,16,1 ; v_add3_u32 v,v,x,0x7fff ; high half: :17064, :17065 first pair).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// halogen::(anonymous namespace)::k_deq_q8g64(unsigned char const*, unsigned short*, unsigned int, unsigned int, long)
// kernarg: src@0, dst@8 (s_load_b128 s[4:7], :16794), a@16 and b@20 (s_load_b64 s[8:9],0x10, :16780), c@24 (s_load_b64 s[2:3],0x18, :16788),
// hidden block_count_x@0x20 (s_load_b32 s12,0x20, :16790).
// Unit = one 64-byte block of uint8 plus a (scale, offset) f16 pair. a = blocks per row, b = total blocks, c = K.
//   Same index scheme as k_deq_q8s32: i = blockIdx.x*256+tid (v_lshl_or_b32 :16781), guard i < b (v_cmpx_gt_u32 :16785),
//   grid-stride block_count_x<<8 (s_lshl_b32 :16811, v_add_nc_u32 :16855), row = i / a, blk = i % a (v_mul_hi_u32 :16816).
//   Row stride = K + K/16 bytes, NOT padded: s_ashr_i64 .,4 on K (:16804), then s_add_u32 s1,s10,s2 adds K (:16806); no "and -16".
//   uint8 at src + row*stride + blk*64: four global_load_b128 at offsets 0/16/32/48 (:16883-16887, one at offset 16 first);
//   (scale, offset) pair = one global_load_b32 at src + row*stride + K + blk*4 (v_lshlrev_b32 .,2 :16865; load :16885).
//   out[j] = bf16_rne( fma((float)u8[j], scale_f16, offset_f16) ), j = 0..63 in byte order: v_cvt_f32_ubyte0..3 (:16893 first),
//   then v_fma_mix_f32 v,v,v41,v41 op_sel:[0,0,1] op_sel_hi:[0,1,1] (:16959 first; 64 of them) = v * lo_half(v41) + hi_half(v41)
//   with one rounding. Stored as 8 x global_store_b128 at dst + (row*K + blk*64)*2 bytes (v_lshlrev_b64 .,7 :16866; stores :17201-17228).
extern "C" __global__ void __launch_bounds__(256)
k_deq_q8g64(const unsigned char* src, unsigned short* dst, unsigned a, unsigned b, long c) {
  long rowbytes = c + c / 16;
  for (unsigned i = blockIdx.x * 256u + threadIdx.x; i < b; i += gridDim.x * 256u) {
    unsigned row = i / a, blk = i % a;
    const unsigned char* rp = src + (long)row * rowbytes;
    unsigned sc = *(const unsigned*)(rp + c + 4L * blk);
    float s = h2f((unsigned short)(sc & 0xffffu));
    float o = h2f((unsigned short)(sc >> 16));
    const unsigned char* q = rp + 64L * blk;
    unsigned short* d = dst + (long)row * c + 64L * blk;
    for (int j = 0; j < 64; j++) d[j] = f2bf_rne(__builtin_fmaf((float)q[j], s, o));
  }
}
