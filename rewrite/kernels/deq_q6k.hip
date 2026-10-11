// Rewrite of k_deq_q6k (Q6_K -> bf16 dequant) from codeobj_53000 (flash_serve, gfx1151).
// Listing: kernels-disasm/codeobj_53000.s (not in git); slice work/kernels-dequant/*k_deq_q6kEPKhPtjjl.s, starts at 0x228b00.
// Line numbers below are slice line numbers.
#include <hip/hip_runtime.h>
// hipcc fuses a*b+c by default; the listing uses separate v_mul/v_add (explicit fmaf where fused).
#pragma clang fp contract(off)

// f32 -> bf16 RNE, no NaN special case: v_bfe_u32 x,16,1 ; v_add3_u32 x,x,b,0x7fff, high half kept (:222-256).
__device__ static inline unsigned f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (u + ((u >> 16) & 1u) + 0x7fffu) >> 16;
}

// k_deq_q6k(const u8* src, u16* dst, u32 G, u32 total, long K)   kernarg: src@0 dst@8 G@16 total@20 K@24, hidden block_count_x@32
//   Grid-stride loop (:4 v_lshl_or t = blockIdx.x<<8 | tid ; :219 t += s11 where s11 = block_count_x<<8 (:12,:32) ; :252 exit when total <= t).
//   One thread = one 16-element scale group. row = t / G, c = t % G (unsigned divide via v_rcp_iflag, :15,:35-48).
//   G is the groups per row (K/16 in use); block = c>>4 (:59 ashr 4), lane l = c&15 (:52 and 15, :50 bit 3).
//   Block base = src + row * (K/256)*0xd2 + block*0xd2 (:26-30, :54-61). Q6_K block: ql[128] qh[64] sc[16]i8 d f16.
//   Loads: sc at +192+l (global_load_i8, :85), qh b128 at 128 + half*32 + (l&1)*16 (:86), ql b128 at half*64 + (l&3)*16 (:87), d at +208 (:88).
//   Element 4k+i: ql nibble = (qlw[k] >> (8i + (l&4))) & 15 (:129-152 v_bfe_u32 ..,4 with v17|{0,8,16,24}),
//   qh bits = (qhw[k] >> (8i + (l&6))) & 3, <<4 & 48 (:112-128, :133-211 v_and_or ..,48), q = (hi|lo) - 32 (:145 v_subrev 32).
//   w = bf16_rne( (d*sc) * (float)q ): fma_mix(d, (float)sc, -0) (:102-104) then v_mul_f32 (:160-227) then RNE (:222-256).
//   Output dst + row*K + c*16 as two b128 stores (:267-268), packed pairs (even in low half).
__device__ static inline void k_deq_q6k_body(const unsigned char* src, unsigned short* dst, unsigned G, unsigned total, long K) {
  long nblk = K / 256;
  for (unsigned t = blockIdx.x * 256u + threadIdx.x; t < total; t += gridDim.x * 256u) {
    unsigned row = t / G, c = t % G;
    int l = c & 15;
    const unsigned char* blk = src + (long)row * nblk * 210 + (long)(c >> 4) * 210;
    int half = l >> 3;
    const unsigned char* qlp = blk + half * 64 + (l & 3) * 16;
    const unsigned char* qhp = blk + 128 + half * 32 + (l & 1) * 16;
    float sc = (float)(int)(signed char)blk[192 + l];
    unsigned short dh;
    __builtin_memcpy(&dh, blk + 208, 2);
    float ds = (float)__builtin_bit_cast(_Float16, dh) * sc;
    int qs = l & 4, hs = l & 6;
    unsigned o[8];
    #pragma unroll
    for (int k = 0; k < 4; k++) {
      unsigned qlw, qhw;
      __builtin_memcpy(&qlw, qlp + 4 * k, 4);
      __builtin_memcpy(&qhw, qhp + 4 * k, 4);
      unsigned wb[4];
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        int q = (int)((((qhw >> (8 * i + hs)) << 4) & 48u) | ((qlw >> (8 * i + qs)) & 15u)) - 32;
        wb[i] = f2bf_rne(ds * (float)q);
      }
      o[2 * k] = (wb[0] & 0xffffu) | (wb[1] << 16);
      o[2 * k + 1] = (wb[2] & 0xffffu) | (wb[3] << 16);
    }
    unsigned* d = (unsigned*)(dst + (long)row * K + (long)c * 16);
    #pragma unroll
    for (int i = 0; i < 8; i++) d[i] = o[i];
  }
}
extern "C" __global__ void k_deq_q6k(const unsigned char* src, unsigned short* dst, unsigned G, unsigned total, long K) {
  k_deq_q6k_body(src, dst, G, total, K);
}
