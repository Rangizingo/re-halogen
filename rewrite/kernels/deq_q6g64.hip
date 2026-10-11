// Rewrite of k_deq_q6g64 (planar 6-bit affine, 64-element groups -> bf16) from codeobj_53000 (flash_serve, gfx1151).
// Listing: kernels-disasm/codeobj_53000.s (not in git); slice work/kernels-dequant/*k_deq_q6g64EPKhPtll.s, starts at 0x22b800.
// Line numbers below are slice line numbers.
#include <hip/hip_runtime.h>
// hipcc fuses a*b+c by default; the listing uses separate v_mul/v_add (explicit fmaf where fused).
#pragma clang fp contract(off)

// f32 -> bf16 RNE, no NaN special case: v_bfe_u32 x,16,1 ; v_add3_u32 x,x,b,0x7fff (:486-515); packed with
// v_and_or_b32 0xffff0000 (odd element high half) | (even element >> 16) (:507-514).
__device__ static inline unsigned f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (u + ((u >> 16) & 1u) + 0x7fffu) >> 16;
}

// k_deq_q6g64(const u8* src, u16* dst, long rows, long K)   kernarg: src@0 dst@8 rows@16 K@24
//   One thread = 32 elements. Threads per row P = K/32 (:6-10 s_lshr 27, ashr 5); total = P*rows (:11-15, bound check :16-20).
//   t = blockIdx.x<<8 | tid (:9,:11); row = t / P, c = t % P (64-bit divide, :22-160 with a 32-bit fast path :157-176).
//   Row layout (planar), row stride S = (K/2 + K/4 + K/16 + 15) & ~15 (:177-197: s0=K>>1, s8=K>>2, s2=K>>4, add 15, and -16):
//     ql plane at +0, K/2 bytes: thread c reads 16 bytes at c*16 (b128, :224)
//     qh plane at +K/2, K/4 bytes: thread c reads 8 bytes at c*8 (b64, :234)
//     sb at +K/2+K/4: one dword per 64 elements at 4*(c>>1) ((c*2)&-4, :235-240): scale f16 low half, bias f16 high half.
//   Element e (0..31): lo = nibble e of the 16 ql bytes (dword e/8, shift 4*(e%8); :243-261 v_and/v_bfe ..,4),
//     hi = bits 2e..2e+1 of the 64-bit qh ((qh<<4 | ...)&48 patterns, :270-310), q = hi*16 + lo in 0..63 (v_and_or ..,48 ; no -32 bias).
//   value = fma(float(q), float(scale), float(bias)) as v_fma_mix_f32 q, sb, sb op_sel:[0,0,1] op_sel_hi:[0,1,1] (:305): single rounding.
//   Output dst + row*K + c*32, 4 x b128 (:517-520).
extern "C" __global__ void k_deq_q6g64(const unsigned char* src, unsigned short* dst, long rows, long K) {
  long P = K / 32;
  long t = (long)blockIdx.x * 256 + threadIdx.x;
  if (t >= P * rows) return;
  long row = t / P, c = t % P;
  long S = (K / 2 + K / 4 + K / 16 + 15) & ~15L;
  const unsigned char* base = src + row * S;
  unsigned ql[4];
  unsigned long long qh;
  unsigned sb;
  __builtin_memcpy(ql, base + c * 16, 16);
  __builtin_memcpy(&qh, base + K / 2 + c * 8, 8);
  __builtin_memcpy(&sb, base + K / 2 + K / 4 + ((c * 2) & ~3L), 4);
  float scale = (float)__builtin_bit_cast(_Float16, (unsigned short)(sb & 0xffffu));
  float bias = (float)__builtin_bit_cast(_Float16, (unsigned short)(sb >> 16));
  unsigned o[16];
  #pragma unroll
  for (int e = 0; e < 32; e++) {
    unsigned lo = (ql[e >> 3] >> (4 * (e & 7))) & 15u;
    unsigned hi = (unsigned)(qh >> (2 * e)) & 3u;
    float v = __builtin_fmaf((float)(int)((hi << 4) | lo), scale, bias);
    unsigned b = f2bf_rne(v);
    if (e & 1) o[e >> 1] |= b << 16; else o[e >> 1] = b;
  }
  unsigned* d = (unsigned*)(dst + row * K + c * 32);
  #pragma unroll
  for (int i = 0; i < 16; i++) d[i] = o[i];
}
