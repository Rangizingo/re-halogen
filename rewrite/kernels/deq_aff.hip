// Rewrite of the "deq_aff" family from codeobj_53000 (flash_serve, gfx1151): k_deq_aff<BITS, PAIR>.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o deq_aff.co deq_aff.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). Instances (line ranges):
//   <5,true>  82461-82865   <5,false> 82865-83247   <4,true> 83247-83568   <4,false> 83568-83872
// The listing shows VOP3 instructions as a ".long <dword1>" line plus a mis-decoded second dword; decoded by hand.
// Cite note: kernel start/end lines in codeobj_53000.s are exact (see rewrite/reports/g22.md). Line numbers inside a kernel that are written
// as :NNNNN were read off a filtered re-listing; for the long unrolled kernels they can be off by tens of lines. Instruction names are exact.
#include <hip/hip_runtime.h>

// Keep float multiply and add separate unless an explicit __builtin_fmaf is written (hipcc fuses a*b+c by default).
#pragma clang fp contract(off)

// halogen::(anonymous namespace)::AffDev, 48 bytes by value at kernarg 16..63 (the kernels s_load it with
// offsets 0x10, 0x20: codeobj_53000.s:83570, :83579-83587, :82467-82473). Fields, all byte offsets from `src`
// except the two strides (inferred from use):
//   o[0] nibble plane base                    (s_load_b64 kernarg+0x10, :83580)
//   o[1] 5th-bit plane base (BITS=5 only)     (s_load_b256 kernarg+0, s10/s11, :82466 region)
//   o[2] scale base: PAIR=false {f16 scale, f16 offset} u32 per group; PAIR=true u16 {scale byte, min byte} per group
//   o[3] stride in bytes of that table per row
//   o[4] PAIR=true only: u32 {f16 d, f16 dmin} per 8 groups, base
//   o[5] stride in bytes of that table per row
struct AffDev { unsigned long long o[6]; };

__device__ static inline float h2f(unsigned h) { return (float)__builtin_bit_cast(_Float16, (unsigned short)h); }

// f32 -> bf16, round to nearest even without a NaN special case: v_bfe_u32 v6,v7,16,1 ; v_add3_u32 v6,v7,v6,0x7fff
// then the high half is kept (codeobj_53000.s:83731, :83762). Pairs are packed low = even element, high = odd
// (v_and_or_b32 v3, 0xffff0000, odd, even>>16 :83783) and written with four global_store_b128 (:83863-83866).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// kernarg: src@0, dst@8, aff@16 (48 B), a@64 (groups per row), b@68 (total groups), c@72 (long, dst row stride in elements)
// Thread i = blockIdx.x*256 | tid (v_lshl_or_b32 v2, s2, 8, v0 :83571); runs while i < b (v_cmpx_gt_u32 :83575) with
// a grid-stride step of block_count_x << 8 (s_load_b32 kernarg+0x50 :83583, s_lshl_b32 s8,s18,8 :83604).
// q = i / a (float-reciprocal unsigned divide with two fixups, :83582-83625), r = i - q*a (:83631).
// Group (q,r) = 32 elements:
//   nibble words: 16 B at src + o[0] + q*(c/2) + r*16 (s_ashr_i64 c/2 :83598, v_mad_u64_u32 :83628, v_lshlrev 4 :83635,
//     global_load_b128 :83645). Element e = nibble (e&7) of word e>>3 (v_and 15 / v_bfe_u32 4,4 ... v_lshrrev 28, :83651-83694).
//   BITS=5: plane word at src + o[1] + q*(c/8) + r*4 (c/8 via sign bias + ashr 3, :82469-82474 region); element e gets
//     ((plane >> e) & 1) << 4 or-ed in (v_and_or_b32 vN, plane<<k, 16, nibble, g22_a5f listing: elem 0 plane<<4, elem 31 plane>>27).
//   PAIR=false: u32 at src + o[2] + o[3]*q + r*4: lo = f16 scale, hi = f16 offset;
//     value = fma(float(nib), scale, offset)  (v_fma_mix_f32 v7, v7, v11, v11 op_sel:[0,0,1] op_sel_hi:[0,1,1] :83699).
//   PAIR=true: u16 at src + o[2] + o[3]*q + r*2 (global_load_u16): sc = low byte, m = high byte (v_cvt_f32_ubyte0 /
//     v_cvt_f32_ubyte1, <4,true> listing: v_cvt_f32_ubyte0 v41,v12; v_cvt_f32_ubyte1 v12,v12);
//     u32 at src + o[4] + o[5]*q + ((r>>1) & -4) (v_ashrrev_i32 1 ; v_and -4): d = low f16, dmin = high f16;
//     v41 = fma_mix(d, float(sc), -0.0) ; v42 = fma_mix(dmin, -float(m), -0.0)  (v_fma_mix_f32 v41,v10,v41,neg(0) and
//     v_fma_mix_f32 v42,v10,-v12,neg(0) op_sel:[1,0,0], the third operand is the inline constant 0 with the neg
//     modifier, read here as -0.0) ; value = fma(float(nib), v41, v42)  (v_fma_f32 v7,v7,v41,v42).
//   Output: dst[q*c + r*32 + e] = bf16_rne(value) (v_mad_u64_u32 v[7:8], s2, q :83652, +r*32 elements :83661).
template <int BITS, bool PAIR>
__device__ static inline void deq_aff_body(const unsigned char* src, unsigned short* dst, AffDev aff, unsigned a,
                                           unsigned b, long c) {
  unsigned stride = gridDim.x << 8;
  for (unsigned i = (blockIdx.x << 8) | threadIdx.x; i < b; i += stride) {
    unsigned q = i / a;
    unsigned r = i - q * a;
    const unsigned* nw = (const unsigned*)(src + aff.o[0] + (unsigned long long)(c / 2) * q + (unsigned long long)r * 16);
    unsigned w[4] = {nw[0], nw[1], nw[2], nw[3]};
    unsigned plane = 0;
    if (BITS == 5) plane = *(const unsigned*)(src + aff.o[1] + (unsigned long long)(c / 8) * q + (unsigned long long)r * 4);
    float sc_a = 0, sc_b = 0;
    if (PAIR) {
      unsigned short u = *(const unsigned short*)(src + aff.o[2] + aff.o[3] * q + (unsigned long long)r * 2);
      unsigned dm = *(const unsigned*)(src + aff.o[4] + aff.o[5] * q + (unsigned long long)((r >> 1) & ~3u));
      float scb = (float)(u & 0xffu), mb = (float)(u >> 8);
      sc_a = __builtin_fmaf(h2f(dm & 0xffffu), scb, -0.0f);
      sc_b = __builtin_fmaf(h2f(dm >> 16), -mb, -0.0f);
    } else {
      unsigned p = *(const unsigned*)(src + aff.o[2] + aff.o[3] * q + (unsigned long long)r * 4);
      sc_a = h2f(p & 0xffffu);
      sc_b = h2f(p >> 16);
    }
    unsigned short* o = dst + (long)q * c + (long)r * 32;
#pragma unroll
    for (int e = 0; e < 32; e++) {
      unsigned nib = (w[e >> 3] >> (4 * (e & 7))) & 15u;
      if (BITS == 5) nib |= ((plane >> e) & 1u) << 4;
      float v = __builtin_fmaf((float)nib, sc_a, sc_b);
      o[e] = f2bf_rne(v);
    }
  }
}

#define DEQ_AFF(NAME, BITS, PAIR)                                                                              \
  extern "C" __global__ void NAME(const unsigned char* src, unsigned short* dst, AffDev aff, unsigned a,      \
                                  unsigned b, long c) {                                                       \
    deq_aff_body<BITS, PAIR>(src, dst, aff, a, b, c);                                                         \
  }

DEQ_AFF(k_deq_aff_5t, 5, true)    // _ZN7halogen12_GLOBAL__N_19k_deq_affILi5ELb1EEEvPKhPtNS0_6AffDevEjjl
DEQ_AFF(k_deq_aff_5f, 5, false)   // _ZN7halogen12_GLOBAL__N_19k_deq_affILi5ELb0EEEvPKhPtNS0_6AffDevEjjl
DEQ_AFF(k_deq_aff_4t, 4, true)    // _ZN7halogen12_GLOBAL__N_19k_deq_affILi4ELb1EEEvPKhPtNS0_6AffDevEjjl
DEQ_AFF(k_deq_aff_4f, 4, false)   // _ZN7halogen12_GLOBAL__N_19k_deq_affILi4ELb0EEEvPKhPtNS0_6AffDevEjjl
