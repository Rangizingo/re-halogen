// Rewrites of the "dn_conv" family from codeobj_53000 (flash_serve, gfx1151): k_dn_conv_rows and k_dn_conv_ab_rows<R,b>.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dn_conv.co dn_conv.hip
// Cites are addresses in codeobj_53000 (repo listing kernels-disasm/codeobj_53000.s, same addresses).
#pragma clang fp contract(off)
#include <hip/hip_runtime.h>

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
__device__ static inline float silu_f(float v) {
  float e = __builtin_amdgcn_exp2f(v * -1.44269502162933349609375f);
  return v / (1.0f + e);
}
__device__ static inline float softplus_v(float v) {
  if (!(20.0f < v)) {
    float e = __builtin_amdgcn_exp2f(v * 1.44269502162933349609375f);
    float s = 1.0f + e;
    bool small = __uint_as_float(0x800000u) > s;
    s = __builtin_amdgcn_ldexpf(s, small ? 32 : 0);
    float l = __builtin_amdgcn_logf(s);
    float p = l * __uint_as_float(0x3f317217u);
    bool fin = __uint_as_float(0x7f800000u) > __builtin_fabsf(l);
    float q = __builtin_fmaf(__uint_as_float(0x3f317217u), l, -p);
    q = __builtin_fmaf(l, __uint_as_float(0x3377d1cfu), q);
    float r = p + q;
    l = fin ? r : l;
    v = l - (small ? __uint_as_float(0x41b17218u) : 0.0f);
  }
  return v;
}

// DnRows is 256 B by value (kernarg 16..271). Fields used by the conv kernels (@23C40C, @23C414: s4 = kernarg+16,
// loads at s4+0x40 and s4+0): ptr[0..7] = state write pointer per row, ptr[8..15] = state read pointer per row (null: use ptr[r]).
// The recurrent kernels read further fields (see dn_recur.hip).
struct DnRows { void* p[32]; };

// One conv step for one channel c and one row (shared by k_dn_conv_rows and the conv half of k_dn_conv_ab_rows):
// reads slots 1..3 of the state (b32 at +2, b16 at +6), x, w; fma chain from 0; silu; bf16 RNE; writes {s1,s2,s3,x}.
__device__ static inline void conv_step(const unsigned short* rd, unsigned short* wr, const unsigned short* w,
                                        unsigned short xv, unsigned short* yp, int c) {
  unsigned short s1 = rd[c * 4 + 1], s2 = rd[c * 4 + 2], s3 = rd[c * 4 + 3];
  float acc = __builtin_fmaf(bf2f(s1), bf2f(w[c * 4 + 0]), 0.0f);   // v_fma_f32 v15,v18,v9,0 (@23C4D4)
  acc = __builtin_fmaf(bf2f(s2), bf2f(w[c * 4 + 1]), acc);          // @23C4F0
  acc = __builtin_fmaf(bf2f(s3), bf2f(w[c * 4 + 2]), acc);          // @23C4F4
  acc = __builtin_fmaf(bf2f(xv), bf2f(w[c * 4 + 3]), acc);          // @23C4FC
  wr[c * 4 + 0] = s1; wr[c * 4 + 1] = s2; wr[c * 4 + 2] = s3; wr[c * 4 + 3] = xv;   // global_store_b64 @23C548
  *yp = f2bf_rne(silu_f(acc));                                       // @23C5AC-23C5C4
}

// k_dn_conv_rows: _ZN7halogen12_GLOBAL__N_114k_dn_conv_rowsEPKtS2_NS0_6DnRowsEPtiS2_S2_PKfS6_PfS7_
// kernarg: x@0, w@8, rows@16 (DnRows, 256 B), y@272, nrows@280, a_in@288, b_in@296, A_log@304, dt_bias@312, beta@320, g@328.
// Two jobs split on blockIdx.x (s_cmp_lt_i32 s2, 40 @23C308):
//  blockIdx.x < 40: conv for channel c = blockIdx.x*blockDim.x + tid (c < 0x2800, @23C330-23C340), looping rows r = 0..nrows-1
//    (s_cmp_lt_i32 nrows,1 skips, @23C344; loop @23C494/@23C5E0). Row r reads x/y at row stride 0x5000 B (v_add 0x5000 @23C46C, @23C5CC),
//    state read ptr = rows.p[8+r] ? rows.p[8+r] : rows.p[r] (s_cmp_eq_u64 @23C424), state write ptr = rows.p[r] (@23C484).
//  blockIdx.x >= 40: gates, i = (blockIdx.x-40)*blockDim.x + tid < nrows*48 (@23C600-23C62C), same math as k_dn_gates
//    (a_in@288 -> s4, b_in@296 -> s6, A_log@304 -> s8, dt_bias@312 -> s10 loaded at s_load_b256 0x120 @23C63C; beta@320 -> s8, g@328 -> s10 at 0x140 @23C6D0).
extern "C" __global__ void k_dn_conv_rows(const unsigned short* x, const unsigned short* w, DnRows rows, unsigned short* y,
                                          int nrows, const unsigned short* a_in, const unsigned short* b_in,
                                          const float* A_log, const float* dt_bias, float* beta, float* g) {
  if (blockIdx.x < 40) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 10240) return;
    for (int r = 0; r < nrows; ++r) {
      unsigned short* wr = (unsigned short*)rows.p[r];
      const unsigned short* rd = rows.p[8 + r] ? (const unsigned short*)rows.p[8 + r] : wr;
      conv_step(rd, wr, w, x[(long)r * 10240 + c], y + (long)r * 10240 + c, c);
    }
    return;
  }
  int i = (blockIdx.x - 40) * blockDim.x + threadIdx.x;
  if (i >= nrows * 48) return;
  int j = i % 48;
  float v = dt_bias[j] + bf2f(a_in[i]);
  float sp = softplus_v(v);
  float eA = __builtin_amdgcn_exp2f(A_log[j] * 1.44269502162933349609375f);
  g[i] = sp * (-eA);
  float e = __builtin_amdgcn_exp2f(bf2f(b_in[i]) * -1.44269502162933349609375f);
  beta[i] = 1.0f / (1.0f + e);
}

// ---------------------------------------------------------------------------------------------------------------
// k_dn_conv_ab_rows<R,b>: _ZN7halogen12_GLOBAL__N_117k_dn_conv_ab_rowsILi{R}ELi{b}EEEvPKtS3_NS0_6DnRowsEPtNS0_5LqGrpEiS3_l
// kernarg: x@0, w@8, rows@16 (DnRows), y@272, grp@280 (LqGrp, 296 B), nb@576 (int), xa@584 (bf16 activations of the LQ part), K@592 (long).
// Two jobs split on blockIdx.x against nb (s_cmp_ge_i32 s2, s4 @86E810 for <1,0>; listing addresses below are <1,0> at 0x86e800):
//  blockIdx.x >= nb: conv step for R rows, c = (blockIdx.x - nb)*blockDim.x + tid < 10240 (@86E820-86E840). The per-row work is the
//    same as k_dn_conv_rows (state slots 1..3 + x, fma chain from 0, silu, bf16 RNE, new state {s1,s2,s3,x}); R rows are unrolled in the
//    vendor (state ptrs rows.p[r] / rows.p[8+r], x and y at row stride 0x5000 B, @872B4C-872EF0 for <2,0>).
//  blockIdx.x < nb: a 4-bit "LQ" matrix-vector product, the same math as k_lq4w_grp (rewrite/kernels/lq4w.hip, g01 group), NC = R activation
//    columns. The LqGrp member layout is the one decoded there: blob@+0, out@+8, out2@+0x10, woff@+0x18, soff@+0x20, sstride@+0x28, R@+0x30,
//    f@+0x38, E@+0x3c, first block@+0x40 (loads @86EB60, @86EB68, @86EC18, @86F22C, @86F304, @86F310). count@568 (s_cmp_gt_i32 s3 ..., @86EADC).
//    Member pick: highest i with blockIdx.x >= start[i] (@86EAE0-86EB44). Rows R_m > 8 use 16 lanes x 1 segment, 16 rows per block;
//    R_m <= 8 use 16 lanes x 16 segments, 1 row per block (v_cmp_gt_i64 R,8 -> s22 = 1 or 16 @86EB8C-86EB98).
//    LUT: T[b] = (rne_bf16(lut[b>>4]) << 16) | rne_bf16(lut[b&15]) (@86EB78-86EBEC). Per 32-weight group: inner = 16 chained v_dot2_f32_bf16
//    over the 16 weight bytes, acc += scale*inner (v_fmac, @86F0A4); sixteen lanes (xor 8,4,2,1) summed by ds_bpermute, segments summed in
//    order from 0.0f (scalar adds @86F27C-86F338), epilogue by E exactly as k_lq4w_grp.
//  b = 1 differs from b = 0 only by preloading scales when K/32 <= 128 (s_cmpk_gt_i32 0x80 / s_cmpk_lt_i32 0x81 appear only in b = 1); the
//  numerics are the same, so one rewrite covers both and the specs use K/32 on both sides of 128.
typedef unsigned short u16;
typedef __bf16 v2bf __attribute__((ext_vector_type(2)));
struct LqGrpEnt {
  const unsigned char* blob; void* out; void* out2;
  long woff, soff, sstride, R;
  float f; int E; int start; int pad;
};
struct LqGrp {
  LqGrpEnt g[4];
  int count; int pad;
};
__device__ static inline float h2f(u16 h) { return (float)__builtin_bit_cast(_Float16, h); }
__device__ static inline unsigned rne_hi(unsigned u) { return u + ((u >> 16) & 1u) + 0x7fffu; }

template <int NC>
__device__ static void dn_conv_ab_body(const u16* x, const u16* w, DnRows rows, u16* y, const LqGrp& gp, int nb,
                                       const u16* xa, long K) {
  __shared__ unsigned tbl[256];
  __shared__ float part[8 * 16];
  const int tid = threadIdx.x;
  const int blk = blockIdx.x;
  if (blk >= nb) {
    int c = (blk - nb) * blockDim.x + tid;
    if (c >= 10240) return;
    for (int r = 0; r < NC; ++r) {
      u16* wr = (u16*)rows.p[r];
      const u16* rd = rows.p[8 + r] ? (const u16*)rows.p[8 + r] : wr;
      conv_step(rd, wr, w, x[(long)r * 10240 + c], y + (long)r * 10240 + c, c);
    }
    return;
  }
  int gi = 0;
  if (gp.count > 1 && blk >= gp.g[1].start) gi = 1;
  if (gp.count > 2 && blk >= gp.g[2].start) gi = 2;
  if (gp.count > 3 && blk >= gp.g[3].start) gi = 3;
  const unsigned char* blob = gp.g[gi].blob;
  void* out = gp.g[gi].out;
  void* out2 = gp.g[gi].out2;
  const long woff = gp.g[gi].woff, soff = gp.g[gi].soff, sstride = gp.g[gi].sstride, R = gp.g[gi].R;
  const float f = gp.g[gi].f;
  const int E = gp.g[gi].E;
  const int first = gp.g[gi].start;
  {
    const float* lut = (const float*)blob;
    tbl[tid] = (rne_hi(__float_as_uint(lut[tid >> 4])) & 0xffff0000u) | (rne_hi(__float_as_uint(lut[tid & 15])) >> 16);
  }
  __syncthreads();
  const int W = 16;
  const int Mrt = (R <= 8) ? 16 : 1;
  const int ROWS = 256 / (W * Mrt);
  const int rl = (tid / W) / Mrt;
  const int seg = (tid / W) % Mrt;
  const int lane = tid % W;
  const long row = (long)(blk - first) * ROWS + rl;
  const bool valid = row < R;
  const long crow = valid ? row : 0;
  const long khalf = (K + ((K >> 63) & 1)) >> 1;
  const long G = (K + ((K >> 63) & 31)) >> 5;
  const unsigned char* wrow = blob + woff + crow * khalf;
  const u16* srow = (const u16*)(blob + soff + crow * sstride);
  long chunk = (G + (Mrt - 1)) / Mrt;
  long g0 = chunk * seg + lane;
  long g1 = chunk * seg + chunk;
  if (G < g1) g1 = G;
  float acc[NC];
  for (int j = 0; j < NC; j++) acc[j] = 0.0f;
  for (long g = g0; g < g1; g += W) {
    const uint4 wv = *(const uint4*)(wrow + g * 16);
    const float sc = h2f(srow[g]);
    const unsigned wd[4] = {wv.x, wv.y, wv.z, wv.w};
    for (int j = 0; j < NC; j++) {
      const uint4* xp = (const uint4*)(xa + (long)j * K + g * 32);
      uint4 xv[4] = {xp[0], xp[1], xp[2], xp[3]};
      const unsigned xw[16] = {xv[0].x, xv[0].y, xv[0].z, xv[0].w, xv[1].x, xv[1].y, xv[1].z, xv[1].w,
                               xv[2].x, xv[2].y, xv[2].z, xv[2].w, xv[3].x, xv[3].y, xv[3].z, xv[3].w};
      float inner = 0.0f;
      for (int n = 0; n < 16; n++) {
        unsigned ent = tbl[(wd[n >> 2] >> (8 * (n & 3))) & 0xffu];
        inner = __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(v2bf, xw[n]), __builtin_bit_cast(v2bf, ent), inner, false);
      }
      acc[j] = __builtin_fmaf(sc, inner, acc[j]);
    }
  }
  for (int j = 0; j < NC; j++)
    for (int o = W / 2; o > 0; o >>= 1) acc[j] = acc[j] + __shfl_xor(acc[j], o, W);
  float sum[NC];
  bool doit;
  long orow;
  if (Mrt == 1) {
    for (int j = 0; j < NC; j++) sum[j] = acc[j];
    doit = valid && lane == 0;
    orow = row;
  } else {
    if (lane == 0)
      for (int j = 0; j < NC; j++) part[j * 16 + rl * Mrt + seg] = acc[j];
    __syncthreads();
    orow = (long)(blk - first) * ROWS + tid;
    doit = tid < ROWS && orow < R;
    const int r2 = tid < ROWS ? tid : 0;
    for (int j = 0; j < NC; j++) {
      float s = 0.0f + part[j * 16 + r2 * Mrt];
      for (int q = 1; q < Mrt; q++) s = s + part[j * 16 + r2 * Mrt + q];
      sum[j] = s;
    }
  }
  if (!doit) return;
  for (int j = 0; j < NC; j++) {
    const float s = sum[j];
    const long idx = (long)j * R + orow;
    if (E == 5) {
      unsigned m = (unsigned)(orow & 511);
      long b5 = orow >> 9;
      u16* base = (u16*)(m < 256 ? out : out2);
      base[b5 * 256 + (m & 255) + (long)j * (R >> 1)] = (u16)(rne_hi(__float_as_uint(s)) >> 16);
    } else if (E == 0) {
      ((u16*)out)[idx] = (u16)(rne_hi(__float_as_uint(s)) >> 16);
    } else if (E <= 4) {
      float v = __uint_as_float(rne_hi(__float_as_uint(s)) & 0xffff0000u);
      if (E == 1 || E == 3) v = f * v;
      if (E == 4) v = s;
      float e = __builtin_amdgcn_exp2f(v * -1.44269502162933349609375f);
      float d = 1.0f + e;
      if (E == 1) {
        ((u16*)out)[idx] = (u16)(rne_hi(__float_as_uint(v / d)) >> 16);
      } else if (E == 2) {
        ((u16*)out)[idx] = (u16)(rne_hi(__float_as_uint(1.0f / d)) >> 16);
      } else {
        float yv = 1.0f / d;
        ((float*)out)[idx] = (E == 3) ? yv + yv : yv;
      }
    }
  }
}

#define AB_ARGS const u16* x, const u16* w, DnRows rows, u16* y, LqGrp grp, int nb, const u16* xa, long K
#define AB_KERNEL(NAME, NC) \
  extern "C" __global__ void NAME(AB_ARGS) { dn_conv_ab_body<NC>(x, w, rows, y, grp, nb, xa, K); }
AB_KERNEL(k_dn_conv_ab_rows_1, 1)
AB_KERNEL(k_dn_conv_ab_rows_2, 2)
AB_KERNEL(k_dn_conv_ab_rows_3, 3)
AB_KERNEL(k_dn_conv_ab_rows_4, 4)
AB_KERNEL(k_dn_conv_ab_rows_5, 5)
AB_KERNEL(k_dn_conv_ab_rows_6, 6)
AB_KERNEL(k_dn_conv_ab_rows_7, 7)
AB_KERNEL(k_dn_conv_ab_rows_8, 8)
