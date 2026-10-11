// Rewrite of q4moe::k_q4moe<A,B,TILE,Tune<0,128>,E>(q4moe::Args), vendor code object 53000, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o q4moe.co q4moe.hip
// "L:n" = kernels-disasm/codeobj_53000.s line n. Instance <1,2,64,Tune<0,128>,2> starts at L:2008013.
// Listing offsets below are "address" cites of the form E16300 (hex offset in the code object) because the 235 VOP3 words
// llvm cannot decode (v_mul_lo_u32, v_add_co_u32, 64-bit shifts, v_mov_b16 ...) were re-decoded by hand (see report).
//
// Args (152 bytes, by value, kernarg offsets from the s_load list L:2008014-2008027):
//   0x00 w base, 0x08 weight byte offset (s_add s20,s22), 0x10 scale byte offset (s20+s24), 0x18 scale row stride (bytes)
//   0x20 lut[16] f32 (s_load_b512 s[4:19]), 0x60 svh, 0x68 suh (E=1 only), 0x70 x (bf16 [row][K]),
//   0x78 idx (B=0 only: x row of a token slot = idx[start+n] / 10), 0x80 items (Item = {expert,k,start,nrows}),
//   0x88 count (int*), 0x90 y (bf16 [slot][N])
// One block (256 threads = 8 waves) per item: blockIdx.x < *count (L:2008015-2008019). Item at items + 16*blockIdx.x.
//
// Shared math (A=1; verified bitwise on the vendor kernels, see report):
//   weight row = K/2 bytes (B=2: 320 = 0x140; B=0: 1280 = 0x500); chunk c = bytes [64c, 64c+64); byte i holds k=2i (low nibble), 2i+1
//   decoded weight = bf16_rne( float(scale) * lut[nibble] ) (v_fma_mix_f32 s, lut, neg(0) = f32 product; RNE = v + ((v>>16)&1) +
//     0x7fff, no NaN case; table lookup by v_perm/v_bfi)
//   scale f16 per (row, group): group = 128 k (E=1,2: at base+soff+row*sstride+2c) or 32 k (E=0: entry 4c+q, global_load_b64)
//   x row for token slot n (0..63 or 0..127) = min(n, nrows-1) + start (v_min_i32 s39-1), B=0 through idx[.] / 10
//   WMMA v_wmma_f32_16x16x16_bf16 D=acc, A=weights (16 channels x 16 k), B=x (16 k x 16 tokens). Two A tiles T0/T1 per wave,
//     ntiles of 16 tokens; acc never cleared between chunks. Per chunk k16 blocks j=0..7 ascending for every accumulator.
//   Fragment element f (0..15) of block j: g=f/8, p=f%8, k = 16j + 8g + ORD[p], ORD={0,2,4,6,1,3,5,7}; same slot both sides.
//   gfx11 w32 accumulator: lane l, vgpr i = D[row 2*i + (l>>4)][col l&15].
//   B=2 (down): T0/T1 rows = expert*2560 + k*256 + wave*32 + 16*T + l16 (v_mad_i64_i32 0xa00, E16474); output y[slot*2560 + ...].
//   B=0 (gate_up): T0 = gate rows expert*1280 + k*128 + wave*16 + l16, T1 = up rows = gate + 640 (v_cndmask 0x280 on lane half,
//     L:1995... listing i1002 FBD6C4-FBD6F0); output y[slot*640 + k*128 + wave*16 + 2i + half].
// A=2 (f16 path, listing i2202 = <2,2,64,..,0> at L:2331473): the LUT is converted to f16 in SGPRs (s_cvt_f16_f32, RNE), packed
//   into byte tables; decoded weight = f16 pk_mul(f16(lut[nib]), scale_f16) (v_pk_mul_f16, VOP3P words cc10xxxx);
//   x is converted bf16 -> f32 -> f16 toward zero (v_cvt_pk_rtz_f16_f32) before it goes to LDS; the WMMA is
//   v_wmma_f32_16x16x16_f16. Same fragment order, same chunk/block order, same epilogues as A=1.
// TILE=128: same arithmetic, items with nrows up to 128 (8 token tiles per wave, 36864 B LDS in the vendor). TILE=64 caps at 64.
// Epilogues:
//   E=2 / E=0 (B=2): y = bf16_rne(acc).
//   E=0 (B=0): g=bf16(acc_gate), u=bf16(acc_up) as f32; y = bf16_rne(u * (g / (1 + exp2(g * -1.44269502f)))) (FC0358-FC0668).
//   E=1 (B=2): per token and 128-channel group, y = bf16_rne(svh[c] * (WHT128(acc) * 0x3db504f3)) (E24E74-E25230, listing i1212e1).
//   E=1 (B=0): G=WHT(gate)*c*svh_g, U=WHT(up)*c*svh_u, P=U*silu(G), then x0..x3 from fma(P_even, suh, +-round(P_odd*suh)),
//     WHT stages 2.., *c, bf16_rne (E05A74-E05F..., listing i1011).
#include <hip/hip_runtime.h>
#include <cstdint>
#pragma clang fp contract(off)

struct Item { int expert, k, start, nrows; };
struct Args {
  const unsigned char* w;       // 0x00
  long woff;                    // 0x08
  long soff;                    // 0x10
  long sstride;                 // 0x18
  float lut[16];                // 0x20
  const void* side0;            // 0x60
  const void* side1;            // 0x68
  const unsigned short* x;      // 0x70
  const int* idx;               // 0x78
  const Item* items;            // 0x80
  const int* count;             // 0x88
  unsigned short* y;            // 0x90
};
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));

__device__ __constant__ static const int ORD[8] = {0, 2, 4, 6, 1, 3, 5, 7};

__device__ static inline unsigned short bf16_rne(float f) {
  unsigned v = __float_as_uint(f);
  return (unsigned short)((v + ((v >> 16) & 1) + 0x7fff) >> 16);
}
__device__ static inline float bf16r(float f) { return __uint_as_float((unsigned)bf16_rne(f) << 16); }
__device__ static inline float f16_to_f32(unsigned short h) {
  float r;
  asm volatile("v_cvt_f32_f16 %0, %1" : "=v"(r) : "v"((unsigned)h));
  return r;
}
__device__ static inline float mul_f32(float a, float b) {
  float r;
  asm volatile("v_mul_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
  return r;
}
// A=2 helpers: f16 weights/activations (v_wmma_f32_16x16x16_f16)
__device__ static inline unsigned short cvt_f16_rne(float f) {   // s_cvt_f16_f32 in the prologue (default round mode)
  unsigned r;
  asm volatile("v_cvt_f16_f32 %0, %1" : "=v"(r) : "v"(f));
  return (unsigned short)r;
}
__device__ static inline unsigned short cvt_f16_rtz(float f) {   // v_cvt_pk_rtz_f16_f32 (x: bf16 -> f32 -> f16 toward zero)
  unsigned r;
  asm volatile("v_cvt_pk_rtz_f16_f32 %0, %1, %2" : "=v"(r) : "v"(f), "v"(0.0f));
  return (unsigned short)(r & 0xffff);
}
__device__ static inline unsigned short mul_f16(unsigned short a, unsigned short b) {   // v_pk_mul_f16, one lane
  unsigned r;
  asm volatile("v_pk_mul_f16 %0, %1, %2" : "=v"(r) : "v"((unsigned)a), "v"((unsigned)b));
  return (unsigned short)(r & 0xffff);
}
__device__ static inline float silu_f(float g) {
  float ex = __builtin_amdgcn_exp2f(g * -1.44269502162933349609375f);  // 0xbfb8aa3b
  return g / (1.0f + ex);                                               // IEEE div (div_scale/rcp/fma/fmas/fixup)
}

// 128-point Walsh-Hadamard transform exactly as the vendor schedules it: v[L][e] = value of channel 4L+e, L = lane 0..31.
// In-register stage (outputs f0..f3), then lane bits 0..3 (row_xmask 1,2,4,8), then bit 4 (permlanex16). A stage on lane bit b:
// lane with bit 0 gets x + partner, lane with bit 1 gets partner + (-x).
__device__ static inline void wht128(float v[32][4]) {
  for (int L = 0; L < 32; L++) {
    float A0 = v[L][0], B0 = v[L][1], C0 = v[L][2], D0 = v[L][3];
    float s0 = A0 + B0, d0 = A0 - B0, s1 = C0 + D0, d1 = C0 - D0;
    v[L][0] = s0 + s1;
    v[L][1] = d0 + d1;
    v[L][2] = s0 - s1;
    v[L][3] = d0 - d1;
  }
  for (int b = 0; b < 5; b++) {
    float nv[32][4];
    for (int L = 0; L < 32; L++)
      for (int e = 0; e < 4; e++) {
        float x = v[L][e], pp = v[L ^ (1 << b)][e];
        nv[L][e] = ((L >> b) & 1) ? (pp + (-x)) : (x + pp);
      }
    for (int L = 0; L < 32; L++)
      for (int e = 0; e < 4; e++) v[L][e] = nv[L][e];
  }
}
// stages after the in-register one (used by B=0 E=1, whose first stage is fused with the suh multiply)
__device__ static inline void wht128_lanes(float v[32][4]) {
  for (int b = 0; b < 5; b++) {
    float nv[32][4];
    for (int L = 0; L < 32; L++)
      for (int e = 0; e < 4; e++) {
        float x = v[L][e], pp = v[L ^ (1 << b)][e];
        nv[L][e] = ((L >> b) & 1) ? (pp + (-x)) : (x + pp);
      }
    for (int L = 0; L < 32; L++)
      for (int e = 0; e < 4; e++) v[L][e] = nv[L][e];
  }
}

// MODE 0: B=2 (down). MODE 1: B=0 (gate_up, gather + silu).
// KCH = K/128 chunks, SG = scale group (128 or 32), NT = 16-token tiles (4 for TILE 64, 8 for TILE 128).
template <int FP, int MODE, int KCH, int SG, int NT, int EPI>
__device__ static inline void body(Args a) {
  constexpr int ROWS = MODE == 0 ? 2560 : 1280;
  constexpr int XS = MODE == 0 ? 640 : 2560;
  constexpr int OS = MODE == 0 ? 2560 : 640;
  constexpr int WROW = KCH * 64;
  if ((int)blockIdx.x >= *a.count) return;
  Item it = a.items[blockIdx.x];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, l16 = lane & 15, half = lane >> 4;
  long row0, row1;
  if (MODE == 0) {
    row0 = (long)it.expert * ROWS + it.k * 256 + wave * 32 + l16;
    row1 = row0 + 16;
  } else {
    row0 = (long)it.expert * ROWS + it.k * 128 + wave * 16 + l16;
    row1 = row0 + 640;
  }
  // x row of each 16-token tile column of this lane
  long xrow[NT];
  for (int n = 0; n < NT; n++) {
    int tok = 16 * n + l16;
    if (tok > it.nrows - 1) tok = it.nrows - 1;
    int slot = it.start + tok;
    xrow[n] = MODE == 0 ? slot : (a.idx[slot] / 10);   // mulhi 0x66666667 then >>2 + sign = /10 (i1002 FBD74C-FBD7B8)
  }
  v8f acc[2][NT];
  for (int t = 0; t < 2; t++)
    for (int n = 0; n < NT; n++) acc[t][n] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
  for (int c = 0; c < KCH; c++) {
    v16s A[2][8];
    for (int t = 0; t < 2; t++) {
      long r = t ? row1 : row0;
      const unsigned char* wr = a.w + a.woff + r * WROW + c * 64;
      unsigned short tab[128 / SG][16];
      for (int g = 0; g < 128 / SG; g++) {
        unsigned short sc = *(const unsigned short*)(a.w + a.soff + r * a.sstride + 2 * (c * (128 / SG) + g));
        if (FP) {
          for (int q = 0; q < 16; q++) tab[g][q] = mul_f16(cvt_f16_rne(a.lut[q]), sc);
        } else {
          float s = f16_to_f32(sc);
          for (int q = 0; q < 16; q++) tab[g][q] = bf16_rne(mul_f32(s, a.lut[q]));
        }
      }
      for (int j = 0; j < 8; j++)
        for (int f = 0; f < 16; f++) {
          int k = 16 * j + 8 * (f >> 3) + ORD[f & 7];
          unsigned char byte = wr[k >> 1];
          A[t][j][f] = (short)tab[k / SG][(k & 1) ? (byte >> 4) : (byte & 15)];
        }
    }
    for (int n = 0; n < NT; n++) {
      const unsigned short* xr = a.x + xrow[n] * XS + c * 128;
      for (int j = 0; j < 8; j++) {
        v16s B;
        for (int f = 0; f < 16; f++) {
          unsigned short xv = xr[16 * j + 8 * (f >> 3) + ORD[f & 7]];
          B[f] = FP ? (short)cvt_f16_rtz(__uint_as_float((unsigned)xv << 16)) : (short)xv;
        }
        for (int t = 0; t < 2; t++) {
          if (FP) acc[t][n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(v16h, A[t][j]), __builtin_bit_cast(v16h, B), acc[t][n]);
          else acc[t][n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(A[t][j], B, acc[t][n]);
        }
      }
    }
  }
  const int chb = MODE == 0 ? wave * 32 : wave * 16;   // channel base of this wave inside the item's channel block
  if (EPI == 0 && MODE == 0) {
    for (int t = 0; t < 2; t++)
      for (int n = 0; n < NT; n++)
        for (int i = 0; i < 8; i++) {
          int tok = 16 * n + l16;
          if (tok < it.nrows) a.y[(long)(it.start + tok) * OS + it.k * 256 + chb + 16 * t + 2 * i + half] = bf16_rne(acc[t][n][i]);
        }
  } else if (EPI == 0 && MODE == 1) {
    for (int n = 0; n < NT; n++)
      for (int i = 0; i < 8; i++) {
        int tok = 16 * n + l16;
        if (tok < it.nrows) {
          float g = bf16r(acc[0][n][i]), u = bf16r(acc[1][n][i]);
          a.y[(long)(it.start + tok) * OS + it.k * 128 + chb + 2 * i + half] = bf16_rne(mul_f32(u, silu_f(g)));
        }
      }
  } else if (MODE == 0) {
    // B=2 E=1: acc tile staged in LDS as Z[token16][256 channels] f32 (ds_store_2addr at 260-float row pitch), wave w takes
    // token rows w and w+8, lane L loads channels 4L..4L+3 of group g.
    __shared__ float Z[16][260];
    const unsigned short* svh = (const unsigned short*)a.side0;
    for (int n = 0; n < NT; n++) {
      __syncthreads();
      for (int t = 0; t < 2; t++)
        for (int i = 0; i < 8; i++) Z[l16][chb + 16 * t + 2 * i + half] = acc[t][n][i];
      __syncthreads();
      for (int pass = 0; pass < 2; pass++) {
        int tr = wave + 8 * pass;
        int tok = 16 * n + tr;
        if (tok >= it.nrows) continue;
        for (int g = 0; g < 2; g++) {
          float v[32][4];
          for (int L = 0; L < 32; L++)
            for (int e = 0; e < 4; e++) v[L][e] = Z[tr][128 * g + 4 * L + e];
          wht128(v);
          for (int L = 0; L < 32; L++)
            for (int e = 0; e < 4; e++) {
              float sc = mul_f32(v[L][e], __uint_as_float(0x3db504f3u));   // 1/sqrt(128), literal bits
              float sv = f16_to_f32(svh[it.k * 256 + 128 * g + 4 * L + e]);
              a.y[(long)(it.start + tok) * OS + it.k * 256 + 128 * g + 4 * L + e] = bf16_rne(mul_f32(sv, sc));
            }
        }
      }
    }
  } else {
    // B=0 E=1: Zg/Zu[token16][128 channels] f32.
    __shared__ float Zg[16][132], Zu[16][132];
    const unsigned short* svh = (const unsigned short*)a.side0;
    const unsigned short* suh = (const unsigned short*)a.side1;
    for (int n = 0; n < NT; n++) {
      __syncthreads();
      for (int i = 0; i < 8; i++) {
        Zg[l16][chb + 2 * i + half] = acc[0][n][i];
        Zu[l16][chb + 2 * i + half] = acc[1][n][i];
      }
      __syncthreads();
      for (int pass = 0; pass < 2; pass++) {
        int tr = wave + 8 * pass;
        int tok = 16 * n + tr;
        if (tok >= it.nrows) continue;
        float G[32][4], U[32][4], X[32][4];
        for (int L = 0; L < 32; L++)
          for (int e = 0; e < 4; e++) {
            G[L][e] = Zg[tr][4 * L + e];
            U[L][e] = Zu[tr][4 * L + e];
          }
        wht128(G);
        wht128(U);
        const float cst = __uint_as_float(0x3db504f3u);
        for (int L = 0; L < 32; L++) {
          float P[4], sh[4];
          for (int e = 0; e < 4; e++) {
            int ch = it.k * 128 + 4 * L + e;
            float g1 = mul_f32(f16_to_f32(svh[ch]), mul_f32(G[L][e], cst));
            float u1 = mul_f32(f16_to_f32(svh[640 + ch]), mul_f32(U[L][e], cst));
            P[e] = mul_f32(u1, silu_f(g1));
            sh[e] = f16_to_f32(suh[ch]);
          }
          float b1 = mul_f32(sh[1], P[1]), b3 = mul_f32(sh[3], P[3]);
          float x0 = __builtin_fmaf(P[0], sh[0], b1), x1 = __builtin_fmaf(P[0], sh[0], -b1);
          float x2 = __builtin_fmaf(P[2], sh[2], b3), x3 = __builtin_fmaf(P[2], sh[2], -b3);
          X[L][0] = x0 + x2;
          X[L][1] = x1 + x3;
          X[L][2] = x0 - x2;
          X[L][3] = x1 - x3;
        }
        wht128_lanes(X);
        for (int L = 0; L < 32; L++)
          for (int e = 0; e < 4; e++)
            a.y[(long)(it.start + tok) * OS + it.k * 128 + 4 * L + e] = bf16_rne(mul_f32(X[L][e], __uint_as_float(0x3db504f3u)));
      }
    }
  }
}

#define K2(P, T, NTV, E, SGV) \
  extern "C" __global__ void __launch_bounds__(256) k_q4moe_a##P##b2e##E##_##T(Args a) { body<P - 1, 0, 5, SGV, NTV, (E == 1) ? 1 : 0>(a); }
#define K0(P, T, NTV, E, SGV) \
  extern "C" __global__ void __launch_bounds__(256) k_q4moe_a##P##b0e##E##_##T(Args a) { body<P - 1, 1, 20, SGV, NTV, E>(a); }
#define ALLK(P)                                    \
  K2(P, 64, 4, 2, 128) K2(P, 64, 4, 1, 128) K2(P, 64, 4, 0, 32)       \
  K2(P, 128, 8, 2, 128) K2(P, 128, 8, 1, 128) K2(P, 128, 8, 0, 32)    \
  K0(P, 64, 4, 1, 128) K0(P, 64, 4, 0, 32) K0(P, 128, 8, 1, 128) K0(P, 128, 8, 0, 32)
ALLK(1)
ALLK(2)
