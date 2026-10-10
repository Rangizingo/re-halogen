// Rewrites of the lq4w family (k_lq4w, k_lq4w_pair, k_lq4w_grp) from codeobj_53000, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o lq4w.co lq4w.hip [-DLQ_N=.. -DLQ_W=.. ...]
// Listing: kernels-disasm/codeobj_53000.s. Cites below use the 12-digit address of the instruction in that
// listing for k_lq4w<1,16,1,false,0,0> (0x54c500..), written "@c5xx" for 0x54c5xx.
// The Mac listing leaves VOP3 ops with a nonzero unused src2 field as ".long"; the cites were read from a
// re-decode (llvm-mc, gfx1100 target, src2 cleared). Addresses are the same.
#include <hip/hip_runtime.h>

typedef unsigned short u16;
typedef __bf16 v2bf __attribute__((ext_vector_type(2)));

__device__ static inline unsigned f2u(float f) { return __float_as_uint(f); }
__device__ static inline float u2f(unsigned u) { return __uint_as_float(u); }
// f32 -> bf16 RNE without NaN special case: v_bfe_u32 t,x,16,1 ; v_add3_u32 x,x,t,0x7fff (@c5c0,@c5d4)
__device__ static inline unsigned rne_hi(unsigned u) { return u + ((u >> 16) & 1u) + 0x7fffu; }
// f16 bits -> f32 (v_cvt_f32_f16, @c6b0)
__device__ static inline float h2f(u16 h) { return (float)__builtin_bit_cast(_Float16, h); }

// 8 byte-addressable bf16 pair table: T[b] = (rne_bf16(lut[b>>4]) << 16) | rne_bf16(lut[b&15])  (@c5c0-@c62c)
//   lut = 16 f32 at the start of the weight blob (global_load_b32 at tid&15 and tid>>4, @c53c,@c544)
//   v8.l = v4.h (@c5fc) ; v_and_or_b32 v7, 0xffff0000, v7, v8 (@c618) ; ds_store_b32 tid*4 (@c62c)

// k_lq4w<NC, W, M, B, E, F>(const u8* blob, const bf16* x, void* out, long K, long R, long woff, long soff,
//                           long sstride, void* out2, float f)
//   kernarg: blob@0 x@8 out@16 K@24 R@32 woff@40 soff@48 sstride@56 out2@64 f@72   (s_load_b512 @c500 covers 0..63)
//   NC = activation columns (1..8), W = lanes per row segment (16|32), M = segments per row (1|2|8),
//   B = 1: f32 output, E = epilogue (0 plain, 1 scale+silu, 2 sigmoid, 3 2*sigmoid(f*x) f32, 4 sigmoid f32,
//   5 split-output), F = unroll flag (the vendor preloads scales when K/32 <= 128; numerics unchanged, tested).
//   256 threads, row = blockIdx.x * (256/(W*M)) + tid/(W*M); the row clamps to 0 for loads when row >= R (@c564).
template <int NC, int W, int M, int B, int E, int F>
__device__ static void lq4w_body(const unsigned char* blob, const u16* x, void* out, long K, long R, long woff,
                                 long soff, long sstride, void* out2, float f) {
  constexpr int TPR = W * M;
  constexpr int ROWS = 256 / TPR;
  __shared__ unsigned tbl[256];
  __shared__ float part[M > 1 ? 64 : 1];
  const int tid = threadIdx.x;
  {
    const float* lut = (const float*)blob;
    unsigned lo = rne_hi(f2u(lut[tid & 15])) >> 16;
    unsigned hi = rne_hi(f2u(lut[tid >> 4])) & 0xffff0000u;
    tbl[tid] = hi | lo;
  }
  __syncthreads();
  const int rl = tid / TPR;
  const long row = (long)blockIdx.x * ROWS + rl;
  const bool valid = row < R;
  const long crow = valid ? row : 0;
  const int lane = tid % W;
  const int seg = (tid / W) % M;
  const long khalf = (K + ((K >> 63) & 1)) >> 1;          // @c558,@c560,@c578
  const long G = (K + ((K >> 63) & 31)) >> 5;             // @c5a4,@c5e8,@c600
  const unsigned char* wrow = blob + woff + crow * khalf;  // @c5e0
  const u16* srow = (const u16*)(blob + soff + crow * sstride);  // @c584,@c658..
  long g0, g1;
  if (M == 1) {
    g0 = lane; g1 = G;
  } else {
    long chunk = (G + (M - 1)) / M;                        // 1_32_2 listing: s_add_i32 s11,s10,1 ; ashr 1
    g0 = chunk * seg + lane;
    g1 = chunk * seg + chunk;
    if (G < g1) g1 = G;                                    // v_min_i32
  }
  float acc[NC];
  for (int j = 0; j < NC; j++) acc[j] = 0.0f;
  for (long g = g0; g < g1; g += W) {
    const uint4 wv = *(const uint4*)(wrow + g * 16);
    const float sc = h2f(srow[g]);
    const unsigned wd[4] = {wv.x, wv.y, wv.z, wv.w};
    for (int j = 0; j < NC; j++) {
      const uint4* xp = (const uint4*)(x + (long)j * K + g * 32);
      uint4 xv[4] = {xp[0], xp[1], xp[2], xp[3]};
      const unsigned xw[16] = {xv[0].x, xv[0].y, xv[0].z, xv[0].w, xv[1].x, xv[1].y, xv[1].z, xv[1].w,
                               xv[2].x, xv[2].y, xv[2].z, xv[2].w, xv[3].x, xv[3].y, xv[3].z, xv[3].w};
      float inner = 0.0f;
      for (int n = 0; n < 16; n++) {
        unsigned ent = tbl[(wd[n >> 2] >> (8 * (n & 3))) & 0xffu];
        inner = __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(v2bf, xw[n]), __builtin_bit_cast(v2bf, ent),
                                                inner, false);   // v_dot2_f32_bf16 x,T,acc (@c84c..@c9c4)
      }
      acc[j] = __builtin_fmaf(sc, inner, acc[j]);          // v_fmac_f32 v17, v18, v22 (@c9d4)
    }
  }
  // butterfly xor W/2 .. 1 inside the W lanes (ds_bpermute + v_add_f32, @caa4..@cb0c)
  for (int j = 0; j < NC; j++)
    for (int o = W / 2; o > 0; o >>= 1) acc[j] = acc[j] + __shfl_xor(acc[j], o, W);
  bool doit;
  float sum[NC];
  if (M == 1) {
    doit = valid && lane == 0;
    for (int j = 0; j < NC; j++) sum[j] = acc[j];
  } else {
    if (lane == 0)
      for (int j = 0; j < NC; j++) part[(j * ROWS + rl) * M + seg] = acc[j];
    __syncthreads();
    doit = (tid < ROWS) && ((long)blockIdx.x * ROWS + tid) < R;
    const int r2 = tid < ROWS ? tid : 0;
    for (int j = 0; j < NC; j++) {
      float s = 0.0f + part[(j * ROWS + r2) * M];
      for (int q = 1; q < M; q++) s = s + part[(j * ROWS + r2) * M + q];
      sum[j] = s;
    }
  }
  if (!doit) return;
  const long orow = (M == 1) ? row : (long)blockIdx.x * ROWS + tid;
  for (int j = 0; j < NC; j++) {
    float s = sum[j];
    if (E == 5) {
      unsigned m = (unsigned)(orow & 511);
      long blk = orow >> 9;
      u16* base = (u16*)(m < 256 ? out : out2);
      base[blk * 256 + (m & 255) + (long)j * (R >> 1)] = (u16)(rne_hi(f2u(s)) >> 16);
    } else if (E == 0) {
      if (B) ((float*)out)[(long)j * R + orow] = s;
      else ((u16*)out)[(long)j * R + orow] = (u16)(rne_hi(f2u(s)) >> 16);
    } else {
      float v = u2f(rne_hi(f2u(s)) & 0xffff0000u);
      if (E == 1 || E == 3) v = f * v;
      if (E == 4) v = s;
      float e = __builtin_amdgcn_exp2f(v * -1.44269502162933349609375f);
      float d = 1.0f + e;
      if (E == 1) {
        ((u16*)out)[(long)j * R + orow] = (u16)(rne_hi(f2u(v / d)) >> 16);
      } else if (E == 2) {
        ((u16*)out)[(long)j * R + orow] = (u16)(rne_hi(f2u(1.0f / d)) >> 16);
      } else {
        float y = 1.0f / d;
        ((float*)out)[(long)j * R + orow] = (E == 3) ? y + y : y;
      }
    }
  }
}

#define LQ_SIG(name) \
  extern "C" __global__ void name(const unsigned char* blob, const u16* x, void* out, long K, long R, long woff, \
                                  long soff, long sstride, void* out2, float f)
#define LQ_DEF(name, NC, W, M, B, E, F) \
  LQ_SIG(name) { lq4w_body<NC, W, M, B, E, F>(blob, x, out, K, R, woff, soff, sstride, out2, f); }

#ifdef LQ_N
// One instance picked by -D flags; the spec's rewrite_kernel is lq4w_sel.
LQ_DEF(lq4w_sel, LQ_N, LQ_W, LQ_M, LQ_B, LQ_E, LQ_F)
#elif !defined(LQ_PAIR_N) && !defined(LQ_GRP_N)
// Default build: the 12 k_lq4w instances on the default decode path.
LQ_DEF(lq4w_1_16_1_0_0_0, 1, 16, 1, 0, 0, 0)
LQ_DEF(lq4w_1_16_1_0_5_0, 1, 16, 1, 0, 5, 0)
LQ_DEF(lq4w_1_32_2_0_0_0, 1, 32, 2, 0, 0, 0)
LQ_DEF(lq4w_1_32_2_0_1_0, 1, 32, 2, 0, 1, 0)
LQ_DEF(lq4w_2_16_1_0_0_0, 2, 16, 1, 0, 0, 0)
LQ_DEF(lq4w_2_16_1_0_5_0, 2, 16, 1, 0, 5, 0)
LQ_DEF(lq4w_2_32_2_0_0_0, 2, 32, 2, 0, 0, 0)
LQ_DEF(lq4w_2_32_2_0_1_0, 2, 32, 2, 0, 1, 0)
LQ_DEF(lq4w_3_16_1_0_0_0, 3, 16, 1, 0, 0, 0)
LQ_DEF(lq4w_3_16_1_0_5_0, 3, 16, 1, 0, 5, 0)
LQ_DEF(lq4w_3_32_2_0_0_0, 3, 32, 2, 0, 0, 0)
LQ_DEF(lq4w_3_32_2_0_1_0, 3, 32, 2, 0, 1, 0)
#endif

// ---------------------------------------------------------------------------------------------
// k_lq4w_pair<NC, 16, F>(const u8* blobA, long woffA, long soffA, long sstrideA, const u8* blobB, long woffB,
//                        long soffB, long sstrideB, const bf16* x, bf16* out, long K, long R)
//   kernarg: blobA@0 woffA@8 soffA@16 sstrideA@24 blobB@32 woffB@40 soffB@48 sstrideB@56 x@64 out@72 K@80 R@88
//   (s_load_b512 s[4:19] + s_load_b256 s[20:27] at 0x40, listing 0x288500 and 0x288514 for <1,16,0>).
//   Two 4-bit tensors (A, B), each with its own LUT table (A at LDS 0, B at LDS 1024: ds_store_2addr_stride64
//   @2886ec, B reads use "offset:1024" @28890c), the same x vector, 16 rows per block, 16 lanes per row.
//   Per group: innerA = chain of 16 dot2 with table A, innerB likewise, scales scA/scB from the two f16 arrays
//   (global_load_d16_b16 / d16_hi_b16, @288790,@288798), accA += scA*innerA, accB += scB*innerB (fmac).
//   Epilogue (@288f0c..@288ff8): a = bf16r(sumA), b = bf16r(sumB); out = bf16_rne(silu(a) * b), silu as in E=1
//   with scale 1 (exp2(a * -1.4427), IEEE division). Which of A or B feeds silu is settled by the compare.
template <int NC, int F>
__device__ static void lq4w_pair_body(const unsigned char* blobA, long woffA, long soffA, long sstrideA,
                                      const unsigned char* blobB, long woffB, long soffB, long sstrideB,
                                      const u16* x, u16* out, long K, long R) {
  __shared__ unsigned tblA[256];
  __shared__ unsigned tblB[256];
  const int tid = threadIdx.x;
  {
    const float* la = (const float*)blobA;
    const float* lb = (const float*)blobB;
    tblA[tid] = (rne_hi(f2u(la[tid >> 4])) & 0xffff0000u) | (rne_hi(f2u(la[tid & 15])) >> 16);
    tblB[tid] = (rne_hi(f2u(lb[tid >> 4])) & 0xffff0000u) | (rne_hi(f2u(lb[tid & 15])) >> 16);
  }
  __syncthreads();
  const int rl = tid >> 4;
  const long row = (long)blockIdx.x * 16 + rl;
  const bool valid = row < R;
  const long crow = valid ? row : 0;
  const int lane = tid & 15;
  const long khalf = (K + ((K >> 63) & 1)) >> 1;
  const long G = (K + ((K >> 63) & 31)) >> 5;
  const unsigned char* wA = blobA + woffA + crow * khalf;
  const unsigned char* wB = blobB + woffB + crow * khalf;
  const u16* sA = (const u16*)(blobA + soffA + crow * sstrideA);
  const u16* sB = (const u16*)(blobB + soffB + crow * sstrideB);
  float accA[NC], accB[NC];
  for (int j = 0; j < NC; j++) { accA[j] = 0.0f; accB[j] = 0.0f; }
  for (long g = lane; g < G; g += 16) {
    const uint4 va = *(const uint4*)(wA + g * 16);
    const uint4 vb = *(const uint4*)(wB + g * 16);
    const float scA = h2f(sA[g]);
    const float scB = h2f(sB[g]);
    const unsigned wa[4] = {va.x, va.y, va.z, va.w};
    const unsigned wb[4] = {vb.x, vb.y, vb.z, vb.w};
    for (int j = 0; j < NC; j++) {
      const uint4* xp = (const uint4*)(x + (long)j * K + g * 32);
      uint4 xv[4] = {xp[0], xp[1], xp[2], xp[3]};
      const unsigned xw[16] = {xv[0].x, xv[0].y, xv[0].z, xv[0].w, xv[1].x, xv[1].y, xv[1].z, xv[1].w,
                               xv[2].x, xv[2].y, xv[2].z, xv[2].w, xv[3].x, xv[3].y, xv[3].z, xv[3].w};
      float ia = 0.0f, ib = 0.0f;
      for (int n = 0; n < 16; n++) {
        unsigned ea = tblA[(wa[n >> 2] >> (8 * (n & 3))) & 0xffu];
        unsigned eb = tblB[(wb[n >> 2] >> (8 * (n & 3))) & 0xffu];
        ia = __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(v2bf, xw[n]), __builtin_bit_cast(v2bf, ea), ia, false);
        ib = __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(v2bf, xw[n]), __builtin_bit_cast(v2bf, eb), ib, false);
      }
      accA[j] = __builtin_fmaf(scA, ia, accA[j]);
      accB[j] = __builtin_fmaf(scB, ib, accB[j]);
    }
  }
  for (int j = 0; j < NC; j++)
    for (int o = 8; o > 0; o >>= 1) {
      accA[j] = accA[j] + __shfl_xor(accA[j], o, 16);
      accB[j] = accB[j] + __shfl_xor(accB[j], o, 16);
    }
  if (!(valid && lane == 0)) return;
  for (int j = 0; j < NC; j++) {
    float a = u2f(rne_hi(f2u(accA[j])) & 0xffff0000u);
    float b = u2f(rne_hi(f2u(accB[j])) & 0xffff0000u);
    float e = __builtin_amdgcn_exp2f(a * -1.44269502162933349609375f);
    float d = 1.0f + e;
    float s = a / d;
    out[(long)j * R + row] = (u16)(rne_hi(f2u(s * b)) >> 16);
  }
}

#ifdef LQ_PAIR_N
extern "C" __global__ void lq4w_pair_sel(const unsigned char* blobA, long woffA, long soffA, long sstrideA,
                                         const unsigned char* blobB, long woffB, long soffB, long sstrideB,
                                         const u16* x, u16* out, long K, long R) {
  lq4w_pair_body<LQ_PAIR_N, LQ_PAIR_F>(blobA, woffA, soffA, sstrideA, blobB, woffB, soffB, sstrideB, x, out, K, R);
}
#endif

// ---------------------------------------------------------------------------------------------
// k_lq4w_grp<NC, W, MB, F>(LqGrp g, const bf16* x, long K)
//   kernarg: g@0 (296 bytes by value) x@296 K@304 (listing 0x293600 for <1,16,1,0>)
//   LqGrp = 4 group entries of 72 bytes + count (int @0x120). Entry layout (offsets read from the s_load_b256
//   at 0x18 @2936b8, s_load_b64 at 0 @2936c0, 0x38 @294304 / @293e54, 0x8 @29430c, 0x10 @293ebc, 0x40 @293d7c):
//     +0 blob  +8 out  +0x10 out2  +0x18 woff  +0x20 soff  +0x28 sstride  +0x30 R  +0x38 f  +0x3c E  +0x40 first block
//   Block b belongs to the highest entry i < count with b >= entry[i].first (@293624..@2936a0).
//   Rows R > 8: base split (MB segments per row); R <= 8: MS segments per row, one row per block
//   (@2936cc: v_cmp_gt_i64 R, 8 ; s_cselect 1/16 for W=16, 2/8 for W=32 @1_32_2:s_cselect_b32 s3,8,2).
//   Epilogue is selected at run time by E (0..5, same functions as k_lq4w E=0..5 with E=0 always bf16);
//   output index is col*R + row. Segment partials are summed in order starting from 0.0f (scalar s_add_f32 chain
//   @293dd4.. for 16 partials, v_add_f32 chain in the 32-lane variant).
struct LqGrpEnt {
  const unsigned char* blob; void* out; void* out2;
  long woff, soff, sstride, R;
  float f; int E; int start; int pad;
};
struct LqGrp {
  LqGrpEnt g[4];
  int count; int pad;
};

template <int NC, int W, int MB, int MS, int F>
__device__ static void lq4w_grp_body(const LqGrp& gp, const u16* x, long K) {
  __shared__ unsigned tbl[256];
  __shared__ float part[8 * 16];
  const int tid = threadIdx.x;
  const int blk = blockIdx.x;
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
    tbl[tid] = (rne_hi(f2u(lut[tid >> 4])) & 0xffff0000u) | (rne_hi(f2u(lut[tid & 15])) >> 16);
  }
  __syncthreads();
  const int Mrt = (R <= 8) ? MS : MB;
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
      const uint4* xp = (const uint4*)(x + (long)j * K + g * 32);
      uint4 xv[4] = {xp[0], xp[1], xp[2], xp[3]};
      const unsigned xw[16] = {xv[0].x, xv[0].y, xv[0].z, xv[0].w, xv[1].x, xv[1].y, xv[1].z, xv[1].w,
                               xv[2].x, xv[2].y, xv[2].z, xv[2].w, xv[3].x, xv[3].y, xv[3].z, xv[3].w};
      float inner = 0.0f;
      for (int n = 0; n < 16; n++) {
        unsigned ent = tbl[(wd[n >> 2] >> (8 * (n & 3))) & 0xffu];
        inner = __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(v2bf, xw[n]), __builtin_bit_cast(v2bf, ent),
                                                inner, false);
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
      base[b5 * 256 + (m & 255) + (long)j * (R >> 1)] = (u16)(rne_hi(f2u(s)) >> 16);
    } else if (E == 0) {
      ((u16*)out)[idx] = (u16)(rne_hi(f2u(s)) >> 16);
    } else if (E <= 4) {
      float v = u2f(rne_hi(f2u(s)) & 0xffff0000u);
      if (E == 1 || E == 3) v = f * v;
      if (E == 4) v = s;
      float e = __builtin_amdgcn_exp2f(v * -1.44269502162933349609375f);
      float d = 1.0f + e;
      if (E == 1) {
        ((u16*)out)[idx] = (u16)(rne_hi(f2u(v / d)) >> 16);
      } else if (E == 2) {
        ((u16*)out)[idx] = (u16)(rne_hi(f2u(1.0f / d)) >> 16);
      } else {
        float y = 1.0f / d;
        ((float*)out)[idx] = (E == 3) ? y + y : y;
      }
    }
  }
}

#ifdef LQ_GRP_N
extern "C" __global__ void lq4w_grp_sel(LqGrp gp, const u16* x, long K) {
  lq4w_grp_body<LQ_GRP_N, LQ_GRP_W, LQ_GRP_MB, LQ_GRP_MS, LQ_GRP_F>(gp, x, K);
}
#endif
