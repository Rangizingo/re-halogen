// Rewrite of the k_lq8w_grp<M, 32, 2> family (flash_serve, gfx1151): grouped q8g64 GEMV with a per-member epilogue.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o lq8w_grp.co lq8w_grp.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git), reference instance k_lq8w_grp<1,32,2> (symbol 0x2f4300, :161193). Cites are codeobj_53000.s:<line>. Some VOP3 instructions print as .long, so the
// branch structure was read from a fresh ROCm-23 llvm-objdump of the same symbol (branch targets are
// byte addresses of that symbol; :161579 is the epilogue dispatch).
//
// Signature: (LqGrp g, unsigned short const* x, long K); kernarg: g@0 (296 bytes), x@0x128, K@0x130.
// LqGrp (reconstructed from the loads at 0x120, 0x88, 0xd0, 0x118 and the member loads; 0x48 bytes per member):
//   member i at 0x48*i: +0x00 W (q8g64 rows), +0x08 out0, +0x10 out1 (mode 5 only), +0x30 long rows N,
//                       +0x38 float alpha, +0x3c int mode, +0x40 int start block.   int count at +0x120.
// Member select (:161196 count load at 0x120, :161225-161229 selects, :161229 s_mulk_i32 0x48): sel = (count>3 && bid>=m3.start) ? 3 : (count>2 && bid>=m2.start) ? 2 : (count>1 && bid>=m1.start) ? 1 : 0
//   (s_cmp_gt_i32 s3,{1,2,3}; s_cmp_ge_i32 s2,{0x88,0xd0,0x118}; s_mulk_i32 s3,0x48).
// Shape picked at run time from N: N < 9 gives W=8 waves per row, 1 row per block; otherwise W=2 waves, 4 rows per block
//   (v_cmp_lt_i64 s[10:11],9 then s_cselect 8:2, 3:1, 0:2: :161240).  Local block = bid - start; row = (local << log2(RPB)) + row-in-block.
// Weights are exactly k_lq8w: row stride K + K/16 (no pad), (scale, offset) f16 pair per 64 weights, w = bf16_rne(fma(u8, scale, offset)),
//   K/16 groups split over W waves by chunk = ceil(ng/W), lanes stride 32, dot2 chain per group from 0, xor 16..1 over the 32 lanes.
// Cross-wave step (ds_store :161542, barrier :161549, ds_load loop :161571): lane 0 of each wave stores to LDS[(row_in_block*W + w)*M + m]; thread t < RPB*M (row_in_block = t/M, m = t%M,
//   the M=2 listing shows v_lshrrev 1 / v_and 1) sums w = 0..W-1 from 0.0 in order, then the epilogue for the member's mode.
// Epilogue modes (dispatch :161579, compares :161586-161592: s_cmp s5 < 3 / > 3 / > 4 / == 5; r = the f32 sum, rb = bf16_rne(r) widened back to f32):
//   0: out0 bf16[m*N + row] = bf16_rne(r)                                   (:161752-161764)
//   1: out0 bf16[m*N + row] = bf16_rne(silu(alpha*rb)), silu(v) = v / (1 + exp2(v * -log2e))   (:161768-161797)
//   2: out0 bf16[m*N + row] = bf16_rne(sigmoid(rb)), sigmoid = 1 / (1 + exp2(rb * -log2e))        (:161710-161745)
//   3: out0 f32 [m*N + row] = 2 * sigmoid(alpha*rb)                          (:161668-161702, v_add_f32 v0,v0,v0)
//   4: out0 f32 [m*N + row] = sigmoid(r)  (r not rounded to bf16)            (:161637-161663)
//   5: bf16_rne(r) split in two planes: base = row & ~511, idx = row - base; idx < 256 -> out0 else out1 with idx -= 256;
//      byte address = plane + m*(N/2)*2 + base + idx*2                       (:161595-161632; M=2 symbol: :162237-162286)
// Divisions are IEEE (v_div_scale/v_div_fmas/v_div_fixup), exp is v_exp_f32 on v * -1.442695 (0xbfb8aa3b).
#include <hip/hip_runtime.h>

__device__ static inline float h2f(unsigned short u) { return (float)__builtin_bit_cast(_Float16, u); }

typedef __bf16 v2bf __attribute__((ext_vector_type(2)));

__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
__device__ static inline float bf_round(float f) { return __uint_as_float((unsigned)f2bf_rne(f) << 16); }
__device__ static inline float dot2(unsigned x, unsigned w, float acc) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(v2bf, x), __builtin_bit_cast(v2bf, w), acc, false);
}
__device__ static inline float sigmoid_(float v) { return 1.0f / (1.0f + __builtin_amdgcn_exp2f(v * -1.44269502162933349609375f)); }
__device__ static inline float silu_(float v) { return v / (1.0f + __builtin_amdgcn_exp2f(v * -1.44269502162933349609375f)); }

struct LqMem {
  const unsigned char* w;  // +0x00
  void* o0;                // +0x08
  void* o1;                // +0x10
  long pad[3];             // +0x18..0x2f (unused by this kernel)
  long n;                  // +0x30
  float alpha;             // +0x38
  int mode;                // +0x3c
  int start;               // +0x40
  int pad2;
};
struct LqGrp {
  LqMem m[4];
  int count;  // +0x120
  int pad;
};
static_assert(sizeof(LqMem) == 72, "member");
static_assert(sizeof(LqGrp) == 296, "grp");

template <int M>
__device__ static inline void lq8w_grp_body(const LqGrp& g, const unsigned short* x, long K) {
  __shared__ float lds[64];
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  int sel = 0;
  if (g.count > 1 && bid >= g.m[1].start) sel = 1;
  if (g.count > 2 && bid >= g.m[2].start) sel = 2;
  if (g.count > 3 && bid >= g.m[3].start) sel = 3;
  const LqMem& mb = g.m[sel];
  const long n = mb.n;
  const bool small = n < 9;
  const int W = small ? 8 : 2;
  const int lgW = small ? 3 : 1;
  const int lgR = small ? 0 : 2;
  const int RPB = 1 << lgR;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int rin = wave >> lgW;
  const int wv = wave & (W - 1);
  const long rbase = (long)(bid - mb.start) << lgR;
  const long row = rbase + rin;
  const bool valid = row < n;
  const long lr = valid ? row : 0;
  const long ng = K / 16;
  const long rowbytes = K + K / 16;
  const unsigned char* rp = mb.w + lr * rowbytes;
  const unsigned char* sc = rp + K;
  const long chunk = (ng + W - 1) / W;
  const long begin = (long)wv * chunk;
  const long end = begin + chunk < ng ? begin + chunk : ng;
  float acc[M];
  for (int m = 0; m < M; m++) acc[m] = 0.0f;
  for (long j = begin + lane; j < end; j += 32) {
    uint4 q4 = *(const uint4*)(rp + 16 * j);
    unsigned so = *(const unsigned*)(sc + 4 * (j >> 2));
    float s = h2f((unsigned short)(so & 0xffffu));
    float o = h2f((unsigned short)(so >> 16));
    unsigned qw[4] = {q4.x, q4.y, q4.z, q4.w};
    unsigned wp[8];
    for (int i = 0; i < 8; i++) {
      unsigned b0 = (qw[i / 2] >> (16 * (i & 1))) & 0xffu;
      unsigned b1 = (qw[i / 2] >> (16 * (i & 1) + 8)) & 0xffu;
      wp[i] = (unsigned)f2bf_rne(__builtin_fmaf((float)b0, s, o)) | ((unsigned)f2bf_rne(__builtin_fmaf((float)b1, s, o)) << 16);
    }
    for (int m = 0; m < M; m++) {
      const uint4* xp = (const uint4*)(x + (long)m * K + 16 * j);
      uint4 x0 = xp[0], x1 = xp[1];
      unsigned xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
      float t = 0.0f;
      for (int p = 0; p < 8; p++) t = dot2(xw[p], wp[p], t);
      acc[m] += t;
    }
  }
  for (int m = 0; m < M; m++) {
    float a = acc[m];
    for (int off = 16; off >= 1; off >>= 1) a = a + __shfl_xor(a, off, 32);
    if (lane == 0) lds[(rin * W + wv) * M + m] = a;
  }
  __syncthreads();
  if (tid < RPB * M) {
    const int rl = tid / M, m = tid % M;
    const long orow = rbase + rl;
    if (orow < n) {
      float r = 0.0f;
      for (int w = 0; w < W; w++) r = r + lds[(rl * W + w) * M + m];
      const float rb = bf_round(r);
      const long off = (long)m * n + orow;
      switch (mb.mode) {
        case 1: ((unsigned short*)mb.o0)[off] = f2bf_rne(silu_(mb.alpha * rb)); break;
        case 2: ((unsigned short*)mb.o0)[off] = f2bf_rne(sigmoid_(rb)); break;
        case 3: { float v = sigmoid_(mb.alpha * rb); ((float*)mb.o0)[off] = v + v; } break;
        case 4: ((float*)mb.o0)[off] = sigmoid_(r); break;
        case 5: {
          long base = orow & ~511L;
          long idx = orow - base;
          unsigned char* plane = (unsigned char*)(idx < 256 ? mb.o0 : mb.o1);
          if (idx >= 256) idx -= 256;
          *(unsigned short*)(plane + (long)m * (n / 2) * 2 + base + idx * 2) = f2bf_rne(r);
        } break;
        default: ((unsigned short*)mb.o0)[off] = f2bf_rne(r); break;
      }
    }
  }
}

#define LQ8WG(M) \
  extern "C" __global__ void __launch_bounds__(256) lq8w_grp_m##M##_r32_w2(LqGrp g, const unsigned short* x, long K) { lq8w_grp_body<M>(g, x, K); }
LQ8WG(1) LQ8WG(2) LQ8WG(3) LQ8WG(4) LQ8WG(5) LQ8WG(6) LQ8WG(7) LQ8WG(8)
