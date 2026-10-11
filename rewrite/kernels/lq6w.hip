// Rewrites of k_lq6w (planar 6-bit affine GEMV, 32 elements per lane, M tokens) and k_lq6w_grp from codeobj_53000
// (flash_serve, gfx1151). Listing: kernels-disasm/codeobj_53000.s (not in git); slices work/kernels-dequant/*k_lq6w*.s.
// Line numbers are slice-relative for k_lq6wILi1ELi16ELi1E (starts at 0x331900) unless stated.
#include <hip/hip_runtime.h>
// hipcc fuses a*b+c by default; the listing uses separate v_mul/v_add (explicit fmaf where fused).
#pragma clang fp contract(off)

typedef __bf16 v2bf __attribute__((ext_vector_type(2)));

// f32 -> bf16 RNE, no NaN special case: v_bfe_u32 x,16,1 ; v_add3_u32 x,x,b,0x7fff (same sequence as k_lq6k / k_deq_q6g64).
__device__ static inline unsigned f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (u + ((u >> 16) & 1u) + 0x7fffu) >> 16;
}

// One 32-element unit of the planar Q6 g64 row layout (same as k_deq_q6g64, see deq_q6g64.hip), dotted with 32 bf16 of x.
//   Row layout: ql plane (K/2 bytes), qh plane (K/4), sb plane (K/16: f16 scale, f16 bias per 64 elements), row stride aligned to 16.
//   Loads for unit c: ql b128 at c*16, qh b64 at K/2 + c*8, sb b32 at K/2+K/4 + 4*(c>>1)  (:80-82 loads; :139-141 v_lshl_or 16/8/..)
//   Element e: q = ((qh >> 2e) & 3) << 4 | nibble e of ql (v_and_or ..,15 / ..,48 patterns :113-190),
//   w = bf16_rne(fma((float)q, scale, bias)) (v_fma_mix_f32 x32 per unit), pairs (even low, odd high).
//   acc = dot2_f32_bf16 chain over the 16 pairs, first pair from 0 (:356-421, 16 v_dot2_f32_bf16).
__device__ static inline float unit_dot(const unsigned char* ql, const unsigned char* qh, const unsigned char* sbp, const unsigned* xw) {
  unsigned qlw[4];
  unsigned long long qhw;
  unsigned sb;
  __builtin_memcpy(qlw, ql, 16);
  __builtin_memcpy(&qhw, qh, 8);
  __builtin_memcpy(&sb, sbp, 4);
  float scale = (float)__builtin_bit_cast(_Float16, (unsigned short)(sb & 0xffffu));
  float bias = (float)__builtin_bit_cast(_Float16, (unsigned short)(sb >> 16));
  float acc = 0.0f;
  #pragma unroll
  for (int p = 0; p < 16; p++) {
    unsigned wb[2];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
      int e = 2 * p + j;
      unsigned lo = (qlw[e >> 3] >> (4 * (e & 7))) & 15u;
      unsigned hi = (unsigned)(qhw >> (2 * e)) & 3u;
      wb[j] = f2bf_rne(__builtin_fmaf((float)(int)((hi << 4) | lo), scale, bias));
    }
    unsigned wp = (wb[0] & 0xffffu) | (wb[1] << 16);
    acc = __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(v2bf, xw[p]), __builtin_bit_cast(v2bf, wp), acc, false);
  }
  return acc;
}

// Generic k_lq6w<M, R, W>(w, x, y, K, N)   kernarg: w@0 x@8 y@16 (bf16) K@24 N@32, block 256 threads, wave32.
//   Same geometry as k_lq6k (see lq6k.hip) with units of 32 elements: U = K/32 units (:19 s_lshr 5), row stride S as in deq_q6g64
//   (:14-37), unit c = beg + lane + R*j, x of unit at x + c*32 (4 x b128, :116-119), token m at x + m*K, y[m*N + row] (bf16 RNE store).
//   Rows per block 256/(R*W). Wave range for W>1: [w*ceil(U/W), min(U, ...)).
template <int M, int R, int W>
__device__ static inline void lq6w_body(const unsigned char* w, const unsigned short* x, unsigned short* y, long K, long N) {
  constexpr int RPB = 256 / (R * W);
  __shared__ float part[M * RPB * W];
  int tid = threadIdx.x;
  int rib = tid / (R * W);
  long row = (long)blockIdx.x * RPB + rib;
  int wave = (tid / 32) % W;
  int lane = tid % R;
  long U = K / 32;
  long S = (K / 2 + K / 4 + K / 16 + 15) & ~15L;
  long chunk = (U + W - 1) / W;
  long beg = (W == 1) ? 0 : wave * chunk;
  long end = (W == 1) ? U : (beg + chunk < U ? beg + chunk : U);
  float acc[M];
  #pragma unroll
  for (int m = 0; m < M; m++) acc[m] = 0.0f;
  if (row < N) {
    const unsigned char* wr = w + row * S;
    for (long c = beg + lane; c < end; c += R) {
      #pragma unroll
      for (int m = 0; m < M; m++) {
        unsigned xw[16];
        __builtin_memcpy(xw, x + (long)m * K + c * 32, 64);
        acc[m] += unit_dot(wr + c * 16, wr + K / 2 + c * 8, wr + K / 2 + K / 4 + ((c * 2) & ~3L), xw);
      }
    }
  }
  #pragma unroll
  for (int m = 0; m < M; m++) {
    if (R == 32) acc[m] += __shfl_xor(acc[m], 16);
    acc[m] += __shfl_xor(acc[m], 8);
    acc[m] += __shfl_xor(acc[m], 4);
    acc[m] += __shfl_xor(acc[m], 2);
    acc[m] += __shfl_xor(acc[m], 1);
  }
  float out[M];
  if (W == 1) {
    #pragma unroll
    for (int m = 0; m < M; m++) out[m] = acc[m];
  } else {
    if ((tid & 31) == 0) {
      #pragma unroll
      for (int m = 0; m < M; m++) part[(m * RPB + rib) * W + wave] = acc[m];
    }
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < M; m++) {
      float s = 0.0f;
      #pragma unroll
      for (int k = 0; k < W; k++) s += part[(m * RPB + rib) * W + k];
      out[m] = s;
    }
  }
  if (row < N && (tid % (R * W)) == 0) {
    #pragma unroll
    for (int m = 0; m < M; m++) y[(long)m * N + row] = (unsigned short)f2bf_rne(out[m]);
  }
}

#define LQ6W_ARGS const unsigned char* w, const unsigned short* x, unsigned short* y, long K, long N
#define LQ6W_ONE(M, R, W) \
  extern "C" __global__ void lq6w_##M##_##R##_##W(LQ6W_ARGS) { lq6w_body<M, R, W>(w, x, y, K, N); }
#define LQ6W_M(M) LQ6W_ONE(M, 16, 1) LQ6W_ONE(M, 32, 1) LQ6W_ONE(M, 32, 2) LQ6W_ONE(M, 32, 8)
LQ6W_M(1) LQ6W_M(2) LQ6W_M(3) LQ6W_M(4) LQ6W_M(5) LQ6W_M(6) LQ6W_M(7) LQ6W_M(8)

// ---------------------------------------------------------------------------------------------------------------
// k_lq6w_grp<M, 32, W>(LqGrp g, const u16* x, long K)   (slices work/kernels-dequant/*k_lq6w_grp*.s; line numbers below are
// slice-relative for k_lq6w_grpILi1ELi32ELi2E, starts at 0x36e600, and ...Li2ELi32ELi2E (starts 0x36f900) for the M=2 notes).
// kernarg: LqGrp by value @0 (296 bytes), x@296, K@304. Up to 4 members of 0x48 bytes, count at 0x120 (:4).
//   member j (base 0x48*j): w@0x00, y1@0x08 (:645 s_load_b64 [0x8]), y2@0x10 (mode 5 only, :618), rows N (i64)@0x30 (:39),
//   scale (f32)@0x38, mode (i32)@0x3c (s_load_b64 s[2:3],[0x38], :603), first block index@0x40 (:40). Bytes 0x18..0x2f are not read.
//   Member select (:4-33): j = 0; j = 1 if count>1 && blockIdx >= start[1]; j = 2 if count>2 && blockIdx >= start[2]; j = 3 likewise
//   (start[] read from +0x40 of members 1..3 at kernarg offsets 0x88, 0xd0, 0x118). local block = blockIdx - start[j] (:40 s_sub).
//   Waves per row W_eff: v_cmp_lt_i64 s17, N, 9 (VOP3 word d4510011 / 0201120a, :43-44) ; s16 = s17 ? 8 : 2 (:46). For W=4 the compare is
//   v_cmp_gt_i64 N, 8 (d454000f / 0201100a) and W_eff = N>8 ? 4 : 8 (W=4 slice :48); W=8 is always 8. So W_eff = (N > 8) ? W : 8.
//   Rows per block = 8/W_eff (shift s13 = 0 for 8, 2 for 2 (:47-52)), row = (local << (3-log2 W_eff)) + ((tid>>5) >> log2 W_eff) (:12,:55).
//   Unit loop exactly as k_lq6w: U = K/32, chunk = ceil(U/W_eff), wave range [w*chunk, min(U, w*chunk+chunk)) (:70-92, v_min :113).
//   Wave reduction xor 16,8,4,2,1 (:540-568), lane 0 -> LDS (:576), barrier (:582). Then threads t < rows_per_block*M (:578 s_cselect 1/4 for
//   M=1; M=2 slice :628 selects 2/8): row = base_row + t/M, token = t%M (M=2 slice :634-640), sum W_eff partials in wave order starting from 0.0
//   (:595-601), valid if row < N, and the epilogue by mode (:603-768), outputs indexed token*N + row (M=2 slice :770-830, v_mul_hi / v_mad of N * token):
//     mode 0: y1 bf16[tok*N+row]  = rne(acc)                                  (:726 area)
//     mode 1: x = bf16val(acc) * scale ; y1 bf16 = rne( x / (1 + exp2(x * -log2e)) )   (:741-768)
//     mode 2: x = bf16val(acc) ; y1 bf16 = rne( 1 / (1 + exp2(x * -log2e)) )   (:700-727)
//     mode 3: x = bf16val(acc) * scale ; y1 f32 = 2 * (1 / (1 + exp2(x * -log2e)))  (:668-695, v_add_f32 v0,v0,v0)
//     mode 4: y1 f32 = 1 / (1 + exp2(acc * -log2e))                             (:647-666)
//     mode 5: bf16 rne(acc) split store: base = ((row & 0x1ff) < 0x100) ? y1 : y2; element index = tok*(N/2) + (row & ~0x1ff)/2 + (row & 0xff)
//             (:618-643: v_cmp_gt_u64 0x100,(row&0x1ff) ; v_cndmask on the pointer ; byte offset (row & 0xfffffe00) + (row&0xff)*2 ; token stride N>>1 in the M=2 slice)
//     mode > 5: no store (:612-616, s_cmp_eq_u32 s3,5 else skip)
//   bf16val(acc) = (rne_bits(acc) & 0xffff0000) as float. exp2 is v_exp_f32; division is the IEEE div_scale/rcp/fma/div_fmas/div_fixup sequence.
struct LqMember {
  const unsigned char* w;
  unsigned short* y1;
  unsigned short* y2;
  char pad0[24];
  long n;
  float scale;
  int mode;
  int start;
  int pad1;
};
struct LqGrp {
  LqMember m[4];
  int count;
};

__device__ static inline float bf16val(float f) { return __uint_as_float(f2bf_rne(f) << 16); }
__device__ static inline float sigm(float v) {  // 1 / (1 + exp2(v * -log2e)), v_mul by 0xbfb8aa3b then v_exp_f32
  float e = __builtin_amdgcn_exp2f(v * -1.44269502162933349609375f);
  return 1.0f / (1.0f + e);
}

template <int M, int W>
__device__ static inline void lq6w_grp_body(const LqGrp& g, const unsigned short* x, long K) {
  __shared__ float part[8 * M];
  int tid = threadIdx.x;
  int bx = blockIdx.x;
  int j = 0;
  if (g.count > 1 && bx >= g.m[1].start) j = 1;
  if (g.count > 2 && bx >= g.m[2].start) j = 2;
  if (g.count > 3 && bx >= g.m[3].start) j = 3;
  const LqMember& mb = g.m[j];
  long N = mb.n;
  int local = bx - mb.start;
  int weff = (W == 8) ? 8 : (N > 8 ? W : 8);
  int lg = weff == 8 ? 3 : (weff == 4 ? 2 : 1);
  int rshift = 3 - lg;
  int wv = tid >> 5;
  int rib = wv >> lg;
  int wir = wv & (weff - 1);
  long row = ((long)local << rshift) + rib;
  int lane = tid & 31;
  long U = K / 32;
  long S = (K / 2 + K / 4 + K / 16 + 15) & ~15L;
  long chunk = (U + weff - 1) / weff;
  long beg = wir * chunk;
  long end = beg + chunk < U ? beg + chunk : U;
  float acc[M];
  #pragma unroll
  for (int m = 0; m < M; m++) acc[m] = 0.0f;
  if (row < N) {
    const unsigned char* wr = mb.w + row * S;
    for (long c = beg + lane; c < end; c += 32) {
      #pragma unroll
      for (int m = 0; m < M; m++) {
        unsigned xw[16];
        __builtin_memcpy(xw, x + (long)m * K + c * 32, 64);
        acc[m] += unit_dot(wr + c * 16, wr + K / 2 + c * 8, wr + K / 2 + K / 4 + ((c * 2) & ~3L), xw);
      }
    }
  }
  #pragma unroll
  for (int m = 0; m < M; m++) {
    acc[m] += __shfl_xor(acc[m], 16);
    acc[m] += __shfl_xor(acc[m], 8);
    acc[m] += __shfl_xor(acc[m], 4);
    acc[m] += __shfl_xor(acc[m], 2);
    acc[m] += __shfl_xor(acc[m], 1);
  }
  if (lane == 0) {
    #pragma unroll
    for (int m = 0; m < M; m++) part[(rib * weff + wir) * M + m] = acc[m];
  }
  __syncthreads();
  int rpb = 8 >> lg;
  if (tid < rpb * M) {
    int r2 = tid / M, tok = tid % M;
    long rw = ((long)local << rshift) + r2;
    if (rw < N) {
      float a = 0.0f;
      for (int k = 0; k < weff; k++) a += part[(r2 * weff + k) * M + tok];
      long to = (long)tok * N;
      float sc = mb.scale;
      switch (mb.mode) {
        case 0: mb.y1[to + rw] = (unsigned short)f2bf_rne(a); break;
        case 1: { float v = bf16val(a) * sc; mb.y1[to + rw] = (unsigned short)f2bf_rne(v / (1.0f + __builtin_amdgcn_exp2f(v * -1.44269502162933349609375f))); break; }
        case 2: mb.y1[to + rw] = (unsigned short)f2bf_rne(sigm(bf16val(a))); break;
        case 3: { float s = sigm(bf16val(a) * sc); ((float*)mb.y1)[to + rw] = s + s; break; }
        case 4: ((float*)mb.y1)[to + rw] = sigm(a); break;
        case 5: {
          unsigned short* base = (rw & 0x1ff) < 0x100 ? mb.y1 : mb.y2;
          base[(long)tok * (N >> 1) + ((rw & ~0x1ffL) >> 1) + (rw & 0xff)] = (unsigned short)f2bf_rne(a);
          break;
        }
        default: break;
      }
    }
  }
}

#define LQ6W_GRP_ONE(M, W) \
  extern "C" __global__ void lq6w_grp_##M##_##W(LqGrp g, const unsigned short* x, long K) { lq6w_grp_body<M, W>(g, x, K); }
#define LQ6W_GRP_M(M) LQ6W_GRP_ONE(M, 2) LQ6W_GRP_ONE(M, 4) LQ6W_GRP_ONE(M, 8)
LQ6W_GRP_M(1) LQ6W_GRP_M(2) LQ6W_GRP_M(3) LQ6W_GRP_M(4) LQ6W_GRP_M(5) LQ6W_GRP_M(6) LQ6W_GRP_M(7) LQ6W_GRP_M(8)
