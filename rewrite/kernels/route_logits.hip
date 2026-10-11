// Rewrites of the "route_logits" family (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o route_logits.co route_logits.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). ".long 0xd4.. / 0xd7.." pairs and the VOP1 words
// 0x7e04.... / 0x7f04.... are encodings llvm-objdump rejects (a 2-operand VOP3 with inline-0 in src2,
// and true16 v_mov_b16 / v_lshrrev_b16): 0xd73c.... v_lshlrev_b64, 0xd7006a../0xd700.... v_add_co_u32,
// 0xd4d3/0xd4d4 v_cmpx_*_i64, 0xd4c9 v_cmpx_lt_u32, 0xd7390002 v_lshrrev_b16 v2, 1, v2, 0x7e043810
// v_mov_b16 v2.l, s16 (blockDim.x), 0x7f043880 v_mov_b16 v2.h, 0.
#include <hip/hip_runtime.h>

// bf16 -> f32 is a left shift by 16 (v_lshlrev_b32 16).
__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// Every plain variant is one f32 dot product per (token, expert), accumulated by v_fmac_f32 over k with a
// blockDim.x-strided loop, then reduced through LDS by halving: for (s = blockDim/2; s != 0; s >>= 1)
// if (tid < s) a[tid] += a[tid+s], with a barrier after every step.

// halogen::(anonymous namespace)::k_route_logits_w32(float const*, float const*, float*, long)
// _ZN7halogen12_GLOBAL__N_118k_route_logits_w32EPKfS2_Pfl, kernarg: x@0 (f32), w@8, out@16, K@24
// grid = (tokens, experts): x row = x + bx*K (s_mul_i32/s_mul_hi_u32 of K and s2, codeobj_53000.s:18263-18272),
// w row = w + by*K (s4 = blockIdx.y, :18259-18267). Loop k = tid; k < K; k += blockDim.x (v_cmp_le_i64 s[14:15], :18298);
// two global_load_b32 (:18292-18293), v_fmac_f32 acc += w*x (:18301). LDS tree (:18318-18342).
// Thread 0 stores out[by*gridDim.x + bx]: gridDim.x is read from the kernarg block_count_x at +32 (global_load_b32 v[3:4],
// :18348), v_mad_u64_u32 v0*by (:18353), plus bx (:18364-18366).
extern "C" __global__ void k_route_logits_w32(const float* x, const float* w, float* out, long K) {
  __shared__ float red[256];
  const int tid = threadIdx.x;
  const float* xr = x + (long)blockIdx.x * K;
  const float* wr = w + (long)blockIdx.y * K;
  float acc = 0.f;
  for (long k = tid; k < K; k += blockDim.x) acc = fmaf(wr[k], xr[k], acc);
  red[tid] = acc;
  __syncthreads();
  for (int s = blockDim.x >> 1; s != 0; s >>= 1) {
    if (tid < s) red[tid] = red[tid + s] + red[tid];
    __syncthreads();
  }
  if (tid == 0) out[(long)blockIdx.y * gridDim.x + blockIdx.x] = red[0];
}

// halogen::(anonymous namespace)::k_route_logits_x32(unsigned short const*, float const*, float*, long)
// _ZN7halogen12_GLOBAL__N_118k_route_logits_x32EPKtPKfPfl, kernarg: x@0 (bf16), w@8, out@16, K@24
// Same as w32 with a bf16 x: x row byte offset is K*bx*2 (s_lshl_b64 ..., 1, codeobj_53000.s:38353), x load
// global_load_u16 (:38375), shifted by 16 (:38384), v_fmac_f32 (:38387); the LDS tree and the store are at
// :38400-38463.
extern "C" __global__ void k_route_logits_x32(const unsigned short* x, const float* w, float* out, long K) {
  __shared__ float red[256];
  const int tid = threadIdx.x;
  const unsigned short* xr = x + (long)blockIdx.x * K;
  const float* wr = w + (long)blockIdx.y * K;
  float acc = 0.f;
  for (long k = tid; k < K; k += blockDim.x) acc = fmaf(wr[k], bf2f(xr[k]), acc);
  red[tid] = acc;
  __syncthreads();
  for (int s = blockDim.x >> 1; s != 0; s >>= 1) {
    if (tid < s) red[tid] = red[tid + s] + red[tid];
    __syncthreads();
  }
  if (tid == 0) out[(long)blockIdx.y * gridDim.x + blockIdx.x] = red[0];
}

// halogen::(anonymous namespace)::k_route_logits_x32_m<M>(unsigned short const*, float const*, float*, long, long)
// _ZN7halogen12_GLOBAL__N_120k_route_logits_x32_mILi{M}EEEvPKtPKfPfll, kernarg: x@0 (bf16), w@8, out@16, K@24, ld@32
// M tokens per block. grid = (ceil(tokens/M), experts). Listing for <8>: codeobj_53000.s:2267587-2267802.
// Token block start is bx*M (s_lshl_b64 s[2:3], s[2:3], 3 at :2267608 for M = 8; shift 2 for M = 4, 1 for M = 2); x row = x + that*K
// elements, token j is j*K elements further (2*K-byte stride). w row = w + by*K (s4 = blockIdx.y). Per k one w load and M
// bf16 x loads (global_load_u16 x8 + global_load_b32 :2267654-2267688), then M v_fmac_f32 (:2267706-2267714) into M accumulators.
// LDS: M arrays of 256 floats at j*1024 (ds_store_2addr_stride64_b32 :2267726 and the next three); the halving tree adds all M
// arrays each step (blockDim/2 from v_lshrrev_b16 0xd7390002 :2267722, v_lshrrev_b32 v2, 1, v2 :2267742, v_cmpx_lt_u32 0xd4c9 :2267752).
// Thread tid < M writes out[ld*by + M*bx + tid] (v_cmpx_gt_u32 8, v0 :2267781, v_mad_u32_u24 0x3fc = tid*1024 :2267783,
// global_store_b32 :2267798).
template <int M>
__device__ static inline void x32_m_body(const unsigned short* x, const float* w, float* out, long K, long ld) {
  __shared__ float red[M * 256];
  const int tid = threadIdx.x;
  const unsigned short* xr = x + ((long)blockIdx.x * M) * K;
  const float* wr = w + (long)blockIdx.y * K;
  float acc[M];
  for (int j = 0; j < M; j++) acc[j] = 0.f;
  for (long k = tid; k < K; k += blockDim.x) {
    const float wv = wr[k];
    for (int j = 0; j < M; j++) acc[j] = fmaf(wv, bf2f(xr[(long)j * K + k]), acc[j]);
  }
  for (int j = 0; j < M; j++) red[j * 256 + tid] = acc[j];
  __syncthreads();
  for (int s = blockDim.x >> 1; s != 0; s >>= 1) {
    if (tid < s)
      for (int j = 0; j < M; j++) red[j * 256 + tid] = red[j * 256 + tid + s] + red[j * 256 + tid];
    __syncthreads();
  }
  if (tid < M) out[ld * (long)blockIdx.y + (long)blockIdx.x * M + tid] = red[tid * 256];
}
extern "C" __global__ void k_route_logits_x32_m2(const unsigned short* x, const float* w, float* out, long K, long ld) {
  x32_m_body<2>(x, w, out, K, ld);
}
extern "C" __global__ void k_route_logits_x32_m4(const unsigned short* x, const float* w, float* out, long K, long ld) {
  x32_m_body<4>(x, w, out, K, ld);
}
extern "C" __global__ void k_route_logits_x32_m8(const unsigned short* x, const float* w, float* out, long K, long ld) {
  x32_m_body<8>(x, w, out, K, ld);
}

// halogen::(anonymous namespace)::k_route_logits_rows<R, E>(unsigned short const*, float const*, float*, long, long)
// _ZN7halogen12_GLOBAL__N_119k_route_logits_rowsILi{R}ELi{E}EEEvPKtPKfPfll, kernarg: x@0 (bf16), w@8, out@16, K@24, ld@32
// R tokens and E experts per block, grid.x = ceil(tokens/R), no grid.y. x row = x + (R*bx)*K, token j at +j*K.
// w is the base pointer for all experts: expert e is w + e*K floats (the w stride is K << 2 bytes, s_lshl_b64 s[4:5], s[10:11], 2 at
// codeobj_53000.s:963294 for <8,2>; x stride K << 1 at :963293, x block start at :963280). Per k: one bf16 x load per token and
// one f32 w load per expert (:963307-963348), accumulator j*E+e += w[e*K+k] * x[j*K+k] (v_fmac_f32 chain :963360-963377 for <8,2>);
// LDS holds R*E arrays of 256 floats at (j*E+e)*1024 (ds_store_2addr_stride64_b32 offsets 0,4,8,..., :963385 and the next seven) and
// the tree adds all of them each step (:963393-963454).
// Thread tid < R*E writes out[e*ld + R*bx + j] with j = tid / E, e = tid % E (v_cmpx_gt_u32 16, v0 :963454; v_and 1 = e, v_mad_u32_u24
// 0x3fc = tid*1024 :963457; global_store_b32 :963482).
template <int R, int E>
__device__ static inline void rows_body(const unsigned short* x, const float* w, float* out, long K, long ld) {
  __shared__ float red[R * E * 256];
  const int tid = threadIdx.x;
  const unsigned short* xr = x + ((long)blockIdx.x * R) * K;
  float acc[R * E];
  for (int a = 0; a < R * E; a++) acc[a] = 0.f;
  for (long k = tid; k < K; k += blockDim.x) {
    float wv[E];
    for (int e = 0; e < E; e++) wv[e] = w[(long)e * K + k];
    for (int j = 0; j < R; j++) {
      const float xv = bf2f(xr[(long)j * K + k]);
      for (int e = 0; e < E; e++) acc[j * E + e] = fmaf(wv[e], xv, acc[j * E + e]);
    }
  }
  for (int a = 0; a < R * E; a++) red[a * 256 + tid] = acc[a];
  __syncthreads();
  for (int s = blockDim.x >> 1; s != 0; s >>= 1) {
    if (tid < s)
      for (int a = 0; a < R * E; a++) red[a * 256 + tid] = red[a * 256 + tid + s] + red[a * 256 + tid];
    __syncthreads();
  }
  if (tid < R * E) out[(long)(tid % E) * ld + (long)blockIdx.x * R + tid / E] = red[tid * 256];
}
#define ROWS(R, E)                                                                                           \
  extern "C" __global__ void k_route_logits_rows_##R##_##E(const unsigned short* x, const float* w, float* out, \
                                                           long K, long ld) {                                \
    rows_body<R, E>(x, w, out, K, ld);                                                                       \
  }
ROWS(8, 2)
ROWS(8, 3)
ROWS(8, 4)
ROWS(4, 5)
ROWS(4, 6)
ROWS(4, 7)
ROWS(4, 8)

// ---- the "_sg" variants: the plain block above plus one extra block that computes shared-expert gates ----
//
// Cites below are lines of codeobj_53000.s for sg1<8,0> (listing 949853-950382) unless another kernel is named.
// SgArg is 48 bytes passed by value at kernarg offset 40 (by_value, size 48 in kernargs.json):
//   +0  tbl         const float*  16-entry palette: global_load_b32 at tbl + 4*(tid & 15) and tbl + 4*(tid >> 4) (:950074-950075)
//   +8  off_codes   long          code bytes start at (char*)tbl + off_codes      (s_add_u32 s3, s4, s6, :950076)
//   +16 off_scales  long          f16 scales start at (char*)tbl + off_scales    (s_add_u32 s9, s4, s12, :950093)
//   +24 aux         long          never read by the gate block (s_load_b128 at 0x28 and s_load_b64 at 0x38 skip 0x30...; 0x40 is not loaded)
//   +32 x           const bf16*   activations of the gate token(s) (s_load_b128 s[4:7], 0x48, :950140)
//   +40 out         float*        sigmoid outputs                          (global_store_b32 v8, v0, s[6:7], :950382)
struct SgArg {
  const float* tbl;
  long off_codes;
  long off_scales;
  long aux;
  const unsigned short* x;
  float* out;
};

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
__device__ static inline float dot2_bf16(unsigned xpair, unsigned wpair, float acc) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2, xpair), __builtin_bit_cast(bf16x2, wpair), acc, false);
}
// f32 -> bf16 bits, round to nearest even with no NaN special case: v_bfe_u32 (u >> 16) & 1 ; v_add3_u32 u + bit + 0x7fff
// (:950099-950105).
__device__ static inline unsigned f2bf_bits(float f) {
  unsigned u = __float_as_uint(f);
  return (u + ((u >> 16) & 1u) + 0x7fffu) >> 16;
}

// Gate block, 256 threads = 8 wave32. Entered when blockIdx.x == arg32 / R (signed division, truncating): the compare
// `arg32 >> k == s2` at the top of every _sg kernel (s_cmp_lg_u64 :949867, s_cbranch_scc0 :949869 jumps here when equal).
//   LUT[256] in LDS: LUT[t] = bf16(tbl[t >> 4]) << 16 | bf16(tbl[t & 15])   (two bf16 roundings, v_bfe_u32/v_add3_u32 0x7fff
//   :950099-950105, v_mov_b16 pair :950108, v_and_or_b32 0xffff0000 and ds_store_b32 offset:8224 :950115)
//   G = K / 32 groups (s_lshr_b64 by 5 with the sign correction, :950084); per = ceil(G / 8) (:950086-950090)
//   wave w covers groups [w*per, min(G, w*per + per)) (v_mul 0xd72c0005 :950094, v_min_i32 :950110); lane l takes
//   g = start + l, +32, +64, ... (v_cmp_lt_i32 :950114, v_add_nc_u32 32 near :950262)
//   per group: 16 code bytes (global_load_b128 at codes + 16 g, :950136), f16 scale (global_load_d16_b16 + v_cvt_f32_f16 at
//   scales + 2 g, :950135 and :950138), 64 bytes of x (4 x global_load_b128, :950173-950176);
//   d = chain over i = 0..15 of v_dot2_f32_bf16(xword[i], LUT[codebyte[i]], d) starting from 0 (:950196-950257),
//   then acc += scale * d with v_fmac_f32 (:950259).
//   wave reduction: five xor steps 16, 8, 4, 2, 1, v = v + partner (ds_bpermute_b32 :950308-950329, v_add_f32 :950310-950334);
//   lane 0 of each wave stores to LDS[w] (ds_store_b32 :950335); barrier; thread 0 sums ((0 + w0) + w1) + ... + w7
//   (ds_load_b128 x2 :950344-950345, v_add_f32 :950347-950359);
//   p = 1 / (1 + exp2(sum * -1.442695)) with IEEE division (v_mul 0xbfb8aa3b :950360, v_exp_f32 :950362, v_add 1.0 :950363,
//   div sequence to v_div_fixup_f32 :950381); out[0] = p.
// rows_sg<R,E,b> repeats the gate block over E activation vectors (x + e*K elements, the byte stride K << 1 is
// s_lshl_b64 s[2:3], s[10:11], 1 at :951529 of rows_sg<8,2,0>): E accumulators (v_fmac_f32 v18 and v14, :951669-951671),
// per-wave partials stored as a pair (ds_store_b64 :951759), and thread e < E sums the 8 waves for vector e
// (ds_load_2addr_b32 :951768-951775) and stores out[e] (global_store_b32 v11, v0, s[6:7], :951808).
// B is the second template argument of the vendor kernel. It selects code that preloads scales when G <= 2048; the
// arithmetic order is the same as far as the tests show (see reports/g06.md).
template <int B, int E>
__device__ static inline void sg_block(const SgArg& sg, long K) {
  __shared__ unsigned lut[256];
  __shared__ float wsum[8 * E];
  const int tid = threadIdx.x;
  lut[tid] = (f2bf_bits(sg.tbl[tid >> 4]) << 16) | f2bf_bits(sg.tbl[tid & 15]);
  __syncthreads();
  const char* base = (const char*)sg.tbl;
  const uint4* codes = (const uint4*)(base + sg.off_codes);
  const unsigned short* scales = (const unsigned short*)(base + sg.off_scales);
  const long G = K >> 5;
  const long per = (G + 7) >> 3;
  const int wv = tid >> 5, lane = tid & 31;
  const long start = (long)wv * per;
  const long stop = (start + per < G) ? start + per : G;
  float acc[E];
  for (int e = 0; e < E; e++) acc[e] = 0.f;
  for (long g = start + lane; g < stop; g += 32) {
    const uint4 c = codes[g];
    const float sc = (float)__builtin_bit_cast(_Float16, scales[g]);
    const unsigned cw[4] = {c.x, c.y, c.z, c.w};
    for (int e = 0; e < E; e++) {
      const uint4* xp = (const uint4*)(sg.x + (long)e * K + g * 32);
      const uint4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
      const unsigned xw[16] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w,
                               x2.x, x2.y, x2.z, x2.w, x3.x, x3.y, x3.z, x3.w};
      float d = 0.f;
      for (int i = 0; i < 16; i++) d = dot2_bf16(xw[i], lut[(cw[i >> 2] >> ((i & 3) * 8)) & 0xffu], d);
      acc[e] = fmaf(sc, d, acc[e]);
    }
  }
  for (int e = 0; e < E; e++) {
    float v = acc[e];
    for (int m = 16; m >= 1; m >>= 1) v = v + __shfl_xor(v, m, 32);
    if (lane == 0) wsum[wv * E + e] = v;
  }
  __syncthreads();
  if (tid < E) {
    float s = 0.0f + wsum[tid];
    for (int w = 1; w < 8; w++) s = s + wsum[w * E + tid];
    const float ex = __builtin_amdgcn_exp2f(s * -1.44269502162933349609375f);
    sg.out[tid] = 1.0f / (1.0f + ex);
  }
}

// halogen::(anonymous namespace)::k_route_logits_sg1<R, B>(unsigned short const*, float const*, float*, long, long, SgArg)
// kernarg: x@0 (bf16), w@8, out@16, K@24, M@32, SgArg@40. R tokens per plain block; plain block bx (< M/R) is the x32_m<R>
// body without grid.y: one logit per token, w is the single row at w (not offset), out[R*bx + tid] for tid < R
// (sg1<8,0>: v_cmpx_gt_u32 8 and global_store_b32 v2, v1, s[2:3] at :950054 with s[2:3] = out + 4*8*bx).
template <int R>
__device__ static inline void sg1_plain(const unsigned short* x, const float* w, float* out, long K) {
  __shared__ float red[R * 256];
  const int tid = threadIdx.x;
  const unsigned short* xr = x + ((long)blockIdx.x * R) * K;
  float acc[R];
  for (int j = 0; j < R; j++) acc[j] = 0.f;
  for (long k = tid; k < K; k += blockDim.x) {
    const float wv = w[k];
    for (int j = 0; j < R; j++) acc[j] = fmaf(wv, bf2f(xr[(long)j * K + k]), acc[j]);
  }
  for (int j = 0; j < R; j++) red[j * 256 + tid] = acc[j];
  __syncthreads();
  for (int s = blockDim.x >> 1; s != 0; s >>= 1) {
    if (tid < s)
      for (int j = 0; j < R; j++) red[j * 256 + tid] = red[j * 256 + tid + s] + red[j * 256 + tid];
    __syncthreads();
  }
  if (tid < R) out[(long)blockIdx.x * R + tid] = red[tid * 256];
}
#define SG1(R, B)                                                                                                \
  extern "C" __global__ void k_route_logits_sg1_##R##_##B(const unsigned short* x, const float* w, float* out, \
                                                          long K, long M, SgArg sg) {                          \
    if ((long)blockIdx.x == M / R) sg_block<B, 1>(sg, K);                                                      \
    else sg1_plain<R>(x, w, out, K);                                                                           \
  }
SG1(2, 0)
SG1(2, 1)
SG1(4, 0)
SG1(4, 1)
SG1(8, 0)
SG1(8, 1)

// halogen::(anonymous namespace)::k_route_logits_rows_sg<R, E, B>(..., SgArg): the rows<R,E> body for blockIdx.x < ld / R,
// the gate block for blockIdx.x == ld / R. ld is kernarg 32 (same compare as above).
#define ROWS_SG(R, E, B)                                                                                           \
  extern "C" __global__ void k_route_logits_rows_sg_##R##_##E##_##B(const unsigned short* x, const float* w,       \
                                                                    float* out, long K, long ld, SgArg sg) {       \
    if ((long)blockIdx.x == ld / R) sg_block<B, E>(sg, K);                                                         \
    else rows_body<R, E>(x, w, out, K, ld);                                                                        \
  }
ROWS_SG(8, 2, 0)
ROWS_SG(8, 2, 1)
ROWS_SG(8, 3, 0)
ROWS_SG(8, 3, 1)
ROWS_SG(8, 4, 0)
ROWS_SG(8, 4, 1)
ROWS_SG(4, 5, 0)
ROWS_SG(4, 5, 1)
ROWS_SG(4, 6, 0)
ROWS_SG(4, 6, 1)
ROWS_SG(4, 7, 0)
ROWS_SG(4, 7, 1)
ROWS_SG(4, 8, 0)
ROWS_SG(4, 8, 1)
