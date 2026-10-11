// Rewrites of the "expert" family from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o expert.co expert.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). VOP3 instructions appear as ".long <dword1>" plus a
// mis-decoded second dword; they are decoded by hand.
// Cite note: kernel start/end lines in codeobj_53000.s are exact (see rewrite/reports/g22.md). Line numbers inside a kernel that are written
// as :NNNNN were read off a filtered re-listing; for the long unrolled kernels they can be off by tens of lines. Instruction names are exact.
#include <hip/hip_runtime.h>

// Keep float multiply and add separate unless an explicit __builtin_fmaf is written (hipcc fuses a*b+c by default).
#pragma clang fp contract(off)

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// f32 -> bf16, RNE without a NaN special case: v_bfe_u32 v1,v0,16,1 ; v_add_nc_u32 ; v_add_nc_u32 0x7fff ; store hi half
// (k_expert_h, codeobj_53000.s:38612-38616 region; k_expert_out v_bfe_u32/v_add3_u32 0x7fff).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// silu(g) = g / (1 + exp2(g * -1.442695)): v_mul_f32 0xbfb8aa3b, v_exp_f32, v_add_f32 1.0, IEEE div sequence
// (div_scale/rcp/fma/div_fmas/div_fixup), same as scale.hip.
__device__ static inline float silu(float g) {
  float e = __builtin_amdgcn_exp2f(g * -1.44269502162933349609375f);
  return g / (1.0f + e);
}

// halogen::(anonymous namespace)::k_expert_h(unsigned short const*, unsigned short const*, int const*, unsigned short*)
// _ZN7halogen12_GLOBAL__N_110k_expert_hEPKtS2_PKiPt  kernarg: x@0 (bf16 [tok][2560]), W@8 (bf16), sel@16, out@24
// Grid (640, nslots): j = blockIdx.x (s2), slot sl = blockIdx.y (s3). tok = sl / 10 (magic 0x66666667 >> 2, :38470-38477),
// e = sel[sl] (:38489 s_load_b32), x row at tok*0x1400 B (:38479), W[e] = 1280 rows of 2560 bf16 (e * 0x640000 B, :38494),
// gate row j at +j*0x1400 B, up row at +0x320000 B (= 640 rows; v_cndmask-decoded v_add_co_u32 0x320000 :38508).
// Each thread: for i = tid; i <= 0x9ff; i += blockDim.x (v_cmp_lt_u32 0x9ff < i, :38524): xv = bf(x[i]);
//   accg = fma(xv, bf(Wgate[i]), accg); accu = fma(xv, bf(Wup[i]), accu)  (v_fmac_f32 v5 / v7, :38520-38521).
// LDS: lds[tid] = accg, lds[256 + tid] = accu (ds_store_2addr_stride64 offset1:4, :38532), tree s = bs/2, bs/4, .. 1:
//   if tid < s: lds[tid] += lds[tid+s] for both halves (:38538-38551), barrier each step.
// Thread 0: g = lds[0], u = lds[256]; out[sl*640 + j] = bf16_rne(u * silu(g)) (v_mul_f32 v0, v1, v0 :38580; store :38585).
extern "C" __global__ void k_expert_h(const unsigned short* x, const unsigned short* W, const int* sel,
                                      unsigned short* out) {
  __shared__ float lds[512];
  int j = blockIdx.x, sl = blockIdx.y, tid = threadIdx.x, bs = blockDim.x;
  int tok = sl / 10;
  int e = sel[sl];
  const unsigned short* xr = x + (long)tok * 2560;
  const unsigned short* wg = W + (long)e * 3276800 + (long)j * 2560;
  const unsigned short* wu = wg + 1638400;
  float accg = 0.0f, accu = 0.0f;
  for (unsigned i = tid; i <= 0x9ffu; i += bs) {
    float xv = bf2f(xr[i]);
    accg = __builtin_fmaf(xv, bf2f(wg[i]), accg);
    accu = __builtin_fmaf(xv, bf2f(wu[i]), accu);
  }
  lds[tid] = accg;
  lds[256 + tid] = accu;
  __syncthreads();
  for (int s = bs >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      lds[tid] = lds[tid + s] + lds[tid];
      lds[256 + tid] = lds[256 + tid + s] + lds[256 + tid];
    }
    __syncthreads();
  }
  if (tid == 0) out[(long)sl * 640 + j] = f2bf_rne(lds[256] * silu(lds[0]));
}

// halogen::(anonymous namespace)::k_expert_out(unsigned short const*, unsigned short const*, int const*, float const*, unsigned short*)
// _ZN7halogen12_GLOBAL__N_112k_expert_outEPKtS2_PKiPKfPt  kernarg: h@0 (bf16 [slot][640]), W@8 (bf16), sel@16, wt@24 (f32 per slot), out@32
// Grid (2560, ntok): j = blockIdx.x (s2), tok = blockIdx.y (s4). W[e] = 2560 rows of 640 bf16 (0x320000 B per expert, :46449).
// acc = 0 (v11); for m = 0..9: s = tok*10 + m; e = sel[s] (:46440 s_load_b32); partial = sum over i = tid; i <= 0x27f; i += bs of
//   fma(bf(h[s*640+i]), bf(W[e][j*640+i]), partial) (v_fmac_f32 v12, v15, v14 :46455); lds[tid] = partial, barrier, tree
//   s = bs/2 .. 1 (lds[tid] = lds[tid+s] + lds[tid], :46488-46497); thread 0: acc = fma(wt[s], lds[0], acc) (v_fmac_f32 v11, s8, v5 :46473).
//   barrier between m iterations (:46427).
// Thread 0: out[tok*2560 + j] = bf16_rne(acc) (v_bfe_u32/v_add3_u32 0x7fff, global_store_d16_hi_b16 :46510-46525).
extern "C" __global__ void k_expert_out(const unsigned short* h, const unsigned short* W, const int* sel, const float* wt,
                                        unsigned short* out) {
  __shared__ float lds[256];
  int j = blockIdx.x, tok = blockIdx.y, tid = threadIdx.x, bs = blockDim.x;
  float acc = 0.0f;
  for (int m = 0; m < 10; m++) {
    int s = tok * 10 + m;
    int e = sel[s];
    const unsigned short* hp = h + (long)s * 640;
    const unsigned short* wp = W + (long)e * 1638400 + (long)j * 640;
    float part = 0.0f;
    for (unsigned i = tid; i <= 0x27fu; i += bs) part = __builtin_fmaf(bf2f(hp[i]), bf2f(wp[i]), part);
    lds[tid] = part;
    __syncthreads();
    for (int st = bs >> 1; st > 0; st >>= 1) {
      if (tid < st) lds[tid] = lds[tid + st] + lds[tid];
      __syncthreads();
    }
    if (tid == 0) acc = __builtin_fmaf(wt[s], lds[0], acc);
    __syncthreads();
  }
  if (tid == 0) out[(long)tok * 2560 + j] = f2bf_rne(acc);
}

// ---------------------------------------------------------------------------------------------------------------
// Warp (one 32-lane wave per output row) quantized expert kernels.
// Weight value of element i of a row, as the vendor computes it before the multiply with x:
//   LUT form (k_expert_h_warp, k_expert_h_grouped_warp): nibble n of byte i>>1 (even i low, odd i high), f16 scale per 32
//     elements; w = bf16_rne(fma(f16 scale, lut[n], -0.0)) with lut = the first 16 floats of the weight blob (staged in LDS
//     by tid < 16: global_load_b32 + ds_store_b32, codeobj_53000.s:46533-46538; v_fma_mix_f32 ..., neg(0) :46737).
//   AFF form (..._aff<4|5,true>), AffDev as in deq_aff.hip: nibble (+ 5th bit from a plane for BITS=5), group u16 {sc,m},
//     u32 {d, dmin} per 8 groups; w = bf16_rne(fma(n, fma(d,sc,-0.0), fma(dmin,-m,-0.0))).
// The vendor then does acc = fma(x_i, w_i, acc) with the bf16 values widened, lane l visiting i = l + 32t in ascending t,
// then a five-step cross-lane reduction (ds_bpermute partners lane|16, +8 (lane<24), +4 (lane<28), +2 (lane<30), +1 (lane!=31)),
// and lane 0 finishes silu(gate) * up.
// ---------------------------------------------------------------------------------------------------------------
struct AffDev { unsigned long long o[6]; };

__device__ static inline float h2f(unsigned h) { return (float)__builtin_bit_cast(_Float16, (unsigned short)h); }

__device__ static inline float wred(float v) {
  int l = threadIdx.x & 31;
  v = v + __shfl(v, l | 16);
  v = v + __shfl(v, l + (l < 24 ? 8 : 0));
  v = v + __shfl(v, l + (l < 28 ? 4 : 0));
  v = v + __shfl(v, l + (l < 30 ? 2 : 0));
  v = v + __shfl(v, l + (l != 31 ? 1 : 0));
  return v;
}

// Row descriptors (byte pointers already advanced to the row).
struct LutRow { const unsigned char* q; const unsigned char* s; };
struct AffRow { const unsigned char* q; const unsigned char* plane; const unsigned char* sc; const unsigned char* dm; };

__device__ static inline float w_lut(const LutRow& r, const float* lut, int i) {
  unsigned byte = r.q[i >> 1];
  unsigned n = (i & 1) ? (byte >> 4) : (byte & 15u);
  float sc = h2f(((const unsigned short*)r.s)[i >> 5]);
  return bf2f(f2bf_rne(__builtin_fmaf(sc, lut[n], -0.0f)));
}

template <int BITS>
__device__ static inline float w_aff(const AffRow& r, int i) {
  unsigned byte = r.q[i >> 1];
  unsigned n = (i & 1) ? (byte >> 4) : (byte & 15u);
  if (BITS == 5) n |= ((((const unsigned*)r.plane)[i >> 5] >> (i & 31)) & 1u) << 4;
  unsigned u = ((const unsigned short*)r.sc)[i >> 5];
  unsigned dm = ((const unsigned*)r.dm)[i >> 8];
  float a = __builtin_fmaf(h2f(dm & 0xffffu), (float)(u & 0xffu), -0.0f);
  float b = __builtin_fmaf(h2f(dm >> 16), -(float)(u >> 8), -0.0f);
  return bf2f(f2bf_rne(__builtin_fmaf((float)n, a, b)));
}

// k_expert_h_warp(x@0, W@8, A@16, B@24, C@32, qoff@40, soff@48, sstride@56, out@64)
// Grid (80, ngroups), block 256. Group g = blockIdx.y covers slots C[B[g] .. B[g+1]) of expert e = A[g] (loads at :46547-46556).
// j = blockIdx.x*8 + (tid>>5) (v_lshl_add_u32 v1, s2, 3, v1 :46565), no bound check. Row = e*1280 + j; gate row at
// W + qoff + row*1280, up row 640 rows (0xc8000 B) later; scales at W + soff + row*sstride, up 640*sstride later (v_mad_u64_u32 :46590-46600).
// Per slot p: slot = C[p], tok = slot/10, x row = x + tok*2560. out[slot*640 + j] = bf16_rne(up * silu(gate)) written by lane 0 (:46759-46770).
extern "C" __global__ void k_expert_h_warp(const unsigned short* x, const unsigned char* W, const int* A, const int* B,
                                           const int* C, long qoff, long soff, long sstride, unsigned short* out) {
  __shared__ float lut[16];
  int tid = threadIdx.x, lane = tid & 31;
  if (tid < 16) lut[tid] = ((const float*)W)[tid];
  __syncthreads();
  int g = blockIdx.y;
  int lo = B[g], hi = B[g + 1];
  if (lo >= hi) return;
  int e = A[g];
  int j = blockIdx.x * 8 + (tid >> 5);
  long row = (long)e * 1280 + j;
  LutRow gr = {W + qoff + row * 1280, W + soff + row * sstride};
  LutRow ur = {gr.q + 640L * 1280, gr.s + 640L * sstride};
  for (int p = lo; p < hi; p++) {
    int slot = C[p];
    const unsigned short* xr = x + (long)(slot / 10) * 2560;
    float ag = 0.0f, au = 0.0f;
    for (int i = lane; i < 2560; i += 32) {
      float xv = bf2f(xr[i]);
      ag = __builtin_fmaf(xv, w_lut(gr, lut, i), ag);
      au = __builtin_fmaf(xv, w_lut(ur, lut, i), au);
    }
    ag = wred(ag);
    au = wred(au);
    if (lane == 0) out[(long)slot * 640 + j] = f2bf_rne(au * silu(ag));
  }
}

// k_expert_h_warp_aff<BITS,true>(x@0, W@8, A@16, B@24, C@32, AffDev@40 (48 B), out@88): same grid and slot loop; rows from AffDev:
// nibble row W + o0 + row*1280, plane W + o1 + row*320 (BITS=5), group table W + o2 + row*o3, dm table W + o4 + row*o5; up rows are 640 rows on
// (+0xc8000, +0x32000, +640*o3, +640*o5, listing: v_mad_u64_u32 0x280 * s10 etc., :2378767-2378790 and the h5 listing).
template <int BITS>
__device__ static inline void h_warp_aff(const unsigned short* x, const unsigned char* W, const int* A, const int* B,
                                         const int* C, AffDev af, unsigned short* out) {
  int tid = threadIdx.x, lane = tid & 31;
  int g = blockIdx.y;
  int lo = B[g], hi = B[g + 1];
  if (lo >= hi) return;
  int e = A[g];
  int j = blockIdx.x * 8 + (tid >> 5);
  long row = (long)e * 1280 + j;
  AffRow gr = {W + af.o[0] + row * 1280, W + af.o[1] + row * 320, W + af.o[2] + row * af.o[3], W + af.o[4] + row * af.o[5]};
  AffRow ur = {gr.q + 640L * 1280, gr.plane + 640L * 320, gr.sc + 640L * af.o[3], gr.dm + 640L * af.o[5]};
  for (int p = lo; p < hi; p++) {
    int slot = C[p];
    const unsigned short* xr = x + (long)(slot / 10) * 2560;
    float ag = 0.0f, au = 0.0f;
    for (int i = lane; i < 2560; i += 32) {
      float xv = bf2f(xr[i]);
      ag = __builtin_fmaf(xv, w_aff<BITS>(gr, i), ag);
      au = __builtin_fmaf(xv, w_aff<BITS>(ur, i), au);
    }
    ag = wred(ag);
    au = wred(au);
    if (lane == 0) out[(long)slot * 640 + j] = f2bf_rne(au * silu(ag));
  }
}
extern "C" __global__ void k_expert_h_warp_aff_5t(const unsigned short* x, const unsigned char* W, const int* A, const int* B,
                                                  const int* C, AffDev af, unsigned short* out) { h_warp_aff<5>(x, W, A, B, C, af, out); }
extern "C" __global__ void k_expert_h_warp_aff_4t(const unsigned short* x, const unsigned char* W, const int* A, const int* B,
                                                  const int* C, AffDev af, unsigned short* out) { h_warp_aff<4>(x, W, A, B, C, af, out); }

// k_expert_h_grouped_warp(x@0, W@8, A@16, B@24, C@32, qoff@40, soff@48, sstride@56, out@64); LDS 20544 B (LUT 64 B at 20480 + four bf16 rows).
// Grid (320, ngroups), block 256. Block (bx, g) owns rows j0 = 2*bx and j0+1 (s_lshl_b32 s20, s2, 1 :46829), expert e = A[g].
// Phase 1 (:46856-46985): for r in {0,1}, i = tid; i <= 0x9ff; i += blockDim: the weight (same formula as k_expert_h_warp) of gate row
//   e*1280+j0+r and up row 640 rows later is stored as bf16 into LDS: gate r0 @0, gate r1 @5120, up r0 @10240, up r1 @15360
//   (ds_store_b16_d16_hi :46923-46924, :46968-46969). barrier (:46973).
// Phase 2 (:46978-47244): warp w handles slots p = B[g] + w, +8, ... < B[g+1] (v_add_nc_u32 v1, 8, v1). slot = C[p], tok = slot/10.
//   Lane l visits i = l + 32*t in ascending order (listing: l + 128*m + 32*q); for each: x = bf16 x[tok][i], four accumulators
//   fma(x, lds row value, acc) (:47025-47066). Five-step reduction as wred, then for each row r: out = bf16_rne(up_r * silu(gate_r)); the
//   two bf16 are packed row0 low, row1 high into one 32-bit store at out + slot*640 + j0 (global_store_b32 :47240).
template <bool AFF, int BITS>
__device__ static inline void h_grouped(const unsigned short* x, const unsigned char* W, const int* A, const int* B, const int* C,
                                        long qoff, long soff, long sstride, const AffDev* af, unsigned short* out) {
  __shared__ float lut[16];
  __shared__ unsigned short wl[4][2560];
  int tid = threadIdx.x, bs = blockDim.x, lane = tid & 31, warp = tid >> 5;
  if (!AFF && tid < 16) lut[tid] = ((const float*)W)[tid];
  __syncthreads();
  int g = blockIdx.y;
  int e = A[g];
  int j0 = blockIdx.x * 2;
  for (int r = 0; r < 2; r++) {
    long row = (long)e * 1280 + j0 + r;
    LutRow lg{}, lu{};
    AffRow ag{}, au{};
    if (!AFF) {
      lg = {W + qoff + row * 1280, W + soff + row * sstride};
      lu = {lg.q + 640L * 1280, lg.s + 640L * sstride};
    } else {
      ag = {W + af->o[0] + row * 1280, W + af->o[1] + row * 320, W + af->o[2] + row * af->o[3], W + af->o[4] + row * af->o[5]};
      au = {ag.q + 640L * 1280, ag.plane + 640L * 320, ag.sc + 640L * af->o[3], ag.dm + 640L * af->o[5]};
    }
    for (int i = tid; i < 2560; i += bs) {
      float wg = AFF ? w_aff<BITS>(ag, i) : w_lut(lg, lut, i);
      float wu = AFF ? w_aff<BITS>(au, i) : w_lut(lu, lut, i);
      wl[r][i] = f2bf_rne(wg);
      wl[2 + r][i] = f2bf_rne(wu);
    }
  }
  __syncthreads();
  int lo = B[g], hi = B[g + 1];
  for (int p = lo + warp; p < hi; p += 8) {
    int slot = C[p];
    const unsigned short* xr = x + (long)(slot / 10) * 2560;
    float g0 = 0.0f, g1 = 0.0f, u0 = 0.0f, u1 = 0.0f;
    for (int i = lane; i < 2560; i += 32) {
      float xv = bf2f(xr[i]);
      g0 = __builtin_fmaf(xv, bf2f(wl[0][i]), g0);
      g1 = __builtin_fmaf(xv, bf2f(wl[1][i]), g1);
      u0 = __builtin_fmaf(xv, bf2f(wl[2][i]), u0);
      u1 = __builtin_fmaf(xv, bf2f(wl[3][i]), u1);
    }
    g0 = wred(g0); g1 = wred(g1); u0 = wred(u0); u1 = wred(u1);
    if (lane == 0) {
      unsigned short r0 = f2bf_rne(u0 * silu(g0)), r1 = f2bf_rne(u1 * silu(g1));
      *(unsigned*)(out + (long)slot * 640 + j0) = (unsigned)r0 | ((unsigned)r1 << 16);
    }
  }
}
extern "C" __global__ void k_expert_h_grouped_warp(const unsigned short* x, const unsigned char* W, const int* A, const int* B,
                                                   const int* C, long qoff, long soff, long sstride, unsigned short* out) {
  h_grouped<false, 0>(x, W, A, B, C, qoff, soff, sstride, nullptr, out);
}
extern "C" __global__ void k_expert_h_grouped_warp_aff_5t(const unsigned short* x, const unsigned char* W, const int* A, const int* B,
                                                          const int* C, AffDev af, unsigned short* out) {
  h_grouped<true, 5>(x, W, A, B, C, 0, 0, 0, &af, out);
}
extern "C" __global__ void k_expert_h_grouped_warp_aff_4t(const unsigned short* x, const unsigned char* W, const int* A, const int* B,
                                                          const int* C, AffDev af, unsigned short* out) {
  h_grouped<true, 4>(x, W, A, B, C, 0, 0, 0, &af, out);
}

// k_expert_out_grouped_warp(h@0, W@8, A@16, B@24, C@32, wt@40 (f32), qoff@48, soff@56, sstride@64, out@72 (f32)); LDS 10304 B.
// Grid (320, ngroups), block 256. Warp w of block (bx, g) owns output row j = bx*8 + w of expert e = A[g]; W[e] has 2560 rows of 640 nibbles
// (320 B per row, row = e*2560 + j: v_mad_i64_i32 0xa00 :49197, v_mad_u64_u32 0x140 :49198), scales at soff + row*sstride.
// Phase 1 (:49218-49270): the warp dequantizes its row into LDS (bf16, 1280 B per warp, lane l: i = l + 32q + 128m, five passes, 640 elements; same
//   formula as k_expert_h_warp: bf16_rne(fma(f16 scale, lut[nib], -0.0)); the four scales of a pass come from one global_load_b64), barrier.
// Phase 2 (:49290-49340): for p = B[g] .. B[g+1]-1 (all warps), slot = C[p]: acc = sum over i = l + 32t of fma(bf(h[slot*640+i]), bf(lds[w][i]), acc),
//   ascending t (v_fmac_f32 v8, v16, v17 :49329), wred, lane 0: out[slot*2560 + j] = wt[slot] * sum (v_mul_f32 :49345, global_store_b32).
extern "C" __global__ void k_expert_out_grouped_warp(const unsigned short* h, const unsigned char* W, const int* A, const int* B,
                                                     const int* C, const float* wt, long qoff, long soff, long sstride, float* out) {
  __shared__ float lut[16];
  __shared__ unsigned short rows[8][640];
  int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid < 16) lut[tid] = ((const float*)W)[tid];
  __syncthreads();
  int g = blockIdx.y;
  int e = A[g];
  int j = blockIdx.x * 8 + warp;
  long row = (long)e * 2560 + j;
  LutRow r = {W + qoff + row * 320, W + soff + row * sstride};
  for (int i = lane; i < 640; i += 32) rows[warp][i] = f2bf_rne(w_lut(r, lut, i));
  __syncthreads();
  int lo = B[g], hi = B[g + 1];
  for (int p = lo; p < hi; p++) {
    int slot = C[p];
    float acc = 0.0f;
    for (int i = lane; i < 640; i += 32) acc = __builtin_fmaf(bf2f(h[(long)slot * 640 + i]), bf2f(rows[warp][i]), acc);
    acc = wred(acc);
    if (lane == 0) out[(long)slot * 2560 + j] = wt[slot] * acc;
  }
}

// k_expert_out_grouped_warp_aff<5,false>(h@0, W@8, A@16, B@24, C@32, wt@40, AffDev@48 (48 B), out@96 (f32)); LDS 10240 B.
// Same structure; weights from AffDev as deq_aff<5,false>: nibble row W + o0 + row*320, plane W + o1 + row*80 (20 words), pair u32 W + o2 + row*o3 + 4*(i>>5)
// {f16 scale, f16 offset}; w = bf16_rne(fma(nib, scale, offset)) (v_fma_mix_f32 v4, v4, v17, v17 op_sel:[0,0,1] op_sel_hi:[0,1,1] :2379080 region).
extern "C" __global__ void k_expert_out_grouped_warp_aff_5f(const unsigned short* h, const unsigned char* W, const int* A, const int* B,
                                                            const int* C, const float* wt, AffDev af, float* out) {
  __shared__ unsigned short rows[8][640];
  int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int g = blockIdx.y;
  int e = A[g];
  int j = blockIdx.x * 8 + warp;
  long row = (long)e * 2560 + j;
  const unsigned char* qr = W + af.o[0] + row * 320;
  const unsigned* pr = (const unsigned*)(W + af.o[1] + row * 80);
  const unsigned* sr = (const unsigned*)(W + af.o[2] + row * af.o[3]);
  for (int i = lane; i < 640; i += 32) {
    unsigned byte = qr[i >> 1];
    unsigned n = (i & 1) ? (byte >> 4) : (byte & 15u);
    n |= ((pr[i >> 5] >> (i & 31)) & 1u) << 4;
    unsigned pair = sr[i >> 5];
    rows[warp][i] = f2bf_rne(__builtin_fmaf((float)n, h2f(pair & 0xffffu), h2f(pair >> 16)));
  }
  __syncthreads();
  int lo = B[g], hi = B[g + 1];
  for (int p = lo; p < hi; p++) {
    int slot = C[p];
    float acc = 0.0f;
    for (int i = lane; i < 640; i += 32) acc = __builtin_fmaf(bf2f(h[(long)slot * 640 + i]), bf2f(rows[warp][i]), acc);
    acc = wred(acc);
    if (lane == 0) out[(long)slot * 2560 + j] = wt[slot] * acc;
  }
}

// k_expert_h_q4(x@0, W@8, sel@16, qoff@24, soff@32, sstride@40, out@48); LDS 8256 B (eight 256-float arrays + 64 B LUT at 8192).
// Grid (160, nslots), block 256. j0 = 4*blockIdx.x (s_lshl_b32 s2, s2, 2), slot = blockIdx.y, tok = slot/10, e = sel[slot].
// LUT = first 16 floats of W (tid < 16, ds_store at 8192). Rows e*1280 + j0 + r (r = 0..3) for gate and 640 rows later for up (nibble rows 1280 B
// apart: offsets 0/1280/2560/3840; scale rows sstride apart). Thread: for i = tid; i < 2560; i += blockDim: x = bf16 x[tok][i], eight accumulators
//   acc = fma(x, bf16_rne(fma(f16 scale[i>>5], lut[nib(i)], -0.0)), acc) in listing order g0 g1 g2 g3 u0 u1 u2 u3 -> LDS arrays 0..7.
//   Tree s = bs/2 .. 1 over all eight arrays (lds[t] = lds[t+s] + lds[t]); threads 0..3: out[slot*640 + j0 + t] = bf16_rne(up_t * silu(gate_t)).
extern "C" __global__ void k_expert_h_q4(const unsigned short* x, const unsigned char* W, const int* sel, long qoff, long soff,
                                         long sstride, unsigned short* out) {
  __shared__ float lut[16];
  __shared__ float red[8][256];
  int tid = threadIdx.x, bs = blockDim.x;
  if (tid < 16) lut[tid] = ((const float*)W)[tid];
  __syncthreads();
  int slot = blockIdx.y;
  int tok = slot / 10;
  int e = sel[slot];
  int j0 = blockIdx.x * 4;
  LutRow gr[4], ur[4];
  for (int r = 0; r < 4; r++) {
    long row = (long)e * 1280 + j0 + r;
    gr[r] = {W + qoff + row * 1280, W + soff + row * sstride};
    ur[r] = {gr[r].q + 640L * 1280, gr[r].s + 640L * sstride};
  }
  const unsigned short* xr = x + (long)tok * 2560;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = tid; i < 2560; i += bs) {
    float xv = bf2f(xr[i]);
    for (int r = 0; r < 4; r++) {
      acc[r] = __builtin_fmaf(xv, w_lut(gr[r], lut, i), acc[r]);
      acc[4 + r] = __builtin_fmaf(xv, w_lut(ur[r], lut, i), acc[4 + r]);
    }
  }
  for (int k = 0; k < 8; k++) red[k][tid] = acc[k];
  __syncthreads();
  for (int s = bs >> 1; s > 0; s >>= 1) {
    if (tid < s)
      for (int k = 0; k < 8; k++) red[k][tid] = red[k][tid + s] + red[k][tid];
    __syncthreads();
  }
  if (tid < 4) out[(long)slot * 640 + j0 + tid] = f2bf_rne(red[4 + tid][0] * silu(red[tid][0]));
}

// k_expert_out_q4(h@0, W@8, sel@16, wt@24, qoff@32, soff@40, sstride@48, out@56 (bf16)); LDS 10304 B (ten 256-float arrays + LUT at 10240).
// Grid (2560, ntok), block 256. j = blockIdx.x, tok = blockIdx.y; the ten slots s = tok*10 + m are handled together: e_m = sel[s]
// (s_load_b256 + s_load_b64 of sel + 10*tok, :45790-45800), row = e_m*2560 + j, nibble row 320 B, scales row*sstride.
// Thread: for i = tid; i <= 0x27f; i += blockDim: acc_m = fma(bf(h[s*640 + i]), bf16_rne(fma(f16 scale[i>>5], lut[nib(i)], -0.0)), acc_m).
// LDS arrays m = 0..9, tree s = bs/2 .. 1; thread 0: acc = fma(wt[tok*10], A0[0], +0.0), then fma(wt[tok*10+m], Am[0], acc) for m = 1..9
// (v_fma_f32 v9, s4, v0, 0 then v_fmac_f32 :46418-46428); out[tok*2560 + j] = bf16_rne(acc) (:46440-46448).
extern "C" __global__ void k_expert_out_q4(const unsigned short* h, const unsigned char* W, const int* sel, const float* wt, long qoff,
                                           long soff, long sstride, unsigned short* out) {
  __shared__ float lut[16];
  __shared__ float red[10][256];
  int tid = threadIdx.x, bs = blockDim.x;
  if (tid < 16) lut[tid] = ((const float*)W)[tid];
  __syncthreads();
  int j = blockIdx.x, tok = blockIdx.y;
  LutRow rr[10];
  for (int m = 0; m < 10; m++) {
    long row = (long)sel[tok * 10 + m] * 2560 + j;
    rr[m] = {W + qoff + row * 320, W + soff + row * sstride};
  }
  float acc[10];
  for (int m = 0; m < 10; m++) acc[m] = 0.0f;
  for (int i = tid; i <= 0x27f; i += bs)
    for (int m = 0; m < 10; m++)
      acc[m] = __builtin_fmaf(bf2f(h[(long)(tok * 10 + m) * 640 + i]), w_lut(rr[m], lut, i), acc[m]);
  for (int m = 0; m < 10; m++) red[m][tid] = acc[m];
  __syncthreads();
  for (int s = bs >> 1; s > 0; s >>= 1) {
    if (tid < s)
      for (int m = 0; m < 10; m++) red[m][tid] = red[m][tid + s] + red[m][tid];
    __syncthreads();
  }
  if (tid == 0) {
    float a = __builtin_fmaf(wt[tok * 10], red[0][0], 0.0f);
    for (int m = 1; m < 10; m++) a = __builtin_fmaf(wt[tok * 10 + m], red[m][0], a);
    out[(long)tok * 2560 + j] = f2bf_rne(a);
  }
}
