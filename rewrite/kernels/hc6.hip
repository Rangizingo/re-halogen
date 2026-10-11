// Rewrites of k_hc6_fused<N> and k_hc6_fused3<N> (N = 1..8): the 6-bit hybrid-code projection of the four hyper-connection streams fused with
// an "lq6" grouped matvec (and, for fused3, a k_hc_scatter_norm<false> role), from flash_serve, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o hc6.co hc6.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). Cites are lines in that file; "f1" = k_hc6_fused<1> (starts :263152, hc6 path :263170-264692,
// LqGrp branch :264693-265590), "f3_1" = k_hc6_fused3<1> (starts :283197).
//
// Same three-role structure as k_hc8_fused / k_hc8_fused3 (rewrite/kernels/hc8.hip, g11), with 6-bit rows instead of 8-bit ones:
//   k_hc6_fused<N>: blocks 0 .. nblk-1 (int kernarg 0x138: s_load_b32 s3 f1 :263154, s_cmp_ge_i32 s2, s3 :263159, s_cbranch_scc0 :263160) run the
//   LqGrp matvec (lq6_role); blocks >= nblk wait for the u64 counter at kernarg 0x160 to reach 0x168 (global_load_b64 glc spin, s_sleep 1: f1 :263243-263255)
//   and run the hc6 path with block offset nblk (s_sub_i32 s3, s2, s3: f1 :263170). The LqGrp role bumps the counter (as in hc8).
//   The hc6 path is the hcup6 body (hcup6.hip): row layout 272 bytes (160 nibble bytes, 80 high-part bytes, 5 group dwords at +240), plane stride 0xaa000,
//   10 lanes of 32 weights per row (f1 :263161-263230: v_mad_i64_i32 0x110 :263174, global_load_b128 / b64 offset:160 / b32 offset:240 :263206-263240).
//   N = 5..8 use the 4-rows-per-block form with the bf16 LDS combine (as hc8_big).
//   Math per weight and per plane: q = nibble | ((hi >> 2w) & 3) << 4; w = bf16_rne(fma(float(q), scale, bias)); chain of 16 v_dot2_f32_bf16 from 0;
//   xor-butterfly 8, 4, 2, 1; sigmoid via v_exp_f32 and IEEE division; fma chain with wm; * 0.25; f store when f != 0; y store bf16 RNE.
//   k_hc6_fused3<N>: blocks < rows (kernarg 0x20) run scatter_role (k_hc_scatter_norm<false>, same code as in hc8.hip), blocks rows .. rows+nblk-1 run lq6_role
//   (waiting on ctr, bumping ctr2), blocks >= rows + nblk wait for ctr2 >= target2 and run the hc6 path reading the scatter output as wm.
#include <hip/hip_runtime.h>

typedef __bf16 v2bf __attribute__((ext_vector_type(2)));
// LqGrp: 4 members of 72 bytes + count (kernarg offset 0, 296 bytes). Fields as in hc8.hip (g11): blob@+0, out@+8, out2@+0x10, R@+0x30, f@+0x38, E@+0x3c, start@+0x40.
struct LqGrpEnt {
  const unsigned char* blob; void* out; void* out2;
  long woff, soff, sstride, R;
  float f; int E; int start; int pad;
};
struct LqGrp {
  LqGrpEnt g[4];
  int count; int pad;
};

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
__device__ static inline unsigned rne_bits(float f) {
  unsigned u = __float_as_uint(f);
  return u + ((u >> 16) & 1u) + 0x7fffu;  // v_bfe_u32 16,1 ; v_add3_u32 0x7fff (f1 :167470-167475)
}
__device__ static inline float widen_rne(float f) { return __uint_as_float(rne_bits(f) & 0xffff0000u); }
__device__ static inline float dot2(unsigned a, unsigned b, float c) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(v2bf, a), __builtin_bit_cast(v2bf, b), c, false);
}
__device__ static inline float sigmoid_widen(float r) {  // f1 :168302-168396
  float e = __builtin_amdgcn_exp2f(r * -1.44269502162933349609375f);
  return widen_rne(1.0f / (1.0f + e));
}

// Decode one lane's 32 weights: nibbles nib[4] (16 bytes), high parts hi (8 bytes), group dword grp -> 16 packed bf16 pairs.
__device__ static inline void dequant6(uint4 nib, uint2 hi, unsigned grp, unsigned pk[16]) {
  const float s = (float)__builtin_bit_cast(_Float16, (unsigned short)(grp & 0xffffu));
  const float b = (float)__builtin_bit_cast(_Float16, (unsigned short)(grp >> 16));
  const unsigned nd[4] = {nib.x, nib.y, nib.z, nib.w};
  const unsigned long long h = ((unsigned long long)hi.y << 32) | hi.x;
  for (int m = 0; m < 16; m++) {
    const int w0 = 2 * m, w1 = 2 * m + 1;
    unsigned q0 = ((nd[w0 >> 3] >> (4 * (w0 & 7))) & 15u) | ((unsigned)((h >> (2 * w0)) & 3u) << 4);
    unsigned q1 = ((nd[w1 >> 3] >> (4 * (w1 & 7))) & 15u) | ((unsigned)((h >> (2 * w1)) & 3u) << 4);
    float v0 = __builtin_fmaf((float)q0, s, b);
    float v1 = __builtin_fmaf((float)q1, s, b);
    pk[m] = (rne_bits(v1) & 0xffff0000u) | (rne_bits(v0) >> 16);
  }
}

// Row n of plane k: 272 bytes (u1 :252508), plane stride 0xaa000 = 696320 (:252518+). Returns the 16 weight pairs of lane `lane` (< 10).
__device__ static inline void load6(const unsigned char* W, int k, long n, unsigned lane, unsigned pk[16]) {
  const unsigned char* row = W + (long)k * 696320 + n * 272;
  const uint4 nib = *(const uint4*)(row + lane * 16);
  const uint2 hi = *(const uint2*)(row + 160 + lane * 8);
  const unsigned grp = *(const unsigned*)(row + 240 + (lane >> 1) * 4);
  dequant6(nib, hi, grp, pk);
}

__device__ static inline float chain16(const unsigned* xp, const unsigned pk[16]) {
  float a = 0.f;
  for (int m = 0; m < 16; m++) a = dot2(xp[m], pk[m], a);
  return a;
}

// N = 1..4 (k_hcup6): 16 rows per block, the 4 planes in-lane.
template <int N>
__device__ static inline void hc6_small(const unsigned short* wm, const unsigned char* W, const unsigned short* x, unsigned short* y, float* f, int nblk) {
  const unsigned tid = threadIdx.x;
  const unsigned lane = tid & 15u, g = tid >> 4;
  const long n = (long)(blockIdx.x - nblk) * 16 + g;
  float p[N][4];
  for (int j = 0; j < N; j++) p[j][0] = p[j][1] = p[j][2] = p[j][3] = 0.f;
  if (lane < 10) {
    for (int k = 0; k < 4; k++) {
      unsigned pk[16];
      load6(W, k, n, lane, pk);
      for (int j = 0; j < N; j++) p[j][k] = chain16((const unsigned*)(x + 320 * j + 32 * lane), pk);
    }
  }
  for (int off = 8; off >= 1; off >>= 1)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < 4; k++) p[j][k] = p[j][k] + __shfl_xor(p[j][k], off);
  if (lane == 0) for (int j = 0; j < N; j++) {
    const unsigned short* wj = wm + (long)j * 10240;
    float s[4];
    for (int k = 0; k < 4; k++) s[k] = sigmoid_widen(widen_rne(p[j][k]));
    float m = __builtin_fmaf(s[0], bf2f(wj[0 * 2560 + n]), 0.0f);
    m = __builtin_fmaf(s[1], bf2f(wj[1 * 2560 + n]), m);
    m = __builtin_fmaf(s[2], bf2f(wj[2 * 2560 + n]), m);
    m = __builtin_fmaf(s[3], bf2f(wj[3 * 2560 + n]), m);
    float o = m * 0.25f;
    if (f) f[(long)j * 2560 + n] = o;
    y[(long)j * 2560 + n] = (unsigned short)(rne_bits(o) >> 16);
  }
}

// N = 5..8 (k_hcup6w): 4 rows per block, group gi = tid >> 4, plane gi & 3, row gi >> 2; N sums as bf16 RNE through LDS, combine on tid < 4N.
template <int N>
__device__ static inline void hc6_big(const unsigned short* wm, const unsigned char* W, const unsigned short* x, unsigned short* y, float* f, int nblk) {
  __shared__ unsigned short lds[16 * N];
  const unsigned tid = threadIdx.x;
  const unsigned lane = tid & 15u, gi = tid >> 4;
  const unsigned k = gi & 3u, rr = gi >> 2;
  const long b = (long)(blockIdx.x - nblk) * 4;
  float p[N];
  for (int j = 0; j < N; j++) p[j] = 0.f;
  if (lane < 10) {
    unsigned pk[16];
    load6(W, k, b + rr, lane, pk);
    for (int j = 0; j < N; j++) p[j] = chain16((const unsigned*)(x + 320 * j + 32 * lane), pk);
  }
  for (int off = 8; off >= 1; off >>= 1)
    for (int j = 0; j < N; j++) p[j] = p[j] + __shfl_xor(p[j], off);
  if (lane == 0)
    for (int j = 0; j < N; j++) lds[gi * N + j] = (unsigned short)(rne_bits(p[j]) >> 16);
  __syncthreads();
  if (tid < 4u * N) {
    const unsigned j = tid % N, r = tid / N;
    const long n = b + r;
    const unsigned short* wj = wm + (long)j * 10240;
    float s[4];
    for (int kk = 0; kk < 4; kk++) s[kk] = sigmoid_widen(bf2f(lds[(r * 4 + kk) * N + j]));
    float m = __builtin_fmaf(bf2f(wj[0 * 2560 + n]), s[0], 0.0f);
    m = __builtin_fmaf(bf2f(wj[1 * 2560 + n]), s[1], m);
    m = __builtin_fmaf(bf2f(wj[2 * 2560 + n]), s[2], m);
    m = __builtin_fmaf(bf2f(wj[3 * 2560 + n]), s[3], m);
    float o = m * 0.25f;
    if (f) f[(long)j * 2560 + n] = o;
    y[(long)j * 2560 + n] = (unsigned short)(rne_bits(o) >> 16);
  }
}


// Cites in wait_counter / scatter_role / bump_counter marked hc8-f1 / hc8-f3_1 are lines of k_hc8_fused<1> / k_hc8_fused3<1> (identical code);
// the hc6 equivalents: wait loop f1 :263243-263263 (s_sleep :263252, s_barrier :263263), scatter role f3_1 rsq :285469, its counter add :285637 (kernarg 0x188),
// LqGrp role counter add f1 :265587, f3_1 :285967.
// Grid sync with the LqGrp blocks: thread 0 spins until *ctr >= target (global_load_b64 glc, v_cmp_le_u64 target <= value, s_sleep 1: f1 :167423-167439),
// then s_barrier. The counter increment (global_atomic_add_u64, f1 :169061) belongs to the LqGrp path: the hc6 path ends with s_cbranch_vccz 843
// from :168432 straight to the s_nop/s_endpgm at 0x2ff718, skipping the atomic (target address checked in the listing).
__device__ static inline void wait_counter(unsigned long long* ctr, unsigned long long target) {
  if (threadIdx.x == 0) {
    while (__hip_atomic_load(ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target) __builtin_amdgcn_s_sleep(1);
  }
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");  // buffer_gl1_inv ; buffer_gl0_inv after the barrier (hc8-f1 :167443-167446)
}

// Role A of k_hc8_fused3<N>: blocks 0 .. rows-1 (rows = int kernarg at 0x20, f3_1 :182601, :182606-182607) run k_hc_scatter_norm<false> on row r = blockIdx.x
// (hc8-f3_1 :184033-184310, same instruction sequence as codeobj_53000.s:1103203-1103489): y0 (arg 0) in/out, p (arg 1), w (arg 2), g (arg 3), out = kernarg 0x150 (s20:21).
// Pass 1: t = fma(w[r], p[r/4][i], y0[r][i]) -> bf16 RNE, stored back into y0 (:184073-184086), ss = fma(rb, rb, ss) (:184087).
// LDS tree sum (:184095-184132); norm = rsq(sum/2560 + 1e-6f) with the 2^-126 rescue (:184139-184172).
// Pass 2: out[r][i] = bf16_rne((norm * y0[r][i]) * (1 + g[(r%4)*2560 + i])) (:184191-184232, :184276-184303).
// Then thread 0 adds 1 to the u64 at kernarg 0x188 (s12:13) (global_atomic_add_u64: :184331). The out matrix is [rows][2560]; the hc8 role reads it as its wm.
__device__ static inline void scatter_role(unsigned short* y0, const unsigned short* p, const float* w, const unsigned short* g, unsigned short* out,
                                           unsigned long long* ctr) {
  __shared__ float sh[256];
  const unsigned r = blockIdx.x;
  const unsigned tid = threadIdx.x;
  const unsigned bd = blockDim.x;
  unsigned short* yr = y0 + (size_t)r * 2560;
  const unsigned short* pr = p + (size_t)(r / 4) * 2560;
  const float wr = w[r];
  float ss = 0.0f;
  for (unsigned i = tid; i < 2560u; i += bd) {
    float t = __builtin_fmaf(wr, bf2f(pr[i]), bf2f(yr[i]));
    unsigned rb = rne_bits(t);
    yr[i] = (unsigned short)(rb >> 16);
    float v = __uint_as_float(rb & 0xffff0000u);
    ss = __builtin_fmaf(v, v, ss);
  }
  sh[tid] = ss;
  __syncthreads();
  for (unsigned s = bd >> 1; s > 0; s >>= 1) {
    if (tid < s) sh[tid] = sh[tid + s] + sh[tid];
    __syncthreads();
  }
  float var = sh[0] / 2560.0f + __uint_as_float(0x358637bdu);
  bool small = var < 0x1p-126f;
  if (small) var = var * 16777216.0f;
  float norm = __builtin_amdgcn_rsqf(var);
  if (small) norm = norm * 4096.0f;
  const unsigned m = r & 3u;
  for (unsigned i = tid; i < 2560u; i += bd) {
    float x = bf2f(yr[i]);
    float gv = 1.0f + bf2f(g[m * 2560 + i]);
    out[(size_t)r * 2560 + i] = (unsigned short)(rne_bits((norm * x) * gv) >> 16);
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) atomicAdd(ctr, 1ull);
}
__device__ static inline void bump_counter(unsigned long long* ctr) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(ctr, 1ull);
}

// LqGrp branch ("lq6" grouped matvec with NC activation columns), k_hc6_fused<1> :264693-265590 (member pick :264696-264722; the instruction pattern is
// the same as k_lq6w_grp, see lq6w.hip / the lq8 branch of hc8.hip, and k_hc8_fused<1> :168433-169043).
// Member pick: highest i with count > i and blk >= start[i] (count at 0x120, starts at 0x88, 0xd0, 0x118: :264696-264722). R = member+0x30 (s_load_b64 [0x30] :264728),
//   R <= 8: 1 row per block, 8 waves split the units; R > 8: 4 rows per block, 2 waves per row (s_cselect 8 : 2 and 3 : 1, 0 : 2: :264732-264734).
// Row layout (the hc6 row layout, K = kernarg 0x130 elements): K/2 nibble bytes, K/4 high-part bytes, K/16 group bytes, row stride aligned up to 16
// ((K/2 + K/4 + K/16 + 15) & ~15: s_add 15 ... s_and -16 :264770-264790).
// Unit = 32 elements: nibbles 16 bytes at row + 16*c, high parts 8 bytes at row + K/2 + 8*c, group dword at row + K/2 + K/4 + 4*(c >> 1).
// Unit range of a wave: [wi*cw, min(nu, wi*cw + cw)), cw = ceil(nu / s1), nu = K/32; lane l takes units start + l, +32, ...
// Per unit: weights as in dequant6, x unit = 32 bf16 at xa + j*K + 32*c, inner = chain of 16 dot2 from 0, acc += inner.
// Wave sum: xor-butterfly 16, 8, 4, 2, 1, lane 0 stores to LDS; after a barrier threads tid < (big ? 1 : 4) sum the s1 partials in order from 0 and
// run the E epilogue exactly as in the lq8 branch of hc8.hip (E = 0 bf16 store, 1 silu * f, 2 sigmoid, 3 2*sigmoid f32, 4 sigmoid of the raw sum f32, 5 split store).
template <int NC>
__device__ static inline void lq6_role(const LqGrp& gp, const unsigned short* xa, long K, int blk) {
  __shared__ float part[NC][8];
  const unsigned tid = threadIdx.x;
  const unsigned wave = tid >> 5, lane = tid & 31u;
  int gi = 0;
  if (gp.count > 1 && blk >= gp.g[1].start) gi = 1;
  if (gp.count > 2 && blk >= gp.g[2].start) gi = 2;
  if (gp.count > 3 && blk >= gp.g[3].start) gi = 3;
  const LqGrpEnt& e = gp.g[gi];
  const long R = e.R;
  const bool big = R <= 8;  // 1 row per block over 8 waves; else 4 rows per block over 2 waves each
  const int s1 = big ? 8 : 2, s8 = big ? 3 : 1, s9 = big ? 0 : 2, rpb = big ? 1 : 4;
  const long rbase = (long)(blk - e.start) << s9;
  const long row = rbase + (wave >> s8);
  const bool valid = row < R;
  const long crow = valid ? row : 0;
  const unsigned wi = wave & (s1 - 1);
  const long nu = K >> 5;
  const long stride = (K / 2 + K / 4 + K / 16 + 15) & ~15L;
  const unsigned char* rowp = e.blob + crow * stride;
  const long cw = (nu + s1 - 1) / s1;
  const long c0 = (long)wi * cw;
  long c1 = c0 + cw;
  if (nu < c1) c1 = nu;
  float acc[NC];
  for (int j = 0; j < NC; j++) acc[j] = 0.0f;
  for (long c = c0 + lane; c < c1; c += 32) {
    const uint4 nib = *(const uint4*)(rowp + c * 16);
    const uint2 hi = *(const uint2*)(rowp + K / 2 + c * 8);
    const unsigned gd = *(const unsigned*)(rowp + K / 2 + K / 4 + ((c * 2) & ~3L));
    unsigned pk[16];
    dequant6(nib, hi, gd, pk);
    for (int j = 0; j < NC; j++) acc[j] = acc[j] + chain16((const unsigned*)(xa + (long)j * K + c * 32), pk);
  }
  for (int j = 0; j < NC; j++)
    for (int o = 16; o >= 1; o >>= 1) acc[j] = acc[j] + __shfl_xor(acc[j], o);
  if (lane == 0)
    for (int j = 0; j < NC; j++) part[j][((wave >> s8) << s8) + wi] = acc[j];
  __syncthreads();
  if ((int)tid < rpb) {
    const long orow = rbase + tid;
    if (orow < R) {
      const float f = e.f;
      const int E = e.E;
      for (int j = 0; j < NC; j++) {
        float sum = 0.0f;
        for (int q = 0; q < s1; q++) sum = sum + part[j][(tid << s8) + q];
        const long idx = (long)j * R + orow;
        if (E == 5) {
          unsigned m = (unsigned)(orow & 511);
          long b5 = orow >> 9;
          unsigned short* base = (unsigned short*)(m < 256 ? e.out : e.out2);
          base[b5 * 256 + (m & 255) + (long)j * (R >> 1)] = (unsigned short)(rne_bits(sum) >> 16);
        } else if (E == 0) {
          ((unsigned short*)e.out)[idx] = (unsigned short)(rne_bits(sum) >> 16);
        } else if (E <= 4) {
          float v = widen_rne(sum);
          if (E == 1 || E == 3) v = f * v;
          if (E == 4) v = sum;
          float ex = __builtin_amdgcn_exp2f(v * -1.44269502162933349609375f);
          float d = 1.0f + ex;
          if (E == 1) {
            ((unsigned short*)e.out)[idx] = (unsigned short)(rne_bits(v / d) >> 16);
          } else if (E == 2) {
            ((unsigned short*)e.out)[idx] = (unsigned short)(rne_bits(1.0f / d) >> 16);
          } else {
            float yv = 1.0f / d;
            ((float*)e.out)[idx] = (E == 3) ? yv + yv : yv;
          }
        }
      }
    }
  }
}

#define HC6_SMALL(N)                                                                                                       \
  extern "C" __global__ void k_hc6_fused_##N(LqGrp grp, const unsigned short* wm, long ld, int nblk, const unsigned char* W, \
                                             const unsigned short* x, unsigned short* y, float* f,                           \
                                             unsigned long long* ctr, unsigned long long target) {                            \
                                                                                                         \
    if ((int)blockIdx.x < nblk) { lq6_role<N>(grp, wm, ld, blockIdx.x);                \
      bump_counter(ctr);                \
      } else { wait_counter(ctr, target);                \
      hc6_small<N>(wm, W, x, y, f, nblk);                \
      }                                                                                                                        \
  }
HC6_SMALL(1) HC6_SMALL(2) HC6_SMALL(3) HC6_SMALL(4)

#define HC6_BIG(N)                                                                                                         \
  extern "C" __global__ void k_hc6_fused_##N(LqGrp grp, const unsigned short* wm, long ld, int nblk, const unsigned char* W, \
                                             const unsigned short* x, unsigned short* y, float* f,                           \
                                             unsigned long long* ctr, unsigned long long target) {                            \
                                                                                                         \
    if ((int)blockIdx.x < nblk) { lq6_role<N>(grp, wm, ld, blockIdx.x);                \
      bump_counter(ctr);                \
      } else { wait_counter(ctr, target);                \
      hc6_big<N>(wm, W, x, y, f, nblk);                \
      }                                                                                                                        \
  }
HC6_BIG(5) HC6_BIG(6) HC6_BIG(7) HC6_BIG(8)

// k_hc6_fused3<N>: roles by block index (hc8-f3_1 :182601-182614). Blocks < rows: scatter_role. Blocks rows .. rows+nblk-1: LqGrp GEMV (not rewritten).
// Blocks >= rows + nblk: hc8 path with b' = blockIdx.x - rows - nblk (hc8-f3_1 :182611-182625), which first waits for *ctr2 >= target2
// (kernarg 0x198 / 0x1a0: s16:17, s18:19 at :182698-182710; the LqGrp role bumps ctr2 at :184664) and reads the scatter output as its wm (kernarg 0x150).
// The hc8 path itself is the same as in k_hc6_fused<N> (identical instruction pattern, e.g. dequant f3_1 :182719-182764, epilogue :183535-183700).
#define HC6_FUSED3(N, PATH)                                                                                                                 \
  extern "C" __global__ void k_hc6_fused3_##N(unsigned short* y0, const unsigned short* p, const float* w, const unsigned short* g, int rows, \
                                              LqGrp grp, unsigned short* wm, long ld, int nblk, const unsigned char* W,                      \
                                              const unsigned short* x, unsigned short* y, float* f, unsigned long long* ctr,                 \
                                              unsigned long long target, unsigned long long* ctr2, unsigned long long target2) {             \
     (void)target;                                                                                                       \
    if ((int)blockIdx.x < rows) {                                                                                                            \
      scatter_role(y0, p, w, g, wm, ctr);                                                                                                    \
    } else if ((int)blockIdx.x < rows + nblk) {                                                                                              \
      wait_counter(ctr, target);                                                                                                             \
      lq6_role<N>(grp, wm, ld, blockIdx.x - rows);                                                                                           \
      bump_counter(ctr2);                                                                                                                    \
    } else {                                                                                                                                 \
      wait_counter(ctr2, target2);                                                                                                           \
      PATH<N>(wm, W, x, y, f, rows + nblk);                                                                                                  \
    }                                                                                                                                        \
  }
HC6_FUSED3(1, hc6_small) HC6_FUSED3(2, hc6_small) HC6_FUSED3(3, hc6_small) HC6_FUSED3(4, hc6_small)
HC6_FUSED3(5, hc6_big) HC6_FUSED3(6, hc6_big) HC6_FUSED3(7, hc6_big) HC6_FUSED3(8, hc6_big)
