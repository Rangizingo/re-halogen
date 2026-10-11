// Rewrites of k_hc8_fused<N> and k_hc8_fused3<N> (8-bit hybrid-code projection of the four hyper-connection streams) from flash_serve, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o hc8.co hc8.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). Line cites below are in that file; "f1" = k_hc8_fused<1> (starts :167332).
//
// COVERAGE (superseded: the LqGrp branch is now rewritten as lq8_role below; original note follows): each vendor kernel is two kernels in one. Blocks 0 .. nblk-1 (nblk = int kernarg at 0x138, :167335, :167340-167341) run the
// generic LqGrp GEMV (code after :168433 in <1>); blocks >= nblk run the hc8 path rewritten here. The rewrite implements the hc8 path
// only; the specs launch with nblk = 0 so every block takes the hc8 path. The LqGrp branch is NOT rewritten (BLOCKED, see the report).
#include <hip/hip_runtime.h>

typedef __bf16 v2bf __attribute__((ext_vector_type(2)));
// LqGrp: 4 members of 72 bytes + count (kernarg offset 0, 296 bytes). Member fields as decoded by g08 (rewrite/reports/g08.md): blob@+0, out@+8, out2@+0x10,
// R@+0x30, f@+0x38, E@+0x3c, start@+0x40 (loads f1 :168471-168472, :168512, :168792, :168819-168820; member pick :168433-168466 with count at 0x120 and starts at
// 0x88, 0xd0, 0x118). woff/soff/sstride are not read by the hc8 fused LqGrp branch (row = blob + row*(K + K/16)).
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

// One 16-byte chunk of u8 codes to eight packed bf16 pairs. w = bf16_rne(fma(float(byte), scale, bias)), scale = low f16, bias = high f16 of the
// group dword (v_cvt_f32_ubyte0..3 then v_fma_mix_f32 v, v, s, s with op_sel [0,0,1], op_sel_hi [0,1,1]: f1 :167448-167468, :167476-167493).
// Pair m holds bytes 2m (low half) and 2m+1 (high half): v_and_or_b32 0xffff0000 merge (f1 :167499).
__device__ static inline void dequant_chunk(uint4 c, unsigned grp, unsigned pk[8]) {
  const float s = (float)__builtin_bit_cast(_Float16, (unsigned short)(grp & 0xffffu));
  const float b = (float)__builtin_bit_cast(_Float16, (unsigned short)(grp >> 16));
  const unsigned dw[4] = {c.x, c.y, c.z, c.w};
  for (int m = 0; m < 8; m++) {
    const unsigned d = dw[m >> 1];
    const int sh = (m & 1) * 16;
    float w0 = __builtin_fmaf((float)((d >> sh) & 0xffu), s, b);
    float w1 = __builtin_fmaf((float)((d >> (sh + 8)) & 0xffu), s, b);
    pk[m] = (rne_bits(w1) & 0xffff0000u) | (rne_bits(w0) >> 16);
  }
}

// Row layout of the code tensor: [plane 0..3][row 0..2559][340 bytes] (plane stride 0xd4000 = 2560*340 :167357-167381; row stride 0x154 :167357).
// A row is 20 chunks of 16 code bytes (320 weights) followed by 5 group dwords at +320 + 4*group (group = chunk >> 2) (:167384, :167407).

// hc8 path of k_hc8_fused<N>, N = 1..4. Block (after nblk) b' = blockIdx.x - nblk; lane = tid & 15, g = tid >> 4; row n = b'*16 + g
// (v_or3 s3<<4, g & 14, g & 1 :167353). Lane l reads chunk l, lanes 0..3 also chunk 16 + l (v_cmpx 20 > v45 :167397).
// x set j = x + 320*j bf16 (offset 640 bytes per set, <2> :169183-169186); lane reads 16 bf16 at 16*chunk.
// Per plane k and set j: acc1 = chain of 8 dot2 over the chunk-l pairs starting from 0; total = 0 + acc1 (:167822-167823), and for lanes < 4
// total += chain2 over chunk 16 + l (:167826-168206). Then xor-butterfly 8,4,2,1 over the 16 lanes (:168212-168263), lane 0 of the group finishes.
// Epilogue per set j (same as k_hcup4<A,0>): s_k = sigmoid_widen(widen_rne(P_jk)); m = fma chain over planes 0..3 with wm[j*10240 + k*2560 + n]
// (wm = kernarg 0x128, :168271-168295, :168398-168409); out = m * 0.25 (:168411); f[j*2560 + n] = out if f != 0 (:168412-168419),
// y[j*2560 + n] = bf16_rne(out) (:168421-168427) (set 1 at +2560 elements: <2> :170427-170440).
template <int N>
__device__ static inline void hc8_small(const unsigned short* wm, const unsigned char* W, const unsigned short* x, unsigned short* y, float* f, int nblk) {
  const unsigned tid = threadIdx.x;
  const unsigned lane = tid & 15u, g = tid >> 4;
  const long n = (long)(blockIdx.x - nblk) * 16 + g;
  float p[N][4];
  for (int j = 0; j < N; j++) p[j][0] = p[j][1] = p[j][2] = p[j][3] = 0.f;
  for (int k = 0; k < 4; k++) {
    const unsigned char* row = W + (long)k * 870400 + n * 340;
    const uint4 c1 = *(const uint4*)(row + lane * 16);
    const unsigned g1 = *(const unsigned*)(row + 320 + (lane >> 2) * 4);
    unsigned pk1[8];
    dequant_chunk(c1, g1, pk1);
    unsigned pk2[8];
    if (lane < 4) {
      const uint4 c2 = *(const uint4*)(row + (16 + lane) * 16);
      const unsigned g2 = *(const unsigned*)(row + 336);
      dequant_chunk(c2, g2, pk2);
    }
    for (int j = 0; j < N; j++) {
      const unsigned* x1 = (const unsigned*)(x + 320 * j + 16 * lane);
      float a1 = 0.f;
      for (int m = 0; m < 8; m++) a1 = dot2(x1[m], pk1[m], a1);
      float tot = 0.0f + a1;
      if (lane < 4) {
        const unsigned* x2 = (const unsigned*)(x + 320 * j + 16 * (16 + lane));
        float a2 = 0.f;
        for (int m = 0; m < 8; m++) a2 = dot2(x2[m], pk2[m], a2);
        tot = tot + a2;
      }
      p[j][k] = tot;
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

// hc8 path of k_hc8_fused<N>, N = 5..8 (f5 :176258). One block covers 4 rows and the 4 planes: group gi = tid >> 4, plane k = gi & 3, row rr = gi >> 2,
// n = (blockIdx.x - nblk)*4 + rr (v_bfe_u32 v0,4,2 * 0xa00 + (tid >> 6) + s3<<2: f5 :176269-176281; code row (plane*2560 + n)*340 :176283).
// Each group dots its single code row with all N x sets (x offsets 0, 640, 1280, ... :176328-176337) with the same chunk/dequant/chain order as hc8_small
// (total = 0 + chain1 [+ chain2 for lanes < 4], :176475-176478), xor-butterfly 8,4,2,1 (:176650-176698), then lane 0 stores the N totals as bf16 RNE
// into LDS at gi*2N bytes (ds_store_b64 + b16 for N = 5: :176705-176730). After a barrier threads tid < 4N take set j = tid % N, row rr = tid / N (:176737-176757),
// read the four plane values at (rr*4 + k)*2N + 2j (:176779-176785), and run the hc8_small epilogue with n = b'*4 + rr (:176787-176912):
// s_k = sigmoid_widen(bf16 value), m = fma chain of wm[j*10240 + k*2560 + n] * s_k, out = m * 0.25, f[j*2560 + n] = out if f != 0, y[j*2560 + n] = bf16_rne(out).
template <int N>
__device__ static inline void hc8_big(const unsigned short* wm, const unsigned char* W, const unsigned short* x, unsigned short* y, float* f, int nblk) {
  __shared__ unsigned short lds[16 * N];
  const unsigned tid = threadIdx.x;
  const unsigned lane = tid & 15u, gi = tid >> 4;
  const unsigned k = gi & 3u, rr = gi >> 2;
  const long b = (long)(blockIdx.x - nblk) * 4;
  const unsigned char* row = W + (long)k * 870400 + (b + rr) * 340;
  const uint4 c1 = *(const uint4*)(row + lane * 16);
  const unsigned g1 = *(const unsigned*)(row + 320 + (lane >> 2) * 4);
  unsigned pk1[8];
  dequant_chunk(c1, g1, pk1);
  unsigned pk2[8];
  if (lane < 4) {
    const uint4 c2 = *(const uint4*)(row + (16 + lane) * 16);
    const unsigned g2 = *(const unsigned*)(row + 336);
    dequant_chunk(c2, g2, pk2);
  }
  float p[N];
  for (int j = 0; j < N; j++) {
    const unsigned* x1 = (const unsigned*)(x + 320 * j + 16 * lane);
    float a1 = 0.f;
    for (int m = 0; m < 8; m++) a1 = dot2(x1[m], pk1[m], a1);
    float tot = 0.0f + a1;
    if (lane < 4) {
      const unsigned* x2 = (const unsigned*)(x + 320 * j + 16 * (16 + lane));
      float a2 = 0.f;
      for (int m = 0; m < 8; m++) a2 = dot2(x2[m], pk2[m], a2);
      tot = tot + a2;
    }
    p[j] = tot;
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

// Grid sync with the LqGrp blocks: thread 0 spins until *ctr >= target (global_load_b64 glc, v_cmp_le_u64 target <= value, s_sleep 1: f1 :167423-167439),
// then s_barrier. The counter increment (global_atomic_add_u64, f1 :169061) belongs to the LqGrp path: the hc8 path ends with s_cbranch_vccz 843
// from :168432 straight to the s_nop/s_endpgm at 0x2ff718, skipping the atomic (target address checked in the listing).
__device__ static inline void wait_counter(unsigned long long* ctr, unsigned long long target) {
  if (threadIdx.x == 0) {
    while (__hip_atomic_load(ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target) __builtin_amdgcn_s_sleep(1);
  }
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");  // buffer_gl1_inv ; buffer_gl0_inv after the barrier (f1 :167443-167446)
}

// Role A of k_hc8_fused3<N>: blocks 0 .. rows-1 (rows = int kernarg at 0x20, f3_1 :182601, :182606-182607) run k_hc_scatter_norm<false> on row r = blockIdx.x
// (f3_1 :184033-184310, same instruction sequence as codeobj_53000.s:1103203-1103489): y0 (arg 0) in/out, p (arg 1), w (arg 2), g (arg 3), out = kernarg 0x150 (s20:21).
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

// LqGrp branch ("lq8" grouped matvec with N activation columns), f1 :168433-169043, f3_1 :183706-184032.
// Member pick: highest i with count > i and blk >= start[i] (:168442-168465). R = member+0x30, big = R <= 8 (v_cmp_lt_i64 8 < ... :168474; confirmed by the harness).
//   R <= 8: 1 row per block, 8 waves split the K chunks; R > 8: 4 rows per block, 2 waves per row (s1 = 8 or 2, shift s8 = 3 or 1, s9 = 0 or 2: :168477-168485).
//   row = ((blk - start) << s9) + (wave >> s8) (:168480, :168487, :168502); rows >= R read row 0 and write nothing (:168510, :168518-168521).
// Row layout = the hc8 row layout: K code bytes then K/16 bytes of (scale, bias) f16 pairs, one per 4 chunks; stride K + K/16 (s22 :168495; :168535).
// Chunk range of a wave: [wi*cw, min(nch, wi*cw + cw)), cw = ceil(nch / s1), nch = K / 16 (:168481-168528, :168543-168547); lane l takes chunks start + l, +32, ...
// (:168549, :168698). Per chunk: 16 code bytes at row + 16*c, group dword at row + K + (c & ~3) (:168552-168566); weights = bf16_rne(fma(byte, scale, bias))
// as in hc8; x chunk = 16 bf16 at xa + j*K + 16*c (:168596-168598); inner = chain of 8 dot2 from 0; acc += inner (:168661-168695).
// Wave sum: xor-butterfly 16, 8, 4, 2, 1 (:168735-168770); lane 0 stores to LDS (:168773-168779). After a barrier threads tid < (big ? 1 : 4) sum the s1 partials
// in order from 0 (:168789-168817) and run the E epilogue exactly as in g08's k_dn_conv_ab_rows (:168818-169040): idx = j*R + row.
template <int NC>
__device__ static inline void lq8_role(const LqGrp& gp, const unsigned short* xa, long K, int blk) {
  __shared__ float part[NC][8];
  const unsigned tid = threadIdx.x;
  const unsigned wave = tid >> 5, lane = tid & 31u;
  int gi = 0;
  if (gp.count > 1 && blk >= gp.g[1].start) gi = 1;
  if (gp.count > 2 && blk >= gp.g[2].start) gi = 2;
  if (gp.count > 3 && blk >= gp.g[3].start) gi = 3;
  const LqGrpEnt& e = gp.g[gi];
  const long R = e.R;
  const bool big = R <= 8;  // s3 = (R < 9): 1 row per block over 8 waves; else 4 rows per block over 2 waves each
  const int s1 = big ? 8 : 2, s8 = big ? 3 : 1, s9 = big ? 0 : 2, rpb = big ? 1 : 4;
  const long rbase = (long)(blk - e.start) << s9;
  const long row = rbase + (wave >> s8);
  const bool valid = row < R;
  const long crow = valid ? row : 0;
  const unsigned wi = wave & (s1 - 1);
  const long nch = (K + ((K >> 63) & 15)) >> 4;
  const long stride = nch + K;
  const unsigned char* rowp = e.blob + crow * stride;
  const long cw = (nch + s1 - 1) / s1;
  const long c0 = (long)wi * cw;
  long c1 = c0 + cw;
  if (nch < c1) c1 = nch;
  float acc[NC];
  for (int j = 0; j < NC; j++) acc[j] = 0.0f;
  for (long c = c0 + lane; c < c1; c += 32) {
    const uint4 cv = *(const uint4*)(rowp + c * 16);
    const unsigned gd = *(const unsigned*)(rowp + K + (c & ~3L));
    unsigned pk[8];
    dequant_chunk(cv, gd, pk);
    for (int j = 0; j < NC; j++) {
      const uint4* xp = (const uint4*)(xa + (long)j * K + c * 16);
      const uint4 x0 = xp[0], x1 = xp[1];
      const unsigned xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
      float inner = 0.f;
      for (int m = 0; m < 8; m++) inner = dot2(xw[m], pk[m], inner);
      acc[j] = acc[j] + inner;
    }
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
__device__ static inline void bump_counter(unsigned long long* ctr) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(ctr, 1ull);
}

#define HC8_SMALL(N)                                                                                                       \
  extern "C" __global__ void k_hc8_fused_##N(LqGrp grp, const unsigned short* wm, long ld, int nblk, const unsigned char* W, \
                                             const unsigned short* x, unsigned short* y, float* f,                           \
                                             unsigned long long* ctr, unsigned long long target) {                            \
                                                                                                         \
    if ((int)blockIdx.x < nblk) { lq8_role<N>(grp, wm, ld, blockIdx.x);                \
      bump_counter(ctr);                \
      } else { wait_counter(ctr, target);                \
      hc8_small<N>(wm, W, x, y, f, nblk);                \
      }                                                                                                                        \
  }
HC8_SMALL(1) HC8_SMALL(2) HC8_SMALL(3) HC8_SMALL(4)

#define HC8_BIG(N)                                                                                                         \
  extern "C" __global__ void k_hc8_fused_##N(LqGrp grp, const unsigned short* wm, long ld, int nblk, const unsigned char* W, \
                                             const unsigned short* x, unsigned short* y, float* f,                           \
                                             unsigned long long* ctr, unsigned long long target) {                            \
                                                                                                         \
    if ((int)blockIdx.x < nblk) { lq8_role<N>(grp, wm, ld, blockIdx.x);                \
      bump_counter(ctr);                \
      } else { wait_counter(ctr, target);                \
      hc8_big<N>(wm, W, x, y, f, nblk);                \
      }                                                                                                                        \
  }
HC8_BIG(5) HC8_BIG(6) HC8_BIG(7) HC8_BIG(8)

// k_hc8_fused3<N>: roles by block index (f3_1 :182601-182614). Blocks < rows: scatter_role. Blocks rows .. rows+nblk-1: LqGrp GEMV (not rewritten).
// Blocks >= rows + nblk: hc8 path with b' = blockIdx.x - rows - nblk (f3_1 :182611-182625), which first waits for *ctr2 >= target2
// (kernarg 0x198 / 0x1a0: s16:17, s18:19 at :182698-182710; the LqGrp role bumps ctr2 at :184664) and reads the scatter output as its wm (kernarg 0x150).
// The hc8 path itself is the same as in k_hc8_fused<N> (identical instruction pattern, e.g. dequant f3_1 :182719-182764, epilogue :183535-183700).
#define HC8_FUSED3(N, PATH)                                                                                                                 \
  extern "C" __global__ void k_hc8_fused3_##N(unsigned short* y0, const unsigned short* p, const float* w, const unsigned short* g, int rows, \
                                              LqGrp grp, unsigned short* wm, long ld, int nblk, const unsigned char* W,                      \
                                              const unsigned short* x, unsigned short* y, float* f, unsigned long long* ctr,                 \
                                              unsigned long long target, unsigned long long* ctr2, unsigned long long target2) {             \
     (void)target;                                                                                                       \
    if ((int)blockIdx.x < rows) {                                                                                                            \
      scatter_role(y0, p, w, g, wm, ctr);                                                                                                    \
    } else if ((int)blockIdx.x < rows + nblk) {                                                                                              \
      wait_counter(ctr, target);                                                                                                             \
      lq8_role<N>(grp, wm, ld, blockIdx.x - rows);                                                                                           \
      bump_counter(ctr2);                                                                                                                    \
    } else {                                                                                                                                 \
      wait_counter(ctr2, target2);                                                                                                           \
      PATH<N>(wm, W, x, y, f, rows + nblk);                                                                                                  \
    }                                                                                                                                        \
  }
HC8_FUSED3(1, hc8_small) HC8_FUSED3(2, hc8_small) HC8_FUSED3(3, hc8_small) HC8_FUSED3(4, hc8_small)
HC8_FUSED3(5, hc8_big) HC8_FUSED3(6, hc8_big) HC8_FUSED3(7, hc8_big) HC8_FUSED3(8, hc8_big)
