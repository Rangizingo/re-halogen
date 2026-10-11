// Rewrites of the "ht_tg" and "ht_tg3" families from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o ht_tg.co ht_tg.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)

typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef _Float16 half_t;

__device__ static inline u16 f2bf(float f) {
  u32 u = __float_as_uint(f);
  return (u16)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
__device__ static inline float xs(float v, int lane, int bit) {
  float p = __shfl_xor(v, bit);
  return ((lane & bit) ? -v : v) + p;
}
#define S3 __uint_as_float(0x3db504f3u)

typedef half_t v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

// HT weight value (f16), see ht_deq.hip / ht_moe.hip.
__device__ static inline half_t ht_w16(u32 x) {
  u32 p = x * 0x83dcd12du;
  u32 s = 0x6400u + (p & 255u) + ((p >> 8) & 255u) + ((p >> 16) & 255u) + (p >> 24);
  return __builtin_fmaf16(__builtin_bit_cast(half_t, (u16)s), __builtin_bit_cast(half_t, (u16)0x1eee),
                          __builtin_bit_cast(half_t, (u16)0xc931));
}
template <int B>
__device__ static inline u32 tile_window(const u32* tw, int pos) {
  constexpr int W = 8 * B;
  int wi = (W - 1) - (pos >> 5), off = pos & 31;
  int wn = wi == 0 ? W - 1 : wi - 1;
  u64 pair = ((u64)tw[wn] << 32) | tw[wi];
  return (u32)(pair >> off) & 0xffffu;
}
template <int B>
__device__ static inline v16h tile_row16(const u32* tw, int rr) {
  v16h r;
#pragma unroll
  for (int h = 0; h < 2; h++)
#pragma unroll
    for (int c = 0; c < 8; c++) {
      int j = 128 * h + 8 * rr + c;
      r[8 * h + c] = ht_w16(tile_window<B>(tw, B * (255 - j)));
    }
  return r;
}
__device__ static inline void lanes5(float* r, int lane, int n) {
  for (int k = 0; k < n; k++) {
    float v = r[k];
    v = ((lane & 1) ? -v : v) + __shfl_xor(v, 1);
    v = ((lane & 2) ? -v : v) + __shfl_xor(v, 2);
    v = ((lane & 4) ? -v : v) + __shfl_xor(v, 4);
    v = ((lane & 8) ? -v : v) + __shfl_xor(v, 8);
    v = ((lane & 16) ? -v : v) + __shfl_xor(v, 16);
    r[k] = v;
  }
}
__device__ static inline void reg4(const float* a, float* r) {
  float p0 = a[0] + a[1], m0 = a[0] - a[1], p1 = a[2] + a[3], m1 = a[2] - a[3];
  r[0] = p0 + p1; r[1] = m0 + m1; r[2] = p0 - p1; r[3] = m0 - m1;
}

// k_ht_tg3<B>(u32 const* w, half const* x, int M, int K, int N, int kb, half const* sv, u16* out, float* part)
// kernarg w@0 x@8 M@16 K@20 N@24 kb@28 sv@32 out@40 part@48. Block 512 (16 waves), grid (ceil(M/128), ceil(N/256), splits).
// codeobj_53000.s:472224-... (k_ht_tg3<4> at :472224). Block x: 128 tokens (s22 = x*128, count min(M - s22, 128), :472250-472252);
// block y: 256 weight rows = two 128-row blocks 2y + (wave>>3), wave&7 = row tile (:472375 v13 = s11 + v88);
// block z: K range [z*kb, min(K/128, (z+1)*kb)) in 128-blocks (:472253-472261), 16 steps per WMMA.
// Weights are the WMMA A operand (tile id (rowblk*K16 + ct)*8 + rt), activations the B operand; accumulate from 0 in k order.
// part != null (s[16:17], :472231): part += z*M*N floats and the raw f32 accumulators are stored, part[tok*N + y*256 +
// 128*half + 4*lane + e] (global_store_b128 :472717, second half +512 B :472858). part == null: the 128-point rot-style
// transform per token (reg4 + 5 lane stages), * 1/sqrt(128), * sv[y*256 + 128*half + 4*lane + e] (fma_mix -0), bf16 RNE
// stores at out[tok*N + ...] (:472848, +256 B for the second half :472976).
template <int B, int NW, int GRPS>
__device__ static inline void tg_gen(const u32* w, const half_t* x, int M, int K, int N, int kb, const half_t* sv, u16* out, float* part) {
  constexpr int RB = NW / 8;          // 128-row blocks per block (tg3: 2, tg: 1)
  constexpr int MAXT = 64 * GRPS;
  constexpr int NT = 4 * GRPS;
  __shared__ float S[128 * RB * 64];
  int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
  int cnt = M - MAXT * bx;
  if (cnt > MAXT) cnt = MAXT;
  int K16 = K / 16, nblk = K / 128;
  int y2 = RB * by + (wave >> 3), rt = wave & 7;
  int k0 = bz * kb, k1 = k0 + kb;
  if (k1 > nblk) k1 = nblk;
  int m = lane & 15;
  v8f acc[NT];
  for (int t = 0; t < NT; t++) acc[t] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
  const half_t* ap[NT];
  for (int t = 0; t < NT; t++) {
    int tk = 16 * t + m;
    if (tk > cnt - 1) tk = cnt - 1;
    ap[t] = x + (long)(MAXT * bx + tk) * K;
  }
  int ntiles = (cnt + 15) >> 4;
  for (int ks = 8 * k0; ks < 8 * k1; ks++) {
    const u32* tw = w + ((long)(y2 * K16 + ks) * 8 + rt) * (8 * B);
    v16h wa = tile_row16<B>(tw, m);
    for (int t = 0; t < ntiles; t++) {
      v16h bv;
      for (int k = 0; k < 16; k++) bv[k] = ap[t][ks * 16 + k];
      acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(wa, bv, acc[t]);
    }
  }
  float* pp = part ? part + (long)bz * M * N : nullptr;
  for (int grp = 0; grp < GRPS; grp++) {
    __syncthreads();
    for (int tt = 0; tt < 4; tt++)
      for (int i = 0; i < 8; i++) S[(128 * (wave >> 3) + 16 * rt + 2 * i + (lane >> 4)) * 64 + 16 * tt + m] = acc[4 * grp + tt][i];
    __syncthreads();
    for (int iter = 0; iter < 64 / NW; iter++) {
      int tl = wave + NW * iter;
      int t = 64 * grp + tl;
      if (t >= cnt) continue;
      long tok = (long)MAXT * bx + t;
      for (int half = 0; half < RB; half++) {
        float a4[4];
        for (int e = 0; e < 4; e++) a4[e] = S[(128 * half + 4 * lane + e) * 64 + tl];
        long col = (long)by * (128 * RB) + 128 * half + 4 * lane;
        if (pp) {
          for (int e = 0; e < 4; e++) pp[tok * N + col + e] = a4[e];
        } else {
          float r4[4];
          reg4(a4, r4);
          lanes5(r4, lane, 4);
          for (int e = 0; e < 4; e++) {
            float y = r4[e] * S3;
            out[tok * N + col + e] = f2bf(__builtin_fmaf((float)sv[col + e], y, -0.0f));
          }
        }
      }
    }
  }
}
extern "C" __global__ void k_ht_tg3_b3(const u32* w, const half_t* x, int M, int K, int N, int kb, const half_t* sv, u16* out, float* part) {
  tg_gen<3, 16, 2>(w, x, M, K, N, kb, sv, out, part);
}
extern "C" __global__ void k_ht_tg3_b4(const u32* w, const half_t* x, int M, int K, int N, int kb, const half_t* sv, u16* out, float* part) {
  tg_gen<4, 16, 2>(w, x, M, K, N, kb, sv, out, part);
}
// k_ht_tg<B>: block 256 (8 waves), 64 tokens per x tile, 128 rows per y tile (s28 = x*64 :493615, s0 = y<<7 :494737).
extern "C" __global__ void k_ht_tg_b3(const u32* w, const half_t* x, int M, int K, int N, int kb, const half_t* sv, u16* out, float* part) {
  tg_gen<3, 8, 1>(w, x, M, K, N, kb, sv, out, part);
}
extern "C" __global__ void k_ht_tg_b4(const u32* w, const half_t* x, int M, int K, int N, int kb, const half_t* sv, u16* out, float* part) {
  tg_gen<4, 8, 1>(w, x, M, K, N, kb, sv, out, part);
}

// k_ht_tg_sum(float const* part, int nsplit, half const* sv, u16* out, int a, int b)
// kernarg part@0 nsplit@8 sv@16 out@24 a@32 b@36   (codeobj_53000.s:21731-22144)
// One wave32 per 128 columns of one row (8 per block): nb = b/128, g < nb*a, row = g/nb, blk = g%nb.
// Sum the split-K partials in order (first load :21971, loop adds :22006-22007, stride a*b floats :21993),
// then the same 128-point butterfly + 1/sqrt(128) + sv multiply + bf16 RNE as k_ht_rot_out_f32.
extern "C" __global__ void k_ht_tg_sum(const float* part, int nsplit, const half_t* sv, u16* out, int a, int b) {
  int nb = b / 128;
  long g = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (g >= (long)nb * a) return;
  int row = (int)(g / nb), blk = (int)(g - (long)row * nb);
  long eo = (long)row * b + blk * 128 + lane * 4;
  float x[4];
  for (int i = 0; i < 4; i++) x[i] = part[eo + i];
  for (int k = 1; k < nsplit; k++) {
    const float* pk = part + (long)k * b * a + eo;
    for (int i = 0; i < 4; i++) x[i] += pk[i];
  }
  float p0 = x[0] + x[1], m0 = x[0] - x[1], p1 = x[2] + x[3], m1 = x[2] - x[3];
  float r[4] = {p0 + p1, m0 + m1, p0 - p1, m0 - m1};
#pragma unroll
  for (int k = 0; k < 4; k++) {
    float v = r[k];
    v = xs(v, lane, 1);
    v = xs(v, lane, 2);
    v = xs(v, lane, 4);
    v = xs(v, lane, 8);
    v = xs(v, lane, 16);
    r[k] = v;
  }
  int so = blk * 128 + lane * 4;
  for (int i = 0; i < 4; i++) {
    float y = r[i] * S3;
    out[eo + i] = f2bf(__builtin_fmaf((float)sv[so + i], y, -0.0f));
  }
}
