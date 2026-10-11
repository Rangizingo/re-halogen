// Rewrites of the "ht_deq" family from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o ht_deq.co ht_deq.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). HT format: analysis/closure/ht-decode.md.
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)

typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

// bf16 RNE without NaN special case: v_and 1 ; v_add3 0x7fff (k_ht_deq_t<4>, codeobj_53000.s:555151-555161)
__device__ static inline u16 f2bf(float f) {
  u32 u = __float_as_uint(f);
  return (u16)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// One HT weight from a 16-bit window of the tile bit string:
//   p = x * 0x83dcd12d (v_mul_lo_u32, :555088 etc.), S = byte sum + 0x6400 (v_sad_u8 p,0,0x6400, :555102),
//   w = fma_f16(f16bits(S), f16(0x1eee), f16(0xc931)) (v_pk_fma_f16 ..., s2=0x1eee, literal 0xc931, :555118),
//   widened with v_cvt_f32_f16, then bf16 RNE.
__device__ static inline u16 ht_weight(u32 x) {
  u32 p = x * 0x83dcd12du;
  u32 s = 0x6400u + (p & 255u) + ((p >> 8) & 255u) + ((p >> 16) & 255u) + (p >> 24);
  _Float16 hs = __builtin_bit_cast(_Float16, (u16)s);
  _Float16 sc = __builtin_bit_cast(_Float16, (u16)0x1eee);
  _Float16 of = __builtin_bit_cast(_Float16, (u16)0xc931);
  _Float16 w = __builtin_fmaf16(hs, sc, of);
  return f2bf((float)w);
}

// 16-bit window at bit position `pos` of the tile bit string U, where U is the 8*B words of the tile read in reverse
// (last word = lowest bits) and the string is circular. For B=4 this is the pair {w[t-1], w[t]} >> 4*kk of ht-decode.md,
// for B=3 the window step is 3 bits (tile = 24 words, stride 0x60 bytes, :554592).
template <int B>
__device__ static inline u32 tile_window(const u32* tw, int pos) {
  constexpr int W = 8 * B;
  int wi = (W - 1) - (pos >> 5), off = pos & 31;
  int wn = wi == 0 ? W - 1 : wi - 1;
  u64 pair = ((u64)tw[wn] << 32) | tw[wi];
  return (u32)(pair >> off) & 0xffffu;
}

// halogen::(anonymous namespace)::k_ht_deq_t<B>(unsigned const* w, int K, int N, unsigned short* out)
// kernarg w@0 K@8 N@12 out@16. One wave32 per 16x16 tile, 8 per block. g < (K/16)*(N/16); with K16 = K/16:
//   q = g / (8*K16), r = g - q*8*K16, ct = r>>3, rt = r&7        (division :554944-554972 and :554787-554941)
//   tile id = (q*K16 + ct)*8 + rt, tile at w + tile*8*B words      (0x60 stride for B=3 :554592; <<7 for B=4)
//   lane t = 16*h + rr holds row 128q + 16rt + rr, columns 16ct + 8h + c (c = 0..7), c = 7 - kk
//   out[row*K + col] (v_mad_i64_i32 with s4 = K, :555051); one b128 store of 8 bf16 (:555166).
template <int B>
__device__ static inline void deq_t(const u32* w, int K, int N, u16* out) {
  long g = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  int K16 = K / 16, N16 = N / 16;
  if (g >= (long)K16 * N16) return;
  long q = g / (8L * K16);
  long r = g - q * 8L * K16;
  int ct = (int)(r >> 3), rt = (int)(r & 7);
  long tile = (q * K16 + ct) * 8 + rt;
  const u32* tw = w + tile * (8 * B);
  int h = lane >> 4, rr = lane & 15;
  long row = q * 128 + rt * 16 + rr;
  u16 o[8];
#pragma unroll
  for (int c = 0; c < 8; c++) {
    int j = 128 * h + 8 * rr + c;
    int pos = B * (255 - j);
    o[c] = ht_weight(tile_window<B>(tw, pos));
  }
  u16* dst = out + row * K + 16 * ct + 8 * h;
#pragma unroll
  for (int c = 0; c < 8; c++) dst[c] = o[c];
}

// ---- k_ht_deq_orig<B, T, W>(u32 const* w, int K, half const* a, half const* b, u16* out, int flag) ----
// kernarg w@0 K@8 a@16 b@24 out@32 flag@40. One block per 128x128 region (rows 128*rb.., cols 128*cb..): flag == 0 gives
// cb = blockIdx.y, rb = blockIdx.x, else cb = blockIdx.x, rb = blockIdx.y (s_cselect s8/s11, listing
// k_ht_deq_orig<4,false,16> codeobj_53000.s:451937-451940). Wave (ct = 8*cb + wave%8) decodes tiles (rb, ct, rt) with
// tile id (rb*K16 + ct)*8 + rt (:451950, <<10) (same decode as k_ht_deq_t, f16 value kept as f32 without bf16 rounding,
// :452310-452341 v_cvt_f32_f16 then ds_store_b32).
// Phase 1 (:452379-452683 loads/butterfly, :452687-452760 second stage): 128-point WHT over the columns of every row:
//   32-point iterative WHT over columns 32g..32g+31 (bit order 0..4, lo+hi / lo-hi), then a 4-point over g (stride 32),
//   * 1/sqrt(128) (v_mul 0x3db504f3), then * a[128*cb + col] (fma_mix, -0), stored back as f32.
// Phase 2 (:452766-453046 and :453049-453153): the same over the rows (axis of 128 rows), * 1/sqrt(128), * b[128*rb + row]
// (fma_mix -0), bf16 RNE (no NaN case), out[(128*rb + row)*K + 128*cb + col].
__device__ static inline void wht32(float* x) {
  for (int len = 1; len < 32; len <<= 1)
    for (int i = 0; i < 32; i += 2 * len)
      for (int j = i; j < i + len; j++) {
        float lo = x[j], hi = x[j + len];
        x[j] = lo + hi;
        x[j + len] = lo - hi;
      }
}
__device__ static inline void wht4_scaled(float* y4, const float* x4) {
  float p0 = x4[0] + x4[1], m0 = x4[0] - x4[1], p1 = x4[2] + x4[3], m1 = x4[2] - x4[3];
  y4[0] = p0 + p1;
  y4[1] = m0 + m1;
  y4[2] = p0 - p1;
  y4[3] = m0 - m1;
}
template <int B>
__device__ static inline void deq_orig(const u32* w, int K, const _Float16* a, const _Float16* b, u16* out, int flag) {
  __shared__ float S[128 * 128];  // [row][col]
  int cb = flag == 0 ? blockIdx.y : blockIdx.x;
  int rb = flag == 0 ? blockIdx.x : blockIdx.y;
  int K16 = K / 16;
  int nt = blockDim.x;
  for (int e = threadIdx.x; e < 128 * 128; e += nt) {
    int r = e >> 7, c = e & 127;
    int rt = r >> 4, rr = r & 15, ctl = c >> 4, cc = c & 15;
    long tile = ((long)rb * K16 + 8 * cb + ctl) * 8 + rt;
    const u32* tw = w + tile * (8 * B);
    int h = cc >> 3, ci = cc & 7;
    int j = 128 * h + 8 * rr + ci;
    u32 x = tile_window<B>(tw, B * (255 - j));
    u32 p = x * 0x83dcd12du;
    u32 s = 0x6400u + (p & 255u) + ((p >> 8) & 255u) + ((p >> 16) & 255u) + (p >> 24);
    _Float16 hv = __builtin_fmaf16(__builtin_bit_cast(_Float16, (u16)s), __builtin_bit_cast(_Float16, (u16)0x1eee),
                                   __builtin_bit_cast(_Float16, (u16)0xc931));
    S[r * 128 + c] = (float)hv;
  }
  __syncthreads();
  const float s3 = __uint_as_float(0x3db504f3u);
  // phase 1: transform + scale over columns, one row per thread iteration
  for (int r = threadIdx.x; r < 128; r += nt) {
    float seg[4][32];
    for (int g = 0; g < 4; g++) {
      for (int j = 0; j < 32; j++) seg[g][j] = S[r * 128 + 32 * g + j];
      wht32(seg[g]);
    }
    for (int j = 0; j < 32; j++) {
      float x4[4] = {seg[0][j], seg[1][j], seg[2][j], seg[3][j]}, y4[4];
      wht4_scaled(y4, x4);
      for (int g = 0; g < 4; g++) {
        float y = y4[g] * s3;
        S[r * 128 + 32 * g + j] = __builtin_fmaf((float)a[128 * cb + 32 * g + j], y, -0.0f);
      }
    }
  }
  __syncthreads();
  // phase 2: over rows, one column per thread iteration
  for (int c = threadIdx.x; c < 128; c += nt) {
    float seg[4][32];
    for (int g = 0; g < 4; g++) {
      for (int j = 0; j < 32; j++) seg[g][j] = S[(32 * g + j) * 128 + c];
      wht32(seg[g]);
    }
    for (int j = 0; j < 32; j++) {
      float x4[4] = {seg[0][j], seg[1][j], seg[2][j], seg[3][j]}, y4[4];
      wht4_scaled(y4, x4);
      for (int g = 0; g < 4; g++) {
        float y = y4[g] * s3;
        int row = 32 * g + j;
        out[((long)(128 * rb + row)) * K + 128 * cb + c] = f2bf(__builtin_fmaf((float)b[128 * rb + row], y, -0.0f));
      }
    }
  }
}
extern "C" __global__ void k_ht_deq_orig_b3(const u32* w, int K, const _Float16* a, const _Float16* b, u16* out, int flag) {
  deq_orig<3>(w, K, a, b, out, flag);
}
extern "C" __global__ void k_ht_deq_orig_b4(const u32* w, int K, const _Float16* a, const _Float16* b, u16* out, int flag) {
  deq_orig<4>(w, K, a, b, out, flag);
}

extern "C" __global__ void k_ht_deq_t3(const u32* w, int K, int N, u16* out) { deq_t<3>(w, K, N, out); }
extern "C" __global__ void k_ht_deq_t4(const u32* w, int K, int N, u16* out) { deq_t<4>(w, K, N, out); }
