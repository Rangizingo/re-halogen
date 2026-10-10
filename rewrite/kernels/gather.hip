// Rewrites of the "gather" family from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o gather.co gather.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

// k_gather_rows(unsigned short const* tab, int const* ids, int n, unsigned short* out)
// _ZN7halogen12_GLOBAL__N_113k_gather_rowsEPKtPKiiPt, kernarg: tab@0, ids@8, n@16 (i32), out@24
// One block per row: row = blockIdx.x, exit if row >= n (codeobj_53000.s:48673-48674).
// id = ids[row]; source row = id / 10 (signed, magic 0x66666667 with shifts, :48687-48692); copy 2560 bf16
// (row stride 0x1400 bytes, :48688-48696; the 0xa00 trip count is :48699, bound 0x9ff :48769) to out[row*2560].
// The vendor has a 4-element unrolled path when blockDim.x == 1 (s_cmp_lg_u32 s1,1, :48693); same result.
extern "C" __global__ void k_gather_rows(const unsigned short* tab, const int* ids, int n, unsigned short* out) {
  int row = blockIdx.x;
  if (row >= n) return;
  int id = ids[row];
  const unsigned short* src = tab + (long)(id / 10) * 2560;
  unsigned short* dst = out + (long)row * 2560;
  for (int j = threadIdx.x; j < 2560; j += blockDim.x) dst[j] = src[j];
}

// k_gather_slots_h(unsigned short const* tab, int const* ids, int n, unsigned short* out)
// _ZN7halogen12_GLOBAL__N_116k_gather_slots_hEPKtPKiiPt, kernarg: tab@0, ids@8, n@16, out@24
// row = blockIdx.x, exit if row >= n (:48967-48968), exit if tid >= 640 (v_cmpx_gt_u32 0x280, :48970).
// id = ids[row] (:48984); source row id (stride 0x500 bytes = 640 bf16, :48992-48993); out[row*640 + j] (:48979-48980).
// j = tid, tid+blockDim, ... < 640 (:49078 bound 0x27f).
extern "C" __global__ void k_gather_slots_h(const unsigned short* tab, const int* ids, int n, unsigned short* out) {
  int row = blockIdx.x;
  if (row >= n) return;
  int id = ids[row];
  const unsigned short* src = tab + (long)id * 640;
  unsigned short* dst = out + (long)row * 640;
  for (int j = threadIdx.x; j < 640; j += blockDim.x) dst[j] = src[j];
}

// ---------------------------------------------------------------------------------------------------------------
// Quantized row gathers. All six kernels share one launch shape (codeobj_53000.s:31220-31230 for iq4nl, same prologue
// in the others): grid (x = token, y = 0..15), thread j < 160 (v_cmpx_gt_u32 0xa0). The segment index is a 64-bit entry
// rows[x*16 + y] (s_lshl_b64 s,7 for x*128 bytes, + y*8, then s_load_b64). Each segment holds 160 weights (5 blocks of 32);
// the dequantized weights go to out[x*2560 + y*160 + j] as bf16 (output stride 0x1400 bytes per token). The loop runs
// j = tid, tid+blockDim, ... < 160 (bound 0x9f); the vendor has an unrolled path for blockDim.x == 1 with the same results.

__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// Software fp16 -> fp32 as the vendor does it (not v_cvt_f32_f16), e.g. codeobj_53000.s:2405779-2405805:
//   e = (h >> 10) & 31, m = h & 0x3ff
//   e == 0:  float(m) * 2^-24 (0x33800000), sign by h bit 15
//   e == 31: bits = (h << 13) | 0x7f800000 with the sign bit forced from h (:2405929-2405933)
//   else:    bits = (e << 23 | m << 13) + 0x38000000, sign from h (:2405924-2405933)
__device__ static inline float h2f(unsigned h) {
  unsigned e = (h >> 10) & 31u, m = h & 0x3ffu;
  float r;
  if (e == 0) {
    r = (float)m * __uint_as_float(0x33800000u);
    return (h & 0x8000u) ? -r : r;
  }
  unsigned bits = e != 31u ? (((e << 23) | (m << 13)) + 0x38000000u) : ((h << 13) | 0x7f800000u);
  bits = (bits & 0x7fffffffu) | (((unsigned)(int)(short)h) & 0x80000000u);
  return __uint_as_float(bits);
}

__device__ static inline unsigned short ld16(const unsigned char* p) { return (unsigned short)(p[0] | (p[1] << 8)); }

// kvalues_iq4nl, the 16-entry table the vendor reads through a pc-relative address (s_getpc_b64 / s_add_u32 0xfffd2a6c,
// codeobj_53000.s:31592-31594, global_load_i8 :31625). Contents are the standard ggml table (checked by the MATCH rows).
__constant__ signed char kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// k_gather_iq4nl(unsigned char const* tab, long long const* rows, unsigned short* out)
// _ZN7halogen12_GLOBAL__N_114k_gather_iq4nlEPKhPKxPt, kernarg: tab@0, rows@8, out@16
// Segment = tab + idx*90 (0x5a, :31443-31447); block b = j >> 5 starts at b*18: f16 scale at +0 (:31611), 16 nibble bytes at +2
// (:31615); j & 16 selects the high nibble, else the low one (:31616-31623); value = float(kvalues[nib]) * float(d)
// (v_fma_mix_f32 with -0 addend, :31630), bf16 RNE (:31631-31634).
extern "C" __global__ void k_gather_iq4nl(const unsigned char* tab, const long long* rows, unsigned short* out) {
  int x = blockIdx.x, y = blockIdx.y;
  long long idx = rows[x * 16 + y];
  const unsigned char* seg = tab + idx * 90;
  unsigned short* dst = out + (long)x * 2560 + y * 160;
  for (int j = threadIdx.x; j < 160; j += blockDim.x) {
    const unsigned char* blk = seg + (j >> 5) * 18;
    float d = h2f(ld16(blk));
    unsigned char q = blk[2 + (j & 15)];
    unsigned nib = (j & 16) ? (q >> 4) : (q & 15);
    dst[j] = f2bf_rne(d * (float)kIq4nl[nib]);
  }
}

// k_gather_ggml<2>: 34-byte blocks (2*...: segment = idx*170 = 0xaa, codeobj_53000.s:2405577-2405582), f16 d at +0,
// int8 q[32] at +2: value = d * float(q[j & 31]) (:2405635, :2405751), bf16 RNE (:2405753-2405755).
extern "C" __global__ void k_gather_ggml2(const unsigned char* tab, const long long* rows, unsigned short* out) {
  int x = blockIdx.x, y = blockIdx.y;
  long long idx = rows[x * 16 + y];
  const unsigned char* seg = tab + idx * 170;
  unsigned short* dst = out + (long)x * 2560 + y * 160;
  for (int j = threadIdx.x; j < 160; j += blockDim.x) {
    const unsigned char* blk = seg + (j >> 5) * 34;
    float d = h2f(ld16(blk));
    dst[j] = f2bf_rne(d * (float)(signed char)blk[2 + (j & 31)]);
  }
}

// k_gather_ggml<3>: 22-byte blocks (segment = idx*110 = 0x6e, :2405834-2405839): f16 d at +0, u32 qh at +2 (global_load_b32 offset:2,
// :2405880), 16 nibble bytes qs at +6 (:2405892, :2405902). For jj = j & 31:
//   jj < 16:  x = (qs[jj] & 15) | ((qh >> jj << 4) & 16)
//   jj >= 16: x = (qs[jj-16] >> 4) | ((qh >> (jj-4)) & 16)           (:2405893-2405909)
//   value = float(x - 16) * d  (v_add3_u32 ..., -16, :2405860; v_mul_f32, :2405865), bf16 RNE (:2405867-2405869).
extern "C" __global__ void k_gather_ggml3(const unsigned char* tab, const long long* rows, unsigned short* out) {
  int x = blockIdx.x, y = blockIdx.y;
  long long idx = rows[x * 16 + y];
  const unsigned char* seg = tab + idx * 110;
  unsigned short* dst = out + (long)x * 2560 + y * 160;
  for (int j = threadIdx.x; j < 160; j += blockDim.x) {
    const unsigned char* blk = seg + (j >> 5) * 22;
    int jj = j & 31;
    float d = h2f(ld16(blk));
    unsigned qh = (unsigned)blk[2] | ((unsigned)blk[3] << 8) | ((unsigned)blk[4] << 16) | ((unsigned)blk[5] << 24);
    unsigned v, hi;
    if (jj < 16) { v = blk[6 + jj] & 15u; hi = (qh >> jj << 4) & 16u; }
    else { v = blk[6 + jj - 16] >> 4; hi = (qh >> (jj - 4)) & 16u; }
    dst[j] = f2bf_rne(d * (float)((int)(hi + v) - 16));
  }
}

// k_gather_ggml<4>: 24-byte blocks (segment = idx*120 = 0x78, :2405970-2405975): f16 d at +0, f16 m at +2 (:2406044, :2406074),
// u32 qh at +4 (:2406013), nibbles at +8 (:2406025, :2406035). x as in ggml<3> but without the -16 (:2405992: v_and_or_b32 ..., 16, ...);
// value = fma(d, float(x), m) (v_fmac_f32 v4, v5, v3, :2405998), bf16 RNE (:2406000-2406003).
extern "C" __global__ void k_gather_ggml4(const unsigned char* tab, const long long* rows, unsigned short* out) {
  int x = blockIdx.x, y = blockIdx.y;
  long long idx = rows[x * 16 + y];
  const unsigned char* seg = tab + idx * 120;
  unsigned short* dst = out + (long)x * 2560 + y * 160;
  for (int j = threadIdx.x; j < 160; j += blockDim.x) {
    const unsigned char* blk = seg + (j >> 5) * 24;
    int jj = j & 31;
    float d = h2f(ld16(blk)), m = h2f(ld16(blk + 2));
    unsigned qh = (unsigned)blk[4] | ((unsigned)blk[5] << 8) | ((unsigned)blk[6] << 16) | ((unsigned)blk[7] << 24);
    unsigned v, hi;
    if (jj < 16) { v = blk[8 + jj] & 15u; hi = (qh >> jj << 4) & 16u; }
    else { v = blk[8 + jj - 16] >> 4; hi = (qh >> (jj - 4)) & 16u; }
    dst[j] = f2bf_rne(__builtin_fmaf(d, (float)(hi | v), m));
  }
}

// k_gather_fp8(unsigned char const* tab, long long const* rows, float scale, unsigned short* out)
// _ZN7halogen12_GLOBAL__N_112k_gather_fp8EPKhPKxfPt, kernarg: tab@0, rows@8, scale@16, out@24
// Segment = tab + idx*160 (0xa0, codeobj_53000.s:31242-31247), one byte per weight: e = (b >> 3) & 15, m = b & 7:
//   e == 0: m * 2^-9 (0x3b000000, :31396); else ldexp(fma(m, 0.125, 1.0), e - 7) (:31395, :31397-31398);
//   negated when the byte is negative as int8 (:31401-31403); times scale (v_mul_f32 s10, :31405); bf16 RNE (:31406-31409).
extern "C" __global__ void k_gather_fp8(const unsigned char* tab, const long long* rows, float scale, unsigned short* out) {
  int x = blockIdx.x, y = blockIdx.y;
  long long idx = rows[x * 16 + y];
  const unsigned char* seg = tab + idx * 160;
  unsigned short* dst = out + (long)x * 2560 + y * 160;
  for (int j = threadIdx.x; j < 160; j += blockDim.x) {
    unsigned b = seg[j];
    int e = (b >> 3) & 15;
    float m = (float)(b & 7);
    float v = e == 0 ? m * __uint_as_float(0x3b000000u) : __builtin_ldexpf(__builtin_fmaf(m, 0.125f, 1.0f), e - 7);
    if ((signed char)b < 0) v = -v;
    dst[j] = f2bf_rne(scale * v);
  }
}
