// Rewrites of the "hc_scatter" family (hyper-connection scatter, fused RMS norm) from flash_serve, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o hc_scatter.co hc_scatter.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). Lines below are in that file.
#include <hip/hip_runtime.h>

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
__device__ static inline unsigned rne_bits(float f) {  // bfe 16,1 ; add3 0x7fff ; rounded value in the high half (:37013-37015)
  unsigned u = __float_as_uint(f);
  return u + ((u >> 16) & 1u) + 0x7fffu;
}

// halogen::(anonymous namespace)::k_hc_scatter(unsigned short const*, unsigned short const*, float const*, unsigned short*)
// _ZN7halogen12_GLOBAL__N_112k_hc_scatterEPKtS2_PKfPt, kernarg: a@0, b@8, w@16, y@24.
// Row r = blockIdx.x. a and y rows = 0x5000 bytes = 10240 bf16 (:36954-36956, :36964-36970); b row = 0x1400 bytes = 2560 bf16 (:36959, :36961, :36971-36972);
// w row = 4 f32 at w + r*16 (:36960, :36965).
// Loop i = tid; i < 0x2800 = 10240 (:37004); stride blockDim (:36975, group_size_x at 0x2c :36953).
// c = i / 2560 (mul 0xcccd, lshr 27 :36976-36980), j = i - 2560*c (:36981). The w index is c (global_load_b32 at w + 4*c :36982, :36999).
// y[i] = bf16_rne(fma(w[c], f32(b[j]), f32(a[i]))) (v_fmac_f32 :37012, bfe/add3 :37013-37015, store high half :37016).
extern "C" __global__ void k_hc_scatter(const unsigned short* a, const unsigned short* b, const float* w, unsigned short* y) {
  size_t r = blockIdx.x;
  for (unsigned i = threadIdx.x; i < 10240u; i += blockDim.x) {
    unsigned c = i / 2560u;
    unsigned j = i - c * 2560u;
    float t = __builtin_fmaf(w[r * 4 + c], bf2f(b[r * 2560 + j]), bf2f(a[r * 10240 + i]));
    y[r * 10240 + i] = (unsigned short)(rne_bits(t) >> 16);
  }
}

// k_hc_scatter_norm<false> and <true>. Block r = blockIdx.x with blockDim = 256 (LDS 1024 / 1152 bytes in the kernarg metadata).
// kernarg: y0@0 (in/out bf16, row r at +r*0x1400 bytes :1103220-1103222), p@8 (bf16, row r/4 at +(r/4)*0x1400 :1103214, :1103223-1103226),
//          w@16 (f32, w[r] :1103211-1103219), g@24 (bf16 [4][2560] table), out@32 (bf16, row r), tab@40 (bf16 [4][4*2560], only <true>),
//          f@48 (f32, only <true>); group_size_x at 0x44 (:1103213).
// Pass 1: for i = tid; i < 2560; i += blockDim: t = fma(w[r], f32(p[i]), f32(y0[i])) (v_fmac_f32 s0 :1103264),
//         rounded to bf16 RNE (:1103265-1103267), stored back into y0 (:1103269), ss = fma(rb, rb, ss) with rb the rounded value widened (:1103268, :1103270).
// Block reduction: sh[tid] = ss ; barrier ; for s = bd/2; s > 0; s >>= 1: if tid < s: sh[tid] += sh[tid + s]; barrier (:1103275-1103312).
// norm = rsq(sh[0] / 2560.0 + 1e-6f): IEEE division by 0x45200000 (:1103319-1103336), + 0x358637bd (:1103337),
//         if value < 0x800000 scale by 0x4b800000 first (:1103339-1103341), v_rsq_f32 (:1103343), and multiply the result by 0x45800000 when scaled (:1103344, :1103348).
// Pass 2: for i: x = f32(y0[i]); v = (norm * x) * (1 + f32(g[(r%4)*2560 + i])) (:1103395-1103403), out[r*2560 + i] = bf16_rne(v) (:1103412-1103416).
//   <true> also: rb = widened rounded out (:1103704); acc_k = fma(f32(tab[k*10240 + (r%4)*2560 + i]), rb, acc_k) for k = 0..3 (:1103707-1103710, loads :1103657-1103676).
//   Reduction: wave32 butterfly over xor 16, 8, 4, 2, 1 for the four accumulators (:1103721-1103791), lane 0 of each wave stores to LDS
//   at wave*16 bytes (:1103794-1103803), barrier, threads 0..3 sum the 8 wave partials in order ((0 + w0) + w1) ... + w7) (:1103812-1103839)
//   and store f[r*4 + k] (:1103813-1103822, :1103840).
template <bool TAB>
__device__ static inline void scatter_norm_body(unsigned short* y0, const unsigned short* p, const float* w, const unsigned short* g,
                                                unsigned short* out, const unsigned short* tab, float* f) {
  __shared__ float sh[256];
  __shared__ float wp[8 * 4];
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
  bool small = var < 0x1p-126f;                  // 0x800000
  if (small) var = var * 16777216.0f;            // 0x4b800000
  float norm = __builtin_amdgcn_rsqf(var);
  if (small) norm = norm * 4096.0f;              // 0x45800000
  const unsigned m = r & 3u;
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  for (unsigned i = tid; i < 2560u; i += bd) {
    float x = bf2f(yr[i]);
    float gv = 1.0f + bf2f(g[m * 2560 + i]);
    float v = (norm * x) * gv;
    unsigned ob = rne_bits(v);
    out[(size_t)r * 2560 + i] = (unsigned short)(ob >> 16);
    if (TAB) {
      float rb = __uint_as_float(ob & 0xffff0000u);
      a0 = __builtin_fmaf(bf2f(tab[0 * 10240 + m * 2560 + i]), rb, a0);
      a1 = __builtin_fmaf(bf2f(tab[1 * 10240 + m * 2560 + i]), rb, a1);
      a2 = __builtin_fmaf(bf2f(tab[2 * 10240 + m * 2560 + i]), rb, a2);
      a3 = __builtin_fmaf(bf2f(tab[3 * 10240 + m * 2560 + i]), rb, a3);
    }
  }
  if (TAB) {
    for (int off = 16; off >= 1; off >>= 1) {
      a0 = a0 + __shfl_xor(a0, off);
      a1 = a1 + __shfl_xor(a1, off);
      a2 = a2 + __shfl_xor(a2, off);
      a3 = a3 + __shfl_xor(a3, off);
    }
    if ((tid & 31u) == 0) {
      unsigned wv = tid >> 5;
      wp[wv * 4 + 0] = a0;
      wp[wv * 4 + 1] = a1;
      wp[wv * 4 + 2] = a2;
      wp[wv * 4 + 3] = a3;
    }
    __syncthreads();
    if (tid < 4) {
      float s = 0.0f + wp[0 * 4 + tid];
      s = s + wp[1 * 4 + tid];
      s = s + wp[2 * 4 + tid];
      s = s + wp[3 * 4 + tid];
      s = s + wp[4 * 4 + tid];
      s = s + wp[5 * 4 + tid];
      s = s + wp[6 * 4 + tid];
      s = s + wp[7 * 4 + tid];
      f[(size_t)r * 4 + tid] = s;
    }
  }
}

// halogen::(anonymous namespace)::k_hc_scatter_norm<false>(...): _ZN7halogen12_GLOBAL__N_117k_hc_scatter_normILb0EEEvPtPKtPKfS4_S2_S4_Pf
// Arguments 5 and 6 (tab, f) are not read (kernarg loads stop at 0x28: :1103205, :1103218).
extern "C" __global__ void k_hc_scatter_norm_f(unsigned short* y0, const unsigned short* p, const float* w, const unsigned short* g,
                                               unsigned short* out, const unsigned short* tab, float* f) {
  scatter_norm_body<false>(y0, p, w, g, out, tab, f);
}

// halogen::(anonymous namespace)::k_hc_scatter_norm<true>(...): _ZN7halogen12_GLOBAL__N_117k_hc_scatter_normILb1EEEvPtPKtPKfS4_S2_S4_Pf
extern "C" __global__ void k_hc_scatter_norm_t(unsigned short* y0, const unsigned short* p, const float* w, const unsigned short* g,
                                               unsigned short* out, const unsigned short* tab, float* f) {
  scatter_norm_body<true>(y0, p, w, g, out, tab, f);
}
