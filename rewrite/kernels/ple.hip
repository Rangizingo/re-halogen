// Rewrites of the "ple" family (per-layer-embedding path) from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o ple.co ple.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

// The vendor fuses only where the listing shows v_fma/v_fmac.
#pragma clang fp contract(off)

typedef long long ll;
typedef unsigned long long ull;

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
// silu(v) = v / (1 + exp2(v * -1.442695)), IEEE division (codeobj_53000.s:32516-32541).
__device__ static inline float silu(float v) {
  float e = __builtin_amdgcn_exp2f(v * -1.44269502162933349609375f);
  return v / (1.0f + e);
}
__device__ static inline float sigm(float v) {
  float e = __builtin_amdgcn_exp2f(v * -1.44269502162933349609375f);
  return 1.0f / (1.0f + e);
}

// Tree sum over LDS as in the other kernels: s = blockDim/2 ... 1, lds[t] = lds[t+s] + lds[t].
__device__ static inline float block_sum(float part, float* lds) {
  int tid = threadIdx.x;
  lds[tid] = part;
  __syncthreads();
  for (unsigned s = blockDim.x >> 1; s != 0; s >>= 1) {
    if (tid < (int)s) lds[tid] = lds[tid + s] + lds[tid];
    __syncthreads();
  }
  return lds[0];
}

// k_ple_gate(unsigned short const* a, unsigned short const* b, float* out)
// _ZN7halogen12_GLOBAL__N_110k_ple_gateEPKtS2_Pf, kernarg: a@0, b@8, out@16
// Block (x = token t, y = gate j). Element offset t*10240 + j*2560 (s_mul 0x2800 and 0xa00, codeobj_53000.s:31651-31652),
// then shifted left by 1 into bytes (v_lshl_add_u64, :31669). Each thread: acc = fma(a, b, acc) over i = tid, tid+blockDim ... < 2560
// (:31690-31691, bound 0x9ff). Tree sum (:31697-31723). Thread 0: t = rsq(2560.0) * total (v_rsq_f32 0x45200000, :31728, s_mul_f32 :31733);
// m = max(|t|, 0x358637bd) (:31735-31736); s = correctly rounded sqrt(m) (v_sqrt_f32 plus fma fixup, scaling constants
// 0x4f800000 / 0xf800000 / 0x37800000, :31738-31772); out[t*4 + j] = t < 0 ? -s : s (:31742-31778, s_lshl2_add_u32 :31764).
extern "C" __global__ void k_ple_gate(const unsigned short* a, const unsigned short* b, float* out) {
  __shared__ float lds[1024];
  long off = (long)blockIdx.x * 10240 + (long)blockIdx.y * 2560;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < 2560; i += blockDim.x) acc = __builtin_fmaf(bf2f(a[off + i]), bf2f(b[off + i]), acc);
  float total = block_sum(acc, lds);
  if (threadIdx.x == 0) {
    float c = 2560.0f;
    asm volatile("" : "+v"(c));
    float t = __builtin_amdgcn_rsqf(c) * total;
    float m = fmaxf(fabsf(t), __uint_as_float(0x358637bdu));
    float s = sqrtf(m);
    out[blockIdx.x * 4 + blockIdx.y] = t < 0.0f ? -s : s;
  }
}

// k_ple_gv(float const* g, unsigned short const* v, unsigned short* out)
// _ZN7halogen12_GLOBAL__N_18k_ple_gvEPKfPKtPt, kernarg: g@0, v@8, out@16
// Block x = token t. i = tid, tid+blockDim ... < 10240 (bound 0x27ff, :31841). j = i / 2560 (v_mul_u32_u24 0xcccd then >> 27, :31810-31811);
// gate = g[t*4 + j] (:31813-31822); c = i - j*2560 (:31823); value = v[t*2560 + c] (stride 0x1400 bytes, :31789-31798, :31839);
// out[t*10240 + i] (stride 0x5000 bytes, :31793-31800) = bf16_rne(sigmoid(gate) * bf16(value)) (:31844-31869).
extern "C" __global__ void k_ple_gv(const float* g, const unsigned short* v, unsigned short* out) {
  long t = blockIdx.x;
  for (int i = threadIdx.x; i < 10240; i += blockDim.x) {
    int j = i / 2560;
    int c = i - j * 2560;
    float gate = g[t * 4 + j];
    float s = sigm(gate);
    out[t * 10240 + i] = f2bf_rne(s * bf2f(v[t * 2560 + c]));
  }
}

// Causal 4-tap dilation-3 conv for one (t, c) with a 9-step history, then silu and a residual add.
//   in    [T][10240] bf16 (row stride 0x5000 bytes): the new inputs   (arg8 in k_ple_out_p / k_ple_out)
//   w     [10240][4] bf16 (c*8 bytes, :32363 neighbours)              (arg16)
//   hist  [10240][9] bf16 (c*18 bytes, v_mad_i64_i32 18, :32363)      (arg24)
//   tap k reads source row j = t - 9 + 3k: j >= 0 -> in[j][c] if j < T, else nothing (acc unchanged, :32378-32380);
//   j < 0 -> hist[c][j + 9] (= hist[c*9 + t + 3k]); acc starts at 0 and each tap is fma(w[k], x, acc)
//   (v_fma ..., 0 for the first, v_fmac after, :32391, :32425, :32460, :32495).
//   result = bf16_rne(silu(acc) + bf16(a[t][c]))  (:32515-32546)
__device__ static inline unsigned short ple_elem(int t, int c, int T, const unsigned short* a, const unsigned short* in,
                                                 const unsigned short* w, const unsigned short* hist) {
  float acc = 0.0f;
  for (int k = 0; k < 4; k++) {
    int j = t - 9 + 3 * k;
    float wk = bf2f(w[(long)c * 4 + k]);
    if (j >= 0) {
      if (j < T) acc = __builtin_fmaf(wk, bf2f(in[(long)j * 10240 + c]), acc);
    } else {
      acc = __builtin_fmaf(wk, bf2f(hist[(long)c * 9 + j + 9]), acc);
    }
  }
  float r = silu(acc) + bf2f(a[(long)t * 10240 + c]);
  return f2bf_rne(r);
}

// k_ple_out_p(unsigned short const* a, unsigned short const* in, unsigned short const* w, unsigned short const* hist, int T, unsigned short* out)
// _ZN7halogen12_GLOBAL__N_111k_ple_out_pEPKtS2_S2_S2_iPt, kernarg: a@0, in@8, w@16, hist@24, T@32, out@40
// t = blockIdx.x (s2), c = blockIdx.y*blockDim.x + tid (v_mad_u64_u32 with s3, :32343), exit if c >= 10240 (0x2800, :32345).
// Residual a, input rows and output share the [t][10240] layout (:32355-32356, :32507).
extern "C" __global__ void k_ple_out_p(const unsigned short* a, const unsigned short* in, const unsigned short* w,
                                       const unsigned short* hist, int T, unsigned short* out) {
  int t = blockIdx.x;
  int c = blockIdx.y * blockDim.x + threadIdx.x;
  if (c >= 10240) return;
  out[(long)t * 10240 + c] = ple_elem(t, c, T, a, in, w, hist);
}

// k_ple_out(...) same kernarg as k_ple_out_p (_ZN7halogen12_GLOBAL__N_19k_ple_outEPKtS2_S2_S2_iPt):
// the same element function, but one block per time step with a loop c = tid, tid+blockDim ... < 10240 (0x27ff, :31986),
// pointers advanced by t*0x5000 bytes (:31888-31946).
extern "C" __global__ void k_ple_out(const unsigned short* a, const unsigned short* in, const unsigned short* w,
                                     const unsigned short* hist, int T, unsigned short* out) {
  int t = blockIdx.x;
  for (int c = threadIdx.x; c < 10240; c += blockDim.x) out[(long)t * 10240 + c] = ple_elem(t, c, T, a, in, w, hist);
}

// k_ple_conv_state_update(unsigned short const* in, unsigned short* hist, int T)
// _ZN7halogen12_GLOBAL__N_123k_ple_conv_state_updateEPKtPti, kernarg: in@0, hist@8, T@16
// c = blockIdx.x*blockDim.x + tid, exit if c >= 10240 (:32144-32146). New history entry j (0..8):
//   T > 8 - j ? in[(T - 9 + j)*10240 + c] : old hist[c*9 + T + j]   (the s_cmp chain, e.g. :32161-32175 for j=0, :32271-32289)
// All nine values are loaded before the stores (:32331-32333).
extern "C" __global__ void k_ple_conv_state_update(const unsigned short* in, unsigned short* hist, int T) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= 10240) return;
  unsigned short nh[9];
  for (int j = 0; j < 9; j++) nh[j] = T > 8 - j ? in[(long)(T - 9 + j) * 10240 + c] : hist[(long)c * 9 + T + j];
  for (int j = 0; j < 9; j++) hist[(long)c * 9 + j] = nh[j];
}

// PleRowPtrs: 16 pointers (128 bytes). k_ple_out_rows reads element [row] (kernarg + 0x18 + row*8, :32562-32568),
// k_ple_ids_rows reads element [8 + row] (kernarg + 0x48 + row*8, :32673-32677).
struct PleRowPtrs { void* p[16]; };

// k_ple_out_rows(unsigned short const* a, unsigned short const* x, unsigned short const* w, PleRowPtrs rp, unsigned short* out)
// _ZN7halogen12_GLOBAL__N_114k_ple_out_rowsEPKtS2_S2_NS0_10PleRowPtrsEPt, kernarg: a@0, x@8, w@16, rp@24, out@152
// One decode step per row: row = blockIdx.x, c = blockIdx.y*blockDim + tid, exit if c >= 10240 (:32556-32558).
// hist = rp.p[row] + c*9 elements (v_mad_i64_i32 18, :32582). Taps use hist[0], hist[3], hist[6] and the new value
// x[row*10240 + c] (loads :32592-32604; fma chain :32617-32628: w0*h0, then +w1*h3, +w2*h6, +w3*x).
// out[row*10240 + c] = bf16_rne(silu(acc) + bf16(a[row*10240 + c])) (:32630-32661). The history is shifted
// (hist[j] = hist[j+1] for j < 8, hist[8] = x) and written back (:32664-32665).
extern "C" __global__ void k_ple_out_rows(const unsigned short* a, const unsigned short* x, const unsigned short* w,
                                          PleRowPtrs rp, unsigned short* out) {
  int row = blockIdx.x;
  int c = blockIdx.y * blockDim.x + threadIdx.x;
  if (c >= 10240) return;
  unsigned short* h = (unsigned short*)rp.p[row] + (long)c * 9;
  long e = (long)row * 10240 + c;
  unsigned short hv[9], xn = x[e];
  for (int j = 0; j < 9; j++) hv[j] = h[j];
  float acc = __builtin_fmaf(bf2f(w[(long)c * 4 + 0]), bf2f(hv[0]), 0.0f);
  acc = __builtin_fmaf(bf2f(w[(long)c * 4 + 1]), bf2f(hv[3]), acc);
  acc = __builtin_fmaf(bf2f(w[(long)c * 4 + 2]), bf2f(hv[6]), acc);
  acc = __builtin_fmaf(bf2f(w[(long)c * 4 + 3]), bf2f(xn), acc);
  out[e] = f2bf_rne(silu(acc) + bf2f(a[e]));
  for (int j = 0; j < 8; j++) h[j] = hv[j + 1];
  h[8] = xn;
}

// k_ple_ids_rows(int const* ids, PleRowPtrs rp, ll const* c, ll const* d, ll const* e, int n, ll* out)
// _ZN7halogen12_GLOBAL__N_114k_ple_ids_rowsEPKiNS0_10PleRowPtrsEPKxS5_S5_iPx
// kernarg: ids@0, rp@8, c@136, d@144, e@152, n@160, out@168. One block per row (blockIdx.x), 16+ threads.
// ctx = (ll*)rp.p[8 + row]: ctx[0], ctx[1] go to LDS slots 0 and 1 (tid < 2, :32679-32684), slot 2 = (ll)ids[row] (tid == 2, :32688-32700).
// flag (thread 0, :32714-32719): v_cndmask(2, eq01, vcc = n != ctx[1]) so flag = 2 if n == ctx[1], else (n == ctx[0] ? 1 : 0) (n sign-extended, :32711).
// Hash, thread tid < 16 (:32726): h = c[0] * L[2]; for k = 1 .. 1 + (tid >> 3):
//   x = k > 2 - flag ? (ll)n : L[2 - k]   (:32786-32794, LDS reads at byte offsets 8 and 0)
//   h ^= c[k] * x (64-bit product, :32765-32779); stop test k+1 - (tid>>3) == 2 (:32768-32770).
// out[row*16 + tid] = e[tid] + h % d[tid] (signed remainder, :32797-32980; same tail as k_ngram_ids).
// Thread 0 then stores ctx[0] = L[1] (old ctx[1]), ctx[1] = (ll)ids[row] (:32993-32995).
extern "C" __global__ void k_ple_ids_rows(const int* ids, PleRowPtrs rp, const ll* c, const ll* d, const ll* e, int n, ll* out) {
  __shared__ ll L[4];
  __shared__ int flag;
  int tid = threadIdx.x, row = blockIdx.x;
  ll* ctx = (ll*)rp.p[8 + row];
  if (tid < 2) L[tid] = ctx[tid];
  if (tid == 2) L[2] = (ll)ids[row];
  __syncthreads();
  if (tid == 0) flag = (ll)n != L[1] ? ((ll)n == L[0] ? 1 : 0) : 2;
  __syncthreads();
  if (tid < 16) {
    ull h = (ull)c[0] * (ull)L[2];
    int last = 1 + (tid >> 3);
    for (int k = 1; k <= last; k++) {
      ll x = k > 2 - flag ? (ll)n : L[2 - k];
      h ^= (ull)c[k] * (ull)x;
    }
    out[(long)row * 16 + tid] = e[tid] + (ll)h % d[tid];
  }
  __syncthreads();
  if (tid == 0) {
    ll l1 = L[1], l2 = L[2];
    ctx[0] = l1;
    ctx[1] = l2;
  }
}
