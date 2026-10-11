// Rewrites of the "ngram" family (n-gram speculation helpers) from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o ngram.co ngram.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

typedef long long ll;
typedef unsigned long long ull;

// k_ngram_ctx_update(int const* tokens, long long* ctx, int n)
// _ZN7halogen12_GLOBAL__N_118k_ngram_ctx_updateEPKiPxi, kernarg: tokens@0, ctx@8, n@16
// Only the thread with blockIdx.x | threadIdx.x == 0 runs (v_or_b32 / v_cmpx_eq_u32 0, codeobj_53000.s:33001-33004).
// a = n >= 2 ? tokens[n-2] : ctx[n]   (s_cmp_lt_i32 n,2 :33010; load tokens[n-2] :33026-33028; ctx[n] :33014-33017)
// b = n >  0 ? tokens[n-1] : ctx[n+1] (s_cmp_gt_i32 n,0 :33031; tokens[n-1] :33045-33051; ctx[n+1] :33036-33039)
// Both loads happen before the store; ctx[0] = a, ctx[1] = b (one 16-byte store, :33058). Values sign-extended to 64 bit.
extern "C" __global__ void k_ngram_ctx_update(const int* tokens, ll* ctx, int n) {
  if ((blockIdx.x | threadIdx.x) != 0) return;
  ll a = n >= 2 ? (ll)tokens[n - 2] : ctx[n];
  ll b = n > 0 ? (ll)tokens[n - 1] : ctx[n + 1];
  ctx[0] = a;
  ctx[1] = b;
}

// k_ngram_segstart(long long const* seq, int m, int key, int* out)
// _ZN7halogen12_GLOBAL__N_116k_ngram_segstartEPKxiiPi, kernarg: seq@0, m@8, key@12, out@16
// No thread guard: every thread does the same serial loop (the vendor has no tid test, :30915-30943).
// last = -1; for i = 0 .. m+1 (s_add_i32 s6,s6,2 :30926; skipped when m < -1 :30918):
//   out[i] = last + 1 (:30930-30933); if seq[i] == (long long)key then last = i (s_cmp_eq_u64 / s_cselect :30935-30936).
extern "C" __global__ void k_ngram_segstart(const ll* seq, int m, int key, int* out) {
  if (m < -1) return;
  int last = -1;
  for (int i = 0; i < m + 2; i++) {
    out[i] = last + 1;
    if (seq[i] == (ll)key) last = i;
  }
}

// k_ngram_hist(int const* ids, long long const* tab, int n, long long* out)
// _ZN7halogen12_GLOBAL__N_112k_ngram_histEPKiPKxiPx, kernarg: ids@0, tab@8, n@16, out@24
// Grid-stride loop over i in [0, n+2) (s_add_i32 n,2 :30846; stride = blockDim.x * hidden block_count_x :30856,30862):
//   out[i] = i >= 2 ? (long long)ids[i-2] (load at offset -8 from ids+i*4, :30894-30896) : tab[i] (:30907).
extern "C" __global__ void k_ngram_hist(const int* ids, const ll* tab, int n, ll* out) {
  int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n + 2; i += stride) out[i] = i >= 2 ? (ll)ids[i - 2] : tab[i];
}

// k_ngram_ids(a, b, c, d, e, n, out)
// _ZN7halogen12_GLOBAL__N_111k_ngram_idsEPKxPKiS2_S2_S2_iPx
// kernarg: a@0 (ll*), b@8 (int*), c@16 (ll*), d@24 (ll*), e@32 (ll*), n@40 (int), out@48 (ll*)
// row = blockIdx.x (s2), only tid < 16 run (v_cmpx_gt_u32 16 :30950). The first loop is serial per thread:
//   lim = row - b[row+2] + 2 (:30967, :30975, :30980)
//   h = c[0] * a[row+2]        64-bit product (:30970-30981)
//   for k = 1 ..: x = (lim >= k && row+2-k >= 0) ? a[row+2-k] : (long long)n   (:31012-31026; s12:s13 = sext(n) :30963)
//       h ^= c[k] * x           (:30991-31001)
//     stop after the term k == 1 + (tid >> 3)   (exit test v1 = k+1 - (tid>>3) == 2, :31002-31006), so tid 0-7 hash 2
//     terms (c[0], c[1]) and tid 8-15 hash 3.
// result = e[tid] + h % d[tid], signed 64-bit remainder with the sign of h (abs/negate sequence :31113-31206),
// stored at out[row*16 + tid] (s_lshl 7 = 128 bytes per row :31031, store :31212).
extern "C" __global__ void k_ngram_ids(const ll* a, const int* b, const ll* c, const ll* d, const ll* e, int n, ll* out) {
  int tid = threadIdx.x;
  if (tid >= 16) return;
  int row = blockIdx.x;
  int lim = row - b[row + 2] + 2;
  ull h = (ull)c[0] * (ull)a[row + 2];
  int last = 1 + (tid >> 3);
  for (int k = 1; k <= last; k++) {
    ll x = (lim >= k && row + 2 - k >= 0) ? a[row + 2 - k] : (ll)n;
    h ^= (ull)c[k] * (ull)x;
  }
  out[(long)row * 16 + tid] = e[tid] + (ll)h % d[tid];
}
