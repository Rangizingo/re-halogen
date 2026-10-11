// Rewrites of the "ht_items" family from the vendor code object codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o ht_items.co ht_items.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). Semantics: analysis/closure/htitem-r3.md.
// Cites below are codeobj_53000.s lines. k_ht_items_slots starts at :22504.
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)

struct HtItem { int expert, tile, begin, end; };

// halogen::(anonymous namespace)::k_ht_items_slots(int const*, int, HtItem*, HtItem*, int*, int*)
// kernarg: ids@0, n@8, A@16, B@24, tok@32, rank@40.  1D grid, i = global thread id.
//   i < 20n : B[i] = {ids[i/20], i - 20*(i/20), i/20, i/20 + 1}   (:22504+37..+42, store b128 to s[6:7])
//   i < 5n  : A[i] = {ids[i/5],  i - 5*(i/5),  i/5,  i/5 + 1}    (mad -5, :22504+60..)
//   i < n   : tok[i] = i/10 (0x66666667 mulhi, ashr 2), rank[i] = i  (:22504+96..+97)
extern "C" __global__ void k_ht_items_slots(const int* ids, int n, HtItem* A, HtItem* B, int* tok, int* rank) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 20 * n) {
    int j = i / 20;
    B[i] = HtItem{ids[j], i - 20 * j, j, j + 1};
  }
  if (i < 5 * n) {
    int j = i / 5;
    A[i] = HtItem{ids[j], i - 5 * j, j, j + 1};
  }
  if (i < n) {
    tok[i] = i / 10;
    rank[i] = i;
  }
}

// k_ht_items_group<G>(int const* ids, int n, HtItem* A, HtItem* B, int* tok, int* rank, int* total)
// kernarg: ids@0, n@8, A@16, B@24, tok@32, rank@40, total@48. One workgroup of 128 threads, thread t = slot t.
//   rank[t]   = stable-sort position of slot t by expert id           (cmp_eq/cmp_lt ladder)
//   tok[rank] = t / 10                                                (0x199a multiply)
//   per expert (first slot only): items = ceil(count / G); v6 = items of lower expert ids
//   A[5*(v6+k) + r] = {expert, r, begin + G*k, begin + count}, r < 5; B[20*(v6+k) + r], r < 20
//   total = sum of items (thread 0)
template <int G>
__device__ static inline void items_group(const int* ids, int n, HtItem* A, HtItem* B, int* tok, int* rank, int* total) {
  __shared__ int s_id[128];
  __shared__ int s_items[128];
  int t = threadIdx.x;
  int id = 0, lt = 0, cnt = 0, before = 0;
  if (t < n) {
    id = ids[t];
    s_id[t] = id;
  }
  __syncthreads();
  bool first = false;
  if (t < n) {
    for (int j = 0; j < n; j++) {
      int o = s_id[j];
      lt += o < id;
      cnt += o == id;
      before += (o == id) & (j < t);
    }
    first = before == 0;
    rank[t] = lt + before;
    tok[lt + before] = t / 10;
  }
  s_items[t] = (t < n && first) ? (cnt + G - 1) / G : 0;
  __syncthreads();
  if (t < n && first) {
    int v6 = 0;
    for (int j = 0; j < n; j++)
      if (s_id[j] < id) v6 += s_items[j];
    int items = s_items[t];
    for (int k = 0; k < items; k++) {
      for (int r = 0; r < 5; r++) A[5 * (v6 + k) + r] = HtItem{id, r, lt + G * k, lt + cnt};
      for (int r = 0; r < 20; r++) B[20 * (v6 + k) + r] = HtItem{id, r, lt + G * k, lt + cnt};
    }
  }
  if (t == 0) {
    int s = 0;
    for (int j = 0; j < n; j++) s += s_items[j];
    *total = s;
  }
}

extern "C" __global__ void k_ht_items_group2(const int* ids, int n, HtItem* A, HtItem* B, int* tok, int* rank, int* total) {
  items_group<2>(ids, n, A, B, tok, rank, total);
}
extern "C" __global__ void k_ht_items_group4(const int* ids, int n, HtItem* A, HtItem* B, int* tok, int* rank, int* total) {
  items_group<4>(ids, n, A, B, tok, rank, total);
}
