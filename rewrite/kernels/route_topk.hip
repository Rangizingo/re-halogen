// Rewrites of the "route_topk" family (flash_serve, gfx1151): k_route_topk, k_route_topk_w and k_route_topk_w_grp<2|4>.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o route_topk.co route_topk.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). ".long 0xd4.. / 0xd7.." lines are VOP3 compares and
// 64-bit adds/shifts that llvm-objdump rejects: 0xd414 v_cmp_gt_f32, 0xd41b v_cmp_ngt_f32, 0xd41d v_cmp_neq_f32,
// 0xd449 v_cmp_lt_u32, 0xd44c v_cmp_gt_u32, 0xd44d v_cmp_ne_u32, 0xd4ca v_cmpx_eq_u32, 0xd7006a../0xd700....
// v_add_co_u32, 0xd73c.... v_lshlrev_b64. The compare operands are read from the second dword that the
// listing shows as a bogus v_cndmask_b32_e32 line.
#include <hip/hip_runtime.h>

// softmax over the ten picked values, shared by the kernels below. exp(x) is v_exp_f32 of x * 0x3fb8aa3b
// (1.44269502); the sum runs e0, e1, ..., e9 in that order; each weight is an IEEE division (div_scale/rcp/
// fma/div_fmas/div_fixup).
__device__ static inline void softmax10(const float* v, float* w) {
  float e[10];
  for (int i = 0; i < 10; i++) e[i] = __builtin_amdgcn_exp2f((v[i] - v[0]) * 1.44269502162933349609375f);
  float s = e[0] + e[1];
  for (int i = 2; i < 10; i++) s = s + e[i];
  for (int i = 0; i < 10; i++) w[i] = e[i] / s;
}

// halogen::(anonymous namespace)::k_route_topk(float const*, int*, float*)
// _ZN7halogen12_GLOBAL__N_112k_route_topkEPKfPiPf, kernarg: x@0, out_idx@8, out_w@16
// One block (512 threads) per row, blockIdx.x = row (s2): x row = x + row*2048 bytes (s_lshl_b64 s[4:5], s[2:3], 11,
// codeobj_53000.s:37032), out offset row*10 (s_mul_i32 s14, s2, 10, :37033).
// LDS: idx[512] at 0, val[512] at 2048, work[512] at 4096, sel[10] at 6144 (6184 bytes).
//   work[tid] = x[row*512 + tid] (:37052, :37061).
//   Ten rounds (s9 from -40 step 4, :37059/:37067/:37069):
//     val[tid] = work[tid] (:37074-37076), idx[tid] = tid (:37077), barrier
//     compare-exchange tree, strides 256,128,64,32,16,8,4,2 (thread tid < stride, :37081-37208) then thread 0 against
//     element 1 (:37209-37222): if val[tid+stride] > val[tid] (strict, v_cmp_gt_f32 of the second dword's operands
//     v2, v1 at :37085/:37101/:37117/:37133/:37149/:37165/:37181/:37197, level 1 at :37214 with v1 = val[1],
//     v2 = val[0]) then idx[tid] = idx[tid+stride], val[tid] = val[tid+stride] (:37089-37092 etc.).
//     So an equal value never replaces the one at the lower tree position. The tree is not a plain
//     "lowest index wins" rule, so the same tree is kept here.
//     thread 0: out_idx[row*10 + round] = idx[0] (global_store_b32, :37236), sel[round] = val[0] (:37237)
//     barrier; the thread whose id equals idx[0] sets work[tid] = -inf (0xff800000, v5 :37034; cmpx_eq_u32 :37246, store :37249)
//   After ten rounds thread 0 loads sel[0..9] (:37255-37256, :37270) and writes ten weights:
//     e_i = v_exp_f32((sel[i] - sel[0]) * 0x3fb8aa3b) (:37262-37310), sum in order e0..e9 (:37276-37310),
//     w_i = e_i / sum (:37312-37448), stored at out_w + row*10*4 (global_store_b128 x2 + b64, :37450-37452).
extern "C" __global__ void k_route_topk(const float* x, int* out_idx, float* out_w) {
  __shared__ int idx_s[512];
  __shared__ float val_s[512];
  __shared__ float work[512];
  __shared__ float sel[16];
  const int tid = threadIdx.x;
  const long row = blockIdx.x;
  work[tid] = x[row * 512 + tid];
  __syncthreads();
  for (int it = 0; it < 10; it++) {
    val_s[tid] = work[tid];
    idx_s[tid] = tid;
    __syncthreads();
    for (int s = 256; s >= 2; s >>= 1) {
      if (tid < s) {
        const float b = val_s[tid + s];
        if (b > val_s[tid]) {
          idx_s[tid] = idx_s[tid + s];
          val_s[tid] = b;
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      const float b = val_s[1];
      if (b > val_s[0]) {
        idx_s[0] = idx_s[1];
        val_s[0] = b;
      }
    }
    __syncthreads();
    if (tid == 0) {
      out_idx[row * 10 + it] = idx_s[0];
      sel[it] = val_s[0];
    }
    __syncthreads();
    if (tid == idx_s[0]) work[tid] = -__builtin_inff();
    __syncthreads();
  }
  if (tid == 0) {
    float w[10];
    softmax10(sel, w);
    for (int i = 0; i < 10; i++) out_w[row * 10 + i] = w[i];
  }
}

// Order used by k_route_topk_w and k_route_topk_w_grp: larger value first; on equal values the smaller
// bit-reversed 9-bit index first (v_bfrev_b32 then v_lshrrev_b32 23).
// A candidate replaces the current best iff cand > cur, or cand == cur and key(cand) < key(cur)
// (codeobj_53000.s:37653-37678 for the first pair, :37686-37699 and the 14 repeats after it for the other 14
// elements, :38055-38074 for each of the five cross-lane steps). NaN never replaces and is never replaced.
__device__ static inline unsigned key9(int idx) { return __brev((unsigned)idx) >> 23; }
__device__ static inline bool better(float cv, int ci, float bv, int bi) {
  return (cv > bv) || (cv == bv && key9(ci) < key9(bi));
}

// halogen::(anonymous namespace)::k_route_topk_w(float const*, int*, float*, int)
// _ZN7halogen12_GLOBAL__N_114k_route_topk_wEPKfPiPfi, kernarg: x@0, out_idx@8, out_w@16, nrows@24
// 256 threads, one wave32 per row: row = blockIdx.x*8 + (tid >> 5) (v_lshl_or_b32 s2, 3, v0>>5, :37459-37461),
// guard nrows > row signed (v_cmpx at :37464). Lane l holds x[row*512 + l + 32*i], i = 0..15
// (sixteen global_load_b32, offsets 0,128,...,1920, :37495-37510).
//   Ten rounds (s3 from 0 to 10, :37594-37596). Round: each lane reduces its 16 values with `better` (:37653-38040),
//   then five xor-butterfly steps over lanes 16,8,4,2,1 with ds_bpermute_b32 of value and index and the same
//   `better` rule (:38049-38117), so every lane ends with the same best. The lane that owns the best element
//   (element index != best index test, v_cmp_ne_u32 0xd44d, :37591-37650) sets it to -inf (0xff800000).
//   best value goes to register v[round] through m0 (v_movreld_b32, :37595); lane 0 stores out_idx[row*10 + round] = best idx
//   (global_store_b32, :38120-38130).
//   Lane 0 then does the same softmax as k_route_topk: e_i = exp2((v_i - v_0)*1.4427), sum e0..e9 in order
//   (:38134-38182), IEEE divides (:38188-38311), stores out_w[row*10..] (:38313-38315).
// ids_lds, when not null, also receives each picked index at ids_lds[row*10 + round] (ds_store_b32 v65, v66,
// codeobj_53000.s:2268432 of the grp kernel, whose address v65 = 4*(row*10) + 4 per round).
__device__ static inline void topk_w_core(const float* x, int* out_idx, float* out_w, int nrows, int* ids_lds) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= nrows) return;
  float v[16];
  for (int i = 0; i < 16; i++) v[i] = x[(long)row * 512 + lane + 32 * i];
  float sel[10];
  for (int it = 0; it < 10; it++) {
    float bv = v[0];
    int bi = lane;
    for (int i = 1; i < 16; i++)
      if (better(v[i], lane + 32 * i, bv, bi)) {
        bv = v[i];
        bi = lane + 32 * i;
      }
    for (int m = 16; m >= 1; m >>= 1) {
      const float ov = __shfl_xor(bv, m, 32);
      const int oi = __shfl_xor(bi, m, 32);
      if (better(ov, oi, bv, bi)) {
        bv = ov;
        bi = oi;
      }
    }
    sel[it] = bv;
    if ((bi & 31) == lane) v[bi >> 5] = -__builtin_inff();
    if (lane == 0) {
      out_idx[(long)row * 10 + it] = bi;
      if (ids_lds) ids_lds[row * 10 + it] = bi;
    }
  }
  if (lane == 0) {
    float w[10];
    softmax10(sel, w);
    for (int i = 0; i < 10; i++) out_w[(long)row * 10 + i] = w[i];
  }
}
extern "C" __global__ void k_route_topk_w(const float* x, int* out_idx, float* out_w, int nrows) {
  topk_w_core(x, out_idx, out_w, nrows, nullptr);
}

// halogen::(anonymous namespace)::k_route_topk_w_grp<G>(float const*, int*, float*, int, HtItem*, HtItem*, int*, int*, int*, int*)
// _ZN7halogen12_GLOBAL__N_118k_route_topk_w_grpILi{G}EEEvPKfPiPfiPNS0_6HtItemES7_S4_S4_S4_S4_
// kernarg: x@0, out_idx@8, out_w@16, nrows@24, A@32 (HtItem*), B@40 (HtItem*), tok@48, rank@56, total@64, inv@72.
// HtItem is {int expert, tile, begin, end}, 16 bytes (analysis/closure/htitem-r3.md).
// One block of 256 threads; only the first 8 rows are processed (one wave per row), so the grid must be 1 block and
// nrows <= 8 (the grouping stage reads the LDS ids of all nrows*10 slots).
//   Stage 1 is k_route_topk_w plus ids[row*10 + round] in LDS (:2268432), barrier (:2268620).
//   Stage 2, thread t < n = nrows*10 (v_cmpx at :2268622; s34 = n at :2268615), e = ids[t]:
//     pos  = #{j < n : ids[j] < e (signed) or (ids[j] == e and j < t)}  (stable sort position, :2268647-2268745, tail :2268759-2268792)
//     cnt  = #{j < n : ids[j] == e}, first = no j < t with ids[j] == e
//     rank[t] = pos (global_store_b32 v5, v3, s[26:27], :2268806); tok[pos] = (t * 6554) >> 16 = t / 10
//     (v_mul_u32_u24 0x199a, v_mov_b16 pair, :2268794-2268801, store :2268807); inv[pos] = t when the inv pointer
//     is not null (s_cmp_lg_u64 s[30:31], 0 at :2268795; store :2268813).
//   Threads t < 128 store items[t] = first ? ceil(cnt / G) : 0 into LDS at 512 + 4t (:2268820-2268827).
//   Barrier (:2268831). A thread with first set (s1) and items > 0 (ds_load v85, :2268974-2268977):
//     before = sum of items[j] over j < n with ids[j] < e (:2268842-2268973)
//     item k = 0 .. items-1: begin = pos + G*k, end = pos + cnt; A[5*(before + k) + r] = {e, r, begin, end} for r = 0..4
//     (stores via v[94:95], :2269113-2269198) and B[20*(before + k) + r] = {e, r, begin, end} for r = 0..19 (v[96:97]);
//     the 20 * v7 and 5 * v7 bases are at :2268983-2268985, the begin step at :2269200.
//   Thread 0 stores total = sum of items[j] over j < n at s[28:29] (:2269210-2269267).
struct HtItem {
  int expert, tile, begin, end;
};
template <int G>
__device__ static inline void grp_body(const float* x, int* out_idx, float* out_w, int nrows, HtItem* A, HtItem* B,
                                       int* tok, int* rank, int* total, int* inv) {
  __shared__ int ids_s[128];
  __shared__ int items_s[128];
  const int t = threadIdx.x;
  topk_w_core(x, out_idx, out_w, nrows, ids_s);
  __syncthreads();
  const int n = nrows * 10;
  int e = 0, pos = 0, cnt = 0;
  bool first = false;
  if (t < n) {
    e = ids_s[t];
    first = true;
    for (int j = 0; j < n; j++) {
      const int q = ids_s[j];
      if (q < e || (q == e && j < t)) pos++;
      if (q == e) cnt++;
      if (q == e && j < t) first = false;
    }
    rank[t] = pos;
    tok[pos] = (int)(((unsigned)t * 6554u) >> 16);
    if (inv) inv[pos] = t;
  }
  if (t < 128) items_s[t] = (t < n && first) ? (cnt + G - 1) / G : 0;
  __syncthreads();
  if (t < n && first && items_s[t] > 0) {
    int before = 0;
    for (int j = 0; j < n; j++)
      if (ids_s[j] < e) before += items_s[j];
    const int nit = items_s[t];
    for (int k = 0; k < nit; k++) {
      const int b = pos + G * k, en = pos + cnt;
      for (int r = 0; r < 5; r++) A[5 * (before + k) + r] = HtItem{e, r, b, en};
      for (int r = 0; r < 20; r++) B[20 * (before + k) + r] = HtItem{e, r, b, en};
    }
  }
  if (t == 0) {
    int s = 0;
    for (int j = 0; j < n; j++) s += items_s[j];
    *total = s;
  }
}
extern "C" __global__ void k_route_topk_w_grp2(const float* x, int* out_idx, float* out_w, int nrows, HtItem* A,
                                               HtItem* B, int* tok, int* rank, int* total, int* inv) {
  grp_body<2>(x, out_idx, out_w, nrows, A, B, tok, rank, total, inv);
}
extern "C" __global__ void k_route_topk_w_grp4(const float* x, int* out_idx, float* out_w, int nrows, HtItem* A,
                                               HtItem* B, int* tok, int* rank, int* total, int* inv) {
  grp_body<4>(x, out_idx, out_w, nrows, A, B, tok, rank, total, inv);
}
