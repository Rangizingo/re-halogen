// Rewrite of the "route_hist" family (flash_serve, gfx1151): k_route_hist.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o route_hist.co route_hist.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). The ".long 0xd4.... / 0xd7...." lines are VOP3
// compares and 64-bit adds that llvm-objdump rejects: 0xd44a is v_cmp_eq_u32, 0xd44c v_cmp_gt_u32,
// 0xd454 v_cmp_gt_i64, 0xd4d4 v_cmpx_*_i64, 0xd7006a../0xd700.... v_add_co_u32, 0xd73c.... v_lshlrev_b64.
#include <hip/hip_runtime.h>

// halogen::(anonymous namespace)::k_route_hist(int const*, int, int*, int*)
// _ZN7halogen12_GLOBAL__N_112k_route_histEPKiiPiS3_, kernarg: ids@0, n@8, blk_hist@16, rank@24
// Launch: block 256 threads, each thread owns 4 consecutive ids; one block covers 1024 ids
// (idx = blockIdx.x*1024 + tid*4 + j, s_lshl_b64 s[4:5], s[2:3], 10 at :47526; v5 = tid*4 + that).
// LDS: ids[1024] ints at 0 (ds_store_b32 v12 = tid*16, :47558/:47573/:47589/:47606) and cnt[512] at 4096.
//   cnt is zeroed by a blockDim-strided loop over 512 slots (ds_store_b32 offset:4096, :47510).
//   id load (global_load_b32, only when idx < n as signed 64-bit: :47547/:47565/:47582/:47598);
//   an id outside [0,512) unsigned, or idx >= n, becomes -1 (v_cmp 0x200 test + v_cndmask -1: :47550...).
//   After a barrier (:47608) every id >= 0 does ds_add_u32 cnt[id] += 1 (:47612/:47618/:47624/:47630).
//   blk_hist[blockIdx.x*512 + e] = cnt[e] for e < 512, strided by blockDim (:47687-47694; s_lshl_b64 11 at :47676).
//   rank[idx] (only for idx < n) = number of earlier ids in this block with the same value, -1 counting
//   as a value: thread tid compares its 4 ids to every id of threads 0..tid-1 (8 ints per step, ds_load_b128
//   x2 and v_cmp_eq_u32 chains :47719-47720; an odd tid adds the 4 ids of thread tid-1 in a tail loop, :47865)
//   and item j also counts equal ids among its own items 0..j-1 (v_add_co_ci accumulations such as :47660 and :47880).
extern "C" __global__ void k_route_hist(const int* ids, int n, int* blk_hist, int* rank) {
  __shared__ int ids_s[1024];
  __shared__ int cnt[512];
  const int tid = threadIdx.x;
  const long base = (long)blockIdx.x * 1024 + (long)tid * 4;
  for (int i = tid; i < 512; i += blockDim.x) cnt[i] = 0;
  int e[4];
  for (int j = 0; j < 4; j++) {
    int v = -1;
    if (base + j < (long)n) {
      int x = ids[base + j];
      v = ((unsigned)x < 512u) ? x : -1;
    }
    e[j] = v;
    ids_s[tid * 4 + j] = v;
  }
  __syncthreads();
  for (int j = 0; j < 4; j++)
    if (e[j] >= 0) atomicAdd(&cnt[e[j]], 1);
  int r[4] = {0, 0, 0, 0};
  for (int p = 0; p < tid * 4; p++) {
    int q = ids_s[p];
    for (int j = 0; j < 4; j++) r[j] += (q == e[j]);
  }
  for (int j = 1; j < 4; j++)
    for (int k = 0; k < j; k++) r[j] += (e[k] == e[j]);
  __syncthreads();
  for (int i = tid; i < 512; i += blockDim.x) blk_hist[(long)blockIdx.x * 512 + i] = cnt[i];
  for (int j = 0; j < 4; j++)
    if (base + j < (long)n) rank[base + j] = r[j];
}
