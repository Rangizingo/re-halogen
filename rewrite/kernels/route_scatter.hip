// Rewrite of the "route_scatter" family (flash_serve, gfx1151): k_route_scatter.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o route_scatter.co route_scatter.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). ".long 0xd4.. / 0xd7.." pairs are VOP3 compares and
// 64-bit adds/shifts that llvm-objdump rejects (0xd4d4 v_cmpx_*_i64, 0xd7006a../0xd700.... v_add_co_u32,
// 0xd73c.... v_lshlrev_b64).
#include <hip/hip_runtime.h>

// halogen::(anonymous namespace)::k_route_scatter(int const*, int, int const*, int const*, int*, int*)
// _ZN7halogen12_GLOBAL__N_115k_route_scatterEPKiiS2_S2_PiS3_, kernarg: ids@0, n@8, boff@16, rank@24, perm@32, slot@40
// Same indexing as k_route_hist: idx = blockIdx.x*1024 + tid*4 + j, j = 0..3 handled in order
// (s_lshl_b64 s[14:15], s[2:3], 10 at codeobj_53000.s:48434; v_lshl_or_b32 v0 = tid*4 | that, :48436).
// Guard idx < n as signed 64-bit (v_cmpx_*_i64 at :48443; the j = 3 item uses v_cmp_gt_i64 s[12:13], :48623).
//   e = ids[idx] (global_load_b32 :48453); skipped unless e < 0x200 unsigned (v_cmpx_gt_u32, :48456).
//   pos = rank[idx] (s[6:7] = kernarg+24) + boff[blockIdx.x*512 + e] (s[2:3] = boff + (block << 11), :48435/:48440;
//   loads :48469-48470) as a 32-bit add (v_add_nc_u32, :48472), then sign-extended for the address (v_ashrrev_i32, :48474).
//   perm[pos] = idx (global_store_b32 v0, :48485) and slot[idx] = pos (:48486).
extern "C" __global__ void k_route_scatter(const int* ids, int n, const int* boff, const int* rank, int* perm,
                                           int* slot) {
  const long base = (long)blockIdx.x * 1024 + (long)threadIdx.x * 4;
  for (int j = 0; j < 4; j++) {
    const long idx = base + j;
    if (idx >= (long)n) continue;
    const int e = ids[idx];
    if ((unsigned)e >= 512u) continue;
    const int pos = rank[idx] + boff[(long)blockIdx.x * 512 + e];
    perm[pos] = (int)idx;
    slot[idx] = pos;
  }
}
