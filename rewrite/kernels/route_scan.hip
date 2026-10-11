// Rewrite of the "route_scan" family (flash_serve, gfx1151): k_route_scan.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o route_scan.co route_scan.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). ".long 0xd4.. / 0xd7.." pairs are VOP3 compares
// and 64-bit adds that llvm-objdump rejects (0xd441 v_cmp_lt_i32, 0xd449 v_cmp_lt_u32, 0xd44d v_cmp_ne_u32,
// 0xd7006a../0xd700.... v_add_co_u32, 0xd73c.... v_lshlrev_b64).
#include <hip/hip_runtime.h>

// halogen::(anonymous namespace)::k_route_scan(int*, int, int*, int*, int*)
// _ZN7halogen12_GLOBAL__N_112k_route_scanEPiiS1_S1_S1_, kernarg: a@0, nb@8, act@16, off@24, tot@32
// One block of 512 threads, thread e owns expert e. a is [nb][512] block-by-expert counts (from k_route_hist), in place.
//   total = sum over b < nb of a[b*512+e]            (8-way unrolled global loads + v_add3_u32, codeobj_53000.s:47949-47992)
//   nz    = 0 < total  (signed)                        (v_cmp_lt_i32 0xd441, :48032)
//   LDS: nz[512] at 0 and total[512] at 2048 (ds_store_2addr_stride64_b32 offset1:8, :48040)
//   inclusive Hillis-Steele scan over both arrays, steps d = 1,2,4,...,256, each step: masked loads of [tid-d]
//   (zero when tid < d, masks from v_cmp_lt_u32 0xd449 such as :48057), barrier, load own, add, store, barrier (:48042-48258)
//   excl = incl_total - total ; r = incl_nz - nz        (v_sub_nc_u32 v6 = incl_total - total at :48264, r = incl_nz - nz at :48278)
//   tot[e] = total for every e (global_store_b32 v1, v2, s[10:11], :48268 and :48294)
//   if nz: act[r] = e, off[r] = excl                    (global_store_b32 :48291-48292)
//   thread 0: n_act = nz_incl[511] (ds_load offset:2044, :48262), grand = total_incl[511] (offset:4092, :48303)
//             off[n_act] = grand (:48308), tot[512] = n_act, tot[513] = grand (global_store_b64 offset:2048, :48309)
//   then a[b*512+e] is replaced by the running exclusive offset: x = excl; for b: t = a[..]; a[..] = x; x += t
//   (unrolled by 8 with a scalar tail, :48357-48413; same result as the plain loop)
extern "C" __global__ void k_route_scan(int* a, int nb, int* act, int* off, int* tot) {
  __shared__ int nz_s[512];
  __shared__ int tot_s[512];
  const int tid = threadIdx.x;
  int total = 0;
  for (int b = 0; b < nb; b++) total += a[(long)b * 512 + tid];
  const int nz = (0 < total) ? 1 : 0;
  nz_s[tid] = nz;
  tot_s[tid] = total;
  __syncthreads();
  for (int d = 1; d < 512; d <<= 1) {
    int pn = 0, pt = 0;
    if (tid >= d) {
      pt = tot_s[tid - d];
      pn = nz_s[tid - d];
    }
    __syncthreads();
    int ct = tot_s[tid], cn = nz_s[tid];
    __syncthreads();
    tot_s[tid] = ct + pt;
    nz_s[tid] = cn + pn;
    __syncthreads();
  }
  const int incl_t = tot_s[tid];
  const int incl_n = nz_s[tid];
  const int excl = incl_t - total;
  tot[tid] = total;
  if (nz) {
    const int r = incl_n - nz;
    act[r] = tid;
    off[r] = excl;
  }
  if (tid == 0) {
    const int n_act = nz_s[511];
    const int grand = tot_s[511];
    off[n_act] = grand;
    tot[512] = n_act;
    tot[513] = grand;
  }
  int x = excl;
  for (int b = 0; b < nb; b++) {
    const int t = a[(long)b * 512 + tid];
    a[(long)b * 512 + tid] = x;
    x += t;
  }
}
