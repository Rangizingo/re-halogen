// Rewrites of the "kv" family (flash_serve, gfx1151): k_kv_scatter_rows (codeobj_53000), k_kv_bt (codeobj_1513000).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o kv.co kv.hip
// Listings: kernels-disasm/codeobj_53000.s and codeobj_1513000.s (not in git). Pure data movement, no arithmetic.
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)

// FdRows, the 160-byte by-value struct at kernarg +0x10 (k_kv_scatter_rows reads it at +0x10+8*b, +0x50+8*b,
// +0x90+4*b, codeobj_53000.s:61930-61943). Layout inferred from those three addressing forms:
//   8 pointers @0, 8 pointers @64, 8 ints @128.
struct FdRows {
  unsigned short* p0[8];
  unsigned short* p1[8];
  int slot[8];
};

// k_kv_scatter_rows(const bf16* K@0, const bf16* V@8, FdRows@16)
//   _ZN7halogen12_GLOBAL__N_117k_kv_scatter_rowsEPKtS2_NS0_6FdRowsE. Grid x = row b (<8), 1D block, 512 elements per row.
//   src row = K/V + b*1024 bytes (s_lshl_b64 s[2:3],10, :61954); dst row = p0[b]/p1[b] + slot[b]*1024 bytes (:61948-61953).
//   for (i = tid; i < 512; i += blockDim.x): p0[b]-row[i] = K-row[i] ; p1[b]-row[i] = V-row[i]
//   (loop exit at i > 0x1ff, :61978; K load d16_b16 :61969, V load d16_hi_b16 :61970; stores :61984-61986).
extern "C" __global__ void k_kv_scatter_rows(const unsigned short* K, const unsigned short* V, FdRows r) {
  int b = blockIdx.x;
  const unsigned short* ks = K + (long)b * 512;
  const unsigned short* vs = V + (long)b * 512;
  unsigned short* kd = r.p0[b] + (long)r.slot[b] * 512;
  unsigned short* vd = r.p1[b] + (long)r.slot[b] * 512;
  for (int i = threadIdx.x; i < 512; i += blockDim.x) {
    kd[i] = ks[i];
    vd[i] = vs[i];
  }
}

// k_kv_bt(const bf16* a@0, const bf16* b@8, long ld@16, int n@24, bf16* c@32, bf16* d@40)
//   _ZN7halogen12_GLOBAL__N_17k_kv_btEPKtS2_liPtS3_. Block of 256 threads, grid (ceil(n/4), ncols/256).
//   Tile (bx, by) = rows 4*bx..4*bx+3 (the thread's row is 4*bx + tid/64, :8283-8288), columns by*256..+255
//   (s14<<8, :8295). Tile base in c and d: bx*4096 + by*2048 bytes (:8332-8337).
//   c: thread t copies 4 consecutive bf16 of a (one global_load_b64, :8327) at row (4bx+t/64), column by*256 + 4*(t&63);
//      row >= n gives zeros (v4:v5 = 0, :8298-8300, still stored). c tile layout (confirmed with an iota run on Ai-core):
//      c[tile + (colchunk=((t&63)>>2))*64 + row*16 + 4*(t&3) + j] (16-column chunks of 4 rows x 16 columns; :8330-8335 uses
//      (t<<5)&0x780 bytes = chunk*128 B).
//   d: thread t takes column by*256 + t of b for rows 4bx..4bx+3 (four global_load_u16 packed pairwise with
//      v_lshl_or_b32 v,16, :8386-8388, :8411-8428), rows >= n give 0 (the four guarded loads, :8349-8374, :8400-8447),
//      and stores one b64: d[tile + 4*t + r] (transposed to [column][row]).
extern "C" __global__ void k_kv_bt(const unsigned short* a, const unsigned short* b, long ld, int n,
                                   unsigned short* c, unsigned short* d) {
  int t = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;
  long tile = (long)bx * 2048 + (long)by * 1024;  // elements
  int row = 4 * bx + (t >> 6);
  unsigned short v[4] = {0, 0, 0, 0};
  if (row < n) {
    const unsigned short* p = a + (long)row * ld + (long)by * 256 + 4 * (t & 63);
    for (int j = 0; j < 4; j++) v[j] = p[j];
  }
  unsigned short* cp = c + tile + ((t & 63) >> 2) * 64 + (t >> 6) * 16 + 4 * (t & 3);
  for (int j = 0; j < 4; j++) cp[j] = v[j];
  unsigned short w[4] = {0, 0, 0, 0};
  for (int r = 0; r < 4; r++)
    if (4 * bx + r < n) w[r] = b[(long)(4 * bx + r) * ld + (long)by * 256 + t];
  unsigned short* dp = d + tile + 4 * t;
  for (int r = 0; r < 4; r++) dp[r] = w[r];
}
