// Rewrites of q4moe::k_q4moe_items<64|128> (work-list builder), vendor code object 53000, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o q4moe_items.co q4moe_items.hip
// Listing: kernels-disasm/codeobj_53000.s (untracked; regenerate with llvm-objdump --mattr=+gfx1151). "L:n" below means that file, line n.
// `.long` words in the listing are 64-bit-operand or VOPC-E ops llvm-objdump could not print; they are read from the raw
// encoding (opcode field) and the compare run is the check.
#include <hip/hip_runtime.h>

struct Item { int expert, k, start, nrows; };

// q4moe::k_q4moe_items<TILE>(int const* ids, int const* offs, int const* cfg, int n, Item* items5, Item* items10, int* counts)
// kernarg ids@0 offs@8 cfg@16 n@24 items5@32 items10@40 counts@48; one block of 512 threads (L:1995227 v_cmp_eq 0x1ff).
// cfg[512] = number of experts E (s_load_b32 s5, s[2:3], 0x800; L:1994939); thread 0 traps (s_trap 2, L:1995239) when
// cfg[513] != n (L:1994942-1994947 to the s_trap at 0xe008e8). Thread t < E loads ids[t] and offs[t], offs[t+1]
// (L:1994967-1994968), cnt = offs[t+1] - offs[t] (L:1994970), tiles = ceil(cnt / TILE) with the signed rounding fix
// (L:1994973-1994979), then an inclusive scan over 512 LDS ints (Hillis-Steele, strides 1..256, L:1994980-1995123).
// If cnt > 0 (v_cmpx_lt_i32 0, v41, L:1995127): first = scan[t] - tiles; for j in 0..tiles-1 the tile index is first + j and
// start_j = offs[t] + TILE*j, rows_j = min(TILE, cnt - TILE*j) (L:1995147, 1995172, 1995189, 1995200).
// Each tile writes 5 records at items5 + 80*tile and 10 records at items10 + 160*tile (v_mad_i64_i32 0x50 / 0xa0,
// L:1995171-1995173), record k = {ids[t], k, start_j, rows_j} (k from the s_mov 1..9 constants and the 4-dword stores,
// L:1995136-1995170, 1995198-1995223). Thread 511 stores counts[0] = total_tiles * 5 and counts[1] = total_tiles * 10
// (v_mul_lo_u32 by inline 5 and 10, L:1995233-1995237; total from LDS offset 2044).
template <int TILE>
__device__ static inline void items_body(const int* ids, const int* offs, const int* cfg, int n, Item* items5, Item* items10,
                                         int* counts) {
  __shared__ int scan[512];
  int t = threadIdx.x;
  int E = cfg[512];
  if (t == 0 && cfg[513] != n) __builtin_trap();
  int e = 0, base = 0, cnt = 0;
  if (t < E) {
    e = ids[t];
    base = offs[t];
    cnt = offs[t + 1] - base;
  }
  int tiles = (cnt + (TILE - 1)) / TILE;
  scan[t] = tiles;
  __syncthreads();
  for (int d = 1; d < 512; d <<= 1) {
    int add = (t >= d) ? scan[t - d] : 0;
    __syncthreads();
    scan[t] += add;
    __syncthreads();
  }
  int first = scan[t] - tiles;
  if (cnt > 0) {
    int rem = cnt, start = base;
    for (int j = 0; j < tiles; j++) {
      int rows = rem < TILE ? rem : TILE;
      int tile = first + j;
      for (int k = 0; k < 10; k++) {
        Item it = {e, k, start, rows};
        if (k < 5) items5[tile * 5 + k] = it;
        items10[tile * 10 + k] = it;
      }
      rem -= TILE;
      start += TILE;
    }
  }
  __syncthreads();
  if (t == 511) {
    int total = scan[511];
    counts[0] = total * 5;
    counts[1] = total * 10;
  }
}
extern "C" __global__ void k_q4moe_items_64(const int* ids, const int* offs, const int* cfg, int n, Item* items5, Item* items10,
                                            int* counts) {
  items_body<64>(ids, offs, cfg, n, items5, items10, counts);
}
extern "C" __global__ void k_q4moe_items_128(const int* ids, const int* offs, const int* cfg, int n, Item* items5, Item* items10,
                                             int* counts) {
  items_body<128>(ids, offs, cfg, n, items5, items10, counts);
}
