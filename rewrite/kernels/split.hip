// Rewrites of the "split" family (row splitting copies) from flash_serve, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o split.co split.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). All moves are bit-exact 16-bit copies.
#include <hip/hip_runtime.h>

// halogen::(anonymous namespace)::k_split_qgate(unsigned short const*, unsigned short*, unsigned short*)
// _ZN7halogen12_GLOBAL__N_113k_split_qgateEPKtPtS3_, kernarg: x@0, q@8, gate@16.
// Row r = blockIdx.x. Input row = 0x6000 bytes (12288 bf16) at x + r*0x6000 (:52194-52198);
// both output rows = 0x3000 bytes (6144 bf16) at +r*0x3000 (:52200, :52202).
// Loop i = tid; i < 0x1800 (6144); i += blockDim (:52208, :52327; :52285-52338 is the scalar tail loop).
// Element index in the input row: ((i*2) & 0x3e00)*2 bytes + (i & 0xff)*2 bytes = (i >> 8)*512 + (i & 0xff) elements
// (:52225-52232, :52303-52308); q takes that element (:52242), gate takes the one 512 bytes = 256 elements later (:52243).
// So each of 24 heads holds [q 256 | gate 256] in the input.
extern "C" __global__ void k_split_qgate(const unsigned short* x, unsigned short* q, unsigned short* gate) {
  const unsigned short* xr = x + (size_t)blockIdx.x * 12288;
  unsigned short* qr = q + (size_t)blockIdx.x * 6144;
  unsigned short* gr = gate + (size_t)blockIdx.x * 6144;
  for (unsigned i = threadIdx.x; i < 6144u; i += blockDim.x) {
    unsigned e = (i >> 8) * 512u + (i & 0xffu);
    qr[i] = xr[e];
    gr[i] = xr[e + 256];
  }
}

// halogen::(anonymous namespace)::k_split_ixqk(unsigned short const*, unsigned short*, unsigned short*, int)
// _ZN7halogen12_GLOBAL__N_112k_split_ixqkEPKtPtS3_i, kernarg: x@0, a@8, b@16, n@24.
// Grid-stride loop over idx < n*0x280 = n*640 (:55640, :55643, :55664; stride = gridDim.x*blockDim.x :55655-55656, :55660-55663).
// r = idx / 640, c = idx % 640: v_mad_u64_u32 by 0xcccccccd/0xcccccccc is mulhi(idx, 0xCCCCCCCCCCCCCCCD) >> 9 (:55678-55690),
// remainder via multiply-add of -640 = 0xfffffd80 (:55693-55696). Compare c > 0x1ff (:55701).
// c >= 512: store to b (arg 2); c < 512: store to a (arg 1) (:55705-55730). Source is x[idx] (:55676).
// Destination layout: a[r*512 + c], b[r*128 + c - 512] (:55716 stores with offset -1024 bytes = -512 elements; confirmed by the harness MATCH, split_ixqk.json).
extern "C" __global__ void k_split_ixqk(const unsigned short* x, unsigned short* a, unsigned short* b, int n) {
  long total = (long)n * 640;
  long stride = (long)gridDim.x * blockDim.x;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    long r = idx / 640;
    int c = (int)(idx - r * 640);
    unsigned short v = x[idx];
    if (c >= 512) b[r * 128 + (c - 512)] = v;
    else a[r * 512 + c] = v;
  }
}
