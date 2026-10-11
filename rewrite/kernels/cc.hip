// Rewrites of the "cc" family from the vendor code object codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o cc.co cc.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). The listing shows some VOP3 instructions as a
// ".long <dword1>" line plus a mis-decoded second dword; they are decoded by hand below
// (v_add_co_u32 = D700, v_lshlrev_b64 = D73C, v_cmpx_gt_i64 = D4D4).
#include <hip/hip_runtime.h>

// Keep float multiply and add separate unless an explicit __builtin_fmaf is written (hipcc fuses a*b+c by default).
#pragma clang fp contract(off)

// halogen::(anonymous namespace)::k_cc_sub(float*, float const*, float const*, long)
// _ZN7halogen12_GLOBAL__N_18k_cc_subEPfPKfS3_l, kernarg: y@0, a@8, b@16, n@24
// i = blockIdx.x*blockDim.x + threadIdx.x as a 64-bit product (v_mad_u64_u32, codeobj_53000.s:63100),
// guard i < n (v_cmpx_gt_i64 exec, n, i :63102; s_cbranch_execz :63104). Grid-stride loop: the step is
// hidden block_count_x * blockDim.x (s_load_b32 at kernarg+32 :63095/:63105; s_mul_hi/s_mul_i32 :63110-63111),
// the loop exits on i >= n (v_cmp_le_i64 :63135, s_cbranch_execnz :63144).
//   y[i] = a[i] - b[i]  (two global_load_b32 :63125-63126, v_sub_f32 :63141, global_store to arg0 :63142)
extern "C" __global__ void k_cc_sub(float* y, const float* a, const float* b, long n) {
  long stride = (long)gridDim.x * blockDim.x;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    y[i] = a[i] - b[i];
}

// halogen::(anonymous namespace)::k_cc_affine(float*, float const*, float const*, float const*)
// _ZN7halogen12_GLOBAL__N_111k_cc_affineEPfPKfS3_S3_, kernarg: out@0, A@8, B@16, C@24; LDS 9280 B.
// Block (16,16): tx = v0 & 0x3ff (:62699), ty = (v0 >> 10) & 0x3ff (:62697). Grid (batch, 8).
// Batch bx owns 128x128 f32 matrices: byte offset bx<<16 (s_lshl_b64 s[2:3],s[2:3],16 :62706).
// Row r = blockIdx.y*16 + ty (v_lshl_add_u32 v0 = s4*16 + v12 :62702; s4 = blockIdx.y :62698).
// Each thread owns 8 columns tx + 16*j. Main loop k0 = 0,16,..,112 (s_cmpk_gt_u32 s2, 0x6f :62748):
//   As[ty][tx] = A[r][k0+tx]            global :62749, ds_store_b32 v16 = 0x2000 + ty*0x44 + tx*4 :62767 (row pitch 17 floats)
//   Bs[ty][tx+16j] = B[k0+ty][tx+16j]   8 loads at +64 B :62758-62765, ds_store_2addr v17 = ty*512 + tx*4 :62769-62775
//   barrier :62777, then acc[j] += As[ty][k] * Bs[k][tx+16j] for k = 0..15 ascending, v_fmac_f32 (FMA),
//   barrier :62976 before the next tile.
// Epilogue (:62979-63145): out[bx][r*128 + tx + 16j] = acc[j] + (C ? C[same index] : 0.0f).
//   The vendor tests C != null with s_cmp_lg_u64 s[14:15], 0 (:62981); when null it adds a zero register
//   (v_mov 0 :62982, then v_add_f32), so -0.0 becomes +0.0. The rewrite adds 0.0f too.
extern "C" __global__ void k_cc_affine(float* out, const float* A, const float* B, const float* C) {
  __shared__ float Bs[16][128];
  __shared__ float As[16][17];
  int tx = threadIdx.x, ty = threadIdx.y;
  long boff = (long)blockIdx.x * 16384;
  int r = blockIdx.y * 16 + ty;
  const float* Ab = A + boff;
  const float* Bb = B + boff;
  float acc[8];
  for (int j = 0; j < 8; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < 128; k0 += 16) {
    As[ty][tx] = Ab[r * 128 + k0 + tx];
    for (int j = 0; j < 8; j++) Bs[ty][tx + 16 * j] = Bb[(k0 + ty) * 128 + tx + 16 * j];
    __syncthreads();
    for (int k = 0; k < 16; k++) {
      float a = As[ty][k];
      for (int j = 0; j < 8; j++) acc[j] = __builtin_fmaf(a, Bs[k][tx + 16 * j], acc[j]);
    }
    __syncthreads();
  }
  for (int j = 0; j < 8; j++) {
    long idx = boff + r * 128 + tx + 16 * j;
    float c = C ? C[idx] : 0.0f;
    out[idx] = acc[j] + c;
  }
}
