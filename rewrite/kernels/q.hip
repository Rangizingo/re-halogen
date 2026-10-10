// Rewrite of the "q" family (k_q_softmax_f32) from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o q.co q.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

#pragma clang fp contract(off)

// k_q_softmax_f32(float const* x, float* out, int n, float temp)
// _ZN7halogen15k_q_softmax_f32EPKfPfif, kernarg: x@0, out@8, n@16, temp@20. One block of 256 threads (LDS stride constants 0x100).
// 1. m_t = max over i = t, t+256, ... < n of x[i], start -inf (codeobj_53000.s:2496, 2518-2531; v_max_f32 with canonicalizing self-max).
// 2. LDS max tree: stages 128, 64, 32, 16, 8, 4, 2, 1 with lds[t] = max(lds[t], lds[t+s]) (:2536-2649); m = lds[0] (copied to lds[257], :2659).
// 3. invt = 1.0f / temp, IEEE division (:2661-2680).
// 4. thread sum s_t = sum over its i of exp2(((x[i] - m) * invt) * 1.442695) in index order (v_sub, v_mul, v_mul 0x3fb8aa3b,
//    v_exp_f32, v_add; :2713-2719).
// 5. LDS add tree, same stages, lds[t] = lds[t+s] + lds[t] (:2731-2806); total = lds[0].
// 6. inv = 1.0f / total, IEEE division (:2827-2849); out[i] = inv * exp2(((x[i] - m) * invt) * 1.442695) (:2875-2882).
extern "C" __global__ void k_q_softmax_f32(const float* x, float* out, int n, float temp) {
  __shared__ float lds[256];
  int tid = threadIdx.x;
  float m = __int_as_float(0xff800000);
  for (int i = tid; i < n; i += 256) m = fmaxf(m, x[i]);
  lds[tid] = m;
  __syncthreads();
  for (int s = 128; s >= 1; s >>= 1) {
    if (tid < s) lds[tid] = fmaxf(lds[tid], lds[tid + s]);
    __syncthreads();
  }
  float mx = lds[0];
  __syncthreads();
  float invt = 1.0f / temp;
  float acc = 0.0f;
  for (int i = tid; i < n; i += 256) {
    float v = invt * (x[i] - mx);
    acc = acc + __builtin_amdgcn_exp2f(v * 1.44269502162933349609375f);
  }
  lds[tid] = acc;
  __syncthreads();
  for (int s = 128; s >= 1; s >>= 1) {
    if (tid < s) lds[tid] = lds[tid + s] + lds[tid];
    __syncthreads();
  }
  float total = lds[0];
  float inv = 1.0f / total;
  for (int i = tid; i < n; i += 256) {
    float v = invt * (x[i] - mx);
    out[i] = inv * __builtin_amdgcn_exp2f(v * 1.44269502162933349609375f);
  }
}
