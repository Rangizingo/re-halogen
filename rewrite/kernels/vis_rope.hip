// Rewrite of k_vis_rope (2D rotary on q|k rows of the packed qkv), codeobj_53000.s:64211-64352 (kernel at 0x26b900).
// _ZN7halogen12_GLOBAL__N_110k_vis_ropeEPfii, kernarg: qkv@0 (f32 rows of 3456), unused@8, mw@12
#include <hip/hip_runtime.h>
// The vendor keeps each separate mul and add separate; every fused op below is an explicit fmaf.
#pragma clang fp contract(off)

// grid.x = row s2, grid.y = head s3, 1D block; row stride 0x3600 bytes = 3456 f32 (:64286), head stride 0x120 bytes = 72 f32 (:64284).
// The int at kernarg 8 is not read by the kernel (only kernarg 0xc is loaded, :64213).
// Row split (:64214-64287): a = s2%2, b = (s2/2)%2, c = (s2/4)%mw, d = s2/(4*mw); iy = 2d + b (v4, :64287), ix = 2c + a (s1, :64282).
// Threads e = tid; e < 36 (:64224), e += blockDim (:64310, :64325):
//   j = (e < 18) ? 2e : 2e - 36           (v11 select with vcc = 18 > e, :64304-64311)
//   pos = (e < 18) ? iy : ix              (v12 select, :64311)
//   f = j / -36.0f (IEEE div, 0xc2100000, :64315-64326); f = f * 0x41135d8e (ln 1e4) (:64328); f = f * 0x3fb8aa3b (log2 e) (:64329)
//   inv = v_exp_f32(f) (:64330); ang = inv * float(pos) (:64331); ang = ang * 0x3e22f983 (1/2pi) (:64332)
//   s = v_sin_f32(ang), c = v_cos_f32(ang) (:64333-64334; input in revolutions)
//   q at base, k at base + 1152 f32 (+0x1000 +512 bytes :64302,64307); pairs (e, e+36) (+144 bytes :64306)
//   x1=q[e] x2=q[e+36] y1=k[e] y2=k[e+36]
//   q[e]    = fma(c, x1, -(s*x2))    (:64340)        q[e+36] = fma(s, x1, c*x2)  (:64341)
//   k[e]    = fma(c, y1, -(s*y2))    (:64339)        k[e+36] = fma(s, y1, c*y2)  (:64342)
extern "C" __global__ void k_vis_rope(float* qkv, int unused, int mw) {
  int s2 = blockIdx.x, s3 = blockIdx.y;
  int a = s2 % 2, b = (s2 / 2) % 2, c0 = (s2 / 4) % mw, d = s2 / (4 * mw);
  int ix = 2 * c0 + a, iy = 2 * d + b;
  float* q = qkv + (long)s2 * 3456 + (long)s3 * 72;
  float* k = q + 1152;
  for (int e = threadIdx.x; e < 36; e += blockDim.x) {
    bool lo = e < 18;
    int j = lo ? 2 * e : 2 * e - 36;
    float pos = (float)(lo ? iy : ix);
    float f = (float)j / -36.0f;
    f = f * __uint_as_float(0x41135d8eu);
    f = f * __uint_as_float(0x3fb8aa3bu);
    float inv = __builtin_amdgcn_exp2f(f);
    float ang = inv * pos;
    ang = ang * 0.15915494f;
    float s = __builtin_amdgcn_sinf(ang), c = __builtin_amdgcn_cosf(ang);
    float x1 = q[e], x2 = q[e + 36], y1 = k[e], y2 = k[e + 36];
    float sx2 = s * x2, sy2 = s * y2, cx2 = c * x2, cy2 = c * y2;
    k[e + 36] = fmaf(s, y1, cy2);
    q[e + 36] = fmaf(s, x1, cx2);
    k[e] = fmaf(c, y1, -sy2);
    q[e] = fmaf(c, x1, -sx2);
  }
}
