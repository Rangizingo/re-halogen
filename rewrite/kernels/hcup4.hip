// Rewrites of k_hcup4<A, B> (A = 1..4, B = 0..2) from flash_serve, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o hcup4.co hcup4.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). Line cites below are in that file.
//
// halogen::(anonymous namespace)::k_hcup4<A,B>(unsigned char const*, long, long, long, unsigned short const*,
//                                              unsigned short const*, unsigned short*, float*)
// kernarg: W@0, a@8, b@16, c@24, x@32, wm@40, y@48, f@56. Block = 256 threads; grid = number of 16-row groups.
//
// Thread map (<1,1> :133670-133672, <1,2> :133957-133962): lane = tid & 15, g = tid >> 4 (16 row groups per block),
// row n = blockIdx.x*16 + g (v_or3 of (g & 14), s2<<4, g & 1: :133692-133693 in <1,1> epilogue, :133966 in <1,2>).
// Only lanes 0..9 of a group load (v_cmpx_gt_u32 10: :133698, :133971): 10 lanes * 32 weights = a 320-wide dot per plane.
// B selects the weight decode, A the number of epilogue sets:
//   B = 2: 16 bytes of codes at W + a + n*160 + k*0x64000 + lane*16 (plane k = 0..3: v_mad 0xa0 :134336, planes +0x64000/+0xc8000/+0x12c000
//          :134347-134356) are XORed as four dwords and converted with v_cvt_f32_u32 (:134033-134048); times the f16 scale
//          at W + b + 2*lane + c*(n + 2560*k) (v_fma_mix with op_sel_hi 1,0,0: :134051-134057; scale address v_mad c*n :134360, plane
//          step c*2560 :134363-134365). Activations x are not read.
//   B = 0: the same four plane/scale addressing, but each code byte indexes a 256-entry LDS table (below) and is accumulated with
//          v_dot2_f32_bf16 against x[lane*32 .. +32) (64 bytes at x + lane*64: :133707-133710, :134331-134333, :134362), bytes in order 0..15
//          (:134508-134665), then times the f16 scale (:134667-134673).
//   B = 1: one shared dot (<1,1> :133704-133805): codes at W + a + lane*16 (no row or plane term, :133700-133704), scale at W + b + 2*lane
//          (:133711-133715); the one result is used for all four planes (the epilogue divides four identical values: :133901-133909).
// LDS table (B = 0, 1; :133676-133694, :134291-134314): entry[tid] = (bf16_rne(cb[tid >> 4]) << 16) | bf16_rne(cb[tid & 15]) where cb is the
// 16 f32 at W (global_load_b32 W + 4*(tid & 15) and W + 4*(tid >> 4)). A code byte v selects entry[v] (v_and 0xff << 2, (>>6)&0x3fc, ...).
// Reduction over the 16 lanes of a group: xor-butterfly 8, 4, 2, 1 with v_add_f32 (:133811-133842, :134063-134123).
// Epilogue on lane 0 of each group (:133843+ ): per plane k, r_k = bf16_rne(sum_k) widened (bfe/add3/and 0xffff0000: :133847-133850),
//   s_k = bf16_rne(1 / (1 + exp2(r_k * -1.442695))) widened (:133851-133925, IEEE division),
//   set 0: m = fma(s0,w[0*2560+n],0); fmac(s1,w[1*2560+n]); fmac(s2,w[2*2560+n]); fmac(s3,w[3*2560+n]) (:133926-133934, w loads :133874-133883)
//          out = m * 0.25 (:133935); f[n] = out when f != 0 (:133943), y[n] = bf16_rne(out) (:133945-133951).
//   sets j = 1..A-1 (set j reads x set j = x + 640*j bytes for B = 0, 1: <2,1> :134938-134945, <2,0> :135721-135724; x offsets 0/640, 16/656, ...):
//          B = 0, 1: every plane (B = 0) or the shared dot (B = 1) is dotted against x set j; the same sigmoid and fma chain run with
//                    wm[j*10240 + k*2560 + n] (<2,1> :135200-135262), out = m * 0.25, y[j*2560 + n], f[j*2560 + n] (:135266-135281).
//          B = 2: no sigmoid (the hash path ignores x): t = fma(w[j*10240 + 0*2560 + n], 0.5, 0), then fmac with 0.5 and planes 1..3
//                 (<2,2> :135609-135666, <3,2> :137411-137516); out = t * 0.25; y[j*2560 + n] = bf16_rne(out); f[j*2560 + n] = out when f != 0.
#include <hip/hip_runtime.h>

typedef __bf16 v2bf __attribute__((ext_vector_type(2)));

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
__device__ static inline unsigned rne_bits(float f) {
  unsigned u = __float_as_uint(f);
  return u + ((u >> 16) & 1u) + 0x7fffu;  // bfe 16,1 ; add3 0x7fff (:133683-133687)
}
__device__ static inline float widen_rne(float f) { return __uint_as_float(rne_bits(f) & 0xffff0000u); }
__device__ static inline float dot2(unsigned a, unsigned b, float c) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(v2bf, a), __builtin_bit_cast(v2bf, b), c, false);
}
__device__ static inline float sigmoid_widen(float r) {
  float e = __builtin_amdgcn_exp2f(r * -1.44269502162933349609375f);
  return widen_rne(1.0f / (1.0f + e));
}

template <int A, int B>
__device__ static inline void hcup4_body(const unsigned char* W, long a, long b, long c, const unsigned short* x,
                                         const unsigned short* wm, unsigned short* y, float* f) {
  __shared__ unsigned lut[256];
  const unsigned tid = threadIdx.x;
  const unsigned lane = tid & 15u, g = tid >> 4;
  const long n = (long)blockIdx.x * 16 + g;
  if (B != 2) {
    const float* cb = (const float*)W;
    lut[tid] = (rne_bits(cb[g]) & 0xffff0000u) | (rne_bits(cb[lane]) >> 16);
  }
  __syncthreads();
  float p[A][4];
  for (int j = 0; j < A; j++) p[j][0] = p[j][1] = p[j][2] = p[j][3] = 0.f;
  if (lane < 10) {
    // x set j: 320 bf16 = 640 bytes at x + j*640, this lane reads 64 bytes at +lane*64 (<2,1> :134938-134945: offsets 0/640, 16/656, ...).
    if (B == 1) {
      // Shared codes and scale for every set and plane (<2,1> :134935, :134950): the same code bytes dotted with x set j.
      const uint4 cw = *(const uint4*)(W + a + lane * 16);
      const unsigned cwv[4] = {cw.x, cw.y, cw.z, cw.w};
      _Float16 h = *(const _Float16*)(W + b + 2 * lane);
      for (int j = 0; j < A; j++) {
        const unsigned* xp = (const unsigned*)(x) + j * 160 + lane * 16;
        float acc = 0.0f;
        for (int q = 0; q < 16; q++) acc = dot2(xp[q], lut[(cwv[q >> 2] >> (8 * (q & 3))) & 0xffu], acc);
        float s = __builtin_fmaf((float)h, acc, 0.0f);
        p[j][0] = p[j][1] = p[j][2] = p[j][3] = s;
      }
    } else {
      for (int k = 0; k < 4; k++) {
        const uint4 cw = *(const uint4*)(W + a + n * 160 + (long)k * 0x64000 + lane * 16);
        _Float16 h = *(const _Float16*)(W + b + 2 * lane + c * (n + 2560L * k));
        if (B == 2) {
          unsigned hash = (cw.y ^ cw.x) ^ cw.z ^ cw.w;
          p[0][k] = __builtin_fmaf((float)h, (float)hash, 0.0f);
        } else {
          // B = 0: every plane's codes are dotted with every x set (<2,0> :135721-135724: x offsets 0, 640, 16, 656, ...).
          const unsigned cwv[4] = {cw.x, cw.y, cw.z, cw.w};
          for (int j = 0; j < A; j++) {
            const unsigned* xp = (const unsigned*)(x) + j * 160 + lane * 16;
            float acc = 0.0f;
            for (int q = 0; q < 16; q++) acc = dot2(xp[q], lut[(cwv[q >> 2] >> (8 * (q & 3))) & 0xffu], acc);
            p[j][k] = __builtin_fmaf((float)h, acc, 0.0f);
          }
        }
      }
    }
  }
  for (int off = 8; off >= 1; off >>= 1)
    for (int j = 0; j < A; j++)
      for (int k = 0; k < 4; k++) p[j][k] = p[j][k] + __shfl_xor(p[j][k], off);
  if (lane != 0) return;
  for (int j = 0; j < A; j++) {
    const unsigned short* wj = wm + (long)j * 10240;
    float o;
    if (B == 2 && j >= 1) {
      // B = 2 sets 1.. : constant 0.5 weights, no sigmoid (<2,2> :135636-135648, <3,2> :137434-137499).
      float t = __builtin_fmaf(bf2f(wj[0 * 2560 + n]), 0.5f, 0.0f);
      t = __builtin_fmaf(0.5f, bf2f(wj[1 * 2560 + n]), t);
      t = __builtin_fmaf(0.5f, bf2f(wj[2 * 2560 + n]), t);
      t = __builtin_fmaf(0.5f, bf2f(wj[3 * 2560 + n]), t);
      o = t * 0.25f;
    } else {
      float s[4];
      for (int k = 0; k < 4; k++) s[k] = sigmoid_widen(widen_rne(p[j][k]));
      float m = __builtin_fmaf(s[0], bf2f(wj[0 * 2560 + n]), 0.0f);
      m = __builtin_fmaf(s[1], bf2f(wj[1 * 2560 + n]), m);
      m = __builtin_fmaf(s[2], bf2f(wj[2 * 2560 + n]), m);
      m = __builtin_fmaf(s[3], bf2f(wj[3 * 2560 + n]), m);
      o = m * 0.25f;
    }
    y[(long)j * 2560 + n] = (unsigned short)(rne_bits(o) >> 16);
    if (f) f[(long)j * 2560 + n] = o;
  }
}


#define HCUP4(A, B)                                                                                                         \
  extern "C" __global__ void k_hcup4_##A##_##B(const unsigned char* W, long a, long b, long c, const unsigned short* x,      \
                                               const unsigned short* wm, unsigned short* y, float* f) {                     \
    hcup4_body<A, B>(W, a, b, c, x, wm, y, f);                                                                              \
  }
HCUP4(1, 0) HCUP4(1, 1) HCUP4(1, 2)
HCUP4(2, 0) HCUP4(2, 1) HCUP4(2, 2)
HCUP4(3, 0) HCUP4(3, 1) HCUP4(3, 2)
HCUP4(4, 0) HCUP4(4, 1) HCUP4(4, 2)
