// Rewrite of the "dnc_prep" family from codeobj_1146000 (flash_serve, gfx1151):
//   halogen::(anonymous namespace)::k_dnc_prep<b1, b2, b3>(u16 const* qkv, f32 const* gate, f32* qn, f32* kn, f32* csum, int n,
//                                                         u16* qh, f32* qs, f32* ks)
//   halogen::(anonymous namespace)::k_dnc_prep_w(...same 9 args...)
// kernarg: qkv@0 gate@8 qn@16 kn@24 csum@32 n@40 qh@48 qs@56 ks@64.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dnc_prep.co dnc_prep.hip
// Listing: kernels-disasm/codeobj_1146000.s (not in git). Cites are for k_dnc_prep<false,false,false> (lines 307757-308094,
// "p000" below) unless another instance is named. Names are inferred from the arithmetic.
//
// What it computes, per (token t, head h), 128 lanes:
//   q = qkv[t][ (h/3)*128 + lane ], k = qkv[t][ 2048 + (h/3)*128 + lane ]  (bf16 -> f32; 0 when t >= n)  (loads at :307890/:307896)
//   qn = q * (rsq(128) * rsq(sum(q*q) + 1e-6)),  kn = k * rsq(sum(k*k) + 1e-6)     (L2 norm, q also scaled by 1/sqrt(128))
//   qh = f16(qn * 16)   when qh != null
//   csum[t][h] = running f32 sum of gate[t'][h] over the chunk, t' <= t           (tid 0 only; :307796-307806)
//   b3 instances additionally write qs[t][h] = rsq(128) * rsq(sum q^2 + eps), ks[t][h] = rsq(sum k^2 + eps),
//   and test qn / kn for null before storing (listing 307069-307420 tail).
// Sums are a fixed tree over 128 lanes: x[l] = x[l] + x[l+64]; then +32, +16, +8, +4, +2, +1 (LDS steps :307922-308040).
// The prep_w kernel (wave-per-token, 3 heads per block, no csum) uses the same tree via ds_bpermute and gives the same sums.
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// v_rsq_f32 with the denormal guard the vendor emits: if s < 0x800000 (as f32) scale s by 2^24 and the result by 2^12 (:308014-308022).
__device__ static inline float rsq_guard(float s) {
  bool lo = __uint_as_float(0x800000) > s;
  float t = lo ? s * __uint_as_float(0x4b800000) : s;
  float r = __builtin_amdgcn_rsqf(t);
  return lo ? r * __uint_as_float(0x45800000) : r;
}

// v_rsq_f32(128.0), computed on the device so it is the hardware value (:307790 v_rsq_f32 v7, 0x43000000).
__device__ static inline float rsq128() {
  float c;
  asm volatile("v_rsq_f32 %0, %1" : "=v"(c) : "v"(128.0f));
  return c;
}

#define EPS_BITS 0x358637bd

// ---- k_dnc_prep<b1,b2,b3>: block 128 threads, grid (ceil(n/64), 48) = (chunk, head) ----
template <bool B3, bool SWAP>
__device__ static inline void prep_body(const unsigned short* qkv, const float* gate, float* qn, float* kn, float* csum, int n,
                                        unsigned short* qh, float* qs, float* ks) {
  __shared__ float red[128];
  __shared__ float run;
  const int tid = threadIdx.x;
  const int chunk = SWAP ? blockIdx.y : blockIdx.x, h = SWAP ? blockIdx.x : blockIdx.y;
  const int g = h / 3;
  const float c128 = rsq128();
  if (tid == 0) run = 0.0f;  // ds_store LDS[512] = 0 (:307759-307763)
  __syncthreads();
  for (int t = 0; t < 64; t++) {
    const int tok = chunk * 64 + t;
    const bool valid = tok < n;
    float q = valid ? bf2f(qkv[(long)tok * 10240 + g * 128 + tid]) : 0.0f;
    float k = valid ? bf2f(qkv[(long)tok * 10240 + 2048 + g * 128 + tid]) : 0.0f;
    const long o = ((long)(unsigned)tok * 48 + h) * 128 + tid;
    const long ti = (long)(unsigned)tok * 48 + h;

    float sums[2];
    for (int pass = 0; pass < 2; pass++) {
      float x = pass == 0 ? q : k;
      __syncthreads();
      red[tid] = x * x;
      __syncthreads();
      if (tid < 64) red[tid] = red[tid + 64] + red[tid];
      __syncthreads();
      if (tid < 32) red[tid] = red[tid + 32] + red[tid];
      __syncthreads();
      if (tid < 16) red[tid] = red[tid + 16] + red[tid];
      __syncthreads();
      if (tid < 8) red[tid] = red[tid + 8] + red[tid];
      __syncthreads();
      if (tid < 4) red[tid] = red[tid + 4] + red[tid];
      __syncthreads();
      if (tid < 2) red[tid] = red[tid + 2] + red[tid];
      __syncthreads();
      if (tid == 0) red[0] = red[1] + red[0];
      __syncthreads();
      sums[pass] = red[0];
    }
    float rq = c128 * rsq_guard(sums[0] + __uint_as_float(EPS_BITS));
    float rk = rsq_guard(sums[1] + __uint_as_float(EPS_BITS));
    float qv = q * rq;
    if (!B3 || qn) qn[o] = qv;
    if (qh) qh[o] = __builtin_bit_cast(unsigned short, (_Float16)(qv * 16.0f));
    if (!B3 || kn) kn[o] = k * rk;
    if (tid == 0) {
      if (B3) {
        qs[ti] = rq;
        ks[ti] = rk;
      }
      float s13 = valid ? gate[ti] : 0.0f;
      float S = s13 + run;
      run = S;
      csum[ti] = S;
    }
    __syncthreads();
  }
}

extern "C" __global__ __launch_bounds__(128) void dnc_prep(const unsigned short* qkv, const float* gate, float* qn, float* kn,
                                                           float* csum, int n, unsigned short* qh, float* qs, float* ks) {
  prep_body<false, false>(qkv, gate, qn, kn, csum, n, qh, qs, ks);
}
// <false,true,false>: same code as <false,false,false> but ctaid.x is the head and ctaid.y the chunk (listing 307420: s_mov_b32 s7,s3 ; s_mul_hi_i32 s0,s2,0x55555556).
extern "C" __global__ __launch_bounds__(128) void dnc_prep_sw(const unsigned short* qkv, const float* gate, float* qn, float* kn,
                                                              float* csum, int n, unsigned short* qh, float* qs, float* ks) {
  prep_body<false, true>(qkv, gate, qn, kn, csum, n, qh, qs, ks);
}
// <false,true,true>: swapped grid as above, plus the qs/ks stores and the null tests (listing 307069)
extern "C" __global__ __launch_bounds__(128) void dnc_prep_b3(const unsigned short* qkv, const float* gate, float* qn, float* kn,
                                                              float* csum, int n, unsigned short* qh, float* qs, float* ks) {
  prep_body<true, true>(qkv, gate, qn, kn, csum, n, qh, qs, ks);
}

// ---- k_dnc_prep_w: block 256 = 8 waves, grid (16, ceil(n/64)) = (q/k group, chunk). Wave w handles tokens chunk*64 + w*8 + i,
// each lane holds elements lane, lane+32, lane+64, lane+96 of q and k. The result is written for the 3 heads 3*g .. 3*g+2.
// Per-lane partial: (x0 + x64) + (x32 + x96) (listing 51000+: lines 181-190 of the dump), then lane tree +16, +8, +4, +2, +1.
// All 3 pointer stores test the pointer for null (s24 = qn, s25 = qh, s26 = kn); qs / ks are stored unconditionally (lane 0).
// gate and csum are not used.
extern "C" __global__ __launch_bounds__(256) void dnc_prep_w(const unsigned short* qkv, const float* gate, float* qn, float* kn,
                                                             float* csum, int n, unsigned short* qh, float* qs, float* ks) {
  __shared__ float red[8][2][32];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int g = blockIdx.x, chunk = blockIdx.y;
  const float c128 = rsq128();
  for (int i = 0; i < 8; i++) {
    const int tok = chunk * 64 + w * 8 + i;
    const bool valid = tok < n;
    float q[4], k[4];
    for (int e = 0; e < 4; e++) {
      q[e] = valid ? bf2f(qkv[(long)tok * 10240 + g * 128 + lane + 32 * e]) : 0.0f;
      k[e] = valid ? bf2f(qkv[(long)tok * 10240 + 2048 + g * 128 + lane + 32 * e]) : 0.0f;
    }
    float pq = (q[0] * q[0] + q[2] * q[2]) + (q[1] * q[1] + q[3] * q[3]);
    float pk = (k[0] * k[0] + k[2] * k[2]) + (k[1] * k[1] + k[3] * k[3]);
    red[w][0][lane] = pq;
    red[w][1][lane] = pk;
    __builtin_amdgcn_wave_barrier();
    for (int off = 16; off >= 1; off >>= 1) {
      float a0 = red[w][0][lane], a1 = red[w][1][lane];
      float b0 = red[w][0][(lane + off) & 31], b1 = red[w][1][(lane + off) & 31];
      __builtin_amdgcn_wave_barrier();
      red[w][0][lane] = a0 + b0;
      red[w][1][lane] = a1 + b1;
      __builtin_amdgcn_wave_barrier();
    }
    float sq = red[w][0][0], sk = red[w][1][0];
    __builtin_amdgcn_wave_barrier();
    float rq = c128 * rsq_guard(sq + __uint_as_float(EPS_BITS));
    float rk = rsq_guard(sk + __uint_as_float(EPS_BITS));
    for (int hh = 0; hh < 3; hh++) {
      const int h = 3 * g + hh;
      for (int e = 0; e < 4; e++) {
        const long o = ((long)(unsigned)tok * 48 + h) * 128 + lane + 32 * e;
        float qv = rq * q[e];  // v_mul v36, v34, v0 : (c*r) * q
        if (qn) qn[o] = qv;
        if (qh) qh[o] = __builtin_bit_cast(unsigned short, (_Float16)(qv * 16.0f));
        if (kn) kn[o] = rk * k[e];
      }
      if (lane == 0) {
        const long ti = (long)(unsigned)tok * 48 + h;
        qs[ti] = rq;
        ks[ti] = rk;
      }
    }
  }
}
