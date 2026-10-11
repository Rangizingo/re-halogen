// Rewrites of the dnc_scan3 family (chunked delta-rule scan) from codeobj_1146000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o scan3.co dnc_scan3.hip
// Listing: kernels-disasm/codeobj_1146000.s. Cites are code-object addresses (@12xxxx), read from a ROCm llvm-objdump
// re-decode (the repo listing prints some 8-byte VOP3 words as .long). Base kernel = k_dnc_scan3<8,8,8,true,0,15,0>
// at @12bc00 (listing line 154482).
#include <hip/hip_runtime.h>

#pragma clang fp contract(off)

// Kernarg layout of every k_dnc_scan3<...> (and scan/scan2/scan4/scan5/scan3t): keys@0 (never read), q@8, kk@16, g@24, v@32,
// w@40, a@48, st@56 (state in/out), idx@64, n@72, o@80 (bf16 out), k@88.
//
// Roles (all proven by the arithmetic below):
//   grid (48 heads, A column slabs), block 128. Head h = blockIdx.x, slab y = blockIdx.y, slab width CW = 128/A columns.
//   S = state [128 rows][CW cols] of head h, f32, address st + n + idx[0]*0x2400000 + h*16384 + y*CW + row*128 + col
//       (@12bc34 s_load idx[0], @12bc5c s_mul_i32 idx*0x9000000 bytes, @12bc48 head<<16 bytes, @12bc14 y*16 floats).
//   q, kk, v, w: [token][48][128] f32 (24576 B per token, 512 B per head). g: [token][48] f32. a: [chunk][48][64][64] f32.
//   Chunk c = 64 tokens (nch = (k+63)>>6, @12bd88/@12bd94).
//   per chunk:
//     e0[t] = expf(g_t), e1[t] = expf(g_63 - g_t)                       (@12bfd4..@12c0f0, LDS 0x4420 / 0x4320)
//     pass 1 (8 tokens a time): u[t][col] = v[t][col] - sum_j w[t][j]*S[j][col]   fma chain j = 0..127 (@12c230..@12cc6c)
//     pass 2: o[t][col] = bf16( (sum_j q[t][j]*S[j][col]) * e0[t] + sum_{j'<=t} a[t][j']*u[j'][col] ), only t < k
//             (@12cd8c..@12d640; first term fma(q0,S0,0), then fmac; bf16 RNE without NaN case, @12d614..@12d634)
//     S *= expf(g_63)                                                    (@12d664..@12d7a0)
//     S[j][col] += sum_{t=0..7} (kk[t][j]*e1[t]) * u[t][col], per 8-token group, in order  (@12d7a0..@12e4e4)
//   after the last chunk S is written back (@12e69c..@12e6cc).

__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// A = column slabs (grid.y). Only A = 8 is instantiated here.
template <int A>
__device__ static inline void scan3_body(const float* __restrict__ q, const float* __restrict__ kk, const float* __restrict__ g,
                                         const float* __restrict__ v, const float* __restrict__ w, const float* __restrict__ a,
                                         float* __restrict__ st, const int* __restrict__ idx, long n, unsigned short* __restrict__ o,
                                         int k) {
  constexpr int CW = 128 / A;       // columns per block
  constexpr int RT = 128 / CW;      // token rows per pass (= A)
  constexpr int SS = CW + 1;        // LDS row stride in floats
  __shared__ float S[128 * SS];
  __shared__ float U[64 * SS];
  __shared__ float T[RT * 129];
  __shared__ float e1[64], e0[64];
  const int tid = threadIdx.x, head = blockIdx.x, y = blockIdx.y;
  const int r = tid / CW, col = tid % CW;
  float* sbase = st + n + (long)idx[0] * 0x2400000L + (long)head * 16384 + y * CW;
  for (int i = tid; i < 128 * CW; i += 128) S[(i / CW) * SS + (i % CW)] = sbase[(i / CW) * 128 + (i % CW)];
  __syncthreads();
  const int nch = (k > 0) ? ((k + 63) >> 6) : 0;
  for (int c = 0; c < nch; ++c) {
    const float glast = g[(long)c * 3072 + 3024 + head];
    if (tid < 64) {
      float gt = g[(long)(c * 64 + tid) * 48 + head];
      e1[tid] = expf(glast - gt);
      e0[tid] = expf(gt);
    }
    __syncthreads();
    // pass 1
    for (int s10 = 0; s10 < 64; s10 += RT) {
      for (int i = 0; i < RT; ++i) T[i * 129 + tid] = w[(long)(c * 64 + s10 + i) * 6144 + head * 128 + tid];
      __syncthreads();
      float acc = v[(long)(c * 64 + s10 + r) * 6144 + head * 128 + y * CW + col];
      for (int j = 0; j < 128; ++j) acc = __builtin_fmaf(-T[r * 129 + j], S[j * SS + col], acc);
      U[(s10 + r) * SS + col] = acc;
      __syncthreads();
    }
    // pass 2
    for (int s10 = 0; s10 < 64; s10 += RT) {
      for (int i = 0; i < RT; ++i) T[i * 129 + tid] = q[(long)(c * 64 + s10 + i) * 6144 + head * 128 + tid];
      __syncthreads();
      const int tl = s10 + r, t = c * 64 + tl;
      if (t < k) {
        float acc = __builtin_fmaf(T[r * 129], S[col], 0.0f);
        for (int j = 1; j < 128; ++j) acc = __builtin_fmaf(T[r * 129 + j], S[j * SS + col], acc);
        float val = acc * e0[tl];
        const float* ar = a + (long)c * 196608 + head * 4096 + tl * 64;
        for (int jp = 0; jp <= tl; ++jp) val = __builtin_fmaf(ar[jp], U[jp * SS + col], val);
        o[(long)t * 6144 + head * 128 + y * CW + col] = f2bf_rne(val);
      }
      __syncthreads();
    }
    // state decay
    const float gl = expf(glast);
    for (int i = tid; i < 128 * CW; i += 128) S[(i / CW) * SS + (i % CW)] *= gl;
    __syncthreads();
    // state update
    for (int s10 = 0; s10 < 64; s10 += RT) {
      for (int i = 0; i < RT; ++i) T[i * 129 + tid] = kk[(long)(c * 64 + s10 + i) * 6144 + head * 128 + tid] * e1[s10 + i];
      __syncthreads();
      for (int i = tid; i < 128 * CW; i += 128) {
        const int j = i / CW, cc = i % CW;
        float s = S[j * SS + cc];
        for (int tt = 0; tt < RT; ++tt) s = __builtin_fmaf(T[tt * 129 + j], U[(s10 + tt) * SS + cc], s);
        S[j * SS + cc] = s;
      }
      __syncthreads();
    }
  }
  for (int i = tid; i < 128 * CW; i += 128) sbase[(i / CW) * 128 + (i % CW)] = S[(i / CW) * SS + (i % CW)];
}

#define SCAN3_A(AA)                                                                                                  \
  extern "C" __global__ __launch_bounds__(128) void scan3_a##AA(                                                    \
      const unsigned short* keys, const float* q, const float* kk, const float* g, const float* v, const float* w,  \
      const float* a, float* st, const int* idx, long n, unsigned short* o, int k) {                                \
    scan3_body<AA>(q, kk, g, v, w, a, st, idx, n, o, k);                                                            \
  }
SCAN3_A(2)
SCAN3_A(4)
SCAN3_A(8)
SCAN3_A(16)
SCAN3_A(32)
