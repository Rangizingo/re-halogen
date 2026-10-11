// Rewrite of the vis_attn family (vision tower attention, f32), codeobj_53000.s.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3
#include <hip/hip_runtime.h>
// The vendor keeps each separate mul and add separate; every fused op below is an explicit fmaf.
#pragma clang fp contract(off)

// k_vis_attn<8,32> (0x7b6b00), codeobj_53000.s:974572-975344.
// _ZN7halogen12_GLOBAL__N_110k_vis_attnILi8ELi32EEEvPKfPfif, kernarg: qkv@0 (f32 rows of 3456), out@8 (f32 rows of 1152), n@16, scale@20
// grid.x = query tile (QR = 8 rows, row0 = 8*blockIdx.x, :974579), grid.y = head (head*72 floats, :974576-974577), block = 72 threads (one per head dim d).
// Q tile -> LDS (rows >= n are zero, :974591-974656). Per key tile of KT = 32 keys (k0 += 32, :975057):
//   K[j][d] = qkv[(k0+j)*3456 + 1152 + head*72 + d], V[j][d] = qkv[(k0+j)*3456 + 2304 + head*72 + d], zero beyond n (:974695-974759).
//   score for e = tid, tid+72, ... < QR*KT, r = e / KT, j = e % KT (:974785-974841):
//     acc = 0; acc = fma(Q[r][d], K[j][d], acc) for d = 0..71 ascending (:974797-974826)
//     s = scale * acc (:974828); s = -inf when k0 + j >= n (v_cndmask 0xff800000 :974839)
//   per row r (:974886-975055), m = running max (init -inf), l = running sum (init 0), o = running output (init 0):
//     mn = max(m, s_0..s_31) (v_max3 chain :974889-974905)
//     alpha = exp2((m - mn) * log2e); p_j = exp2((s_j - mn) * log2e)   (0x3fb8aa3b, v_exp_f32 :974906-974916)
//     l = fma(l, alpha, p_0); l = l + p_j for j = 1..31 ascending (:974925, 974932, 974936...)
//     o = V[0][d] * p_0; o = fma(o_old, alpha, o); o = fma(V[j][d], p_j, o) for j = 1..31 (:974924,974928,974933...)
//   m = mn
// final (:975098-975110): out[(row0+r)*1152 + head*72 + d] = o / l (IEEE division) for row0 + r < n.
// The V[j]/p_j pairing and the sum order are read from register use; confirmed only by MATCH.
template <int QR, int KT>
__device__ static void vis_attn_body(const float* qkv, float* out, int n, float scale) {
  __shared__ float Qs[QR][72];
  __shared__ float Ks[KT][72];
  __shared__ float Vs[KT][72];
  __shared__ float Ss[QR][KT];
  const int d = threadIdx.x, head = blockIdx.y, row0 = blockIdx.x * QR;
  const float L = __uint_as_float(0x3fb8aa3bu);
  for (int r = 0; r < QR; r++) Qs[r][d] = (row0 + r < n) ? qkv[(long)(row0 + r) * 3456 + head * 72 + d] : 0.0f;
  float m[QR], l[QR], o[QR];
  for (int r = 0; r < QR; r++) { m[r] = __uint_as_float(0xff800000u); l[r] = 0.0f; o[r] = 0.0f; }
  if (n > 0) {
    for (int k0 = 0; k0 < n; k0 += KT) {
      for (int j = 0; j < KT; j++) {
        bool ok = k0 + j < n;
        Ks[j][d] = ok ? qkv[(long)(k0 + j) * 3456 + 1152 + head * 72 + d] : 0.0f;
        Vs[j][d] = ok ? qkv[(long)(k0 + j) * 3456 + 2304 + head * 72 + d] : 0.0f;
      }
      __syncthreads();
      for (int e = d; e < QR * KT; e += 72) {
        int r = e / KT, j = e % KT;
        float acc = 0.0f;
        for (int c = 0; c < 72; c++) acc = fmaf(Qs[r][c], Ks[j][c], acc);
        float s = scale * acc;
        Ss[r][j] = (k0 + j >= n) ? __uint_as_float(0xff800000u) : s;
      }
      __syncthreads();
      for (int r = 0; r < QR; r++) {
        float mn = m[r];
        for (int j = 0; j < KT; j++) mn = fmaxf(mn, Ss[r][j]);
        float alpha = __builtin_amdgcn_exp2f((m[r] - mn) * L);
        float p0 = __builtin_amdgcn_exp2f((Ss[r][0] - mn) * L);
        float ls = fmaf(l[r], alpha, p0);
        float os = Vs[0][d] * p0;
        os = fmaf(o[r], alpha, os);
        for (int j = 1; j < KT; j++) {
          float p = __builtin_amdgcn_exp2f((Ss[r][j] - mn) * L);
          ls = ls + p;
          os = fmaf(Vs[j][d], p, os);
        }
        l[r] = ls; o[r] = os; m[r] = mn;
      }
      __syncthreads();
    }
  }
  for (int r = 0; r < QR; r++)
    if (row0 + r < n) out[(long)(row0 + r) * 1152 + head * 72 + d] = o[r] / l[r];
}

extern "C" __global__ void k_vis_attn_8_32(const float* qkv, float* out, int n, float scale) {
  vis_attn_body<8, 32>(qkv, out, n, scale);
}

// ---- k_vis_attn_w<W, HL>: WMMA bf16 flash attention, 16 query rows per wave ----
// W = waves per block (block = 32*W threads), HL = true splits P into hi + lo bf16 and runs a second WMMA.
// Listings: W=1 HL=1 :965494 (0x7aa200), W=1 HL=0 :966666, W=2 HL=1 :967796, W=2 HL=0 :968990, W=8 HL=1 :970133,
// W=8 HL=0 :971210, W=4 HL=1 :972238, W=4 HL=0 :973425. All share one structure; HL=1 only adds the lo-P WMMA group
// (w1_1 vs w1_0: second group of 5 v_wmma with A = second P block, same B).
// _ZN7halogen12_GLOBAL__N_112k_vis_attn_wILi<W>ELb<HL>EEEvPKfPfif, kernarg: qkv@0, out@8, n@16 (i32), scale@20 (f32)
// grid.x = ceil(n / (16*W)), grid.y = head; wave w of block b owns rows row0 = 16*(W*b + w) .. +15 (s_lshl_b32 s2,4 :966674, v57 = tid>>5 in W>1).
// Per wave, all in LDS as bf16 (truncation, ds_store_b16_d16_hi):
//   Q[16][80] (cols >= 72 or rows >= n are 0; cooperative load over e < 1280, row = e/80 via 0xcccd mul, :966704-966723)
//   per key tile of 16 keys (ceil(n/16) tiles, :966747-966749): K[key][80], V^T[col][key], zero when key >= n or col >= 72 (:966908-967017)
// S tile (:967042-967062): acc = 0; 5 x wmma_f32_16x16x16_bf16(A = Q cols 16c..16c+15 of row lane%16, B = K row lane%16 same cols) chained;
//   s = scale * acc (v_mul_f32 s9 :967047); s = -inf where key >= n (v_cndmask 0xff800000 + s0 = n > key, :967039-967058)
//   D layout: register r of lane l holds row 2r + l/16, col l%16 (stores at row*64 + col*4 with 0x2000 base :967059-967062)
// Softmax, lane t < 16 handles row t (:967063-967228), m = LDS 0x2400, alpha 0x2440, l 0x2480, init m = -inf, l = 0:
//   mn = max(m_old, s_0..s_15) (v_max3 :967071-967078)
//   pv = (mn > -1e29) (0xefa18f08, v_cmp_lt_f32 literal < v54 writes s0 :967082); p_j = pv ? exp2((s_j - mn)*log2e) : 0 (:967084-967158)
//   av = (m_old > -1e29) (:967159); alpha = av ? exp2((m_old - mn)*log2e) : 0 (:967148-967163)
//   sum = ((p_0 + p_1) + p_2) ... + p_15 ascending (:967107-967221); l = fma(alpha, l_old, sum) (v_fmac :967222); m = mn
//   P packed to bf16 (truncation; INFERRED from .long VOP3 words), HL: lo = trunc(p - hi) (INFERRED, same as k_vis_softmax lo path)
// O update (:966833-966857): O_c (5 tiles of 16 cols, D layout as above) *= alpha[row] (v_mul_f32), then O_c = wmma(P_hi, V^T tile c, O_c);
//   HL: then wmma(P_lo, V^T tile c, O_c)
// final (:967269-): out[row][c*16 + col] = (1.0f / l[row]) * O (IEEE div then v_mul_f32), for row < n and col < 72 (guard INFERRED).
typedef short v16s_t __attribute__((ext_vector_type(16)));
typedef float v8f_t __attribute__((ext_vector_type(8)));

__device__ static inline unsigned short trunc_bf16(float f) { return (unsigned short)(__float_as_uint(f) >> 16); }

template <int W, bool HL>
__device__ static void vis_attn_w_body(const float* qkv, float* out, int n, float scale) {
  __shared__ unsigned short Qs[W][16][80];
  __shared__ unsigned short Ks[16][80];
  __shared__ unsigned short Vt[80][16];
  __shared__ float Ssh[W][16][16];
  __shared__ unsigned short Phi[W][16][16];
  __shared__ unsigned short Plo[W][16][16];
  __shared__ float sm_m[W][16], sm_l[W][16], sm_a[W][16];
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, head = blockIdx.y;
  const int row0 = 16 * (W * blockIdx.x + wave);
  const float L = __uint_as_float(0x3fb8aa3bu);
  const float NEG = __uint_as_float(0xefa18f08u);
  const float NINF = __uint_as_float(0xff800000u);
  for (int e = tid; e < W * 1280; e += 32 * W) {
    int w = e / 1280, ee = e % 1280, r = ee / 80, c = ee % 80;
    int grow = 16 * (W * blockIdx.x + w) + r;
    Qs[w][r][c] = (grow < n && c < 72) ? trunc_bf16(qkv[(long)grow * 3456 + head * 72 + c]) : (unsigned short)0;
  }
  if (lane < 16) { sm_m[wave][lane] = NINF; sm_l[wave][lane] = 0.0f; }
  v8f_t O[5];
  for (int c = 0; c < 5; c++) O[c] = (v8f_t){0, 0, 0, 0, 0, 0, 0, 0};
  const int ntiles = (n + 15) / 16;
  for (int t = 0; t < ntiles; t++) {
    const int k0 = 16 * t;
    __syncthreads();
    for (int e = tid; e < 1280; e += 32 * W) {
      int key = e / 80, c = e % 80;
      bool ok = (k0 + key < n) && c < 72;
      Ks[key][c] = ok ? trunc_bf16(qkv[(long)(k0 + key) * 3456 + 1152 + head * 72 + c]) : (unsigned short)0;
      Vt[c][key] = ok ? trunc_bf16(qkv[(long)(k0 + key) * 3456 + 2304 + head * 72 + c]) : (unsigned short)0;
    }
    __syncthreads();
    {
      v8f_t acc = (v8f_t){0, 0, 0, 0, 0, 0, 0, 0};
      for (int c = 0; c < 5; c++) {
        v16s_t a, b;
        for (int k = 0; k < 16; k++) {
          a[k] = (short)Qs[wave][lane & 15][c * 16 + k];
          b[k] = (short)Ks[lane & 15][c * 16 + k];
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
      }
      for (int r = 0; r < 8; r++) {
        float s = scale * acc[r];
        int col = lane & 15, row = 2 * r + (lane >> 4);
        Ssh[wave][row][col] = (k0 + col < n) ? s : NINF;
      }
    }
    __syncthreads();
    if (lane < 16) {
      float s[16];
      for (int j = 0; j < 16; j++) s[j] = Ssh[wave][lane][j];
      float mo = sm_m[wave][lane], lo_ = sm_l[wave][lane];
      float mn = mo;
      for (int j = 0; j < 16; j++) mn = fmaxf(mn, s[j]);
      bool pv = NEG < mn, av = NEG < mo;
      float p[16];
      for (int j = 0; j < 16; j++) p[j] = pv ? __builtin_amdgcn_exp2f((s[j] - mn) * L) : 0.0f;
      float alpha = av ? __builtin_amdgcn_exp2f((mo - mn) * L) : 0.0f;
      float sum = p[0] + p[1];
      for (int j = 2; j < 16; j++) sum = sum + p[j];
      sm_l[wave][lane] = fmaf(alpha, lo_, sum);
      sm_m[wave][lane] = mn;
      sm_a[wave][lane] = alpha;
      for (int j = 0; j < 16; j++) {
        unsigned short hi = trunc_bf16(p[j]);
        Phi[wave][lane][j] = hi;
        if (HL) Plo[wave][lane][j] = trunc_bf16(p[j] - __uint_as_float((unsigned)hi << 16));
      }
    }
    __syncthreads();
    for (int c = 0; c < 5; c++)
      for (int r = 0; r < 8; r++) O[c][r] = sm_a[wave][2 * r + (lane >> 4)] * O[c][r];
    for (int c = 0; c < 5; c++) {
      v16s_t a, b;
      for (int k = 0; k < 16; k++) {
        a[k] = (short)Phi[wave][lane & 15][k];
        b[k] = (short)Vt[c * 16 + (lane & 15)][k];
      }
      O[c] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, O[c]);
    }
    if (HL) {
      for (int c = 0; c < 5; c++) {
        v16s_t a, b;
        for (int k = 0; k < 16; k++) {
          a[k] = (short)Plo[wave][lane & 15][k];
          b[k] = (short)Vt[c * 16 + (lane & 15)][k];
        }
        O[c] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, O[c]);
      }
    }
  }
  __syncthreads();
  for (int r = 0; r < 8; r++) {
    int lrow = 2 * r + (lane >> 4);
    int grow = row0 + lrow;
    float rr = 1.0f / sm_l[wave][lrow];
    for (int c = 0; c < 5; c++) {
      int col = c * 16 + (lane & 15);
      if (grow < n && col < 72) out[(long)grow * 1152 + head * 72 + col] = rr * O[c][r];
    }
  }
}

extern "C" __global__ void k_vis_attn_w_1_1(const float* q, float* o, int n, float s) { vis_attn_w_body<1, true>(q, o, n, s); }
extern "C" __global__ void k_vis_attn_w_1_0(const float* q, float* o, int n, float s) { vis_attn_w_body<1, false>(q, o, n, s); }
extern "C" __global__ void k_vis_attn_w_2_1(const float* q, float* o, int n, float s) { vis_attn_w_body<2, true>(q, o, n, s); }
extern "C" __global__ void k_vis_attn_w_2_0(const float* q, float* o, int n, float s) { vis_attn_w_body<2, false>(q, o, n, s); }
extern "C" __global__ void k_vis_attn_w_4_1(const float* q, float* o, int n, float s) { vis_attn_w_body<4, true>(q, o, n, s); }
extern "C" __global__ void k_vis_attn_w_4_0(const float* q, float* o, int n, float s) { vis_attn_w_body<4, false>(q, o, n, s); }
extern "C" __global__ void k_vis_attn_w_8_1(const float* q, float* o, int n, float s) { vis_attn_w_body<8, true>(q, o, n, s); }
extern "C" __global__ void k_vis_attn_w_8_0(const float* q, float* o, int n, float s) { vis_attn_w_body<8, false>(q, o, n, s); }
