// Rewrites of the "deq_q4cp" family (q4c planar dequant to bf16/f32), vendor code object 53000, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o deq.co deq_q4cp.hip
// Listing: kernels-disasm/codeobj_53000.s (untracked; regenerate with llvm-objdump --mattr=+gfx1151). "L:n" below means that file, line n.
// The ISA has `.long` words where llvm-objdump could not decode 64-bit VALU ops; those are read from
// context (inferred) and the compare run is the check.
#include <hip/hip_runtime.h>

// f32 -> bf16, RNE without NaN special case: v_bfe_u32 ;; v_add3_u32 ..., 0x7fff ;; store high half (L:16372-16374).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
// v_fma_mix_f32 d, scale_f16, lut_f32, neg(0) with op_sel_hi:[1,0,0] = fma(f32(scale), lut, -0.0) (L:16370).
// Inline asm pins the operand order: with two NaN inputs the hardware result depends on it (a plain fmaf
// gave a different NaN payload than the vendor on random-byte inputs).
__device__ static inline float fma_scale(unsigned short sc, float lut) {
  float d;
  unsigned s = sc;
  asm("v_fma_mix_f32 %0, %1, %2, neg(0) op_sel_hi:[1,0,0]" : "=v"(d) : "v"(s), "v"(lut));
  return d;
}

// halogen::(anonymous namespace)::k_deq_q4cp(unsigned char const*, unsigned short*, long, long, long, long, long)
// _ZN7halogen12_GLOBAL__N_110k_deq_q4cpEPKhPtlllll, kernarg x@0 y@8 a0@16(unused) n@24 dat@32 sco@40 scs@48
// One block per row r = blockIdx.x (s2). x holds a 16-entry f32 LUT at +0 (global_load_b32 v0, v0, s[12:13] with
// v0 = nibble*4, L:16367), packed nibbles at x+dat + r*(n/2) (s6 = x+dat, row term s_ashr_i64 (n + n>>63), 1; L:16302-16313),
// f16 group-32 scales at x+sco + r*scs (L:16314-16318, v_lshrrev 4 then and -2 = (i>>5)*2, L:16346-16349).
// Thread loop i = tid; i < n; i += blockDim (L:16365-16366, 16329); even i takes the low nibble (L:16354-16360).
// out y[r*n + i] = bf16_rne(fma(f16 scale, lut[nib], -0)) (L:16370-16374).
extern "C" __global__ void k_deq_q4cp(const unsigned char* x, unsigned short* y, long a0, long n, long dat, long sco, long scs) {
  long r = blockIdx.x;
  const unsigned char* d = x + dat + r * (n / 2);
  const unsigned char* s = x + sco + r * scs;
  unsigned short* o = y + r * n;
  for (long i = threadIdx.x; i < n; i += blockDim.x) {
    unsigned b = d[i >> 1];
    unsigned nib = (i & 1) ? (b >> 4) : (b & 15);
    unsigned short sc = *(const unsigned short*)(s + ((i >> 5) << 1));
    float lut = *(const float*)(x + nib * 4);
    o[i] = f2bf_rne(fma_scale(sc, lut));
  }
}

// k_deq_q4cp_f32(unsigned char const*, float*, long x5): same body, f32 store, 4-byte output stride (L:18173, 18210).
extern "C" __global__ void k_deq_q4cp_f32(const unsigned char* x, float* y, long a0, long n, long dat, long sco, long scs) {
  long r = blockIdx.x;
  const unsigned char* d = x + dat + r * (n / 2);
  const unsigned char* s = x + sco + r * scs;
  float* o = y + r * n;
  for (long i = threadIdx.x; i < n; i += blockDim.x) {
    unsigned b = d[i >> 1];
    unsigned nib = (i & 1) ? (b >> 4) : (b & 15);
    unsigned short sc = *(const unsigned short*)(s + ((i >> 5) << 1));
    float lut = *(const float*)(x + nib * 4);
    o[i] = fma_scale(sc, lut);
  }
}

// k_deq_q4cp_sel(unsigned char const*, unsigned short*, int const*, long a, long n, long dat, long sco, long scs)
// _ZN7halogen12_GLOBAL__N_114k_deq_q4cp_selEPKhPtPKilllll, kernarg x@0 y@8 sel@16 a@24 n@32 dat@40 sco@48 scs@56
// s_load_b512 s[4:19] (L:17841) gives x=s4, y=s6, sel=s8, a=s10, n=s12, dat=s14, sco=s16, scs=s18.
// b = blockIdx.x; q = b / a (64-bit divide, L:17846-18004), rem = b - q*a (L:18035-18036);
// row = sel[q] * a + rem (sign-extended int, L:18016, 18028-18038). Data and scale rows come from `row`
// (L:18039-18062), the output row from b itself (y + b*n*2, L:18021-18068). Element body as k_deq_q4cp (L:18087-18122).
extern "C" __global__ void k_deq_q4cp_sel(const unsigned char* x, unsigned short* y, const int* sel, long a, long n, long dat, long sco, long scs) {
  long b = blockIdx.x;
  long q = b / a;
  long rem = b - q * a;
  long row = (long)sel[q] * a + rem;
  const unsigned char* d = x + dat + row * (n / 2);
  const unsigned char* s = x + sco + row * scs;
  unsigned short* o = y + b * n;
  for (long i = threadIdx.x; i < n; i += blockDim.x) {
    unsigned bb = d[i >> 1];
    unsigned nib = (i & 1) ? (bb >> 4) : (bb & 15);
    unsigned short sc = *(const unsigned short*)(s + ((i >> 5) << 1));
    float lut = *(const float*)(x + nib * 4);
    o[i] = f2bf_rne(fma_scale(sc, lut));
  }
}

// ---- k_deq_q4cp_v8 and the v8u/v32/probe family ----
// Pack two bf16 into a dword the way the vendor does: v_lshrrev_b32 lo,16 ;; v_and_or_b32 0xffff0000 (L:16587).
__device__ static inline unsigned pack_bf16x2(float lo, float hi) {
  return ((unsigned)f2bf_rne(hi) << 16) | f2bf_rne(lo);
}

// k_deq_q4cp_v8(unsigned char const*, unsigned short*, long A, long B, long C, long D, long E)
// _ZN7halogen12_GLOBAL__N_113k_deq_q4cp_v8EPKhPtlllll, kernarg x@0 y@8 A@16 B@24 C@32 D@40 E@48 (hidden block_count_x@56)
// LDS (64 bytes static, group_segment 64): tid<16 copies the f32 LUT x[tid] to lds[tid] (L:16393-16397), s_barrier (L:16403).
// B8 = (B + ((B>>63)&7)) >> 3 (L:16401-16407); work items total = B8*A (L:16408-16412).
// gid = blockIdx.x*blockDim.x + tid (v_mad_u64_u32, L:16417), grid-stride loop with stride block_count_x*blockDim.x
// (L:16440-16442, 16543-16558). row = gid / B8, c = gid - row*B8 (64-bit divide, L:16603-16774).
// One dword of 8 nibbles at x + C + row*(B/2) + c*4 (L:16475, 16490; row stride (B/2) - 4*B8 folded into the gid*4 term);
// f16 scale at x + D + row*E + (c/4)*2 (L:16491-16509). Nibble j is bits 4j..4j+3 (shifts 0,2,6,10,...,26 then and 60 on a
// value that is already x4: L:16512-16527), looked up in LDS (L:16528-16535), fma_mix by the scale (L:16560-16576),
// bf16 RNE, packed pairwise, one global_store_b128 at y + row*B*2 + c*16 (L:16549, 16600).
extern "C" __global__ void k_deq_q4cp_v8(const unsigned char* x, unsigned short* y, long A, long B, long C, long D, long E) {
  __shared__ float lut[16];
  if (threadIdx.x < 16) lut[threadIdx.x] = ((const float*)x)[threadIdx.x];
  __syncthreads();
  long B8 = (B + ((B >> 63) & 7)) >> 3;
  long total = B8 * A;
  long stride = (long)gridDim.x * blockDim.x;
  for (long gid = (long)blockIdx.x * blockDim.x + threadIdx.x; gid < total; gid += stride) {
    long row = gid / B8;
    long c = gid - row * B8;
    unsigned w = *(const unsigned*)(x + C + row * (B / 2) + c * 4);
    unsigned short sc = *(const unsigned short*)(x + D + row * E + ((c / 4) << 1));
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) v[j] = fma_scale(sc, lut[(w >> (4 * j)) & 15]);
    uint4 o;
    o.x = pack_bf16x2(v[0], v[1]);
    o.y = pack_bf16x2(v[2], v[3]);
    o.z = pack_bf16x2(v[4], v[5]);
    o.w = pack_bf16x2(v[6], v[7]);
    *(uint4*)(y + row * B + c * 8) = o;
  }
}

// ---- k_deq_q4cp_v8u<N, FromCb> ----
// DeqCb is the 64-byte by-value struct at kernarg +56: 16 f32 LUT entries (FromCb: thread 0 copies it from the kernarg block
// into LDS with four ds_store_b128, L:73459-73473 of v8u<2,1>; else tid<16 copies x[tid] as in k_deq_q4cp_v8, L:72909-72917).
struct DeqCb { float lut[16]; };

// Scale conversion and multiply: v_cvt_f32_f16 then v_mul_f32 scale, lut (L:73002, 73043). Asm pins the operand order for NaNs.
__device__ static inline float mul_scale(float sc, float lut) {
  float d;
  asm("v_mul_f32 %0, %1, %2" : "=v"(d) : "v"(sc), "v"(lut));
  return d;
}
__device__ static inline float cvt_h(unsigned short h) {
  float d;
  unsigned s = h;
  asm("v_cvt_f32_f16 %0, %1" : "=v"(d) : "v"(s));
  return d;
}

// halogen::(anonymous namespace)::k_deq_q4cp_v8u<N, FromCb>(unsigned char const*, unsigned short*, unsigned a, unsigned b,
//   long n, long dat, long sco, long scs, DeqCb)
// kernarg x@0 y@8 a@16 b@20 n@24 dat@32 sco@40 scs@48 cb@56, hidden block_count_x@120.
// N items per thread per tile (tile = N*256 consecutive item ids).
// Tile loop: tile = blockIdx.x; run while tile*256*N < b (s_lshl_b32 s3,s2,8|9 ; s_cmp_ge_u32 s3,s17; L:72920-72923, 73476-73479);
// tile += block_count_x (s_load_b32 s0,s[0:1],0x78 ; s_add_i32 s2,s0,s2; L:73011, 73015).
// Item id g = tile*256*N + tid + 256*j, guarded g < b (L:72929, 73484-73487, 73511). row = g / a, c = g - row*a (32-bit
// unsigned divide via v_rcp_iflag_f32, L:72953-72968). Elements 8c..8c+7 of the row: dword at x + dat + row*(n/2) + c*4
// (s18 = n/2 signed, L:72931-72935, 72981-72991), f16 scale at x + sco + row*scs + (((8c)>>4)&~1) (L:72983, 72988),
// eight LDS lookups, v_mul_f32 by the f32 scale, bf16 RNE, pairwise pack, one 16-byte store (L:73023-73105).
template <int N, bool FromCb>
__device__ static inline void deq_v8u_body(const unsigned char* x, unsigned short* y, unsigned a, unsigned b, long n,
                                           long dat, long sco, long scs, const DeqCb& cb) {
  __shared__ float lut[16];
  if (FromCb) {
    if (threadIdx.x == 0)
      for (int k = 0; k < 16; k++) lut[k] = cb.lut[k];
  } else {
    if (threadIdx.x < 16) lut[threadIdx.x] = ((const float*)x)[threadIdx.x];
  }
  __syncthreads();
  long half_n = (n + ((n >> 63) & 1)) >> 1;
  for (unsigned tile = blockIdx.x; (unsigned long)tile * 256 * N < b; tile += gridDim.x) {
#pragma unroll
    for (int j = 0; j < N; j++) {
      unsigned g = tile * 256u * N + threadIdx.x + 256u * j;
      if (g >= b) continue;
      unsigned row = g / a;
      unsigned c = g - row * a;
      unsigned w = *(const unsigned*)(x + dat + (long)row * half_n + ((long)c * 8 >> 1));
      unsigned short h = *(const unsigned short*)(x + sco + (long)row * scs + (((c * 8u) >> 4) & 0xffffffeu));
      float sc = cvt_h(h);
      float v[8];
#pragma unroll
      for (int k = 0; k < 8; k++) v[k] = mul_scale(sc, lut[(w >> (4 * k)) & 15]);
      uint4 o;
      o.x = pack_bf16x2(v[0], v[1]);
      o.y = pack_bf16x2(v[2], v[3]);
      o.z = pack_bf16x2(v[4], v[5]);
      o.w = pack_bf16x2(v[6], v[7]);
      *(uint4*)(y + (long)row * n + (long)c * 8) = o;
    }
    __syncthreads();
  }
}

// ---- k_deq_q4cp_v32<FromCb> ----
// halogen::(anonymous namespace)::k_deq_q4cp_v32<FromCb>(unsigned char const*, unsigned short*, unsigned a, unsigned b,
//   long n, long dat, long sco, long scs, DeqCb)   same kernarg layout as v8u; one item = 32 elements, 256 items per tile.
// Item g = tile*256 + tid, guarded g < b (L:71609-71612, 71654); row = g / a, c = g - row*a (L:71637-71658).
// 16 data bytes (global_load_b128) at x + dat + row*(n/2) + c*16 (L:71663-71682, v5 = c<<5 then >>1: L:71660, 71671),
// f16 scale at x + sco + row*scs + (c*2) (L:71665-71668), one v_cvt_f32_f16 (L:71684). Each dword gives 8 LUT lookups
// as in v8u, v_mul_f32 by the scale, bf16 RNE, packed; four global_store_b128 at y + (row*n + 32c)*2 + 16k (L:71971-71974).
template <bool FromCb>
__device__ static inline void deq_v32_body(const unsigned char* x, unsigned short* y, unsigned a, unsigned b, long n,
                                           long dat, long sco, long scs, const DeqCb& cb) {
  __shared__ float lut[16];
  if (FromCb) {
    if (threadIdx.x == 0)
      for (int k = 0; k < 16; k++) lut[k] = cb.lut[k];
  } else {
    if (threadIdx.x < 16) lut[threadIdx.x] = ((const float*)x)[threadIdx.x];
  }
  __syncthreads();
  long half_n = (n + ((n >> 63) & 1)) >> 1;
  for (unsigned tile = blockIdx.x; (unsigned long)tile * 256 < b; tile += gridDim.x) {
    unsigned g = tile * 256u + threadIdx.x;
    if (g < b) {
      unsigned row = g / a;
      unsigned c = g - row * a;
      uint4 w = *(const uint4*)(x + dat + (long)row * half_n + (long)c * 16);
      unsigned short h = *(const unsigned short*)(x + sco + (long)row * scs + ((c << 1) & 0xffffffeu));
      float sc = cvt_h(h);
      unsigned wd[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int q = 0; q < 4; q++) {
        float v[8];
#pragma unroll
        for (int k = 0; k < 8; k++) v[k] = mul_scale(sc, lut[(wd[q] >> (4 * k)) & 15]);
        uint4 o;
        o.x = pack_bf16x2(v[0], v[1]);
        o.y = pack_bf16x2(v[2], v[3]);
        o.z = pack_bf16x2(v[4], v[5]);
        o.w = pack_bf16x2(v[6], v[7]);
        *(uint4*)(y + (long)row * n + (long)c * 32 + 8 * q) = o;
      }
    }
    __syncthreads();
  }
}
extern "C" __global__ void k_deq_q4cp_v32_0(const unsigned char* x, unsigned short* y, unsigned a, unsigned b, long n, long dat,
                                            long sco, long scs, DeqCb cb) {
  deq_v32_body<false>(x, y, a, b, n, dat, sco, scs, cb);
}
extern "C" __global__ void k_deq_q4cp_v32_1(const unsigned char* x, unsigned short* y, unsigned a, unsigned b, long n, long dat,
                                            long sco, long scs, DeqCb cb) {
  deq_v32_body<true>(x, y, a, b, n, dat, sco, scs, cb);
}

// ---- k_deq_q4cp_probe<A, B> ----
// halogen::(anonymous namespace)::k_deq_q4cp_probe<A, B>(unsigned char const*, unsigned short*, unsigned a, unsigned b,
//   long n, long dat, long sco, long scs, DeqCb)  -- same kernarg layout as v8u. A items per thread per tile (A = 1: 256-item tiles,
// L:69143; A = 4: 1024-item tiles, s_lshl_b32 s10, s2, 10, L:65207). The LUT always comes from DeqCb (four ds_store_b128, L:69137-69140)
// except B = 3, which has no LDS (L:69973-70036). Per item g: row = g / a, c = g - row*a. Variants of the per-item body:
//   B = 1: no memory reads. w = g * 0x9e3779b1 (v_mul_lo_u32, L:69180-69181); scale = fma(float(g & 7), 0.25f, 1.0f)
//          (v_and 7, v_cvt_f32_ubyte0, v_fmamk_f32 0x3e800000; L:69186-69192; the operand is the item id g, which the
//          compare run at a % 8 != 0 shows, not the column c); output as v8u (bf16 pairs, one b128 store).
//   B = 2: loads as v8u (full row); the eight bf16 values are folded into one word h = p0 ^ (p1<<1) ^ (p2<<2) ^ (p3<<3) with
//          p0 = b1<<16 | b0, p1 = (b3<<16 & 0x7fff0000) | b2, p2 = (b5<<16 & 0x3fff0000) | b4, p3 = (b7<<16 & 0x1fff0000) | b6
//          (v_and_or_b32 masks, v_lshlrev, v_xor_b32, v_xor3_b32; L:69782-69800); each thread xors its items' words; at the end
//          if the xor == 0x9e3779b9 the thread stores its own index as a u16 at y[tid] (v_cmpx_eq_u32 0x9e3779b9, global_store_b16;
//          L:66754-66758). Nothing else is written.
//   B = 3: loads w and the f16 scale as v8u (full row), s = bits of float(scale); stores the four dwords w+s, w+2s, w+3s, w+4s
//          (v_lshl_add_u32 s<<2 + w, then three v_sub_nc_u32; L:70082-70103). No table.
//   B = 4: as v8u (FromCb) but the data and scale reads use row & 0xff (v_and_b32 0xff, row; L:70359-70375); the output index
//          still uses the full row.
__device__ static inline unsigned probe_hash(float v[8]) {
  unsigned b[8];
#pragma unroll
  for (int k = 0; k < 8; k++) b[k] = f2bf_rne(v[k]);
  unsigned p0 = (b[1] << 16) | b[0];
  unsigned p1 = ((b[3] << 16) & 0x7fff0000u) | b[2];
  unsigned p2 = ((b[5] << 16) & 0x3fff0000u) | b[4];
  unsigned p3 = ((b[7] << 16) & 0x1fff0000u) | b[6];
  return p0 ^ (p1 << 1) ^ (p2 << 2) ^ (p3 << 3);
}

template <int A, int B>
__device__ static inline void deq_probe_body(const unsigned char* x, unsigned short* y, unsigned a, unsigned b, long n, long dat,
                                             long sco, long scs, const DeqCb& cb) {
  __shared__ float lut[16];
  if (B != 3) {
    if (threadIdx.x == 0)
      for (int k = 0; k < 16; k++) lut[k] = cb.lut[k];
  }
  __syncthreads();
  long half_n = (n + ((n >> 63) & 1)) >> 1;
  unsigned chk = 0;
  for (unsigned tile = blockIdx.x; (unsigned long)tile * 256 * A < b; tile += gridDim.x) {
#pragma unroll
    for (int j = 0; j < A; j++) {
      unsigned g = tile * 256u * A + threadIdx.x + 256u * j;
      if (g >= b) continue;
      unsigned row = g / a;
      unsigned c = g - row * a;
      unsigned lrow = (B == 4) ? (row & 0xffu) : row;
      unsigned w;
      float sc;
      unsigned short h = 0;
      if (B == 1) {
        w = g * 0x9e3779b1u;
        sc = __builtin_fmaf((float)(g & 7u), 0.25f, 1.0f);  // g, not c: the two differ when a % 8 != 0 (N=100003 run)
      } else {
        w = *(const unsigned*)(x + dat + (long)lrow * half_n + ((long)c * 8 >> 1));
        h = *(const unsigned short*)(x + sco + (long)lrow * scs + (((c * 8u) >> 4) & 0xffffffeu));
        sc = cvt_h(h);
      }
      if (B == 3) {
        unsigned s = __float_as_uint(sc);
        uint4 o;
        o.x = w + s;
        o.y = w + 2u * s;
        o.z = w + 3u * s;
        o.w = w + 4u * s;
        *(uint4*)(y + (long)row * n + (long)c * 8) = o;
      } else {
        float v[8];
#pragma unroll
        for (int k = 0; k < 8; k++) v[k] = mul_scale(sc, lut[(w >> (4 * k)) & 15]);
        if (B == 2) {
          chk ^= probe_hash(v);
        } else {
          uint4 o;
          o.x = pack_bf16x2(v[0], v[1]);
          o.y = pack_bf16x2(v[2], v[3]);
          o.z = pack_bf16x2(v[4], v[5]);
          o.w = pack_bf16x2(v[6], v[7]);
          *(uint4*)(y + (long)row * n + (long)c * 8) = o;
        }
      }
    }
    __syncthreads();
  }
  // The stored value is the constant 1: the undecoded `.long 0x7e003881` (VOP1, src0 = inline constant 1, vdst = v0) runs between
  // the address shift and the store (L:69968-69969), and the vendor run writes 0x0001 at y[0] for thread 0.
  if (B == 2 && chk == 0x9e3779b9u) y[threadIdx.x] = 1;
}
#define DEQ_PROBE(A, B)                                                                                                       \
  extern "C" __global__ void k_deq_q4cp_probe_##A##_##B(const unsigned char* x, unsigned short* y, unsigned a, unsigned b, \
                                                        long n, long dat, long sco, long scs, DeqCb cb) {                 \
    deq_probe_body<A, B>(x, y, a, b, n, dat, sco, scs, cb);                                                                \
  }
DEQ_PROBE(1, 1)
DEQ_PROBE(1, 2)
DEQ_PROBE(1, 3)
DEQ_PROBE(1, 4)
DEQ_PROBE(4, 1)
DEQ_PROBE(4, 2)
DEQ_PROBE(4, 3)
DEQ_PROBE(4, 4)

#define DEQ_V8U(NAME, N, FROMCB)                                                                                    \
  extern "C" __global__ void NAME(const unsigned char* x, unsigned short* y, unsigned a, unsigned b, long n, long dat, \
                                  long sco, long scs, DeqCb cb) {                                                    \
    deq_v8u_body<N, FROMCB>(x, y, a, b, n, dat, sco, scs, cb);                                                       \
  }
DEQ_V8U(k_deq_q4cp_v8u_1_0, 1, false)
DEQ_V8U(k_deq_q4cp_v8u_1_1, 1, true)
DEQ_V8U(k_deq_q4cp_v8u_2_0, 2, false)
DEQ_V8U(k_deq_q4cp_v8u_2_1, 2, true)
DEQ_V8U(k_deq_q4cp_v8u_4_0, 4, false)
DEQ_V8U(k_deq_q4cp_v8u_4_1, 4, true)
DEQ_V8U(k_deq_q4cp_v8u_8_0, 8, false)
DEQ_V8U(k_deq_q4cp_v8u_8_1, 8, true)
