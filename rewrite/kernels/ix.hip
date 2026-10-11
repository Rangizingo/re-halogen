// Rewrites of the "ix" (indexer) family from codeobj_53000 / codeobj_1513000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o ix.co ix.hip
// Listing: kernels-disasm/codeobj_53000.s and codeobj_1513000.s (not in git); cites are codeobj_53000.s unless marked 1513000:.
#include <hip/hip_runtime.h>

// Products are fused (v_fmac_f32) only where the listing shows it.
#pragma clang fp contract(off)

typedef long long ll;

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// k_ix_gather_k(unsigned short const* a, int na, unsigned short const* b, int nc, unsigned short* out, long x, long rows)
// _ZN7halogen12_GLOBAL__N_113k_ix_gather_kEPKtiS2_iPtll, kernarg: a@0, na@8, b@16, nc@24, out@32, x@40, rows@48
// Grid-stride copy of rows*128 bf16 (s_lshl_b64 rows,7 = element count, codeobj_53000.s:55210; stride = gridDim.x*blockDim.x from the
// hidden block_count_x and group_size_x, :55204-55222; guard t < count :55215). For t: r = t >> 7, c = t & 127 (:55585-55587);
// logical row r' = r + x (v_add_co_u32 with s6/s7 = x, :55589-55591). If r' < nc (v_cmp_gt_i64 s[2:3], :55593) the source is a
// at row r' - (nc - na) (:55594-55603), otherwise b at row r' - nc. Row stride 256 bytes (v_lshlrev_b64 8, :55603), column
// c*2 bytes (:55587, :55610); out[t] = element (u16 load :55616, store :55620).
extern "C" __global__ void k_ix_gather_k(const unsigned short* a, int na, const unsigned short* b, int nc, unsigned short* out,
                                         ll x, ll rows) {
  ll total = rows << 7;
  ll stride = (ll)gridDim.x * blockDim.x;
  for (ll t = (ll)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += stride) {
    ll r = (t >> 7) + x;
    int c = (int)(t & 127);
    const unsigned short* src = r < (ll)nc ? a + (r - ((ll)nc - na)) * 128 : b + (r - nc) * 128;
    out[t] = src[c];
  }
}

// k_ix_scores_strip(unsigned short const* q, unsigned short const* k, float* s, long ld, int off, int bias, int rows, int cols)
// _ZN7halogen12_GLOBAL__N_117k_ix_scores_stripEPKtS2_Pfliiii, kernarg: q@0, k@8, s@16, ld@24, off@32, bias@36, rows@40, cols@44
// Block (x = key tile, y = query tile), 256 threads = 16 query rows x 16 keys (tid >> 4, tid & 15).
// q0 = y*16 (:55740); nk = (off + q0 + bias + 16) / 4 with signed round-toward-zero division (:55742-55752, the block's causal limit);
// col0 = x*16 (:55751). If col0 >= nk the block only writes zeros to every (row < rows, col < cols) cell (:55753-55819).
// Otherwise thread (qi, kj): row = q0 + qi, col = col0 + kj; for row < rows and col < cols (:55928-55929):
//   count = (qi + off + q0 + bias + 1) / 4 (:55931-55937); if col < count:
//     for head h = 0..3: acc_h = sum over d = 0..127 in order of fma(q[(off+row)*512 + h*128 + d], k[col*128 + d], acc_h)
//       (Q rows are 512 bf16 = 1024 bytes in LDS :55943, K rows 128 bf16 with 272-byte LDS stride :55944; loads :55951-55954,
//        fmac chains :55963-56005 and three more copies for the other heads at LDS offsets 512/768/1024)
//     score = rsq(128.0f) * (((0 + relu(acc_0)) + relu(acc_1)) + relu(acc_2)) + relu(acc_3)   (:56191-56205; relu = max(0, x) after a
//       self-max canonicalization, v_rsq_f32 of 0x43000000)
//   else score = 0; s[row*ld + col] = score (:56207-56229).
extern "C" __global__ void k_ix_scores_strip(const unsigned short* q, const unsigned short* k, float* s, ll ld, int off, int bias,
                                             int rows, int cols) {
  int tid = threadIdx.x, qi = tid >> 4, kj = tid & 15;
  int q0 = blockIdx.y * 16, col0 = blockIdx.x * 16;
  int nk = (off + q0 + bias + 16) / 4;
  int row = q0 + qi, col = col0 + kj;
  if (!(row < rows && col < cols)) return;
  float score = 0.0f;
  if (col0 < nk) {
    int count = (qi + off + q0 + bias + 1) / 4;
    if (col < count) {
      float h[4];
      for (int hh = 0; hh < 4; hh++) {
        float acc = 0.0f;
        for (int d = 0; d < 128; d++)
          acc = __builtin_fmaf(bf2f(q[(long)(off + row) * 512 + hh * 128 + d]), bf2f(k[(long)col * 128 + d]), acc);
        h[hh] = acc;
      }
      float c = 128.0f;
      asm volatile("" : "+v"(c));
      float sum = 0.0f + fmaxf(0.0f, h[0]);
      sum = sum + fmaxf(0.0f, h[1]);
      sum = sum + fmaxf(0.0f, h[2]);
      sum = sum + fmaxf(0.0f, h[3]);
      score = __builtin_amdgcn_rsqf(c) * sum;
    }
  }
  s[(ll)row * ld + col] = score;
}

// ---------------------------------------------------------------------------------------------------------------
// WMMA scoring kernels (codeobj_1513000). Same function as k_ix_scores_strip but the 128-long dot products run on the matrix
// unit: D(16x16) = A(16x16) * B(16x16) + C, A = a 16-row query slice, B = 16 key columns, one 16-element chunk of the 128 head dims per
// instruction, 8 chunks per head in order with the accumulator chained (1513000:13908-14001, 14911-15032), 4 heads in separate accumulators
// that start from zero. Per wave32 lane l: A row = l & 15 (16 consecutive bf16 = one chunk of that Q row, lanes 16-31 repeat lanes 0-15,
// ds_load_b128 pairs at 32-byte steps :13843-13874), B column = l & 15 (16 consecutive bf16 of K row min(col, cols-1), global_load_b128
// pairs at 32-byte steps :13884-13900); D element i of lane l is row 2*i + (l >> 4), column l & 15 (the row index (lane>>4)&1 | q0 plus
// 2 per element, :14011, :14060).
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

__device__ static inline void ix_wmma_tile(const unsigned short* q, const unsigned short* k, float* s, ll ld, int off, int bias, int rows,
                                           int cols, int q0, int kcol) {
  int lane = threadIdx.x & 31;
  int arow = q0 + (lane & 15);
  int kc = min(kcol + (lane & 15), cols - 1);
  v8f acc[4];
  for (int h = 0; h < 4; h++) {
    v8f c = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int ch = 0; ch < 8; ch++) {
      v16s a, b;
      for (int j = 0; j < 16; j++) {
        a[j] = arow < rows ? (short)q[(long)(off + arow) * 512 + h * 128 + ch * 16 + j] : (short)0;
        b[j] = (short)k[(long)kc * 128 + ch * 16 + j];
      }
      c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    }
    acc[h] = c;
  }
  float rs = 128.0f;
  asm volatile("" : "+v"(rs));
  rs = __builtin_amdgcn_rsqf(rs);
  int col = kcol + (lane & 15);
  for (int i = 0; i < 8; i++) {
    int row = q0 + (lane >> 4) + 2 * i;
    if (row < rows && col < cols) {
      float sum = 0.0f + fmaxf(0.0f, acc[0][i]);
      sum = sum + fmaxf(0.0f, acc[1][i]);
      sum = sum + fmaxf(0.0f, acc[2][i]);
      sum = sum + fmaxf(0.0f, acc[3][i]);
      int count = (row + off + bias + 1) / 4;
      s[(ll)row * ld + col] = col < count ? rs * sum : 0.0f;
    }
  }
}

// k_ix_scores_wmma(q, k, s, ld, off, bias, rows, cols)   (codeobj_1513000)
// _ZN7halogen12_GLOBAL__N_116k_ix_scores_wmmaEPKtS2_Pfliiii, kernarg: q@0, k@8, s@16, ld@24, off@32, bias@36, rows@40, cols@44
// 128 threads = 4 waves, block (x = 64 keys, y = 16 query rows): q0 = y*16 (s3<<4, :13682), col0 = x*64 (s2<<6, :13683),
// wave w covers keys col0 + w*16 + (l & 15) (:13818-13841). A block whose col0 >= (off + q0 + bias + 16)/4 only writes zeros
// (:13684-13750); that is the same value the per-cell rule col < (row + off + bias + 1)/4 gives, so one rule is enough here.
extern "C" __global__ void k_ix_scores_wmma(const unsigned short* q, const unsigned short* k, float* s, ll ld, int off, int bias,
                                            int rows, int cols) {
  int w = threadIdx.x >> 5;
  ix_wmma_tile(q, k, s, ld, off, bias, rows, cols, blockIdx.y * 16, blockIdx.x * 64 + w * 16);
}

// k_ix_scores_loop(q, k, s, ld, off, bias, rows, cols, tpb)   (codeobj_1513000)
// _ZN7halogen12_GLOBAL__N_116k_ix_scores_loopEPKtS2_Pfliiiii, kernarg: ..., tpb@48
// 256 threads = 8 waves, block x = 128 keys (s2<<7, :14394), wave w covers keys col0 + w*16 + (l & 15) (:14393-14402); the K rows
// for the block are loaded once into registers (:14420-14436). Block y walks query tiles t = y*tpb ... min(y*tpb + tpb, ceil(rows/16)) - 1
// (:14378-14392), each tile loaded into LDS and scored as above (:14900-15032), eight stores per lane (:14729-15406).
extern "C" __global__ void k_ix_scores_loop(const unsigned short* q, const unsigned short* k, float* s, ll ld, int off, int bias,
                                            int rows, int cols, int tpb) {
  int w = threadIdx.x >> 5;
  int t0 = blockIdx.y * tpb;
  int t1 = min(t0 + tpb, (rows + 15) / 16);
  for (int t = t0; t < t1; t++) ix_wmma_tile(q, k, s, ld, off, bias, rows, cols, t * 16, blockIdx.x * 128 + w * 16);
}

// ---------------------------------------------------------------------------------------------------------------
// Top-k selection.
// k_ix_select_strip(float const* sc, long ld, int a, int b, int* ids, int* counts, int stride)
// _ZN7halogen12_GLOBAL__N_117k_ix_select_stripEPKfliiPiS3_i, kernarg: sc@0, ld@8, a@16, b@20, ids@24, counts@32, stride@40
// One block of 256 threads per row (row = blockIdx.x = s2). Output row index orow = a + row (:56240); nb = (a + row + b + 1) / 4 with the
// signed round-toward-zero fixup (:56242-56250) = number of visible 4-token blocks; the score row is sc[row*ld ...] (s_mul_i32 s18, s10, s2,
// :56299). k = stride.
//   nb <= k: identity list, ids[orow*k + i] = i for i < nb, counts[orow] = nb (:56252-56282).
//   nb >  k: four-pass radix select on the raw u32 bit pattern of the scores, most significant byte first: pass p histograms byte p of
//     every score whose higher bytes equal the threshold prefix found so far (ds_add_u32 on a 256-bin LDS histogram, :56318-56330, :56578-56589,
//     :56811-56822, :57044-57055), then scans bins 255 down to 0 adding bin counts to the number of larger keys until that count reaches k
//     (ds_load offsets 1020 ... 992, s_cmp_ge_u32, :56358-56420). After four passes T is the k-th largest raw key and `above` the number of
//     keys larger than T (stored with T at LDS offset 2048, :57230-57234, :56525).
//     The list is the columns col < nb in ascending order with key > T, plus the first (k - above) columns with key == T: flag = key == T gets an
//     LDS prefix scan (:57290-57539), out position = running count + exclusive prefix (:57550-57561). ids[orow*k + pos] = col; counts[orow] = k (:57572).
extern "C" __global__ void k_ix_select_strip(const float* sc, ll ld, int a, int b, int* ids, int* counts, int stride) {
  __shared__ unsigned hist[256];
  __shared__ unsigned thr, rem_s;
  __shared__ int cnt;
  int tid = threadIdx.x, row = blockIdx.x;
  int orow = a + row;
  int nb = (a + row + b + 1) / 4;
  int k = stride;
  const unsigned* key = (const unsigned*)(sc + (ll)row * ld);
  int* oid = ids + (ll)orow * k;
  if (nb <= k) {
    for (int i = tid; i < nb; i += 256) oid[i] = i;
    if (tid == 0) counts[orow] = nb;
    return;
  }
  unsigned T = 0;
  unsigned above = 0;
  for (int pass = 0; pass < 4; pass++) {
    int shift = 24 - 8 * pass;
    unsigned pmask = pass == 0 ? 0u : (0xffffffffu << (shift + 8));
    hist[tid] = 0;
    __syncthreads();
    for (int i = tid; i < nb; i += 256) {
      unsigned v = key[i];
      if ((v & pmask) == (T & pmask)) atomicAdd(&hist[(v >> shift) & 0xff], 1u);
    }
    __syncthreads();
    if (tid == 0) {
      unsigned cum = above;
      int bin = 255;
      for (; bin >= 0; bin--) {
        unsigned nc = cum + hist[bin];
        if (nc >= (unsigned)k) break;
        cum = nc;
      }
      thr = T | ((unsigned)bin << shift);
      rem_s = cum;
    }
    __syncthreads();
    T = thr;
    above = rem_s;
    __syncthreads();
  }
  if (tid == 0) {
    unsigned need = (unsigned)k - above;
    unsigned ties = 0;
    int pos = 0;
    for (int i = 0; i < nb; i++) {
      unsigned v = key[i];
      bool sel = v > T || (v == T && ties < need);
      if (v == T) ties++;
      if (sel) oid[pos++] = i;
    }
    counts[orow] = pos;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Long-row top-k: k_ix_select_slice<T> cuts a score row into slices of 1024*T columns, each slice emits its own top-k candidates, and
// k_ix_select_merge picks the global top-k among the candidates. All three use one selection rule:
//   key = raw u32 bits of the score; T* = k-th largest key (found by a 32-step bit search "largest X with count(key >= X) >= k",
//   unsigned compares, e.g. codeobj_53000.s:1149270, 1149287-1149304, 1149346, with 0 as the answer when fewer than k keys exist);
//   selected = key > T*, plus the first (k - #(key > T*)) columns with key == T* in column order (the per-group counts are clamped
//   with v_maxmin_i32 against the remaining quota, :1149578-1149866); the ids are written in ascending order (prefix compaction).
__device__ static unsigned kth_largest_key(const unsigned* key, const unsigned char* valid, int n, int k) {
  unsigned t = 0;
  for (int bit = 31; bit >= 0; bit--) {
    unsigned cand = t | (1u << bit);
    int cnt = 0;
    for (int i = 0; i < n; i++) cnt += (valid == nullptr || valid[i]) && key[i] >= cand;
    if (cnt >= k) t = cand;
  }
  return t;
}

// Writes the selected positions (indices into key[]) in ascending order through put(); returns the count.
// valid == nullptr means every position takes part.
template <typename F>
__device__ static int topk_positions(const unsigned* key, const unsigned char* valid, int n, int k, F put) {
  int total = 0;
  for (int i = 0; i < n; i++) total += (valid == nullptr || valid[i]);
  if (total <= k) {
    int pos = 0;
    for (int i = 0; i < n; i++) if (valid == nullptr || valid[i]) put(pos++, i);
    return pos;
  }
  unsigned t = kth_largest_key(key, valid, n, k);
  int above = 0;
  for (int i = 0; i < n; i++) above += (valid == nullptr || valid[i]) && key[i] > t;
  int need = k - above, ties = 0, pos = 0;
  for (int i = 0; i < n; i++) {
    if (!(valid == nullptr || valid[i])) continue;
    bool sel = key[i] > t || (key[i] == t && ties < need);
    if (key[i] == t) ties++;
    if (sel) put(pos++, i);
  }
  return pos;
}

// k_ix_select_slice<T>(float const* sc, long ld, int a, int b, int k, int d, int* o1, int* o2, int* o3, int* o4)
// kernarg: sc@0, ld@8, a@16, b@20, k@24, d@28, o1@32, o2@40, o3@48, o4@56. Block (x = row, y = slice), 1024 threads, T values per thread
// (T = 8, 16, 32 -> 8192, 16384, 32768 columns per slice: s_lshl_b32 s25, s3, 13 at :1149032, 14 at 1150283; s_min_i32 0x2000 / 0x4000 / 0x8000).
// nb = (a + row + b + 1) / 4 (:1148969-1148978).
//   d == 1 and nb <= k: o1[(a+row)*k + i] = i for i < nb, o2[a+row] = nb (:1148979-1149028).
//   d  > 1 and slice start >= nb: o4[d*row + slice] = 0 (:1150203-1150210).
//   otherwise the top-k of columns start .. min(start + 1024*T, nb) - 1 as above; d == 1: o1[(a+row)*k + pos] = column, o2[a+row] = count
//   (:1150175-1150182, s28 = o1 + k*(a+row)*4 at :1149409-1149414); d > 1: o3[(d*row + slice)*k + pos] = column,
//   o4[d*row + slice] = count (:1150185-1150193, s28 = o3 + ((d*row+slice)*k)*4 at :1149417-1149436).
template <int SLICE>
__device__ static void slice_body(const float* sc, ll ld, int a, int b, int k, int d, int* o1, int* o2, int* o3, int* o4) {
  if (threadIdx.x != 0) return;
  int row = blockIdx.x, slice = blockIdx.y;
  int nb = (a + row + b + 1) / 4;
  if (d == 1 && nb <= k) {
    int* o = o1 + (ll)(a + row) * k;
    for (int i = 0; i < nb; i++) o[i] = i;
    o2[a + row] = nb;
    return;
  }
  ll start = (ll)slice * SLICE;
  ll span = nb - start;
  if (d > 1 && span < 1) { o4[(ll)d * row + slice] = 0; return; }
  int n = (int)(span < SLICE ? span : SLICE);
  const unsigned* key = (const unsigned*)(sc + (ll)row * ld + start);
  int* o = d == 1 ? o1 + (ll)(a + row) * k : o3 + ((ll)d * row + slice) * k;
  int cnt = topk_positions(key, nullptr, n, k, [&](int pos, int i) { o[pos] = (int)start + i; });
  if (d == 1) o2[a + row] = cnt; else o4[(ll)d * row + slice] = cnt;
}
extern "C" __global__ void k_ix_select_slice8(const float* sc, ll ld, int a, int b, int k, int d, int* o1, int* o2, int* o3, int* o4) {
  slice_body<8192>(sc, ld, a, b, k, d, o1, o2, o3, o4);
}
extern "C" __global__ void k_ix_select_slice16(const float* sc, ll ld, int a, int b, int k, int d, int* o1, int* o2, int* o3, int* o4) {
  slice_body<16384>(sc, ld, a, b, k, d, o1, o2, o3, o4);
}
extern "C" __global__ void k_ix_select_slice32(const float* sc, ll ld, int a, int b, int k, int d, int* o1, int* o2, int* o3, int* o4) {
  slice_body<32768>(sc, ld, a, b, k, d, o1, o2, o3, o4);
}

// k_ix_select_merge(float const* sc, long ld, int a, int k, int nsl, int const* p, int const* q, int* o1, int* o2)
// _ZN7halogen12_GLOBAL__N_117k_ix_select_mergeEPKfliiiPKiS4_PiS5_, kernarg: sc@0, ld@8, a@16, k@20, nsl@24, p@32, q@40, o1@48, o2@56
// One block (1024 threads) per row. Candidate slot idx in 0 .. nsl*k - 1 (16 per thread: idx = wave*512 + 32*j + lane, :57667-57671) belongs to
// slice idx / k (signed division by k, :57691-57725) at position idx % k; it is valid if position < q[row*nsl + slice] (:57738-57741);
// its id is p[row*nsl*k + idx] (:57744), -1 otherwise (v9 = -1); the key is sc[row*ld + id] or 0 for an invalid slot (:57750-57771).
// Selection over the slots in idx order as above, o1[(a+row)*k + pos] = id, o2[a+row] = count.
extern "C" __global__ void k_ix_select_merge(const float* sc, ll ld, int a, int k, int nsl, const int* p, const int* q, int* o1, int* o2) {
  if (threadIdx.x != 0) return;
  int row = blockIdx.x;
  int total = nsl * k;
  const int* pr = p + (ll)row * total;
  const int* qr = q + (ll)row * nsl;
  const unsigned* base = (const unsigned*)(sc + (ll)row * ld);
  unsigned t = 0;
  int nvalid = 0;
  for (int idx = 0; idx < total; idx++) nvalid += (idx % k) < qr[idx / k];
  int* o = o1 + (ll)(a + row) * k;
  auto keyof = [&](int idx) -> unsigned { return (idx % k) < qr[idx / k] ? base[pr[idx]] : 0u; };
  auto isvalid = [&](int idx) -> bool { return (idx % k) < qr[idx / k]; };
  if (nvalid <= k) {
    int pos = 0;
    for (int idx = 0; idx < total; idx++) if (isvalid(idx)) o[pos++] = pr[idx];
    o2[a + row] = pos;
    return;
  }
  for (int bit = 31; bit >= 0; bit--) {
    unsigned cand = t | (1u << bit);
    int cnt = 0;
    for (int idx = 0; idx < total; idx++) cnt += isvalid(idx) && keyof(idx) >= cand;
    if (cnt >= k) t = cand;
  }
  int above = 0;
  for (int idx = 0; idx < total; idx++) above += isvalid(idx) && keyof(idx) > t;
  int need = k - above, ties = 0, pos = 0;
  for (int idx = 0; idx < total; idx++) {
    if (!isvalid(idx)) continue;
    unsigned key = keyof(idx);
    bool sel = key > t || (key == t && ties < need);
    if (key == t) ties++;
    if (sel) o[pos++] = pr[idx];
  }
  o2[a + row] = pos;
}
