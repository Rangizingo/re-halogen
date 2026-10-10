# halogen-flash-server — ATTENTION kernel family, objdump-level writeup

Target: `flash_serve` fatbin device code, gfx1151 (`amdgcn-amd-amdhsa`, LLVM
OpenCL C, wave32). Listings: `analysis/kernels-disasm/codeobj_*.s`
(`llvm-objdump -d --mattr=+gfx1151`); per-kernel slices in
`work/kernels/*.s`. Kernel-argument maps and LDS sizes below come from the
`NT_AMDGPU_METADATA` note of the owning code object (msgpack
`amdhsa.kernels`).

**Provenance note (correction to the tasking):** the attention family is split
across **two** code objects, not one. Prefill/decode tile kernels
(`k_attn_fa2*`, `k_attn_qs*`, `k_attn_fa`, `k_kv_bt`) live in
**codeobj_1513000.s**; the flash-decode, selection, pool-write and RoPE
kernels (`k_attn_fd*`, `k_attn_sel`, `k_attn`, `k_kv_scatter_rows`,
`k_rope*`) live in **codeobj_53000.s**. Addresses below are cited against the
owning listing.

## 0. Family map

| kernel (entry) | codeobj | wgs | LDS (B) | MMA | role |
|---|---|---|---|---|---|
| `k_attn_fa2<1,1,1,3,0,0>` (0x61700) | 1513000 | 384 | 42240 | 32× `v_wmma_f32_16x16x16_bf16` | FA2 prefill, FaSel-gated |
| `k_attn_fa2<2,1,1,3,0,0>` (0x101d00) | 1513000 | 768 | 63744 | 32× wmma | FA2 prefill, 2× block |
| `k_attn_fa2<1,1,1,5,1,1>` (0x2fd00) | 1513000 | 384 | 42240 | 32× wmma | FA2, single-buffered variant |
| `k_attn_fa` (0x19c00) | 1513000 | — | — | 32× wmma | FA1-style predecessor (no FaSel, 0 barriers) |
| `k_attn_qs<1,0>` (0x170800) | 1513000 | 128 | 64340 | 32× wmma | query-stationary decode, contiguous KV |
| `k_attn_qs_bt` (0x25e00) | 1513000 | 128 | 34880 | 32× wmma | qs + block-table paging |
| `k_attn_qs_bt2<16,0>` (0x1a4a00) | 1513000 | 64 | 17616 | 16× wmma | qs block-table v2, tile 16 |
| `k_attn_qs_bt2w<7,7>` (0x18a100) | 1513000 | 64 | 9216 | 16× wmma | qs_bt2 "w" (wave-narrow) |
| `k_attn_qs_bt4<3>` (0x1d6800) | 1513000 | 128 | 12880 | 8× wmma | qs block-table v4 |
| `k_attn_qs_bt4x<1>` (0x1dd800) | 1513000 | 128 | — | 16× wmma | qs_bt4 extended |
| `k_attn_fd<1>` (0x8e9c00) | 53000 | 256 | 15456 | 384× `v_dot2_f32_bf16` | flash-decode split-K, f32 partials |
| `k_attn_fd_merge` (0x269100) | 53000 | 256 | 0 | — | merge split partials → bf16 out |
| `k_attn_sel` (0x266e00) | 53000 | 1024 | 17440 | — | sparse KV-block selection (top-k) |
| `k_attn` (0x267b00) | 53000 | 1024 | — | — | tiny single-tile attention fallback |
| `k_kv_bt` (0x25a00) | 1513000 | 1024 | 0 | — | KV page copy into pool (block table) |
| `k_kv_scatter_rows` (0x268600) | 53000 | 1024 | 0 | — | KV row scatter into pool |
| `k_rope` (0x25c500), `k_rope_yarn` (0x25c800), `k_rope_w` (0x25cb00), `k_rope_w_mrope` (0x25d900), `k_rope_blocks` (0x25ec00) | 53000 | 1024 | 0 | — | RoPE at write time (plain / YaRN / mRoPE) |

No `v_mfma_*` appears anywhere in this family: gfx1151 uses wave-level
`v_wmma_f32_16x16x16_bf16` (RDNA-style WMMA, 16×16×16 bf16→f32) for the tile
kernels and `v_dot2_f32_bf16` for the flash-decode dot products.

## 1. `k_attn_fa2` — FlashAttention-2 prefill

Signature (metadata, `k_attn_fa2<...>vPKtS3_S3_lS3_PtPKiiiNS_5FaSelE`):
`Q(PKt,ro) K(PKt,ro) V(PKt,ro) stride(l,i64) bias(PKt,ro) O(Pt,wo)
sel_idx(PKi,ro) n(i32) m(i32) FaSel(32 B by value @0x40)`.
`<1,1,1,3,0,0>`: wgs 384 (12 waves), 207 VGPR, LDS 42 240 B.

Structure (all addresses `codeobj_1513000.s`, kernel entry 0x61700):

- **Prologue** 0x61700–0x61e4f: `s_load_b256`/`s_load_b128` pull the whole
  kernarg block into SGPRs (0x61704 `s_load_b256 s[8:15],0x30` covers
  `sel_idx/n/m/FaSel`; 0x6170c `s_load_b128 s[20:23],0x50` covers FaSel
  fields). `FaSel` is a 32-byte by-value struct (4×u64: selected-block index
  list pointer + count/stride fields) consumed entirely through SGPRs.
- **Q·Kᵀ (S tile)** 0x61e50–0x62008: 16× `v_wmma_f32_16x16x16_bf16`
  (0x61e64…0x62008), A-operand = Q fragments `v[130:137]` (LDS-resident,
  `ds_load_b128 v171` offsets 0…11536 stride 768), B-operand = K fragments
  `v[146:153]` (`ds_load_b128` offsets 0/16 + 768·k), 16 f32 accumulators
  `v[1:128]` — a 16-tile (256-row) Q block against one 16-token K tile.
- **FaSel sparse gating** 0x62010–0x62260: three small loops
  (0x62010..0x62070, 0x620c0..0x6214c, 0x621c0..0x62260) implement the KV-block
  iteration. The index loop loads int32 block indices through the FaSel
  pointer (`global_load_b32 v130, v130, s[20:21]` @0x62044 — base register
  pair from the FaSel struct), scales them
  (`v_mad_u64_u32 v[133:134], null, s10, v133` @0x620E4), and issues
  `global_load_b128` (0x62124, 0x6227C–0x62588: 16 loads, offsets 0…480)
  that land in the K LDS tile (`ds_store_b128 … offset:31488` @0x62134).
  So **only the KV blocks named by `sel_idx`/FaSel are ever fetched** —
  dense attention is the degenerate case where the list covers all blocks.
- **Barriers / double-buffering**: 4× `s_barrier`
  (0x62078, 0x6226C, 0x62620, 0x6459C) bracket the K/V tile stores and the
  P-tile store — classic FA2 two-stage pipeline (load K(k+1) while computing
  S(k), load V(k) while softmax). The `<1,1,1,5,1,1>` variant has only 2
  barriers (0x30670, 0x3086C in its slice) — single-buffered; the 4th
  template arg (0…7, 26 instantiations) selects unroll/stage depth.
- **Online softmax** 0x62620–0x6459b: running row max `m` via 16×
  `v_max_f32_e32` (0x62C88…0x630DC) plus `v_max3_f32` (0x63C28…0x63C88);
  `v_exp_f32_e32` (hardware exp2; the scale factor `0x3fb8aa3b` = log2(e)
  appears at 0x65A20) computes `p = exp2((s−m)·log2e)` (0x63144…0x63B70);
  P is rounded to bf16 with RNE (`v_bfe_u32 …,16,1` + `v_add3_u32 …,0x7fff`,
  e.g. 0x63174/0x6317C) and stored into LDS (`ds_store_b16_d16_hi`
  0x63188…0x63B70, `ds_store_b128` 0x644D8/0x644E0).
- **P·V (O tile)** 0x622F4–0x625B0: 16× `v_wmma_f32_16x16x16_bf16`
  accumulating into `v[130:137]`, B-operand = V fragments from LDS.
- **Rescale + epilogue** 0x648C8–0x6E208: running-sum rescale factor
  `exp2(m_old−m_new)` (`v_exp_f32` 0x64B74…), final normalization
  `O = acc/l` with **IEEE-correct division** (`v_rcp_f32` + Newton FMA +
  `v_div_scale_f32`/`v_div_fmas_f32`/`v_div_fixup_f32`, 0x65A8C–0x65AC0,
  136 `v_rcp_f32` sites total), output rounded to bf16 RNE
  (`v_bfe_u32 v3,v2,16,1` 0x6DFB4 + `v_add3_u32 v2,v2,v3,0x7fff` 0x6DFBC)
  and stored `global_store_d16_hi_b16` (0x6DB48…0x6E208, 128 stores).

`k_attn_fa` (0x19c00) is the same shape without the FaSel argument and with
**zero** `s_barrier` — an FA1-style single-buffered predecessor kept in the
binary.

## 2. `k_attn_qs*` — query-stationary decode

`qs` = **query-stationary**: the Q tile is loaded once and kept resident
(80× `global_load_b128` in the prologue of `qs<1,0>`, 0x170800; Q fragments
re-read from LDS at 0x1A5204 `ds_load_b128 v[184:187], v96`), while K/V
stream through LDS per KV tile. Softmax is the same online scheme
(`v_max_f32` 0x1A5620…, `v_exp_f32` 0x1A56C8…, single `v_rcp_f32`
normalization at the end — 0x172E2C/0x172ECC in `qs<1,0>`), output one bf16
RNE store stream (`global_store_d16_hi_b16` 0x172F58).

- `k_attn_qs<1,0>` (0x170800, wgs 128, LDS 64 340): contiguous KV
  (`stride l` by-value i64), main loop 0x1710C0..0x171D54 (805 instrs,
  32 wmma).
- `k_attn_qs_bt` (0x25e00, wgs 128, LDS 34 880): adds a **block-table**
  pointer (4th buffer arg). K/V addresses are computed per page
  (`global_load_b64` 0x25B0C-style page loads; 96× `global_load_u16` +
  64× `global_load_b64`), main loop 0x26774..0x27D5C (1402 instrs).
- `k_attn_qs_bt2<a,b>` (0x1a4a00 for `<16,0>`, wgs 64, LDS 17 616):
  second-gen paging. The KV loop (0x1A5004..0x1A62B4, 1196 instrs) gathers
  32× `global_load_b64` K rows from **four page base addresses**
  (`v[110:111] v[156:157] v[158:159] v[182:183]`, offsets 0…896 stride 128,
  0x1A5104–0x1A51FC) — i.e. one KV tile assembled from up to 4 pool pages —
  then 8× wmma for S (0x1A5388–0x1A53DC) and 8× wmma for O with V streamed
  **directly from global** (16× `global_load_b128` 0x1A54C0–0x1A5538,
  wmma 0x1A5544–0x1A55A8). Template arg `a` ∈ {1,2,4,8,16,32,64} = KV tile
  size; `b` ∈ {1,2,3,7,9,11} = secondary tiling/flag.
- `k_attn_qs_bt2w<7,7>` (0x18a100, wgs 64, LDS 9 216): "w" = wave-narrow
  variant (LDS halved, 16 wmma); `k_attn_qs_bt4<3>` (0x1d6800, wgs 128,
  LDS 12 880, 8 wmma, 5 barriers) and `qs_bt4x<1>` (0x1dd800) are 4-page
  gather variants (16× `global_load_b64` page loads at 0x1D6E7C…).

All qs variants read pool K/V as **raw bf16** (`global_load_b128/b64/u16`)
straight into WMMA bf16 operands — no conversion instruction on the data
path (see §7).

## 3. `k_attn_fd*` — flash-decode split-K + merge

- `k_attn_fd<1>` (codeobj_53000.s:0x8e9c00, wgs 256, LDS 15 456, 201 VGPR):
  args `PKt KV, FdRows(160 B by value), PKi, PKi, i, Pt f32 partials(wo),
  PKt m/l(wo), i`. No WMMA: scores via 384× `v_dot2_f32_bf16` (bf16 dot
  pairs), wave reductions via 540× `ds_bpermute_b32`, online max/exp
  (`v_max_f32` 72, `v_exp_f32` 48). Writes **f32 partial outputs**
  (`global_store_b128` 0x8F1D08–0x8F1D94) plus per-split m/l.
- `k_attn_fd_merge` (codeobj_53000.s:0x269100, wgs 256): args
  `PKf partials, PKf m/l, i, Pt out(bf16,wo), PKt`. Loop 0x2691BC–0x269328:
  running max over split maxima (`v_max_f32` 0x2691C8/0x2691F0), rescale
  `exp2((m_i−m)·log2e)` (`v_mul_f32 0x3fb8aa3b` 0x26930C + `v_exp_f32`
  0x269314), accumulate partials and denominators (`v_fmac_f32_e32`
  0x26931C/0x269324), normalize with IEEE division (`v_rcp_f32` 0x269338 +
  `v_div_scale/v_div_fmas/v_div_fixup` 0x26934C–0x269380). Second stage
  0x269400–0x269484 loads a bf16 value (`global_load_u16` 0x269400,
  bf16→f32 via `v_lshlrev_b32 …,16` 0x26940C), computes
  `1/(1+exp2(−x·log2e))` (0x269414–0x26947C) and applies it, then bf16 RNE
  store (`v_bfe_u32 v3,v2,16,1` 0x26948C, `v_add3_u32 …,0x7fff` 0x269494,
  `global_store_d16_hi_b16` 0x2694B4).

## 4. `k_attn_sel` — sparse KV-block selection

codeobj_53000.s:0x266e00, wgs 1024, LDS 17 440. Args: 5 read-only buffers
(scores, Q, K, index lists ×2), `Pt` selection output (wo), `i, i`,
`PKt`. Structure: index-list gather loop (0x266ED0–0x266FA8),
`s_barrier` + `buffer_gl0_inv` (0x266FC8/0x266FCC) between stages, a
top-k/max-selection loop (0x267628..0x2679AC, 225 instrs) using
`v_max_f32_e32` (3 sites) + 38× `v_cndmask_b32` rank/keep predicates,
`global_load_u16` of candidate scores (0x26774C–0x26780C) and one
`global_store_d16_hi_b16` (0x26764C) writing the selection result. Its
output feeds the `sel_idx`/FaSel inputs of `k_attn_fa2` (§1): **FaSel gates
which KV blocks the FA2 loop touches** — the block-index loop at
0x62010..0x62070 iterates exactly the list this kernel produces.

## 5. `k_kv_bt` — block-table page copy

codeobj_1513000.s:0x25a00, wgs 1024, LDS 0. Args
`PKt src, PKt idx, l(i64), i, Pt dst(wo), PKt dst2(wo)`. Body: 4×
`global_load_u16` (0x25C44, 0x25CCC, 0x25D18, 0x25D78) — bf16 elements —
packed pairwise (`v_lshl_or_b32 v4, v0, 16, v4` 0x25C50, 0x25D24) and
written `global_store_b64` (0x25BCC, 0x25C88). **Pure data movement: no
arithmetic, no conversion** — KV rows are copied into pool pages addressed
through the block table. `k_kv_scatter_rows` (codeobj_53000.s:0x268600)
is the row-scatter twin: `global_load_d16_b16` 0x2686E4 +
`global_load_d16_hi_b16` 0x2686EC → `global_store_b16` 0x268740 +
`global_store_d16_hi_b16` 0x26874C, row targets from the 160-byte `FdRows`
struct. Also pure bf16 copy.

## 6. RoPE at write time

All five RoPE kernels (codeobj_53000.s) follow one pattern: load bf16
(`global_load_u16`), compute the rotation angle from the position
(`v_log_f32`/`v_exp_f32` for YaRN/mRoPE scaling, e.g. `k_rope_w`
`v_log_f32` 0x25D054, `v_exp_f32` 0x25D13C), rotate with hardware
`v_sin_f32`/`v_cos_f32` (`k_rope` 0x25C748/0x25C74C; `k_rope_yarn`
0x25C97C/0x25C980; `k_rope_w` 0x25D15C/0x25D160; `k_rope_blocks`
0x25EDD8/0x25EDDC; `k_rope_w_mrope` 0x25DF9C/0x25DFAC), combine
(`v_fma_f32 v8, v9, v8, -v11` 0x25C75C + `v_fmac_f32_e32 v10, v9, v7`
0x25C76C), round to bf16 RNE (`v_and_b32 …,1,v11` 0x25C77C +
`v_add3_u32 …,0x7fff` 0x25C788) and store bf16
(`global_store_d16_hi_b16` 0x25C7A4/0x25C7BC; yarn 0x25C9E8/0x25CA00;
rope_w 0x25D1C0/0x25D1C8; mrope 0x25DFFC/0x25E004; blocks
0x25EE34/0x25EE4C). `k_rope_blocks(Pt i, f, i)` takes the KV buffer
**in-place** (`Pt`, read-write) — RoPE is applied to K at prompt/prefill
time, once, before the values are scattered into the pool. No attention
kernel contains `v_sin`/`v_cos` — rotation is never re-applied at read time.

## 7. bf16 packing / rounding

The tasking suggested `v_cvt_pk`/`v_perm`; the actual ISA usage is the
RDNA4-style store-half scheme: bf16↔f32 widening is `v_lshlrev_b32 …,16`
(bf16 in high half = f32), narrowing is RNE via
`v_bfe_u32 x, v, 16, 1` (round bit) + `v_add3_u32 v, v, x, 0x7fff` +
`global_store_d16_hi_b16` (store high half). This appears at every kernel
boundary: FA2 epilogue 0x6DFB4/0x6DFBC, qs 0x172F28/0x172F38,
qs_bt2 0x1A5008/0x1A5020, fd_merge 0x26948C/0x269494, rope 0x25C77C/0x25C788.
The only `v_cvt_*` instructions in the attention kernels are
`v_cvt_i32_f32` (exp2 argument reduction for `v_exp_f32`) — **zero
format-conversion cvts on the K/V data path** in any attention kernel.

## 8. Acceptance claim: "prompt cache resume is bitwise" — kernel-level evidence

The claim requires: cached K/V are written to the pool **once, as bf16**,
and resume/decode **reads those exact bits** with no recompute/requantize.

1. **Write path (once, bf16):** K/V reach the pool through exactly three
   mechanisms, all bf16-exact:
   - RoPE kernels apply rotation at write time and store bf16 RNE
     (§6; `k_rope_blocks` in-place 0x25EE34/0x25EE4C).
   - `k_kv_scatter_rows` (0x268600) and `k_kv_bt` (0x25a00) are pure
     bf16 copies (§5) — load bf16, store bf16, no arithmetic.
   - The FA2 prefill kernel itself stores its K/V tiles into LDS as raw
     `ds_store_b128` of the loaded bits (0x62134) and its pool writes are
     the same RNE bf16 stores as everything else.
2. **Read path (as-is):** every decode kernel loads pool K/V with raw
   `global_load_b128/b64/u16` and feeds the loaded registers **directly**
   into `v_wmma_f32_16x16x16_bf16` / `v_dot2_f32_bf16` operands
   (qs<1,0> 0x1710C0 loop → 0x171160 wmma; qs_bt2 0x1A5104 loads →
   0x1A5388 wmma; fa2 0x62124/0x6227C loads → 0x61E64/0x622F4 wmma).
   No `v_cvt_*`, no `v_sin/v_cos`, no scale multiply touches K/V values on
   this path (§7). A resumed request re-reading pool positions therefore
   consumes bit-identical bf16 values.
3. **Determinism of the arithmetic that does run:** softmax uses hardware
   `v_exp_f32` (exp2) with the fixed log2e constant `0x3fb8aa3b`, and all
   normalizations use the IEEE-correct division sequence
   (`v_rcp_f32` + Newton + `v_div_scale/v_div_fmas/v_div_fixup`,
   e.g. 0x65A8C–0x65AC0, 0x269330–0x269380) — correctly-rounded, not
   rcp-approximate. Fixed work partition (uniform wgs, per-kernel tile
   counts) keeps reduction orders fixed for a given KV length.

**Verdict (kernel scope): SUPPORTED.** Cached K/V are written once as bf16
(RoPE applied at write time) and read back as raw bits into the MMA
operands; no kernel in the family recomputes or requantizes cached K/V.

## Open items

- Host-side half of the claim: that resume **dispatches** the decode
  kernels over the existing pool rows without re-running the prefill
  QKV+RoPE for cached positions (HALOGEN_CACHE_INPLACE bookkeeping in
  `flash_serve`) is not covered here — needs a dispatch-site pass over
  `main_pseudo.c`/`gf_*.c`.
- Exact semantics of fa2 template args 4–6 (0…7 / 1,1) and qs_bt2 arg `b`
  (1…11) inferred from barrier/LDS/wmma deltas only; no symbol-level
  confirmation.
- `k_attn` (0x267b00) and `k_attn_fa` (0x19c00) kept in the binary with no
  observed dispatch evidence in this pass; presumed fallback/legacy.
- `FaSel` field names (4×u64 at kernarg 0x40) are inferred from use
  (pointer + counts); struct layout not cross-checked against host
  dispatch code.
