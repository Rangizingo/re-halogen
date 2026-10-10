# Closure: U17, U18, U24, U25 (n-gram lanes, dv_quant8, float-key select, dv_quant codebook)

Inputs: slices in `work/kernels-spec/` (file names abbreviated below), decompiles in
`work/decomp/flash_serve/` (0.17.3) and `work/decomp/flash_serve_0172/`. Slice line
numbers are lines of the slice file. Cap: about 45 tool calls; limits are stated per ID.

## U17: n-gram order per lane in k_ngram_ids

Verdict: CLOSED at kernel level (orders decoded from the loop bound). Host-side
launch site and the checkpoint tensor contents not read, so the head count per order
is inferred from "16 lanes, two orders".

Answer: lanes 0-7 hash a 2-token window, lanes 8-15 hash a 3-token window. So 16 lanes
= 2 orders (2-gram, 3-gram) x 8 heads, lane = order_index*8 + head (head = lane & 7,
inferred). Every lane uses the same multiplier table; lanes differ only in window
length, vocab modulus and table offset.

How decoded (`k_ngram_ids` slice, non-blank listing order):
- `v_lshrrev_b32 v1, 3, v0` then `v_sub_nc_u32 v3, 0, v1`: v3 = -(lane >> 3), 0 for
  lanes 0-7, -1 for lanes 8-15.
- Loop exit test: `v_add_nc_u32 v1, s20, v3` then `v_cmp_eq_u32 vcc_lo, 2, v1` then
  `s_and_not1_b32 exec_lo, exec_lo, s22`. s20 is the term counter k (init
  `s_mov_b32 s20, 1`, `s_add_i32 s20, s20, 1` per pass). Lanes with lane>>3 == 0 exit
  when k == 2 (two terms folded: one pre-loop, one in the loop); lanes with lane>>3 == 1
  exit when k == 3 (three terms).
- Fold: pre-loop `s_mul_i32/s_mul_hi_u32` of `s_load_b64 s[16:17],s[8:9]` (multiplier
  B[0]) by `s_load_b64 s[18:19],s[6:7],0x10` (token at index row+2, s6 = tokens + row*8).
  Each pass loads the next B[k] (`s_load_b64 s[24:25],s[8:9]`, `s_add_u32 s8,s8,8`),
  multiplies by the next older token (index row+2-k, `s14` decremented per pass) and
  `s_xor_b64 s[16:17], s[18:19], s[16:17]`. h = xor over k of B[k] * tok[row+2-k].
- Out-of-segment substitution: `s_cmp_ge_i32 s21, s20` where
  `s21 = row - segstart[row+2] + 2` (`s_load_b32 s21,s[6:7],0x8`, `s_sub_i32 s21,s2,s21`,
  `s_add_i32 s21,s21,2`) plus `s_cmp_gt_i32 s14,-1`; if either fails the token is
  replaced by kernarg 0x28 (`s_load_b32 s12,s[0:1],0x28`, `s_mov_b64 s[18:19],s[12:13]`),
  a boundary/pad id.
- Tail: `global_load_b64 v[3:4], v11, s[10:11]` (per-lane value from kernarg 0x18, used
  as the signed 64-bit modulus through the `v_rcp_f32`/`v_mad_u64_u32` expansion) and
  `global_load_b64 v[5:6], v11, s[6:7]` (per-lane value from kernarg 0x20, added after
  the modulus: `v_add_co_ci_u32 v1, null, v6, v1`), then `s_lshl_b64 s[2:3], s[2:3], 7`
  and `global_store_b64 v11, v[0:1], s[0:1]`: out row stride 128 B, 8 B per lane,
  id = (h mod A[lane]) + C[lane]. Kernarg 0x18 = vocab sizes and 0x20 = offsets is
  inferred from use (modulus, additive base) and from the loader names below.

Host evidence:
- `work/decomp/flash_serve_0172/018876b0.c:2364` `"ple_embedding.ngram_heads_vocab_sizes",0x25`
  and `:2383` `"ple_embedding.ngram_heads_offsets",0x21`, each followed by
  `FUN_01898900(param_1,&local_170,0x10)`: the trailing 0x10 is 16 (inferred to be the
  expected element count, one per lane). These strings are absent from the 0.17.3
  `flash_serve` decompile by grep (`ngram_heads`, `heads_offsets`, `heads_vocab`: no hits
  in `work/decomp/flash_serve/*.c`); 0.17.3 shows only the kernel registration
  (`01968680.c:3638`, `k_ngram_ids` string, stub `PTR_FUN_019f83a8`).
- `01904930.c` (`FUN_01904930`, ngram_early_launch, near the end): `uVar10 = lVar11 *
  0x10; plVar2[9] = uVar10;` and `plVar2[6] + lVar11 * 0x80`: 16 ids per token, 0x80
  bytes per token, matching the kernel's 128 B row stride and 16 lanes.

Remains: the real vocab_sizes and offsets values (checkpoint data, not in the
decompile), which would show the per-head table partition; the 0.17.3 launch of
`k_ngram_ids` (the stub pointer's relocation target is zero in the file, so the stub
address was not resolved); lane-to-head ordering inside an order.

## U18: k_dv_quant8 block layout and argument roles; consumer of the int8 q rows

Verdict: PARTIAL. Block shape and compute are pinned; byte placement of the output and
the host launcher are not. One earlier claim is corrected below: the host keeps f32 q
rows, not int8.

Pinned (`k_dv_quant8` slice, signature bf16*, int*, int*, int, uint8*, int):
- One thread per 32-value block, 1-D grid. Kernarg 0x28 (arg5) is `s3`; lines 9-15 do
  `s_lshr_b32 s5,s8,27`, `s_add_i32 s2,s3,s5`, `s_ashr_i32 s9,s2,5`: s9 = arg5/32 =
  blocks per row. Kernarg 0x18 (arg3, `s_load_b32 s6`) multiplies it: `s_mul_i32 s4,s9,s6`
  (line 18) is the thread bound (arg5/32)*arg3. The thread id is split with a 64-bit
  divide (reciprocal expansion, lines ~25-150) into (row, block-in-row). Inferred roles:
  arg3 = number of rows, arg5 = row length in elements.
- Input: 4 x `global_load_b128` at offsets 0/16/32/48 (lines 248-251) = 64 bytes = 32
  bf16 per block. One `global_load_b32 v0,v[0:1],off` (line 256) reads an int through one
  of the two `int*` args (the row map); it feeds `v_mad_i64_i32 v[1:2], null, v0, s2,
  s[0:1]` (line 346): out base + map_value * stride. So an int row map picks the
  destination row. Which of arg1/arg2 is source map versus destination map is not
  resolved.
- Math: absmax of 32 values, `v_div_fixup_f32 v8, v2, 0x42fe0000, v1` (line 344) =
  absmax/127, then `v_cvt_f16_f32 v0, v8` (line 347): the scale is IEEE **f16**. The 32
  quants are 32 `global_store_b8` at offsets 0..31 from one base (lines 409-1235). The
  scale is one `global_store_b16 v[1:2], v0` (line 1236) at `v_lshlrev_b32 v5, 1, v5`
  (2 bytes x block index) added to the mapped row base. So layout is split planes per
  destination row: f16 scales at 2*block, int8 quants at 32 B per block from another
  base. The plane start inside a row (the s2 stride and the v[3:4] address) was not
  decoded.

Host evidence for the consumer:
- `018f9fb0.c:5-18` (`FUN_018f9fb0`, q_keep): `if ((7 < param_2) || ...)
  fprintf(...,"q_keep: row %d outside 0..%d\n",(ulong)param_2,7);` then
  `hipMemcpyAsync(*(long *)(param_1 + 0x60) + (ulong)(param_2 * 0x3ca00) * 4,
  *(long *)(param_1 + 0xd80),0xf2800,3,0)`. Kind 3 is device to device; the size 0xf2800 =
  993,280 bytes = 248,320 f32 (0x3ca00) into slot `param_2` of an 8-slot buffer. Nothing
  quantizes here. 248,320 is the logit row width (vocab, inferred); the kept q row is a
  full f32 row.
- `019098d0.c:20-25` `"draft_sample: q row %d was not kept"` and
  `lVar5 = (ulong)param_8 * 0xf2800 + *(long *)(param_2 + 0x60)`: draft_sample reads the
  kept f32 slot (or the live row `param_2+0xd80` when the slot index is negative) and
  passes it to the sample launcher `FUN_01908c30`. `0190aa50.c:28` residual_sample has the
  same guard text.
- So the statement in `analysis/kernels-speculation.md` section 4 that the engine keeps a
  quantized copy of the q rows is not supported by the host code read: the kept rows are
  f32 and are consumed by draft_sample and residual_sample. Whether `k_dv_quant8` and
  `k_dv_quant` are launched in 0.17.3 is untraced: `k_dv_*` shows only as registration
  strings in `01968680.c`, and the stub addresses were not resolved (same limit as U17).
- done-line `dv_tokens, dv_misses, dv_switches`: the done line is built in
  `0184c9a0.c:99-103`, a `snprintf` of `"D %ld %s ... %.1f %ld %ld %ld %ld"`. The last
  three `%ld` args are `uVar11, uVar10, uVar14`, loaded from `*(param_13 + 0xa0)`, `+0xa8`,
  `+0xb0` (`0184c9a0.c:90-95`, zero when `param_13 == 0`). That fixes the stats-object
  offsets of the three counters (inferred mapping: field order equals dv_tokens,
  dv_misses, dv_switches, from wire-protocol.md section 3.3 positions 27-29). The writers
  of those fields were not found, so what a dv miss or switch counts is open.

Remains: stub addresses and launch args of `k_dv_quant8`; which int arg is which; the
plane offsets; writers of stats offsets 0xa0/0xa8/0xb0.

## U24: float-to-sortable-key plus byte extraction in the sampling kernels

Verdict: CLOSED at kernel level for `k_accept_prob_f32` (read for this section).
`k_sample_f32` and `k_resid_sample_f32` were not re-read; for them it is inferred from
the identical key transform. Purpose: an LDS radix select whose digit scan stops on a
cumulative count (top-k) or a cumulative probability mass (top-p, nucleus), with the mass
in integer fixed point so the pick is deterministic.

Evidence (`k_accept_prob_f32` slice, lines 548-660):
- Key: `v_not_b32 v14,v15`; `v_or_b32 v16, 0x80000000, v15`; `v_cmp_gt_i32 vcc_lo, 0,
  v15`; `v_cndmask_b32 v14, v16, v14, vcc_lo`: key = x < 0 ? ~x : x | 0x80000000 (orders
  as unsigned). `v_xor_b32 v16, v14, v13` then `v_cmpx_gt_u32 0x1000000, v16`: only
  elements whose top 8 key bits equal the prefix held in v13 continue. This is the
  "match the digits already chosen" filter of an MSD radix select.
- Digit: `v_bfe_u32 v14, v14, 16, 8` (bits 16-23), then `v_lshlrev_b32 v17, 2, v14` and
  `v_lshlrev_b32 v18, 3, v14`: the digit indexes a 256-bucket histogram in LDS.
- Two histograms per bucket: `ds_add_u32 v17, v5 offset:3072` (element count) and
  `ds_add_u64 v18, v[14:15]` (u64 mass). The mass is `v_sub_f32 v15,v15,v10` (logit minus
  max), `v_mul_f32 v15,v8,v15` (x 1/T), `v_mul_f32 v15, 0x3fb8aa3b, v15` (x log2 e),
  `v_exp_f32`, `v_fma_f32 v15, 0x56000000, v15, 0.5`, `v_trunc_f32`, a two-limb split with
  `0x2f800000`/`0xcf800000`, two `v_cvt_u32_f32`: round(exp((x-max)/T) * 2^47) as u64.
  Integer adds are order independent, so the result does not depend on thread timing.
- Scan: after `s_barrier`, a loop starting at `s_movk_i32 s14, 0x100` counts down 256
  buckets (`s_add_i32 s14,s14,-1`), accumulating bucket counts and masses
  (`ds_load_b32 ... offset:4108/4092`, `ds_load_b64 ... offset:4096/2040`). It stops on
  `s_cmp_le_u32 s13, s29` (cumulative count reaches s13, gated by s12 through
  `s_and_b32 s34,s12,s34`) or `v_cmp_ge_u64 vcc_lo, v[3:4], v[1:2]` (cumulative mass
  reaches a target, gated by s28). Two stop rules, one on count and one on mass: top-k and
  top-p. The chosen digit is shifted back (`s_lshl_b32 s14,s14,16`), OR-ed into the prefix
  (`v_or_b32 v3, s14, v13`) and stored to LDS. This pass resolves bits 16-23 with bits
  24-31 already fixed, so the threshold has 16-bit key resolution (sign, exponent, 7
  mantissa bits: the bf16 grid), inferred.
- Not decoded: which of s12, s13, s28 come from which `SampleRow` or kernel fields
  (kernarg loads at slice lines 4-5: `s_load_b256 s[16:23]`, `s_load_b128 s[24:27],
  s[0:1],0x20`).
- Host: `019098d0.c` builds a 32-byte SampleRow: `local_68[0] = param_1 * DAT_019fe1e8`
  (temperature times HALOGEN_S2_TSKEW, `01909e00.c`), `local_68[1] = 0`, `local_60 =
  (uint)DAT_019fe1c8` (HALOGEN_S2_NUCLEUS, `01909da0.c`; offset 8), `local_58 = param_3`
  (offset 0x10), param_4/param_5 at 0x18/0x1c, sent by `hipMemcpy(...,0x20,1)`. The
  warning text there names nucleus, qoff and tskew together as the falsifier knobs
  ("FALSIFIER ACTIVE: nucleus %.4f, qoff %+.4f, tskew %.4f"). Inferred: SampleRow+8 is a
  nucleus float, 0 = off (`-(uint)(0.0 < DAT_019fe1c8)` builds the enable mask), which fits
  the mass-stop rule. In this draft_sample path SampleRow+0xc is 0, so the
  `s15 = SampleRow+0xc` seen in `k_smpg_accept_final` is not a nucleus value (U23 stays
  open).
- Launch: `01908c30.c:21-24` pushes grid (n,1,1), block 0x100 (256 threads) for the
  sampler, consistent with a 256-bucket histogram.

Remains: field map of SampleRow into s12/s13/s28; whether the non-falsifier sampler feeds
a real top-p or top-k through the same fields; line-level confirmation in `k_sample_f32`
and `k_resid_sample_f32`.

## U25: what k_dv_quant quantizes to

Verdict: CLOSED for the codebook (inferred identity: NF4, 16 levels, nearest level by a
midpoint compare ladder). Block size and storage layout not decoded.

Evidence (`k_dv_quant` slice, signature `EPKtPKiS3_iPhli`): `v_cmp_lt_f32` against
literals uses exactly 15 distinct thresholds (grep counts 64, 64, 63, 60, 53, 52, 51, 51,
49, 47, 46, 46, 44, 42, 42; sum 775 = the known total). Decoded from the hex (bf16-exact,
low half 0x0000 or 0x8000):

| hex | value |
|---|---|
| 0xbf590000 | -0.84766 |
| 0xbf1c0000 | -0.60938 |
| 0xbeeb0000 | -0.45898 |
| 0xbeae0000 | -0.33984 |
| 0xbe708000 | -0.23486 |
| 0xbe0d0000 | -0.13770 |
| 0xbd3a0000 | -0.04541 |
| 0x3d230000 | +0.03979 |
| 0x3df68000 | +0.12036 |
| 0x3e508000 | +0.20361 |
| 0x3e958000 | +0.29199 |
| 0x3ec78000 | +0.38965 |
| 0x3f008000 | +0.50195 |
| 0x3f248000 | +0.64258 |
| 0x3f5c8000 | +0.86133 |

(Decoded by hand from the IEEE fields; no tool computed them.) 15 thresholds split
[-1, 1] into 16 cells, so the output is a 4-bit index. The thresholds equal the midpoints
between adjacent levels of the NF4 (NormalFloat4) codebook (-1, -0.6962, -0.5251, -0.3949,
-0.2844, -0.1848, -0.0911, 0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1)
to bf16 rounding: (-1 + -0.6962)/2 = -0.848 vs -0.8477; (0.5626 + 0.7230)/2 = 0.6428 vs
0.6426; (0.7230 + 1)/2 = 0.8615 vs 0.8613. The NF4 list is from memory of the published
QLoRA table, not re-fetched, so the identity stays labelled inferred even though all 15
values match. A compare ladder instead of round-and-clamp is explained: a non-uniform
codebook has no closed-form index.
- Scale: `v_cvt_f16_f32 v39, v39` (slice line 356): f16 block scale. Divides (64 IEEE
  divides, earlier count) bring values into [-1, 1], so the scale is the block absmax
  (inferred; 16 `v_max3_f32` do the absmax).
- Staging: 4 `ds_store_b128` (lines 318-322) and many `ds_load_b32` through the body; the
  nibble pack goes through LDS (two 4-bit codes per byte, inferred from the `uint8*`
  output and the 4-bit cell count).

Remains: block size (64 divides suggest 64 values per thread, inferred, unchecked), nibble
order within a byte, scale placement in the output, and whether the host launches it in
0.17.3 (same stub-resolution limit as U18).
