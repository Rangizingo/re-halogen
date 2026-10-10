# Speculation / sampling / acceptance kernel family: objdump-level analysis

> **Corrections (2026-10-10, after host-side passes, see analysis/closure/sampling-host.md and sampling-host-r2.md).** (1) `SampleRow` offset 0xc is `min_p`, not a host-supplied uniform; section 1's reading of `s15` as a uniform threshold is withdrawn. (2) The `mode` argument of `k_smpg_accept_final` (the `s13 == 3` compare) is the value of `HALOGEN_S3_FALSIFY` (default 0), not a greedy-versus-sampled switch. (3) The on-GPU splitmix64 draw in `k_smpg_pick` and `k_smpg_rpick` is derived from the per-request seed; the copy from the parsed wire seed into the params is not yet located (U23). (4) `k_dv_quant` uses a codebook compare ladder, read as NF4 (inferred), see closure/ngram-dv-host.md.

Phase 2, family 5 of 5 (speculation). Evidence: 43 per-kernel slices in
`work/kernels-spec/` (llvm-objdump 21, `--mattr=+gfx1151`), instruction-class
counts in `tmp/stats-spec.csv`, demangled signatures (`c++filt`), host strings
in `analysis/strings-tools.txt`. Method: rank by stats, grep instruction
classes, read targeted windows. Line numbers below are in the slice files.
`work/kernels-spec/` also holds 21 `k_dn_fused16` variants; those belong to
`analysis/kernels-moe.md` and are not covered here.

## 0. Family map

| group | kernels | what it does |
|---|---|---|
| sampler | `k_sample_f32`, `k_resid_sample_f32`, `k_q_softmax_f32`, `k_sigmoid(_f32)` | draw a token from a logit row; draw from the residual distribution after a rejection |
| acceptance | `k_accept_prob_f32`, `k_smpg_accept_final` | per-position accept/reject of a drafted token |
| SMPG graph sampler | `k_smpg_max/mass/level<0..3>/pick/rmass/rpick/qwrite` | multi-level reduction (max, mass, level search, pick) used by sampled speculation; `r*` variants work on the residual |
| penalties | `k_pen_rep_rows_f32`, `k_pen_scatter_rows_f32` | repetition penalty on logit rows |
| n-gram | `k_ngram_ids/hist/segstart/ctx_update` | hashed n-gram table ids (section 3) |
| draft-verify | `k_dv_quant`, `k_dv_quant8`, `k_dv_scatter` | block int8 quantize, row scatter (section 4) |

Sizes (stats CSV, instructions): `k_dv_quant` 6065, `k_resid_sample_f32` 1710,
`k_sample_f32` 1312, `k_dv_quant8` 1263, `k_accept_prob_f32` 1167,
`k_smpg_accept_final` 345, `k_ngram_ids` 265, `k_dv_scatter` 147.

Signatures:

```
k_accept_prob_f32 (const float* logits, const int* draft_ids, const SampleRow* rows,
                   float* p_out, float* q_out, int n, int m)
k_sample_f32      (const float* logits, const SampleRow* rows, int* tok, float* a, float* b, int n)
k_resid_sample_f32(const float* p, const float* q, const SampleRow* rows,
                   int* tok, float* a, int* c, int n, int m)
k_smpg_accept_final(const float*, const int* draft_ids, const SampleRow*, const SmpScratch*,
                    float* accept, float* logratio, int, int, int)
```

## 1. Acceptance is probabilistic, not a greedy argmax compare

`k_smpg_accept_final` (slice lines 255-348) ends with this chain, in order:

1. `v_log_f32` of a probability (scaled by `0x3f317217` = ln 2, with the
   denormal pre-scale `0x4f800000`), i.e. a natural log.
2. `v_sub_f32` of two logits, `v_mul_f32` by a reciprocal temperature
   (`v_div_fixup_f32 v5, ..., s22, 1.0` is `1/T`), then `v_mul_f32` by
   `0x3fb8aa3b` (log2 e) and `v_exp_f32`: exp((logit_a - logit_b)/T).
3. `v_div_scale/v_rcp/v_div_fmas/v_div_fixup` full-precision divide of that
   exponential by a second value: the ratio p/q.
4. `v_cmp_nle_f32 vcc, s15, v0`: compare the ratio to `s15`, loaded from the
   `SampleRow` at byte offset 0xc (`s_load_b32 s15, s[2:3], 0xc`, line ~268).
   Result selects between the ratio and 0 (`v_cndmask_b32 v3, v2, 0, s0`).
5. The token-id compare `s_cmp_lt_u32 s0, s2` ORs into the same predicate
   (reject when the drafted id is outside the row's accepted range).
6. Two `global_store_b32`: an accept value (1.0 when mode `s13 == 3`) and the
   log-ratio.

The chain computes exp((a - b)/T), divides it by a second value, then compares
the exp numerator `v0` with `s15` (`v_cmp_nle_f32 vcc, s15, v0`, slice line
306). `s15` comes from `s_load_b32 s15, s[2:3], 0xc` (slice line 15) on the
32-byte `SampleRow` (row index shifted by 5; field 0 feeds the `1/T` divide).
This has the shape of a speculative-sampling accept test: accept when a
probability clears a threshold. That it is exactly min(1, p/q) against a
uniform draw is inferred: this kernel has no explicit min with 1.0 and no
uniform generation, and the compare operand is the exp numerator, not the
ratio. The kernel also has an early fall-through that stores 1.0 when the mode
argument `s13 == 3` (line ~300); what the other mode values mean was not
traced, so which mode is greedy is open. The greedy commit itself (trunk
argmax) is the falsifier-gate argument in `analysis/report.md` section 3, not
re-derived here. Counts of `v_exp_f32` / `v_log_f32` (grep; the stats CSV
`exp` column adds the two): `k_accept_prob_f32` 7/1, `k_resid_sample_f32` 9/1,
`k_sample_f32` 7/2, `k_smpg_accept_final` 1/1 (log at slice line 270, exp at
284), `k_smpg_pick` 1/2.

**There is an in-kernel RNG, in the pick kernels, not in the accept kernel.**
`k_smpg_pick` (slice lines 184-212) and `k_smpg_rpick` (lines 234-264) contain
the splitmix64 constants 0x9e3779b97f4a7c15, 0xbf58476d1ce4e5b9 and
0x94d049bb133111eb (12 hits each by grep, scalar `s_mul_i32`, `s_mul_hi_u32`,
`s_xor_b64`, shifts 30 and 27). The first pass of this writeup grepped vector
ops only and missed them; the independent review caught it. So the sampled draw
is generated on the GPU from a seed. Update 2026-10-10 (U23 CLOSED, `closure/sampler-seed-r3.md`): the seed is
SampleRow+0x10, copied unchanged from the wire; the counter at +0x18 is the
token position; `s15` in the accept kernel is SampleRow+0xc, min_p clamped.

The other xors are not a divide expansion either. `k_smpg_level<1..3>` each
have one non-sign xor (level1 slice line 697). `k_accept_prob_f32` (lines
555-567), `k_sample_f32` and `k_resid_sample_f32` carry a float-to-sortable-key
transform: `v_not_b32`, `v_or_b32 0x80000000`, a sign compare, `v_cndmask_b32`,
`v_xor_b32` (key = x < 0 ? ~x : x | 0x80000000), and `k_accept_prob_f32` then
extracts a byte with `v_bfe_u32 ,16,8`. That is a radix-style select over float
keys, probably a top-k, top-p or sorted search (U24, purpose not traced).

Consequence for "speculation can't change output": the kernels alone do not
prove it. Greedy: see report.md section 3. Sampled: accept-by-probability plus
residual resample (`k_resid_sample_f32`) is the standard distribution-preserving
scheme and the instruction chain is consistent with it (inferred). Bitwise
reproducibility of sampled runs rests on the splitmix64 seed source (U23,
now closed: seed plus token position; same-seed repeats matched in run 2).

## 2. Sampler / SMPG structure

- `k_sample_f32` and `k_resid_sample_f32` are row-wise: a max pass, an
  exp-sum pass, an inverse-CDF scan against the row's uniform. They carry
  LDS tree reductions (110 and 156 `ds_*` ops) and 4 `v_cmp_ge_u64` each from
  the 64-bit index math.
- `k_smpg_*` splits the same work for long rows into stages through the
  `SmpScratch` buffer: `max` (254 instrs, 9 stores, no exp) finds the row
  max; `mass` and `level<0..3>` compute block masses at four granularities;
  `pick`/`rpick` walk down the levels to the chosen index; `qwrite` writes the
  draft distribution row (`k_q_softmax_f32` is the plain-softmax variant).
  The four `level` instantiations differ only in the stage constant (393,
  739, 762, 748 instrs).
- `k_pen_rep_rows_f32` and `k_pen_scatter_rows_f32` have no transcendental
  ops (`exp` 0, `rcp` 0 for scatter): repetition penalty is a gather,
  multiply/divide by a per-row factor, store.

## 3. n-gram kernels: id hashing, nothing else

All four take `long long` token arrays and write `long long` ids.

- `k_ngram_ctx_update` (59 instrs): one thread, writes the last two tokens of
  the history as a 16-byte pair (`global_store_b128`) to a context slot.
- `k_ngram_segstart` (30 instrs): scans token ids for the segment-boundary
  value (`s_cmp_eq_u64`) and writes the running segment index.
- `k_ngram_hist` (69 instrs): builds the shifted token history, substituting
  boundary values for out-of-segment positions.
- `k_ngram_ids` (265 instrs): 16 lanes (`v_cmpx_gt_u32 16, v0`), one 8-byte id
  each, row stride 128 B (`s_lshl_b64 ..., 7`). Per lane a scalar
  multiply-xor fold over the window (`s_mul_i32/s_mul_hi_u32` 64-bit multiply,
  `s_xor_b64`, lines ~40-60), then a signed 64-bit modulo (the `v_rcp_f32` +
  `v_mad_u64_u32` expansion) by a per-lane table size.

Float ops: zero `v_exp`, zero `v_fma`, zero LDS. The only float instructions
are the reciprocal seed inside the compiler's 64-bit modulo expansion
(`v_cvt_f32_u32`, `v_fmamk_f32`, `v_rcp_f32`, `v_mul_f32`, slice lines
102-112). Integer hash, modulo, store.
This matches the host strings `ple.ple_embedding.ngram_heads_offsets`,
`ngram_heads_vocab_sizes` and `HALOGEN_NGRAM_TABLE` (the 47.7 GiB lookup
file): the kernels produce per-head row ids into the n-gram embedding table;
the table rows are gathered by the host (`HALOGEN_NGRAM_GATHER_THREADS`).
So "the n-gram table is a pure lookup" holds at kernel level: hash, modulo,
index. No learned compute and no continuation match runs on the GPU in this
family. Which n-gram order each lane serves (16 lanes = heads x orders) is
inferred, not proven.

## 4. k_dv_* kernels: int8 block quantize and a row scatter

`k_dv_quant8(const bf16*, const int*, const int*, int, uint8*, int)`
(1263 instrs, no LDS):

- 4 `global_load_b128` bring in 32 bf16 values (shifted to f32 with
  `v_lshlrev_b32 16`).
- `v_max3_f32` chain over `|v|` (16 ops, 32 values): block absmax.
- `v_div_scale/v_div_fixup` by the literal `0x42fe0000` = 127.0: scale =
  absmax/127, an IEEE-correct divide, not `v_rcp`.
- per value: `v_div_*` by the scale, `v_rndne_f32` (round to nearest even),
  `v_minmax_f32` clamp to +/-127 (`0x42fe0000`, `0xc2fe0000`),
  `v_cvt_i32_f32`, then `global_store_b8` / `global_store_d16_hi_b8`.
- one `global_store_b16` for the scale.

This is symmetric int8 block quantization with a 16-bit scale, the same shape
as GGUF Q8_0 (32 values, 1 scale). The exact byte layout of the output block
and which of the two int arguments is the row map is not pinned down (bounded
open).

`k_dv_quant` (6065 instrs, `long` row stride) is not the same quantizer. The
independent review caught this and grep confirms: 36 `ds_` ops, 128
`v_div_scale` (64 IEEE divides), 775 `v_cmp_lt_f32`, 16 `v_max3_f32`, one
`v_cvt_f16_f32`, and no `v_rndne_f32` and no `v_cvt_i32_f32`. A compare ladder
in place of round-and-clamp points to a non-uniform codebook (nearest-level
search) with an f16 block scale. Unresolved (U25).

`k_dv_scatter(const float* src, const int* idx, float* dst, int n)` (147
instrs) is a gather despite the name: it loads `idx[i]`, tests
`v_cmpx_lt_i32 -1`, loads `src[idx[i]]` when the index is non-negative, and
stores `dst[i] = idx[i] >= 0 ? src[idx[i]] : -inf` (default `0xff800000`).

What "dv" stands for is not shown by the kernels. They first appear at 0.17.2
and 0.17.3 adds host guards `q_keep: row %d outside 0..%d` and "was not kept"
(`analysis/version-deltas.md`): the engine keeps a quantized copy of the
sampled "q rows" (the draft distribution) for the acceptance walk. That fits
the kernels: `k_dv_quant8` packs a q row to int8 blocks, `k_dv_scatter`
gathers rows by index (negative index = -inf), and `k_smpg_qwrite` /
`k_q_softmax_f32` produce the q rows. So the p/q ratio in section 1 reads q
from this int8 copy (inferred from names and release notes, the load of the
quantized rows inside the accept kernels was not traced).

## 4b. Do the falsify and sampled-speculation flags change the math?

Open question carried from `analysis/version-deltas.md`. Device-side answer:
no. 0.17.2 to 0.17.3 ships zero device-code change (code-object sha256 lineage
in version-deltas) while adding `HALOGEN_SAMPLED_DEPTH`, `HALOGEN_SAMPLED_GATE`
and `HALOGEN_WALK_FALSIFY` host strings, and no kernel in this family is
named for or branches on a falsify mode. The flags therefore act on the host:
which rows, uniforms and mode values are fed to the same kernels, and what is
audited. Whether the host feeds different acceptance inputs in falsify mode is
host-side and untraced (bounded open item 6).

## 5. Claims closed

| claim | verdict | evidence |
|---|---|---|
| greedy vs probabilistic acceptance | the accept kernel has the shape of a probability-threshold test (inferred to be min(1, p/q)); which mode value is greedy is open | `k_smpg_accept_final` log/exp/divide/compare chain, `s15` from `SampleRow+0xc` |
| in-kernel RNG | exists: splitmix64 in `k_smpg_pick` and `k_smpg_rpick`; seed source untraced | constants at pick lines 184-212, rpick 234-264 |
| n-gram table is a pure lookup | holds: integer hash plus modulo, only float ops are the divide's reciprocal seed | `k_ngram_ids` |
 holds for greedy (argmax commit); for sampled, by distribution preservation (inferred) | section 1 |

## 6. Bounded open

1. Exact `SampleRow` field map (offset 0xc = uniform is inferred from use).
2. Per-lane n-gram order in `k_ngram_ids`.
3. `k_dv_quant`/`k_dv_quant8` output block layout and argument roles.
4. What consumes the `k_dv_*` output (host dispatch not traced).
5. Seed and counter source of the splitmix64 draw and what `s15` holds: closed (U23, `closure/sampler-seed-r3.md`).
6. Host feeding of acceptance inputs under `HALOGEN_WALK_FALSIFY`.
7. Purpose of the float-key radix select in the accept and sample kernels (U24).
8. What `k_dv_quant` quantizes to (compare-ladder codebook, U25).
