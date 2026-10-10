# Indexer (sparse-attention) kernel family — objdump-level analysis

Phase 2, indexer path. Evidence: per-kernel slices in `work/kernels/`
(llvm-objdump 21 `--mattr=+gfx1151`), AMDGPU metadata from the four code
objects (`llvm-readelf --notes work/fatbin/*.hsaco`), demangled signatures
(`c++filt`), changelog 0.12.0/0.9.1 (`work/CHANGELOG.md`), flag strings in
`analysis/strings-tools.txt`.

## 0. Family map

The model is Qwen3.8-Flash-Next: "QSA with its micro-block indexer" (0.1.0
changelog). Each indexer layer scores every **4-token block** of the context
and keeps the **top 512** blocks (2,048 tokens; `HALOGEN_INDEXER_BUDGET`,
default 512 blocks, range 2048–8192 tokens, rounded to 16 — 0.9.1).

Three stages, three kernel groups:

| stage | kernels | code object |
|---|---|---|
| score (Q·K per block) | `k_ix_scores_loop`, `k_ix_scores_wmma` | `codeobj_1513000.hsaco` (the attention object: 62 kernels, all `k_attn_fa*`, `k_attn_qs*`, `k_kv_bt`, + the 2 ix kernels) |
| select (top-k over a score row) | `k_dnc_scan`, `scan2`, `scan3` (104 variants), `scan3t/w/x`, `scan4`, `scan5`, `k_dnc_prep(_w)`, `k_dnc_cmp_words`, `k_dnc_ut/2/3/4/5` | `codeobj_1146000.hsaco` (175 kernels — the whole dnc family lives here) |
| attend over selected blocks | `k_dnc_att`, `att2`, `att6<16/32/64,3 bools>`, `att7<32,{2,4,8}>`, `att8`, `att9<{0,4,8,12,16,-4,-8}>` | `codeobj_1146000.hsaco` |

`dnc` = the select+attend pipeline ("dense-neighbourhood cache" reading of the
letters is a guess; the family is unambiguously the indexer's block-select +
sparse attention: signatures match the changelog's description exactly).

Flags naming the path (halogen-tools strings): `HALOGEN_IX_SELECT`,
`HALOGEN_IX_WMMA`, `HALOGEN_IX_WMMA_POISON` — the last is the same
negative-control pattern as `HALOGEN_S2_*`/`HALOGEN_S3_FALSIFY` (report.md §3):
a switch that poisons the WMMA scoring path so the harness can prove the
select's block lists are identical with the matrix-unit path sabotaged.

## 1. Scoring: `k_ix_scores_loop` / `k_ix_scores_wmma`

Signatures (demangled):

```
k_ix_scores_loop (const bf16* Q, const bf16* K, float* scores, long stride,
                  int n, int m, int a, int b, int c)   // 5 ints
k_ix_scores_wmma (const bf16* Q, const bf16* K, float* scores, long stride,
                  int n, int m, int a, int b)          // 4 ints
```

Metadata (codeobj_1513000):

| kernel | wg | vgpr | LDS |
|---|---|---|---|
| `k_ix_scores_wmma` | 128 | 219 | 16,640 B |
| `k_ix_scores_loop` | 256 | 147 | 33,280 B |

Instruction mix (both): 32× `v_wmma_f32_16x16x16_bf16`, 64× `ds_load_b128`,
20/8× `global_load_b128`, `v_max_f32`+`v_dual_max_f32` (39/48),
`v_rsq_f32` (loop variant), **no `v_exp`**, no atomics.

Reading: bf16 in / f32 scores out; 16×16×16 WMMA tiles with block keys
staged in LDS; the max chain is the per-row running max over the produced
scores (top-k bookkeeping seed), `v_rsq` in the loop variant folds the query's
RMS-norm scale into the scores. Pure dot-product scoring — no softmax —
matching "the indexer scores every 4-token block".

The 0.12.0 scoring rewrite ("keeps 128 block keys per workgroup in registers
and walks the query rows, with the same products in the same order") is
visible in the shape: `wmma` is the register-resident variant — 219 VGPRs
(128 keys × bf16 fragments in registers), one 11-instruction prologue loop
and one 7-instruction tail loop, i.e. the key set is loaded once and the
kernel is almost fully unrolled over the 16×16 tiles; `loop` is the old
re-read shape — a 505-instruction main loop (`0x48c→0x1458`) that walks the
key set per query strip, 256 threads, 33 KB LDS. Same WMMA op, same tile
size, same bf16 inputs → "same products in the same order" is structurally
true: identical 16×16×16 bf16→f32 accumulation, only the loop nest moved.

## 2. Select: the `k_dnc_scan*` family

Signatures (all 12-arg scan variants share this shape):

```
k_dnc_scan3<8,8,8,true,10,4,0>(const bf16* keys, const f32* s0..s6,
    float* out, const int* idx, long n, bf16* sel, int k)
k_dnc_scan3x<...>(same 12 + 4 extra f32/bf16 pointers
    + DncTapArgs (40 B by value) + DncNormFold (40 B by value))
```

Metadata (codeobj_1146000):

| kernel | wg | vgpr | LDS | notes |
|---|---|---|---|---|
| `k_dnc_scan` (old) | **1024** | 73 | 21,824 | one fat workgroup per row |
| `k_dnc_scan3<8,8,8,true,10,4,0>` | **128** | 82 | 17,696 | bisection |
| `k_dnc_scan3t<8,8,8,true>` | 128 | 133 | 17,536 | tree-reduction variant |
| `k_dnc_scan3w<2,4,true,false,0..8>` | 128 | 161 | 47,104 | WMMA/f16 variant |
| `k_dnc_scan3x<2,4,...>` | 128 | **255** | 61,952 | tap/norm-fold variant |

Key structural facts:

* **No atomics, no ballot, no cross-workgroup sync anywhere in the scan
  family** (grep for `atomic|ballot|readfirstlane` = 0 hits in scan/scan3/scan3t;
  1 `v_perm`-class hit in scan3w/x). All cross-wave communication is
  LDS + `s_barrier` (8 barriers in scan3, 51 in scan3t) with
  `s_and_saveexec_b32`/`s_and_not1_b32` exec-masked segmented loops. The
  select is therefore **deterministic by construction**: fixed work partition,
  fixed reduction order, no data-dependent scheduling. This is the kernel-level
  ground for the changelog's "the same blocks, the same tie rule, the same
  order" (0.12.0) and "byte-identical output" acceptance.
* **Bisection over a threshold**: the scan3 main loop (`0x3b0→0x2a5c`,
  1,237 instructions, nested 24–29-instruction inner loops) loads a scalar
  threshold candidate `s10` via `s_load_b32`, computes `v_sub_f32 s10, v29`
  (τ − score), then the classic **exp2 Cody–Waite sequence** —
  `0x3fb8aa3b` = log2(e), `0x32a5705f` = 2.2146721e-8, `v_rndne_f32`,
  `v_cvt_i32_f32`, `v_exp_f32` (= exp2 on RDNA) — and accumulates with
  170× `v_fmac_f32` + 131× `v_fma_f32`. That is Σ exp(score − τ): a soft
  count of scores above τ, refined by bisection on τ (the 24-instruction
  inner loops are the slice-wise partial counts; 320× `ds_load_2addr_b32`
  reads score pairs from LDS). Matches "register-resident bisection over
  slices of the row, spread across the GPU" (0.12.0) — 128-thread workgroups
  (vs the old 1024-thread one-workgroup-per-row `k_dnc_scan`, whose slice
  count is the whole visible block list).
* **Template grid**: `scan3<A,B,C,D,E,F,G>` with E ∈ {1,2,3,8,10,12,14,16,18,20}
  × F ∈ {1,2,4,8,15} (104 instantiations) = slice-count × slice-width
  dispatch: the grid picks the bisection geometry by row length (decode =
  one row per layer spread across slices; prefill = thousands of rows).
  `scan3w<2,4,true,false,P>` P ∈ 0..8 = 9 slice positions; `scan3w` is the
  WMMA variant: 76× `v_wmma_f32_16x16x16_f16` + 673/657×
  `v_fma_mixlo/hi_f16` — the slice histograms are accumulated on the matrix
  units in f16.
* **Output encoding**: scan3w/scan3t/scan3x write the selected set with
  `global_store_d16_hi_b16` after `mul 0x3b800000 (=1/256)`, `v_bfe bit16`,
  `add 0x7fff` — round-to-nearest bf16 of value/256, stride `0x3000`
  (3,072 f32 lanes) — i.e. the block list is written as bf16-packed
  indices/keys into a strided arena, not as raw ints. The `int* idx` +
  `bf16* sel` + `int k` tail args carry the count and the k (budget).
* `scan3x` adds 4 pointers + `DncTapArgs`/`DncNormFold` 40-byte by-value
  structs (names recovered from the mangled metadata in codeobj_1146000):
  the newest variant folds the per-block norm scale and extra taps inside
  the select, 255 VGPRs, 112 WMMA f16 ops, a 3,195-instruction main loop.
* `k_dnc_cmp_words(const u32* a, const u32* b, long n, u64* out)` —
  word-wise key comparison with hidden grid dims (HIP grid-gem); the
  "block-key read" comparator used by the select path (768 MiB/step at 1M
  in the changelog's roofline note).
* `k_dnc_ut/2/3/4/5` (ut5 = 12 args, 60 instantiations) — the update/commit
  kernels of the same pipeline (write selected keys, fold norms).

## 3. Attend over selected blocks: `k_dnc_att*`

Signatures: `k_dnc_att(const f32* Q, const f32* K, const f32* V, f32* out,
int n)` (att6 adds `bf16* out2, const bf16* idx, const f32*, const f32*`).
Metadata: att wg=1024/LDS 33 KB; att2 wg=1024/LDS 41 KB; att8 wg=256/LDS 37 KB
(8× `v_wmma_f32_16x16x16_f16` — matrix-unit attention over the selected set);
att6 wg=256 (16/32/64 = block width, 3 bool flags = bf16-out / rope-fold /
norm-fold variants); att9 has **negative** template args (`<-4>, <-8>`) =
backward window offsets.

All att kernels carry the exp2 Cody–Waite constants (`0x3fb8aa3b`,
`0x32a5705f`, clamp `0x42b17218` = 88.0 = exp-overflow guard) → standard
online softmax over the selected blocks only. `k_dnc_att` strides rows by
48 f32 (`v_mad_u64_u32 …, 48`) = 12 selected 4-token blocks × 4 tokens per
tile, i.e. the kernel consumes the select's output directly.

`k_dnc_prep<3 bools>`: `v_rsq_f32` (RMS-norm), 20 barriers, wg=1024, LDS 516 B
— per-token Q/K/V preparation feeding the pipeline; `k_dnc_prep_w` the
WMMA-weighted variant.

## 4. What this says about the 0.12.0 "byte-identical indexer rewrite"

1. **Both generations ship in the same code object** (old `k_dnc_scan`,
   wg=1024, and the 104-variant `scan3*` family, wg=128, in
   codeobj_1146000; old `k_ix_scores_loop` and new `k_ix_scores_wmma` in
   codeobj_1513000). The engine can dispatch either; the changelog's
   bit-for-bit comparison of "block lists and scores" is exactly what this
   pairing enables, and `HALOGEN_IX_WMMA_POISON` is the sabotage switch for
   the negative control.
2. **Determinism is structural**: no atomics, no ballots, fixed LDS+barrier
   reductions, fixed WMMA tile order, identical bf16 inputs. The select's
   bisection counts with exp2(fma) chains in a fixed order; the scorer's
   accumulation order is the WMMA tile order, unchanged between variants.
   Nothing in the ISA sequence can reorder arithmetic between the two
   generations.
3. **The perf story is visible in the shapes**: decode's serial
   one-workgroup-per-row walk (1024-thread scan over the whole row) is
   replaced by 128-thread slice bisection spread across workgroups; the
   scorer's per-16-row key re-read (505-instruction loop) is replaced by
   219-VGPR register-resident keys.

## 5. Open items

* Exact semantics of `scan3` template args D/E/F/G and of `DncTapArgs` /
  `DncNormFold` field layout (40 B each) — needs host-side dispatch
  decompile in `flash_serve` (the launcher computes grid dims per row
  length; not yet located).
* `k_dnc_ut4/ut5` 60-variant flag meanings (bool flags + int 0..3).
* Which att variant serves decode vs prefill at 0.17.3 (dispatch table in
  flash_serve not traced; `HALOGEN_ATTN_*` flags in report.md §2 are the
  selectors).
* The letters `dnc` themselves (family naming) remain a guess.
