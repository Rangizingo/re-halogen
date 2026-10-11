# Refuter round 2, PR #24 (p7-12-wave2, head f65b568)

Verdicts: claim A UPHELD, claim B UPHELD, claim C UPHELD. REFUTED: 0.

Checked out f65b568 detached in a private worktree; nothing committed to p7-12-wave2, nothing pushed, no PR comment. Repo edits: this report,
`rewrite/specs/refuter2/` (178 attack specs generated on the Mac, relative `rewrite_src`), `rewrite/reports/refuter2-pr24-files/` (generator and
runner scripts saved as `.txt` so the lint hook does not treat them as repo Python, plus `ai-core-logs-and-manifests.tgz`, the raw run logs and
manifests, 117 KB). Ai-core: own dir `~/refuter2/` only (`work/cmp` and `work/fatbin` symlinked to `~/re-halogen`; `CMP_TMP=~/refuter2/tmp`; the
`.service-down` marker was never created or removed; no service touched; only my own PIDs killed; no `pkill -f`). Peak about 14 of my own
compare.py streams at once with load average 14 on 32 cores; the orchestrator's compile jobs ran alongside.

Every run is `python3 -I rewrite/harness/compare.py <spec> --kernel <mangled> --seeds ... --sizes ... --no-table` (bitwise, no `tol`), driven by
`runman.sh` over TSV manifests (`rr.sh` / `rr2.sh` wrap it). The compare harness counts "agreeing on an output neither side wrote" as MISMATCH diff=0
(UNTOUCHED); those are reported separately and never counted as MATCH.

## Claim A: round-1 NaN-sign and attn_fa2 findings are fixed: UPHELD

### A1. Round-1 reproducers against the replacement files (all MATCH), with the original file as negative control

| family | replacement | reproducer | fixed | original (control) |
| --- | --- | --- | --- | --- |
| dnc_ut | dnc_ut_c.hip | round-1 `dnc_ut__nan.json`, seeds 31,32,33, N=2,5 | MATCH 6/6 | MISMATCH 6/6, diff=281 first_diff=15991 (round 1: 281 / 15991) |
| dnc_ut3 | dnc_ut3_c.hip | `dnc_ut3__nan.json` | MATCH 6/6 | MISMATCH 6/6, diff=211 first_diff=18372 (round 1: 211 / 18372) |
| ht_deq | ht_deq_c.hip | `ht_deq_orig_b3_f0_t256__nan.json`, N=3,5 | MATCH 6/6 | MISMATCH 6/6 |
| i4r_gu | i4r_gu_c.hip | `i4r_gu_epi__nan.json`, N=8,101 | MATCH 6/6 | MISMATCH 3/6 |
| vis_rope | vis_rope_c.hip | `vis_rope_b32__nan.json`, N=64,77 | MATCH 6/6 | MISMATCH 6/6 |
| ht_gemv dn_g | ht_gemv_b_c.hip | `ht_gemv_b_dn_g_3_8_2_{2,0,4}` with x and sv NaN, 0.4% and 2% density, 2 data seeds, N=200,40, seeds 31,32,33 | MATCH 216 of 216 runs (36 specs) | 12 of 12 "both" specs MISMATCH (first run `diff=3 max_ulp=65536`, the sign bit); x-only and sv-only specs MATCH on both files, as round 1 found |
| attn_fa2 A=2 | attn_fa2_d.hip | all 34 round-1 specs: `fa2_def_all`, `fa2A2c__*` constant-score scan, `sw_..._edge_{k,q,v}_*` | 34 of 34 specs MATCH (105 runs, seeds 1,2,3) | 13 of 34 MISMATCH on attn_fa2.hip (same ones as round 1: scan s=-1.1e29, -2e29, -5e29, -9.9e29; k/q 1e30; v inf, nan, 3.4e38) |

Commands (on Ai-core, `~/refuter2`): `bash a1.sh` (five families), `python3 -I ht1.py; bash runman.sh man_ht1.tsv 31,32,33` (ht_gemv),
`bash a2.sh` (attn_fa2). Excerpt, `logs/a1.log`:

```
[FIXED] == old/dnc_ut_fix.json : MATCH=6 other=0
[ORIG] == old/dnc_ut_orig.json : MATCH=0 other=6
MISMATCH dnc_ut_orig.json seed=31 N=2 n=1572864 diff=281 max_abs=inf max_ulp=4.29497e+09 first_diff=15991 tol=bitwise
```

The ht_gemv round-1 spec file referenced a data file that no longer existed; I regenerated x and sv fills with the same special list (+-inf, -0,
+-1e-40, +-3e38, NaN) and checked that the original ht_gemv_b.hip still fails on them, so the reproducer is live.

### A2. New attacks on the replacement files: no mismatch

All float input buffers of every committed MATCH spec of the target files get file fills: the original value distribution with a fraction of
elements replaced by a pick from a special-value zoo. The zoo per dtype has 12 NaN patterns (+-qNaN, +-sNaN, payloads, all-ones, 0x7fff-style),
+-inf, +-0, denormals (smallest, largest, 0x00400000 style), and +-max. Specs: `nanfuzz.py` (writes specs and data), manifests from the
match-table (`mkman*.py`). Runs listed as "rows" are one (spec, instance) pair run over 2 sizes x 2 seeds unless noted.

| attack | scope | result |
| --- | --- | --- |
| 2% NaN-zoo density, all float buffers at once | 212 (spec, instance) rows = 160 ht_gemv_b_c/ht_gemv_c rows (all 94 ht_gemv instances) + 52 attn_fa2_d dense rows (all 26 instances); plus 24 rows for dnc_ut_c, dnc_ut3_c, ht_deq_c (9 instances), i4r_gu_c, vis_rope_c | 0 mismatches (640 runs on the 212 rows, 96 runs on the 24 rows) |
| 20% mixed zoo (NaN, inf, denormal, max, -0), all buffers | same 236 rows | 0 mismatches (640 + 96 runs) |
| 50% NaN zoo | same 236 rows | 0 mismatches (640 + 96 runs) |
| dense-path specs of ht_gemv_gu_t (HtSd job, wg >= n), 94 rows | 2% NaN and 20% mixed | 0 mismatches (188 + 188 runs) |
| one buffer at a time (x, sv, su, pf, a, b, gate ...) at 5% NaN | 1,462 rows (ht_gemv token and dense paths, attn_fa2 dense) plus 56 rows for the five other families; this covers the "NaN in sv alone" shape that all-buffers fuzz can miss | 0 mismatches, 0 errors |
| inf-heavy vis_rope (10% +-inf) | vis_rope_c, N=64,77 | MATCH 4/4; original vis_rope.hip MISMATCH 4/4 |
| valid odd sizes | ht_gemv rows with N in {1,3,5,17, table N-1, N-2} (2% NaN and 20% mixed, seeds 133,134): 320 rows, 3,608 MATCH runs, 0 mismatches with diff>0, 0 errors, 680 UNTOUCHED-only (neither kernel writes that output at that N). Other families: dnc_ut/ut3 N=1,3,4; ht_deq N=1,2,4; i4r_gu N=1,7,9,100,102,4099; vis_rope N=1,3,5,17,65,76,78,4099 (48 rows, 176 runs, 0 mismatches). attn_fa2_d dense N=1,3,5,17,47,49,99,101 (104 rows, 832 runs, 0 mismatches) | 0 mismatches |
| sparse FaSel (attn_fa2_d), 26 instances x 13 variants x N=48,100 x seeds 1,2 | cnt all 0; ids all equal (3); ids with duplicates; descending ids; ids 0x7fffffff and negatives; cnt=64 ascending; random ids; tilemask all 0; all ones; 0x55555555; tilemask 0 with cnt 0; tilemask all ones with equal ids; ids null with tilemask set | 364 rows, 0 mismatches |
| sparse FaSel + small and large sel offset | 6 instances (A=1 and A=2: `<1,1,1,5,1,1>`, `<2,1,1,5,1,1>`, `<2,1,1,3,0,0>`, `<1,1,1,0,0,0>`, `<2,1,1,7,0,0>`, `<2,0,0,0,0,0>`), N=1,17,33,127, seeds 3,4,5 (sel0 = 26, 39, 52) | 84 rows, 1,008 MATCH runs, 0 mismatches |
| sparse FaSel + sentinel scores | constant q,k so every unmasked score s = -5e28, -9e28, -1.1e29, -2e29, -9.9e29, -1.01e30, -2e30, +1e20, -1e20; selection = random ids / tilemask 0x5555 / dense; same 6 instances | 162 rows, 0 mismatches |
| sparse FaSel + NaN zoo in q,k,v,gate | `_sp` and `_sp2` specs of all 26 instances, 2% NaN and 20% mixed | 112 rows, 448 runs, 0 mismatches |

Sensitivity of the fuzz (negative controls, so the "0 mismatches" lines mean something): the same manifests repointed at the original files
MISMATCH on ht_gemv_b (44 of 44 rows at 50% NaN, 44 of 44 at 20% mixed) and on dnc_ut, dnc_ut3, ht_deq, i4r_gu (all rows at 50% NaN and at 20% mixed). vis_rope escapes the 50%-NaN fill (no inf in it) but not the 20% mixed fill (24 of 24 control rows of that manifest mismatch) or the inf-heavy fill.
The all-buffers fills do not expose the ht_gemv gu_gs ordering bug (it needs NaN in sv or su alone); the one-buffer-at-a-time fills do: original
`ht_gemv.hip` on `gu_gs` isolating sv or su MISMATCH (diff>0) on 15 of 16 rows, fixed `ht_gemv_c.hip` MATCH on all. Sparse controls: the 11 sparse
variants and the sentinel + sparse specs on `attn_fa2_c.hip` (g25, no FaSel) MISMATCH 10 of 11 and 6 of 6 (`cnt64` with a null tilemask is
equivalent to dense, so it matches by construction).

Two false alarms, both mine, both explained before I trusted them: (1) my first odd-size pass used sizes above the fixed buffer extents of
the committed specs (dn_t N=801 reads x row 40 of 40, gu_t dense N=21 against an N=20 spec); 9 runs flaked with NaN-sign differences, the rewrite output
changed between identical runs (3 md5s in 30 runs) while the vendor output stayed fixed, and the same spec at N=799/776 was stable in 20 of 20 runs. That
is an out-of-bounds read of my own making, not a kernel defect; all later odd sizes stay inside the spec's buffers. (2) My first 3- and 4-group
`ht_lin_grp` specs scaled the partial-sum `yoff` by tokens-per-pass wrong for S>1, so groups overlapped and one instance flaked; fixed in the
generator (`mkgrp.py`) and rerun clean (see B, structural scans).

## Claim B: the three real integer gaps are fixed and nothing like them remains: UPHELD

### B1. Ten random EQUIVALENT lines: `random.seed(7)`, `random.sample` over the 610 non-comment lines of `const-audit-exceptions.d/g30..g34.tsv`

All ten hold. Each cite was re-read in `kernels-disasm-v2/` (never the old listing) and the rewrite lines read; where a spec reaches the bound I
re-ran it with fresh seeds.

| # | file:line, instance, value | what the vendor listing shows | rewrite | verdict, reaching run |
| --- | --- | --- | --- | --- |
| 1 | g31:203 k_dnc_scan3x<4,4,1,0,0,0,0,0,0> 0x2e | codeobj_1146000.s:291483 `s_cmp_gt_i32 s7, 46`; s7 = k - chunk base (`:211FD0 s_sub_i32 s7, s50, s1`); row t uses its own address if s7 > t else the fallback v131/v132 (row k-1) | dnc_scan3x_b.hip:91 `tok=min(base+t,k-1)`, :100 `t<valid` | holds. dnc_scan3x_b_a4f.json, seeds 101,102, N=43..50, 63..65, 108..113: 34 of 34 MATCH |
| 2 | g31:26 k_dnc_scan3<8,8,8,1,3,1,0> 0x37f | :35976 `v_cmp_lt_u32 0x37f, v34`, v34 += 0x80 per pass: tid+128j <= 895 gives 8 passes; global stride 0x6000, LDS stride 0x204 | dnc_scan3_b.hip tiles 8 rows per step | holds (static; pass count is a template constant, not an input) |
| 3 | g33:2 k_sample_f32 0x1000000 | :553 `v_cmpx_gt_u32 0x1000000, v16`, v16 = key ^ prefix, digit `v_bfe_u32 16,8` (level 2) | sample.hip lim = 1<<(32-8*(lv-1)) = 2^24 at lv=2, digit shift 24-8*(lv-1) = 16 | holds. sample_a, sample_edge, seeds 111,112, N=1000,4096: 8 of 8 MATCH |
| 4 | g30:51 k_linear_q4_mt<8,3,2,256> 0x300 | :756891 `v_add_co_u32 v28, 0x300, v3` then `v_cmp_le_i64 K, v28`, v3 += 0x200: "i + 3B < K" continues the unguarded two-chunk loop, otherwise a guarded tail | linear_q4.hip:88 plain `i += B` chain | holds. linear_q4_mt_8_3_2_256.json, K = 1, 255..257, 511..513, 767..769, 1023..1025, 1543, seeds 111,112: 28 of 28 MATCH |
| 5 | g30:76 k_lq4w<5,32,2,0,0,1> 0xa | :538299 `v_lshrrev_b16 v3.l, 10, v1.l` of tid*0xcd (= tid/5 for tid<20), :538303 `v_cmp_gt_u32 20, v0`, row guard against R; column tid%5, two segment partials summed 0+p0+p1 | lq4w.hip:96 tid<ROWS, loop j<NC, guard row<R | holds. lq4w_g30_tail_c5.json N=1001,1002,1003,1005,1007, seeds 111,112: 10 of 10 |
| 6 | g33:146 k_ix_select_slice<32> 0xc | :1089729 `v_writelane_b32 v96, s3, 12` and :1090609 region `v_readlane_b32 s3, v96, 12`: lane index of an SGPR spill; I scanned the whole kernel body (:1088882-1091375) for any other operand 12 / 0xc outside lane ops, none | ix.hip keeps no per-lane state | holds (static) |
| 7 | g30:98 k_lq4w<5,16,1,0,0,1> 0x81 (text cites the NC=1 listing) | :523576 `s_cmpk_lt_i32 s12, 0x81` with s12 = low half of (K+bias)>>5 (:523564-523572), :523579 `s_cmpk_gt_i32 s12, 0x80`: scale-preload path select | lq4w.hip:66 one loop for any G | holds. NC=1: lq4w_g30_thr_c1.json G = 126..130, 255..257, seeds 111,112: 16 of 16. NC=5 (the instance the row is named for): lq4w_g30_thr_c5.json, same G: 16 of 16 |
| 8 | g32:6 k_ht_gemv_gu_gs 0x1d | :1140469 `s_ashr_i32 s10, s2, 31; s_lshr_b32 s10, s10, 29` = signed wg/8; `s_cmp_ge_i32 s24, s3` (I >= 5*cnt) branches to :1142762 `s_nop; s_sendmsg; s_endpgm`, no second job. This is the same shift pattern as the real gu_t gap, so I checked for a dense job here: there is none | ht_gemv.hip:338 `I = blockIdx.x >> 3; if (I >= 5*cnt) return` | holds (static, contrast with gu_t) |
| 9 | g34:11 k_dn_step_pre 0x7f800000 | :28083 `v_cmp_gt_f32 s3, 0x7f800000, \|v4\|` in the inlined softplus; preceding `v_cmpx_nlt_f32 20.0` | dn_step.hip:25 `fabsf(l) < inf`, `!(20 < v)` | holds (static) |
| 10 | g30:61 k_linear_q4_mt<8,8,2,256> 0x300 | :767631 same pattern as #4 | same | holds. linear_q4_mt_8_8_2_256.json, same K list, seeds 111,112: 28 of 28 |

### B2. "Nothing like them remains": two structural scans

The audit's literal scan is blind to two shapes: (i) a path chosen by a pointer null test (`s_cmp_eq/lg_u64 x, 0` has no literal) and (ii) a
job chosen by comparing blockIdx with a kernarg field. Both were behind the real gaps (FaSel ids/tilemask; the HtSd dense job). I scanned all
listings for both and checked whether any MATCH spec reaches both sides.

(i) Null tests: 329 vendor kernels contain `s_cmp_*_u64 ptr, 0`; I mapped each tested register to its kernarg offset (`nullargs.py`) and
classified every committed MATCH spec's value at that offset (`nullstate.py`). Single-sided (never null or never non-null in any MATCH spec):
dn_conv_ab_rows (read pointers rows.p[8+r]), dnc_att6, dnc_prep, dnc_ut5, hc6/hc8 fused, hcup4/6/6w/8/8w, exp_dn_w<1> (out2), ht_lin_grp<4,1,1,0,0,1> (rot_out). For each I wrote the missing side:

- 118 null-flip specs (the always-non-null pointer set to 0; manifest `man_null.tsv`): 118 rows, 472 runs, 0 mismatches (null1.log 60 rows/240 runs, null2.log 58 rows/232 runs).
- dn_conv_ab_rows with non-null read pointers rows.p[8+r]: 28 specs, 112 runs, 0 mismatches.
- exp_dn_w<1> with the residual path (out2 non-null): `rewrite/specs/refuter2/exp_dn_w_1_resid.json`, seeds 161,162, N=1,3,7: 6 of 6.
- ht_lin_grp<4,1,1,0,0,1> with rot_out set (3 block-order variants A, B, C): 12 of 12 on the compared outputs (the rot-only blocks do not run in this instance in either kernel).

(ii) blockIdx vs kernarg splits: scanning `s_cmp s2, <kernarg reg>` (`bidscan.py`) and evaluating each spec's value against its grid found two
families where the group-start fields of members 2 and 3 never lie inside the launched grid in any MATCH spec: `dn_conv_ab_rows` LqGrp (starts at
kernarg 488 and 560, all 16 instances) and `ht_lin_grp` HtGrp (starts at 196 and 268, all 33 instances). The specs only ever used 2 groups. I built
3-group and 4-group specs (64 + 64 rows; member shapes copied from member 1; `yoff` scaled by tokens per pass):

- dn_conv_ab_rows count=3 and 4, R=1..8, E-variants a and c, N=33,48, seeds 151,152: 64 rows, 256 runs, 0 mismatches.
- ht_lin_grp ngroups=3 and 4 on all instances with a multi spec, N=512,1024, seeds 151,152: 64 rows, 256 runs, 0 mismatches (after fixing my generator, see A2).

I did not map `s_cmp_*_u64` tests whose register comes from a computed struct pointer (ht_lin_grp group-internal out2, ht_tg/ht_tg3, dnc_att6 second
pointer); for those I only confirmed that MATCH specs include both states of the same pointer in other instances of the family. They are the
remaining blind spot of this check. No blockIdx split other than the two above is single-sided with a job (the rest of the list are `blockIdx >= count`
early-exit guards or false positives where s2 is reused).

## Claim C: all 1,738 instances have a MATCH row on the rewrite of record: UPHELD

`cov2.py`/`cov3.py` (run from the repo root): the 1,738 distinct names in `analysis/function-map-kernel-stubs.tsv` col 2; the MATCH rows of
`rewrite/match-table.md` (30,446 rows, 30,437 MATCH) parsed by instance, spec, result; the spec must be tracked in git at f65b568, its `rewrite_src` (resolved against the spec
directory, falling back to the repo-relative form) must exist, and its basename must equal the end of the `superseded.tsv` chain.

```
stub rows 1738 unique 1738
ok 1738
no MATCH row: 0
fail end-of-chain: 0
instances in table not in stubs 0
```

Failing instances: none. Minimum number of end-of-chain MATCH rows for an instance: 3 (ht_moe dn2w/dn2x, ple_out_rows). By end file: ht_gemv_c 64 instances,
ht_gemv_b_c 30, attn_fa2_d 26 (1,914 rows, 72 or more per instance), attn_qs_c 32, ht_deq_c 10, dnc_ut_c, dnc_ut3_c, i4r_gu_c, vis_rope_c 1 each. For 740 of the
1,738 instances the end file does not contain the `rewrite_kernel` name literally; those are macro-pasted names (ht_lin, ht_gemv, i4r_rows_b `##` patterns) and the MATCH row
itself proves the symbol exists.

Observations that do not change the verdict:

1. `rewrite/kernels/attn_fa2_s.hip` (g34's intermediate sparse copy, 312 MATCH rows) is not in `superseded.tsv`, so a chain-end check treats it as a rewrite of record.
   It is superseded in fact by attn_fa2_d.hip; the 26 attn_fa2 instances have 72+ rows each on attn_fa2_d.hip, so no instance depends on it. Suggest adding the line.
2. Commit 8478f0d repointed 234 attn_fa2 specs in place to attn_fa2_c.hip (superseded by _d). Their match-table rows mix runs on the original file (before)
   and on _c (g25). They are not end-of-chain and are not used above; the evidence for those instances is the `attn_fa2_d_*` copies.
3. `ht_gemv_c.hip` was edited in place by g14n after g32b's 1,170 rows ran. The 5,133 g14n rows are the evidence on the current file.

## Not covered (explicit)

- No value-range attack beyond NaN, inf, zero, denormal and max in this round (round 1 covered clamp and sentinel scans on other families).
- The three families with no float input (ht_deq_t3/t4 and the integer-only list from round 1) cannot take NaN.
- B2 does not cover pointer tests whose register is loaded through a computed struct pointer (listed above).
- Fuzz density is random; "0 mismatches" is evidence, not proof. The controls above show the fuzz does catch the known defects.

## Reproduce

Scripts are in `rewrite/reports/refuter2-pr24-files/*.txt` (copy without the `.txt`). Attack specs generated on the Mac are in
`rewrite/specs/refuter2/` (relative `rewrite_src`; `file` fills point at `/home/rangizingo/re-halogen/rewrite/specs/*_data/`). Fuzz specs generated on Ai-core
(`rewrite/specs/refuter2/*__n02.json` and similar, data under `~/refuter2/data`) are regenerable with `nanfuzz.py`/`nanfuzz_iso.py`/`mkfa2.py` and the manifests
in the tarball; they are not committed because their data files are about 3.5 GB.
