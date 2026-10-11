# Refuter report, PR #24 (p7-12-wave2, tip e91bd65): U22 CLOSED claim

Verdicts: 1 REFUTED (NaN sign bit, ht_gemv dn_g/dn_t), 2 UPHELD, 3 UPHELD.
Repo files edited: none. Ai-core: work only in ~/refuter2 (removed afterwards); no services touched.
The Ai-core spec copies were deleted at cleanup; the finding's specs are regenerated in refuter-pr24-files/ (generators mksp.py, mkspec.py).

## 1. Range attack: REFUTED

Method. PR kernels rsynced to ~/refuter2/rewrite; spec copies point compare.rewrite_src at /home/rangizingo/refuter2/rewrite/specs/../kernels/<x>.hip.
Harness and vendor fatbins from ~/re-halogen. Command shape (from ~/re-halogen, CMP_TMP=$HOME/refuter2/tmp):
  python3 -I ~/re-halogen/rewrite/harness/compare.py --no-table <spec> --kernel <mangled> --seeds 31,32 --sizes <list>
Families (committed spec -> instance): lq6w_1_16_1, lq6k_1_16_1_0, ht_lin_b3_s1_a1, ht_gemv_b_dn_g_3_8_2_2, g22b_gl1_mix, q4moe_a1b2e2_64,
dn_fused16_base, hc6_f3_1, route_logits_w32. Plus k_sample_f32 (sample_a/b/edge/tie).
Fill modes applied to every float rand buffer (f32/bf16/f16): big (1e18; f16 6e4), max (3e38; f16 6.5e4), tiny (1e-30; f16 1e-7),
denorm (1e-40; f16 3e-8), zero, pos (0..big), mixed (-1e-3..big); then file fills with ~0.4% specials
(+-inf, -0, +-1e-40, +-3e38, and NaN in the "nan" files, none in "inf" files) per dtype.
Sizes: the committed ones, then odd ones (4099, 50021, 131, 389, 199, 201, 5, 7, 17, 101, 3, 19, 41, 161, 2561, 2599).

Results.
- 63 spec-mode runs x seeds 31,32 x committed sizes: 236 MATCH. The other lines were MISMATCH diff=0 with UNTOUCHED notes
  (dn_fused16 zero/tiny/denorm, hc6 tiny/zero): both kernels leave `st`/`y0` equal to the pre-fill, harness rule "agreeing on an unwritten output proves nothing". Not a divergence.
- Odd sizes (27 runs): 100 MATCH, 4 harmless MISMATCH of the same kind, none with diff > 0.
- inf/NaN file fills on the 9 families (18 runs): all MATCH except ht_gemv dn_g with NaN (below).

### Finding: NaN sign bit, ht_gemv dn_g / dn_t
Spec: refuter-pr24-files/ht_gemv_b_dn_g_3_8_2_2__fnan.json (committed ht_gemv_b_dn_g_3_8_2_2.json with x and sv replaced by file fills of sp_f16_nan.bin).
Kernel: _ZN7halogen12_GLOBAL__N_114k_ht_gemv_dn_gILi3ELi8ELi2ELi2EEEvPKjPKDF16_PKNS0_6HtItemEPKiS5_PtNS0_4HtSdEi
Rewrite: rewrite/kernels/ht_gemv_b.hip, compare output `out` (bf16, 102400 elements, N=200).
Command: compare.py --no-table ...__fnan.json --kernel <above> --seeds 31,32 --sizes 200
Output:
  MISMATCH ... ht_gemv_b_dn_g_3_8_2_2__fnan.json seed=31 N=200 n=102400 diff=5 max_abs=inf max_ulp=65536 first_diff=12126 tol=bitwise
  MISMATCH ... seed=32 N=200 n=102400 diff=8 max_abs=inf max_ulp=65536 first_diff=2382 tol=bitwise
  (seed 33: diff=3 first_diff=42282)
Differing output: `out` only. Seed 31 byte dump (vendor out.bin vs rewrite out.bin, uint16): element 12126 0x7fc0 vs 0xffc0; same pair at 42282, 73002, 75562, 76126. Every differing element is a NaN whose sign differs.
Isolation: x NaN only (xonly spec) MATCH on seeds 31,32,33; sv NaN only (svonly spec) MATCH; inf without NaN MATCH. Needs NaN in both x and sv.
Same signature (all float fills NaN, seeds 31,32,33, N=200): dn_g_3_8_2_0 diff 5/8/3 (first_diff 12126/2382/42282), dn_g_3_8_2_4 diff 5/8/3,
dn_t_3_8_2_1 diff 16/13/9 (n=112640, first_diff 1322, 2382, 32042 for seeds 31..33). For dn_t I did not dump bytes; the ulp 65536 matches the sign-bit pattern.
Hypothesis (not checked in ISA): operand order of a commutative float op; AMD returns the first NaN operand, so the kernels choose different NaNs (+qNaN vs generated -qNaN).
Severity low: value is NaN either way; but "bitwise on all inputs" does not hold for non-finite inputs.

### k_sample_f32 with NaN/inf logits: inconclusive
Specs sample_a/b/edge/tie with x filled from NaN or inf files (4096, 1000 and 100003 sizes): wherever a row has a non-finite value, both vendor and rewrite leave ids, lp_all, lp_kept untouched (diff=0), harness flags UNTOUCHED -> MISMATCH. Cases with output (small N or rare inf) MATCH.

## 2. Coverage attack: UPHELD
Own script over analysis/function-map-kernel-stubs.tsv (col 2), rewrite/match-table.md, rewrite/specs:
- 1738 unique mangled names; 1738 have a MATCH row; min 3 MATCH rows per instance.
- Every spec file named in a row exists (0 missing).
- For every instance at least one MATCH spec has compare.rewrite_src resolving to an existing rewrite/kernels/*.hip with no "neg" in the path (0 instances without one). Spec files with neg in the name are input-data cases (*_gneg, *_negzero), not control copies; negctl outputs live in rewrite/tables/*-negctl.md, excluded from the merge.
- 9 MISMATCH rows (k_dnc_scan3x i32 x6, k_dnc_ut5 norm_rs3 seed 2, k_vis_bias_gelu seeds 1 and 3); all belong to instances with later MATCH rows (footer note).
- Thin: k_dnc_cmp_words n=1, k_ngram_ctx_update n=4, k_count_diff_u32 n=4.

## 3. Constant audit: UPHELD (5 of 5)
Method: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 on Ai-core, llvm-objdump -d --mcpu=gfx1151, own extractor of hex/float immediates (skipping memory ops, offsets, waitcnt) from the rewrite and from kernels-disasm-v2/codeobj_53000.s.
- k_scale_silu: identical (0xbfb8aa3b, 1.0, 0x7fff, 0xffff).
- k_vis_bias_gelu: identical 24 literal instructions (0x3d372713, 0x3f4c422a, 0xc2ce8ed0, 0x42b17218, 0x3ca908c9, 0xbd5c1c4e, 0x3e088382, 0xbeaaaa99, 0xbbbac73d, ...); cmp ngt vs le is equivalent.
- k_rope_w_yarn: 8 vs 8, no difference. k_rmsnorm_grouped: 11 vs 11, no difference.
- k_sample_f32: 40 vs 40; vendor-only 0x10000 0x1000000 0x7f800000 0x7fffffff 0xff; rewrite-only 0x1f8 0x41f00000 0x7f8 0xbfc 0xc00; agrees with the audit row.
  Vendor tests (x & 0x7fffffff) < 0x7f800000 (finite); rewrite uses v_cmp_class_f32 mask 0x1f8 (finite). Same behaviour; the audit's "sign-flipped 0x7f800000" exception text names the wrong mechanism.
Limit of the audit itself: it checks vendor literals appear somewhere in the rewrite, not where they are used.
