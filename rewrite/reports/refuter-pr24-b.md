# Refuter PR #24 (head 88b9b35), second refuter

Scope was changed mid-run by the orchestrator: attack 2 (range attacks) stopped after attn_fa2; the NaN sweep over all families is in `refuter-pr24-nan.md`. Items 1 and 3 were finished before the change.

## Item 1, coverage: UPHELD with one wording correction

Script: `refuter-pr24-files/cov.py` (run from the worktree root: `python3 -I cov.py`; needs the heuristics for macro-generated kernel names).

- `analysis/function-map-kernel-stubs.tsv` col 2: 1,738 rows, 1,738 distinct.
- `rewrite/match-table.md`: 1,738 distinct instance names, none missing, none extra, every instance has at least one MATCH row. Not "exactly one": 48 instances have 3 MATCH rows, most have 6 or more, max 120 (`cov.py` histogram).
- Every MATCH row names a spec that exists in `rewrite/specs/`; that spec's `rewrite_src` (relative to the spec dir) exists and contains `rewrite_kernel` literally, or as a macro-pasted name (ht_lin `ht_lin_b##BITS##_s##S##_a##A`, ht_gemv, i4r_rows_b `_gl_g*`; the regex matches the `##` pattern, not an expanded symbol; a MATCH row itself needs the symbol to exist at launch). The spec also appears in a `rewrite/tables/*.md` row for that instance, except `k_scale_silu` (rows exist only in match-table.md, original commit a006eed).
- 11 non-MATCH rows remain in the table (9 flagged by `startswith('MATCH')`: dnc_scan3x x6, dnc_ut5 x1, vis_bias_gelu x2), all noted at match-table.md:19416 with later MATCH rows.

## Item 3, five hand audits: UPHELD (5 of 5)

Picks (`random.seed(24)`, `random.sample` over the 19,394 MATCH rows): k_ht_gemv_dn_ts<4,8,4,false>, k_smpg_rpick, k_exp_gu_w<32,1,0>, k_attn_fa2<1,1,true,0,0,false>, k_lbfw<5,4>. Re-run seeds 7,8,9 at the row's N (20, 1, 3, 48, 512), `compare.py --no-table`: all 15 runs MATCH, diff=0 bitwise (e.g. `lbfw_5_4.json seed=9 N=512 n=2624 diff=0`).

Listing cites re-read in `kernels-disasm-v2` (cites use old-listing line numbers; v2 has a different line count, 2,280,854 vs 2,409,480 for codeobj_53000, so I located by instruction address):
- lbfw: old :141888-141900 (`v_dot2_f32_bf16` chain, 0x2D7C40-0x2D7C8C) is at v2 :132928-132940; RNE `v_bfe_u32 16,1` + `v_add3_u32 0x7fff` + `global_store_d16_hi_b16` at v2 :132991-132997. Matches rewrite.
- smpg_rpick: cite ":8099 v_cmp_lt_f32" is `.long 0xd411000a` in the old listing; v2 :7746 decodes it as `v_cmp_lt_f32_e64 s10, 0, v1` (the `0 < R` test). Atomic add at old :8719-8738 confirmed (`global_atomic_add_u32`, 0x21C124).
- exp_gu: epilogue old :2401959-2402010 is `v_mul_f32 -log2e`, `v_exp_f32`, add 1.0, IEEE div sequence (`v_div_scale/fmas/fixup`), mul, bf16 RNE store; rewrite does `exp2f(gt*-log2e)`, `gt/(1+ex)`, `ut*y`, RNE. No clamp in either.
- attn_fa2: exp2s (old :131563-131572, v2 :127579-127587: -126 test, +64, `v_exp_f32`, `v_ldexp -64`) and the alpha guard (old :132214-132229, v2 :128152-128166: `v_cmpx_lt_f32 0xefa18f08`, 0xefa18f08 = -1.0000000150e29) match the rewrite.
- ht_gemv dn_ts: re-run only (another agent owns ht_gemv); cite not re-read.

## Item 2, range attacks: REFUTED (attn_fa2 A=2), other families not reached

Only attn_fa2 was attacked before the scope change. Tool: `refuter-pr24-files/mk.py` (derives a spec by swapping float fills for file fills with specials) and `fa2const.sh`. Base spec `attn_fa2_2_1_1_5_1_1_edge.json`, kernel `_ZN7halogen12_GLOBAL__N_110k_attn_fa2ILi2ELi1ELb1ELi5ELi1ELb1EEEvPKtS3_S3_lS3_PtPKiiiNS_5FaSelE`, rewrite `k_attn_fa2_2_1_1_5_1_1`.

Reproducers (in `refuter-pr24-files/specs/`, data in `refuter-pr24-files/data/`; paths inside are absolute Ai-core paths under /home/rangizingo/refuter-pr24):

1. `fa2_def_all.json`, q/k/v/gate with 5%/5%/2%/5% specials (inf, nan, denormals, clamp neighbours, -1e29 sentinel neighbours): `compare.py --seeds 1,2,3 --sizes 48,100`. Seed 1 MISMATCH at N=48 and N=100: diff=9, first_diff=16907, vendor 0xffc0 (NaN) vs rewrite finite (0x3e39, 0x2c27, 0x3f15, 0xbe88, 0x8000, 0xbd45, 0x3e96, 0x0, 0x3dcf). Seeds 2,3 MATCH.
2. `sw_attn_fa2_2_1_1_5_1_1_edge_v_inf.json` (v = +inf at 2%, rest in [-1,1]), seed 1, N=48: diff=17,713 of 393,344, first_diff=6, vendor NaN vs rewrite finite (e.g. elem 6 vendor 0xffc0, rewrite 0xbf7d). Token 0, V column 6 has inf only at key 26, a masked key.
3. `fa2A2c__2e29.json` (q = 1e15 everywhere, k constant so every unmasked score s = -2e29): seed 1, N=48 diff=98,304 (rows 0-15, first_diff=0, vendor NaN, rewrite -0.7734375, 0.40039062, ...). Scan of s (same file family, `fa2const.sh`): MATCH for s in {1e20, -1e20, -5e28, -9e28, -1.01e30 and below, 1e30 and above}; MISMATCH for s in [-1.1e29, -9.9e29], i.e. between the -1e29 alpha guard (0xefa18f08) and the -1.0e30 initial m (0xf149f2ca).
4. q or k with 1e30 at 1% (`sw_..._k_1e30.json`, `sw_..._q_1e30.json`, built by `mk.py`), seed 1, N=48: diff 2,304 and 3,072, same NaN-vs-finite pattern.

Control: the A=1 instance `k_attn_fa2<1,1,true,5,1,true>` MATCHes the same constant-score scan (s = -5e28, -2e29, -9.9e29, seed 1, N=48).

Root cause (hypothesis supported by experiment, ISA loop bound not read): `fa_core` computes `ntiles` from the sub-tile's own last row (`lastpos = row0 + 15 + base`, `rewrite/kernels/attn_fa2.hip:97`). With A=2 the vendor appears to use the block-wide bound, so sub-tile 0 also runs the trailing tile that is fully masked. That extra tile has P=0 for every key, so it normally adds nothing, but (a) 0 * V=inf is NaN in the WMMA, and (b) its alpha = exp2s(m_old - m_new) is skipped when m_old <= -1e29 (alpha=0), which wipes O and l and gives 0 * (1/0) = NaN. `refuter-pr24-files/attn_fa2_probe.hip` (copy with `lastpos += 16*(1-sub)`) turns k_1e30 and fa2_def_all into MATCH on seeds 1,2,3 at N=48,100, but is not exact (s=-2e29 then mismatches at rows 32+; v_inf seed 1 N=48 still differs), so the exact vendor bound is open.

Impact: real softmax inputs never reach |s| ~ 1e29 and V is finite; the divergence needs V=inf/NaN in masked positions or scores near -1e29. It does show the MATCH rows say nothing about those inputs, and that the rewrite's loop structure differs from the vendor's for A=2. Not tested on the other 25 attn_fa2 instances.

Not reached: expf/exp2 clamp scans on other families, attn_qs, dn_fused16, ht_moe, hc, quantize and sampler range attacks (the NaN sweep covers all but ht_gemv at one spec each).

## Counts

UPHELD: item 1 (with wording correction), item 3. REFUTED: item 2 (attn_fa2 A=2; separately 5 NaN-sign mismatches in `refuter-pr24-nan.md`).
