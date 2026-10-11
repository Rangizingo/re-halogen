# Orchestrator re-runs of agent MATCH rows (p7-12)

Each group: files copied from the agent worktree into the main checkout, sha1 compared with the copies on Ai-core, a sample of MATCH rows re-run with seeds the agent did not use (`--no-table`), and one negative control of my own.

## g04 (lbfw, linear_f32, linear_m, linear_mt, linear, bf16), 2026-10-10

- Table: 45 distinct instances with MATCH (`awk -F'|' '$5 ~ /^ *MATCH/ {print $2}' rewrite/tables/g04.md | sort -u | wc -l`), negctl 6 MISMATCH rows.
- sha1 of the 6 .hip files and 3 sampled specs: Mac copy equals Ai-core copy.
- Re-runs, seeds 7 and 11: k_lbfw<3,2> at N 512 and 1003, 4 of 4 MATCH bitwise; k_linear_f32_q4_mt<8,5,4,256> at 1024 and 1002, 4 of 4 MATCH; k_bf16_to_f32 exhaustive (65,536 patterns) MATCH.
- Negative controls on k_lbfw<3,2> seed 7: RNE constant 0x7fff to 0x7ffe still MATCHes (it changes only outputs whose low 16 bits plus the LSB equal 0x8001, about 1 in 65,536; 1,600 outputs here). This is the MATCH limit stated at the top of match-table.md, not a harness fault. Truncation (0x7fff to 0) gives MISMATCH, 794 of 1,600 outputs differ, max_ulp 1.

## g08 (dn_recur, dn_step, dn_gates, dn_conv, conv, conv1d), 2026-10-10

- Table: 35 distinct instances with MATCH, 0 MISMATCH rows; negctl 27 MISMATCH rows. sha1 of the 6 .hip files: Mac equals Ai-core.
- Re-runs, seeds 7 and 11: k_conv1d_silu<true> at N 7 and 301 (81,920 and 3,092,480 outputs), 4 of 4 MATCH; k_dn_recur_step<0,false> 2 of 2 MATCH (811,104 outputs each).
- Negative control: norm epsilon 0x358637bd (1e-6) to 0x3727c5ac (1e-5) on k_dn_recur_step<0,false> still MATCHes on the agent's spec (qkv in [-2, 2]: the epsilon is below one ulp of the sum of squares). Added `rewrite/specs/dn_recur_step_s0_small.json` (qkv in [-0.001, 0.001]): the rewrite MATCHes (seeds 1, 2, 3 in `rewrite/tables/orch.md`, and seed 7), the epsilon change gives MISMATCH, 8,341 outputs differ, max_ulp 25. Brief addendum sent to the running agents.
- g08 follow-up (edge addendum): 14 `_small`/`_edge` specs, 657 MATCH rows, 66 negctl MISMATCH rows (`collect.sh` counts). The rsq_guard denormal path and the softplus denormal guard are shown unreachable (arguments at least 1e-6 and 1.0, report g08.md:106-107), so no spec can reach them.

## g12 (add, sigmoid, glu, embed, gather, slot, ngram, argmax, rmsnorm, act, ple, q, ix), 2026-10-10

- Table: 49 distinct instances with MATCH, 0 MISMATCH; negctl 20 MISMATCH rows (plus one insensitive 1-ulp epsilon row kept and labelled by the agent). sha1 of 13 .hip files: Mac equals Ai-core.
- BLOCKED: k_resid_sample_f32 (a full sampler, 1,713 listing lines; not started). Goes to the wave 2 sampler group.
- Re-run, seeds 7 and 11: k_rmsnorm_gated_w on rmsnorm_gated_w_edge.json (x in +-1e-4), 2 of 2 MATCH.
- Negative control, epsilon 1e-6 to 1e-5 (rmsnorm.hip:32): MISMATCH on the edge spec, 1,664 of 1,728 outputs differ, max_ulp 222; on the wide spec rmsnorm_gated_w_a.json only 1 output differs (still MISMATCH, but by chance).

## g02 (lq8s32, lq8w, lq8w_grp, deq_q8s32, deq_q8g64), 2026-10-10

- 106 distinct instances with MATCH, 0 MISMATCH rows; negctl 35 MISMATCH rows (1 MATCH row documented by the agent). sha1 of 5 .hip: Mac equals Ai-core.
- Re-run seeds 7 and 11: k_lq8w_grp<5,32,2> at N 77, 2 of 2 MATCH.

## g06 (route_logits, route_topk, route_hist, route_scan, route_scatter, hist, count), 2026-10-10

- 41 distinct instances with MATCH, 0 MISMATCH rows; negctl 38 MISMATCH rows. sha1 of 7 .hip: Mac equals Ai-core.
- Re-run seeds 7 and 11: k_route_topk_w on route_topk_w_ties.json at 13 and 40 rows, 4 of 4 MATCH.
- Negative control, softmax log2e constant to 1.4 (route_topk.hip:15): MATCH on the ties spec (all ten picked values equal, so every exponent is 0 and the constant drops out), MISMATCH on route_topk_w_rand.json (400 of 832 outputs differ). The rand spec covers the constant.

## g09 (dnc_scan3x), 2026-10-10, partial

- 8 of 21 instances MATCH; 13 BLOCKED (b1=F, i2 4-6 timing probes, b2=T, A=4), continued by agent g09b. The 6 MISMATCH rows in g09.md (g09.md:41-46, dnc_scan3x_i32.json, n=1,572,960, diff=0) are from a spec version that also compared the n3 counter, which returns to 0 after two arrivals and so reads as UNTOUCHED (report g09.md table); the later rows of that spec compare sel/out only (n=1,572,864) and MATCH (QA on PR #23).
- B, C, D, G (ledger U12) are k_dnc_scan3 template args, not scan3x (report g09.md "Template args"); they go to agent g16 (dnc_scan3).

## g11 (hc8, hcup4, hcup4w, hc_scatter, hc_mixed, hc_di, hyper, split, block, inject, hc_fold), 2026-10-10

- 43 distinct instances with MATCH in g11.md; LqGrp branch of the 16 hc8 kernels in `tables/g11-lq.md` (96 MATCH rows; negctl `g11-lq-negctl.md`). sha1 of 11 .hip: Mac equals Ai-core. negctl 12 MISMATCH rows in g11-negctl.md.
- Re-run seeds 7 and 11: k_hc8_fused<2> on hc8_lq_2.json (LqGrp branch), 2 of 2 MATCH.

## g07 (attn_fd, attn_fa2, qk, rope, kv, attn_fa, attn, attn_sel), 2026-10-10

- 45 distinct instances with MATCH, 0 MISMATCH rows; negctl 29 MISMATCH rows. sha1 of 8 .hip: Mac equals Ai-core. Edge specs present for attn_fd_merge, attn, attn_sel, attn_fa, attn_fa2 (3), qk.
- Re-run seeds 7 and 11: k_attn_fa2<1,1,false,0,0,false> at N 48 (295,040 outputs), 2 of 2 MATCH.
- Negative control, log2e 0x3fb8aa3b to 0x3fb8aa3c (1 ulp, attn_fa2.hip:30): MISMATCH, 16 outputs differ, max_ulp 176.

## g01 (lq4w, 488 instances), 2026-10-10

- 488 distinct instances with MATCH (3,348 rows), 0 MISMATCH; negctl in g01-negctl.md. Report written by the agent directly into the main checkout. harness/ and kernargs.json in the agent worktree are byte-identical to main (`diff -r`, `cmp`).
- Instance selection: one templated lq4w.hip; each instance is compiled with `HIPCC_COMPILE_FLAGS_APPEND="-DLQ_N.. -DLQ_F.."` set by `rewrite/specs/lq4w{,_pair,_grp}_run_all.sh` (needs `LQ_TSV`, a copy of function-map-kernel-stubs.tsv, on Ai-core). A row is still a comparison against the named vendor instance, whatever the flags.
- Re-run seeds 7 and 11: k_lq4w<1,16,1,true,3,1> at N 1000 and 2800, 4 of 4 MATCH; k_lq4w<1,32,8,false,5,1> seed 7 at 1024 and 3072, 2 of 2 MATCH.
- Controls: building the E=5 instance with LQ_F=0 instead of 1 still MATCHes at N 3072 (F changes the loop structure only, no numeric effect; not a valid control). RNE bias 0x7fff to 0 (lq4w.hip:15) on <1,16,1,true,3,1>: MISMATCH, 171 of 1,000 outputs differ.

## g05 (exp_gu, exp_dn, deq_q4cp, q4moe_items), 2026-10-10, partial

- 66 of 86 instances MATCH, 0 MISMATCH rows; negctl 11 MISMATCH rows. sha1 of 4 .hip: Mac equals Ai-core. The 20 k_q4moe GEMM instances are BLOCKED (5,476 to 41,499 listing lines each), continued by agent g05b.
- Re-run seeds 7 and 11: k_exp_gu_w<16,2,0> on exp_gu_w_16_2_0_small.json, 2 of 2 MATCH. Control, silu log2e 1.44269502 to 1.4426 (exp_gu.hip:114): MISMATCH, 278 of 6,432 outputs differ.

## g16 (dnc_scan3 family), 2026-10-10, partial

- 20 of 83 instances MATCH in g16.md, but the rows name specs `sw_A{2,4,8,16,32}.json` that exist only in /tmp/g16/specs on Ai-core, not in rewrite/specs. Not reproducible from the repo: g16b asked to commit them and re-run the 20 instances into g16b.md. g16.md rows do not count until then.
- U12 B, C, D, G (report g16.md "U12"): instances differing in one of them match the same rewrite, so no numeric effect on the tested inputs. Open before U12 closes: what D's extra global_load_b32 reads (a spec that varies it), G's index range, listing cites for B/C. Assigned to g16b.
- spec check (all collected groups g01, g02, g04-g09, g11, g12, g11-lq, orch): every spec named by a MATCH row exists in rewrite/specs. Only g16 failed it.

## g19 (dnc_scan3w, dn_fused16), 2026-10-10, partial

- 4 of 29 instances MATCH (the 4 non-probe dnc_scan3w); negctl 15 MISMATCH rows; specs all committed. Left: 8 scan3w timing probes (agent g19b: reproduce the probe's skipped-work output bitwise) and 17 dn_fused16 (agent g19c).

## g21 (hc6, hcup8, hcup8w, hcup6, hcup6w), 2026-10-10

- 32 distinct instances with MATCH, 0 MISMATCH; specs all committed. Agent wrote directly into the main checkout; tables copied from Ai-core.
- Re-run seeds 7 and 11: k_hcup8<1> at N 37, 2 of 2 MATCH.
- Controls: the agent's sigmoid constant change -1.44269502 to -1.4427 (4e-6 relative) is not detected (bf16 output rounding, the MATCH limit). Mine, 1.44269502 to 1.40 (hcup8.hip:24): MISMATCH, 956 of 5,248 outputs differ.

## g18 (vis_*, vis_attn), 2026-10-10, partial

- 21 of 53 instances MATCH; 32 attn_qs not started (agent g18b). Specs all committed. The 2 MISMATCH rows (k_vis_bias_gelu, vis_bias_gelu.json seeds 1 and 3, N 100003, 1-2 ulp on 1-2 outputs) predate the agent's `fp contract(off)` fix; the same spec MATCHes afterwards (g18.md:59-64), and a 1e6-point sweep MATCHes.
- Re-run seed 7: k_vis_bias_gelu on vis_bias_gelu_sweep.json, 1,000,064 outputs, MATCH.
- BRIEF.md second addendum: fp contract(off) and dense sweeps for element-wise f32 kernels.

## g09b (dnc_scan3x remaining 13), 2026-10-10

- With g09: 21 of 21 dnc_scan3x instances have MATCH rows (`awk` over g09.md + g09b.md). g09b.md 0 MISMATCH; negctl 24 MISMATCH rows; specs all committed. Agent wrote into the main checkout; tables copied from Ai-core. The i2 = 4, 5, 6 instances are state-update ablations, not clock probes; no bytes excluded.
- Gap closed by orchestrator: the b2=T k-bound guards were tested only at N 128 and 100 (report g09b.md:22). Re-run k_dnc_scan3x<2,4,T,T,T,0,0,0,F> on dnc_scan3x_b_b2t_b3t.json seed 7 at N 1, 37, 63, 64, 65: 5 of 5 MATCH.

## g20 (dnc_att, att2, att6, att7, att8, att9, ut, ut2, ut3, cmp), 2026-10-10

- 24 distinct instances with MATCH, 0 MISMATCH; negctl 13 MISMATCH rows; specs all committed; sha1 of 10 .hip Mac equals Ai-core.
- Re-run seeds 7 and 11: k_dnc_att9 m8 on dnc_att9_m8.json at N 2 and 5, 4 of 4 MATCH. Control, causal mask `j <= row` to `j < row` (dnc_att9.hip:64): MISMATCH, 14,544 outputs differ.

## g19b (dnc_scan3w timing probes P=1..8), 2026-10-10

- 8 of 8 instances MATCH (240 rows, 0 MISMATCH); specs committed (48). Re-run seeds 7 and 11: probe P8 at N 128, 2 of 2 MATCH.
- Caveats carried into match-table.md for these two rows: P5 reads LDS 0x2800.. that the kernel never writes; the vendor output equals the rewrite with that table = 0 in every run here, which is empirical (LDS is not zeroed by hardware in general). P8 uses VGPRs v10..v13 that the listing never writes; their content was measured on the hardware (a fixed pattern) and the rewrite reproduces it. Both are debug probes (HALOGEN_DN_PROBE timing decomposition) whose outputs are not meaningful values.

## g15 (linear_q4, linear_q8, linear_q8s32), g17 (ht_moe, ht_deq, ht_rot, ht_items, ht_tg, ht_tg3), g19c (dn_fused16), 2026-10-11

- g15: 74 instances MATCH, 0 MISMATCH; negctl 45. g17: 71 MATCH, 0 MISMATCH; negctl 16. g19c: 17 MATCH, 0 MISMATCH; negctl 10 (agent wrote into the main checkout). Specs all committed; sha1 Mac equals Ai-core for g15 (1 .hip) and g17 (5 .hip).
- Re-runs seeds 7 and 11 (2 of 2 MATCH each): k_linear_q4 on linear_q4_single_small.json N 1024; k_ht_items_group<2> on ht_items_group2.json N 100; k_dn_fused16<1,F,F,F> on dn_fused16_base.json N 100.
- Control, RNE bias 0x7fff to 0 (dn_fused16.hip:42): MISMATCH, 307,007 of 1,400,832 outputs differ.

## g23 (sample, accept, pen_scatter, pen_rep, smpg_*, nll, resid) and g10 (dnc_ut5, dnc_prep, dnc_ut4), 2026-10-11

- g23: 20 of 20 instances MATCH (the families hold 20 instances, not the 22 in the brief), 0 MISMATCH; negctl 107; input data files in rewrite/specs/g23_data/ (collect.sh now copies directories). sha1 of 15 .hip Mac equals Ai-core.
- g10: 35 of 35 MATCH; the one MISMATCH row (dnc_ut5_norm_rs3.json seed 2 N 1000, n=12,582,912, max_abs inf with diff 0) is from an earlier spec version (later rows of the same case have n=18,874,368 and MATCH); negctl 8.
- Re-runs seeds 7 and 11 (2 of 2 MATCH each): k_pen_scatter_rows_f32 on pen_scatter_rows.json N 100003; k_dnc_ut4<4,4,F,F,F> on dnc_ut4_base.json N 1000.
- Control on pen_scatter: my first attempt edited 0x3ca00, which appears only in a comment (no-op, MATCH); the code constant 248320 to 248319: MISMATCH, 6 of 1,489,920 outputs differ (only rows whose ids land past the shifted slot change).

## g18b (attn_qs), 2026-10-11, partial

- 27 of 32 MATCH (331 rows, 0 MISMATCH), negctl 128 rows; specs committed. 5 BLOCKED (qs<2,0>, qs<3,0> stale LDS with run-to-run vendor variation; bt2<2,0>, <32,0>, <64,0> unset VGPRs) and weak controls (bt2<4,0>, sentinels) go to agent g18c, with the new compare.py per-output exclude option (BRIEF fourth addendum).
- Re-run seeds 7 and 11: attn_qs_0_1.json at N 203: see next line.
- MATCH seed=7 N=203 diff=0;MATCH seed=11 N=203 diff=0;

## g22 (i4r_rows, i4r_fold, i4r_gu, expert, moe, deq_aff, dv, cc, scatter, esc, sel), 2026-10-11, partial

- 33 of 44 MATCH, 0 MISMATCH; negctl 52; specs committed. Worktree i4r_gu.hip / i4r_rows.hip differed from the Ai-core copies that produced the rows by two comment lines only (diff), synced. In-kernel :NNNNN cites may be off by tens of lines (agent note). 11 left (HtItem group, gl/gl_o, k_dv_quant) to agent g22b using committed file fills.
- Re-run seeds 7 and 11 on dv_scatter.json N 100003: MATCH seed=7 N=100003 diff=0;MATCH seed=11 N=100003 diff=0;

## g05b (20 k_q4moe GEMMs), g14 (ht_gemv, partial), 2026-10-11

- g05b: 20 of 20 MATCH (207 rows), negctl 22; sha1 Mac equals Ai-core. Re-run seeds 7 and 11: q4moe_a1b2e0_128.json N 16 (1,779,200 outputs), 2 of 2 MATCH. Control, RNE bias 0x7fff to 0 (q4moe.hip:65): MISMATCH, 142,375 outputs differ (a first sed matched nothing because the literal has no `u` suffix; checked with diff before running).
- g14: 64 of 94 MATCH (1,082 rows, 0 MISMATCH), negctl 36; agent wrote into the main checkout. Re-run seeds 7 and 11: ht_gemv_gu_t_4_8_0_0.json N 400, 2 of 2 MATCH. 30 BLOCKED (HtFold, HtSd layouts) to agent g14b.
- Listing defect found by g18c: old kernels-disasm has 212,067 `.long` lines; kernels-disasm-v2 (llvm-objdump 23.1.3) has 0. BRIEF fifth addendum.

## g16b (dnc_scan3 family, U12 B, C, D, G), 2026-10-11

- 83 of 83 MATCH (882 rows, 0 MISMATCH), including g16's 20 re-run from committed specs dnc_scan3_sw_A*.json; negctl 18 MISMATCH and 8 designed-insensitive MATCH rows. sha1 Mac equals Ai-core.
- Listing cites re-checked in kernels-disasm-v2/codeobj_1146000.s: :139471 and :146343 are the <8,8,8,F> and <8,8,8,T> kernel starts; global_load count 14 and 14 in their ranges; G=1 range :149995-152141 has 22 v_movrel and 16 global_load; :150241-150245 `s_mov_b32 m0, s11` ... `v_movreld_b32 v3, v44`, :150267-150269 `v_movrels_b32 v46, v3`.
- Re-run seeds 7 and 11 at N 1 and 300 on dnc_scan3_b_base.json: <8,8,8,T,0,15,0> and <8,8,8,T,0,15,1>, 8 of 8 MATCH.

## g13 (ht_lin, 97) and g22b (11 left from g22), 2026-10-11

- g13: 97 of 97 MATCH, 0 MISMATCH; negctl 7; sha1 equal; collected after the session's hook pause, not yet re-run by the orchestrator.
- g22b: 11 of 11 MATCH (414 rows), negctl 44 (4 explained insensitive mutants in its report); cites in kernels-disasm-v2; data files in rewrite/specs/g22b_data. With g22: 44 of 44. Re-run seeds 7 and 11: k_i4r_rows_gl_g<1,2> on g22b_glg12_p1_edge.json N 1, 2 of 2 MATCH. g22b found I4rSd = HtSd (passed to g14b) and corrected g22's fl_g block-to-item map (blockIdx/40).

## g18c (attn_qs remaining 5 + g18b weak controls), 2026-10-11

- 30 of 32 attn_qs instances MATCH with g18b + g18c (614 g18c rows, 0 MISMATCH, no exclusions); spec check clean; attn_qs_c.hip sha1 Mac equals Ai-core. bt2<2,0>, <32,0>, <64,0> were a listing artifact (old kernels-disasm `.long`), now bitwise.
- g18b's rewrite was wrong in the score range near the -1e29 sentinel (no `ST` handling) for bt4, bt4x, bt2w and bt2<0,x>; its recorded specs never reached that range. Orchestrator check: attn_qs_c_bt4_0_mid.json N 40: g18c rewrite seeds 7 and 11 MATCH; g18b's attn_qs.hip on the same spec MISMATCH, 18,432 outputs differ. For these instances the g18c rows are the evidence; match-table.md must cite them.
- qs<2,0>, qs<3,0>: V comes from 48 KiB of LDS the kernel never writes (no global V load); 5 vendor runs differ in 21-27k of 37,120 bytes. Deterministic and bitwise under a priming kernel (agent's scratch runs, 36 of 36, controls mismatch); needs a compare.py prime option before rows exist (agent g18d).
- attn_qs_c_3_0.json N+700: the kernel masks the K row with v_and 0x1ff, so rows up to 511 are read (report g18c.md); keeps reads in bounds.
