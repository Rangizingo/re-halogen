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

- 8 of 21 instances MATCH; 13 BLOCKED (b1=F, i2 4-6 timing probes, b2=T, A=4), continued by agent g09b. The 6 MISMATCH rows in g09.md are diff=0 UNTOUCHED results on the i32 spec (n3 returns to 0 after two arrivals, report g09.md table), not value differences; the instance has MATCH rows on sel/out outputs.
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
