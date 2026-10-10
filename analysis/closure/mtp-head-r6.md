# MTP head sixth pass: head step and target step from the live launch trace

## Scope

Closes `analysis/closure/mtp-head-r5.md` gap 1 (ordered kernels of one head step, record 48) and residuals 1-3 (head Linear kernels, the `+0x723`/`+0x748` router path choice, the router-logit kernel) from what run 4 actually launched. Nothing was run in this pass; it reads the committed capture.

Evidence: `analysis/evidence/run-r4-kargs.jsonl.gz` (27,833 lines; read with `python3 -I`). Kinds: `kname` maps handle `h` to the mangled name (106 rows), `trace` is a kernel launch (25,333), `matmul` is one `hipblasLtMatmul` (2,220). Trace and matmul rows share one `seq` counter; sorted by `seq` they number 27,553 and `seq` equals the list index (0..27552), so every cite below is a `seq`. Idle gaps over 1.5 s in `ns` (after seq 6804, 11066, 17907, 23290) split 5 requests, matching `analysis/evidence/run-r4-kargs-trace.txt`. Request order from `run-r4-probe6-results.txt`: R0 A300_cold, R1 A300_warm, R2 A1100_ext (800 new tokens on 300 cached), R3 B1100_cold, R4 A1100_warm. Each ran 4 output tokens.

Labels: **observed** = read from the trace; **inferred** = reasoned, stated as such. Grids are the `grid` x field of the `trace` row unless a pair is shown. Template args are as demangled from `kname`.

How the head step was told from the target step: the head step is the only sequence that starts `k_embed_gather, k_rmsnorm_grouped, lq8w, k_rmsnorm_grouped, lq8w<4,..>, k_hyper_seed_add`, which is r5 steps 1-6 exactly (launches `17f35fc`, `17f36ca`, fc_embedding, `17f37e0`, fc_hidden, `17f3d49`). The target step starts `k_embed_gather, k_hyper_seed` (no `_add`).

## Verdict: gap 1 (head step ordered kernels, round segmentation) CLOSED

Closed for the two head-step shapes that ran: decode (n = 1 row, 39 launches) and prompt-length (n = 300, 800, 1100 rows, 78 launches). The trace gives ordered kernels and grids; the static launch-site tie is made where r5 names a site and marked unmapped where it does not.

### 1. Request segmentation (observed)

Every request is: target prefill (none if cached), a head step over the prompt rows, then repeating `[head step] + [target verify step]`, then a trailing head step. The embed grid of a target step is its verify width n.

| request | target prefill | head over prompt | rounds (seq ranges) | tail head |
|---|---|---|---|---|
| R0 A300_cold | 0..2508 (2509 launches, embed grid 300) | 2509..2586 (78) | 2587..2625 head, 2626..3991 T(3); 3992..4030 head, 4031..5384 T(2); 5385..5423 head, 5424..6763 T(1) | 6764..6804 (41) |
| R1 A300_warm | none | none | 6805..6806 lone argmax pair; 6807..6848 head (42), 6849..6887 head, 6888..8253 T(3); 8254..8292 head, 8293..9646 T(2); 9647..9685 head, 9686..11025 T(1) | 11026..11066 (41) |
| R2 A1100_ext | 11067..13611 (2545, embed grid 800) | 13612..13689 (78, grid 800) | 13690..13728 head, 13729..15094 T(3); 15095..15133 head, 15134..16487 T(2); 16488..16526 head, 16527..17866 T(1) | 17867..17907 (41) |
| R3 B1100_cold | 17908..20416 (2509, embed grid 1100) | 20417..20494 (78, grid 1100) | 20495..20533 head, 20534..21899 T(3); 21900..21938 head, 21939..23290 T(2) (1352 launches) | none |
| R4 A1100_warm | none | none | 23291..23292 lone argmax pair; 23293..23334 head (42), 23335..23373 head, 23374..24739 T(3); 24740..24778 head, 24779..26132 T(2); 26133..26171 head, 26172..27511 T(1) | 27512..27552 (41) |

Counts (observed):

- Decode head step: 39 launches, 0 `hipblasLtMatmul`. The step at 2587 is the reference; seven steps compared equal on kernel name and grid: 2587, 3992, 5385, 8254, 13690, 20495, 23335 (python over the gz). The other 39-launch steps have the same length by embed spacing.
- Variants: 41 launches = the 39 plus a second `k_argmax_part`/`k_argmax_final` pair as the last two launches (seq 6803-6804, 11065-11066, 17906-17907, 27551-27552); 42 launches = the 39 plus `k_block_pool`, `k_rmsnorm_grouped`, `k_rope_blocks` after `k_ix_gather_k` (seq 6820-6822 and 23306-23308; `k_ix_gather_k` grid 2 there), and no extra pair. The 42 variant is the first head step of each warm request. Block pooling for a new indexer block is inferred.
- Target steps: T(1) 1340, T(2) 1354, T(3) 1366 launches. T(2) and T(3) end with `k_ngram_ctx_update` and `k_ple_conv_state_update` (seq 3990-3991 after the first T(3)); T(1) and the last T(2) of R3 (ends seq 23290) do not. Inferred: those two are post-step state updates, skipped when the request is finishing.
- Verify width is 3, 2, 1 in R0, R1, R2, R4 and 3, 2 in R3. `D` lines in `run-r4-probe6-results.txt` give drafter 1, rounds 2, commit 3 for all five, so the D-line `rounds` field is not the number of verify steps in the trace (3 in R0, R1, R2, R4; 2 in R3, refuter on PR #18); whether the width-1 step counts as a speculative round is not settled by this run. Inferred, not proven: n is capped by remaining output tokens (4 total, 1 from prefill), and R3's T(2) committed 2 tokens while the other steps committed 1. The trace has launches, not accept counts.
- The lone argmax pair at the start of warm requests (seq 6805-6806, 23291-23292) comes 3.0 s after the previous request and before the first head step. `D` lines report 0.2 ms to first token for A300_warm and A1100_warm. Inferred: the first token of a fully cached prompt is an argmax over saved logits. Call site not identified.

### 2. One MTP draft step (head, layer record 48), decode shape n = 1 (observed, seq 2587..2625)

Same 39 kernels in the same order at the other six steps listed above (offsets +0..+38 from the embed). "r5 step" is the row in `mtp-head-r5.md` section 1.1; "site" is the r5 VMA where r5 gives one.

| off | seq | kernel (instance) | grid | r5 step, site |
|---|---|---|---|---|
| 0 | 2587 | `k_embed_gather` | 1 | step 1, launch 17f35fc |
| 1 | 2588 | `k_rmsnorm_grouped` (2560, 1 group) | 1 | step 2, 17f36ca |
| 2 | 2589 | `k_lq8w<1,16,1>` (fc_embedding, N 2560) | 160 | step 3, call 17f36f6 |
| 3 | 2590 | `k_rmsnorm_grouped` (10240, 1 group) | 1 | step 4, 17f37e0 |
| 4 | 2591 | `k_lq8w<4,16,1>` (fc_hidden, 4 rows per token) | 160 | step 5, call 17f3811 |
| 5 | 2592 | `k_hyper_seed_add` | 1 | step 6, 17f3d49 |
| 6 | 2593 | `k_rmsnorm_grouped` (4 groups) | 4 | layer, first hc norm (`0189abe0`, 179adc7 class) |
| 7 | 2594 | `k_hc8_fused<1>` | 244 | layer hc stage; r5 listed `k_hc6_fused<7>` (179f13c); this instance is not in r5, site unmapped |
| 8 | 2595 | `k_lq8w<1,16,1>` | 40 | layer, unmapped |
| 9 | 2596 | `k_split_ixqk` | 3 | `018b86d0`, 17b8871 |
| 10 | 2597 | `k_rmsnorm_grouped` | 4 | `018b86d0`, 17b897b class |
| 11 | 2598 | `k_rope_w` | 1 | `018b86d0` helper, not in r5 |
| 12 | 2599 | `k_ix_gather_k` | 1 | `018b86d0` helper, not in r5 |
| 13 | 2600 | `k_lq8w<1,16,1>` (attention input projection) | 768 | unmapped |
| 14 | 2601 | `k_split_qgate` | 1 | `018b86d0`, 17b8ff7 |
| 15 | 2602 | `k_lq8w<1,16,1>` | 32 | attention projection, unmapped |
| 16 | 2603 | `k_lq8w<1,16,1>` | 32 | attention projection, unmapped |
| 17 | 2604 | `k_qk_prep` | 27 | `018b86d0`, 17bb2ae |
| 18 | 2605 | `k_attn_fd<false>` | 16 x 2 | `018b86d0`, one of 17bbaab, 17bc390, 17bc5e4 (not recoverable from the trace) |
| 19 | 2606 | `k_attn_fd_merge` | 1 x 24 | `018b86d0`, one of 17bbbc0, 17bc4a5, 17bc6fe |
| 20 | 2607 | `k_lq8w<1,32,2>` (attention out) | 640 | unmapped |
| 21 | 2608 | `k_hc8_fused3<1>` | 248 | second hc stage, unmapped |
| 22 | 2609 | `k_lq8w<1,16,1>` (shared expert) | 40 | unmapped |
| 23 | 2610 | `k_lq8w<1,16,1>` (shared expert) | 40 | unmapped |
| 24 | 2611 | `k_glu` | 3 | shared expert gate; not `k_glu_slots` (17dd3a8) |
| 25 | 2612 | `k_lq8w<1,16,1>` (shared expert down) | 160 | unmapped |
| 26 | 2613 | `k_route_logits_sg1<8,0>` | 65 | `018d7510`, see residual 3 |
| 27 | 2614 | `k_route_topk_w` | 1 | `018d7510` `LAB_018d853d`, see residual 2 |
| 28 | 2615 | `k_exp_gu_w<16,2,0>` | 400 | MoE; not r5's `k_expert_h` (17da13d) |
| 29 | 2616 | `k_exp_dn_w<0>` | 320 | MoE; not r5's `k_expert_out` (17da274) |
| 30 | 2617 | `k_hc_scatter` | 1 | `018ab0a0`, 17ab27f |
| 31 | 2618 | `k_rmsnorm_grouped` (4 groups) | 4 | step 8, 17f3e86 |
| 32 | 2619 | `k_lq8w<1,32,2>` (mixer down, N 320) | 80 | step 9, call 17f3eb2 |
| 33 | 2620 | `k_scale_silu` | 2 | step 10, 17f3fa6 |
| 34 | 2621 | `k_lq8w<1,16,1>` (mixer up, N 10240) | 640 | step 11, call 17f3fd2 |
| 35 | 2622 | `k_hc_mixed_sig` | 1 | step 12, 17f4272 (default fused path; no `k_sigmoid`/`k_hc_mixed` pair) |
| 36 | 2623 | `k_lq6k<1,16,1,true>` (lm_head, 248,320 rows = 15,520 x 16) | 15520 | step 13, call 17f4298 |
| 37 | 2624 | `k_argmax_part` | 64 | step 14, 17f4d02; 64 = default `HALOGEN_ARGMAX_BLOCKS` 0x40 |
| 38 | 2625 | `k_argmax_final` | 1 | step 14, 17f4f9b |

Checks against r5: steps 1-6 and 8-14 appear in r5's order with r5's kernels (step 12 as `k_hc_mixed_sig`, the default). The layer body (offsets 6..30, 25 launches, `k_lq8w` and the attention and MoE kernels between steps 6 and 8) is the interior r5 could only bound. Step 13 is `lq6k<1,16,1,true>`, the same instance that ends T(1) at seq 6761, so the head reuses the trunk lm_head; both steps use `k_embed_gather`, so it also reuses the trunk embedding (r5 had both from static code, now observed).

Names in r5's address-ordered lists that appear nowhere in the 106 `kname` rows of the run: `k_kv_scatter_rows`, `k_sel_tilemask`, `k_hcup4w`, `k_lq6w_grp`, `k_hc6_fused`, `k_route_topk_w_grp`, `k_expert_h`, `k_expert_out`. `k_lq4w` exists in the run (target layers) but the head never launched it. r5's lists are supersets, as r5 warned.

### 3. Head step, prompt-length shape (n = 300, 800, 1100; observed, seq 2509..2586)

Kernel names are identical for the three (78 launches, 16 `hipblasLtMatmul`, python count over seq 2509..2586); grids scale with n. Order for n = 300:

- 2509 `k_embed_gather` (300); 2510 `k_rmsnorm_grouped`; 2511-2512 `k_deq_q8g64` + MM (fc_embedding); 2513 `k_rmsnorm_grouped`; 2514-2515 `k_deq_q8g64` + MM (fc_hidden); 2516 `k_hyper_seed_add`.
- Layer, 2517..2574: `k_rmsnorm_grouped` (1200), `k_deq_q8g64` x2 + MM, `k_hc_di_epi` (380), `k_deq_q8g64` + MM, `k_hc_mixed_sig`, `k_deq_q8g64` + MM, `k_split_ixqk`, `k_rmsnorm_grouped`, `k_rope_w`, `k_ix_gather_k`, `k_block_pool`, `k_rmsnorm_grouped`, `k_rope_blocks`, `k_deq_q8g64` + MM, `k_split_qgate`, 2 x (`k_deq_q8g64` + MM), `k_rmsnorm_grouped` x2, `k_rope_w` x2, `k_attn_fa2<2,1,true,5,1,true>` (grid 20, block 768), `k_deq_q8g64` + MM, `k_hc_scatter_norm<false>`, `k_deq_q8g64` x2 + MM, `k_hc_di_epi`, `k_deq_q8g64` + MM, `k_hc_mixed_sig`, shared expert (2 x (`k_deq_q8g64` + MM), `k_glu` 750, `k_deq_q8g64` + MM), `k_linear_f32_q4` (1 x 300, 2563), `k_sigmoid_f32` (2564), `k_route_logits_x32_m<8>` (64 x 300, 2565), `k_route_topk_w` (38, 2566), `k_route_hist`, `k_route_scan`, `k_route_scatter`, `k_q4moe_items<64>`, `k_q4moe<1,0,64,..>` (2790), `k_q4moe<1,2,64,..>` (5580), `k_slot_moe_sum_rows` (5 x 300), `k_hc_scatter` (300, 2574).
- Steps 8-12: 2575 `k_rmsnorm_grouped`, 2576-2577 `k_deq_q8g64` + MM (down), 2578 `k_scale_silu`, 2579-2580 `k_deq_q8g64` + MM (up), 2581 `k_hc_mixed_sig`. Step 13: 2582 `k_lq6k<1,16,1,true>` (15,520 blocks, one row). Step 14: 2583-2584 argmax pair. Then 2585-2586 a second argmax pair, 31 ms after the first and with no lm_head before it (16.7 ms after in R2; seq 13687-13689).
- `k_route_topk_w` grids: 38 / 100 / 138 for n = 300 / 800 / 1100 (ceil(n/8)).

### 4. One target decode step (observed; exact for T(1), seq 5424..6763)

1340 launches:

- Prefix (3): `k_embed_gather` 5424, `k_hyper_seed` 5425, `k_rmsnorm_grouped` (grid 4) 5426.
- 48 layers. Layer starts are the `k_lq4w<1,32,2,false,1,0>` (grid 80) before every second `k_hcup4` (96 `hcup4` per step in all three widths). Layer types from `k_attn_fd` presence: attention on layers 3, 7, 11, ... 47 (12 layers), DeltaNet elsewhere (36 layers), pattern `DDDA` x 12. Starts: layer 0 at seq 5427, 1 at 5470, 2 at 5497, 3 at 5524, 6 at 5611, 7 at 5636, 11 at 5743, 15 at 5852, 47 at 6724.
- Launches per layer (python segmentation): layer 0 43, DeltaNet 27 (17 layers), DeltaNet 25 (18 layers), attention 30 (3 layers), 31 (3), 32 (3), 33 (2 layers: 3 and 43), layer 47 = 33 + 7 tail. Check: 3 + 43 + 17 x 27 + 18 x 25 + 3 x 30 + 3 x 31 + 3 x 32 + 2 x 33 + 40 = 1340.
- DeltaNet layer (27, layer 1, seq 5470..5496): `lq4w<1,32,2,..,1,0>` 80, `hcup4<1,0>` 160, `lbfw<1,4>` 1, `inject_gate` 1, `lq4w<1,16,1,..>` 640, `lq4w<1,16,1,..>` 384, `lbfw<1,2>` 6, `lbfw<1,2>` 6, `dn_step_pre` 41, `dn_recur_step<0,false>` 48, `lq8s32<1,32,2>` 640, `hc_scatter_norm<false>` 4, `lq4w<1,32,2,..>` 80, `hcup4` 160, `lbfw<1,4>`, `inject_gate`, `lq8s32<1,16,1>` 40, `lq8s32<1,16,1>` 40, `glu` 3, `lq8s32<1,16,1>` 160, `linear_f32` 1, `sigmoid_f32` 1, `route_logits_x32_m<8>` 64, `route_topk_w` 1, `exp_gu_w<16,2,0>` 400, `exp_dn_w<0>` 320, `hc_scatter_norm<false>` 4. The 25-launch layers swap the `lq8s32` calls for `lq4w` ones and use `lq4w_pair<1,16,0>` (40) for the shared expert (layer 6, seq 5611..5635).
- Layer 0 (43, seq 5427..5469) is the DeltaNet layer with its last `hc_scatter_norm` replaced by `k_hc_scatter` plus 16 extras: `k_ngram_hist`, `k_ngram_segstart`, `k_ngram_ids`, `k_ngram_ctx_update`, `k_gather_iq4nl`, `lq4w<1,16,1>` 640, `rmsnorm_grouped`, `lq4w<1,16,1>` 160, `rmsnorm_grouped`, `k_ple_gate`, `k_ple_gv`, `rmsnorm_grouped`, `k_ple_out_p`, `k_ple_conv_state_update`, `k_add_bf16`, `rmsnorm_grouped`.
- Attention layer (33, layer 3, seq 5524..5556): `lq4w` 80, `hcup4` 160, `lbfw<1,4>`, `inject_gate`, `lbfw<1,2>` 80, `split_ixqk` 3, `rmsnorm_grouped` 4, `rope_w` 1, `ix_gather_k` 1, `lq6k<1,16,1,false>` 768, `split_qgate` 1, `lq8s32<1,16,1>` 32 x 2, `qk_prep` 27, `attn_fd<false>` 16 x 2, `attn_fd_merge` 1 x 24, `lbfw<1,4>` 320, `hc_scatter_norm` 4, then the MoE half as for DeltaNet. Layers 7 and 11 (30 and 32 launches, seq 5636..5665 and 5743..5774) use `lq4w<1,16,1,false,5,0>` 768 for the input projection and have no `split_qgate`; this is the `5,0` template family of r5's 17ba745 `k_lq4w<7,16,1,false,5,0>`, seen here at width 1. Layers 15 and 47 use the `lq6k` form.
- Tail (seq 6757..6763): `lbfw<1,4>` 40 (mixer down), `scale_silu` 2, `linear_m<8>` 1280 (mixer up), `hc_mixed_sig` 1, `lq6k<1,16,1,true>` 15520 (lm_head), `argmax_part` 64, `argmax_final` 1.
- T(2) and T(3) keep the layer skeleton with `lq4w<n,..>`, `hcup4<n,0>` and grids scaled to n, one `ix_gather_k` launch per row (two in T(2), three in T(3)), `route_logits_rows<8,n>` instead of `x32_m<8>`, `linear_f32` grid (1, n), and the two trailing state updates.

Contrast at width 1: the target step has no `lq8w`; the head has no `lq4w`, `lq8s32`, `lbfw`, `hcup4`, `dn_*`, `linear_f32`, `sigmoid_f32` or `route_logits_x32_m`. The only Linear kernel they share is `lq6k<1,16,1,true>` (lm_head, seq 2623 and 6761).

## Verdict: residual 1 (head Linear kernels) CLOSED

Observed. At n = 1 the head's Linear kernels are all `k_lq8w` (three instances) plus the shared `lq6k` for lm_head; no `k_lq4w*`, no `k_lq6w*`, no `hipblasLtMatmul`. 13 of the 39 launches:

| instance | grid | role (offset above) | seq |
|---|---|---|---|
| `k_lq8w<1,16,1>` | 160 | fc_embedding (2) | 2589 |
| `k_lq8w<4,16,1>` | 160 | fc_hidden (4) | 2591 |
| `k_lq8w<1,16,1>` | 40 | pre-attention (8) | 2595 |
| `k_lq8w<1,16,1>` | 768 | attention input (13) | 2600 |
| `k_lq8w<1,16,1>` | 32, 32 | attention (15, 16) | 2602, 2603 |
| `k_lq8w<1,32,2>` | 640 | attention out (20) | 2607 |
| `k_lq8w<1,16,1>` | 40, 40, 160 | shared expert (22, 23, 25) | 2609, 2610, 2612 |
| `k_lq8w<1,32,2>` | 80 | mixer down 10240 to 320 (32) | 2619 |
| `k_lq8w<1,16,1>` | 640 | mixer up 320 to 10240 (34) | 2621 |
| `k_lq6k<1,16,1,true>` | 15520 | lm_head, shared with trunk (36) | 2623 |

The `k_lq8w` family matches r5's `k_lq8*` guess for the q8g64 head tensors; `k_lq4w*` and `k_lq6w*` do not run in the head in this capture. At prompt widths (n >= 300) the same tensors run as `k_deq_q8g64` then `hipblasLtMatmul` (16 pairs, seq 2511..2580). Static launch sites for `k_lq8w` stay unmapped: r5 lists none, they sit behind `018a7520` and its helpers.

## Verdict: residual 2 (router top-k path, `+0x723`/`+0x748`) CLOSED for the path that ran; flag values unread

Observed: the head layer ran `k_route_topk_w` at every width: grid 1 at n = 1 (seq 2614) and grid ceil(n/8) = 38 / 100 / 138 at n = 300 / 800 / 1100 (seq 2566, and offset +57 in the larger prompt-length steps). No `k_route_topk_w_grp` or plain `k_route_topk` exists among the 106 `kname` rows; the only topk kernel in the run is `k_route_topk_w`.

That is the path r5 predicted (`LAB_018d853d` with `DAT_019fcd68 == 1`, `D:018d7510.c:445-455`). The flags did not decide it: the grouped path is gated by `6 < local_ee4 - 2` (`D:018d7510.c:380`, unsigned), so n = 1 wraps to a large value and n >= 300 exceeds 8; the head only ran n in {1, 300, 800, 1100}, so the flag test at `:381`/`:395` was not reached. They matter only for n in 2..8, a width the head did not have here (the n = 2 and n = 3 steps in the trace are target steps, which also chose `k_route_topk_w`, for example seq 2626..3991). The stored flag values stay unread (r5 inferred zero).

## Verdict: residual 3 (router-logit kernel) CLOSED

Observed; the kernel depends on n, and it is not plain `k_route_logits_x32` as r5 guessed from the one visible launch:

- n = 1 decode: `k_route_logits_sg1<8,0>`, grid 65, seq 2613 (offset +26 in every head decode step). No `k_linear_f32`/`k_sigmoid_f32` precedes it in the head, though both precede the router in every target layer (for example seq 5447-5448 in layer 0), so the shared-expert gate appears to be fused into this kernel. Inferred; the 65th block over the trunk's 64 fits an added gate block.
- n = 300 / 800 / 1100: `k_route_logits_x32_m<8>`, grid 64 x n, seq 2565 (offset +56 in the larger steps), preceded by `k_linear_f32_q4` (1 x n, seq 2563) and `k_sigmoid_f32` (seq 2564). The instance is `x32_m<8>`, the multi-row template.
- Target steps for contrast: width 1 `x32_m<8>` (grid 64) with `linear_f32` and `sigmoid_f32`; widths 2 and 3 `k_route_logits_rows<8,2>` and `<8,3>` (grid 64).

Static tie (inferred): the sg1 branch is in `work/decomp/flash_serve/018d7510.c` from line 201 (`if (local_eaa == '\x01')`), guarded by `HALOGEN_ROUTE_SG` (default true when unset), a nonzero `*(local_e90 + 0x688)` and n <= 8 (`0xfffffff7 < uVar5 - 9`); `case 1` jumps to `code_r0x018d81b7`. The decompiled launch configs for cases 2-4 are `0x100000041` (65 blocks), matching the observed grid. The stub `k_route_logits_sg1<8,0>` is at `analysis/function-map-kernel-stubs.tsv:911` (address 01941d10). Why the head record satisfies `local_eaa` and `+0x688` while the trunk records do not is inferred from the missing `linear_f32`/`sigmoid_f32`, not read from a record.

## Residuals

1. Static launch sites for the `k_lq8w` instances, `k_hc8_fused<1>`, `k_hc8_fused3<1>`, `k_rope_w`, `k_ix_gather_k`, `k_glu`, `k_exp_gu_w`, `k_exp_dn_w`: none is in r5's direct-site lists; they sit inside helpers. Needs a launch-site pass over `018a7520` and the hc and decode-MoE paths.
2. Which of the three `k_attn_fd` / `k_attn_fd_merge` sites r5 lists ran is not recoverable (same kernel, same stub).
3. The second argmax pair (after the head step at n >= 300 and at request tails) and the lone pair at warm-request start have no identified call site. Gaps are 3 to 31 ms after the first argmax, and 3.0 s at warm starts. The cached-logits reading is inferred.
4. Accept counts per round and the remaining-budget cap on verify width are not in the trace (inferred only). A field map of the `D` line would settle it.
5. The 42-launch head variant's block-pool path is attributed to a new indexer block (inferred); the block size is not checked.
6. Flag values `+0x723`, `+0x748` of record 48 and the `local_eaa` / `+0x688` conditions are unread; they matter only at head widths 2..8.
7. r5 residuals 4 (hand-decoded VOP3 tie rules), 6 (0x300 B block at `+0x25880`) and 7 (draft-vocab items) are untouched by this capture.
