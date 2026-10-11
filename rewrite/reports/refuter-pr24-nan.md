# Refuter PR #24: NaN/special-value attack across all kernel families

Scope (orchestrator): same attack as the previous refuter's ht_gemv finding, on one representative instance of every rewrite/kernels family except ht_gemv. Head 88b9b35. Run on Ai-core in ~/refuter-pr24, seeds 31,32,33, committed sizes (smallest plus a middle one from match-table.md for the chosen spec), `compare.py --no-table`, bitwise.

## Method

- Driver: `refuter-pr24-nan-files/drive2.py`. For each `rewrite/kernels/*.hip`, picks the first committed spec (name order) that has a float-dtype input buffer, keeps its original fill (rand range, const, iota) and replaces it with a file fill where 0.4% of elements are specials: +inf, -inf, -0, +1e-40, -1e-40, +3e38, -3e38, NaN (same list as the previous refuter's mksp.py). Every float input buffer of the spec gets this, so NaN is in all float inputs at once. Output buffers are not touched, except in-place buffers with a rand fill (vis_rope qkv).
- Vendor kernel name = first MATCH instance of that spec in match-table.md. lq4w gets its `-DLQ_*` flags from the mangled name (as lq4w_run_all.sh does).
- Skipped: ht_gemv.hip, ht_gemv_b.hip (another agent), smpg_common.h (header), attn_qs_prime.hip and dnc_scan3w_probe_d_prime.hip (priming helpers, not instances).
- Results (one JSON line per family): `refuter-pr24-nan-files/nan_results.jsonl` (rows for lq4w and vis_rope in it are from the first pass; final values below).
- Caveat on MATCH: a MATCH proves vendor and rewrite agree on that spec; it does not prove NaN reached the output (not measured per family). Seeds 31..33 share one file per buffer (only SEED-driven expressions change).

## Totals (141 families run)

| result | families |
| --- | --- |
| MATCH (all 6 runs bitwise) | 116 |
| MISMATCH | 5 |
| UNTOUCHED-inconclusive | 3 |
| not applicable (integer-only inputs) | 15 |
| ERROR | 0 |

## MISMATCH (5), all NaN sign-bit / payload only

Every differing element is NaN in both vendor and rewrite; only the sign bit differs. Same signature as the ht_gemv finding (AMD returns the first NaN operand of a commutative op; the two codes order operands or generate NaNs differently; not checked in ISA).

Reproducer specs: `refuter-pr24-nan-files/specs/nan/<name>__nan.json`, input data `refuter-pr24-nan-files/data/nan/`. The specs hold absolute Ai-core paths (`/home/rangizingo/refuter-pr24/...`); copy the two dirs there to rerun. Command: `compare.py <spec> --kernel <kernel> --seeds 31,32,33 --sizes <sizes> --no-table`.

| family | kernel | spec | first failing case | diff | first_diff | vendor vs rewrite |
| --- | --- | --- | --- | --- | --- | --- |
| dnc_ut | `_ZN7halogen12_GLOBAL__N_18k_dnc_utEPKtPKfS4_S4_PfS5_i` | specs/nan/dnc_ut__nan.json | seed=31 N=2 (out1, f32, n=1572864) | 281 | 15991 | 0xffc00000 vs 0x7fc00000 |
| dnc_ut3 | `_ZN7halogen12_GLOBAL__N_19k_dnc_ut3EPKtPKfS4_S4_PfS5_i` | specs/nan/dnc_ut3__nan.json | seed=31 N=2 (out1, f32) | 211 | 18372 | 0xffc00000 vs 0x7fc00000 |
| ht_deq | `_ZN7halogen12_GLOBAL__N_113k_ht_deq_origILi3ELb0ELi8EEEvPKjiPKDF16_S5_Pti` | specs/nan/ht_deq_orig_b3_f0_t256__nan.json | seed=31 N=3 (out, bf16, n=98368) | 2 | 94385 | 0x7fc0 vs 0xffc0 (also 97457) |
| i4r_gu | `_ZN7halogen12_GLOBAL__N_112k_i4r_gu_epiEPKfiPKDF16_S4_PDF16_` | specs/nan/i4r_gu_epi__nan.json | seed=31 N=101 (out, f16, n=64704) | 24 | 6716 | 0x7e00 vs 0xfe00; adjacent element 6717 is 0xfe00 vs 0x7e00 (pairs swapped) |
| vis_rope | `_ZN7halogen12_GLOBAL__N_110k_vis_ropeEPfii` | specs/nan/vis_rope_b32__nan.json | seed=31 N=64 (qkv f32 in place, n=221184) | 7 | 5049 | 0x7fc00000 vs 0xffc00000 (also 5472, 13982, 90754, 167065, 208254) |

Counts above are from seed 31 (first size); all 6 runs of each spec MISMATCH (seeds 31,32,33 at two sizes). Per-element dumps came from `nandiff.py` (same dir). Severity low (NaN either way), but the "bitwise" claim does not hold for NaN inputs in these five, and the i4r_gu pair swap suggests the rewrite's operand order differs from the vendor's in the epilogue (hypothesis, ISA not read).

## UNTOUCHED-inconclusive (3)

On rows holding a non-finite value, both vendor and rewrite leave the output buffers at their pre-launch fill (diff=0), so the harness reports UNTOUCHED. Nothing can be concluded about NaN handling. Matches the previous refuter's note on k_sample_f32.

| family | spec | kernel | note |
| --- | --- | --- | --- |
| sample | sample_a.json | `_ZN7halogen12k_sample_f32EPKfPKNS_9SampleRowEPiPfS6_i` | 3 of 6 runs MATCH, N=100003 runs UNTOUCHED (n=48, diff=0) |
| resid | resid_am0.json | `_ZN7halogen18k_resid_sample_f32EPKfS1_PKNS_9SampleRowEPiPfS5_ii` | all 6 UNTOUCHED |
| esc | esc_arbiter_dec.json | `_ZN7halogen12_GLOBAL__N_113k_esc_arbiterEPKfiPiNS_8AmxMasksEPKi` | all 6 UNTOUCHED |

## Not applicable (15)

No committed spec for these families has a float-dtype input; inputs are u8/u16/u32/i32/i64 bit data (random bits already include NaN patterns only where the kernel reinterprets them, not tested here): count, deq_q4cp, deq_q6g64, deq_q6k, dnc_cmp, hist, ht_items, ngram, q4moe_items, route_hist, route_scan, route_scatter, sel, split, vis_patchify.

## MATCH (116 families)

accept act add argmax attn_fa2 attn_fa attn_fd attn attn_qs attn_qs_c attn_sel bf16 block cc conv1d conv deq_aff deq_q8g64 deq_q8s32 dn_conv dn_fused16 dn_gates dn_recur dn_step dnc_att dnc_att2 dnc_att6 dnc_att7 dnc_att8 dnc_att9 dnc_prep dnc_scan3_b dnc_scan3 dnc_scan3w dnc_scan3w_probe_d dnc_scan3w_probe dnc_scan3x dnc_scan3x_b dnc_ut2 dnc_ut4 dnc_ut5 dv embed exp_dn exp_gu expert dv_quant i4r_rows_b gather glu hc6 hc8 hc_di hc_fold hc_mixed hc_scatter hcup4 hcup4w hcup6 hcup8 ht_lin ht_moe ht_rot ht_tg hyper i4r_fold i4r_rows inject ix scale kv lbfw linear linear_f32 linear_m linear_mt linear_q4 lq4w lq6k lq6w lq8s32 lq8w_grp lq8w moe nll pen_rep pen_scatter ple q4moe q qk rmsnorm rope route_logits route_topk scatter sigmoid slot smpg_accept smpg_level smpg_mass smpg_max smpg_pick smpg_qwrite smpg_rmass smpg_rpick vis_acc vis_add vis_attn vis_bias vis_ln vis_narrow vis_ohead vis_pos vis_qkv vis_softmax

## Side finding (separate from the NaN attack, from the first refuter task, not yet written up)

attn_fa2 instances with A=2 (checked on `k_attn_fa2<2,1,true,5,1,true>`) differ from the vendor when scores or V hold non-NaN extremes, not only NaN. Cause (supported by experiment): the rewrite's tile loop bound is per 16-row sub-tile (`ntiles` from `row0 + 15 + base`, attn_fa2.hip:97); the vendor appears to run the block-wide bound, so sub-tile 0 also processes the fully masked trailing tile. Effects: (a) V=+-inf in a masked key position gives NaN in the vendor (0*inf) and a finite value in the rewrite; (b) with every unmasked score s in (-1e30, -1e29) the extra tile's alpha is 0 (m_old below the -1e29 test at the sentinel compare), the vendor row ends as 0*(1/0)=NaN, rewrite stays finite. A=1 instance `<1,1,true,5,1,true>` MATCHes the same inputs. A scratch copy with `lastpos += 16*(1-sub)` made k=1e30 and the mixed-special spec MATCH (probe only, not committed). Full write-up with specs follows in the main refuter report.
