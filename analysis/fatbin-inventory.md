# fatbin inventory — clang offload bundle v2 in flash_serve

Source: `binaries/flash_serve`, bundle at file offset 0x52000, 23,739,040 B. Carved by `scripts/carve_fatbin.py`; machine-readable manifest: `work/fatbin-manifest.json`.

| code object | file offset | size | ELF e_machine | kernel symbols |
|---|---|---|---|---|
| `codeobj_53000.hsaco` | 0x53000 | 17,765,424 B | 0xe0 | 2982 |
| `codeobj_1146000.hsaco` | 0x1146000 | 3,549,744 B | 0xe0 | 350 |
| `codeobj_14aa000.hsaco` | 0x14aa000 | 422,544 B | 0xe0 | 34 |
| `codeobj_1513000.hsaco` | 0x1513000 | 1,976,992 B | 0xe0 | 124 |

sha256: `codeobj_53000.hsaco` 18937428b544e8a5…; `codeobj_1146000.hsaco` 3e44d5a242b83af3…; `codeobj_14aa000.hsaco` 5a82ad4cb1c64b6b…; `codeobj_1513000.hsaco` dbaafa57d9b3bd60…

## Kernel symbol families per code object (top 12 by count)

### codeobj_53000.hsaco (2982 symbols)

- `k_add_bf16` ×4
- `k_accept_prob_f32` ×2
- `k_act_dense` ×2
- `k_act_rows` ×2
- `k_argmax_f32` ×2
- `k_argmax_f32_legacy` ×2
- `k_argmax_final` ×2
- `k_argmax_part` ×2
- `k_attn` ×2
- `k_attn_fd_merge` ×2
- `k_attn_sel` ×2
- `k_bf16_to_f32` ×2
- `k_block_pool` ×2
- `k_cc_affine` ×2
- `k_cc_sub` ×2
- `k_conv1d_step` ×2
- `k_conv_state_from_prefill` ×2
- `k_count_diff_u32` ×2
- `k_deq_q4cp` ×2
- `k_deq_q4cp_f32` ×2
- `k_deq_q4cp_sel` ×2
- `k_deq_q4cp_v8` ×2
- `k_deq_q6g64` ×2
- `k_deq_q6k` ×2
- `k_deq_q8g64` ×2
- `k_deq_q8s32` ×2
- `k_dn_conv_rows` ×2
- `k_dn_gates` ×2
- `k_dn_step_pre` ×2
- `k_dv_quant` ×2
- `k_dv_quant8` ×2
- `k_dv_scatter` ×2
- `k_embed_gather` ×2
- `k_embed_image` ×2
- `k_esc_arbiter` ×2
- `k_exp_dn_aff` ×2
- `k_exp_dn_w2` ×2
- `k_exp_dn_w_q8` ×2
- `k_expert_h` ×2
- `k_expert_h_grouped` ×2
- `k_expert_h_grouped_warp` ×2
- `k_expert_h_q4` ×2
- `k_expert_h_warp` ×2
- `k_expert_out` ×2
- `k_expert_out_grouped` ×2
- `k_expert_out_grouped_warp` ×2
- `k_expert_out_q4` ×2
- `k_expert_out_warp` ×2
- `k_f32_to_bf16` ×2
- `k_gate_mul` ×2
- `k_gather_fp8` ×2
- `k_gather_iq4nl` ×2
- `k_gather_rows` ×2
- `k_gather_slots_h` ×2
- `k_glu` ×2
- `k_glu_rows` ×2
- `k_glu_slots` ×2
- `k_hc_di_epi` ×2
- `k_hc_fold` ×2
- `k_hc_mixed` ×2
- `k_hc_mixed_sig` ×2
- `k_hc_scatter` ×2
- `k_hcup4w` ×2
- `k_hist_from_cstate` ×2
- `k_ht_items_slots` ×2
- `k_ht_rot_in` ×2
- `k_ht_rot_out` ×2
- `k_ht_rot_out_f32` ×2
- `k_ht_rot_rows` ×2
- `k_ht_rot_rows_h` ×2
- `k_ht_tg_sum` ×2
- `k_hyper_seed` ×2
- `k_hyper_seed_add` ×2
- `k_i4r_fold` ×2
- `k_i4r_fold_rows` ×2
- `k_i4r_gu_epi` ×2
- `k_i4r_rows_epi` ×2
- `k_i4r_rows_fl` ×2
- `k_i4r_rows_fl_o` ×2
- `k_i4r_rows_rot` ×2
- `k_idx_scores` ×2
- `k_idx_select` ×2
- `k_inject_gate` ×2
- `k_inject_sum` ×2
- `k_ix_gather_k` ×2
- `k_ix_scores_strip` ×2
- `k_ix_select_merge` ×2
- `k_ix_select_strip` ×2
- `k_kv_scatter_rows` ×2
- `k_linear` ×2
- `k_linear_f32` ×2
- `k_linear_f32_q4` ×2
- `k_linear_q4` ×2
- `k_linear_q8` ×2
- `k_linear_q8s32` ×2
- `k_moe_sum` ×2
- `k_ngram_ctx_update` ×2
- `k_ngram_hist` ×2
- `k_ngram_ids` ×2
- `k_ngram_segstart` ×2
- `k_nll_final` ×2
- `k_nll_init` ×2
- `k_pen_rep_rows_f32` ×2
- `k_ple_conv_state_update` ×2
- `k_ple_gate` ×2
- `k_ple_gv` ×2
- `k_ple_ids_rows` ×2
- `k_ple_out` ×2
- `k_ple_out_p` ×2
- `k_ple_out_rows` ×2
- `k_q_softmax_f32` ×2
- `k_qk_prep` ×2
- `k_resid_sample_f32` ×2
- `k_rmsnorm_gated` ×2
- `k_rmsnorm_gated_w` ×2
- `k_rmsnorm_grouped` ×2
- `k_rope` ×2
- `k_rope_blocks` ×2
- `k_rope_blocks_mrope` ×2
- `k_rope_blocks_mrope_yarn` ×2
- `k_rope_blocks_yarn` ×2
- `k_rope_w` ×2
- `k_rope_w_mrope` ×2
- `k_rope_w_mrope_yarn` ×2
- `k_rope_w_yarn` ×2
- `k_rope_yarn` ×2
- `k_route_hist` ×2
- `k_route_logits_w32` ×2
- `k_route_logits_x32` ×2
- `k_route_scan` ×2
- `k_route_scatter` ×2
- `k_route_topk` ×2
- `k_route_topk_w` ×2
- `k_sample_f32` ×2
- `k_scale_silu` ×2
- `k_scatter_slots_dn` ×2
- `k_sel_tilemask` ×2
- `k_sigmoid` ×2
- `k_sigmoid_f32` ×2
- `k_slot_moe_sum_rows` ×2
- `k_slot_sum` ×2
- `k_slot_sum_rows` ×2
- `k_smpg_accept_final` ×2
- `k_smpg_mass` ×2
- `k_smpg_max` ×2
- `k_smpg_pick` ×2
- `k_smpg_qwrite` ×2
- `k_smpg_rmass` ×2
- `k_smpg_rpick` ×2
- `k_split_ixqk` ×2
- `k_split_qgate` ×2
- `k_vis_acc` ×2
- `k_vis_add` ×2
- `k_vis_bias` ×2
- `k_vis_bias_gelu` ×2
- `k_vis_ln` ×2
- `k_vis_narrow` ×2
- `k_vis_ohead_rows` ×2
- `k_vis_patchify` ×2
- `k_vis_pos_add` ×2
- `k_vis_qkv_split` ×2
- `k_vis_rope` ×2
- `k_vis_softmax` ×2

### codeobj_1146000.hsaco (350 symbols)

- `k_dnc_att` ×2
- `k_dnc_att2` ×2
- `k_dnc_att8` ×2
- `k_dnc_cmp_words` ×2
- `k_dnc_prep_w` ×2
- `k_dnc_scan` ×2
- `k_dnc_ut` ×2
- `k_dnc_ut2` ×2
- `k_dnc_ut3` ×2

### codeobj_14aa000.hsaco (34 symbols)


### codeobj_1513000.hsaco (124 symbols)

- `k_attn_fa` ×2
- `k_attn_qs_bt` ×2
- `k_ix_scores_loop` ×2
- `k_ix_scores_wmma` ×2
- `k_kv_bt` ×2

