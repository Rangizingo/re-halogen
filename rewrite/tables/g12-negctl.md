# Kernel match table (U22, PLAN.md p7-12)

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_110k_add_bf16EPKtS2_Pti | add | add_out.json seed=1 N=4096 | MISMATCH n=4160 diff=4085 max_abs=49497 max_ulp=49497 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_19k_sigmoidEPKtPtl | sigmoid | sigmoid_bf16.json seed=1 N=4096 | MISMATCH n=4160 diff=2823 max_abs=73 max_ulp=73 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_15k_gluEPKtS2_Ptl | glu | glu.json seed=1 N=4096 | MISMATCH n=4160 diff=3589 max_abs=128 max_ulp=128 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_114k_embed_gatherEPKtPKiPt | embed | embed_gather.json seed=1 N=5 | MISMATCH n=12864 diff=12800 max_abs=1 max_ulp=1 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_113k_gather_rowsEPKtPKiiPt | gather | gather_rows.json seed=1 N=19 | MISMATCH n=48704 diff=35798 max_abs=49280 max_ulp=49280 first_diff=5120 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_slot_sum_rowsEPKtPKiPKfPt | slot | slot_sum_rows.json seed=1 N=3 | MISMATCH n=7744 diff=7680 max_abs=32768 max_ulp=32768 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_111k_ngram_idsEPKxPKiS2_S2_S2_iPx | ngram | ngram_ids.json seed=1 N=7 | MISMATCH n=128 diff=112 max_abs=2.52559e+06 max_ulp=2.52559e+06 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_argmax_f32EPKfiPiNS_8AmxMasksE | argmax | argmax_f32_ties.json seed=1 N=4096 | MISMATCH n=8 diff=4 max_abs=4095 max_ulp=4095 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_117k_rmsnorm_groupedEPKtS2_Ptii | rmsnorm | rmsnorm_grouped_a.json seed=1 N=5 | MATCH n=2624 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_111k_act_denseEPKtiiPfS3_ | act | act_dense.json seed=1 N=100 | MISMATCH n=216 diff=100 max_abs=32.1953 max_ulp=2.19653e+09 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_111k_ple_out_pEPKtS2_S2_S2_iPt | ple | ple_out_p.json seed=1 N=5 | MISMATCH n=51264 diff=50494 max_abs=34324 max_ulp=34324 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen15k_q_softmax_f32EPKfPfif | q | q_softmax.json seed=1 N=1000 | MISMATCH n=1008 diff=1000 max_abs=0.000309484 max_ulp=4.34531e+06 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_117k_ix_scores_stripEPKtS2_Pfliiii | ix | ix_scores_strip.json seed=1 N=100 | MISMATCH n=4344 diff=650 max_abs=1.06897 max_ulp=8.38861e+06 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_117k_rmsnorm_groupedEPKtS2_Ptii | rmsnorm | rmsnorm_grouped_a.json seed=1 N=5 | MISMATCH n=2624 diff=2560 max_abs=16165 max_ulp=16165 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_117k_rmsnorm_groupedEPKtS2_Ptii | rmsnorm | rmsnorm_grouped_edge.json seed=1 N=5 | MISMATCH n=2624 diff=2558 max_abs=222 max_ulp=222 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_rmsnorm_gatedEPKtS2_S2_Pti | rmsnorm | rmsnorm_gated_edge.json seed=1 N=5 | MISMATCH n=2624 diff=2560 max_abs=222 max_ulp=222 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_117k_rmsnorm_gated_wEPKtS2_S2_Ptli | rmsnorm | rmsnorm_gated_w_edge.json seed=1 N=13 | MISMATCH n=1728 diff=1664 max_abs=222 max_ulp=222 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_ple_gateEPKtS2_Pf | ple | ple_gate_edge.json seed=1 N=3 | MISMATCH n=20 diff=12 max_abs=0.00216228 max_ulp=1.33806e+07 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_117k_ix_select_stripEPKfliiPiS3_i | ix | ix_select_strip_ties.json seed=1 N=300 | MISMATCH n=731 diff=279 max_abs=303 max_ulp=303 first_diff=205 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_117k_ix_select_sliceILi8EEEvPKfliiiiPiS4_S4_S4_ | ix | ix_select_slice8_multi_ties.json seed=1 N=20000 | MISMATCH n=796 diff=20 max_abs=16802 max_ulp=16802 first_diff=64 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_117k_ix_select_mergeEPKfliiiPKiS4_PiS5_ | ix | ix_select_merge_one_ties.json seed=1 N=30000 | MISMATCH n=211 diff=51 max_abs=17671 max_ulp=17671 first_diff=143 | bitwise | 2026-10-10 |
