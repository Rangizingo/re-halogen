# Kernel match table (U22, PLAN.md p7-12)

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_19k_deq_affILi4ELb1EEEvPKhPtNS0_6AffDevEjjl | deq_aff | deq_aff_4t_edge.json seed=1 N=64 | MISMATCH n=4096 diff=32 max_abs=0 max_ulp=0 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_19k_deq_affILi4ELb1EEEvPKhPtNS0_6AffDevEjjl | deq_aff | deq_aff_4t_edge.json seed=2 N=64 | MISMATCH n=4096 diff=32 max_abs=0 max_ulp=0 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_19k_deq_affILi4ELb1EEEvPKhPtNS0_6AffDevEjjl | deq_aff | deq_aff_4t_edge.json seed=3 N=64 | MISMATCH n=4096 diff=32 max_abs=0 max_ulp=0 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_18k_cc_subEPfPKfS3_l | cc | cc_sub.json seed=1 N=4096 | MISMATCH n=4160 diff=4096 max_abs=1999.88 max_ulp=2.29744e+09 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_18k_cc_subEPfPKfS3_l | cc | cc_sub.json seed=2 N=4096 | MISMATCH n=4160 diff=4096 max_abs=1999.7 max_ulp=2.29752e+09 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_18k_cc_subEPfPKfS3_l | cc | cc_sub.json seed=3 N=4096 | MISMATCH n=4160 diff=4096 max_abs=1998.86 max_ulp=2.29749e+09 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_118k_scatter_slots_dnEPKtiPKiPKfPf | scatter | scatter_slots_dn.json seed=1 N=5 | MISMATCH n=12864 diff=5 max_abs=108.275 max_ulp=2.21213e+09 first_diff=2559 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_118k_scatter_slots_dnEPKtiPKiPKfPf | scatter | scatter_slots_dn.json seed=2 N=5 | MISMATCH n=12864 diff=5 max_abs=109.147 max_ulp=2.21304e+09 first_diff=2559 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_118k_scatter_slots_dnEPKtiPKiPKfPf | scatter | scatter_slots_dn.json seed=3 N=5 | MISMATCH n=12864 diff=5 max_abs=104.275 max_ulp=2.20508e+09 first_diff=2559 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_19k_moe_sumEPKtS2_PKfPt | moe | moe_sum.json seed=1 N=17 | MISMATCH n=43584 diff=5 max_abs=0.125 max_ulp=1 first_diff=24650 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_19k_moe_sumEPKtS2_PKfPt | moe | moe_sum.json seed=2 N=17 | MISMATCH n=43584 diff=6 max_abs=0.125 max_ulp=1 first_diff=8073 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_19k_moe_sumEPKtS2_PKfPt | moe | moe_sum.json seed=3 N=17 | MISMATCH n=43584 diff=15 max_abs=0.125 max_ulp=1 first_diff=2987 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_114k_sel_tilemaskEPKiS2_iiiPjii | sel | sel_tilemask.json seed=1 N=100 | MISMATCH n=120 diff=3 max_abs=64 max_ulp=64 first_diff=24 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_114k_sel_tilemaskEPKiS2_iiiPjii | sel | sel_tilemask.json seed=2 N=100 | MISMATCH n=120 diff=4 max_abs=256 max_ulp=256 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_114k_sel_tilemaskEPKiS2_iiiPjii | sel | sel_tilemask.json seed=3 N=100 | MISMATCH n=120 diff=2 max_abs=256 max_ulp=256 first_diff=56 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_113k_esc_arbiterEPKfiPiNS_8AmxMasksEPKi | esc | esc_arbiter_ninf_edge.json seed=1 N=4096 | MISMATCH n=8 diff=7 max_abs=4001 max_ulp=4001 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_113k_esc_arbiterEPKfiPiNS_8AmxMasksEPKi | esc | esc_arbiter_ninf_edge.json seed=2 N=4096 | MISMATCH n=8 diff=7 max_abs=4001 max_ulp=4001 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_113k_esc_arbiterEPKfiPiNS_8AmxMasksEPKi | esc | esc_arbiter_ninf_edge.json seed=3 N=4096 | MISMATCH n=8 diff=7 max_abs=4001 max_ulp=4001 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12k_dv_scatterEPKfPKiPfi | dv | dv_scatter.json seed=1 N=100003 | MISMATCH n=100067 diff=58 max_abs=inf max_ulp=1 first_diff=3374 | bitwise | 2026-10-11 |
| _ZN7halogen12k_dv_scatterEPKfPKiPfi | dv | dv_scatter.json seed=2 N=100003 | MISMATCH n=100067 diff=68 max_abs=inf max_ulp=1 first_diff=336 | bitwise | 2026-10-11 |
| _ZN7halogen12k_dv_scatterEPKfPKiPfi | dv | dv_scatter.json seed=3 N=100003 | MISMATCH n=100067 diff=82 max_abs=inf max_ulp=1 first_diff=416 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_i4r_gu_epiEPKfiPKDF16_S4_PDF16_ | i4r_gu | i4r_gu_epi_edge.json seed=1 N=101 | MISMATCH n=64704 diff=68 max_abs=4 max_ulp=3 first_diff=102 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_i4r_gu_epiEPKfiPKDF16_S4_PDF16_ | i4r_gu | i4r_gu_epi_edge.json seed=2 N=101 | MISMATCH n=64704 diff=61 max_abs=2 max_ulp=8 first_diff=1968 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_i4r_gu_epiEPKfiPKDF16_S4_PDF16_ | i4r_gu | i4r_gu_epi_edge.json seed=3 N=101 | MISMATCH n=64704 diff=71 max_abs=2 max_ulp=7 first_diff=349 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_i4r_foldEPKfiS2_PKDF16_PKtS2_Pt | i4r_fold | i4r_fold.json seed=1 N=13 | MISMATCH n=33344 diff=6 max_abs=0.0625 max_ulp=2 first_diff=8426 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_i4r_foldEPKfiS2_PKDF16_PKtS2_Pt | i4r_fold | i4r_fold.json seed=2 N=13 | MISMATCH n=33344 diff=6 max_abs=0.0625 max_ulp=4 first_diff=3719 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_i4r_foldEPKfiS2_PKDF16_PKtS2_Pt | i4r_fold | i4r_fold.json seed=3 N=13 | MISMATCH n=33344 diff=5 max_abs=0.0625 max_ulp=2 first_diff=5807 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_expert_hEPKtS2_PKiPt | expert | expert_h.json seed=1 N=33 | MISMATCH n=21184 diff=1658 max_abs=0.125 max_ulp=1 first_diff=41 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_expert_hEPKtS2_PKiPt | expert | expert_h.json seed=2 N=33 | MISMATCH n=21184 diff=1613 max_abs=0.0625 max_ulp=1 first_diff=29 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_expert_hEPKtS2_PKiPt | expert | expert_h.json seed=3 N=33 | MISMATCH n=21184 diff=1742 max_abs=0.03125 max_ulp=1 first_diff=6 | bitwise | 2026-10-11 |
| _ZN7halogen11k_dv_quant8EPKtPKiS3_iPhi | dv | dv_quant8_edge.json seed=1 N=100 | MISMATCH n=272064 diff=48032 max_abs=129 max_ulp=129 first_diff=32 | bitwise | 2026-10-11 |
| _ZN7halogen11k_dv_quant8EPKtPKiS3_iPhi | dv | dv_quant8_edge.json seed=2 N=100 | MISMATCH n=272064 diff=43136 max_abs=129 max_ulp=129 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen11k_dv_quant8EPKtPKiS3_iPhi | dv | dv_quant8_edge.json seed=3 N=100 | MISMATCH n=272064 diff=43456 max_abs=129 max_ulp=129 first_diff=96 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_115k_expert_h_warpEPKtPKhPKiS6_S6_lllPt | expert | expert_h_warp_f16.json seed=1 N=4 | MISMATCH n=7744 diff=7680 max_abs=1.20446e-07 max_ulp=25913 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_115k_expert_h_warpEPKtPKhPKiS6_S6_lllPt | expert | expert_h_warp_f16.json seed=2 N=4 | MISMATCH n=7744 diff=7670 max_abs=3.45171e-08 max_ulp=25698 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_115k_expert_h_warpEPKtPKhPKiS6_S6_lllPt | expert | expert_h_warp_f16.json seed=3 N=4 | MISMATCH n=7744 diff=7678 max_abs=9.0804e-08 max_ulp=25995 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_127k_expert_h_grouped_warp_affILi4ELb1EEEvPKtPKhPKiS7_S7_NS0_6AffDevEPt | expert | expert_h_grouped_warp_aff_4t_f16.json seed=1 N=4 | MISMATCH n=7744 diff=2671 max_abs=7.2142e+08 max_ulp=39466 first_diff=1 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_127k_expert_h_grouped_warp_affILi4ELb1EEEvPKtPKhPKiS7_S7_NS0_6AffDevEPt | expert | expert_h_grouped_warp_aff_4t_f16.json seed=2 N=4 | MISMATCH n=7744 diff=6100 max_abs=1.59384e+09 max_ulp=39821 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_127k_expert_h_grouped_warp_affILi4ELb1EEEvPKtPKhPKiS7_S7_NS0_6AffDevEPt | expert | expert_h_grouped_warp_aff_4t_f16.json seed=3 N=4 | MISMATCH n=7744 diff=4325 max_abs=1.48688e+09 max_ulp=39914 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_115k_expert_out_q4EPKtPKhPKiPKflllPt | expert | expert_out_q4_f16.json seed=1 N=3 | MATCH n=7744 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_115k_expert_out_q4EPKtPKhPKiPKflllPt | expert | expert_out_q4_f16.json seed=2 N=3 | MATCH n=7744 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_115k_expert_out_q4EPKtPKhPKiPKflllPt | expert | expert_out_q4_f16.json seed=3 N=3 | MATCH n=7744 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_115k_expert_out_q4EPKtPKhPKiPKflllPt | expert | expert_out_q4_f16.json seed=1 N=3 | MISMATCH n=7744 diff=7680 max_abs=1.00045 max_ulp=31083 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_115k_expert_out_q4EPKtPKhPKiPKflllPt | expert | expert_out_q4_f16.json seed=2 N=3 | MISMATCH n=7744 diff=7680 max_abs=1.00023 max_ulp=30961 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_115k_expert_out_q4EPKtPKhPKiPKflllPt | expert | expert_out_q4_f16.json seed=3 N=3 | MISMATCH n=7744 diff=7680 max_abs=1.00044 max_ulp=31080 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_i4r_rowsILi640EEEvPKhliPKDF16_iPKiPf | i4r_rows | i4r_rows_640_f16_xd1.json seed=1 N=3 | MISMATCH n=448 diff=384 max_abs=22.29 max_ulp=2.18501e+09 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_i4r_rowsILi640EEEvPKhliPKDF16_iPKiPf | i4r_rows | i4r_rows_640_f16_xd1.json seed=2 N=3 | MISMATCH n=448 diff=384 max_abs=33.0111 max_ulp=2.1846e+09 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_i4r_rowsILi640EEEvPKhliPKDF16_iPKiPf | i4r_rows | i4r_rows_640_f16_xd1.json seed=3 N=3 | MISMATCH n=448 diff=384 max_abs=31.2158 max_ulp=2.18552e+09 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_113k_i4r_rows_flEPKhlPKDF16_PKiPfPKfS4_PKtS9_PtPj | i4r_rows | i4r_rows_fl_f16.json seed=1 N=3 | MATCH n=7744 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_113k_i4r_rows_flEPKhlPKDF16_PKiPfPKfS4_PKtS9_PtPj | i4r_rows | i4r_rows_fl_f16.json seed=2 N=3 | MATCH n=7744 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_113k_i4r_rows_flEPKhlPKDF16_PKiPfPKfS4_PKtS9_PtPj | i4r_rows | i4r_rows_fl_f16.json seed=3 N=3 | MISMATCH n=7744 diff=125 max_abs=17.5 max_ulp=32325 first_diff=2688 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_114k_i4r_rows_epiEPKhliPKfPKDF16_S6_PKiPf | i4r_rows | i4r_rows_epi_f16_edge.json seed=1 N=12 | MISMATCH n=1600 diff=528 max_abs=7.0625 max_ulp=763904 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_114k_i4r_rows_epiEPKhliPKfPKDF16_S6_PKiPf | i4r_rows | i4r_rows_epi_f16_edge.json seed=2 N=12 | MISMATCH n=1600 diff=240 max_abs=0.746094 max_ulp=16128 first_diff=384 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_114k_i4r_rows_epiEPKhliPKfPKDF16_S6_PKiPf | i4r_rows | i4r_rows_epi_f16_edge.json seed=3 N=12 | MISMATCH n=1600 diff=865 max_abs=7.76562 max_ulp=53888 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_114k_i4r_rows_rotEPKhliPKtPKDF16_PKiPf | i4r_rows | i4r_rows_rot_f16.json seed=1 N=12 | MISMATCH n=1600 diff=1536 max_abs=0.005867 max_ulp=22048 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_114k_i4r_rows_rotEPKhliPKtPKDF16_PKiPf | i4r_rows | i4r_rows_rot_f16.json seed=2 N=12 | MISMATCH n=1600 diff=1536 max_abs=0.00553894 max_ulp=66432 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_114k_i4r_rows_rotEPKhliPKtPKDF16_PKiPf | i4r_rows | i4r_rows_rot_f16.json seed=3 N=12 | MISMATCH n=1600 diff=1536 max_abs=0.00740051 max_ulp=262144 first_diff=0 | bitwise | 2026-10-11 |
