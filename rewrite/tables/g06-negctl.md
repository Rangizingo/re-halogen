# Kernel match table (U22, PLAN.md p7-12)

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_118k_hist_from_cstateEPKtPt | hist | hist_from_cstate.json seed=1 N=256 | MISMATCH n=30784 diff=10240 max_abs=64869 max_ulp=64869 first_diff=20480 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_118k_hist_from_cstateEPKtPt | hist | hist_from_cstate.json seed=2 N=256 | MISMATCH n=30784 diff=10240 max_abs=64619 max_ulp=64619 first_diff=20480 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_116k_count_diff_u32EPKjS2_lPj | count | count_diff_half.json seed=1 N=5000 | MISMATCH n=4 diff=1 max_abs=1 max_ulp=1 first_diff=1 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_116k_count_diff_u32EPKjS2_lPj | count | count_diff_half.json seed=2 N=5000 | MISMATCH n=4 diff=1 max_abs=1 max_ulp=1 first_diff=1 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_route_histEPKiiPiS3_ | route_hist | route_hist_wide.json seed=1 N=3000 | MISMATCH n=4792 diff=1225 max_abs=7 max_ulp=7 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_route_histEPKiiPiS3_ | route_hist | route_hist_wide.json seed=2 N=3000 | MISMATCH n=4792 diff=1222 max_abs=8 max_ulp=8 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_route_scanEPiiS1_S1_S1_ | route_scan | route_scan_sparse.json seed=1 N=3 | MISMATCH n=3272 diff=1007 max_abs=796 max_ulp=796 first_diff=9 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_route_scanEPiiS1_S1_S1_ | route_scan | route_scan_sparse.json seed=2 N=3 | MISMATCH n=3272 diff=1023 max_abs=766 max_ulp=766 first_diff=1 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_route_scatterEPKiiS2_S2_PiS3_ | route_scatter | route_scatter_iota.json seed=1 N=3000 | MISMATCH n=6294712 diff=2517 max_abs=1 max_ulp=1 first_diff=41 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_route_scatterEPKiiS2_S2_PiS3_ | route_scatter | route_scatter_iota.json seed=2 N=3000 | MISMATCH n=6294712 diff=2511 max_abs=1 max_ulp=1 first_diff=518 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_route_topkEPKfPiPf | route_topk | route_topk_ties.json seed=1 N=3 | MISMATCH n=92 diff=30 max_abs=386 max_ulp=386 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_route_topkEPKfPiPf | route_topk | route_topk_ties.json seed=2 N=3 | MISMATCH n=92 diff=30 max_abs=510 max_ulp=510 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_114k_route_topk_wEPKfPiPfi | route_topk | route_topk_w_ties.json seed=1 N=13 | MISMATCH n=292 diff=130 max_abs=511 max_ulp=511 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_114k_route_topk_wEPKfPiPfi | route_topk | route_topk_w_ties.json seed=2 N=13 | MISMATCH n=292 diff=130 max_abs=511 max_ulp=511 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_118k_route_topk_w_grpILi2EEEvPKfPiPfiPNS0_6HtItemES7_S4_S4_S4_S4_ | route_topk | route_topk_w_grp2_same.json seed=1 N=3 | MISMATCH n=3374 diff=1676 max_abs=455 max_ulp=455 first_diff=20 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_118k_route_topk_w_grpILi2EEEvPKfPiPfiPNS0_6HtItemES7_S4_S4_S4_S4_ | route_topk | route_topk_w_grp2_same.json seed=2 N=3 | MISMATCH n=3374 diff=1676 max_abs=455 max_ulp=455 first_diff=20 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_120k_route_logits_x32_mILi8EEEvPKtPKfPfll | route_logits | route_logits_x32_m8.json seed=1 N=2560 | MISMATCH n=64 diff=48 max_abs=0.256003 max_ulp=1.00458e+07 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_120k_route_logits_x32_mILi8EEEvPKtPKfPfll | route_logits | route_logits_x32_m8.json seed=2 N=2560 | MISMATCH n=64 diff=48 max_abs=0.256004 max_ulp=4.29498e+06 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_119k_route_logits_rowsILi8ELi2EEEvPKtPKfPfll | route_logits | route_logits_rows_8_2.json seed=1 N=2560 | MISMATCH n=48 diff=32 max_abs=0.255003 max_ulp=9.97867e+06 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_119k_route_logits_rowsILi8ELi2EEEvPKtPKfPfll | route_logits | route_logits_rows_8_2.json seed=2 N=2560 | MISMATCH n=48 diff=32 max_abs=0.255002 max_ulp=4.27815e+06 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_122k_route_logits_rows_sgILi8ELi2ELi0EEEvPKtPKfPfllNS0_5SgArgE | route_logits | route_logits_rows_sg_8_2_0.json seed=1 N=2560 | MISMATCH n=64 diff=2 max_abs=8.76188e-06 max_ulp=147 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_122k_route_logits_rows_sgILi8ELi2ELi0EEEvPKtPKfPfllNS0_5SgArgE | route_logits | route_logits_rows_sg_8_2_0.json seed=2 N=2560 | MISMATCH n=64 diff=2 max_abs=1.3411e-05 max_ulp=450 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_118k_route_logits_sg1ILi8ELi0EEEvPKtPKfPfllNS0_5SgArgE | route_logits | route_logits_sg1_8_0.json seed=1 N=2560 | MATCH n=48 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_118k_route_logits_sg1ILi8ELi0EEEvPKtPKfPfllNS0_5SgArgE | route_logits | route_logits_sg1_8_0.json seed=2 N=2560 | MATCH n=48 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_route_histEPKiiPiS3_ | route_hist | route_hist_edge.json seed=1 N=3000 | MISMATCH n=4792 diff=1590 max_abs=516 max_ulp=516 first_diff=511 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_route_histEPKiiPiS3_ | route_hist | route_hist_edge.json seed=2 N=3000 | MISMATCH n=4792 diff=1593 max_abs=517 max_ulp=517 first_diff=511 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_route_scatterEPKiiS2_S2_PiS3_ | route_scatter | route_scatter_edge.json seed=1 N=3000 | MISMATCH n=6294712 diff=270 max_abs=6.29035e+06 max_ulp=6.29035e+06 first_diff=2093059 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_route_scatterEPKiiS2_S2_PiS3_ | route_scatter | route_scatter_edge.json seed=2 N=3000 | MISMATCH n=6294712 diff=274 max_abs=6.29029e+06 max_ulp=6.29029e+06 first_diff=2093063 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_route_scanEPiiS1_S1_S1_ | route_scan | route_scan_edge.json seed=1 N=3 | MISMATCH n=3272 diff=770 max_abs=518 max_ulp=518 first_diff=1 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_route_scanEPiiS1_S1_S1_ | route_scan | route_scan_edge.json seed=2 N=3 | MISMATCH n=3272 diff=792 max_abs=518 max_ulp=518 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_route_topkEPKfPiPf | route_topk | route_topk_edge.json seed=1 N=3 | MATCH n=92 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_route_topkEPKfPiPf | route_topk | route_topk_edge.json seed=2 N=3 | MATCH n=92 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_114k_route_topk_wEPKfPiPfi | route_topk | route_topk_w_edge.json seed=1 N=13 | MATCH n=292 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_114k_route_topk_wEPKfPiPfi | route_topk | route_topk_w_edge.json seed=2 N=13 | MATCH n=292 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_118k_route_logits_sg1ILi8ELi0EEEvPKtPKfPfllNS0_5SgArgE | route_logits | route_logits_sg1_8_0_tie.json seed=1 N=2560 | MISMATCH n=48 diff=1 max_abs=9.02414e-05 max_ulp=1514 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_118k_route_logits_sg1ILi8ELi0EEEvPKtPKfPfllNS0_5SgArgE | route_logits | route_logits_sg1_8_0_tie.json seed=2 N=2560 | MISMATCH n=48 diff=1 max_abs=0.000251085 max_ulp=8425 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_118k_route_logits_sg1ILi8ELi0EEEvPKtPKfPfllNS0_5SgArgE | route_logits | route_logits_sg1_8_0_small.json seed=1 N=2560 | MATCH n=48 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_118k_route_logits_sg1ILi8ELi0EEEvPKtPKfPfllNS0_5SgArgE | route_logits | route_logits_sg1_8_0_small.json seed=2 N=2560 | MATCH n=48 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_route_topkEPKfPiPf | route_topk | route_topk_rand.json seed=1 N=3 | MISMATCH n=92 diff=30 max_abs=6.10948e-07 max_ulp=82 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_route_topkEPKfPiPf | route_topk | route_topk_rand.json seed=2 N=3 | MISMATCH n=92 diff=30 max_abs=6.55651e-07 max_ulp=88 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_114k_route_topk_wEPKfPiPfi | route_topk | route_topk_w_rand.json seed=1 N=13 | MISMATCH n=292 diff=130 max_abs=7.97212e-07 max_ulp=107 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_114k_route_topk_wEPKfPiPfi | route_topk | route_topk_w_rand.json seed=2 N=13 | MISMATCH n=292 diff=130 max_abs=8.41916e-07 max_ulp=113 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_route_topkEPKfPiPf | route_topk | route_topk_edge.json seed=1 N=3 | MISMATCH n=92 diff=30 max_abs=2.48551e-05 max_ulp=8348 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_route_topkEPKfPiPf | route_topk | route_topk_edge.json seed=2 N=3 | MISMATCH n=92 diff=30 max_abs=1.9908e-05 max_ulp=12934 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_118k_route_logits_sg1ILi8ELi0EEEvPKtPKfPfllNS0_5SgArgE | route_logits | route_logits_sg1_8_0_edge.json seed=1 N=2560 | MISMATCH n=48 diff=1 max_abs=1.29314e-06 max_ulp=5554 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_118k_route_logits_sg1ILi8ELi0EEEvPKtPKfPfllNS0_5SgArgE | route_logits | route_logits_sg1_8_0_edge.json seed=2 N=2560 | MISMATCH n=48 diff=1 max_abs=5.96671e-18 max_ulp=28177 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_118k_route_logits_sg1ILi8ELi0EEEvPKtPKfPfllNS0_5SgArgE | route_logits | route_logits_sg1_8_0_small.json seed=1 N=2560 | MATCH n=48 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_118k_route_logits_sg1ILi8ELi0EEEvPKtPKfPfllNS0_5SgArgE | route_logits | route_logits_sg1_8_0_small.json seed=2 N=2560 | MATCH n=48 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
