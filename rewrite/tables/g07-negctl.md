# Kernel match table (U22, PLAN.md p7-12)

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_16k_ropeEPtiiifiPKi | rope | rope_k_rope_edge.json seed=1 N=7 | MISMATCH n=2016 diff=1470 max_abs=inf max_ulp=65536 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_16k_ropeEPtiiifiPKi | rope | rope_k_rope_edge.json seed=2 N=7 | MISMATCH n=2016 diff=1500 max_abs=inf max_ulp=65536 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_16k_ropeEPtiiifiPKi | rope | rope_k_rope_edge.json seed=3 N=7 | MISMATCH n=2016 diff=1476 max_abs=inf max_ulp=65536 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_117k_kv_scatter_rowsEPKtS2_NS0_6FdRowsE | kv | kv_k_kv_scatter_rows.json seed=1 N=5 | MISMATCH n=16640 diff=2560 max_abs=0.03125 max_ulp=1 first_diff=512 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_117k_kv_scatter_rowsEPKtS2_NS0_6FdRowsE | kv | kv_k_kv_scatter_rows.json seed=2 N=5 | MISMATCH n=16640 diff=2560 max_abs=0.03125 max_ulp=1 first_diff=512 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_117k_kv_scatter_rowsEPKtS2_NS0_6FdRowsE | kv | kv_k_kv_scatter_rows.json seed=3 N=5 | MISMATCH n=16640 diff=2560 max_abs=0.03125 max_ulp=1 first_diff=512 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_19k_qk_prepEPtS1_PKtS3_S3_iiifiPKiNS0_6FdRowsE | qk | qk_k_qk_prep_small.json seed=1 N=5 | MISMATCH n=49920 diff=35840 max_abs=0.132568 max_ulp=28966 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_19k_qk_prepEPtS1_PKtS3_S3_iiifiPKiNS0_6FdRowsE | qk | qk_k_qk_prep_small.json seed=2 N=5 | MISMATCH n=49920 diff=35840 max_abs=0.133545 max_ulp=28826 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_19k_qk_prepEPtS1_PKtS3_S3_iiifiPKiNS0_6FdRowsE | qk | qk_k_qk_prep_small.json seed=3 N=5 | MISMATCH n=49920 diff=35840 max_abs=0.130859 max_ulp=28526 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_19k_attn_fdILb0EEEvPKtNS0_6FdRowsEPKiS6_iPfS7_i | attn_fd | attn_fd_false_ns3.json seed=1 N=100 | MISMATCH n=37184 diff=37152 max_abs=0.0176668 max_ulp=1.96475e+09 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_19k_attn_fdILb0EEEvPKtNS0_6FdRowsEPKiS6_iPfS7_i | attn_fd | attn_fd_false_ns3.json seed=2 N=100 | MISMATCH n=37184 diff=37152 max_abs=0.0175085 max_ulp=1.96587e+09 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_19k_attn_fdILb0EEEvPKtNS0_6FdRowsEPKiS6_iPfS7_i | attn_fd | attn_fd_false_ns3.json seed=3 N=100 | MISMATCH n=37184 diff=37151 max_abs=0.0162191 max_ulp=1.96496e+09 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_attn_fd_mergeEPKfS2_iPtPKt | attn_fd | attn_fd_merge_edge.json seed=1 N=3 | MATCH n=18464 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_attn_fd_mergeEPKfS2_iPtPKt | attn_fd | attn_fd_merge_edge.json seed=2 N=3 | MATCH n=18464 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_attn_fd_mergeEPKfS2_iPtPKt | attn_fd | attn_fd_merge_edge.json seed=3 N=3 | MATCH n=18464 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_16k_attnEPKtS2_S2_S2_PtiiS2_ | attn | attn_k_attn_edge.json seed=1 N=100 | MISMATCH n=24608 diff=145 max_abs=0.00390625 max_ulp=57 first_diff=12321 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_16k_attnEPKtS2_S2_S2_PtiiS2_ | attn | attn_k_attn_edge.json seed=2 N=100 | MISMATCH n=24608 diff=2 max_abs=0.000488281 max_ulp=1 first_diff=14669 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_16k_attnEPKtS2_S2_S2_PtiiS2_ | attn | attn_k_attn_edge.json seed=3 N=100 | MISMATCH n=24608 diff=126 max_abs=0.00390625 max_ulp=121 first_diff=5121 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_attn_selEPKtS2_S2_PKiS4_PtiiS2_ | attn_sel | attn_sel_edge.json seed=1 N=100 | MISMATCH n=18464 diff=10 max_abs=0.00195312 max_ulp=3 first_diff=11786 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_attn_selEPKtS2_S2_PKiS4_PtiiS2_ | attn_sel | attn_sel_edge.json seed=2 N=100 | MISMATCH n=18464 diff=1 max_abs=7.62939e-06 max_ulp=2 first_diff=2664 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_attn_selEPKtS2_S2_PKiS4_PtiiS2_ | attn_sel | attn_sel_edge.json seed=3 N=100 | MATCH n=18464 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_attn_fa2ILi2ELi1ELb1ELi0ELi0ELb0EEEvPKtS3_S3_lS3_PtPKiiiNS_5FaSelE | attn_fa2 | attn_fa2_2_1_1_0_0_0_edge.json seed=1 N=48 | MATCH n=393344 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_attn_fa2ILi2ELi1ELb1ELi0ELi0ELb0EEEvPKtS3_S3_lS3_PtPKiiiNS_5FaSelE | attn_fa2 | attn_fa2_2_1_1_0_0_0_edge.json seed=2 N=48 | MATCH n=393344 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_attn_fa2ILi2ELi1ELb1ELi0ELi0ELb0EEEvPKtS3_S3_lS3_PtPKiiiNS_5FaSelE | attn_fa2 | attn_fa2_2_1_1_0_0_0_edge.json seed=3 N=48 | MATCH n=393344 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_attn_fa2ILi2ELi1ELb1ELi5ELi1ELb1EEEvPKtS3_S3_lS3_PtPKiiiNS_5FaSelE | attn_fa2 | attn_fa2_2_1_1_5_1_1_edge.json seed=1 N=48 | MATCH n=393344 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_attn_fa2ILi2ELi1ELb1ELi5ELi1ELb1EEEvPKtS3_S3_lS3_PtPKiiiNS_5FaSelE | attn_fa2 | attn_fa2_2_1_1_5_1_1_edge.json seed=2 N=48 | MATCH n=393344 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_attn_fa2ILi2ELi1ELb1ELi5ELi1ELb1EEEvPKtS3_S3_lS3_PtPKiiiNS_5FaSelE | attn_fa2 | attn_fa2_2_1_1_5_1_1_edge.json seed=3 N=48 | MATCH n=393344 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_19k_attn_faEPKtS2_S2_lS2_PtPKiii | attn_fa | attn_fa_k_attn_fa.json seed=1 N=48 | MISMATCH n=295040 diff=3 max_abs=0.00012207 max_ulp=1 first_diff=180687 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_19k_attn_faEPKtS2_S2_lS2_PtPKiii | attn_fa | attn_fa_k_attn_fa.json seed=2 N=48 | MISMATCH n=295040 diff=5 max_abs=3.05176e-05 max_ulp=1 first_diff=264677 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_19k_attn_faEPKtS2_S2_lS2_PtPKiii | attn_fa | attn_fa_k_attn_fa.json seed=3 N=48 | MISMATCH n=295040 diff=4 max_abs=3.05176e-05 max_ulp=1 first_diff=71675 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_attn_fd_mergeEPKfS2_iPtPKt | attn_fd | attn_fd_merge_edge.json seed=1 N=3 | MISMATCH n=18464 diff=286 max_abs=0.000183105 max_ulp=35 first_diff=2307 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_attn_fd_mergeEPKfS2_iPtPKt | attn_fd | attn_fd_merge_edge.json seed=2 N=3 | MISMATCH n=18464 diff=322 max_abs=0.00012207 max_ulp=67 first_diff=524 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_attn_fd_mergeEPKfS2_iPtPKt | attn_fd | attn_fd_merge_edge.json seed=3 N=3 | MISMATCH n=18464 diff=335 max_abs=0.00390625 max_ulp=87 first_diff=775 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_attn_fa2ILi2ELi1ELb1ELi5ELi1ELb1EEEvPKtS3_S3_lS3_PtPKiiiNS_5FaSelE | attn_fa2 | attn_fa2_2_1_1_5_1_1_edge.json seed=1 N=48 | MISMATCH n=393344 diff=1173 max_abs=0.00390625 max_ulp=232 first_diff=10547 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_attn_fa2ILi2ELi1ELb1ELi5ELi1ELb1EEEvPKtS3_S3_lS3_PtPKiiiNS_5FaSelE | attn_fa2 | attn_fa2_2_1_1_5_1_1_edge.json seed=2 N=48 | MISMATCH n=393344 diff=1448 max_abs=0.00390625 max_ulp=28848 first_diff=5460 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_attn_fa2ILi2ELi1ELb1ELi5ELi1ELb1EEEvPKtS3_S3_lS3_PtPKiiiNS_5FaSelE | attn_fa2 | attn_fa2_2_1_1_5_1_1_edge.json seed=3 N=48 | MISMATCH n=393344 diff=1407 max_abs=0.00390625 max_ulp=29726 first_diff=13415 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_attn_fa2ILi2ELi1ELb1ELi0ELi0ELb0EEEvPKtS3_S3_lS3_PtPKiiiNS_5FaSelE | attn_fa2 | attn_fa2_2_1_1_0_0_0_edge.json seed=1 N=48 | MISMATCH n=393344 diff=1578 max_abs=0.00390625 max_ulp=333 first_diff=10499 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_attn_fa2ILi2ELi1ELb1ELi0ELi0ELb0EEEvPKtS3_S3_lS3_PtPKiiiNS_5FaSelE | attn_fa2 | attn_fa2_2_1_1_0_0_0_edge.json seed=2 N=48 | MISMATCH n=393344 diff=1937 max_abs=0.00390625 max_ulp=29330 first_diff=5460 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_attn_fa2ILi2ELi1ELb1ELi0ELi0ELb0EEEvPKtS3_S3_lS3_PtPKiiiNS_5FaSelE | attn_fa2 | attn_fa2_2_1_1_0_0_0_edge.json seed=3 N=48 | MISMATCH n=393344 diff=1688 max_abs=0.00390625 max_ulp=29726 first_diff=13415 | bitwise | 2026-10-10 |
