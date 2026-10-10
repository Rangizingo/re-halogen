# Kernel match table (U22, PLAN.md p7-12)

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_116k_hyper_seed_addEPKtS2_Pt | hyper | hyper_seed_add.json seed=1 N=3 | MISMATCH n=30720 diff=30691 max_abs=16.0312 max_ulp=33280 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_19k_hc_foldEPKtPtf | hc_fold | hc_fold.json seed=1 N=3 | MISMATCH n=7744 diff=7678 max_abs=0.125 max_ulp=3 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_113k_split_qgateEPKtPtS3_ | split | split_qgate.json seed=1 N=3 | MISMATCH n=36992 diff=18432 max_abs=64785 max_ulp=64785 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_block_poolEPKtPti | block | block_pool.json seed=1 N=3 | MISMATCH n=448 diff=384 max_abs=1.34375 max_ulp=52 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_113k_inject_gateEPKtPflf | inject | inject_gate.json seed=1 N=4096 | MISMATCH n=4160 diff=4096 max_abs=0.00997531 max_ulp=83468 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_111k_hc_di_epiEPKtPtPflf | hc_di | hc_di_epi.json seed=1 N=50 | MISMATCH n=16328 diff=200 max_abs=0.997409 max_ulp=5.58797e+06 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_hc_mixedEPKtS2_PtPfS3_ | hc_mixed | hc_mixed.json seed=1 N=3 | MISMATCH n=30912 diff=30713 max_abs=2.2519 max_ulp=3.35544e+06 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_hc_scatterEPKtS2_PKfPt | hc_scatter | hc_scatter.json seed=1 N=3 | MISMATCH n=30784 diff=30663 max_abs=11.625 max_ulp=33140 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_17k_hcup4ILi2ELi0EEEvPKhlllPKtS5_PtPf | hcup4 | hcup4_20.json seed=1 N=37 | MISMATCH n=10368 diff=2368 max_abs=0.157776 max_ulp=3.35502e+06 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_18k_hcup4wEPKtS2_S2_PtPfS3_i | hcup4w | hcup4w.json seed=1 N=40 | MISMATCH n=409792 diff=409536 max_abs=0.166728 max_ulp=3.35544e+06 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_111k_hc8_fusedILi1EEEvNS0_5LqGrpEPKtliPKhS4_PtPfPyy | hc8 | hc8_fused1.json seed=1 N=37 | MISMATCH n=5248 diff=1184 max_abs=0.150263 max_ulp=3.34534e+06 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_hc8_fused3ILi1EEEvPtPKtPKfS4_iNS0_5LqGrpES2_liPKhS4_S2_PfPyySB_y | hc8 | hc8_f3_1.json seed=1 N=37 | MISMATCH n=25857 diff=11424 max_abs=2.35352 max_ulp=2.0805e+09 first_diff=0 | bitwise | 2026-10-10 |
