# Kernel match table (U22, PLAN.md p7-12)

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_115k_dnc_cmp_wordsEPKjS2_lPy | dnc_cmp | dnc_cmp_words.json seed=1 N=4096 | MISMATCH n=1 diff=1 max_abs=2438 max_ulp=2438 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_19k_dnc_attEPKfS2_S2_Pfi | dnc_att | dnc_att_edge.json seed=1 N=2 | MISMATCH n=393216 diff=57443 max_abs=inf max_ulp=4.38555e+08 first_diff=128 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_dnc_att2EPKfS2_S2_Pfi | dnc_att2 | dnc_att2_edge.json seed=1 N=2 | MISMATCH n=393216 diff=57443 max_abs=inf max_ulp=4.38555e+08 first_diff=128 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_dnc_att7ILi32ELi4EEEvPKfS3_S3_Pfi | dnc_att7 | dnc_att7_r4_edge.json seed=1 N=2 | MISMATCH n=393216 diff=57443 max_abs=inf max_ulp=4.38555e+08 first_diff=128 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_dnc_att6ILi32ELb0ELb0ELb0EEEvPKfS3_S3_PfiPtPKtS3_S3_ | dnc_att6 | dnc_att6_32_0_0_0_small.json seed=1 N=2 | MISMATCH n=786432 diff=108267 max_abs=915 max_ulp=915 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_dnc_att6ILi32ELb0ELb0ELb0EEEvPKfS3_S3_PfiPtPKtS3_S3_ | dnc_att6 | dnc_att6_32_0_0_0_edge.json seed=1 N=2 | MISMATCH n=786432 diff=57443 max_abs=inf max_ulp=4.38555e+08 first_diff=128 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_dnc_att8EPKfS2_S2_Pfi | dnc_att8 | dnc_att8_small.json seed=1 N=2 | MISMATCH n=393216 diff=153984 max_abs=3.35967e-08 max_ulp=1.4434e+09 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_dnc_att8EPKfS2_S2_Pfi | dnc_att8 | dnc_att8_edge.json seed=1 N=2 | MISMATCH n=393216 diff=55299 max_abs=inf max_ulp=4.29497e+09 first_diff=128 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_dnc_att9ILi4EEEvPKfS3_S3_Pfi | dnc_att9 | dnc_att9_4_edge.json seed=1 N=2 | MISMATCH n=393216 diff=55299 max_abs=inf max_ulp=4.29497e+09 first_diff=128 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_dnc_att9ILi4EEEvPKfS3_S3_Pfi | dnc_att9 | dnc_att9_4_small.json seed=1 N=2 | MISMATCH n=393216 diff=153984 max_abs=3.36107e-08 max_ulp=8.38861e+06 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_18k_dnc_utEPKtPKfS4_S4_PfS5_i | dnc_ut | dnc_ut_edge.json seed=1 N=2 | MISMATCH n=1572864 diff=65215 max_abs=inf max_ulp=4.29497e+09 first_diff=6272 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_19k_dnc_ut2EPKtPKfS4_S4_PfS5_i | dnc_ut2 | dnc_ut2_edge.json seed=1 N=2 | MISMATCH n=1572864 diff=65215 max_abs=inf max_ulp=4.29497e+09 first_diff=6272 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_19k_dnc_ut3EPKtPKfS4_S4_PfS5_i | dnc_ut3 | dnc_ut3_edge.json seed=1 N=2 | MISMATCH n=1572864 diff=65215 max_abs=inf max_ulp=4.29497e+09 first_diff=6272 | bitwise | 2026-10-11 |
