# Kernel match table (U22, PLAN.md p7-12)

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_110k_dnc_prepILb0ELb0ELb0EEEvPKtPKfPfS6_S6_iPtS6_S6_ | dnc_prep | dnc_prep_base.json seed=1 N=200 | MISMATCH n=4730880 diff=2457600 max_abs=3672 max_ulp=3.00798e+07 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_19k_dnc_ut4ILi4ELi1ELb0ELb0ELb0EEEvPKtPKfS5_S5_PfS6_i | dnc_ut4 | dnc_ut4_base.json seed=1 N=256 | MISMATCH n=3145728 diff=2311069 max_abs=1.90735e-05 max_ulp=3.12957e+06 first_diff=6145 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_19k_dnc_ut5ILb1ELb0ELi0ELb0ELb0ELb0EEEvPKtPKfS5_S5_PfS6_iPtS5_S3_S7_S6_ | dnc_ut5 | dnc_ut5_base.json seed=1 N=256 | MISMATCH n=4718592 diff=1037248 max_abs=7.62939e-06 max_ulp=592218 first_diff=6145 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_dnc_prepILb0ELb0ELb0EEEvPKtPKfPfS6_S6_iPtS6_S6_ | dnc_prep | dnc_prep_small.json seed=1 N=200 | MISMATCH n=4730880 diff=3686400 max_abs=524 max_ulp=4.29048e+06 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_19k_dnc_ut4ILi4ELi1ELb0ELb0ELb0EEEvPKtPKfS5_S5_PfS6_i | dnc_ut4 | dnc_ut4_edge.json seed=1 N=256 | MATCH n=3145728 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_19k_dnc_ut4ILi4ELi1ELb0ELb0ELb0EEEvPKtPKfS5_S5_PfS6_i | dnc_ut4 | dnc_ut4_edge.json seed=1 N=256 | MATCH n=3145728 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_19k_dnc_ut5ILb1ELb0ELi0ELb0ELb0ELb0EEEvPKtPKfS5_S5_PfS6_iPtS5_S3_S7_S6_ | dnc_ut5 | dnc_ut5_edge.json seed=1 N=256 | MATCH n=4718592 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_19k_dnc_ut5ILb1ELb0ELi0ELb0ELb0ELb0EEEvPKtPKfS5_S5_PfS6_iPtS5_S3_S7_S6_ | dnc_ut5 | dnc_ut5_edge.json seed=1 N=256 | MATCH n=4718592 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_19k_dnc_ut4ILi4ELi1ELb0ELb0ELb0EEEvPKtPKfS5_S5_PfS6_i | dnc_ut4 | dnc_ut4_edge.json seed=1 N=256 | MISMATCH n=3145728 diff=170873 max_abs=inf max_ulp=4.29497e+09 first_diff=10112 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_19k_dnc_ut4ILi4ELi1ELb0ELb0ELb0EEEvPKtPKfS5_S5_PfS6_i | dnc_ut4 | dnc_ut4_edge.json seed=1 N=256 | MISMATCH n=3145728 diff=149172 max_abs=inf max_ulp=4.29497e+09 first_diff=16384 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_19k_dnc_ut5ILb1ELb0ELi0ELb0ELb0ELb0EEEvPKtPKfS5_S5_PfS6_iPtS5_S3_S7_S6_ | dnc_ut5 | dnc_ut5_edge.json seed=1 N=256 | MISMATCH n=4718592 diff=308066 max_abs=inf max_ulp=4.29497e+09 first_diff=13184 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_19k_dnc_ut5ILb1ELb0ELi0ELb0ELb0ELb0EEEvPKtPKfS5_S5_PfS6_iPtS5_S3_S7_S6_ | dnc_ut5 | dnc_ut5_edge.json seed=1 N=256 | MISMATCH n=4718592 diff=117267 max_abs=inf max_ulp=4.29497e+09 first_diff=16384 | bitwise | 2026-10-11 |
