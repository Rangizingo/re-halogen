# Kernel match table (U22, PLAN.md p7-12)

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_18k_ht_linILi4ELi1ELb0ELb0ELb0ELb0EEEvPKjPKtiiiiPKDF16_S7_PfPjPvS5_ | ht_lin | ht_lin_b4_s1_a0.json seed=1 N=128 | MISMATCH n=128 diff=5 max_abs=0.75 max_ulp=6 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_18k_ht_linILi4ELi1ELb0ELb0ELb0ELb0EEEvPKjPKtiiiiPKDF16_S7_PfPjPvS5_ | ht_lin | ht_lin_b4_s1_a0_small.json seed=1 N=128 | MISMATCH n=128 diff=128 max_abs=0.00235748 max_ulp=29970 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_18k_ht_linILi4ELi1ELb0ELb0ELb0ELb0EEEvPKjPKtiiiiPKDF16_S7_PfPjPvS5_ | ht_lin | ht_lin_b4_s1_a0.json seed=1 N=128 | MATCH n=128 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_ht_lin_grpILi4ELi2ELb0ELb0ELb0ELb0EEEvNS0_5HtGrpEPKtiiPfPj | ht_lin | ht_grp_b4_s2_multi.json seed=1 N=512 | MISMATCH n=9728 diff=255 max_abs=257 max_ulp=34276 first_diff=128 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_ht_lin_grpILi4ELi1ELb0ELb0ELb0ELb0EEEvNS0_5HtGrpEPKtiiPfPj | ht_lin | ht_grp_b4_s1.json seed=1 N=128 | MATCH n=128 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_ht_lin_grpILi4ELi8ELb0ELb0ELb0ELb0EEEvNS0_5HtGrpEPKtiiPfPj | ht_lin | ht_grp_b4_s8_multi.json seed=1 N=1024 | MATCH n=75264 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_ht_lin_grpILi4ELi8ELb0ELb0ELb0ELb0EEEvNS0_5HtGrpEPKtiiPfPj | ht_lin | ht_grp_b4_s8_multi.json seed=2 N=1024 | MATCH n=75264 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_ht_lin_grpILi4ELi8ELb0ELb0ELb0ELb0EEEvNS0_5HtGrpEPKtiiPfPj | ht_lin | ht_grp_b4_s8_multi.json seed=3 N=1024 | MATCH n=75264 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_ht_lin_grpILi4ELi1ELb0ELb0ELb0ELb0EEEvNS0_5HtGrpEPKtiiPfPj | ht_lin | ht_grp_b4_s1.json seed=1 N=128 | MATCH n=128 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_ht_lin_grpILi4ELi1ELb0ELb0ELb0ELb0EEEvNS0_5HtGrpEPKtiiPfPj | ht_lin | ht_grp_b4_s1.json seed=1 N=128 | MISMATCH n=128 diff=128 max_abs=63.8125 max_ulp=33736 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_18k_ht_linILi4ELi2ELb0ELb0ELb0ELb0EEEvPKjPKtiiiiPKDF16_S7_PfPjPvS5_ | ht_lin | ht_lin_b4_s2_a0_split.json seed=1 N=128 | MISMATCH n=1024 diff=518 max_abs=0.015625 max_ulp=814848 first_diff=30 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_ht_lin_grpILi4ELi1ELb1ELb0ELb0ELb1EEEvNS0_5HtGrpEPKtiiPfPj | ht_lin | ht_grp_b4_s1_x.json seed=1 N=128 | MISMATCH n=656 diff=1 max_abs=0.16536 max_ulp=5.54857e+06 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_ht_lin_grpILi4ELi1ELb1ELb0ELb0ELb1EEEvNS0_5HtGrpEPKtiiPfPj | ht_lin | ht_grp_b4_s1_x.json seed=1 N=128 | MISMATCH n=656 diff=512 max_abs=37.1421 max_ulp=2.19601e+09 first_diff=0 | bitwise | 2026-10-11 |
