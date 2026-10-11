# Kernel match table (U22, PLAN.md p7-12)

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_112k_dnc_scan3wILi2ELi4ELb1ELb0ELi5EEEvPKtPKfS5_S5_S5_S5_S5_PfPKilPti | dnc_scan3w | dnc_scan3w_probe_d_p5.json seed=1 N=128 | MISMATCH n=1966080 diff=1178837 max_abs=306.906 max_ulp=2.16406e+09 first_diff=393216 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_dnc_scan3wILi2ELi4ELb1ELb0ELi8EEEvPKtPKfS5_S5_S5_S5_S5_PfPKilPti | dnc_scan3w | dnc_scan3w_probe_d_p8.json seed=1 N=128 | MISMATCH n=1966080 diff=1570156 max_abs=105.659 max_ulp=2.19128e+09 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_dnc_scan3wILi2ELi4ELb1ELb0ELi8EEEvPKtPKfS5_S5_S5_S5_S5_PfPKilPti | dnc_scan3w | dnc_scan3w_probe_d_p8.json seed=1 N=128 | MISMATCH n=1966080 diff=1568428 max_abs=44.8702 max_ulp=2.20456e+09 first_diff=0 | bitwise | 2026-10-11 |
