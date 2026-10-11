# Kernel match table (U22, PLAN.md p7-12)

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_17k_hcup8ILi1EEEvPKhPKtS5_PtPf | hcup8 | hcup8_1.json seed=1 N=37 | MISMATCH n=5248 diff=1184 max_abs=0.150263 max_ulp=3.34534e+06 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_17k_hcup8ILi1EEEvPKhPKtS5_PtPf | hcup8 | hcup8_1_edge.json seed=1 N=37 | MATCH n=5248 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_17k_hcup6ILi1EEEvPKhPKtS5_PtPf | hcup6 | hcup6_1.json seed=1 N=37 | MISMATCH n=5248 diff=1184 max_abs=0.107422 max_ulp=3.35506e+06 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_17k_hcup6ILi1EEEvPKhPKtS5_PtPf | hcup6 | hcup6_1.json seed=1 N=37 | MISMATCH n=5248 diff=1181 max_abs=0.0976562 max_ulp=2.05474e+09 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_111k_hc6_fusedILi1EEEvNS0_5LqGrpEPKtliPKhS4_PtPfPyy | hc6 | hc6_lq_1.json seed=1 N=37 | MISMATCH n=5342 diff=9 max_abs=0.999654 max_ulp=8.38861e+06 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_111k_hc6_fusedILi1EEEvNS0_5LqGrpEPKtliPKhS4_PtPfPyy | hc6 | hc6_fused1.json seed=1 N=37 | MISMATCH n=5248 diff=1184 max_abs=0.107422 max_ulp=3.35506e+06 first_diff=0 | bitwise | 2026-10-11 |
