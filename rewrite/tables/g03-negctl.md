# Kernel match table (U22, PLAN.md p7-12)

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_16k_lq6kILi1ELi16ELi1ELb0EEEvPKhPKtPvll | lq6k | lq6k_1_16_1_0.json seed=1 N=4096 | MISMATCH n=8320 diff=3415 max_abs=33743 max_ulp=33743 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_16k_lq6wILi1ELi16ELi1EEEvPKhPKtPtll | lq6w | lq6w_1_16_1.json seed=1 N=4096 | MISMATCH n=4160 diff=1369 max_abs=33377 max_ulp=33377 first_diff=9 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_lq6w_grpILi1ELi32ELi2EEEvNS0_5LqGrpEPKtl | lq6w | lq6w_grp_1_2_small.json seed=1 N=4096 | MISMATCH n=17512 diff=2158 max_abs=254 max_ulp=254 first_diff=16 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_19k_deq_q6kEPKhPtjjl | deq_q6k | deq_q6k.json seed=1 N=4096 | MISMATCH n=3145760 diff=3043670 max_abs=51958 max_ulp=51958 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_111k_deq_q6g64EPKhPtll | deq_q6g64 | deq_q6g64.json seed=1 N=4096 | MISMATCH n=3145760 diff=1854292 max_abs=50460 max_ulp=50460 first_diff=0 | bitwise | 2026-10-11 |
