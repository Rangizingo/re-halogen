# Kernel match table (U22, PLAN.md p7-12)

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_18k_lq8s32ILi1ELi16ELi1ELb0EEEvPKhPKtPvll | lq8s32 | lq8s32_m1_r16_w1_f0.json seed=1 N=77 | MISMATCH n=141 diff=10 max_abs=16 max_ulp=9 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_18k_lq8s32ILi1ELi16ELi1ELb0EEEvPKhPKtPvll | lq8s32 | lq8s32_m1_r16_w1_f0.json seed=2 N=77 | MISMATCH n=141 diff=7 max_abs=4 max_ulp=2 first_diff=12 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_18k_lq8s32ILi1ELi16ELi1ELb0EEEvPKhPKtPvll | lq8s32 | lq8s32_m1_r16_w1_f0.json seed=3 N=77 | MISMATCH n=141 diff=6 max_abs=4 max_ulp=1 first_diff=2 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_18k_lq8s32ILi1ELi16ELi1ELb0EEEvPKhPKtPvll | lq8s32 | lq8s32_m1_r16_w1_f0.json seed=1 N=1003 | MISMATCH n=1067 diff=114 max_abs=16 max_ulp=27 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_18k_lq8s32ILi1ELi16ELi1ELb0EEEvPKhPKtPvll | lq8s32 | lq8s32_m1_r16_w1_f0.json seed=2 N=1003 | MISMATCH n=1067 diff=122 max_abs=32 max_ulp=13 first_diff=12 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_18k_lq8s32ILi1ELi16ELi1ELb0EEEvPKhPKtPvll | lq8s32 | lq8s32_m1_r16_w1_f0.json seed=3 N=1003 | MISMATCH n=1067 diff=115 max_abs=32 max_ulp=17 first_diff=2 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_16k_lq8wILi1ELi16ELi1EEEvPKhPKtPtll | lq8w | lq8w_m1_r16_w1.json seed=1 N=77 | MISMATCH n=141 diff=3 max_abs=8 max_ulp=2 first_diff=30 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_16k_lq8wILi1ELi16ELi1EEEvPKhPKtPtll | lq8w | lq8w_m1_r16_w1.json seed=2 N=77 | MATCH n=141 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_16k_lq8wILi1ELi16ELi1EEEvPKhPKtPtll | lq8w | lq8w_m1_r16_w1.json seed=3 N=77 | MISMATCH n=141 diff=3 max_abs=16 max_ulp=12 first_diff=22 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_16k_lq8wILi1ELi16ELi1EEEvPKhPKtPtll | lq8w | lq8w_m1_r16_w1.json seed=1 N=1003 | MISMATCH n=1067 diff=30 max_abs=64 max_ulp=13 first_diff=30 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_16k_lq8wILi1ELi16ELi1EEEvPKhPKtPtll | lq8w | lq8w_m1_r16_w1.json seed=2 N=1003 | MISMATCH n=1067 diff=29 max_abs=16 max_ulp=7 first_diff=85 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_16k_lq8wILi1ELi16ELi1EEEvPKhPKtPtll | lq8w | lq8w_m1_r16_w1.json seed=3 N=1003 | MISMATCH n=1067 diff=39 max_abs=32 max_ulp=50 first_diff=22 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_lq8w_grpILi1ELi32ELi2EEEvNS0_5LqGrpEPKtl | lq8w | lq8w_grp_m1_r32_w2_a.json seed=1 N=77 | MISMATCH n=426 diff=2 max_abs=1.17431e-10 max_ulp=16527 first_diff=63 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_lq8w_grpILi1ELi32ELi2EEEvNS0_5LqGrpEPKtl | lq8w | lq8w_grp_m1_r32_w2_a.json seed=2 N=77 | MISMATCH n=426 diff=1 max_abs=3.66374e-15 max_ulp=16896 first_diff=39 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_lq8w_grpILi1ELi32ELi2EEEvNS0_5LqGrpEPKtl | lq8w | lq8w_grp_m1_r32_w2_a.json seed=3 N=77 | MISMATCH n=426 diff=4 max_abs=1.81198e-05 max_ulp=33297 first_diff=44 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_lq8w_grpILi1ELi32ELi2EEEvNS0_5LqGrpEPKtl | lq8w | lq8w_grp_m1_r32_w2_a.json seed=1 N=1003 | MISMATCH n=2278 diff=22 max_abs=2.90871e-05 max_ulp=53677 first_diff=63 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_lq8w_grpILi1ELi32ELi2EEEvNS0_5LqGrpEPKtl | lq8w | lq8w_grp_m1_r32_w2_a.json seed=2 N=1003 | MISMATCH n=2278 diff=23 max_abs=1.16825e-05 max_ulp=82360 first_diff=455 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_lq8w_grpILi1ELi32ELi2EEEvNS0_5LqGrpEPKtl | lq8w | lq8w_grp_m1_r32_w2_a.json seed=3 N=1003 | MISMATCH n=2278 diff=23 max_abs=2.31266e-05 max_ulp=64908 first_diff=44 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_111k_deq_q8s32EPKhPtjjl | deq_q8s32 | deq_q8s32.json seed=1 N=77 | MISMATCH n=179936 diff=941 max_abs=0.5 max_ulp=1 first_diff=165 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_111k_deq_q8s32EPKhPtjjl | deq_q8s32 | deq_q8s32.json seed=2 N=77 | MISMATCH n=179936 diff=913 max_abs=0.5 max_ulp=1 first_diff=292 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_111k_deq_q8s32EPKhPtjjl | deq_q8s32 | deq_q8s32.json seed=3 N=77 | MISMATCH n=179936 diff=879 max_abs=0.5 max_ulp=1 first_diff=204 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_111k_deq_q8s32EPKhPtjjl | deq_q8s32 | deq_q8s32.json seed=1 N=1003 | MISMATCH n=2343072 diff=11831 max_abs=0.5 max_ulp=1 first_diff=165 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_111k_deq_q8s32EPKhPtjjl | deq_q8s32 | deq_q8s32.json seed=2 N=1003 | MISMATCH n=2343072 diff=12110 max_abs=0.5 max_ulp=1 first_diff=292 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_111k_deq_q8s32EPKhPtjjl | deq_q8s32 | deq_q8s32.json seed=3 N=1003 | MISMATCH n=2343072 diff=11850 max_abs=0.5 max_ulp=1 first_diff=204 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_111k_deq_q8g64EPKhPtjjl | deq_q8g64 | deq_q8g64.json seed=1 N=77 | MISMATCH n=182400 diff=166959 max_abs=2.5 max_ulp=32510 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_111k_deq_q8g64EPKhPtjjl | deq_q8g64 | deq_q8g64.json seed=2 N=77 | MISMATCH n=182400 diff=166747 max_abs=2.5 max_ulp=32510 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_111k_deq_q8g64EPKhPtjjl | deq_q8g64 | deq_q8g64.json seed=3 N=77 | MISMATCH n=182400 diff=166547 max_abs=2.5 max_ulp=32508 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_111k_deq_q8g64EPKhPtjjl | deq_q8g64 | deq_q8g64.json seed=1 N=1003 | MISMATCH n=2375168 diff=2168958 max_abs=2.5 max_ulp=32512 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_111k_deq_q8g64EPKhPtjjl | deq_q8g64 | deq_q8g64.json seed=2 N=1003 | MISMATCH n=2375168 diff=2170000 max_abs=2.5 max_ulp=32512 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_111k_deq_q8g64EPKhPtjjl | deq_q8g64 | deq_q8g64.json seed=3 N=1003 | MISMATCH n=2375168 diff=2169530 max_abs=2.5 max_ulp=32512 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_lq8w_grpILi1ELi32ELi2EEEvNS0_5LqGrpEPKtl | lq8w | lq8w_grp_m1_r32_w2_small.json seed=1 N=77 | MISMATCH n=426 diff=15 max_abs=2.38419e-07 max_ulp=4 first_diff=2 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_lq8w_grpILi1ELi32ELi2EEEvNS0_5LqGrpEPKtl | lq8w | lq8w_grp_m1_r32_w2_small.json seed=2 N=77 | MISMATCH n=426 diff=11 max_abs=1.19209e-07 max_ulp=2 first_diff=7 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_lq8w_grpILi1ELi32ELi2EEEvNS0_5LqGrpEPKtl | lq8w | lq8w_grp_m1_r32_w2_small.json seed=3 N=77 | MISMATCH n=426 diff=12 max_abs=1.19209e-07 max_ulp=2 first_diff=4 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_lq8w_grpILi1ELi32ELi2EEEvNS0_5LqGrpEPKtl | lq8w | lq8w_grp_m1_r32_w2_small.json seed=1 N=1003 | MISMATCH n=2278 diff=167 max_abs=2.38419e-07 max_ulp=4 first_diff=2 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_lq8w_grpILi1ELi32ELi2EEEvNS0_5LqGrpEPKtl | lq8w | lq8w_grp_m1_r32_w2_small.json seed=2 N=1003 | MISMATCH n=2278 diff=172 max_abs=2.38419e-07 max_ulp=4 first_diff=7 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_lq8w_grpILi1ELi32ELi2EEEvNS0_5LqGrpEPKtl | lq8w | lq8w_grp_m1_r32_w2_small.json seed=3 N=1003 | MISMATCH n=2278 diff=164 max_abs=2.38419e-07 max_ulp=4 first_diff=4 | bitwise | 2026-10-10 |
