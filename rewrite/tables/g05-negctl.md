# Kernel match table (U22, PLAN.md p7-12)

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_110k_exp_gu_wILi16ELi2ELi0EEEvPKtPKhPKilllPt | exp_gu | exp_gu_w_16_2_0_small.json seed=1 N=1 | MISMATCH n=6432 diff=4495 max_abs=0.5 max_ulp=120 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_exp_gu_wILi16ELi2ELi0EEEvPKtPKhPKilllPt | exp_gu | exp_gu_w_16_2_0_small.json seed=2 N=1 | MISMATCH n=6432 diff=3722 max_abs=1 max_ulp=298 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_exp_gu_wILi16ELi2ELi0EEEvPKtPKhPKilllPt | exp_gu | exp_gu_w_16_2_0_small.json seed=3 N=1 | MISMATCH n=6432 diff=4636 max_abs=0.25 max_ulp=111 first_diff=1 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_114k_deq_q4cp_v8uILi2ELb0EEEvPKhPtjjllllNS0_5DeqCbE | deq_q4cp | deq_q4cp_v8u_2_0.json seed=1 N=4096 | MATCH n=12352 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_114k_deq_q4cp_v8uILi2ELb0EEEvPKhPtjjllllNS0_5DeqCbE | deq_q4cp | deq_q4cp_v8u_2_0.json seed=2 N=4096 | MISMATCH n=12352 diff=64 max_abs=inf max_ulp=65536 first_diff=9216 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_114k_deq_q4cp_v8uILi2ELb0EEEvPKhPtjjllllNS0_5DeqCbE | deq_q4cp | deq_q4cp_v8u_2_0.json seed=3 N=4096 | MISMATCH n=12352 diff=64 max_abs=inf max_ulp=65536 first_diff=5696 | bitwise | 2026-10-10 |
| _ZN5q4moe13k_q4moe_itemsILi64EEEvPKiS2_S2_iPNS_4ItemES4_Pi | q4moe | q4moe_items_64_s70.json seed=1 N=512 | MISMATCH n=155664 diff=49862 max_abs=35840 max_ulp=35840 first_diff=20 | bitwise | 2026-10-10 |
| _ZN5q4moe13k_q4moe_itemsILi64EEEvPKiS2_S2_iPNS_4ItemES4_Pi | q4moe | q4moe_items_64_s70.json seed=2 N=512 | MISMATCH n=155664 diff=49892 max_abs=35845 max_ulp=35845 first_diff=20 | bitwise | 2026-10-10 |
| _ZN5q4moe13k_q4moe_itemsILi64EEEvPKiS2_S2_iPNS_4ItemES4_Pi | q4moe | q4moe_items_64_s70.json seed=3 N=512 | MISMATCH n=155664 diff=49847 max_abs=35850 max_ulp=35850 first_diff=20 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_exp_dn_wILi0EEEvPKtPKhPKiPKflllPtS3_S9_SA_ | exp_dn | exp_dn_w_0_plain.json seed=1 N=1 | MISMATCH n=2592 diff=498 max_abs=0.000244141 max_ulp=1 first_diff=5 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_exp_dn_wILi0EEEvPKtPKhPKiPKflllPtS3_S9_SA_ | exp_dn | exp_dn_w_0_plain.json seed=2 N=1 | MISMATCH n=2592 diff=496 max_abs=0.000488281 max_ulp=1 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_exp_dn_wILi0EEEvPKtPKhPKiPKflllPtS3_S9_SA_ | exp_dn | exp_dn_w_0_plain.json seed=3 N=1 | MISMATCH n=2592 diff=469 max_abs=0.000244141 max_ulp=1 first_diff=1 | bitwise | 2026-10-10 |
