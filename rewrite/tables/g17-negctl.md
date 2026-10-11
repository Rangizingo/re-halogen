# Kernel match table (U22, PLAN.md p7-12)

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_116k_ht_items_groupILi2EEEvPKiiPNS0_6HtItemES5_PiS6_S6_ | ht_items | ht_items_group2.json seed=1 N=100 | MISMATCH n=10280 diff=1325 max_abs=1 max_ulp=1 first_diff=3 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_111k_ht_rot_inEPKtPKDF16_PDF16_i | ht_rot | ht_rot_in.json seed=1 N=100 | MISMATCH n=256064 diff=31 max_abs=0.00390625 max_ulp=1 first_diff=14145 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_ht_rot_outEPtPKDF16_ii | ht_rot | ht_rot_out_640_zero.json seed=1 N=100 | MISMATCH n=64000 diff=34200 max_abs=0 max_ulp=0 first_diff=3 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_110k_ht_deq_tILi4EEEvPKjiiPt | ht_deq | ht_deq_t4.json seed=1 N=40 | MISMATCH n=163904 diff=157753 max_abs=0.015625 max_ulp=30430 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_113k_ht_deq_origILi4ELb0ELi16EEEvPKjiPKDF16_S5_Pti | ht_deq | ht_deq_orig_b4_f0_t512.json seed=1 N=5 | MISMATCH n=163904 diff=29 max_abs=0.0078125 max_ulp=3 first_diff=2384 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_111k_ht_tg_sumEPKfiPKDF16_Ptii | ht_tg | ht_tg_sum_s3.json seed=1 N=100 | MISMATCH n=64064 diff=1 max_abs=0.125 max_ulp=1 first_diff=53544 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_18k_ht_tg3ILi4EEEvPKjPKDF16_iiiiS5_PtPf | ht_tg3 | ht_tg3_b4_part.json seed=1 N=200 | MISMATCH n=307264 diff=307200 max_abs=11.1461 max_ulp=2.17027e+09 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_17k_ht_tgILi4EEEvPKjPKDF16_iiiiS5_PtPf | ht_tg | ht_tg_b4_out_zero.json seed=1 N=200 | MISMATCH n=102464 diff=51400 max_abs=0 max_ulp=0 first_diff=2 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_111k_ht_moe_dnILi4ELi8ELi1EEEvPKjPKDF16_PKNS0_6HtItemES5_Pt | ht_moe | ht_moe_dn_b4_zero.json seed=1 N=1 | MISMATCH n=409856 diff=8122 max_abs=0 max_ulp=0 first_diff=896 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_111k_ht_moe_guILi4ELi8ELi1EEEvPKjPKDF16_PKNS0_6HtItemEPKiS5_S5_PDF16_ | ht_moe | ht_moe_gu_b4_a.json seed=1 N=1 | MISMATCH n=102656 diff=17 max_abs=1 max_ulp=1 first_diff=6873 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_111k_ht_moe_guILi4ELi8ELi1EEEvPKjPKDF16_PKNS0_6HtItemEPKiS5_S5_PDF16_ | ht_moe | ht_moe_gu_b4_edge.json seed=1 N=1 | MATCH n=102656 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_ht_moe_gu3ILi4ELi1ELi0EEEvPKjPKDF16_PKNS0_6HtItemEPKiS5_S5_PDF16_ | ht_moe | ht_moe_gu3_b4_a.json seed=1 N=1 | MATCH n=192256 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_ht_moe_dn3ILi4ELi1ELi0EEEvPKjPKDF16_PKNS0_6HtItemES5_Pt | ht_moe | ht_moe_dn3_b4_a.json seed=1 N=1 | MISMATCH n=768256 diff=1 max_abs=0.125 max_ulp=1 first_diff=244076 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_ht_moe_gu2ILi4ELi64ELi0EEEvPKjPKDF16_PKNS0_6HtItemEPKiS5_S5_PDF16_ | ht_moe | ht_moe_gu2_b4_a.json seed=1 N=1 | MISMATCH n=192256 diff=13 max_abs=0.5 max_ulp=1 first_diff=6209 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_ht_moe_dn2ILi4ELi64ELi0EEEvPKjPKDF16_PKNS0_6HtItemES5_Pt | ht_moe | ht_moe_dn2_b4_a.json seed=1 N=1 | MATCH n=768256 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_113k_ht_moe_dn2xILi4ELi64ELi6ELi0EEEvPKjPKDF16_PKNS0_6HtItemES5_Pt | ht_moe | ht_moe_dn2x_b4_a.json seed=1 N=1 | MISMATCH n=768256 diff=30419 max_abs=91.5625 max_ulp=33896 first_diff=768 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_111k_ht_moe_guILi4ELi8ELi1EEEvPKjPKDF16_PKNS0_6HtItemEPKiS5_S5_PDF16_ | ht_moe | ht_moe_gu_b4_a.json seed=1 N=1 | MISMATCH n=102656 diff=51 max_abs=1 max_ulp=3 first_diff=446 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_ht_moe_gu3ILi4ELi1ELi0EEEvPKjPKDF16_PKNS0_6HtItemEPKiS5_S5_PDF16_ | ht_moe | ht_moe_gu3_b4_a.json seed=1 N=1 | MISMATCH n=192256 diff=36 max_abs=1 max_ulp=1 first_diff=6209 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_ht_moe_dn2ILi4ELi64ELi0EEEvPKjPKDF16_PKNS0_6HtItemES5_Pt | ht_moe | ht_moe_dn2_b4_a.json seed=1 N=1 | MISMATCH n=768256 diff=15210 max_abs=90 max_ulp=33896 first_diff=768 | bitwise | 2026-10-11 |
