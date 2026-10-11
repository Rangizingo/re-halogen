# Kernel match table (U22, PLAN.md p7-12)

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_19k_vis_addEPfPKfl | vis_add | vis_add_f32.json seed=1 N=4096 | MISMATCH n=4160 diff=4096 max_abs=199.982 max_ulp=2.24079e+09 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_19k_vis_addEPfPKfl | vis_add | vis_add_f32.json seed=1 N=100003 | MISMATCH n=100067 diff=100003 max_abs=199.999 max_ulp=2.24081e+09 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_19k_vis_accEPfPKfl | vis_acc | vis_acc_f32.json seed=1 N=4096 | MISMATCH n=4160 diff=4096 max_abs=199.982 max_ulp=2.24079e+09 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_19k_vis_accEPfPKfl | vis_acc | vis_acc_f32.json seed=1 N=100003 | MISMATCH n=100067 diff=100003 max_abs=199.999 max_ulp=2.24081e+09 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_vis_narrowEPKfPtl | vis_narrow | vis_narrow_small.json seed=1 N=4096 | MISMATCH n=4160 diff=2008 max_abs=7.34684e-40 max_ulp=1 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_vis_narrowEPKfPtl | vis_narrow | vis_narrow_small.json seed=1 N=100003 | MISMATCH n=100067 diff=49517 max_abs=7.34684e-40 max_ulp=1 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_vis_biasEPfPKtli | vis_bias | vis_bias_c7.json seed=1 N=4096 | MISMATCH n=4160 diff=4096 max_abs=0.00271988 max_ulp=1.45971e+06 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_vis_biasEPfPKtli | vis_bias | vis_bias_c7.json seed=1 N=100003 | MISMATCH n=100067 diff=100003 max_abs=0.00271988 max_ulp=1.96857e+09 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_vis_bias_geluEPKfPKtPtli | vis_bias | vis_bias_gelu.json seed=1 N=4096 | MISMATCH n=4160 diff=10 max_abs=0.000488281 max_ulp=5 first_diff=404 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_vis_bias_geluEPKfPKtPtli | vis_bias | vis_bias_gelu.json seed=1 N=100003 | MISMATCH n=100067 diff=379 max_abs=0.0078125 max_ulp=128 first_diff=404 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_vis_bias_geluEPKfPKtPtli | vis_bias | vis_bias_gelu_small.json seed=1 N=4096 | MATCH n=4160 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_vis_bias_geluEPKfPKtPtli | vis_bias | vis_bias_gelu_small.json seed=1 N=100003 | MATCH n=100067 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_vis_bias_geluEPKfPKtPtli | vis_bias | vis_bias_gelu_edge.json seed=1 N=4096 | MATCH n=4160 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_vis_bias_geluEPKfPKtPtli | vis_bias | vis_bias_gelu_edge.json seed=1 N=100003 | MATCH n=100067 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_vis_bias_geluEPKfPKtPtli | vis_bias | vis_bias_gelu_sweep.json seed=1 N=1000000 | MISMATCH n=1000064 diff=1 max_abs=0.000976562 max_ulp=1 first_diff=827088 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_116k_vis_ohead_rowsEPKfPfi | vis_ohead | vis_ohead_r64.json seed=1 N=64 | MISMATCH n=73728 diff=70065 max_abs=103.971 max_ulp=2.20444e+09 first_diff=72 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_116k_vis_ohead_rowsEPKfPfi | vis_ohead | vis_ohead_r64.json seed=1 N=77 | MISMATCH n=88704 diff=84300 max_abs=103.998 max_ulp=2.20449e+09 first_diff=72 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_vis_qkv_splitEPKfPtS3_S3_i | vis_qkv | vis_qkv_b128.json seed=1 N=64 | MISMATCH n=221184 diff=73619 max_abs=7.95312 max_ulp=33021 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_vis_qkv_splitEPKfPtS3_S3_i | vis_qkv | vis_qkv_b128.json seed=1 N=77 | MISMATCH n=266112 diff=88586 max_abs=7.96875 max_ulp=33022 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_vis_ropeEPfii | vis_rope | vis_rope_b64.json seed=1 N=64 | MISMATCH n=221184 diff=32609 max_abs=2.6226e-06 max_ulp=5.13483e+06 first_diff=3475 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_vis_ropeEPfii | vis_rope | vis_rope_b64.json seed=1 N=77 | MISMATCH n=266112 diff=40059 max_abs=2.6226e-06 max_ulp=5.13483e+06 first_diff=3475 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_18k_vis_lnEPKfPKtS4_Ptif | vis_ln | vis_ln_edge.json seed=1 N=1152 | MISMATCH n=4608 diff=4602 max_abs=1.09375 max_ulp=31840 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_18k_vis_lnEPKfPKtS4_Ptif | vis_ln | vis_ln_edge.json seed=1 N=1001 | MISMATCH n=4004 diff=4001 max_abs=1.07812 max_ulp=31851 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_18k_vis_lnEPKfPKtS4_Ptif | vis_ln | vis_ln_b256.json seed=1 N=1152 | MATCH n=4608 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_18k_vis_lnEPKfPKtS4_Ptif | vis_ln | vis_ln_b256.json seed=1 N=1001 | MATCH n=4004 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_113k_vis_softmaxEPKfPtS3_if | vis_softmax | vis_softmax_edge.json seed=1 N=777 | MISMATCH n=6216 diff=96 max_abs=5.96046e-08 max_ulp=166 first_diff=2 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_113k_vis_softmaxEPKfPtS3_if | vis_softmax | vis_softmax_edge.json seed=1 N=4096 | MISMATCH n=32768 diff=511 max_abs=2.98023e-08 max_ulp=764 first_diff=5161 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_113k_vis_pos_addEPfPKtiiii | vis_pos | vis_pos_edge.json seed=1 N=192 | MISMATCH n=221184 diff=221184 max_abs=2 max_ulp=2.13009e+09 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_113k_vis_pos_addEPfPKtiiii | vis_pos | vis_pos_edge.json seed=1 N=205 | MISMATCH n=236160 diff=236160 max_abs=2 max_ulp=2.13056e+09 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_114k_vis_patchifyEPKhPtiiii | vis_patchify | vis_patchify_main.json seed=1 N=48 | MISMATCH n=73728 diff=73468 max_abs=0.0078125 max_ulp=15232 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_114k_vis_patchifyEPKhPtiiii | vis_patchify | vis_patchify_main.json seed=1 N=77 | MISMATCH n=118272 diff=117532 max_abs=0.0078125 max_ulp=15232 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_vis_bias_geluEPKfPKtPtli | vis_bias | vis_bias_gelu_small.json seed=1 N=4096 | MISMATCH n=4160 diff=4096 max_abs=0.0098877 max_ulp=128 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_115k_vis_bias_geluEPKfPKtPtli | vis_bias | vis_bias_gelu_small.json seed=1 N=100003 | MISMATCH n=100067 diff=100003 max_abs=0.0102539 max_ulp=128 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_18k_vis_lnEPKfPKtS4_Ptif | vis_ln | vis_ln_b256.json seed=1 N=1152 | MISMATCH n=4608 diff=734 max_abs=0.015625 max_ulp=67 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_18k_vis_lnEPKfPKtS4_Ptif | vis_ln | vis_ln_b256.json seed=1 N=1001 | MISMATCH n=4004 diff=668 max_abs=0.015625 max_ulp=417 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_113k_vis_pos_addEPfPKtiiii | vis_pos | vis_pos_main.json seed=1 N=192 | MISMATCH n=221184 diff=220017 max_abs=3.96875 max_ulp=2.14611e+09 first_diff=1152 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_113k_vis_pos_addEPfPKtiiii | vis_pos | vis_pos_main.json seed=1 N=205 | MISMATCH n=236160 diff=232675 max_abs=3.96875 max_ulp=2.14611e+09 first_diff=1152 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_vis_attnILi8ELi32EEEvPKfPfif | vis_attn | vis_attn_8_32.json seed=1 N=200 | MISMATCH n=230400 diff=230400 max_abs=0.0396777 max_ulp=1.30135e+06 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_vis_attnILi8ELi32EEEvPKfPfif | vis_attn | vis_attn_8_32.json seed=1 N=77 | MISMATCH n=88704 diff=88704 max_abs=0.113892 max_ulp=2.75343e+06 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_vis_attnILi8ELi32EEEvPKfPfif | vis_attn | vis_attn_8_32_edge.json seed=1 N=200 | MISMATCH n=230400 diff=1825 max_abs=1.43051e-06 max_ulp=426 first_diff=2088 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_vis_attnILi8ELi32EEEvPKfPfif | vis_attn | vis_attn_8_32_edge.json seed=1 N=77 | MISMATCH n=88704 diff=372 max_abs=9.53674e-07 max_ulp=71 first_diff=312 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_vis_attn_wILi4ELb1EEEvPKfPfif | vis_attn | vis_attn_w_4_1_edge.json seed=1 N=77 | MISMATCH n=88704 diff=1952 max_abs=0.0121326 max_ulp=874871 first_diff=432 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_vis_attn_wILi4ELb1EEEvPKfPfif | vis_attn | vis_attn_w_4_1_edge.json seed=1 N=200 | MISMATCH n=230400 diff=5912 max_abs=0.0190043 max_ulp=2.65111e+06 first_diff=4176 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_vis_attn_wILi1ELb0EEEvPKfPfif | vis_attn | vis_attn_w_1_0_edge.json seed=1 N=77 | MISMATCH n=88704 diff=1771 max_abs=0.000204563 max_ulp=463 first_diff=1080 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_vis_attn_wILi1ELb0EEEvPKfPfif | vis_attn | vis_attn_w_1_0_edge.json seed=1 N=200 | MISMATCH n=230400 diff=5611 max_abs=0.00350189 max_ulp=230541 first_diff=1969 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_vis_attn_wILi1ELb0EEEvPKfPfif | vis_attn | vis_attn_w_1_0.json seed=1 N=77 | MISMATCH n=88704 diff=88704 max_abs=6.93798e-05 max_ulp=922 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_vis_attn_wILi1ELb0EEEvPKfPfif | vis_attn | vis_attn_w_1_0.json seed=1 N=200 | MISMATCH n=230400 diff=230400 max_abs=2.99215e-05 max_ulp=638 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_vis_attn_wILi1ELb0EEEvPKfPfif | vis_attn | vis_attn_w_1_0.json seed=1 N=77 | MATCH n=88704 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_vis_attn_wILi1ELb0EEEvPKfPfif | vis_attn | vis_attn_w_1_0.json seed=1 N=200 | MATCH n=230400 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_vis_attn_wILi2ELb0EEEvPKfPfif | vis_attn | vis_attn_w_2_0.json seed=1 N=77 | MISMATCH n=88704 diff=88704 max_abs=0.0203475 max_ulp=502992 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_112k_vis_attn_wILi2ELb0EEEvPKfPfif | vis_attn | vis_attn_w_2_0.json seed=1 N=200 | MISMATCH n=230400 diff=230400 max_abs=0.0138535 max_ulp=464327 first_diff=0 | bitwise | 2026-10-11 |
