# Kernel match table (U22, PLAN.md p7-12)

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_113k_bf16_to_f32EPKtPfl | bf16 | bf16_to_f32.json seed=1 N=4096 | MISMATCH n=4160 diff=4096 max_abs=8.71609e+20 max_ulp=2.79701e+09 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_18k_linearEPKtS2_Ptl | linear | linear_dot.json seed=1 N=1003 | MISMATCH n=128 diff=1 max_abs=0.25 max_ulp=1 first_diff=5 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_16k_lbfwILi2ELi4EEEvPKtS3_Ptll | lbfw | lbfw_2_4.json seed=1 N=1003 | MISMATCH n=2070 diff=2003 max_abs=40 max_ulp=33514 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_110k_linear_mILi8EEEvPKtS3_Ptll | linear_m | linear_m_8.json seed=1 N=1003 | MISMATCH n=122 diff=46 max_abs=0.5 max_ulp=31245 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_111k_linear_mtILi8ELi3ELi256EEEvPKtS3_Ptll | linear_mt | linear_mt_3.json seed=1 N=1003 | MISMATCH n=157 diff=2 max_abs=0.25 max_ulp=1 first_diff=21 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_118k_linear_f32_q4_mtILi8ELi2ELi4ELi256EEEvPKhPKtPflllll | linear_f32 | linear_f32_q4_mt_2.json seed=1 N=1002 | MISMATCH n=122 diff=48 max_abs=0.266167 max_ulp=255384 first_diff=0 | bitwise | 2026-10-10 |
