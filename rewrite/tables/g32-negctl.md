# Kernel match table (U22, PLAN.md p7-12)

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_114k_ht_gemv_gu_tILi3ELi8ELi2ELi0EEEvPKjPKDF16_PKiS5_S5_PDF16_NS0_4HtSdEi | ht_gemv | ht_gemv_gu_t_3_8_2_dense_s2b.json seed=1 N=40 | MISMATCH n=5120 diff=1 max_abs=6.10352e-05 max_ulp=1 first_diff=2269 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_114k_ht_gemv_gu_tILi3ELi8ELi2ELi0EEEvPKjPKDF16_PKiS5_S5_PDF16_NS0_4HtSdEi | ht_gemv | ht_gemv_gu_t_3_8_2_dense_s5bm.json seed=1 N=140 | MATCH n=56320 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_114k_ht_gemv_gu_tILi3ELi8ELi2ELi0EEEvPKjPKDF16_PKiS5_S5_PDF16_NS0_4HtSdEi | ht_gemv | ht_gemv_gu_t_3_8_2_dense_s2b.json seed=1 N=40 | MISMATCH n=5120 diff=270 max_abs=1 max_ulp=75 first_diff=6 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_114k_ht_gemv_gu_tILi3ELi8ELi2ELi0EEEvPKjPKDF16_PKiS5_S5_PDF16_NS0_4HtSdEi | ht_gemv | ht_gemv_gu_t_3_8_2_dense_s5bm.json seed=1 N=140 | MISMATCH n=56320 diff=730 max_abs=1 max_ulp=75 first_diff=6 | bitwise | 2026-10-11 |
