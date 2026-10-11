# g30 negative controls

Rewrite copies with a deliberate bound error, run on the reaching specs: lq4w.hip:52 with G-1 above 128; linear_q4.hip:89 dropping the last element above K=1000. MISMATCH expected only above the injected bound.

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_16k_lq4wILi1ELi16ELi1ELb0ELi0ELi1EEEvPKhPKtPvlllllS6_f | lq4w | lq4w_g30_thr_c1.json seed=1 N=100 | MATCH n=24 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_16k_lq4wILi1ELi16ELi1ELb0ELi0ELi1EEEvPKhPKtPvlllllS6_f | lq4w | lq4w_g30_thr_c1.json seed=1 N=128 | MATCH n=24 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_16k_lq4wILi1ELi16ELi1ELb0ELi0ELi1EEEvPKhPKtPvlllllS6_f | lq4w | lq4w_g30_thr_c1.json seed=1 N=129 | MISMATCH n=24 diff=24 max_abs=36 max_ulp=174 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_16k_lq4wILi1ELi16ELi1ELb0ELi0ELi1EEEvPKhPKtPvlllllS6_f | lq4w | lq4w_g30_thr_c1.json seed=1 N=300 | MISMATCH n=24 diff=24 max_abs=26.25 max_ulp=33005 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_114k_linear_q4_mtILi16ELi2ELi2ELi128EEEvPKhPKtPtlllll | linear_q4 | linear_q4_mt_16_2_2_128.json seed=1 N=255 | MATCH n=138 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_114k_linear_q4_mtILi16ELi2ELi2ELi128EEEvPKhPKtPtlllll | linear_q4 | linear_q4_mt_16_2_2_128.json seed=1 N=384 | MATCH n=138 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_114k_linear_q4_mtILi16ELi2ELi2ELi128EEEvPKhPKtPtlllll | linear_q4 | linear_q4_mt_16_2_2_128.json seed=1 N=1001 | MISMATCH n=138 diff=91 max_abs=1.79688 max_ulp=125 first_diff=0 | bitwise | 2026-10-11 |
| _ZN7halogen12_GLOBAL__N_114k_linear_q4_mtILi16ELi2ELi2ELi128EEEvPKhPKtPtlllll | linear_q4 | linear_q4_mt_16_2_2_128.json seed=1 N=1002 | MISMATCH n=138 diff=76 max_abs=2 max_ulp=134 first_diff=0 | bitwise | 2026-10-11 |
