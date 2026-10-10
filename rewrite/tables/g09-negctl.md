# Kernel match table (U22, PLAN.md p7-12)

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_112k_dnc_scan3xILi2ELi4ELb1ELb0ELb1ELi1ELi0ELi1ELb0EEEvPKtPKfS5_S5_S5_S5_S5_PfPKilPtiS3_S3_S3_S5_NS_10DncTapArgsENS_11DncNormFoldE | dnc_scan3x | dnc_scan3x_def_edge.json seed=1 N=100 | MISMATCH n=2359296 diff=614400 max_abs=1.58203 max_ulp=217 first_diff=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_dnc_scan3xILi2ELi4ELb1ELb0ELb1ELi1ELi0ELi1ELb0EEEvPKtPKfS5_S5_S5_S5_S5_PfPKilPtiS3_S3_S3_S5_NS_10DncTapArgsENS_11DncNormFoldE | dnc_scan3x | dnc_scan3x_tap.json seed=1 N=100 | MISMATCH n=3932160 diff=2749523 max_abs=1.86212 max_ulp=2.10995e+09 first_diff=393216 | bitwise | 2026-10-10 |
