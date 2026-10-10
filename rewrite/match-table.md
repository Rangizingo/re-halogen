# Kernel match table (U22, PLAN.md p7-12)

What a MATCH row proves: the rewrite and the vendor kernel wrote identical output bytes (raw bits, NaN payloads included, plus a guard region past the end) for that spec, and both wrote it (an output equal to its pre-launch fill fails as UNTOUCHED). It does not prove identical intermediate math: a change that disappears in the output rounding (for example a 1-ulp change of an f32 constant before a bf16 store) still matches (refuter on PR #19). Element-wise kernels should also run an exhaustive input spec (every bf16 bit pattern) where the input type allows.

| instance | family | test spec | result | tolerance | date |
| --- | --- | --- | --- | --- | --- |
| _ZN7halogen12_GLOBAL__N_112k_scale_siluEPKtPtlf | scale | k_scale_silu.json seed=1 N=4096 | MATCH n=4160 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_scale_siluEPKtPtlf | scale | k_scale_silu.json seed=2 N=4096 | MATCH n=4160 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_scale_siluEPKtPtlf | scale | k_scale_silu.json seed=3 N=4096 | MATCH n=4160 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_scale_siluEPKtPtlf | scale | k_scale_silu.json seed=1 N=100003 | MATCH n=100067 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_scale_siluEPKtPtlf | scale | k_scale_silu.json seed=2 N=100003 | MATCH n=100067 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_scale_siluEPKtPtlf | scale | k_scale_silu.json seed=3 N=100003 | MATCH n=100067 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_scale_siluEPKtPtlf | scale | k_scale_silu_wide.json seed=1 N=4096 | MATCH n=4160 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_scale_siluEPKtPtlf | scale | k_scale_silu_wide.json seed=2 N=4096 | MATCH n=4160 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_scale_siluEPKtPtlf | scale | k_scale_silu_wide.json seed=3 N=4096 | MATCH n=4160 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_scale_siluEPKtPtlf | scale | k_scale_silu_wide.json seed=1 N=100003 | MATCH n=100067 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_scale_siluEPKtPtlf | scale | k_scale_silu_wide.json seed=2 N=100003 | MATCH n=100067 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
| _ZN7halogen12_GLOBAL__N_112k_scale_siluEPKtPtlf | scale | k_scale_silu_wide.json seed=3 N=100003 | MATCH n=100067 diff=0 max_abs=0 max_ulp=0 | bitwise | 2026-10-10 |
