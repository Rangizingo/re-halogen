# Refuter 3, PR #24 (p7-12-wave2 @ 70be761)

Worktree at 70be761, kernels-disasm-v2 symlinked. Nothing committed or pushed. Scripts were run from the agent scratchpad.

## 1. pytest 145 passed, 4 skipped: UPHELD
`python3 -m pytest -q tests` -> `145 passed, 4 skipped in 49.39s`.

## 2. p7-12 proof `1738 1738`, exit 0: UPHELD
The quoted one-liner printed `1738 1738`, `exit 0`.

## 3. U22 row, AUDIT OK count, ledger: UPHELD
- `grep -c '^| U22 .*| CLOSED |$' analysis/unknowns.md` -> 1
- `grep -c 'AUDIT OK' rewrite/const-audit.md` -> 1738; `grep -c 'AUDIT GAP\|AUDIT ERROR'` -> 0
- Status column counted from the table: CLOSED 25, PARTIAL 0, OPEN-BOUNDED 1, UNOBSERVABLE 1 (25 0 1 1). The Totals line (unknowns.md:43) says CLOSED 25.
- `(exception:` rows in const-audit.md: 900, matching the "900 rows" in the U22 row, match-table caveat and PLAN.

## 4. Coverage on the rewrite of record: UPHELD
Own script (scratchpad/cov.py): chains from rewrite/superseded.tsv (11 links, all ends exist in rewrite/kernels/); for each of the 1,738 instances, collect the spec names on its MATCH rows, resolve each spec in rewrite/specs/ and read compare.rewrite_src, require at least one whose file is the end of its chain.
Output: `1738 bad 0`; `spec names without committed file/src: 0 of 3673`.
Note: the original (superseded) files still have MATCH rows (e.g. ht_gemv.hip 180, attn_fa2.hip 29), which the match-table caveat states. No instance relies on them alone.

## 5. Integer gate soundness: UPHELD, with two notes
Synthetic listings through const_audit.audit() (scratchpad/syn.py), bound 0xa0 in `v_cmp_lt_u32 vcc_lo, 0xa0, v1`:
- same bound -> AUDIT OK
- bound changed to 0x90 and 0xc8 -> `AUDIT GAP int 0xa0 (bound)` (both)
- bound missing -> `AUDIT GAP int 0xa0 (bound)`
- bound+1 (0xa1) and bound-1 (0x9f) -> AUDIT OK (as specified)
- unread literal 0x12345 absent from rewrite -> `AUDIT GAP int 0x00012345 (no reader)`; present -> OK
- float 1-ulp change -> `AUDIT GAP 0x3e22f983`
- changed bound with excused={0xa0} -> `AUDIT OK (exception: ...)`

Notes (not refutations): (a) a +-1 change of a bound passes by design, so a loop bound off by one is not caught by the audit; only the MATCH rows cover it. (b) the exception path accepts any reason text.

8 random exception rows (random.seed(70).sample of the 900; script pick.py). Vendor lines read in kernels-disasm-v2/codeobj_53000.s, with the enclosing kernel header:
1. k_exp_gu_w<8,1,0>: cites :2269273-4 `v_cmp_lt_u32 0x47`, `v_cmpx_gt_u32 0x48`. Present, in k_exp_gu_w<8,1,1> (sibling instance, as the text says). exp_gu.hip:84 `for (j*R + l < 80...)`. OK.
2. k_linear_q4_mt<16,6,4,128>: :642940 `v_add_co_u32 ... 0x380`. Present in this instance. linear_q4.hip:88-89 `for (long i = t; i < K; i += B)`. OK.
3. k_lq4w<6,32,2,1,4,0>: 0xa at :857887 `v_lshrrev_b16 ... 10` (k_lq4w<6,32,2,1,0,1>, sibling). lq4w.hip:96-97 tid<ROWS guard. OK. 0x1b at :522805 `s_lshr_b32 ... 27` (k_lq4w<1,32,2,0,0,1>, sibling); lq4w.hip:52 G = (K + ((K>>63)&31)) >> 5. OK.
4. k_lq4w<6,16,1,0,1,0>: 0x1b cite as above, sibling instance, "scripted check" over 460 instances. OK.
5. k_lq4w<5,16,1,0,1,1>: 0x80/0x81 at :523576 `s_cmpk_lt_i32 s12, 0x81`, :523579 `s_cmpk_gt_i32 s12, 0x80` (k_lq4w<1,16,1,0,0,1>, sibling); lq4w.hip:66 one loop `g0..g1 step W`. OK.
6. k_hc8_fused3<3>: :171568,:171993 `v_cmpx_gt_u32 20, v44` and :172651 `s_lshr_b32 ... 28` (k_hc8_fused3<1>, sibling). hc8.hip:79 and :89 `if (lane < 4)`, :259 nch = (K + ((K>>63)&15)) >> 4. OK.
7. k_lq8s32<3,32,2,1>: :561496 `v_cmp_gt_u32_e64 s0, 12, v0` (k_lq8s32<3,32,2,0>, sibling). lq8s32.hip:120 `if (tid < RPB)`. OK.
8. k_ht_gemv_dn_t<4,8,1,1>: :1218998 / :1151977 / :1191679 `v_min_u32 ..., 35, ...` in dn_t<3,8,1,0>, dn_g, dn_ts (siblings, "all 40 checked"). ht_gemv.hip:113 `const int ct = 5 * w + i;`. OK.

Observation: 7 of 8 cited vendor lines are in a sibling instance of the row's own template family, not the row's own instance. The text names the sibling and says how the others were checked (scripted, "same in all N"). I did not re-run those scripted all-instance checks. Every cited line holds the stated value, so UPHELD, but "checked per instance" is only partly true for these rows.

## 6. Diff a02fcc3..70be761: UPHELD
`git diff --name-only`: PLAN.md, analysis/unknowns.md, rewrite/const-audit-exceptions.d/g32.tsv, g34.tsv, rewrite/const-audit.md, rewrite/match-table.md, rewrite/reports/p712c.md. Nothing under rewrite/kernels, specs, tables, harness, superseded.tsv.
- const-audit.md: 9 rows replaced (k_dn_conv_ab_rows<1,1>, 8 k_ht_moe_gu2/gu2w). Cells 0-6 identical and the verdict text before `(exception` identical in all 9; only the exception text differs.
- match-table.md: the Coverage caveat now says "every vendor literal accounted for in the rewrite of record", states floats/integer bounds/other-integers-by-role (so MATCH is the evidence for those), and one caveat line is added (ht_gemv_c.hip edited in place by g14n after the g32b rows). The 11,043-row table body has no change.
- unknowns.md U22 row: same wording, "1,738 AUDIT OK after s7", "900 rows carry cited exceptions", other integers by role, "3 real gaps".
- p712c.md: a "Superseded in s7" paragraph added. PLAN: p7-12c / s6 / s7 text updated.
- Exception-cite checks: g32 gu2 cites (:1395993, :1396001 for gu2<3,64,0>; :1408994, :1409002 for gu2<4,64,0>) and g34 dn_conv (:1043429, :1043420) are in the diff; I did not recheck those individual lines beyond the 8-row sample.

## Verdict
No claim REFUTED. Caveats above: bound +-1 passes by design; most exception cites point at sibling instances.
