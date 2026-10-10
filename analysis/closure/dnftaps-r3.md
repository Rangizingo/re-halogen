# U11 round 3: DnfTaps tap count per template arg, per-format bias table

Verdict: PARTIAL (see "Verdict" at the end for exactly what remains).

Paths: kernel slices are `work/kernels-spec/` (K-SPEC) and `work/kernels-dequant/` (K-DEQ) in the main checkout; `kernels-disasm/codeobj_53000.s` is the full disassembly (DIS53). All slice line numbers are from the original slice files (`grep -n` on the file, comments included). The k_dn_fused16 slice name is `_ZN7halogen12_GLOBAL__N_112k_dn_fused16ILi<N>ELb0ELb0ELb0EEEvPKtPKfS5_PfPKilPtiPyNS0_7DnfTapsENS0_7DnfNormE.s`, written `dnf<N>` below.

## 1. Corrections to rounds 1 and 2

1. DnfTaps is 40 bytes, not 16. Code-object metadata for all 17 `k_dn_fused16` instances (`llvm-readobj --notes work/fatbin/codeobj_14aa000.hsaco`): arg at `.offset: 72` `.size: 40` `by_value` (DnfTaps), arg at `.offset: 112` `.size: 24` `by_value` (DnfNorm), `.kernarg_segment_size: 136`. Counts: 17 of 17 kernels have a 40 B arg and `kernarg_segment_size: 136` (grep -c, output 17 and 17).
2. Layout, consistent with that size and with the kernel loads: DnfTaps = `int32 idx[3]` + 4 B pad (0x48..0x57) + three 8 B pointers (0x58, 0x60, 0x68). Kernel loads: dnf1:331 `s_load_b256 s[12:19], s[0:1], 0x48` (idx0..2, pad, ptr0, ptr1) and dnf1:333 `s_load_b64 s[26:27], s[0:1], 0x68` (ptr2). The three pointers belong to DnfTaps, not DnfNorm. Round 1 said "DnfNorm = 3 pointers, one per tap"; the 40 B size says those pointers are inside DnfTaps (host side: 5 qwords, `019e4bd0.c:86-90` passes `CONCAT44(iVar14,iVar13)`, `CONCAT44(uStack_54,iVar6)`, `local_50`, `uStack_48`, `uVar5` = 2 index qwords + 3 masked pointers = 40 B, then `uVar10,uVar12,uVar7` = 24 B = DnfNorm). Host/kernel size match is the evidence; the field names are inferred.
3. DnfNorm (0x70..0x87) is never read by any of the 13 measured kernels: the only `s_load` from the kernarg segment at offsets above 0x68 would be 0x70/0x78/0x80, and the full load list per instance (section 2) has none. Inferred: DnfNorm is dead in the `<N,F,F,F>` family.
4. Round 1 stated the `s12/s13/s14` compare ladder is absent for N=0,2,3. That was a grep artifact. It is present in all 13 (section 2).

## 2. Tap count per template arg: 3 for every N in 0..12

Tap count = number of idx slots the loop tests = 3. Evidence, one row per instance (K-SPEC `dnf<N>`; the `0x48` load, the `0x68` load, and the five `s_cmp_lg_u32 s12|s13|s14, <loopctr>` lines):

| N | `s_load_b256 ..0x48` | `s_load_b64 ..0x68` | cmp s12 | cmp s13 | cmp s14 | cmp s13 (2nd) | cmp s14 (2nd) | taps |
|---|---|---|---|---|---|---|---|---|
| 0 | dnf0:322 | dnf0:316 | 2808 | 2946 | 2948 | 3117 | 3285 | 3 |
| 1 | dnf1:331 | dnf1:333 | 2885 | 2985 | 2987 | 3156 | 3324 | 3 |
| 2 | dnf2:327 | dnf2:321 | 2840 | 2978 | 2980 | 3149 | 3317 | 3 |
| 3 | dnf3:326 | dnf3:320 | 2837 | 2975 | 2977 | 3146 | 3314 | 3 |
| 4 | dnf4:326 | dnf4:320 | 2844 | 2982 | 2984 | 3153 | 3321 | 3 |
| 5 | dnf5:327 | dnf5:321 | 2699 | 2837 | 2839 | 3153 | 3321 | 3 |
| 6 | dnf6:327 | dnf6:321 | 2847 | 2985 | 2987 | 3156 | 3324 | 3 |
| 7 | dnf7:327 | dnf7:321 | 2842 | 2980 | 2982 | 3151 | 3319 | 3 |
| 8 | dnf8:327 | dnf8:321 | 2840 | 2978 | 2980 | 3149 | 3317 | 3 |
| 9 | dnf9:327 | dnf9:321 | 2837 | 2975 | 2977 | 3146 | 3314 | 3 |
| 10 | dnf10:327 | dnf10:321 | 2798 | 2939 | 2941 | 3110 | 3278 | 3 |
| 11 | dnf11:331 | dnf11:333 | 2845 | 2945 | 2947 | 3116 | 3284 | 3 |
| 12 | dnf12:347 | dnf12:349 | 2833 | 2972 | 2974 | 3143 | 3311 | 3 |

Each cell is the grep -n hit of the quoted instruction (e.g. dnf1:2885 `s_cmp_lg_u32 s12, s54`, dnf1:2985 `s_cmp_lg_u32 s13, s54`, dnf1:2987 `s_cmp_lg_u32 s14, s54`). Only s12, s13, s14 are ever compared; s15 is the pad. Nothing else in the 13 slices reads the idx slots.

Structure that is identical for N = 0..12 (grep -c per slice): `v_wmma` 37, `global_load` 80, `ds_load` 190, `s_barrier` 5, `v_exp_f32` 19, `v_rsq_f32` 3, one `s_mul_hi_i32 sX, s2, 0x55555556` (divide by 3, workgroup id into 16 x 3; dnf0:43, dnf1:199, dnf5:198, dnf12:221). Per-instance metadata (`llvm-readobj --notes`): `.group_segment_fixed_size` 24512 for all 13, `.max_flat_workgroup_size` 256, vgpr 191 or 192, sgpr 66 (N=0) or 72 (N=1..11) or 73 (N=12).

Answer for the table's first column: tap count is 3 for every template arg 0..12. It does not vary with N. A tap is live when its idx slot is not -1 (host: `019e4bd0.c:35-36` index*0x40 + 0x40 constants, see round 1 for the 0x14230 constants) and its norm pointer is non-null.

## 3. What template arg N selects: a timing probe index (new)

N is not a tap count. Evidence from the kernels:

- N=0 has no clock reads and never loads kernarg 0x40 (`Py`). Its tail is plain `global_store_b32` then `s_endpgm` (dnf0:3607). grep for `s_sendmsg_rtn_b64` and `global_store_b64` in dnf0: no matches.
- N=1..11 each contain 3 `s_sendmsg_rtn_b64 sX, sendmsg(MSG_RTN_GET_REALTIME)` (a GPU realtime clock read), load `Py` (`s_load_b64 s[0:1], s[0:1], 0x40`) in the epilogue by one lane (`v_cmpx_eq_u32 0, v9`), and do exactly one `global_store_b64 v2, v[0:1], s[0:1] offset:K` of an accumulated 64-bit tick count (s6:s7). Accumulate: dnf1:400 `s_sub_i32 s3, s6, s28` then dnf1:402 `s_add_i32 s6, s3, s54` (s6 += t_b - t_a style, inferred from the register flow).
- The store offset is 8*(N-1): N=1 none (dnf1:3648), N=2 `offset:8` (dnf2:3641), N=3 16 (dnf3:3640), N=4 24 (dnf4:3647), N=5 32 (dnf5:3645), N=6 40 (dnf6:3648), N=7 48 (dnf7:3643), N=8 56 (dnf8:3641), N=9 64 (dnf9:3638), N=10 72 (dnf10:3602), N=11 80 (dnf11:3608). So `Py` is a u64 array of 11 slots, slot N-1 written by variant N.
- Clock reads sit at a different place per N (K-SPEC line numbers of the three `s_sendmsg_rtn_b64`; entry read is ~195-196 in all): N=1: 196, 399, 2861; N=2: 195, 392, 416; N=3: 195, 405, 543; N=4: 195, 538, 657; N=5: 195, 2871, 2981; N=6: 195, 766, 768; N=7: 195, 759, 820; N=8: 195, 812, 1531; N=9: 195, 1525, 2027; N=10: 195, 2022, 2251; N=11: 196, 2257, 2819. The accumulate `s_sub_i32 sX, s6, sY` lines for N=1..11: 400, 418, 544, 658, 2983, 770, 822, 1533, 2029, 2252, 2820. `s_barrier` lines are at ~750/1520/2030/2300/2970 in each, so the clock pairs bracket different phases of the same kernel body. Which phase each N brackets is NOT mapped (would need per-region labelling of the WMMA, LDS and norm stages).
- N=12 is a different probe: dnf12:197 `s_sendmsg_rtn_b64 s[24:25], sendmsg(MSG_RTN_GET_REALTIME)`, dnf12:198 `s_load_b64 s[26:27], s[0:1], 0x40`, dnf12:201 `s_getreg_b32 s14, hwreg(HW_REG_HW_ID1)`, dnf12:217 `global_store_b64 v65, v[64:65], s[14:15] offset:1024` (per-workgroup entry record with hardware id), and at exit dnf12:3632 `s_sendmsg_rtn_b64 s[0:1]`, dnf12:3645 `global_store_b64 v4, v[0:1], s[2:3] offset:128`, dnf12:3646 `... offset:512`. N=12 therefore records per-block start time + HW id and end time, not a phase interval. The second `s_endpgm` (dnf12:3631 and 3649) is the extra exit path.
- The compute part is otherwise unchanged across N (section 2 counts), so N=0 is the production kernel and N=1..12 are instrumented copies.

Host side:
- `019e2a20` (function-map.tsv:2954 `dn_fused_launch_select`) disassembly (ai-core, r2 `pd` @ 0x18e2a20): `cmpq $0x0,0x448(%rsp)` then `je 18e2b73` (0x18e2a55-0x18e2a5e); not taken: `mov 0x450(%rsp),%r10d; dec r10d; cmp $0xb,%r10d; ja` exit, so stack arg at 0x450 is N in 1..12 (jump table). Taken: 0x18e2b73 builds a default taps block (`mov qword [rsp+0x58], -1`, `mov dword [rsp+0x60], -1`, zero pointers) and `call 0x18e4250`, the N=0 launcher. So: "0x448 arg == 0" selects the uninstrumented kernel, non-zero selects the probe variant N. Inferred: the 0x448 arg is the `Py` output buffer (the kernel only runs the probe stores when it exists).
- `HALOGEN_DN_PROBE` exists: `019cf410.c:8-15` `getenv("HALOGEN_DN_PROBE")`, `strtol`, prints `*** HALOGEN_DN_PROBE=%d: GDN output is WRONG (timing decomposition only) ***`. So the project has a documented "timing decomposition" probe whose value is a small integer; `019ca280.c:451` tests `DAT_019ff760 - 4U < 3` (probe in 4..6) for the GDN step launcher. That the same integer becomes the `k_dn_fused16` N is inferred only; the code path from `019cf410` to the `019e2a20` call was not found.
- Nobody calls `019e2a20`. `objdump -d` on ai-core for `call|jmp` to `18e2a20` returned no hits, r2 `axt 0x18e2a20` returned nothing, and no rip-relative reference. `019e4bd0` (the live dispatcher) calls only `019e4250` (the N=0 route; `objdump` hits for `call 18e4bd0` at 0x17af4a9 and 0x17aff00). So the N=1..12 probe variants have no static call site in this binary (dead, or reached by a path the search cannot see). Label: not found.
- Naming note, inferred: `DN` in `HALOGEN_DN_*` (HALOGEN_DN_PROBE message says "GDN output") reads as DeltaNet, so `k_dn_fused16` is probably a fused gated-DeltaNet step, with DnfTaps = short-conv tap positions. The earlier "down-projection" gloss is doubtful. Not proven.

## 4. Per-format bias table

"Bias" here = the constant subtracted from the stored integer before scaling. Cells cite the dequant slices (K-DEQ) unless noted. The DnfTaps kernels themselves carry no format constant; this table closes the format-side half of U11.

| Format (dtype code) | Kernel | Stored integer | Bias / zero point | Evidence |
|---|---|---|---|---|
| q4c family (5), 4-bit symmetric | `k_i4r_rows_fl_g<2>` (also `k_lq4w` family) | unsigned nibble | 8 (via f16 1024 trick) | `k_i4r_rows_fl_g<2>`:284 `v_or_b32_e32 v49, 0x64006400, v50` (f16 pair 1024.0 + n); :365 `v_mul_i32_i24_e32 v1, 0xe408, v49`. f16 0xe408 = -1032.0 (computed here with python `2**(25-15)*(1+8/1024)` and sign bit set = -1032.0). (1024+n) - 1032 = n - 8. Interpretation of the `v_mul_i32_i24` as the packed subtract is carried from kernels-moe.md:244-248, not independently re-derived: inferred |
| q6k (dtype code not checked) | `k_deq_q6k` | 6-bit unsigned | 32 | `k_deq_q6k`:163 `v_subrev_nc_u32_e32 v0, 32, v0`; `grep -c 'v_subrev_nc_u32_e32 v[0-9]*, 32, v'` = 16; :118 `v_cvt_f32_i32_e32`, :121 `v_fma_mix_f32 v13, v16, v13, neg(0)` (value * scale, zero addend) |
| q8s32 (11) | `k_deq_q8s32` | signed int8 | 0 | `k_deq_q8s32`:112 `v_bfe_i32 v33, v3, 0, 8` (24 such, grep -c); :153 `v_fma_mix_f32 v11, v15, v11, neg(0) op_sel_hi:[1,0,0]` (scale only, addend 0); scale fp16 from :91 `global_load_d16_b16 v15` |
| q8g64 (7) | `k_deq_q8g64` | unsigned byte | per-group f16 offset (no constant) | :116-119 `v_cvt_f32_ubyte0/1/2/3_e32`; 64 x `v_fma_mix_f32 vA, vA, v41, v41 op_sel:[0,0,1] op_sel_hi:[0,1,1]` (grep -c = 64): value*scale_lo + offset_hi of one packed f16 pair |
| q6g64 (24) | `k_deq_q6g64` | unsigned | per-group f16 offset (no constant) | :342 `v_fma_mix_f32 v2, v2, v4, v4 op_sel:[0,0,1] op_sel_hi:[0,1,1]`; 32 such (grep -c) |
| q4a (14) | `k_deq_aff<4,b>` (dequant-host-r2.md:44) | unsigned nibble | per-group f16 offset (no constant) | aff4:78 `global_load_b128 v[3:6]` (nibbles), :81 `global_load_b32 v11` (packed scale+offset f16), nibble `v_bfe_u32 v34, v6, 4, 4` (:94 region), `v_cvt_f32_ubyte0_e32 v7, v7` (:119 region, grep to locate), :132 `v_fma_mix_f32 v7, v7, v11, v11 op_sel:[0,0,1] op_sel_hi:[0,1,1]` (32 such). Zero point rides in the per-group offset half of v11; there is no per-format constant. Closes the "q4a zero point kernarg" item: it is data, loaded by `global_load_b32` at :81 |
| q5a (15) | `k_deq_aff<5,b>` | 5-bit unsigned = nibble + (bit4 plane & 16) | per-group f16 offset (no constant) | aff5:92 `global_load_b128 v[6:9]`, :93 and :97 `global_load_b32` (scale/offset pairs); :136 `v_and_or_b32 v9, v70, 16, v9` (merge 5th bit: value in 0..31); :142 `v_cvt_f32_ubyte0_e32`; :160 `v_fma_mix_f32 v10, v10, v4, v4 op_sel:[0,0,1] op_sel_hi:[0,1,1]` (32 such) |
| iq4nl (13) | `k_gather_iq4nl` (codeobj_53000) | 4-bit index into a 16-entry table | none; non-uniform lookup `kIq4nlValues` | symbol `_ZN7halogen12_GLOBAL__N_112kIq4nlValuesE` LOCAL OBJECT size 16 at 0x2103c0 in codeobj_53000 (`llvm-readelf -s`). Bytes (read from `binaries/flash_serve` at file 0x2633c0, python `-I`): -127 -104 -83 -65 -49 -35 -22 -10 1 13 25 38 53 69 89 113 (matches the ggml iq4_nl table; the same 16 bytes also at `halogen-tools` 0x25f3c0). Consumer: DIS53:31592 `s_getpc_b64 s[6:7]` at 0x23d950, pc+4 = 0x23d954, `s_add_u32 s6, s6, 0xfffd2a6c` (DIS53 line 31593) gives 0x23d954 - 0x2d594 = 0x2103c0; DIS53:31625 `global_load_i8 v3, v3, s[6:7]` (nibble indexes the table, signed byte); DIS53:31630 `v_fma_mix_f32 v3, v7, v3, neg(0)` (scale f16 * table value, no offset). Block = 18 B (f16 scale + 16 nibble bytes): DIS53:31419 kernel start, `s_mul_i32 s1, s11, 0x5a` (0x5a = 90 B per 5 blocks, 160 elements, `v_cmpx_gt_u32 0xa0`) |

Not covered: q5_0 (21), q5_1 (22), q8_0 (20). `k_gather_ggml<2|3|4>` (DIS53:2405553, 2405809, 2405945) are the likely kernels. `<4>` merges a 5th bit with `v_and_or_b32 v3, v8, 16, v7` then `v_cvt_f32_ubyte0_e32` and `v_fmac_f32` (DIS53 line 2405945 + 47..53) and reads a 0xa0-element / 0x78 B row, which does not obviously match q5_0 (22 B per 32) or q5_1 (24 B per 32), so the block-size mapping and bias are unproven.

## Verdict

PARTIAL.

Closed:
- Tap count per template arg: 3 for N = 0..12, all 13 instances measured, each cell cited (section 2). The DnfTaps struct is 40 B (3 idx + pad + 3 pointers); this overturns the round-1 16 B size and the round-1 "DnfNorm = 3 pointers" reading.
- Template arg N meaning: N=0 uninstrumented kernel, N=1..11 timing probes writing a u64 to `Py[N-1]`, N=12 per-workgroup start time + HW id trace (section 3), from the kernel code.
- Bias table for q6k (32), q8s32 (0), q4a and q5a (per-group f16 offset, no constant), q8g64 and q6g64 (per-group f16 offset), iq4nl (16-entry table, values listed), 4-bit symmetric (8, carried interpretation).

Remains:
1. Which kernel phase each probe N=1..11 brackets (per-N region labelling of the clock pairs; line numbers are listed above).
2. Whether `HALOGEN_DN_PROBE` is what supplies N to `019e2a20`, and the caller of `019e2a20` (no static call or jmp or rip-relative reference was found; it may be dead code in this build).
3. q5_0, q5_1, q8_0 biases and their gather/dequant kernel (`k_gather_ggml<2|3|4>` format mapping).
4. The 4-bit symmetric "8" still rests on the `0xe408` / `0x64006400` reading from kernels-moe.md; the `v_mul_i32_i24` use was not re-derived.

## Reproduce

```bash
cd ~/re-halogen   # or /Users/peterb/Documents/AI/re-halogen
# tap count: 0x48 load and the five compares, per N
cd work/kernels-spec
for n in 0 1 2 3 4 5 6 7 8 9 10 11 12; do
  f=$(ls | grep "k_dn_fused16ILi${n}ELb0ELb0ELb0E")
  grep -n -E 's_load_b(256|64) .*s\[0:1\], 0x(48|68)|s_cmp_lg_u32 s1[234], ' "$f"
done
# struct sizes (40 B DnfTaps at 72, 24 B DnfNorm at 112, kernarg 136)
/opt/homebrew/opt/llvm/bin/llvm-readobj --notes work/fatbin/codeobj_14aa000.hsaco | grep -c '\.size:           40$'
# probe stores and clock reads per N
grep -n -E 's_sendmsg_rtn_b64|global_store_b64|s_getreg' work/kernels-spec/*k_dn_fused16ILi5ELb0ELb0ELb0E*.s
# host dispatcher
ssh ai-core 'cd ~/re-halogen && r2 -q -e scr.color=0 -c "pd 22 @ 0x18e2a20; pd 22 @ 0x18e2b73" binaries/flash_serve'
ssh ai-core 'cd ~/re-halogen && objdump -d --no-show-raw-insn binaries/flash_serve | grep -E "(call|jmp).*18e2a20"'   # no output
# bias evidence
grep -c 'v_subrev_nc_u32_e32 v[0-9]*, 32, v' work/kernels-dequant/*k_deq_q6kEPKhPtjjl.s       # 16
grep -n 'v_fma_mix_f32.*op_sel:\[0,0,1\]' work/kernels-dequant/*k_deq_affILi4ELb0E*.s | head -2
# iq4nl table: symbol and consumer
/opt/homebrew/opt/llvm/bin/llvm-readelf -s work/fatbin/codeobj_53000.hsaco | grep kIq4nl
python3 -I -c "d=open('binaries/flash_serve','rb').read(); print([b-256 if b>127 else b for b in d[0x2633c0:0x2633d0]])"
sed -n '31592,31593p;31625p;31630p' kernels-disasm/codeobj_53000.s
```
