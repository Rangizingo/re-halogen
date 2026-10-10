# U12 and U14 closure: host dispatch evidence (flash_serve 0.17.3)

Method: grep over work/decomp/flash_serve/*.c (getenv strings, hipPushCallConfiguration constants), kernel registration tables (019e0890.c, 019f3590.c), kernarg loads in work/kernels/.
Caveat for every host launch below: Ghidra marks `__hipPushCallConfiguration` as noreturn, so the decompile stops at each launch. Grid/block constants survive; kernel arguments and the stub that is called do not. Anything that needs the argument list is marked inferred or left open.
Packed grid constant: low dword = x, high dword = y; z is the next argument. Example `0x800000030` = grid (48, 8).

## Finding that reframes U12: k_dnc_* is the DeltaNet chunk kernel family, not the indexer

- `019d7b70.c:201` `"/src/halogen/src/dn_chunk.hip"`; `019d7b70.c:27` `iVar3 = *(int *)(param_1 + 0x68) + 0x3f;` then `>> 6` gives `local_a4`, the number of 64-token chunks (nc).
- `019d7b70.c` fwrite text `"dn_chunk_forward: norm_read set without qinv/kinv"`. function-map names it dn_chunk_forward (high).
- Every HALOGEN_DN_* getenv sits in this path. "dnc" = dn chunk (inferred from file name and function name, strongly supported).
- The real indexer host code is `018b4750` ix_select (HALOGEN_IX_SELECT, HALOGEN_IX_WMMA, HALOGEN_INDEXER_BUDGET). Nothing in the k_dnc launch paths touches it.
- Consequence (inferred): the kernels-indexer.md section 2 label "select / top-k over a score row" for scan/scan3/ut/att is not supported by the host. Names fit the chunked gated-delta rule instead (scan = triangular scan within a 64 chunk, ut = UT transform, att = intra-chunk attention, prep = gate/beta prep). Kernel-side facts in that doc (VGPR, LDS, WMMA counts) are unaffected.

---

## U12

### Verdict: PARTIAL

### 1. scan3 template args A..G

Registered shapes (019e0890.c, `k_dnc_scan3I...`):
- Grid family: `<8,8,8,true,E,F,0>`, E in {1,2,3,8,10,12,14,16,18,20}, F in {1,2,4,8,15} = 50 variants.
- Base family: `<A,B,C,D,0,15,0>` = 19 variants, plus one `<8,8,8,true,0,15,1>`.

Answers:
- **E = HALOGEN_DN_M10P.** `019cf460.c`: value kept only if `< 4` or in `8..20` else 0; default 0. `019ca280.c:53` `switch(DAT_019ff770)` has cases 0,1,2,3,default,8,10,12,14,16,18 (case 0 = E 0, the base family; default catches 20 and odd values). This matches E in {1,2,3,8,10..20}. Case-to-instantiation mapping is inferred (arguments lost); count and value set match exactly.
- **F = HALOGEN_DN_M10G.** `019cf4a0.c`: default 0xf, values 1..15 kept else 15. `019ca280.c:56-78` tests power of two (`DAT_019ff780 - 1 < (DAT_019ff780 ^ DAT_019ff780 - 1)`), takes ctz, `switch(iVar4)` cases 0..3 = F 1,2,4,8; non power of two falls to the plain launch = F 15. F values {1,2,4,8,15} match exactly.
- **A = grid.y (inferred, strong).** In the S14 switch (`019ca280.c:287`, `switch(DAT_019ff740)`) grid y for cases 1..9 is 8,16,32,16,32,4,4,4,2. Registration order of the first nine base variants (019e0890.c:580-628) has A = 8,16,32,16,32,4,4,4,2. Same sequence. Grid x is a constant 0x30 = 48 in every case, block is 128 (`0x100000080`).
- **B, C, D:** B is the second tile int (values 2..32), C in {0,4,8,16,32}, D bool. Host evidence does not name them. D = true in the whole E/F grid and in 3 base variants; false elsewhere. STILL OPEN from host side.
- **G:** 0 everywhere except one variant `<8,8,8,true,0,15,1>`. No host selector found. STILL OPEN.
- **Default selection at 0.17.3:** HALOGEN_DN_S14 default 0x22 = 34 (`019c9fe0.c`: `return 0x22;`, values above 0x24 clamp to 0x22). Case 0x22 breaks out of the switch (`019ca280.c:387`), then grid `0x200000030` = (48, 2), block 128 (`019ca280.c:314,380,383`). Grid y = 2 fits A = 2, the scan3w/scan3x `<2,4,...>` family (inferred). S14=34 needs fp16 mirrors: fwrite `"HALOGEN_DN_S14=34: no fp16 mirrors in this scratch; running 33\n"` when `*(param_1+0x70/0x78/0x80) == 0`. So 34 = fp16-WMMA variant (scan3w/x), 33 = fallback. The M10P/M10G grid is not used by default (M10P default 0).

### 2. DncTapArgs and DncNormFold layouts

Verdict: PARTIAL (size and kernarg placement derived; field contents not).

- Both are 40 B by value (kernels-indexer.md; mangled names `NS_10DncTapArgsENS_11DncNormFoldE` in work/fatbin-manifest.json:3018).
- Kernarg placement (derived from the signature and 8 B alignment, then checked against loads): 12 base args end at 0x5c (`k` int at 0x58), 4 more pointers at 0x60..0x7f, DncTapArgs at 0x80..0xa7, DncNormFold at 0xa8..0xcf. Check in `work/kernels/_ZN7halogen12_GLOBAL__N_112k_dnc_scan3xILi2ELi4ELb1ELb0ELb1ELi1ELi0ELi2ELb0E...s`: `s_load_b256 s[44:51], s[0:1], 0x80` and `s_load_b64 s[0:1], s[0:1], 0xa0` read tap bytes 0..39 (32 + 8); `s_load_b64 s[8:9], s[0:1], 0xc0` reads NormFold offset 0x18 (0xa8 + 0x18). In this variant only one 8 B NormFold field is loaded.
- Host side: construction code is lost to the noreturn truncation. Related host facts: `019d7b70.c:188` `hipMemsetAsync(DAT_019ff7e0,0,0x28,...)` after `hipMalloc(&DAT_019ff7e0,0x28)`, a zeroed 40 B device buffer (same size, role inferred only). Chunk-args struct offsets seen: +0x68 token count, +0x70/+0x78/+0x80 fp16 mirrors, +0x88 qinv, +0x90 kinv, +0x98 norm_read mode (HALOGEN_DN_NORM_READ default 2), +0xd8/+0xe8 tested in a mode-2 branch.
- Remaining: per-field types and meaning. Needs disassembly of the host launch site (not the decompile) or per-field tracing in the scan3x kernel.

### 3. ut4 / ut5 variant flags

Verdict: PARTIAL.

- ut5 registrations (019e0890.c) are `<bool,bool,int,bool,bool,bool>` with the first bool fixed true; 24 distinct, each registered twice (48 registration strings). Distinct set: second bool in {0,1}; int in {0..3}; int 0 and 2 only with (0,0,0); int 1 with (b3,b4,b5) in {000,010,100,110}; int 3 with {000,010,011,100,110,111}. The doc's "60" does not match this module's count (24 distinct); not re-derived for other modules.
- Host env knobs feeding this launch (defaults from getter decompiles): HALOGEN_DN_UT5 default on (`019d7b70.c:95`, `*pcVar4 != '0'`), HALOGEN_DN_ATT_UT 1 (`019dddc0`), HALOGEN_DN_UT_RS 3 (`019dddf0`), HALOGEN_DN_UT_BURST 1 (`019dde20`), HALOGEN_DN_UT_KL 0 (`019dde50`), HALOGEN_DN_UT_MP 0 (bool, `019d7b70.c`). The ut5 branch needs S3A == 4 (default 4), S22 != 0 (default 1), XH != 0 (default 1); launch grid (48, nc), block 256 (`019d7b70.c:122`).
- Inferred, not proven: the int arg is HALOGEN_DN_UT_RS (range 0..3, default 3 is the richest variant set). The three trailing bools are among BURST, KL, MP in some order; b5 exists only with int 3 and implies b4, suggesting a nested option. Needs the argument setup the decompile loses.
- ut4: registered `<4,1,0,0,0>`, `<4,4,0,0,0>`, `<8,1,0,0,0>`, `<8,1,1,0,0>`, `<8,1,1,1,0>`, `<8,1,1,1,1>`. Selector is the S3A knob (`019dc670`, default 4; `019d7b70.c` tests `DAT_019ff794 < 3` and `== 4`) but case-to-variant is lost. STILL OPEN.

### 4. Which k_dnc_att variant serves decode vs prefill

Verdict: PARTIAL.

- Selector: HALOGEN_DN_S12 (`019dcbe0.c`: default 4, valid 0..10) stored in `DAT_019ff828`; launcher `019dd130.c`, `switch(DAT_019ff828)` cases 1..10 plus default.
- Grid by case, nc = `**(uint **)(param_1+8)`, block always 256, y = 0x30 = 48 (`| 0x3000000000`): cases 1, 2, 9, default: x = nc; cases 3, 4, 6, 7, 8: x = 2*nc; case 5: x = 4*nc. Case 10 first reads HALOGEN_DN_ATT_SCALE (`019dfc10`, accepted range -12..16, default 4).
- Special default path: `if (DAT_019ff828 != 4 || DAT_019ff7b0 == 0)` else grid `nc << 0x21 | 0x30` = (48, 2*nc). With defaults S12 = 4 and HALOGEN_DN_XH = 1 this path is taken.
- att9 template ints {0,4,8,12,16,-4,-8} are the ATT_SCALE value (range -12..16, default 4 gives att9<4>). Inferred from range and set match.
- att7<32,{2,4,8}> are three variants, matching the three 2*nc cases 6,7,8. Inferred.
- Decode vs prefill: no separate branch in this family. Every launch is a function of nc = ceil(tokens/64); a decode step would be nc = 1 if routed here. GDN decode appears to have its own knobs (HALOGEN_GDN_STEP_N, HALOGEN_DN_STEP_SR `0193ef80`); not traced. The named att kernel for the default path is not recoverable from the decompile. STILL OPEN.

### 5. Grid dims per row length

Verdict: CLOSED for the dimension arithmetic; scan variant mapping partial.

- Row length enters only as nc = ceil(tokens/64). Launches in `019d7b70.c`: `(ulong)local_a4 | 0x3000000000,1,0x100000100` = (nc, 48) block 256 (lines 91, 139-146); `(ulong)local_a4 << 0x20 | 0x30,1,0x100000100` = (48, nc) block 256 (122, 135); `<< 0x20 | 0x10,1,0x100000100` = (16, nc) for the final launch after the scratch hipMalloc; `| 0x30,1,0x100000080` = (48, nc) block 128 for norm_read paths.
- The scan launcher `019ca280.c` has no nc dependence: fixed (48, A), block 128, A from the variant case.
- Scratch sizing (`019d7b70.c:157`): `nc*0x9000 + nc*0xc0000 (if +0x70 set) + nc*0x180000 (if +0x20 set)`.
- 48 is a hard-coded constant (probably the head count of the shipped model, inferred).

---

## U14

### Verdict: PARTIAL

### 1. fa2 template args 4-6 (`k_attn_fa2<A,B,bool,OPT,X,Y>`)

Registered set (019f3590.c): A in {1,2}, B in {0,1}, third bool, OPT in {0..7}, X in {0,1}, Y bool; X=1 and Y appear only with OPT=5: `<..,5,1,false>` and `<..,5,1,true>`.

- **Arg 4 = HALOGEN_FA_OPT, CLOSED.** `019eb320` default 5. `019e9920.c:98` `switch(DAT_019ffa08 & 7)` cases 0..7, each launching one instantiation (same switch in `019eb380.c`).
- **Arg 5 = HALOGEN_FA_S21, strong inference.** `019eb350` default 1. The special path is `(DAT_019ffa08 & 7) == 5` and `(DAT_019ffa18 & 1)` (`019e9920.c:95`), the only place the OPT=5, X=1 variants can be reached.
- **Arg 6 = HALOGEN_FA_LPT, inferred.** `019f29e0` default 1; in `019eb380.c` the special path with LPT nonzero uses grid `(uVar2*2) + 0x100000000` = flat 1-D (2*tiles, 1) instead of (tiles, 2). Only OPT=5 variants carry the bool, consistent.
- **Args 1-2 (inferred):** A tracks block size. `019e9920.c` path (sparse FaSel): block 0x180 = 384, grid (ceil(n/16), 2) (`uVar5 = (n+15)>>4`). `019eb380.c` path (dense default): block 0x300 = 768, grid (ceil(tiles/2), 2). A = 1 vs 2 = one vs two 16-row q tiles per block. HALOGEN_FA_VB (`019eb2c0`, default 1) and HALOGEN_FA_RS (`019eb2f0`, default 1) must both be nonzero for the fa2 branch (`DAT_019ff9f8 == 0 || DAT_019ff9e8 == 0` skips it); which of VB/RS is A and which is B is inferred.
- Version select: `019e97f0` reads HALOGEN_FA_TQ: unset gives 2 for dense, 1 for sparse (FaSel present); TQ=0 returns 0, TQ=1 returns 1, other values return 2.

### 2. qs_bt2<a,b> arg b

CLOSED (with a correction to kernels-attention.md).
- `b` = HALOGEN_ATTN_QS_BT2_OPT (`019ef070`, default 0). `019ee360.c:69` `switch(DAT_019ffa68)` has cases 0,1,2,3,7,9,0xb = registered b set {0,1,2,3,7,9,11} (those instantiations have a = 0).
- `a` = HALOGEN_ATTN_QS_BT2_PROBE (`019ef020`, prints "attention output is WRONG (timing decomposition only)"). Powers of two 1..64 via a ctz switch (`019ee360.c:109-144`); error text `"HALOGEN_ATTN_QS_BT2_PROBE=%d is not instantiated (1 2 4 8 16 32 64, one at a time)"` (line 144). So `a` is a debug stage selector, and the production default is `<0,0>`, not `<16,0>`. The doc's "tile 16" reading of `<16,0>` is not supported.
- qs_bt2w: `019ee360.c:42` `uVar1 = (WPE*16 + OPT) - 99` selects combos (WPE,OPT) = (6,3),(6,7),(7,7),(8,7), registered as `w<3,6>,<7,6>,<7,7>,<7,8>`, so `w<OPT,WPE>` (HALOGEN_ATTN_QS_BT2_WPE `019ef0a0`).
- Launch shape: grid (`in_stack_00000008`, 2), block 0x40 = 64, dynamic LDS = HALOGEN_ATTN_QS_BT2_LDS << 10 (`019ef0d0`).

### 3. FaSel layout

PARTIAL (field names STILL OPEN).
- The host builds it on the stack in `019e9920.c`: `local_80 = param_4; local_78 = param_11; uStack_70 = (undefined4)param_12; local_68 = param_13; uStack_60 = param_14; uStack_5c = uStack0000000000000044;` where the last is the high dword of `FUN_01853fb0`'s result (HALOGEN_INDEXER_BUDGET; default pair 0x20000000800, i.e. 2048 tokens and 0x200; accepted range 0x800..0x2000, rounded down to a multiple of 16).
- Ghidra types and offsets here are unreliable (param_12 is a long tested `!= 0` for sparse mode, yet narrowed to 32 bits). The 4 x u64 slots cannot be assigned from this; source values are as listed.
- Proven: FaSel is only populated when `param_12 != 0` (sparse mode). `019e9920.c:81` exits with `"attn_fa: HALOGEN_FA_TQ=0 is the v1 A/B reference and has no block-sparse path."` otherwise.

### 4. Legacy k_attn and k_attn_fa dispatch

- **k_attn_fa: dispatched only with HALOGEN_FA_TQ=0 and no FaSel (kernel identity inferred).** `019e9920.c:90` `__hipPushCallConfiguration((ulong)uVar5 | 0x1800000000,1,0x100000020,1,0)` = grid (ceil(n/16), 24), block 32; this is the `iVar1 == 0` path the error text calls the v1 reference. Default TQ gives 2 (dense) or 1 (sparse), so it is never taken by default. Registered at 019f3590.c:13.
- **k_attn: STILL OPEN.** Registered in 01968680.c:2909 (`k_attnEPKtS2_S2_S2_PtiiS2_`) next to k_attn_sel. No launch site found; the noreturn truncation hides every stub call, so absence of evidence is not proof of non-use. `019ede50.c` launches grid ((n+3)/4, 2) block 256 for an unidentified kernel (not shown to be k_attn).

## What remains
- U12: B/C/D/G meaning, DncTapArgs/DncNormFold field contents, ut4/ut5 flag-to-env mapping, att kernel name for the default path. All need the host launch-site disassembly (call stubs and by-value struct stores) or a re-decompile with noreturn cleared on `__hipPushCallConfiguration`.
- U14: FaSel field names, A/B mapping of fa2 args 1-2, k_attn launch site.
