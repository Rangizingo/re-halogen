# U18 third pass: k_dv_quant8 output planes and the done-line dv counter writers

Inputs read: work/kernels-spec/_ZN7halogen11k_dv_quant8EPKtPKiS3_iPhi.s (1266 lines; windows 1-24, 150-262, 340-412, 1199-1237), work/decomp/flash_serve/{0184c9a0,01845e50,01882d20,018f5320,018715c0,01871780,01851770,01860e60}.c, analysis/closure/moe-dv-host-r2.md, ngram-dv-host.md. `work/` is not in this worktree; it was read from /Users/peterb/Documents/AI/re-halogen (read-only). Static only, no binary run. Slice line numbers below are line numbers in that .s file. Ghidra address = file vaddr + 0x100000.

Decoder caveat: the slice's objdump leaves some VOP3 64-bit ops as `.long 0xd700....`. Their operands are decoded below by hand from the raw second dword (stated where used). VOP3 word0 bits 7:0 = vdst, word1 bits 8:0 = src0, 17:9 = src1; values below 256 are SGPRs, 256+ are VGPRs.

## 1. Output plane offsets (CLOSED, one inference flagged)

Kernarg map (from the s_load operands): arg0 `bf16*` at 0x00, arg1 `int*` at 0x08, arg2 `int*` at 0x10, arg3 `int` at 0x18, arg4 `uint8*` at 0x20, arg5 `int` at 0x28.
- `s_load_b128 s[4:7], s[0:1], null` (line 225) loads arg0 into s[4:5] and arg1 into s[6:7].
- `s_load_b64 s[4:5], s[0:1], 0x10` (line 207) loads arg2.
- `s_load_b64 s[0:1], s[0:1], 0x20` (line 226) loads arg4 (the output base).
- `s_load_b32 s3, s[0:1], 0x28` (line 4) is arg5, called A below (2560 on the host launch).

### Row stride
The row-stride computation (slice lines 213-217):
```
s_lshr_b32 s2, s8, 28        ; s8 = A >> 31 (line 9), so this is the +15 rounding term
s_add_i32  s2, s3, s2
s_ashr_i32 s2, s2, 4         ; A / 16
s_add_i32  s2, s2, s3        ; S = A + A/16
```
Cited as the four consecutive instructions at slice lines 213-217 (`s_lshr_b32 s2, s8, 28` ... `s_add_i32 s2, s2, s3`). S = output row stride in bytes. For A = 2560 (the only host launch, moe-dv-host-r2.md U18): S = 2560 + 160 = 2720 = 80 blocks x 34 bytes.

### Destination row base
- Row index r = tid / (A/32) (the 64-bit divide, slice lines ~25-190; thread bound `s_mul_i32 s4, s9, s6` line 18 = (A/32)*arg3). Block-in-row b = tid - r*(A/32). The exact register path through the undecoded `.long 0xd72c/0xd72d` ops was not decoded, so r and b are inferred from the divide shape.
- Destination map: slice line 230-231 region: `.long 0xd7006a00 ; v_cndmask_b32_e32 v1, s6, v14, vcc_lo` is `v_add_co_u32 v0, vcc, s6, v14` (src0 = s6 = arg1 low, src1 = v14), where v[14:15] = r << 2 (the `.long 0xd73c000e` / `v_cndmask_b32_e32 v1, 2, v2` pair at slice lines 215-216: shift count 2 on v2). Then slice line 256 `global_load_b32 v0, v[0:1], off` loads `arg1[r]`.
- Slice line 346 (file line 342 in the 340-360 window, address 0x2253C0): `v_mad_i64_i32 v[1:2], null, v0, s2, s[0:1]`. So

  `rowbase_dst(r) = arg4 + (int64) arg1[r] * S`

### Planes inside a destination row (stores)
- Quant plane: 32 `global_store_b8` per thread at immediate offsets 0..31 (first store `global_store_d16_hi_b8 v[37:38], v0, off` line 359 / `global_store_b8 v[37:38], v40, off` line 409 at offset 0; last `global_store_b8 v[3:4], v9, off offset:31` line 1235). The base for offset 0 is `.long 0xd7006a25 ; v_cndmask ... v1, v1, v3` = `v_add_co_u32 v37, vcc, v1, v3` (word1 = 0x02020701: src0 = 0x101 = v1, src1 = 0x103 = v3) at line 354-355, followed by `v_add_co_ci_u32_e64 v38, null, v2, v4, vcc_lo` (line 356). v[3:4] = (b << 5) sign-extended: `v_lshlrev_b32_e32 v3, 5, v5` (line 228) and `v_ashrrev_i32_e32 v4, 31, v3` (line 230), with v5 = b. So

  `quant(r, b, k) = arg4 + arg1[r]*S + 32*b + k`, k = 0..31 (value = int8, round-nearest-even, clamp +/-127, per moe-dv-host-r2.md / kernels-speculation.md section 4).
- Scale plane: single `global_store_b16 v[1:2], v0, off` (line 1236, v0 = f16 absmax/127 from `v_cvt_f16_f32_e32 v0, v8`, line 347). Its address:
  - line 1226-1228: `.long 0xd7006a01` with word1 `0x02000701` (line 1226 operand dword: src0 = 0x101 = v1, src1 = (0x02000701 >> 9) & 0x1ff = 3 = s3 = A): `v_add_co_u32 v1, vcc, v1, s3`; then `v_add_co_ci_u32_e64 v2, null, s8, v2, vcc_lo` (line 1228; s8 = A >> 31 = 0, the high half of sext(A)). So rowbase + A bytes.
  - line 1224: `v_lshlrev_b32_e32 v5, 1, v5` gives 2*b, then `v_ashrrev_i32_e32 v6, 31, v5` and the second add pair (line 1231 `.long 0xd7006a01`, `v_add_co_ci_u32_e64 v2, null, v2, v6, vcc_lo` line 1230) add 2*b.

  `scale(r, b) = arg4 + arg1[r]*S + A + 2*b` (f16, 2 bytes)

Answer, per destination row d (d = arg1[r]) of stride S = A + A/16:

| bytes | content |
|---|---|
| [0, A) | int8 quants, block b at 32*b .. 32*b+31 |
| [A, A + A/16) | f16 scales, block b at A + 2*b |

For A = 2560: quants [0, 2560), scales [2560, 2720), S = 2720. A/16 = 2*(A/32) holds exactly the scales, so rows are dense with no padding (arithmetic: 80*2 = 160 = 2560/16). Inferred (not proven): the A/32 blocks-per-row is exact (A multiple of 32; 2560 is).

### Source side (closes the "which int arg is which" question)
- Source map = **arg2**. Line 207 loads arg2 into s[4:5]; the `v_add_co_u32 v3, vcc, s4, v14` at slice lines 221-222 area (`v_cndmask_b32_e32 v1, s4, v14, vcc_lo` line 221, ci pair at line 222 `v_add_co_ci_u32_e64 v4, null, s5, v15`) forms `arg2 + 4r`; line 227 `global_load_b32 v1, v[3:4], off` reads `arg2[r]`; line 235 (0x225134) `v_mad_i64_i32 v[0:1], null, v1, s3, 0` = `arg2[r] * A` (elements); then `<<1` (the `.long 0xd73c0006`/`v_cndmask_b32_e32 v1, 1, v3` pair, shift 1) plus `+ 2*32*b` and arg0 give the four `global_load_b128` at slice lines 248-251 (offsets 0/16/32/48 = 32 bf16).

  `src(r, b) = (const char*) arg0 + 2*(arg2[r]*A + 32*b)`, 64 bytes read.
- Destination map = arg1 (above). The vector at 0x38(%rsp) vs 0x40(%rsp) in the host launch (moe-dv-host-r2.md) was not re-matched to arg1/arg2 here, so which std::vector<int> is "source token ids" and which is "destination slots" is unproven on the host side; the kernel side is certain (arg2 = source rows, arg1 = destination rows).

Net: k_dv_quant8 copies/quantizes row `arg2[r]` of a bf16 matrix (rows of A elements, A=2560) into slot `arg1[r]` of a packed int8 table whose rows are 2720 bytes (quants then f16 scales). Combined with the vocab-row source (lm_head, inferred in the earlier pass), this builds the draft-vocab lm_head quantized weight table, not a q-row copy. This matches the earlier finding that kept q rows are f32 (018f9fb0.c).

## 2. Writers of the done-line dv counters (CLOSED)

The done line reads `*(param_13+0xa0)`, `+0xa8`, `+0xb0` (0184c9a0.c:90-95 `uVar11 = *(undefined8 *)(param_13 + 0xa0); uVar10 = *(ulong *)(param_13 + 0xa8); uVar14 = *(undefined8 *)(param_13 + 0xb0);`) and prints them last (0184c9a0.c:99-103). param_13 is the per-request object (01845e50.c:157 passes `plVar18 = plVar13[2]`, 01845e50.c:150-155 read other counters from the same object).

The only writer found for those three offsets is FUN_01882d20 (vaddr 0x1782d20), grep over all decomp files for `0xa0)`, `0xa8)`, `0xb0)` together; other files with all three offsets (0184a0f0 move-assign, 01852ca0 unrelated fields, 01884460, 018897f0) do not increment them (0184a0f0.c:363,380,410 are moves). 01882d20.c, quoted in full where relevant:

```
 9  if (*param_1 < 1) { return 0; }
12  iVar1 = param_1[2];
13  if (-1 < iVar1) {
14    *(long *)(param_4 + 0xa0) = *(long *)(param_4 + 0xa0) + 1;                 // dv_tokens
16    bVar3 = FUN_018f5320(param_2,iVar1);
17    *(long *)(param_4 + 0xa8) = *(long *)(param_4 + 0xa8) + (ulong)(bVar3 ^ 1); // dv_misses
18  }
19  if ((char)param_1[5] == '\0') {
20    FUN_018715c0(param_1,param_2,param_3);
23    ... (2 < iVar1) && (0xf < window size)
27      FUN_01871780(param_1);
29      if (iVar2 != iVar1) { *(long *)(param_4 + 0xb0) = *(long *)(param_4 + 0xb0) + 1; }  // dv_switches
```
- Mapping of offsets to done-line fields: +0xa0 = dv_tokens, +0xa8 = dv_misses, +0xb0 = dv_switches (field order equals wire-protocol.md section 3.3 positions 27-29; the order is from the snprintf arg order, so inferred but consistent).
- param_1 is the draft-vocab selector (an `int` struct): `[0]` = number of candidate vocab subsets N, `[2]` = current subset id (-1 = none), `[3]` = call counter, `[4]` = ring head, `[5]` byte = frozen flag, vector at `+0x18..+0x20` = window of recent tokens (capacity 0x40, 018715c0.c:36-45), array at `+0x30` = per-subset hit counts. param_2 is the engine context; param_3 is the token; param_4 is the request object that holds the counters.
- FUN_018f5320(ctx, subset, token) (018f5320.c): bitset membership. `lVar1 = *(long *)(param_1 + 0x70)`; `token < 0x3ca00` (248,320 rows, same vocab size as the logit row, 018f9fb0.c); bit = `*(ulong *)(*(long *)(lVar1 + 0x30) + subset*0x18 + (token>>6)*8) >> (token & 63)`. Returns 1 if the table or subset is absent. In 01882d20.c:16 the third argument is not passed in the decompile (param_3 is parked in rdx at line 15 `param_3 = param_3 & 0xffffffff`), so the token is passed in a register the decompiler dropped; inferred, since 018715c0.c passes it explicitly in the same pattern.
- Semantics: dv_tokens = tokens fed to the selector while a subset is active; dv_misses = those tokens that are NOT in the active subset's bitset (a miss means the reduced draft vocab could not have produced it); dv_switches = times the selector changed the active subset. The selector (FUN_018715c0) adds each token to a 64-entry window and bumps `count[i] += inSubset_i(token)` for all N subsets, evicting the oldest (018715c0.c:21-47). Every 4th call once the window holds > 15 tokens it runs FUN_01871780 (01882d20.c:23-27), which scores subsets from the counts (a geometric-weighted hit rate, floating-point formula at 01871780.c:29-80; not fully decoded) and writes the winner to `param_1[2]` (01871780.c:197 `param_1[2] = iVar3;`). The exact scoring formula is not decoded; only that it picks a subset by hit-rate is read.
- Callers (per-token): 01851770.c:106 `FUN_01882d20(param_1 + 0x4a,param_1[0x21],uVar2,param_1);` (loop over accepted tokens, same loop does `param_1[5] = param_1[5] + 1`), 01851dc0.c:284 (same shape), 01860e60.c:15 `FUN_01882d20(param_1 + 0x250,...)`, 0182ea80.c:8357, 8532, 8635, 8940 (`pauVar8 + 0x25` with `param_4 = pauVar8`). `param_1 + 0x4a` in a `long *` = byte 0x250 of the request object, the same object that carries +0xa0..+0xb0, consistent with param_4 = the request.

Link to U18 kernels: the subset bitsets live in the same structure that FUN_01891c50 loads from `HALOGEN_DRAFT_VOCAB_FILE` (moe-dv-host-r2.md); the int8 table built by k_dv_quant8 is the reduced lm_head for the active subset (inferred from names plus the A=2560 row length; the join between the selector's subset id and the table rows (arg1 destination slots) is not traced).

## Verdict

CLOSED for the two goals asked: (1) output plane offsets as a formula in kernel args (above, S = A + A/16, quants at [0,A), f16 scales at A + 2b, dest row = arg1[r], source row = arg2[r]); (2) writers of the done-line dv counters (FUN_01882d20, offsets +0xa0/+0xa8/+0xb0 = dv_tokens/dv_misses/dv_switches).

What remains outside the stated goal (bounded, not blocking the row): host-side match of the two uploaded std::vector<int> (0x38/0x40(%rsp)) to arg1/arg2; the scoring formula in FUN_01871780 (constants at DAT_00113b58..00113f30 not read); the non-8-bit branch at 0x1792530 (`k_dv_quant`); how the selector's active subset id indexes the quantized table. The dest row r and block b split from tid is inferred (undecoded `.long` ops).

## Reproduce

```
cd /Users/peterb/Documents/AI/re-halogen
sed -n '1,24p;195,262p;340,360p;1199,1237p' work/kernels-spec/_ZN7halogen11k_dv_quant8EPKtPKiS3_iPhi.s
grep -n 'global_store\|s_load' work/kernels-spec/_ZN7halogen11k_dv_quant8EPKtPKiS3_iPhi.s
cat work/decomp/flash_serve/01882d20.c work/decomp/flash_serve/018f5320.c
sed -n 1,50p work/decomp/flash_serve/018715c0.c
sed -n 85,110p work/decomp/flash_serve/0184c9a0.c
cd work/decomp/flash_serve && grep -n 'FUN_01882d20' *.c
# VOP3 field decode check (word1 0x02000701): python3 -c "w=0x02000701;print(w&0x1ff,(w>>9)&0x1ff)"  # 257, 3
```
