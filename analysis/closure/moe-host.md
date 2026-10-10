# MoE host and kernel closure: U09, U10, U11

Inputs read: work/kernels-dequant/ (k_ht_items_group<2>, k_ht_moe_gu3<3,1,0>), work/kernels-spec/ (k_dn_fused16 x13), work/decomp/flash_serve/ (018be390, 01917610, 01928680, 0192bc30, 0192c150, 019e4250, 019e4bd0, 019e2a20), binaries/flash_serve .rodata (Ghidra base = file offset + 0x100000, per analysis/npu-engine.md:3 for the sibling binary; verified here only by the value found below being consistent).
Not found: any host function that writes HtItem records field by field. The only writer found is the GPU kernel k_ht_items_group. Host launches are hipPushCallConfiguration stubs whose argument setup Ghidra dropped.
Note: work/ was read from the shared checkout /Users/peterb/Documents/AI/re-halogen (the worktree has no work/ copy needed for reading); this file is written in the worktree copy of the repo.

## U09 HtItem byte layout

Verdict: PARTIAL (size corrected, field meaning inferred)

Answer in plain words: HtItem is 16 bytes, not 72. The "72 B" in analysis/kernels-moe.md s1.3 is a wrong reading. 0x48 in k_ht_moe_gu3 is a per-lane LDS stride, not the item stride. Each work item is four 32-bit ints, one record per workgroup. Inferred layout: {expert id, tile/slot index, token range begin, token range end}.

Evidence:
- k_ht_moe_gu3<3,1,0> kernarg 0x10 is the `PKHtItem` (signature order PKj, PKDF16, PKHtItem). work/kernels-dequant/_ZN7halogen12_GLOBAL__N_112k_ht_moe_gu3ILi3ELi1ELi0EEEvPKjPKDF16_PKNS0_6HtItemEPKiS5_S5_PDF16_.s:6-10: `s_lshl_b64 s[4:5], s[2:3], 4` (workgroup id << 4), `s_add_u32 s4, s20, s4` (s[20:21] = kernarg 0x10), `s_load_b128 s[24:27], s[4:5], null`. So the record is 16 B, indexed by blockIdx.x.
- Same file :13-15: `s_sub_i32 s0, s27, s26` then `s_min_i32 s27, s0, 0x80`. Field 3 minus field 2 = token count, capped at 128 (the M128 tile). So fields [2],[3] are a [begin,end) token range.
- Same file :85 `s_mul_i32 s2, s24, 0x12c000` (field 0 scales a per-expert weight stride of 1,228,800 B, inferred expert id); :132/:137 `s_mul_i32 s14, s25, 0xa0` and :391 `s_lshl_b32 s0, s25, 7` (field 1 scales a 160 B stride and a x128 row offset, inferred tile or slot index; exact meaning not proven).
- The 0x48 origin: same file :32 `v_mul_u32_u24_e32 v88, 0x48, v5` where v5 = lane>>3 (:27); it feeds `ds_*` addressing (LDS rows of 72 B). It never touches the HtItem pointer.
- Writer side, work/kernels-dequant/_ZN7halogen12_GLOBAL__N_116k_ht_items_groupILi2EEEvPKiiPNS0_6HtItemES5_PiS6_S6_.s:500-589: `global_store_b128 ... off offset:32`, 48, 64, ... 304, plus offset 0 and 16. That is 20 consecutive 16-B records (0..304 plus 16 = 320 B) per loop pass, written to both HtItem output pointers (:16 `s_load_b256 s[20:27], s[0:1], 0x10`; store bases s[20:21]/s[22:23] added at :491 and :495). Loop advances `v_add_nc_u32 v81, 20` (:489) and `v_add_nc_u32 v3, 2` (:587). Dword 2 and 3 of each store are filled from v3/v4 (:499, :507) and dword 0/1 from per-expert values, consistent with {expert, tile, begin, end} but not decoded to register-level proof.
- 0x199a = 6554 at :196 is `v_mul_u32_u24 v1, 0x199a, v0`. 6554 = ceil(65536/10), so this is probably a divide-by-10 magic multiply (inferred), not a 6554-B record.
- Host sizing agrees with 16 B: work/decomp/flash_serve/018be390.c:315 `uVar20 = (long)local_1db8 * 0x10 + (DAT_019fc878 + 0xff & ...)` carves a per-slot 16-B array in the forward arena (next carve at :321 is `* 0x40`, a different array). Inferred to be the HtItem list; tied to local_1db8 only by arithmetic.

What remains: register-level decode of which source value lands in each dword of the k_ht_items_group store (field order, and what dword 1 counts); why 20 records per pass (probably 20 tiles per expert slot, not shown). The other k_ht_moe_* kernels (dn3, dn2x, gu2) were not re-read; assumed to use the same 16-B load.

## U10 k_ht_tg and grouped GEMV variants

Verdict: PARTIAL

k_ht_tg (role CLOSED, name expansion STILL OPEN):
- Role: small-batch (decode, 1..8 tokens) HT trunk GEMM with split-K into an f32 partial buffer, finished by k_ht_tg_sum. Not "token gate". Signature `k_ht_tg<3|4>(const unsigned*, const DF16*, int, int, int, int, const DF16*, unsigned short*, float*)` (slice names _ZN7halogen12_GLOBAL__N_17k_ht_tgILi3EEEvPKjPKDF16_iiiiS5_PtPf.s; k_ht_tg3 has the same args). k_ht_tg_sum signature `(const float*, int, const DF16*, unsigned short*, int, int)` takes the f32 partials back to f16.
- Host selection, work/decomp/flash_serve/01917610.c:61-77: `if (param_6 < 9)` (token count under 9) and `param_4 == 0`, then `DAT_019fe680 = FUN_0192bc30()` (HALOGEN_HT_TG_DEC); `if (DAT_019fe680 != 0 ... DAT_019fe680 <= param_6 && FUN_01928680(...))` takes the tg path, otherwise `FUN_01928f40` (k_ht_lin/gemv path).
- 0192bc30.c: HALOGEN_HT_TG_DEC unset gives 0 (tg off); accepted 1..8 (`iVar1 - 1U < 8`). So tg is OFF by default and, when enabled, handles n in [TG_DEC, 8].
- 0192c150.c: HALOGEN_HT_TG_BLOCKS default 0x50 = 80, valid 1..4096 (`iVar1 - 1U < 0x1000`). HALOGEN_HT_TG_M128 getter at 0192c190 (not read further).
- Launcher 01928680.c:11-24: runs only if scratch `DAT_019fe360 != 0` and `K*n <= DAT_019fe358` (scratch capacity), grid `(ceil(K/128)*n + 7) >> 3` blocks of 256 threads (`0x100000100`).
- "tg" = "trunk GEMM" fits HALOGEN_HT_TRUNK_GEMM (019330f0) but is inferred only.

Grouped GEMV variants (k_ht_gemv_dn_g, k_ht_gemv_gu_gs): PARTIAL.
- Difference from the token-indexed siblings, from mangled signatures: gu_t `(PKj, PKDF16, PKi, PKi, PKi, PDF16, HtSd, int)`; gu_ts `(PKj, PKDF16, PKi, PKi, PKi, PDF16, Pf, Pj)`; gu_gs `(PKj, PKDF16, PKHtItem, PKi, PKi, PKi, PKi, PDF16, Pf, Pj)`; dn_t `(..., PKi, PKi, Pt, HtFold)`; dn_g `(PKj, PKDF16, PKHtItem, PKi, PKi, Pt, HtSd, int)`. So: `_t` = indexed by token slot via int arrays; `_g` = driven by the HtItem list (items grouped per expert, from k_ht_items_group / k_route_topk_w_grp); `_s` adds a split-K reduce with f32 scratch `Pf` and counters `Pj`. gu_gs = grouped + split; dn_g = grouped down. Inferred from signatures, not from the kernel bodies.
- Instances present (file names in work/kernels-dequant/): gu_gs<3,8,2>, gu_gs<3,8,4>; dn_g<3|4,8,2|4,0|2|4>. Third template arg of gu_gs = split S in {2,4}, matching the host split values below.
- Host choice, work/decomp/flash_serve/018be390.c:181-307 (decode MoE path, entered when `topk < 9 && HALOGEN_HT_DEC_FUSE (default on) && n < 0x19a (410) && param_1+0x378 == 0`):
  - split `iVar8` = HALOGEN_HT_GU_SPLIT (FUN_01940b00) if set, else 8 when topk > 1, else 1 (:235-241).
  - Token-indexed path (`_t`/`_ts`) when any of: n > 12 (`0xc < param_3`), HALOGEN_HT_GEMV_KL != 0 (`iVar23`), split != 8, HALOGEN_HT_DN_FOLD (FUN_01940b50) == 0, topk < 2, or `DAT_019fcb08 == 0` (:273-275). Grid there is `n*0x32 (+ iVar7)` x 1, 256 threads (:282).
  - Otherwise (n <= 12, KL == 0, split 8, DN_FOLD on, topk >= 2) the grid is `n*0x32*iVar8` with block 32/64/128 for `iVar8` 8/4/2 (:284-306), and `(uVar18>>1 & 0x7f) == 3` picks the 3-bit vs 4-bit instance. This is the only branch whose grid grows with the split factor, consistent with split kernels. Which stub in this branch is the `_g`/`_gs` one cannot be read: the decompile shows only hipPushCallConfiguration with no stub address.
- What remains: map each hipPushCallConfiguration to its kernel stub (needs stub addresses at the call sites, absent from the decompile); read gu_gs/dn_g bodies to confirm "grouped" beyond signatures; where `_g` (HtItem) is chosen over `_t` is not shown in 018be390.c lines 160-340.

## U11 DnfTaps and bias table

Verdict: PARTIAL (struct layout solid, "tap count per template arg" is the wrong question, bias table partial)

DnfTaps:
- Answer: tap count does not depend on template arg 0..12. DnfTaps is a fixed 3-slot struct (three int32 plus 4 B padding, 16 B) at kernarg 0x48..0x57. Each slot is -1 (tap absent) or a K position `(index+1)*64`. What the template argument 0..12 selects is still open.
- Kernel evidence: work/kernels-spec/_ZN7halogen12_GLOBAL__N_112k_dn_fused16ILi1ELb0ELb0ELb0EEEvPKtPKfS5_PfPKilPtiPyNS0_7DnfTapsENS0_7DnfNormE.s:331 `s_load_b256 s[12:19], s[0:1], 0x48`, :333 `s_load_b64 s[26:27], s[0:1], 0x68`. Only s12, s13, s14 are compared: :2885 `s_cmp_lg_u32 s12, s54`, :2985 `s_cmp_lg_u32 s13, s54`, :2987 `s_cmp_lg_u32 s14, s54`, with the loop counter stepped at :2884 `s_add_i32 s54, s54, 16`. s15 (0x54) is overwritten at :342 `s_cselect_b32 s15, -1, 0` without being read (padding). Pointers at 0x58 (s16:17), 0x60 (s18:19), 0x68 (s26:27) are three row-offset-adjusted pointers (:360-368 add s6/s7): DnfNorm = 3 pointers, one per tap (inferred).
- Host evidence: work/decomp/flash_serve/019e4bd0.c:35-36 `iVar13 = (int)lVar1 * 0x40 + _DAT_00114230; iVar14 = (int)((ulong)lVar1 >> 0x20) * 0x40 + _UNK_00114234;` masked by sign (:30-33, -1 when index < 0), :40-41 third slot `iVar6 = iVar3 * 0x40 + 0x40` or -1, and the call at :86-90 passes `CONCAT44(iVar14,iVar13)` and `CONCAT44(uStack_54,iVar6)` (uStack_54 is uninitialised: the pad). The constants at Ghidra 0x114230/0x114234 read from binaries/flash_serve file offset 0x14230 are `4000 0000 4000 0000` = 0x40, 0x40 (xxd; assumes base +0x100000). So each slot = (tap index + 1) * 64.
- A tap is only live with a non-null DnfNorm pointer: 019e4bd0.c masks the 16-B norm pair by the same sign mask (:47-52) and zeroes the third pointer when `iVar3 < 0` (:21-24, :39-41).
- Count of 3 holds for all 13 instances measured: slices `<0..12,F,F,F>` in work/kernels-spec/ have identical v_perm_b32 (80), v_fma_mix (160), v_wmma (37), ds_load (190), global_load (80) (grep -c per slice). The `s_cmp s12/s13/s14, s54` ladder (1+2+2) is present in N=1,4..12; in N=0,2,3 it is absent (cmps54=0), so those are compiled differently. That is the only structural difference measured.
- Host variant switch: 019e2a20.c:12-47 `switch(in_stack_00000028)` cases 1..12, all grid `0x100000030` (48 x 1), block `0x100000100` (256). HALOGEN_DN_FUSED_GRAM and HALOGEN_DN_FUSED_PAIR (019e4bd0.c:50-70) choose the gram/pair kernel.

Per-format bias table: PARTIAL, from existing slice evidence only, not re-extracted.
- 4-bit symmetric (q4c family, HT 4-bit): bias 8. analysis/kernels-moe.md:244-248: `v_or_b32 0x64006400` (1024.0 pair) and `v_mul_i32_i24 v1, 0xe408, v42`. f16 0xe408 = sign 1, exponent 25 (2^10), mantissa 8/1024 = -1032.0 (arithmetic checked here), so (1024+n) - 1032 = n - 8.
- q6k: -32 (`v_subrev_nc_u32` x16, kernels-moe.md:210).
- q4a asymmetric zero point: `v_xor_b32` x5 (kernels-moe.md:202) and 512 `v_bfi_b32` (kernels-moe.md:275). Inferred: the zero point comes from group data, not a constant, so no per-format constant exists for q4a. Not proven.
- q8s32, q5a, iq4nl biases not extracted.

What remains for U11: what template arg N 0..12 selects (unproven); constants for q5a, q8s32, iq4nl; the q4a zero-point source (which kernarg).

## Corrections to existing docs
- analysis/kernels-moe.md:98 and :362: "HtItem stride = 72 B" should read 16 B; 0x48 is an LDS lane stride.
- analysis/kernels-moe.md:158-159: "k_ht_tg = token gate" should read small-batch split-K trunk GEMM (name expansion inferred).
