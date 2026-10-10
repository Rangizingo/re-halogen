# U09 round 3: HtItem field meanings from the device-side writer

Verdict: CLOSED for the record layout (size, four fields, meanings). Wiring residue is listed at the end and does not change the layout.

Static only. No binary run. All slice paths are under `work/kernels-dequant/` unless noted; line numbers are lines of the slice file (the `.s` files are `llvm-objdump --mattr=+gfx1151` output).

Short names used below:

- IG2 = `_ZN7halogen12_GLOBAL__N_116k_ht_items_groupILi2EEEvPKiiPNS0_6HtItemES5_PiS6_S6_.s` (`<4>` has the same 659-line shape)
- SLOTS = `_ZN7halogen12_GLOBAL__N_116k_ht_items_slotsEPKiiPNS0_6HtItemES4_PiS5_.s` (103 lines)
- RT = `work/kernels/_ZN7halogen12_GLOBAL__N_118k_route_topk_w_grpILi2EEEvPKfPiPfiPNS0_6HtItemES7_S4_S4_S4_S4_.s` (1471 lines; not in kernels-dequant)
- GUGS = `..._115k_ht_gemv_gu_gsILi3ELi8ELi2EEEv...s`, DNG = `..._114k_ht_gemv_dn_gILi3ELi8ELi2ELi2EEEv...s`, GU3 = `..._112k_ht_moe_gu3ILi3ELi1ELi0EEEv...s`, GLG = `..._115k_i4r_rows_gl_gILi1ELi2EEEv...s`

## Result

```
struct HtItem {        // 16 B, written by the GPU, read as one s_load_b128 / global_load_b128
  int32 expert;        // +0   expert id (index into the per-layer expert weight blob)
  int32 tile;          // +4   weight-matrix tile index inside that expert: 0..4 (gate_up) or 0..19 (down)
  int32 begin;         // +8   first position in the expert-sorted slot order (inclusive)
  int32 end;           // +12  end position of this expert's run (exclusive)
};
```

"Slot" = token * 10 + k, k in 0..9 (topk = 10 is baked into the kernels, see C5). "Position" = index in the slot list after a stable sort by expert id.

## Writer: k_ht_items_group<2>

Kernel args (from the mangled name `PKi, i, PHtItem, PHtItem, Pi, Pi, Pi` and the loads): 0x00 expert id per slot `int*` (IG2:5-8 `s_load_b64 s[4:5], null`, `global_load_b32 v1, v2, s[4:5]`), 0x08 slot count n (IG2:2 `s_load_b32 s30, ..., 0x8`), 0x10 HtItem array A (s[20:21]), 0x18 HtItem array B (s[22:23]), 0x20 int array (s[24:25]), 0x28 int array (s[26:27]), 0x30 int total (s[28:29]) (IG2:16-17 `s_load_b256 s[20:27], s[0:1], 0x10`, `s_load_b64 s[28:29], s[0:1], 0x30`). One workgroup, thread t handles slot t (t < n <= 128).

### Step 1: per-slot rank and count (IG2:20-195)

Each thread stages ids into LDS (IG2:9 `ds_store_b32 v2, v1`) then loops over all j. The hidden VOP3 words decode as follows (second dword of each pair is printed by objdump as a wrong VOP2 line; fields decoded by hand from the raw words):

- `.long 0xd44a....` = `v_cmp_eq_u32` (opcode 0x4a): ids[j] == ids[t]. It feeds `v_add_co_ci v5, null, 0, v5` (IG2:176 region, tail loop at :168-188), so v5 = count of slots with my expert.
- `.long 0xd441....` = `v_cmp_lt_i32` (0x41): ids[j] < ids[t]. With `s_and_b32 s0, s0, vcc_lo` (eq AND j < t, IG2:178) and `s_or_b32 vcc_lo, s1, s0` then `v_add_co_ci v3, null, 0, v3, vcc_lo` (IG2:181-182): v3 += (ids[j] < ids[t]) OR (equal AND j < t). So v3 = position of slot t in a stable sort by expert id.
- For the first slot of each expert (no earlier equal id) v3 = number of slots with a smaller expert id = start of that expert's run, v3 + v5 = end of the run. IG2:367 `v_add_nc_u32_e32 v4, v3, v5` computes exactly that: v4 = begin + count.

Stores of this step:
- IG2:198 `global_store_b32 v2, v3, s[26:27]`: int array at kernarg 0x28 [slot] = position (rank).
- IG2:196 `v_mul_u32_u24_e32 v1, 0x199a, v0` then `.long 0x7f083880` and `.long 0x7e083981` (IG2:201-202). Decoded: VOP1 opcode 0x1c with vdst 132 / src 128 = `v_mov_b16 v4.h, 0`; vdst 4 / src 385 (0x181 = v1.hi) = `v_mov_b16 v4.l, v1.h`. 0x199a = 6554 and (t*6554)>>16 = floor(t/10) for t < 128, so v4 = t / 10 = token index. IG2:207 `global_store_b32 v[6:7], v4, off` with v[6:7] = s[24:25] + (position << 2) (IG2:199 `.long 0xd73c0006` = `v_lshlrev_b64` by 2 of v3). So int array at 0x20 [position] = token index of the slot at that position.
- Item count per expert: IG2:220 `ds_store_b32 v2, v1 offset:512` with v1 = (count + 1) >> 1 for first-slot threads (IG2:213-218 `v_add_nc_u32 v1, 1, v5` / `lshrrev 31` / `ashrrev 1`), else 0. So items per expert = ceil(count / 2) for `<2>`. (For `<4>` the same code in its own slice is expected to use ceil(count / 4); not re-read here.)
- IG2:654 `global_store_b32 v0, v1, s[28:29]`: thread 0 sums all LDS counts (IG2:622-653, `ds_load_b128 ... offset:512`) and stores the total item count at kernarg 0x30.

### Step 2: record emission (IG2:225-593)

Each first-slot thread loops over its items `v85 = ceil(count/2)` (IG2:233-239 `ds_load_b32 v85, v2 offset:512`, `v_cmp_lt_i32 0, v85`; loop-back `s_cbranch_execnz` at IG2:591). Per iteration it writes two groups of records:

Array A (kernarg 0x10, s[20:21]): 5 records per item.
- Base: `.long 0xd73c0056` (`v_lshlrev_b64 v[86:87], 4, v[83:84]`, IG2:483) then `.long 0xd7006a5e` (`v_add_co_u32 v94, vcc_lo, s20, v86`, IG2:490). v83 starts at 5 * v6 (IG2:372 `v_lshl_add_u32 v83, v6, 2, v6` = v6*4+v6) and advances by 5 per iteration (IG2:586 `v_add_nc_u32_e32 v83, 5, v83`). v6 = sum of item counts of all experts with a smaller id (IG2:289-330, `ds_load_b32 v7 offset:512` taken only when ids[j] < my id). So A record index = 5 * (items of lower experts + k).
- Stores to v[94:95]: offsets 0, 16, 32, 48, 64 (IG2:585 `offset:16`, IG2:586 region `off`, IG2:502 `offset:32`, IG2:503 `offset:48`, IG2:511 `offset:64`). Five 16-B records.

Array B (kernarg 0x18, s[22:23]): 20 records per item.
- Base: `.long 0xd73c0058` (`v_lshlrev_b64 v[88:89], 4, v[81:82]`, IG2:485), `.long 0xd7006a60` (`v_add_co_u32 v96, vcc_lo, s22, v88`, IG2:494). v81 advances by 20 per iteration (IG2:489 `v_add_nc_u32_e32 v81, 20, v81`); its start value comes from `.long 0xd72c0051` at IG2:370 (VOP3 op 0x32c, src0 v6, src1 inline const 20 = v6 * 20; decoded by hand, the 20 step on :489 corroborates it).
- Stores to v[96:97]: offsets 0, 16, 32, ... 304 (IG2:500 `offset:32` through IG2:589 `offset:304`; offsets 0 and 16 at IG2:581-582 region). 20 records.

Record contents (per record r, 4 dwords):
- dword 0 = v1 = the expert id reloaded from LDS (IG2:30 / :225 `ds_load_b32 v1, v2`). Constant across all records of the item.
- dword 1 = the tile index: for array A the sequence of SGPR/VGPR constants 0 (s1 = 0 at IG2:368 `v_mov_b32_e32 v2, s1`), 1 (IG2:423 `s_mov_b32 s3, 1`), 2, 3, 4 (IG2:369 `s_mov_b32 s2, 4`); for array B the same plus `s_mov_b32 s2, 5` ... `s_mov_b32 s2, 19` (IG2:442 `s_mov_b32 s2, 19`). Every one of the 20 B records takes its dword 1 from a distinct constant 0..19.
- dword 2 = v3 (`v_dual_mov_b32 v88, v3 :: v_dual_mov_b32 v89, v4` pattern, IG2:507, :517, :530 ...) = begin, advancing by the group size: IG2:587 `v_add_nc_u32_e32 v3, 2, v3` (G = 2 for `<2>`).
- dword 3 = v4 = begin_of_expert + count = end of the expert's run. Not advanced across iterations (v4 is only written once, IG2:367).
So the 5 (or 20) records of one item are identical except dword 1, and items of one expert differ only in dword 2 (stepped by G).

## Second writer: k_ht_items_slots (cleaner, confirms the layout)

SLOTS is the no-grouping variant (one item per slot). Args: 0x00 `int*` expert id per ITEM, 0x08 n, 0x10 HtItem A, 0x18 HtItem B, 0x20 `int*`, 0x28 `int*` (SLOTS:3-7).
- SLOTS:12 `s_mul_i32 s0, s3, 20` bounds the B block to n*20 threads. SLOTS:14 is the /5 magic (`0x66666667`), `ashr 3` at SLOTS:21 gives j = i / 20. SLOTS:38 `global_load_b32 v3, v[3:4], off` = ids[j]. SLOTS:39 `v_mad_u64_u32 v[4:5], null, 0xffffffec, v7, v[0:1]` = i - 20*j (tile 0..19). SLOTS:42 `global_store_b128 v[8:9], v[3:6], off` with v3 = ids[j], v4 = i - 20j, v5 = j, v6 = j + 1 (SLOTS:40-41 `v_mov v5, v7`, `v_add_nc_u32 v6, 1, v7` at SLOTS:25). Stored to s[6:7] (kernarg 0x18).
- SLOTS:44 `s_mul_i32 s0, s3, 5`: the A block. SLOTS:76 `v_mad_u64_u32 v[4:5], null, v7, -5, v[0:1]` = i - 5j (tile 0..4); SLOTS:79 same b128 store to s[4:5] (kernarg 0x10).
- SLOTS:99-100: `global_store_b32 v[1:2], v5, off` with v5 = (i*0x66666667 >> 2) = i / 10 into s[8:9] (kernarg 0x20), and `global_store_b32 v[3:4], v0, off` with v0 = i into s[10:11] (kernarg 0x28).

This is the identity-permutation version of IG2: rank[i] = i (0x28), token[i] = i / 10 (0x20), range = [j, j+1). It confirms field order {expert, tile, begin, end}, the 5 and 20 tile counts, and that kernarg 0x20 = token index per position, 0x28 = rank per slot, both independent of the decode in IG2.

## Third writer: k_route_topk_w_grp (fused router + grouping)

RT carries the same grouping code (same 20 and 5 record store ladder: RT:1312-1401 `global_store_b128 ... offset:32 ... offset:304`; `0x199a` at RT:993; rank store RT:1005 `global_store_b32 v5, v3, s[26:27]`; token store RT:1006 `global_store_b32 v[7:8], v4, off`; total count RT:1466 `global_store_b32 v0, v1, s[28:29]`). Slot count is n*10: RT:814 `s_mul_i32 s34, s33, 10`. It also stores the topk ids and weights first (RT:805-807 `global_store_b128 v13, v[1:4], s[10:11]` + `offset:16` + b64 `offset:32` = 40 B per token = 10 x 4 B). Its output args sit 0x10 higher than IG2's (s[20:27] from 0x20, s[28:31] from 0x40, RT:810-811). Same HtItem layout.

## Consumers (each field tied to its use)

k_ht_gemv_gu_gs<3,8,2> (GUGS):
- Item pick: workgroup id >> 3 (GUGS:5-11), bound `*count * 5` (GUGS:13 `s_mul_i32 s3, s3, 5`, count at kernarg 0x18). Record: `s_load_b128 s[8:11], s[8:9], null` at (kernarg 0x10 + 16 * item) (GUGS:20-22). So A is indexed by 5*item+tile, consistent with 5 tiles/item.
- dword 0 (s8): `s_mul_i32 s23, s8, 0x12c000` (GUGS:118; 1,228,800 B expert stride). expert id.
- dword 1 (s9): `s_mul_i32 s27, s9, 0x3c000` (GUGS:120). 0x12c000 / 0x3c000 = 5 (checked), 0x3c000 / 0x600 = 160 and 0x12c000 / 0x600 = 800 (0x600 is the row stride: `v_mad_i64_i32 v[2:3], null, 0x600, v14, s[30:31]`, GUGS:130). So dword 1 selects a 160-row tile of the 800-row gate_up matrix: 5 tiles.
- dword 2 (s10) and dword 3 (s11): GUGS:28 `s_add_i32 s20, s11, -1`; lanes below 0x140 take s10, lanes above take s10 + 1 (`v_cmp_lt_u32 0x13f, v4`, `v_add_co_ci v1, null, s10, 0`, GUGS:46-49), `v_min_i32 v5, s20, v1` clamps to end - 1, then GUGS:60 `global_load_b32 v5, v[5:6], off` from kernarg 0x20 (s[12:13]) at that position: the token index for the position. So begin and end bound a position window of G = 2 tokens, clamped by the expert's end. This matches IG2 stepping begin by 2 while keeping end fixed.

k_ht_gemv_dn_g<3,8,2,2> (DNG):
- DNG:1579 `global_load_b32 v2, v1, s[18:19]` (count) then DNG:1591 `global_load_b128 v[1:4], v1, s[0:1]` with s[0:1] = kernarg 0x10 + 16 * workgroup id (DNG:1587-1590). `v_readfirstlane s4, v1` (expert), `s3, v2` (tile) (DNG:1594-1595).
- DNG:1629-1632: `s_mul_i32 s6, s4, 0x96000` (614,400 B per expert) and `s_mul_i32 s7, s3, 0x7800` (30,720 B per tile); 614400 / 30720 = 20 (checked). So dn uses 20 tiles per expert, array B.
- dwords 2 and 3: DNG:1599-1606 `v_add_nc_u32 v1, -1, v4` (end - 1), `v_add_co_ci v2, null, 0, v3, vcc` for lanes > 0x4f (begin + 1), `v_min_i32 v5, v2, v1`, then `v_mad_i64_i32 v[5:6], null, 0x500, v5, s[14:15]` = row address in the gate_up output (0x500 B per position). Same [begin, min(begin+1, end-1)] window, indexed by position.

k_i4r_rows_gl_g<1,2> (GLG):
- GLG:1462 `s_mul_i32 s0, s4, 5` with s4 = workgroup / 20 (GLG:1452-1456 `0x66666667`, `ashr 3`): record 5*item of that array. GLG:1470 `global_load_b64 v[1:2], v0, s[0:1] offset:8` loads dwords 2-3 only, `s_min_i32 s3, s1, 2` with s1 = end - begin (GLG:1475-1477): row count = min(end - begin, 2). Dword 0 is loaded separately at GLG:1471. Because every record of an item carries the same begin/end, reading record 5*item is enough.

k_ht_moe_gu3<3,1,0> (GU3), already in moe-host.md, re-checked:
- GU3:7 `s_load_b128 s[24:27], s[4:5], null` at (kernarg 0x10 + 16 * workgroup id) (GU3:3-7).
- GU3:13 `s_sub_i32 s0, s27, s26` then GU3:15 `s_min_i32 s27, s0, 0x80`: token count = min(end - begin, 128). GU3:85 `s_mul_i32 s2, s24, 0x12c000` (expert). GU3:137 region `s_mul_i32 s14, s25, 0xa0`: dword 1 * 160 = a row offset, i.e. the same 160-row tile unit as GUGS (0x3c000 B = 160 rows * 0x600 B). The first-pass "x160/x128" reading is therefore a tile index, not a generic slot index.
- GU3 consumes a one-record-per-workgroup array whose writer is not one of the three kernels above (see residue R2).

## Corrections to earlier notes

- C1. analysis/closure/moe-host.md:11 and moe-dv-host-r2.md:U09 "inferred layout {expert id, tile/slot index, token range begin, token range end}": confirmed, with dword 1 pinned to "tile index in the expert's weight matrix" (5 tiles of 160 rows for gate_up, 20 tiles of 30,720 B for down).
- C2. moe-host.md:18 says IG2 writes "20 consecutive 16-B records per pass to both HtItem output pointers". Corrected: array A (kernarg 0x10) gets 5 records per item, array B (kernarg 0x18) gets 20 records per item, with different base strides (5 vs 20 records per item, IG2:586 and :489). Both loops advance once per item.
- C3. moe-host.md:20 ties the host arena carve at 018be390.c:315 (`local_1db8 * 0x10`, then `* 0x40`) to "the HtItem list". The device writers need at least 80 B (array A) and 320 B (array B) per item, and items <= slots. 16 B and 64 B per slot do not fit; that carve is not the HtItem arrays (inferred; the real item arrays' sizing was not found).
- C4. "dword 2 token range begin" is not a token index. It is a position in the expert-sorted slot order. The token index comes from the int array at kernarg 0x20 indexed by that position (GUGS:60).
- C5. topk = 10 is hardcoded in the device code (`0x199a` multiply for slot/10, `0x66666667 >> 2` in SLOTS:99, `s_mul_i32 s34, s33, 10` in RT:814). The host grid factor `0x32 = 50 = 10 * 5` (moe-dv-host-r2.md) is consistent.

## Residue (does not affect the layout)

- R1. Host wiring: which of arrays A and B and which of the int arrays at 0x20 and 0x28 reach each consumer is not read from the host (Ghidra dropped the hipLaunchKernel argument setup, per moe-host.md:4). The strides in consumers fix it for gu_gs (A, 5 tiles) and dn_g (B, 20 tiles). GLG divides the value it loads by 10 (GLG:1536 `0x66666667`), so it reads a slot-valued array, not the token array at 0x20; which array that is was not resolved.
- R2. The writer for k_ht_moe_gu2/gu3/dn* item arrays (prefill-size path, ranges up to 128 tokens) was not identified. Field meanings there follow from the same consumers; the writer would have end - begin = chunk size. Not needed for the layout.
- R3. dn_g's count pointer (kernarg 0x18, DNG:1579): the bound is `workgroup >= *count`, but array B has 20 records per item. Whether the host passes the item count or 20 * item count was not determined (the writer only emits item count at kernarg 0x30, IG2:654). The grid (`200*n + 20*rsi`, moe-dv-host-r2.md U10) suggests a separate bound. Inferred, unverified.
- R4. `k_ht_items_group<4>` was not re-read; G = 4 is inferred from the `<4>` template parameter and the identical 659-line shape.

## Reproduce

```bash
cd /Users/peterb/Documents/AI/re-halogen/work/kernels-dequant
# writer: stores, constants, hidden-opcode lines
grep -n -E 'global_store|0x199a|add_nc_u32_e32 (v3, 2|v81, 20|v83, 5)|s_load_b(256|64) s\[(20:27|28:29)\]' *k_ht_items_groupILi2*
# second writer
grep -n -E 's_mul_i32 s0, s3|0xffffffec|-5, v\[0:1\]|global_store' *k_ht_items_slots*
# consumers: expert/tile strides and item loads
grep -n -E '0x12c000|0x3c000|s_add_i32 s20, s11, -1|s_mul_i32 s3, s3, 5' *k_ht_gemv_gu_gsILi3ELi8ELi2*
grep -n -E '0x96000|0x7800|global_load_b128 v\[1:4\], v1, s\[0:1\]' *k_ht_gemv_dn_gILi3ELi8ELi2ELi2*
grep -n -E 'offset:8|s_min_i32 s3, s1, 2|s_mul_i32 s0, s4, 5' *k_i4r_rows_gl_gILi1ELi2*
# arithmetic
python3 -c "print(0x96000/0x7800, 0x12c000/0x3c000, 0x3c000/0x600, 0x12c000/0x600)"   # 20.0 5.0 160.0 800.0
# hidden VOP3 opcodes: opcode = (word >> 16) & 0x3ff; 0xd44a -> 0x4a v_cmp_eq_u32, 0xd441 -> 0x41 v_cmp_lt_i32,
# 0xd73c -> v_lshlrev_b64, 0xd7006a.. -> v_add_co_u32. VOP1 0x7e083981 -> op (w>>9)&0xff = 0x1c v_mov_b16 v4.l, v1.h
python3 -c "
for w in (0xd44a0008,0xd4410007,0x7f083880,0x7e083981): print(hex(w),'op',hex((w>>16)&0x3ff),'vop1 op',hex((w>>9)&0xff),'vdst',(w>>17)&0xff,'src0',w&0x1ff)"
```
