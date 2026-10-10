# U14 round 4: FaSel +0x08 id unit and +0x10 role

Scope: close the two open items from `fasel-r3.md`: (1) the unit of the ids in the `+0x08` list, (2) the role of `+0x10`. Static only; no binary was run. Kernel cites are `file:line` in `kernels-disasm/` (addresses are code-object offsets as printed in the listing). Host cites reuse r3 (`work/decomp/flash_serve/018b5ef0.c`, `01884460.c`, `019e9920.c`).

## Verdict: CLOSED

Both open items are answered from kernel instructions. Unit: one id = one 4-token block. Role of `+0x10`: per-row count of valid ids, used by FA2 as the end bound of a forward-only scan of the sorted id list. Residuals (below) do not change either meaning.

| Item | Answer | Status |
|---|---|---|
| +0x08 unit | id = 4-token block index (column of the indexer score row). 4 ids = one 16-token KV tile. | proven (writer and both consumers agree); one op decode inferred |
| +0x10 role | `counts[row]`: number of valid ids in that row's list. Not a sentinel. | proven (writer); consumer use proven by the compare shape |
| +0x08 order | ascending (compaction in column order); consumer relies on it | proven for `k_ix_select_strip`, inferred for `merge` |
| fa2 equality compare | tests "is token group `4*blk+i` in the list"; sets a 4-bit nibble of a 16-bit per-token mask | proven |

## 1. Writer of the +0x08 / +0x10 buffers: `k_ix_select_*`

Host: `018b5ef0.c:291` and `:346` call `FUN_018b4750(..., local_110 (ids), local_120 (counts), local_138 (stride))` with `local_110 = local_138 * lVar13 * 4 + _DAT_019fc730` (`:86`; `local_138 = (long)DAT_019fc574` at `:85`) and `local_120 = row * 4 + _DAT_019fc738` (`:88` for counts). `FUN_018b4750` (`018b4750.c`) is the launcher that picks `k_ix_select_strip` (short rows) or `k_ix_select_slice<8/16/32>` + `k_ix_select_merge` (long rows; error string `"ix_select: no candidate scratch for %d slices x %d rows\n"`). Stubs: `analysis/function-map-kernel-stubs.tsv` rows `018b5a60` (strip), `018b5b30/5c20/5d10` (slice), `018b5e00` (merge).

Kernel `k_ix_select_strip(float const* scores, long ld, int a, int b, int* ids, int* counts, int stride)`, `kernels-disasm/codeobj_53000.s:56233`. Kernarg map from the loads at lines 56235-56237: `s[4:5]` = `a`,`b` (0x10), `s[12:13]` = ids (0x18), `s[14:15]` = counts (0x20), `s20` = stride (0x28).

Row and block count (workgroup id `s2` = row):

```
56241 s_add_i32 s16, s4, s2          ; s16 = a + row
56242 s_add_i32 s6, s16, s5          ; + b
56243 s_add_i32 s6, s6, 1            ; + 1
56249 s_add_i32 s3, s6, s3           ; (signed round-toward-zero /4 fixup, 56246-56249)
56250 s_ashr_i32 s17, s3, 2          ; s17 = (a + row + b + 1) / 4
56252 s_cmp_gt_i32 s17, s20          ; blocks > stride ?
```

`s17` = number of visible **4-token blocks** (`>>2`) for that row. The scores row is indexed by this block number (`kernels-indexer.md` section 1: the indexer scores every 4-token block).

Short path (`s17 <= s20`, no selection needed): identity list.

```
56267 global_store_b32 v[1:2], v3, off     ; ids[row*stride + i] = i   (v3 starts at lane id, += 0x100)
56281 v_mov_b32_e32 v6, s17                ; count = s17
```

Select path (`s17 > s20`): a compaction pass over the score row. The lane's column `v9 = s11 + lane` is stored at the exclusive-prefix position, with a running total in `v6`:

```
57236 v_mov_b32_e32 v6, 0                   ; running count
57296 ds_store_b32 v5, v12 offset:1024      ; per-lane 0/1 "selected" flag, then LDS prefix scan (57296-57539)
57561 global_store_b32 v[13:14], v9, off    ; ids[row*stride + prefix] = column index v9
57572 global_store_b32 v0, v6, s[0:1]       ; counts[row] = v6   (s[0:1] = counts + row*4, 57563-57571)
```

The stored id is the **score-row column = 4-token block index**, in **ascending column order** (prefix-sum compaction, no re-sort), and `counts[row]` is the number of ids written. In the short path count = number of visible blocks; in the select path count = number of flagged columns (that it equals `stride` is inferred; the threshold derivation was not read).

`k_ix_select_merge` (`codeobj_53000.s:57661`, long-row variant) ends the same way: 16 unrolled `global_store_b32 ..., v9..v24` (59772-60254) write ids, and `global_store_b32 v3, v0, s[0:1]` (60292) writes `counts[row]` as a `v_add3_u32` chain of per-lane counts (60262-60291). Same format; identical ordering is inferred (merge comparator not read). The `slice<N>` kernels only produce candidates for merge (not read).

Host zeroing that makes "count" meaningful: `018b5ef0.c:158` `hipMemsetAsync(local_120,0,(long)(int)param_7 << 2,0)`; rows with no top-k call keep count 0.

## 2. Unit of the ids, from the two consumers

### k_sel_tilemask: 4 ids -> 1 mask bit

`kernels-disasm/codeobj_53000.s:60296`. Per row it loads `counts[row]` (`s_load_b32 s15,[s10 + row*4]`), loops `i < count` over `ids[row*arg7 + i]` (arg 7 = stride), then:

```
60385 v_ashrrev_i32_e32 v4, 7, v2      ; word index  = id >> 7
60386 v_lshrrev_b32_e32 v2, 2, v2      ; bit source  = id >> 2
60389 .long 0xd5180002                  ; VOP3 op 0x118 = v_lshlrev_b32 (gfx11 VOP2 0x18): v2 <- 1 << (id>>2)  [inferred decode]
60391 ds_or_b32 v4, v2                  ; OR bit into LDS mask word
```

`id >> 2` is the mask bit number: one mask bit per **4 ids**. One id is a 4-token block, so one mask bit = 16 tokens, matching the host mask geometry (`r12d = ceil((lastblk16+1)/32)` words, r3 section 3) and the FA2 K tile of 16 tokens. Word `id >> 7` = `(id>>2) >> 5`, 32 bits per word, so one u32 word covers 512 tokens (128 ids). The shifts `>>2` and `>>7` are decoded ops (60385-60386); the `1 << x` step is an undecoded VOP3 word. The kernel's second block confirms the bit scheme with decoded ops: `s_lshl_b32 s21, 1, s17` (60438) and `s_ashr_i32 s4, s17, 5` (just above it) on `s17 = (row+start+1) >> 4`, a 16-token block number with word = block >> 5.

Second job of tilemask (inferred from 0x2666cc-0x266770): for the row's own position `p = row + start` it ORs the bit of 16-token block `p>>4` when `(p>>4) == ((p+1)>>4)` (the `s_cmp_lt_i32 s16, s17` skip at 0x2666fc). This forces the in-progress block into the mask regardless of the id list. Not needed for the slot meanings.

### k_attn_fa2: 4 ids -> one 16-bit token mask, 4 bits per id

`k_attn_fa2<1,1,true,3,0,false>`, `kernels-disasm/codeobj_1513000.s:50925`. Per KV tile `s7` (16 tokens, the loop in r3 section 3), the kernel derives the 4-token block base `s8 = s7 << 2` (`s_lshl_b32 s8, s7, 2`, 0x62634) and checks the row's id list against `s8 + {0,1,2,3}`:

```
51682 v_cmp_gt_i32_e32 vcc_lo, s8, v132     ; list[cursor] < s8 ?  advance cursor
51690 v_cmp_eq_u32_e32 vcc_lo, v158, v132   ; v132 = cursor+1; end of list when cursor+1 == count
51756 v_cmp_eq_u32_e32 vcc_lo, s8, v132     ; list[cursor] == s8 ?
51758 v_cndmask_b32_e64 v133, 0, 15, vcc_lo ; id s8   present -> mask bits 0..3   = 0x000f
51830 v_cmp_eq_u32_e32 vcc_lo, s9, v132     ; id s8|1
51831 v_cndmask_b32_e64 v130, 0, 0xf0, vcc_lo   ; -> 0x00f0 (51833 ORs into v133)
51906 v_cndmask_b32_e64 v130, 0, 0xf00, vcc_lo  ; -> 0x0f00 (third id)
51979 v_cndmask_b32_e64 v130, 0, 0xf000, vcc_lo ; -> 0xf000 (fourth id)
51989 v_cmp_eq_u32_e32 vcc_lo, 0xffff, v133 ; all 16 tokens kept ?
52013 v_and_b32_e32 v131, 1, v133           ; then bit-per-token tests (1,2,4,8,0x10,... 52029-52139+)
```

`v133` is a 16-bit per-token keep mask for the 16-token K tile: id `4*blk+i` present in the list enables tokens `4i..4i+3` (nibble `0xf << 4i`). Hence **one id covers exactly 4 tokens**. Tokens whose bit is clear are masked out of the scores by the later per-bit tests (`v_and ... 1,2,4,...` then `v_cmp` / `ds_load` of scores); the masking op itself (set to -inf) was not read, inferred.

Nibbles forced on: when the 4-token block number is at or beyond `v162` (`v_ashrrev_i32_e32 v162, 2, v1`, line 51183, a value `>>2`; what `v1` holds was not traced, inferred as the row's own position), the code sets the nibble without consulting the list (51700 `v_dual_mov_b32 v133, 15`, then `v_or_b32 v133, 0xf0, v133` at 0x62830, `0xf00` at 51845, `0xf000` at 0x62AB0). Inferred reading: diagonal/tail blocks are always dense; the list only filters earlier blocks.

Tile-skip: the tilemask bit (`+0x00`) decides whether the whole 16-token tile is visited (r3); the `+0x08/+0x10` pair only refines inside a visited tile, per row.

## 3. Role of +0x10 (counts)

Writers: `k_ix_select_strip` count store `codeobj_53000.s:57572` (select path) and `:56281` + tail at 0x263110 (identity path; `s_branch 1484` at 56283 jumps to the same store); `k_ix_select_merge` `:60292`. Value = number of ids in the row's list. Not a sentinel, not a score.

Readers:

1. `k_sel_tilemask` loop bound (`counts[row]` load, then lanes run `i < count`).
2. `k_attn_fa2`: `global_load_b32 v158, v[1:2], off` (`codeobj_1513000.s:50986`); `v158` is zero-initialized (`v_mov_b32_e32 v158, 0`, 50972) so a row with no list has count 0. In the scan, `v_cmp_eq_u32 vcc, v158, v132` (51690, 51746, 51820, 51895, 51969) with `v132 = cursor + 1` ends the forward scan at the last valid element. That is the equality compare r3 flagged: it is the end-of-list test, not a match against an id. The id match is the separate `v_cmp_eq_u32 s8/s9, v132` series in section 2.

Cursor design (inferred from the compare shape): `v145` is a persistent index into the row's list (`v[141:142]` base + `v145*4`, 51656 onward); it only moves forward (`v_add_nc_u32 v132, 1, v145` at 51685, never decremented) and advances while `list[cursor] < 4*blk` (51682). That is correct only for an ascending list, which section 1 shows the writer produces.

## Residuals (none changes a meaning)

1. Decode of VOP3 words `.long 0xd5180002` (tilemask 60389) as `v_lshlrev_b32` is from the gfx11 opcode table, not a decoder run. The `1<<(id>>2)` bit position is consistent with the decoded `s_lshl_b32 1, s17` block, so confidence is high.
2. `v1` source of `v162` (dense-tail threshold, `codeobj_1513000.s:51183`) not traced.
3. Which host arg feeds `k_ix_select_strip` kernarg 0x10 (`a`) and 0x14 (`b`); the call at `018b5ef0.c:291` passes `0` in a matching position (inferred `a = 0`, row offset already folded into the `ids`/`counts` bases).
4. Whether the select-path flag count always equals `stride` (threshold/tie rule), and whether `k_ix_select_merge` emits ascending ids; merge comparator and `slice<N>` candidate kernels not read.
5. The third and fourth id compares reuse `s9` at 51905; the `s8|2` / `s8|3` reassignment is assumed from the `0xf00` / `0xf000` nibbles, not read.

## Reproduce

```bash
cd ~/Documents/AI/re-halogen
sed -n 56233,56290p kernels-disasm/codeobj_53000.s            # strip: block count, identity path, count
sed -n 57225,57240p kernels-disasm/codeobj_53000.s            # select path init (v6 = 0)
sed -n 57555,57575p kernels-disasm/codeobj_53000.s            # id and count stores
sed -n 60262,60293p kernels-disasm/codeobj_53000.s            # merge count store
sed -n 60296,60395p kernels-disasm/codeobj_53000.s            # tilemask: id>>7, id>>2
sed -n 60410,60445p kernels-disasm/codeobj_53000.s            # tilemask own-block bit
sed -n 51630,51770p kernels-disasm/codeobj_1513000.s          # fa2 cursor scan and nibble 0xf
sed -n 51820,51850p kernels-disasm/codeobj_1513000.s          # nibbles 0xf0, 0xf00
sed -n 51986,52030p kernels-disasm/codeobj_1513000.s          # 0xffff test, per-token bits
sed -n 280,296p work/decomp/flash_serve/018b5ef0.c            # host call of the select launcher
```
