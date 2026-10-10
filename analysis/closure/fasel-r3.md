# U14 round 3: FaSel slot meanings

Scope: give each of the 32 bytes of `halogen::(anonymous)::FaSel` (kernarg 0x40 of `k_attn_fa2<...>`) a meaning, by tracing the host stores in the layer function to their sources and matching them to the kernel's use of the same SGPRs. Static only; no binary was run.

Address conventions: VMA = ELF address (r2); Ghidra = VMA + 0x100000. Host layer function: Ghidra `018b86d0` (VMA `17b86d0`). FA launcher: `019e9920` (VMA `18e9920`).

## Verdict: CLOSED

All five stored fields (+0x00, +0x08, +0x10, +0x18, +0x1c) are traced to a host source and matched to a kernel use. Residuals that do not change a slot meaning are listed at the end.

| Slot | Type | Meaning | Host source | Kernel use | Status |
|---|---|---|---|---|---|
| +0x00 | `u32*` | per-q-tile KV-block bitmask (1 bit per KV block of the FA2 loop), output of `k_sel_tilemask` | `[0x18fc750]` (DAT_019fc750) | tile row pointer, bit test per KV block | proven |
| +0x08 | `int*` | per-query-row selected-block id list, row stride = +0x1c ints | `[0x18fc730]` (DAT_019fc730) | row pointer; also the sparse-mode switch | proven |
| +0x10 | `int*` | per-query-row count of valid entries in the +0x08 list | `[0x18fc738]` (DAT_019fc738) | one int loaded per row | proven (use of the loaded value inferred) |
| +0x18 | `i32` | mask row stride in u32 words (words per q tile) | `r12d` computed at `17bb356-17bb387` | multiplies tile index in the +0x00 pointer | proven |
| +0x1c | `i32` | block_topk = row stride of the +0x08 list in ints (default 512) | `[0x18fc574]` (DAT_019fc574, high dword of the indexer budget pair) | multiplies row index in the +0x08 pointer | proven |

Dense mode: all 32 bytes zero (`17bbbce`). The sparse test in the launcher is "+0x08 != 0" (section 2).

## 1. Host store sequence (the 32 bytes)

VMA `17bbc99-17bbcb2`, call to `18e9920` at `17bbcfa`:

```
17bbc99: mov  qword [rsp+0x3b8], rbx       ; +0x00
17bbca1: movupd [rsp+0x3c0], xmm0          ; +0x08, +0x10
17bbcaa: mov  dword [rsp+0x3d0], r12d      ; +0x18
17bbcb2: mov  dword [rsp+0x3d4], 0         ; +0x1c, overwritten by callee
```

Copied by value onto the outgoing stack at `17bbcc1-17bbcd9` (`movupd xmm0,[rsp+0x3f8]; movupd xmm1,[rsp+0x408]; movupd [rsp+0x30],xmm1; movupd [rsp+0x20],xmm0`; the `sub rsp,0x40` at `17bbcbd` shifts the 0x3b8..0x3d7 block to 0x3f8..0x417).

Two sources reach rbx/xmm0/r12d:

- Dense: `17bbbce: xorpd xmm0,xmm0; 17bbbd2: xor ebx,ebx; 17bbbd4: xor r12d,r12d`, entered by `17bb346: je 0x17bbbce` after `17bb336: cmp byte [rsp+0x1e0],0`. All 32 bytes zero.
- Sparse: `17bb567-17bb57e`, then `jmp 0x17bbbd7`:

```
17bb567: mov   rbx, qword [0x018fc750]        ; +0x00
17bb56e: movupd xmm0, xmmword [0x018fc730]    ; +0x08 = [fc730], +0x10 = [fc738]
17bb576: mov   r13, qword [rsp+0xb8]
17bb57e: jmp   0x17bbbd7
```

`r12d` is set at `17bb356-17bb387` (section 3). A branch-target scan of a 3000-instruction r2 listing from `17bb000` found `17bb346` and `17bb57e` as the only entries to `17bbbce`/`17bbbd7`; other branches into `17bbb00-17bbd00` are static-init guards (`17bbbe6/bf5/c04`, `17bbc64`). A jump from beyond the listing is not excluded (inferred). The `17bb583` branch (taken when `[0x18fca38] <= 0` or n < `[0x18fca38]`, `17bb305/17bb30f`) is a different path and does not reach `17bbc99` in that listing.

## 2. Slots +0x08, +0x10, +0x1c: indexer top-k output

Host producer, `work/decomp/flash_serve/018b5ef0.c`:

```
018b5ef0.c:85:  local_138 = (long)DAT_019fc574;
018b5ef0.c:86:  local_110 = local_138 * lVar13 * 4 + _DAT_019fc730;
018b5ef0.c:88:  local_120 = lVar13 * 4 + _DAT_019fc738;
018b5ef0.c:158: hipMemsetAsync(local_120,0,(long)(int)param_7 << 2,0)
018b5ef0.c:291: FUN_018b4750(pcVar1,DAT_019fc728,iVar15,local_b8,local_188,0,local_110,local_120,
```

Reading: ids for row `lVar13` start at `fc730 + 512*row*4` bytes (stride `DAT_019fc574` ints); the count is the int at `fc738 + row*4`; counts are zeroed for `param_7` rows before the top-k launch (`FUN_018b4750` at `:291`, `:346`). `DAT_019fc574` is the high dword of `FUN_01853fb0` (HALOGEN_INDEXER_BUDGET, default pair low 0x800, high 0x200), and the same value is what the launcher stores at +0x1c: `019e9920.c:33` `uStack0000000000000044 = DAT_019fc574;` and `18e996c: mov eax,[0x18fc574]; mov [r12+0x1c],eax` (indexer-attn-host-r2.md section 3). The three buffers are arena offsets: `01884460.c:1235` `_DAT_019fc730 = puVar20 + (long)local_9b0;`, `:1293` `_DAT_019fc738 = local_9e8;`, `:1296` `DAT_019fc750 = local_6b8 + local_958;` (sizes not derived).

Kernel side, `k_attn_fa2<1,1,true,3,0,false>` at 0x61700, `kernels-disasm/codeobj_1513000.s` line 50925:

```
 s_load_b256 s[8:15], s[0:1], 0x30     ; s[8:9]=kernarg 0x30 ptr, s10=n, s11=m, s[12:13]=FaSel+0x00, s[14:15]=FaSel+0x08
 s_load_b128 s[20:23], s[0:1], 0x50    ; s[20:21]=FaSel+0x10, s22=FaSel+0x18, s23=FaSel+0x1c
 s_cmp_eq_u64 s[14:15], 0               ; 0x61778  +0x08 == 0 ?
 v_mad_i64_i32 v[1:2], null, s23, v129, 0   ; 0x617ac  row * (+0x1c)
 v_cndmask_b32_e32 v1, s14, v1, vcc_lo      ; 0x617c4  + (+0x08) base
 v_add_co_ci_u32_e64 v142, null, s15, v2, vcc_lo   ; 0x617cc  per-row list pointer in v[141:142]
 v_cndmask_b32_e32 v1, s20, v1, vcc_lo      ; 0x61820  (+0x10) base
 global_load_b32 v158, v[1:2], off          ; 0x6182c  one int per row into v158
```

`v129` is the row: `v_add_lshl_u32 v155,s2,v2,4` then `v_or_b32 v129,v155,v154` (0x61794, 0x6179c), `s2` = workgroup x id, `v154 = lane & 15`, so `row = tile*16 + (lane&15)`. The row bound is `v_cmp_gt_i32 vcc,s14,v129` at 0x617fc with `s14` reloaded from kernarg 0x38 (n) at 0x617d8. The `*4` between row-times-stride and the base add sits in an undecoded VOP3 `.long 0xd73c0001` (0x617b8), so the byte scale is inferred; the int-element stride matches the host's `*4` in `018b5ef0.c:86`. `v158` and `v[141:142]` feed the compare loops at 0x62620-0x62a58 (`v_cmp_eq_u32 vcc, v158, v130` at 0x626d8, 0x627c8, 0x62904, 0x62a44).

Proven: +0x08 is a row-strided int list (stride +0x1c); +0x10 is read as one int per row; both match the host layout of fc730/fc738. Inferred: what the compare loops do with v158 and the list (row-exact filter or count bound on top of the tile mask).

### The sparse switch is "+0x08 != 0"

`019e9920.c:74` `if (param_12 != 0) {`. Ghidra's `param_12` (a `long`) is the outgoing stack slot at call-site `[rsp+0x28]`. Call-site layout at `17bbcdf-17bbcf5`: `[rsp]` = `[0x18fc7c8]` (kernarg 0x30 pointer), `[rsp+8]` = n, `[rsp+0x10]` = `[r13+0x2e4]` (m), `[rsp+0x18]` = 0, FaSel at `[rsp+0x20..0x3f]`. `[rsp+0x28]` is FaSel+0x08. So the launcher's "sparse mode" test is on the id-list pointer. `019e9920.c:78-85` then exits with `"attn_fa: HALOGEN_FA_TQ=0 is the v1 A/B reference and has no block-sparse path."` when `FUN_019e97f0(1) == 0`. This sharpens indexer-attn-host.md section 3: `param_12` is the +0x08 slot, not an independent flag.

## 3. Slots +0x00 and +0x18: the tile mask

Producer launch in the layer function, VMA `17bb34c-17bb50a`:

```
17bb34c: mov edi,1
17bb351: call 0x18e97e0              ; wrapper: FUN_019e97f0(1) << 4  (18e97e0: call 18e97f0; shl eax,4)
17bb371: mov r15d, eax               ; tile rows (16 at default sparse, version select 1)
17bb356-17bb374: r12d = (n + len - 1) >> 4   ; last 16-token KV block index (len = *(int*)[rsp+0xb0], n = [rsp+4])
17bb378-17bb387: r12d = (r12d + 32) >> 5     ; = ceil((lastblk+1)/32) u32 words
17bb38b-17bb395: ebx = (n + r15 - 1) / r15   ; q tiles
17bb3a2-17bb3bd: __hipPushCallConfiguration grid (q tiles,1,1), block 0x100
17bb3da/3e1/3fe: rbx=[fc730]  rbp=[fc738]  r13=[fc750]
17bb50a: call hipLaunchKernel, handle 0x18f6650
```

Handle `0x18f6650` relocates to `0x017bdc30` (`ir~18f6650`), and `analysis/function-map-kernel-stubs.tsv` row `018bdc30` is `k_sel_tilemask(int const*, int const*, int, int, int, unsigned int*, int, int)`. Argument pointer array at `17bb44a-17bb4a8`, in kernel order:

| arg | stub type | host value |
|---|---|---|
| 0 | `int const*` | `[fc730]` selected ids |
| 1 | `int const*` | `[fc738]` counts |
| 2 | `int` | `r15d` rows per tile (16 default sparse) |
| 3 | `int` | n (copy at `[rsp+0x88]`) |
| 4 | `int` | `*(int*)[rsp+0xb0]` start position |
| 5 | `unsigned*` | `[fc750]` bitmask output |
| 6 | `int` | `r12d` words per tile row |
| 7 | `int` | `[fc574]` block_topk |

So +0x00 is arg 5 of the producer and +0x18 is its arg 6, unchanged. Buffer shape check: with the debug byte `[0x18fca48] == 1`, `17bb53a-17bb55a` does `hipMemset([fc750], 0xff, qtiles*r12d*4)` (`movsxd rdx,ebx; imul rdx,r13; shl rdx,2; mov esi,0xff`; `ebx` restored from `[rsp+0xa8]`, `r13` from `[rsp+0xe0]` = r12d saved at `17bb3d2`), i.e. `qtiles` rows of `r12d` u32 words.

Producer kernel `k_sel_tilemask` (`kernels-disasm/codeobj_53000.s:60296`, 0x266500): a first loop zeroes an LDS word array bounded by arg 6 (`s_load_b64 s[12:13],0x28`; `v_cmp_gt_i32 vcc,s12,v0`; `ds_store_b32 v2,v1`); per row `s14` of the tile (`s14 = s4*s2 .. min(s5, s14+s4)`; s2 tile id, s4 arg 2, s5 arg 3) it loads `s15 = counts[row]` (`s_load_b32 s15,[s10+row*4]`), reads `ids[row*arg7 + i]` (`s_mul_i32 s18,s14,s7`, `s7` = arg 7), and does `ds_or_b32` of bit `(id>>2)&31` into LDS word `id>>7` (0x266680-0x2666b0). The mask is thus the union over the tile's rows of the selected ids. The final store to arg 5 was not read (inferred from the `write_only` buffer tag, `tmp/notes_codeobj_53000.txt` near line 9207).

Consumer, FA2 kernel, 0x61838-0x6186c and 0x62010-0x62060:

```
61838: s_cmp_eq_u64 s[12:13], 0                      ; +0x00 null?
6183c: s_mov_b64 s[20:21], 0                         ; null => no mask
61848/61858: s_mul_hi_u32 / s_mul_i32 s22*s2          ; tile * (+0x18)
6185c: s_lshl_b64 s[0:1],s[0:1],2                    ; * 4 bytes
61864: s_add_u32 s20,s12,s0 ; 61868: s_addc_u32 s21,s13,s1   ; tile row pointer
62014-62020: s_add_i32 s1,s7,1 ; s_cmp_eq_u32 s7,s12 ; s_cbranch_scc1   ; KV block loop s7 .. s12
62024: s_and_not1_b32 vcc_lo,exec_lo,s22             ; s22 = (+0x00 != 0) (0x61b60/61b78)
62030: s_lshr_b32 s1,s7,3 ; 62038: s_and_b32 s1,s1,0x1ffffffc   ; word byte offset (blk>>5)*4
62044: global_load_b32 v130, v130, s[20:21]
62050: v_lshrrev_b32 v130,s7,v130 ; 62058: v_and_b32 v130,1,v130   ; bit blk&31
```

Proven: +0x00 is null-checked, indexed by `tile * (+0x18)` u32 words, and tested per KV block `s7` as bit `s7&31` of word `s7>>5`; null makes every block taken. The host word count formula (one bit per 16-token KV block) matches the FA2 K tile of 16 tokens (kernels-attention.md section 1).

Correction to older docs: the `PKi` at kernarg 0x30 that kernels-attention.md calls `sel_idx` is not a selected-block list. The host fills `[0x18fc7c8]` by `17bb315-17bb329` (`mov rdi,[0x18fc7c8]; mov edx,4; mov rsi,r12; mov ecx,1; call hipMemcpy`, kind 1 = host to device) from the host int `*[rsp+0xb0]`, and the kernel reads one int from it (`s_load_b32 s26,[s8:s9]`, 0x61734) and adds it to the row index for the causal bound (`s_add_i32 s0,s26,s0`, 0x61894). Inferred name: pointer to the start position (past length).

## Residuals (none changes a slot meaning)

1. Id unit in `k_sel_tilemask`: it uses `id>>2` while FA2 iterates 16-token blocks, so indexer ids look like 4-token units, or the kernel maps them by a factor of 4. Undecoded VOP3 `.long` words (0x266648, 0x2666a8) hide exact ops. Inferred.
2. Exact role of per-row ids/counts inside the FA2 compare loops (0x62620-0x62a58). Inferred.
3. Sizes of the fc730/fc738/fc750 arena buffers (`01884460.c:1235,1293,1296`).
4. What sets `[rsp+0x1e0]` (dense vs sparse, `17bb336`). Candidate: `018b86d0.c:148` `local_228 = (uint)((DAT_019fca08 ^ 1) & *(long *)(lVar8 + 0x4f8) != 0);`, not matched at instruction level.

## Reproduce

```bash
cd ~/Documents/AI/re-halogen
# 1. FaSel stores, dense and sparse sources
ssh ai-core 'cd ~/re-halogen && r2 -e scr.color=0 -q -c "pd 3000 @ 0x17bb000" binaries/flash_serve' > /tmp/b.txt
grep -nE "0x017bbc99|0x017bbca1|0x017bbcaa|0x017bbcb2|0x017bbbce|0x017bb567|0x017bb56e" /tmp/b.txt
sed -n 125,260p /tmp/b.txt                               # sparse block 17bb2f5-17bb57e + tilemask launch
grep -E "j[a-z]+ 0x17bb[b-c][0-9a-f]{2}$" /tmp/b.txt     # branch-target scan
# 2. tilemask stub identity and tile-row wrapper
ssh ai-core 'cd ~/re-halogen && r2 -e scr.color=0 -q -c "ir~18f6650" binaries/flash_serve'   # -> 0x017bdc30
awk -F'\t' 'tolower($1)=="018bdc30"' analysis/function-map-kernel-stubs.tsv
ssh ai-core 'cd ~/re-halogen && r2 -e scr.color=0 -q -c "pd 4 @ 0x18e97e0" binaries/flash_serve'
# 3. host producers
grep -n "019fc730\|019fc738\|019fc750" work/decomp/flash_serve/01884460.c work/decomp/flash_serve/018b5ef0.c
sed -n 74,85p work/decomp/flash_serve/019e9920.c
# 4. kernel side
sed -n 50925,50990p kernels-disasm/codeobj_1513000.s
sed -n 50925,53000p kernels-disasm/codeobj_1513000.s | sed -n '242,256p;462,482p'
sed -n 60296,60420p kernels-disasm/codeobj_53000.s
```
