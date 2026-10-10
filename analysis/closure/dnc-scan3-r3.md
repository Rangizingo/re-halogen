# U12 round 3: scan3 / ut4 / ut5 / att selectors, Tap and NormFold layouts, `dnc` name (flash_serve 0.17.3)

Checked: analysis/closure/indexer-attn-host.md, indexer-attn-host-r2.md, analysis/kernels-indexer.md s5, analysis/closure/moe-host.md (DnfTaps), work/decomp/flash_serve/{019ca280,019d7b70,019dd130,018ab5c0,01895270,019cf410,019ddd90,018b4520,018b4550}.c, work/kernels/*scan3x*.s and *scan3I*.s, `objdump -d`, `objdump -s` of .rodata jump tables, `objdump -R` for handles, binaries/flash_serve strings.

Conventions: VMA = Ghidra address minus 0x100000. Every site below is VMA. Method: full `objdump -d` of binaries/flash_serve, handle to stub map from `objdump -R | grep RELATIVE` joined with analysis/function-map-kernel-stubs.tsv, jump tables read from `.rodata` and resolved by hand (target = table base + s32 entry). "Proven" = read in disassembly or kernel ISA. "Inferred" is written out.

## Verdict

PARTIAL. Closed: ut4 and ut5 flag meanings, scan3 args A/E/F and the full S14 / SCAN / M10P / M10G case-to-stub maps, scan3x args 3 to 9, scan3w last arg, att S12 and ATT_SCALE maps, default att variant, DncTapArgs and DncNormFold layouts (field types and kernel behavior), `dnc` name, the decode-vs-prefill routing (from the decompile). Still open: semantic labels for scan3 B, C, D, G (only measured code effects), scan3x b2 beyond its code effect, contents/roles of the four NormFold pointers, who writes non-default Tap slots, and which stub the n==1 decode step launches. Section 9 lists them.

## 1. ScanArgs: the 0xf0-byte struct that every dnc launcher takes

The scan launcher `019ca280` (VMA 18ca280, rbx = arg) and `019d7b70` (dn_chunk_forward, r12 = arg) read one struct. Copy size: `01895270.c` `memcpy(local_128,param_3,0xf0)` and `17af18c: mov $0xf0,%edx` / `17af191: call memcpy` (gdn_layer_forward builds it at 0x500(%rsp)).

| ScanArgs off | use (host load in launcher) | kernel arg |
|---|---|---|
| +0x00 | `18ca381: mov (%rbx),%rdi` keys (u16 const*) | 0x00 |
| +0x18..+0x40 | `18ca384..` rsi,rdx,rcx,r8,r9 then `movups 0x40(%rbx)` six f32 const* | 0x08..0x30 |
| +0x48 out, +0x50 idx, +0x58 n (long), +0x60 sel (u16*), +0x68 k (int) | `18ca398: movups 0x40` ... `18ca3a4: mov 0x68(%rbx),%r10d` | 0x38..0x58 |
| +0x70, +0x78, +0x80 | `18cc58b: movups 0x70(%rbx),%xmm2`, `mov 0x80(%rbx),%r11`; guard `18cc6bd: cmpq $0x0,0x70(%rbx)` / 0x78 / 0x80 = fp16 mirrors (u16 const*) | extra ptr 0x60,0x68,0x70 |
| +0x88 | qinv (named in pass 1, with +0x90 kinv): not passed to any scan3x site; the default att6 launch loads +0x80, +0x88 and +0x90 (`18dd1c3: mov 0x80(%rax),%r10`, `18dd1ca: mov 0x88(%rax),%r11`, `18dd1d1: mov 0x90(%rax),%rax`) as its last three pointer args | |
| +0x90 | kinv (f32 const*), passed only when scan3x b3 = true: `18cc592: mov 0x90(%rbx),%r14` stored to `0x48(%rsp)`; when b3 = false `18ce31c: movq $0x0,0x48(%rsp)` | extra ptr 0x78 |
| +0x98 | norm_read (HALOGEN_DN_NORM_READ, default 2): `18cc908: mov 0x98(%rbx),%eax`; dn_chunk_forward aborts with "norm_read set without qinv/kinv" if nonzero and +0x88 or +0x90 is null (`019d7b70.c:31-37`) | |
| +0xa0..+0xc7 | DncTapArgs (40 B), copied `18cc5f0: movups 0xa0(%rbx),%xmm3`, `0xb0`, `18cc5e4: mov 0xc0(%rbx),%r15` | kernarg 0x80..0xa7 |
| +0xc8..+0xef | DncNormFold (40 B); `18ce111: lea 0xc8(%rbx),%r15` then `movups (%r15)`, `0x10(%r15)`, `0x20(%r15)` | kernarg 0xa8..0xcf |

Kernarg placement proven by the stack args at the stub call: Tap 0x50..0x77(%rsp), NormFold 0x78..0x9f(%rsp), e.g. `18cc5d4: movups %xmm4,0x88(%rsp)`, `18cc5df: movups %xmm3,0x78(%rsp)`, `18cc603: movups %xmm3,0x50(%rsp)`, `18cc62a: call 18d72d0`.

## 2. DncTapArgs layout (40 B): CLOSED for types, inferred for roles

Host default (proven, two independent sites):
- `01895270.c:` `local_88 = 0xffffffffffffffff; local_80 = 0xffffffff;` at ScanArgs +0xa0 and +0xa8, rest zero.
- `17abbe3: movq $0xffffffffffffffff,0xa00(%rsp)` and `17abbef: movl $0xffffffff,0xa08(%rsp)`.

Layout: `{ i32 slot[3] @ +0x00,+0x04,+0x08 (default -1) ; i32 pad @ +0x0c ; ptr p[3] @ +0x10,+0x18,+0x20 (default 0) }`. Same 3-slot-plus-3-pointer shape as DnfTaps in analysis/closure/moe-host.md.

Kernel proof (k_dnc_scan3x<2,4,true,false,false,1,4,0,false>, work/kernels/_ZN7halogen12_GLOBAL__N_112k_dnc_scan3xILi2ELi4ELb1ELb0ELb0ELi1ELi4ELi0ELb0EEEvPKtPKfS5_S5_S5_S5_S5_PfPKilPtiS3_S3_S3_S5.s):
- `:103 s_load_b256 s[36:43], s[0:1], 0x80` and `:126 s_load_b64 s[0:1], s[0:1], 0xa0` load all 40 B. s36,s37,s38 are the slots, s[40:41], s[42:43], s[0:1] the pointers.
- Slots compared by equality with the block loop index s31 (loop var, `:184 s_cmp_eq_u32 s31, s53`, 64 positions per block via `:187 s_lshl_b32 s40, s31, 6`): `:2658 s_cmp_lg_u32 s31, s36`, `:2793 s_cmp_lg_u32 s31, s37`, `:2795 s_cmp_lg_u32 s31, s38`.
- When equal, the kernel copies an LDS array to the slot's pointer in a loop: `:2868 ds_load_b32 v13, v11` ... `:2874 global_store_b32 v[9:10], v13, off` with address v[9:10] from v119/v121 (= `:156-161 v_cndmask ... s40/s42` pointer select, `:158 v_add_co_ci_u32_e64 v119, null, s41`). So: slot k = block index at which the kernel dumps its LDS state to p[k]. Dump content is inferred (LDS bytes, 0x1f7f bound, stride 8).
- Default -1 can never equal a block index, so no dump happens unless the host sets a slot (proven by compare, not by a non-default host writer, see section 9).

Where taps come from on the host (inferred): `01895270.c` launches the scan three times with n = `*(int*)(ctx+0x4bc)`, `+0x4c0`, `+0x4c4` and output buffers `ctx+0x4e8/0x4f0/0x4f8`; gdn_layer_forward tests the same three counts against Tap slot sign (`17af2dc: cmpl $0x0,0x4bc(%r12)` with `17af2e8: cmpl $0x0,0x5a0(%rsp)` / `0x5a4` / `0x5a8`). Not traced further.

## 3. DncNormFold layout (40 B): CLOSED for types

`{ ptr n0 @ +0x00 ; ptr n1 @ +0x08 ; ptr n2 @ +0x10 ; ptr n3 @ +0x18 ; i32 w @ +0x20 ; pad 4 }`.

Host stores (gdn_layer_forward, VMA):
```
17af250: mov %rdx,0x5c8(%rsp)   ; n0 = *(r15+0x268)
17af25f: mov %rdx,0x5d0(%rsp)   ; n1 = DAT 18fc638
17af276: mov %rcx,0x5d8(%rsp)   ; n2 = (mode==2) ? 18fc958 (hipMalloc buf) : 18fc660
17af285: mov %rcx,0x5e0(%rsp)   ; n3 = DAT 18fc7b8
17af29d: mov %ecx,0x5e8(%rsp)   ; w = (mode==3 ? -1 : 0) | 0x80
```
mode = DAT 18fc948 = HALOGEN_DN_NORM_FOLD (getter 018b4550, default 0). 18fc660, 18fc638, 18fc7b8 are arena pointers set in `01884460.c:1259,1276,1280`. Fused-path variant: `17af358..17af389` n3 = 0, w = 0x80.

Kernel proof: scan3x<...,i3=1> (`work/kernels/..ILi1ELi0ELi1ELb0...s`, copy in scratchpad k101): `:12 s_load_b256 s[24:31], s[0:1], 0xa8` and `:13 s_load_b32 s8, s[0:1], 0xc8` read all five fields; `:205 s_cvt_f32_i32 s62, s8` converts w to f32 (so w = 128.0 normally, -1.0 in mode 3). i3=2 loads only n3: `s_load_b64 ... 0xc0` (measured over all 21 scan3x slices: 0xa8 b256 + 0xc8 b32 only in i3=1, 0xc0 b64 only in i3=2, nothing in i3=0).

Host zero-fills NormFold in every scan3x site except i3=1 (copy from `(%r15)`, `18ce7e9..18ce806`) and i3=2 (see section 4).

## 4. scan3 / scan3x / scan3w / scan2 case maps (all proven from jump tables)

Selectors, in `019ca280` order:

1. `HALOGEN_DN_SCAN` (019cef20, default 0). `18ca2aa: cmp $0x3,%rax; ja`, table 0x48dac. 0: continue to step 2. 1: scan2<32,2,2> (handle site `18ca977`). 2: scan2<16,2,2> (`18ca5a3`). 3: scan2<32,4,2> (`18ca78d`). 4 and up: scan2<32,2,4> (`18cab61`). All grid `0x800000030`, block 128. (Table 0x48dac entries 0x018815ed.. resolve to blocks 18ca2c4 / 18ca799 / 18ca3c5 / 18ca5af; nearest handle site after each.)
2. `HALOGEN_DN_M10P` (019cf460; value kept if <4 or 8..20 else 0). `18ca315: cmp $0x12,%rax`, table 0x48dbc. 0 goes to step 4. 1,2,3,8,10,12,14,16,18 each select the grid family with E = that value (resolved targets 18cacdb, 18cab8b, 18caf83, 18cb02d, 18cac31, 18cad85, 18cae2f, 18cb181, 18cb0d7); all other in-range entries and >18 go to 18caed9 = E 20.
3. `HALOGEN_DN_M10G` (019cf4a0): inside each E block: `18cab94: tzcnt %eax,%eax`, `cmp $0x3`, table (E=2 block 0x48e78: 0x18cabbd, 0x18cb8bb, 0x18cb6db, 0x18cb7cb). These land on the calls for F = 1, 2, 4, 8 in order. Non power of two: `18cab9a: jbe 18cb22b` = F 15. So F = M10G when it is 1,2,4,8, else 15. Table read for the E=2 block only; other E blocks have the same shape (inferred).
4. `HALOGEN_DN_S14` (019c9fe0, default 34, clamp). `18ca335: dec %eax; cmp $0x23,%eax`, table 0x48e98. Full map (S14 : stub, grid.y from `movabs`):

| S14 | stub | grid |
|---|---|---|
| 1 | scan3<8,16,0,F,0,15,0> | y 8 |
| 2 | scan3<16,16,0,F> | y 16 |
| 3 | scan3<32,16,0,F> | y 32 |
| 4 | scan3<16,8,0,F> | y 16 |
| 5 | scan3<32,32,0,F> | y 32 |
| 6 | scan3<4,16,0,F> | y 4 |
| 7 | scan3<4,8,0,F> | y 4 |
| 8 | scan3<4,4,0,F> | y 4 |
| 9 | scan3<2,8,0,F> | y 2 |
| 10, 11, 12, 13 | scan4<8,16,{1,2,4,8}> | y 8 |
| 14 | scan5<8,16,1,T> | y 8 |
| 15, 16 | scan5<8,8,1,{T,F}> | y 8 |
| 17, 18, 19 | scan3<8,{8,4,2},0,F> | y 8 |
| 20..23 | scan3<8,8,{4,8,16,32},F> | y 8 |
| 24, 25, 26 | scan3<8,8,{4,8,16},T> | y 8 |
| 27 | scan3<8,8,8,T,0,15,1> (G=1) | y 8 |
| 28 | scan3t<8,8,8,T> | y 8 |
| 29, 30 | scan3w<8,4,F,F,0>, scan3w<4,4,T,F,0> | y 8, y 4 |
| 31 | scan3w<2,4,T,F,P>, P = HALOGEN_DN_PROBE (section 5) | y 2 |
| 32 | scan3w<2,4,T,T,0> | y 2 |
| 33 | scan3x<2,4,F,F,F,0,0,0,F> (the "running 33" fallback, jumps to common tail 18cda25, call `18cda73`) | y 2 |
| 34 (default) | scan3x family, A=2 | y 2 |
| 35 | scan3x family, b2 = true, A=2 | y 2 |
| 36 | scan3x family, A=4 | y 4 |

Case 31 target address 0x18cd3df verified by `18cd3df: mov 0x3237a(%rip),%r15d # 18ff760` and `18cd412: cmp $0x7,%r15d`. Case 33 verified at `18cd478` (y 2, zeroed mirror and NormFold args at `18cd4d0..18cd506`, jump to `18cda25`).

### scan3 template args (A..G)
- **A = grid y. CLOSED.** Every base call's `movabs` high word equals A: <8,16>: `18ca350 movabs $0x800000030`; <32,32>: `18ccabb movabs $0x2000000030`; <4,16>: `18cd0ac movabs $0x400000030`; <2,8>: `18cd36a movabs $0x200000030`; <16,8>: `18ccdee movabs $0x1000000030`. Same for scan3x (A=2: `18cc6a9 movabs $0x200000030`; A=4: `18cda8d movabs $0x400000030,%r12`). x is always 0x30 = 48, block 0x80 = 128 (`0x100000080`).
- **E = HALOGEN_DN_M10P. CLOSED** (table above). **F = HALOGEN_DN_M10G if in {1,2,4,8} else 15. CLOSED.**
- **B, C, D, G: host cannot name them (no env selects them, they are fixed per S14 case).** Measured code effects over the 70 slices in work/kernels (instruction count from the `.s`, histogram by mnemonic):
  - C: C=4, 8, 16 give identical instruction count (1761 with A=B=8, D=F) and differ only in LDS offsets and scheduling (first diff at index 622 is a register rename, ds_load offset0/offset1 constants differ); C=32 is smaller (1517, ds_load 244 vs 340). So C is a layout/stride constant, not a structural switch (inferred label: LDS row stride).
  - D: true adds exactly one `global_load_b32` and about 61 instructions of scheduling noise (`s_delay_alu` +58, `v_fmac_f32` +9) at fixed A,B,C: 1761 to 1822. Semantic open.
  - G: G=1 adds dynamic register indexing: `v_movreld_b32` +12 and `v_movrels_b32` +10, `v_cmp_lt_u32` +19, `s_mov_b32` +54 (2144 vs 1822 instructions). G selects a runtime-indexed register array instead of an unrolled one (inferred).
  - B: non-monotonic code size (B=2: 704, 4: 637, 8: 2039..2928, 16: 1698..1845, 32: 1111; barriers 11, 9, 37..51, 27, 15). B sets tile geometry; no host meaning.

### scan3x args <A, B, b1, b2, b3, i1, i2, i3, b4>

| arg | meaning | proof |
|---|---|---|
| A (2 or 4) | grid y | above |
| B (4) | constant | all 21 slices use 4 |
| b1 | fp16 mirror pointers used | b1=false only at fallback 33 (`18cda73`); its kernel loads nothing at 0x60..0x7f (`s_load_b512 @0x8, b128 @0x48, b32 @0x58, b256 @0x80, b64 @0xa0` only). b1=true kernels load `b128 @0x60` etc. |
| b2 | S14 = 35 variant (A=2) | b2=true only reached from S14=35 (`18cc6a9..18cc6e7`, `18cdb9a`, `18cdfff`). Kernel effect: 64 unrolled bounds guards `s_cmp_gt_u32 s50, 63 ... 0` each followed by `s_cselect_b32 vcc_lo` and a `global_load_b32` (diff +64 loads, +62 cmps vs b2=false; kb2.s `:122-:131`), where s50 = k (`s_load_b32 s50, 0x58`). Inferred: tail-guarded last block of 64. |
| b3 | kinv pointer present (NORM_READ == 2) | b3=true sites pass ScanArgs+0x90 (`18cc592: mov 0x90(%rbx),%r14`), b3=false pass 0 (`18ce31c: movq $0x0,0x48(%rsp)`); kernel b3=true loads the 4th extra pointer in one `s_load_b256 @0x60`, false loads `b128 @0x60` + `b64 @0x70`. Branch: `18cc6e7: cmpl $0x2,0x98(%rbx); jne 18cdb9a` (b3=true only when norm_read == 2) |
| i1 | min(HALOGEN_DN_SCAN_UB, 2) | `18cdb9a: call 18d5ba0; cmp $0x2,%eax; jl` (>=2 gives i1=2), `18cdefc/18ce021: call 18d5ba0; test %ebp,%ebp; je` (0 gives i1=0, else 1). 18d5ba0 memoizes `019ddd90` = getenv HALOGEN_DN_SCAN_UB, default 1. |
| i2 | HALOGEN_DN_PROBE when 4, 5, 6 | `18cc8f9: mov 0x32e61(%rip),%eax # 18ff760; add $0xfffffffc,%eax; cmp $0x3,%eax; setb`; with norm_read == 0 and SCAN_UB == 1; `18cc95b: cmp $0x5,%ebp; je` PROBE 5; `18cc964: cmp $0x4,%ebp` PROBE 4; else PROBE 6. 18ff760 is the cache of `019cf410` = HALOGEN_DN_PROBE. |
| i3 | NormFold mode: 0 none, 1 full copy (all five fields), 2 n3 only (w < 0) | i3=1: `18ce761: mov 0xd8(%rbx),%rax; test; ... call 18d5ba0; cmp $1` (needs NormFold.n2 != 0 and SCAN_UB == 1) then site `18ce851`. i3=2: additionally `18ce128: cmpl $0x0,0xe8(%rbx); jns` skipped, i.e. w < 0 (NORM_FOLD mode 3), site `18ce215`. |
| b4 | HALOGEN_DN_LDS_BAR | `18ce868: call 18d6370` (`019d6370` getter, getenv HALOGEN_DN_LDS_BAR) `test %al,%al`, requires SCAN_UB != 0, site `18ce970`. |

PROBE: `019cf410.c` prints "HALOGEN_DN_PROBE=%d: GDN output is WRONG (timing decomposition only)" for any nonzero value (format string in .rodata). So scan3x i2 in {4,5,6} are timing-probe kernels that skip work: 3295 / 3503 / 3151 instructions vs 4355 for i2=0 (mnemonic count of the slices; `ds_load_b128` -88, `s_cselect_b32` -128).

## 5. scan3w last argument = HALOGEN_DN_PROBE

kernels-indexer.md s2 calls `scan3w<2,4,true,false,P>` P 0..8 "9 slice positions". Wrong: S14=31 selects P by `18cd40f: dec %r15d; cmp $0x7,%r15d; ja` on the PROBE cache 18ff760, table 0x48f28. Resolved: PROBE 1..8 map to P = 1..8 (targets 18cd42c, 18cddcc, 18cdce8, 18cdd34, 18cdc9c, 18cde18, 18cde64, 18cdd80), PROBE 0 or >8 gives P = 0 (`18cdeb0`). These are the same timing probes (wrong output by design).

## 6. ut4 and ut5 flags: CLOSED

All in `019d7b70` (VMA 18d7b70). Globals (decompile `019d7b70.c:40-120`): 19ff794 = HALOGEN_DN_S3A (default 4), 19ff7b0 = HALOGEN_DN_XH (1), 19ff838 = HALOGEN_DN_S22 (1), 19ff890 = HALOGEN_DN_UT5 (on unless "0"), 19ff8a0 = HALOGEN_DN_ATT_UT (1), 19ff8b0 = HALOGEN_DN_UT_RS (3), 19ff8c0 = HALOGEN_DN_UT_BURST (1), 19ff8d0 = HALOGEN_DN_UT_KL (0), 19ff8e4 = HALOGEN_DN_UT_MP (0). Grid: ut5 `18d8bb6: shl $0x20,%rdi; or $0x30,%rdi` = (48, nc) block 256; ut4 `movabs $0x3000000000,%rbx` = (nc, 48) block 256.

S3A map (`18d8ac8: mov 0x26cc6(%rip),%ebp # 18ff794`, `cmp $0x3,%ebp; jl`):

| S3A | kernel pair |
|---|---|
| <3: 0 (or other) | k_dnc_ut (`18d925f`) then att launcher `18d9287: call 18dd130` |
| 1 | k_dnc_ut2 (`18d9079`) then k_dnc_att2 (`18d9166`) |
| 2 | k_dnc_ut3 (`18d8e71`) then att launcher |
| 3, 6+ | ut4<4,1,F,F,F> (`18d98e4`, via `18d94f3: cmp $0x5,%ebp; jne 18d98b5`) |
| 5 | ut4<4,4,F,F,F> (`18d952b`) |
| 4 | see below |

S3A = 4 (`18d8ad7: cmp $0x4,%ebp; sete %cl`, `18d8add: cmpl $0x0,0x26d54(%rip) # 18ff838` = S22, `18d8af1: cmpl $0x0,18ff7b0` = XH):
- S22 == 0: ut4<8,1,F,F,F> (`18d9395`).
- S22 != 0, XH == 0: ut4<8,1,T,F,F> (`18d8f99`).
- S22 != 0, XH != 0, UT5 == 0 (`18d8b0d: cmpb $0x1,0x26d7c(%rip) # 18ff890; jne 18d93a1`): ut4<8,1,T,T,MP> with MP = HALOGEN_DN_UT_MP == 1 (`18d93b0: movzbl 18ff8e4,%ebx; cmp $0x1,%bl`): `18d94c0` <8,1,T,T,T> if MP, `18d9609` <8,1,T,T,F> otherwise.
- S22 != 0, XH != 0, UT5 != 0: ut5.

**ut4<N, M, b1, b2, b3>**: N = 8 iff S3A == 4, else 4; M = 4 iff S3A == 5, else 1; b1 = S22 != 0 (only with S3A == 4); b2 = XH != 0 and UT5 off; b3 = UT_MP. (Names of N and M inferred as row-block and column-block factors; their values follow S3A.)

**ut5<T, b_norm, rs, b3, b4, b5>** (`18d8b29..18d8bda`):
- first bool: always true; no site passes false (all 24 handles checked: `18d8d58..18db7e1`).
- second bool = `ScanArgs+0x98 != 0` (norm_read): `18d8b9f: mov 0x98(%r12),%ebx` ... `18d8bda: test %ebx,%ebx; je 18d962e`. `18d962e` leads to the 12 sites with second bool false; fallthrough to the true ones.
- int rs = HALOGEN_DN_UT_RS: `18d8bea: cmp $0x1,%ebp; je`, `cmp $0x3`, `cmp $0x2`, else rs = 0. Values 0..3 as in the registrations.
- b3 = `ATT_UT != 0 && ScanArgs+0x70 != 0 && ScanArgs+0x80 != 0` (r13b: `18d8b29 cmpl $0,18ff8a0; setne %r13b`, `18d8b34 cmpq $0,0x80(%r12); setne %r14b`, `18d8b41 cmpq $0,0x70(%r12); setne %r15b`, `and`s at `18d8b6f` and `18d8b88`).
- b4 = `UT_BURST != 0` (r14d), b5 = `UT_BURST != 0 && UT_KL != 0` (r15b, `18d8b8b..18d8b9c`). b4 and b5 apply only for rs = 3; rs = 1 uses b3 and b4 only (b5 false: decision tree `18d9a68: test %r13b; je 18da338; test %r14d; je 18daa37` ends in the four rs=1 sites); rs = 2 and rs = 0 have no flags (`18d8d58`, `18d9d2c`).
- Consequence for the registration count: the 24 distinct ut5 stubs are exactly the reachable set {rs, b3, b4, b5, norm}; the "60" in kernels-indexer.md is not reproduced here.

## 7. Attention (k_dnc_att*) launcher `019dd130` (VMA 18dd130): CLOSED for selectors

Selector HALOGEN_DN_S12 (019dcbe0, default 4). `18dd2d1: dec %eax; cmp $0x9,%eax; ja 18dd837`, table 0x48f58:

| S12 | stub |
|---|---|
| 1 | att6<64,F,F,F> |
| 2 | att6<64,T,F,F> |
| 3 | att6<32,F,F,F> |
| 4 | att6<32,T,F,F> |
| 5 | att6<16,F,F,F> |
| 6, 7, 8 | att7<32,2>, att7<32,4>, att7<32,8> |
| 9 | att8 (call `18dd8fc`) |
| 10 | att9<ATT_SCALE> |
| 0 or >10 | att v1 (call `18dd897`) |

att9: `18dd51a: mov 18ff8f0,%r14d; add $0x8,%r14d; rol $0x1e,%r14d; cmp $0x6; ja 18dd95f`, table 0x48f80: (HALOGEN_DN_ATT_SCALE + 8) >> 2 indexes {-8, -4, 0, 4, 8, 12, 16}; a value that is not a multiple of 4 gives index > 6 and falls to att9<4> (`18dd95f`). 019dfc10 clamps to -12..16, default 4.

**Default path** (S12 == 4 and XH != 0, `18dd13e: cmpl $0,...; setne %cl; ... cmp $0x4,%eax; sete %dl; and`): grid `18dd175: shl $0x21,%rdi; or $0x30` = (48, 2*nc), block 256 (`18dd17d: movabs $0x100000100`). Variant by `18dd166: mov 0x98(%rax),%ebp; test %ebp,%ebp; je 18dd35e`: norm_read != 0 gives handle `18dd2c5` = **att6<32,true,true,true>**; norm_read == 0 gives `18dd47d` = **att6<32,true,true,false>**. Default norm_read is 2, so the default att stub is att6<32,T,T,T>. Third bool of att6 = norm_read != 0 on this path; the first two are fixed true here. (Mapping handle to stub via `objdump -R`: 0x18fa... slots; the stub names come from analysis/function-map-kernel-stubs.tsv.)

The att6 width arg 64/32/16 matches grid x: S12=1,2: nc; 3,4: 2*nc; 5: 4*nc (pass 1). So width = 64 / (x multiplier) tokens per block.

## 8. Decode versus prefill; name `dnc`

- **dnc launchers have no n-based branch of their own** (nc = ceil(n/64) from `019d7b70.c:19-23`). The split is upstream. `018ab5c0.c` (gdn_layer_forward, only caller `018f2060.c:140`; called for every token count): for n == 1 with `*(int*)(ctx+0x2e8) > 0` and `ctx[0x408] == 0` it takes a single-token step launch instead (`018ab5c0.c:187 if (param_3 != 1 || iVar4 < 1) goto LAB_018ab65a;` and `:523-526 if ((param_3 == 1) && (0 < iVar4) && (param_1[0x408] == 0)) __hipPushCallConfiguration(0x100000029,1,0x100000100,...)` = grid (41,1) block 256). For 2 <= n <= HALOGEN_GDN_STEP_N (default 0, `018b0e10`) or `ctx[1] != 0` it goes to a multi-row step path (`:184`, `:529`). Otherwise (n above those bounds, or the ctx flags off) the chunk path runs: `17af4b0: call 18d7b70` (dn_chunk_forward) unless the fused flag takes `18e4bd0`. So the dnc att/ut/scan kernels are the prefill (chunked) path; decode (n = 1) is served by the step kernels when ctx+0x2e8 > 0. Source is the decompile plus one disassembly caller site; the step kernel's stub identity was not resolved (section 9).
- **`dnc` = dn_chunk.** The binary embeds the source path `/src/halogen/src/dn_chunk.hip` (strings offset 0x1f8cf; used as the error-file name at `019d7b70.c:201`) and the function string `dn_chunk_forward: norm_read set without qinv/kinv` (0x180a3). All k_dnc_* are registered by one block (`019e0890.c`, 350 `k_dnc_` mentions) and launched only from `019ca280` (dn_scan_launcher), `019d7b70` (dn_chunk_forward) and `019dd130`, all steered by HALOGEN_DN_*. The kernels walk 64-token blocks (`k34.s:187 s_lshl_b32 s40, s31, 6`, `:2309 s_add_i32 s40, s40, 64`). "DN" is GDN/DeltaNet (strings `attn_gdn`, "GDN output is WRONG"). So dnc = DeltaNet chunk; the name is a source-file prefix, strong inference, no string spells it out.

## 9. Not closed

1. scan3 B, C, D, G: no host selector distinguishes them; only code effects measured (section 4). Labels "LDS stride" (C) and "dynamic register index" (G) are inferred.
2. scan3x b2: code effect shown (64 bounded loads against k), the reason for S14=35 not found.
3. NormFold n0..n3: host sources are arena pointers (`01884460.c:1259,1276,1280`) and `ctx+0x268`; what each buffer holds is not traced. Only kernel behavior of w (i32 to f32, 128.0 or -1.0) is proven.
4. Tap: a non-default slot writer was not found in gdn_layer_forward (stores to the Tap region only at the two -1 defaults). Whether any 0.17.3 path sets a slot is unverified; test needed: launch-time capture of kernarg 0x80 for scan3x.
5. The n == 1 decode step launch (grid (41,1)): stub not identified (no `movabs $0x100000029` literal in the disassembly, the decompile builds it arithmetically).
6. M10P/M10G table check was done on the E=2 block only (other blocks inferred identical).

## Reproduce

```bash
# full disassembly (ai-core), handles and tables
ssh ai-core 'cd ~/re-halogen && objdump -d --no-show-raw-insn binaries/flash_serve' > full.dis
ssh ai-core 'cd ~/re-halogen && objdump -R binaries/flash_serve | grep RELATIVE' > rela.txt
# handle slot -> stub: join rela addends (stub VMA) with analysis/function-map-kernel-stubs.tsv (Ghidra - 0x100000)
# jump tables: target = table_base + s32 entry
ssh ai-core 'cd ~/re-halogen && objdump -s --start-address=0x48e98 --stop-address=0x48f28 binaries/flash_serve'   # S14
ssh ai-core 'cd ~/re-halogen && objdump -s --start-address=0x48dbc --stop-address=0x48e08 binaries/flash_serve'   # M10P
ssh ai-core 'cd ~/re-halogen && objdump -s --start-address=0x48f28 --stop-address=0x48f9c binaries/flash_serve'   # PROBE, S12, ATT_SCALE
# launcher windows
objdump -d --start-address=0x18ca280 --stop-address=0x18cf460 binaries/flash_serve   # scan launcher
objdump -d --start-address=0x18d7b70 --stop-address=0x18dc670 binaries/flash_serve   # dn_chunk_forward
objdump -d --start-address=0x18dd130 --stop-address=0x18ddbf0 binaries/flash_serve   # att launcher
objdump -d --start-address=0x17ab5c0 --stop-address=0x17b0c90 binaries/flash_serve   # gdn_layer_forward (ScanArgs build, 17af250 NormFold)
strings -a -t x binaries/flash_serve | grep -E 'dn_chunk.hip|norm_read set'
# kernel proofs
grep -nE 's_load_b|s_cmp_lg_u32 s31' work/kernels/*k_dnc_scan3xILi2ELi4ELb1ELb0ELb0ELi1ELi4ELi0ELb0*.s
for f in work/kernels/*scan3x*; do grep -oE 's_load_b[0-9]+ [^,]+, s\[0:1\], 0x[0-9a-f]+' "$f"; done
```
