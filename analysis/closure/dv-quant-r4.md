# U18 fourth pass: consumer of the k_dv_quant tables, host index vectors, 4-bit branch

Inputs read: analysis/closure/dv-quant-r3.md, moe-dv-host-r2.md (U18 section), work/decomp/flash_serve/{01891c50,018f46b0,018f51e0,018f5c80,018f6d80,0197a020,01968680}.c, work/kernels-spec/_ZN7halogen10k_dv_quantEPKtPKiS3_iPhli.s (6068 lines; grep plus windows 2735-2775 and 6028-6045), work/kernels-spec/_ZN7halogen12k_dv_scatterEPKfPKiPfi.s (lines 1-40), and `objdump -d` / `objdump -s` of binaries/flash_serve on ai-core (VMA 0x1791a00-0x1792c00, 0x17f46b0, 0x17f4a00-0x17f4b10, 0x17f5c80-0x17f69b0, rodata 0x46b30). Ghidra address = VMA + 0x100000. Static only; nothing was run. Disassembly cites are `VMA: instruction`.

The whole draft-vocab setup is one function at VMA 0x1791c50 (prologue at 0x1791c50, `ret` at 0x17929fd). Ghidra only decompiled its first 212 bytes as FUN_01891c50 (01891c50.c:1-24); the rest, including both launches, has no decompile. Its caller is 01884460.c:606-607 (`if (*(char *)(param_1 + 0x5be) == '\x01') { FUN_01891c50(param_1); }`).

## 1. Consumer of the int8 rows (CLOSED)

The consumer is the draft-vocab lm_head GEMV in FUN_018f46b0 (VMA 0x17f46b0, label string "lm_head.weight"), which launches the existing `k_lq8s32<M,16|32,1|8,true>` kernels through FUN_018f5c80. It is not a q-row consumer.

Chain, each link cited:
1. Setup stores the table pointer in the draft-vocab object: `0x1791dc6: mov %rax,0x48(%rbp)` after `0x1791dad: imul $0xaa0,%rcx,%rdi` (8-bit allocation n * 2720 bytes, matching S = A + A/16 in r3) and `0x1791dc1: call 1792e30` (allocator). The object is published at `0x17928f8: mov %rbp,0x70(%rbx)` (rbx = engine ctx), so `*(ctx+0x70)` is the draft-vocab object.
2. FUN_018f46b0 reads it: 018f46b0.c:52 `lVar1 = *(long *)(param_1 + 0x70);` then 018f46b0.c:69 `if (*(int *)(lVar1 + 0x58) == 8) {`, 018f46b0.c:83 `lStack_110 = *(long *)(lVar1 + 0x48);` (table pointer into the weight descriptor slot +0x18, the q8s32 slot per analysis/closure/dequant-host.md), 018f46b0.c:85 `FUN_018f5c80(&local_128,param_2,*(undefined8 *)(lVar1 + 0x78),*(undefined4 *)(lVar5 + 0x20),1)`.
3. FUN_018f5c80 (VMA 0x17f5c80) args are (desc, x, y, rows, n_tokens). Its launches use `lea 0x101be6(%rip),%rdi # 18f8570` (0x17f6983) and 15 sibling pointers 0x18f84a0..0x18f8590. Those registrations are `k_lq8s32<N,32|16,8|1,true>` (01968680.c:3865-3866, 3995-3996, 4015-4016, e.g. `"_ZN7halogen12_GLOBAL__N_18k_lq8s32ILi1ELi32ELi8ELb1EEEvPKhPKtPvll"`). Kernel args stored at 0x17f6905-0x17f6914: weight = `desc+0x18` (the table), `0x50(%rsp)` = x (hidden), `0x48(%rsp)` = y, `movq $0xa00,0x40(%rsp)` = K 2560, rows. So the table rows are dotted with the hidden state and written as f32 to `*(obj+0x78)`.
4. A second call covers the subset's second range: 018f46b0.c:109-111 `lStack_90 = (long)*(int *)(lVar5 + 0x24) * 0xaa0 + *(long *)(*(long *)(param_1 + 0x70) + 0x48);` then `FUN_018f5c80(&local_a8,param_2, (long)*(int *)(lVar5 + 0x20) * 4 + *(long *)(*(long *)(param_1 + 0x70) + 0x78), *(int *)(lVar5 + 0x28),1)`. Row stride 0xaa0 = 2720 = S from r3.
5. Scatter back to the full vocab: launch at VMA 0x17f4acb `lea 0x101966(%rip),%rdi # 18f6438` (k_dv_scatter registration pointer; 018f46b0.c:164 `__hipPushCallConfiguration(0x1000003ca,1,0x100000100,1,0,0)` = 970 x 256 = 248,320 threads). Args at 0x17f4a3d-0x17f4a63: src = `*(*(ctx+0x70)+0x78)` (the GEMV f32 scratch), map = `*(*(*(ctx+0x70)+0x60) + subset*8)` (per-subset int table), dst = `ctx+0xd80`, n = `movl $0x3ca00,0x4(%rsp)`. Kernel (k_dv_scatter slice): `v_mov_b32_e32 v3, 0xff800000` (-inf), `v_cmpx_lt_i32_e32 -1, v2` (v2 = map[t]), and if map[t] > -1 loads `src[map[t]]`, then `global_store_b32 v[0:1], v3` to `dst[t]`. So dst[t] = map[t] >= 0 ? src[map[t]] : -inf.
6. Fallback (no subset or `+0xb00 != 0`): 018f46b0.c:167 `FUN_018e6f20(param_1 + 0x608,param_2,*(undefined8 *)(param_1 + 0xd80),0x3ca00,1);` full lm_head into the same `ctx+0xd80` row.

Subset selection (018f46b0.c:53-62, mode from FUN_018f51e0 at 018f51e0.c:6-26): `HALOGEN_DRAFT_VOCAB` unset or "1" = mode 1 (use `ctx+0x78`), "0" = off, "cycle" = mode 2 (hash of step, `(param_3 * -0x61c8864f) >> 0xb`), "check" = 4, any other string = mode 3 (named list, resolved by FUN_017f5380 into `ctx+0x88`, 0x17928ff-0x1792904).

Correction to analysis/closure/mtp-head.md:90: it calls `*(model+0x70)+0x78 + off*4` a "token-id slice". It is the f32 GEMV output scratch (allocated 4 * max(range end) bytes at 0x17928e0-0x17928e7, never H2D-copied in setup, passed as `y` above, read as `src` by k_dv_scatter). The token ids per slot are in the ids vector `obj[0]` and in the per-subset map table (section 2).

## 2. Host code that builds the arg1/arg2 vectors (CLOSED)

Both vectors are built per 8192-token shard of the vocab, in the setup function at VMA 0x1791c50.

File parse: `0x1791d4c: call 187a020` (Ghidra FUN_0197a020, the HALOGEN_DRAFT_VOCAB_FILE parser). It checks magic `*plStack_10088 != 0x31434f56444748` (ASCII "HGDVOC1", 0197a020.c:46), version 1 at +8 (0197a020.c:46 `(int)plStack_10088[1] != 1`), vocab = 0x3ca00 at +0xc (0197a020.c:53), N ids from the u32 at +0x10 and M subsets from +0x14 (0197a020.c:51-52), total size `M*0x20 + N*4 + 0x20` (0197a020.c:54), ids copied from +0x20 (`memcpy((void *)*param_2,plStack_10088 + 4,uVar7 * 4)`, 0197a020.c:60), each id checked `< 0x3ca00` (0197a020.c:61). So `obj[0]` = N token ids (vocab ids, not cache slots). Subset records are 0x20 bytes (16-byte name plus three ints: A, second-range start, second-range count, 0197a020.c:78-83), stored in `obj+0x18` as 0x30-byte entries (018f46b0.c uses `+0x20 / +0x24 / +0x28`).

Shard bucketing (0x1791e87-0x1791f6a):
```
1791e8b: mov (%rax,%r15,4),%eax        ; id = ids[i]
1791e8f: lea 0x1fff(%rax),%ecx ... 1791e9a: sar $0xd,%ecx   ; shard = id >> 13
1791e70: mov %r15d,(%rbx)              ; bucket[shard].push_back(i)
```
The bucket array is 0x2e8 bytes = 31 vectors of 24 bytes (0x1791df6 `mov $0x2e8,%edi`); 31 * 8192 = 253,952 >= 248,320.

Per shard, with b = shard index (loop head 0x179204a-0x17920cb): rows in shard = `min(0x794 - 64*b, 64) << 7` (0x1792069-0x17920a9; 0x794 = 1940 units of 128 rows = 248,320). They are dequantized from the HT lm_head planes `ctx+0x638/0x640/0x648` (stride `ctx+0x650`) by `0x17920cb: call 17f5480` (FUN_018f5480, reads `HALOGEN_HT_DEQ_KFAST/PRE/W16`, 018f5480.c:13,26,39; K = `mov $0xa00,%ecx` at 0x17920c2) into a 0x2800000-byte buffer (`0x1791f90: mov $0x2800000,%esi; call hipMalloc`; 0x2800000 = 8192 rows * 2560 * 2 B bf16). That buffer is `0x48(%rsp)` = kernel arg0. That the dequant writes into it is inferred from the stack push at 0x17920c7 (not decoded further).

Vector contents (0x17920d4-0x179239e):
```
17920e4: call 1895040          ; V1 = copy of bucket[b]   (Ghidra 01995040, vector assign; inferred from shape)
17922b1: shl $0xd,%eax         ; eax = b << 13   (eax = r14d from 17922ae)
17922bf: mov 0x0(%rbp),%rcx    ; rcx = ids base (obj[0])
17922e0: movslq (%r12,%rdi,4),%r8      ; r8 = V1[j]
17922e4: mov (%rcx,%r8,4),%r8d         ; ids[V1[j]]
17922e8: sub %eax,%r8d ; 17922eb: mov %r8d,0x0(%r13,%rdi,4)   ; V2[j] = ids[V1[j]] - (b<<13)
1792367: mov 0x40(%rsp),%rdi ; mov %r12,%rsi ; call hipMemcpy   ; dev 0x40(%rsp) <- V1
1792384: mov 0x38(%rsp),%rdi ; mov %r13,%rsi ; call hipMemcpy   ; dev 0x38(%rsp) <- V2
```
Launch args (0x1792412-0x1792494, q8): `0x40(%rsp)` stored to `0xc0(%rsp)` = pointer slot 1 = **arg1**; `0x38(%rsp)` stored to `0xb8(%rsp)` = slot 2 = **arg2**; n = `(end-begin)>>2` of V1 (0x179242b `shr $0x2,%rsi`); `[rbp+0x48]` = arg4; `movl $0xa00,0x34(%rsp)` = A.

Result, closing the r3 host-side question:
- arg1 (destination row in the quantized table) = V1[j] = the **draft-vocab slot** i: the position of the token in the file's ids array (0..N-1). It is not a vocab id and not a cache slot.
- arg2 (source row in the bf16 staging buffer) = V2[j] = `ids[i] - 8192*shard`: the token's **row inside the current 8192-row lm_head shard**.
- One launch per non-empty shard (`0x1792058-0x179205d: cmp ... je 1792020` skips empty buckets), up to 31, each followed by `hipGetLastError` and `hipDeviceSynchronize` (0x1792690-0x179269d). Correction to moe-dv-host-r2.md:96 ("launched once"): it is a loop over shards, one launch each, at draft-vocab setup only. Grid per launch `ceil(80 * n / 256)` blocks of 0x100 threads (0x17923a4-0x17923d2; push config `0x100000100` at 0x17923f0).

The per-subset token map (the `map` in section 1 step 5) is built in the same function (0x1792731-0x17928ec): `memset(tab, 0xff, 0xf2800)` (0xf2800 = 248,320 ints, 0x179277d-0x1792785), `tab[ids[c]] = c` for c < A (0x17927a0-0x17927b2), `tab[ids[off + c]] = A + c` for c < B (0x17927d0-0x17927ea), uploaded by hipMemcpy (0x1792809) and its device pointer appended to the vector at `obj+0x60..0x70` (0x17928c5-0x17928d4). That is the map k_dv_scatter takes. It holds scratch positions (c, A+c), which equal the positions the two GEMV ranges write into `*(obj+0x78)`.

## 3. The 4-bit branch at VMA 0x1792530 (CLOSED, residuals below)

Selector: `0x17923e1: cmpl $0x8,0x58(%rbp); jne 1792530`. The width at `obj+0x58` is set at 0x1791d68-0x1791d8f:
```
1791d68: lea -0x17681dc(%rip),%rdi   # 29b93 "HALOGEN_DRAFT_VOCAB_Q8" ; getenv
1791d7b: cmpb $0x31,(%rax) ; sete %cl                ; first char == '1'
1791d81: lea 0x4(,%rcx,4),%eax                       ; 4 or 8
1791d8a: mov $0x4,%eax                               ; env unset -> 4
1791d8f: mov %eax,0x58(%rbp)
```
(`objdump -s` of 0x29b93 reads "HALOGEN_DRAFT_VOCAB_Q8"; `strings -t x` agrees.) So the **default is 4-bit**; 8-bit only when `HALOGEN_DRAFT_VOCAB_Q8` starts with '1'. This corrects r3/moe-dv-host-r2, which treated 4-bit as the "other" path without naming the selector.

Kernel variant: `0x1792661: lea 0x163dc0(%rip),%rdi # 18f6428` = k_dv_quant (01968680.c:65-66 `"_ZN7halogen10k_dv_quantEPKtPKiS3_iPhli"`). Seven args stored at 0x120-0x150(%rsp): arg0 `0x48(%rsp)` bf16 staging, arg1 `0x40(%rsp)`, arg2 `0x38(%rsp)` (same vectors as q8), arg3 n, arg4 `[rbp+0x48]` table base (0x1792574), arg5 long `[rbp+0x50]` (0x1792578 `mov 0x50(%rbp),%r8`, stored 0xa8(%rsp)), arg6 `movl $0xa00,0xd4(%rsp)` = A (0x17925a8). Kernarg offsets in the slice: A at 0x30 (slice line 4 `s_load_b32 s11, s[2:3], 0x30`), n at 0x18 (line 6), arg4/arg5 pair at 0x20 (line 2749 `s_load_b128 s[4:7], s[2:3], 0x20`).

Layout (allocation at setup, 0x1791d92-0x1791de4):
- `0x1791d98-0x1791da4`: `0x50(%rbp) = n*0x500 + 0x40` (shl 8, times 5, +0x40) = 1280*n + 64. This is the scale-plane offset passed as arg5.
- Allocation (`0x1791db6-0x1791dbe`): 1280n + 64 + 160n = 1440n + 64 bytes.
- `0x1791dca-0x1791de4`: for width 4, `hipMemcpy(table, rodata 0x46b30, 0x40, H2D)`: a 64-byte codebook at table[0..64). The 16 floats at 0x46b30 (from `objdump -s`, unpacked with python struct) are `-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0910, 0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0`: the NF4 levels, confirming the identity ngram-dv-host.md U25 inferred from the 15 midpoints.
- Quant plane: `dest = arg4 + arg1[r] * (A/2) + 16*b`, stores at immediate offsets 64..79 (slice lines 2749-2770: 2749 `s_load_b128 s[4:7], s[2:3], 0x20`, 2752 `global_load_b32 v30, v[1:2], off`, 2755 `s_ashr_i32 s0, s11, 1`, 2758 `v_mad_i64_i32 v[1:2], null, v30, s0, s[4:5]`, `v_lshlrev_b32_e32 v33, 4, v4`; stores from 2770 `global_store_d16_hi_b8 v[1:2], v0, off offset:64` through `offset:79`, slice line 6040). So 1280 bytes per row (2560 nibbles), 16 bytes per 32-value block, starting at table+64. Two 4-bit codes per byte is inferred from 16 bytes per 32 elements; nibble order not decoded.
- Scale plane: slice lines 6028-6041: `s_add_u32 s0, s4, s6` (arg4 + arg5), `s_lshl_b32 s2, s10, 1` (s10 = A>>5 = 80, so 160), `v_mad_i64_i32 v[3:4], null, v30, s2, s[0:1]`, `v_lshlrev_b32_e32 v5, 1, v4`, `global_store_b16 v[3:4], v0, off`. So `scale(r,b) = arg4 + arg5 + arg1[r]*160 + 2*b`, one f16 per block, 160 bytes per row.

| table bytes | content |
|---|---|
| [0, 64) | 16 x f32 NF4 codebook |
| [64, 64 + 1280n) | nibble codes, 1280 B per row, 16 B per 32-block |
| [64 + 1280n, 64 + 1440n) | f16 block scales, 160 B per row |

Consumer for 4-bit: 018f46b0.c:123-132 builds the descriptor `lStack_d0 = 0x40; local_c8 = *(long *)(lVar1 + 0x50); uStack_c0 = 0xa0;` with `local_128 = *(undefined8 *)(lVar1 + 0x48)` and calls FUN_018f6d80, which per analysis/closure/dequant-host-r2.md launches only `k_lq4w` (q4c planar). The second range (018f46b0.c:156-157) advances `lStack_50 = lStack_d0 + start*0x500` and `local_48 = local_c8 + start*0xa0`, exactly the quant-row (1280) and scale-row (160) strides written by the kernel. So the 4-bit table is read as a q4c-planar weight with quant-plane offset 0x40, scale-plane offset `obj+0x50`, scale row stride 0xa0. The scatter tail is shared (k_dv_scatter at VMA 0x17f4acb; the 4-bit GEMV call is at 0x17f4a0a).

## Verdict: CLOSED

All three open items are answered with cites: (1) consumer = draft-vocab lm_head GEMV (`k_lq8s32` for 8-bit, `k_lq4w` for 4-bit) feeding `k_dv_scatter` into the `ctx+0xd80` logit row; (2) arg1 = draft-vocab slot index (position in the file's ids array), arg2 = token row inside the 8192-row lm_head shard, built per shard in the loop at VMA 0x179204a; (3) the 4-bit path is the default (`HALOGEN_DRAFT_VOCAB_Q8` unset), runs k_dv_quant with a 64-byte NF4 codebook header and the table layout above.

## Residuals (bounded, none block the row)

1. Nibble order inside each of the 16 bytes per block is not decoded (pack instructions in the 6068-line slice not read).
2. That `k_lq4w` reads quant offset `+0x48`, scale offset `+0x50` and scale row stride `+0x58` of its descriptor is read from the host descriptor (018f46b0.c:128-157) and the matching kernel writes, not from a `k_lq4w` slice; VMA 0x17f6e57 loads `0x58(%r14)` but the field map was not checked.
3. The HT dequant destination of the call at 0x17920cb (stack argument) is inferred to be the 0x2800000 buffer; the source-offset arithmetic `ctx+0x638 + 0x650*64b*5<<13` was read but not interpreted.
4. Only one caller of FUN_018f46b0 exists: `call 17f46b0` at VMA 0x17f4298, in a function Ghidra has not split out; "this is the MTP head step" in mtp-head.md stays inferred.
5. The dv_switches subset scorer (r3 residual) and how FUN_017f5380 resolves a list name to `ctx+0x88` are unread.

## Reproduce

```
cd /Users/peterb/Documents/AI/re-halogen
sed -n 1,24p work/decomp/flash_serve/01891c50.c
sed -n 40,170p work/decomp/flash_serve/018f46b0.c
sed -n 40,90p work/decomp/flash_serve/0197a020.c
sed -n 2745,2775p work/kernels-spec/_ZN7halogen10k_dv_quantEPKtPKiS3_iPhli.s
sed -n 6028,6045p work/kernels-spec/_ZN7halogen10k_dv_quantEPKtPKiS3_iPhli.s
sed -n 1,40p work/kernels-spec/_ZN7halogen12k_dv_scatterEPKfPKiPfi.s
ssh ai-core 'cd ~/re-halogen && objdump -d --no-show-raw-insn --start-address=0x1791c50 --stop-address=0x1792c00 binaries/flash_serve'
ssh ai-core 'cd ~/re-halogen && objdump -d --no-show-raw-insn --start-address=0x17f4a00 --stop-address=0x17f4b10 binaries/flash_serve'
ssh ai-core 'cd ~/re-halogen && objdump -s --start-address=0x46b30 --stop-address=0x46b70 binaries/flash_serve'   # NF4 codebook
python3 -c "import struct;h='000080bfb13932bf316b06bfa032cabe4ea291be88483dbe6c78babd00000000fffaa23de3ca243edd047c3e3903ad3eb7a4e13eab07103fb213393f0000803f';print(struct.unpack('<16f',bytes.fromhex(h)))"
```
