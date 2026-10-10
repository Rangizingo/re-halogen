# Sampler seed and counter, third pass (U23)

Scope: close the half of U23 left PARTIAL by sampling-host-r2.md: where the parsed wire seed lands in the per-request sampling params at +0x10, what the counter is, and what `s15` in `k_smpg_accept_final` holds.

Conventions. Ghidra address = ELF vaddr + 0x100000 (`FUN_01860330` = vaddr 0x1760330). Decomp files are `work/decomp/flash_serve/<addr>.c`. Disassembly lines are `objdump -d -M att binaries/flash_serve` (vaddr: instruction). Kernel slices are `work/kernels-spec/*.s:line`. "Inferred" marks anything not read directly. Static only, nothing run.

## Result in one paragraph

The wire seed is a u64 parsed at stack `rsp+0x558`. It sits at offset +0x10 of a 0x40-byte `SampleParams` block (`optional<SampleParams>`, engaged flag at +0x40) embedded at offset 0x1c8 of the request object. That block travels by whole-struct `memcpy(+0x188, 0x81 bytes)` twice (parse stack to heap request, heap request to a stack copy), then the slot constructor `FUN_01860330` copies it field by field into the slot at slot+0x150, so `slot[0x32]` = slot+0x150 and `*(slot[0x32]+0x10)` is the seed unchanged. Every sampling launch packs that seed verbatim into byte 0x10 of a 32-byte `SampleRow` and adds a per-token counter at +0x18 and a stream tag at +0x1c. The counter is not an RNG state that advances: it is `engine+0x2e8 + 1 (+ j)`, the absolute sequence position of the token being drawn. The kernel (and a host mirror) hash `seed ^ counter*K1 ^ tag*golden` with splitmix64. `s15` in the accept kernel is `SampleRow+0xc` = min_p clamped to [0,1], not a seed or counter.

## 1. Wire seed to request object (U23 hop A)

Parse (carried from sampling-host-r2.md, re-read here):
- `173422e: lea 0x548(%rsp),%rsi` temp, `0x54c` top_k, `0x550` top_p, `0x554` min_p, then `1734276: callq _M_extractIy` into `0x90(%rsp)` and `1734295: mov %rax,0x558(%rsp)`. The seed is a whole u64 at rsp+0x558.
- Validation sets the engaged flag: `1734f1b: seta 0x588(%rsp)` after `movss 0x548(%rsp),%xmm0; ucomiss` (temp > 0). So `0x588` = optional engaged flag, block = 0x548..0x587 (0x40 bytes), flag = +0x40.

Which object that block belongs to. The block layout matches the request object field for field: params at request+0x1c8, engaged flag at request+0x208 (checked as `*(char *)(lVar96 + 0x208) == '\x01'` at 0182ea80.c:7690 and 01851dc0.c:188, presence/frequency/repeat at +0x1e0/+0x1e4/+0x1e8 at 01851dc0.c:174). With parse block at 0x548, the parse-side request object base is rsp+0x380 (0x548 - 0x1c8). The seed is request+0x1d8.

The copy (this is the missing hop):
- Parse stack object to heap request. Disassembly at the end of the request build:
  ```
  173cc13: mov    %rax,0x88(%rsp)        ; new Request (heap), saved
  173cc2d: mov    %rax,%rdi
  173cc30: lea    0x380(%rsp),%rsi       ; the parse-side object
  173cc38: call   174a0f0                ; Ghidra 0184a0f0, move_request_state
  ```
  Inside it, the whole sampler block is one memcpy: 0184a0f0.c:276 `memcpy(param_1 + 0x62,param_2 + 0x62,0x81);` (uint indices, so byte offset 0x188, length 0x81 = 0x188..0x208 inclusive, covering params at 0x1c8, seed at 0x1d8 and the flag at 0x208). Disassembly: `174a603: lea 0x188(%rbx),%rdi; 174a60a: lea 0x188(%r14),%rsi; 174a611: mov $0x81,%edx; 174a616: call memcpy`.
- Heap request to the stack copy handed to the slot constructor. `1732743`, `173280f`, `173d3ad: call 1747c40` with `lea 0xa00(%rsp),%rdi` and the heap request (`mov 0x88(%rsp),%rsi`) as source. 01847c40.c:120 `memcpy(param_1 + 0x31,param_2 + 0x31,0x81);` and disassembly `1747f1d: lea 0x188(%r14),%rdi; 1747f24: lea 0x188(%rbx),%rsi; 1747f2b: mov $0x81,%edx; 1747f30: call memcpy`. Destination base rsp+0xa00, so its params are at rsp+0xbc8 and its flag at rsp+0xc08.
- Why earlier searches found no reader of rsp+0x558: the reader is a memcpy through a base register (`lea 0x188(%reg)`), and the Ghidra decompile of 0182ea80 stops at `_Znwm(0x2b0)` (0182ea80.c:2191, "does not return" artifact), so the slot allocation and constructor call never appear in the .c. They are in the disassembly (next section).

## 2. Slot constructor stores the seed at slot+0x160 (hop B)

Two call sites, same shape (stack args in `sub $8,%rsp; push` form):
```
1732bed: mov  $0x2b0,%edi ; call _Znwm            ; new Slot, 0x2b0 bytes
1732c09: cmpb $0x0,0xc08(%rsp)                    ; engaged flag of the stack copy
1732c17: lea  0xbc8(%rsp),%rax
1732c1f: cmovne %rax,%r8                          ; r8 = &sample params, or NULL
1732c4b: call 1760330                             ; FUN_01860330(slot, engine, ..., r8 = param_5, ...)
```
and again at `173d69a..173d6f8` (with `173d6b6: cmpb $0x0,0xc08(%rsp)`, `173d6c4: lea 0xbc8(%rsp),%rax`). `0xbc8` is the stack copy's params (section 1), so param_5 is the seed-bearing block, or NULL when no SAMPLE line (greedy).

`FUN_01860330` (01860330.c):
- 01860330.c:92-101 `if (param_5 == 0) { ... *(undefined4 *)(param_1 + 0x2a) = 0x3f800000; ... puVar18 = 0; }` (no SAMPLE: temp 1.0, rest zero, pointer null).
- 01860330.c:105 `puVar18 = param_1 + 0x2a;` (slot+0x150), then field copy: 01860330.c:106-107 `uVar16 = *param_5; uVar17 = param_5[1];` (temp, top_k, top_p, min_p), 01860330.c:108-109 `uVar1 = *(undefined4 *)(param_5 + 2); uVar2 = *(undefined4 *)((long)param_5 + 0x14);` (the two dwords of the seed at param_5+0x10), and the stores `01860330.c:127 *(undefined4 *)(param_1 + 0x2c) = uVar1;` plus the matching high dword at `(long)param_1 + 0x164`. Slot+0x160 holds the seed, bit for bit.
- 01860330.c:134 `param_1[0x32] = puVar18;` so `slot[0x32]` = slot+0x150 = the pointer every sampler reads, and `*(slot[0x32] + 0x10)` = slot+0x160 = the seed.
- 01860330.c:150 `*(undefined4 *)(param_1 + 0x46) = 0;` slot+0x230 initialised to 0 (this field is the position snapshot, section 3).
- 01860330.c:162-164 `uVar19 = *(undefined1 *)((long)param_5 + 0x3c);` and `*(undefined1 *)(param_1 + 0x49) = uVar19` (the flag byte at params+0x3c, same byte the parser sets at `0x584`).

No transform of the seed anywhere on this path: parse (u64) > memcpy > memcpy > two-dword copy > use.

## 3. Counter source (U23 half two)

The counter is a sequence position, not a stepping RNG state.

- `engine+0x2e8` is read as the current position. 01882b50.c:41-43: `lVar1 = *(long *)(param_1 + 0x108); iVar3 = *(int *)(lVar1 + 0x2e8); *(int *)(param_1 + 0x230) = iVar3;` so `slot+0x230` is a per-round snapshot of `engine+0x2e8` (`slot+0x108` is the engine pointer, `slot[0x21]` in the constructor). What `+0x2e8` counts is inferred from use: it is compared against the context capacity (`iVar3 + 2 <= *(int *)(lVar1 + 0x118)`, 01882b50.c:52) and bumped as tokens are committed, so it is the committed token count.
- Plain sampled step: 01882b50.c:61-63 `FUN_019098d0(**(undefined4 **)(param_1 + 400), *(undefined8 *)(param_1 + 0x108), *(undefined8 *)(*(undefined4 **)(param_1 + 400) + 4), *(int *)(param_1 + 0x230) + 1, 1, param_1 + 0x234, param_1 + 0x238, 0xffffffff);` Ghidra shows this signature with `param_1` as a float (temp, xmm0): temp = `params[0]`, seed = `*(params+0x10)` (the `+ 4` on an `undefined4*`), counter = `slot+0x230 + 1`, tag = literal 1.
- Speculative round: 01851dc0.c:63-64 `iVar14 = *(int *)(lVar16 + 0x2e8); iVar11 = iVar14 + 1;` then draws at counters `iVar11`, `iVar14 + 2`, `(int)uVar19 + iVar11` (01851dc0.c:67, 73, 86). Draft j gets `pos + 1 + j`.
- Batched path: 0182ea80.c:8459 `iVar74 = *(int *)(*(long *)(pauVar8[0x10] + 8) + 0x2e8); uVar35 = iVar74 + 1;` passed as the counter of the same call at 0182ea80.c:8465.
- Verify/accept host draw: 0182ea80.c:7637 `uVar57 = *(uint *)(&ppVar41->__align + 0x46) + 1;` (slot+0x230 + 1), used at 7642 (sample_row), 7673-7681 (host accept uniform) and 7688 (residual).

So two requests with the same seed give the same stream; two positions in one request give different draws; a rejected draft and the resample at the same position use different tags (below), so they do not reuse a draw.

## 4. Where the seed and counter go on the device (SampleRow +0x10, +0x18, +0x1c)

Every sampling launch builds 32-byte rows and `hipMemcpy`s them to `engine+0x9a0`:
- 01909630.c:44-46 (sample_row): `local_40 = *(undefined8 *)(param_3 + 4); local_38 = param_4; local_34 = param_5;` then 01909630.c:47 `hipMemcpy(*(undefined8 *)(param_1 + 0x9a0),&local_50,0x20,1)`. Row: +0 temp, +4 top_k, +8 top_p, +0xc min_p (clamped), +0x10 seed (`param_3 + 4` floats = byte 0x10 of the params), +0x18 counter (`param_4`), +0x1c tag (`param_5`).
- 019098d0.c:60-62 (draft_sample): `local_58 = param_3; local_50 = param_4; local_4c = param_5;` seed, counter, tag (callers pass 1).
- 0190aa50.c:70-72 (residual_sample): `local_58 = *(undefined8 *)(param_3 + 4); local_4c = 3; local_50 = param_4;` seed, tag 3, counter.
- 019085c0.c:101 (`local_128 = *(undefined8 *)(param_3 + 4);`) and 019085c0.c:101 (sample_rows, 0x40-byte stride per row, no caller found in the decomp, inferred unused or inlined).
- 01909e50.c:240 (accept_probs) packs n identical rows (+0x10 seed copied, `local_200 = 0` at 01909e50.c:238 zeroes +0x18/+0x1c). The accept kernel does not draw random numbers.

Kernel side, `k_smpg_pick` (same hash in `k_smpg_rpick` and `k_sample_f32`):
- pick:14-15 `s_load_b32 s33, s[0:1], 0xc` (min_p) and `s_load_b128 s[28:31], s[0:1], 0x10`: s[28:29] seed, s30 counter, s31 tag.
- pick:182-212: `s_mul_i32 s37, s30, 0xd1b54a32` / `s_mul_hi_u32 s39, s30, 0xd192ed03` / `s_mul_i32 s30, s30, 0xd192ed03` (counter * 0xd1b54a32d192ed03), `s_mul_i32 s40, s31, 0x9e3779b9` / `0x7f4a7c15` (tag * 0x9e3779b97f4a7c15), `s_xor_b64` of seed, counter product and tag product, `s_add_u32 s28, s28, 0x7f4a7c15; s_addc_u32 s29, s29, 0x9e3779b9`, then `s_lshr_b64 .., 30`, `* 0xbf58476d1ce4e5b9`, `s_lshr_b64 .., 27`, `* 0x94d049bb133111eb`, `s_lshr_b32 s28, s28, 8` on the high dword (= z >> 40, a 24-bit value).
- rpick:12 `s_load_b128 s[40:43], s[20:21], 0x10`, rpick:230 and 239 same multiplies. k_sample_f32 slice lines 1095 and 1098 same.

Host mirror of the exact same function (used for the accept coin flip, which is drawn on the host, not the GPU):
```
0182ea80.c:7673-7675  uVar91 = ((ulong)uVar57 * -0x2e4ab5cd2e6d12fd ^
                           *(ulong *)((&ppVar41->__align)[0x32] + 0x10) ^ 0x3c6ef372fe94f82a) + 0x9e3779b97f4a7c15;
0182ea80.c:7676  uVar91 = (uVar91 >> 0x1e ^ uVar91) * -0x40a7b892e31b1a47;
0182ea80.c:7682  ... (float)(uint)((uVar91 >> 0x1b ^ uVar91) * -0x6b2fb644ecceee15 >> 0x28) * _DAT_00113b34
```
`-0x2e4ab5cd2e6d12fd` = 0xd1b54a32d192ed03, `-0x40a7b892e31b1a47` = 0xbf58476d1ce4e5b9, `-0x6b2fb644ecceee15` = 0x94d049bb133111eb, `0x3c6ef372fe94f82a` = 2 * 0x9e3779b97f4a7c15 (tag 2), `_DAT_00113b34` = 2^-24 (r2: `pf f @ 0x13b34` = 5.96046448e-08). Identical code at 0182ea80.c:8483-8487 and 01851dc0.c:147-153 (there `iVar14 == 3` forces the counter offset to 0).

Stream tags found (host args or constants):

| tag | where | use |
|---|---|---|
| 0 | 0182ea80.c:7642 sample_row call, 5th arg literal 0 | normal sampled token |
| 1 | 019098d0 callers (01882b50.c:62, 01851dc0.c:67-86, 0182ea80.c:8465) | MTP draft proposal |
| 2 | host-only, constant `0x3c6e...f82a` | accept coin flip: reject when `min(1, p/q) <= u` (0182ea80.c:7680-7681) |
| 3 | 0190aa50.c:71 | residual resample after reject |

The accept compare direction: `if (dVar112 <= u) {` branch calls `FUN_0190aa50` (residual) at 0182ea80.c:7688, else the draft token is kept (7709-7711), so accept iff `u < min(1, p/q)`. `DAT_00113bb8` = 1.0 (r2 `pf F @ 0x13bb8`).

Splitmix64 constants on the host side: 0x9e3779b97f4a7c15 and the multipliers appear in 0182ea80.c:7673-7682, 8483-8487 and 01851dc0.c:147-153 (the mirror above). The other hits for 0x9e3779b97f4a7c15 in 0182ea80.c:3848, 3862 are a separate per-token hash (`0x3c6ef372fe94f82a` stepping, writes `uVar57 + (uVar57 == 0)` into a table, so a nonzero-id generator for the pld/ngram-style tables, not the draw), and `-0x61c8864680b583eb` hits in 01866f80.c:44, 01866350.c:78 and 0187c240.c:459 are cache-key hash mixing. None of these touch the sampler seed (inferred from context, not traced).

## 5. What `s15` holds in `k_smpg_accept_final`

- Kernel: accept_final slice lines 3-15. `s_load_b256 s[4:11], s[0:1], null` loads the first four pointer kernargs; s[8:9] is kernarg 0x10 (`const SampleRow *`, per the mangled name `PKNS_9SampleRowE` third pointer). `s_lshl_b64 s[2:3], s[20:21], 5; s_add_u32 s2, s8, s2` forms `rows + workgroup_id * 32`. Then `s_load_b32 s22, s[2:3], null` (temp) and `s_load_b32 s15, s[2:3], 0xc`: **s15 = SampleRow+0xc of this block's row = min_p**. Use: line 306 `v_cmp_nle_f32_e32 vcc_lo, s15, v0` (encoding `7C38000F`).
- Host: the rows buffer is the device buffer at `engine+0x9a0`, filled by 01909e50.c:240 `hipMemcpy(*(undefined8 *)(param_1 + 0x9a0),&local_218,uVar12 << 5,1)`. Row +0xc is `param_4[3]` (the params min_p) clamped: 01909e50.c:93-95 `if (param_4[3] <= DAT_00113b10) { if (param_4[3] < 0.0) { local_210 = (ulong)(uint)fVar3; ...` (negative gives 0) and 01909e50.c:146 `local_210 = CONCAT44(_UNK_001145b4,fVar3);` for > 1 (`_UNK_001145b4` = 1.0, r2 `pf f @ 0x145b4`). So s15 is the clamped min_p in [0,1]; the earlier "uniform threshold" reading stays withdrawn. It is not the seed and not the counter.
- Launch binding for the mode argument (kernarg 0x34 = HALOGEN_S3_FALSIFY) stays as in sampling-host-r2.md U16; that part is unchanged.

## 6. Verdict

CLOSED for what U23 asked, with these items marked inferred rather than read:

1. Which struct the 0x548 block belongs to (request object at rsp+0x380) is inferred from the offset match (params 0x1c8, flag 0x208, `memcpy +0x188 len 0x81` covering both) and confirmed by the single call `174a0f0` taking `lea 0x380(%rsp)` as source. Not confirmed by reading a decompiled struct definition (none exists, binary is stripped).
2. `engine+0x2e8` as "committed token count" is inferred from use (capacity compare, +1 per drawn token), not from a named field.
3. The `Ghidra` .c for 0182ea80 cannot show the constructor call; the evidence for it is the disassembly at 1732bed..1732c4b and 173d69a..173d6f8.
4. Default seed when SAMPLE is absent: the sampler is never reached (param_5 NULL, greedy), so the zero default never feeds a draw. When SAMPLE is present but has no seed field the parser default is zero (init `movdqu` of a zero xmm0 over 0x54c..0x567, inferred zero register).

What a runtime test would add (not required to close the static question): send the same request with a fixed `seed` twice and compare token streams (expect identical), then change only the seed (expect different), then send two requests differing only in prompt length (the counter is position, so the first sampled token uses counter = prompt_len + 1; two different prompts with the same seed should not share draws). That would confirm item 2.

## Reproduce

```bash
# parse side: seed lands at rsp+0x558, engaged flag at rsp+0x588
ssh ai-core 'cd ~/re-halogen && objdump -d --no-show-raw-insn -M att binaries/flash_serve --start-address=0x1734200 --stop-address=0x17342a0 | grep -n "0x548\|0x558\|_M_extractIy"'
ssh ai-core 'cd ~/re-halogen && objdump -d --no-show-raw-insn -M att binaries/flash_serve > /tmp/full.s'

# hop A: parse object to heap request, then heap request to stack copy
grep -n "173cc30\|173cc38" /tmp/full.s ; awk '/^ 174a60[0-9a-f]:/' /tmp/full.s ; awk '/^ 1747f[12][0-9a-f]:/' /tmp/full.s
sed -n 276p work/decomp/flash_serve/0184a0f0.c ; sed -n 120p work/decomp/flash_serve/01847c40.c

# hop B: slot constructor call sites and field copy
awk '/^ 1732(bed|c09|c17|c1f|c4b):/' /tmp/full.s ; awk '/^ 173d(69a|6b6|6c4|6f8):/' /tmp/full.s
sed -n 92,134p work/decomp/flash_serve/01860330.c

# counter and tags
sed -n 41,43p work/decomp/flash_serve/01882b50.c ; sed -n 61,63p work/decomp/flash_serve/01882b50.c
sed -n 63,64p work/decomp/flash_serve/01851dc0.c ; sed -n 7637p work/decomp/flash_serve/0182ea80.c
sed -n 44,47p work/decomp/flash_serve/01909630.c ; sed -n 70,72p work/decomp/flash_serve/0190aa50.c ; sed -n 60,62p work/decomp/flash_serve/019098d0.c

# kernel hash and s15
sed -n 14,15p work/kernels-spec/_ZN7halogen11k_smpg_pickEPKfPKNS_9SampleRowEPKNS_10SmpScratchES1_S1_PiPfS9_ii.s
sed -n 182,212p work/kernels-spec/_ZN7halogen11k_smpg_pickEPKfPKNS_9SampleRowEPKNS_10SmpScratchES1_S1_PiPfS9_ii.s
sed -n 3,15p work/kernels-spec/_ZN7halogen19k_smpg_accept_finalEPKfPKiPKNS_9SampleRowEPKNS_10SmpScratchEPfSA_iii.s
ssh ai-core 'cd ~/re-halogen && r2 -q -c "pf f @ 0x145b4; pf f @ 0x13b10; pf f @ 0x13b34" binaries/flash_serve'   # 1.0, 1.0, 2^-24
```
