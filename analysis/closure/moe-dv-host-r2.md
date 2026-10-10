# MoE and draft-vocab host closure, second pass: U09, U10, U11, U18

Inputs: first-pass files analysis/closure/moe-host.md and ngram-dv-host.md (read fully), analysis/function-map-kernel-stubs.tsv, work/decomp/flash_serve/*.c, work/kernels-dequant/ slices, binaries/flash_serve.
Note: work/ is not in this worktree; it was read from /Users/peterb/Documents/AI/re-halogen (read-only). This file is written in the worktree.

## Method change that unlocked this pass

The Ghidra decompile drops everything after `__hipPushCallConfiguration` (noreturn assumption), so launch sites were invisible to grep. The ELF is disassemblable directly. `objdump -d binaries/flash_serve` (Apple llvm-objdump 21) works; ELF vaddr = Ghidra address minus 0x100000 (checked: Ghidra 018cb100 stub = vaddr 0x17cb100; the stub's callers show as `callq 0x17cb100`). Stub vaddr to kernel name was joined from function-map-kernel-stubs.tsv. Disassembly was kept in the scratchpad, not in the repo. Quoted lines below are `fs.dis` line numbers / vaddrs from that objdump output (reproduce with `objdump -d --no-show-raw-insn binaries/flash_serve`).

Decode-MoE launcher = Ghidra FUN_018be390 = vaddr 0x17be390 (018be390.c). Its call list was extracted for every stub (see U10).

## U09 HtItem record

Verdict: PARTIAL (size CLOSED at 16 B, confirmed by a second consumer; field meanings improved to a stride-based inference; host does not fill the records)

Answer: no host code writes HtItem fields. The records are built on the GPU by `k_ht_items_group<2|4>` and read by `k_ht_moe_*` and `k_ht_gemv_gu_gs/dn_g`. Launch is in the host at vaddr 0x17c1966 / 0x17c1d5c (see U10). The "72 bytes" is not the record size (first pass already said so; this pass adds the second consumer):

- `k_ht_gemv_gu_gs<3,8,2>` slice lines 18-22: `s_lshl_b64 s[10:11], s[24:25], 4` / `s_add_u32 s8, s8, s10` / `s_load_b128 s[8:11], s[8:9], null`. Index = workgroup id >> 3 (lines 7-10 `s_ashr_i32 ... 3`), stride 16 B.
- Host arena carve for the item array, 018be390.c:315 `(long)local_1db8 * 0x10` (16 B per slot) and asm 0x17c1831-ish `shlq $0x4, %rdi` (vaddr 0x17c1834) in the non-grouped branch.

Field map (inferred from how each dword is consumed; no host writer exists to name them):
- dword 0, offset 0: expert id. gu_gs line 118 `s_mul_i32 s23, s8, 0x12c000` (1,228,800 B per-expert weight stride); same constant in k_ht_moe_gu3 (first pass).
- dword 1, offset 4: slice index inside the expert, 0..4. gu_gs line 120 `s_mul_i32 s27, s9, 0x3c000`; 0x3c000 = 245,760 and 1,228,800 / 245,760 = 5 (arithmetic checked). The item count bound is `*count * 5` (line 13 `s_mul_i32 s3, s3, 5`) and the host grid uses 0x32 = 50 = 10*5. So five row-slices per expert. Inferred.
- dword 2, offset 8: token range begin. k_ht_moe_gu3 `s_sub_i32 s0, s27, s26` (first pass).
- dword 3, offset 12: token range end (gu_gs line 29 `s_add_i32 s20, s11, -1` is a last-index loop bound). Inferred.
- The first pass read dword 1 as "tile or slot index, x160 / x128"; the gu_gs consumer gives the 245,760 stride, which fits a five-way slice better. Both are inference; not proven which kernel's reading is the general one.

What remains: decode which source value lands in each dword inside `k_ht_items_group` stores (slice :500-589), and the count of 20 records per pass; k_ht_moe_dn* item usage not re-read.

## U10 k_ht_tg and the grouped GEMV variants

Verdict: CLOSED for the grouped path selection (with a correction to the first pass); PARTIAL for k_ht_tg (launch of k_ht_tg and k_ht_tg_sum not found)

### Correction: the grouped path is the default, and its switch is HALOGEN_HT_DEC_GROUP

The first pass labelled `FUN_01940b50` as HALOGEN_HT_DN_FOLD. Its body says otherwise:
- 01940b50.c: `getenv("HALOGEN_HT_DEC_GROUP")`, unset gives 2, accepts only 2 or 4 (`(iVar1 - 2U & 0xfffffffd) == 0`), else 0. Stored in DAT_019fe8d0 (vaddr 0x18fe8d0).
- HALOGEN_HT_DN_FOLD is read separately into DAT_019fe8c0 (018be390.c ~line 255) and is not in the path test.

### Selection logic (vaddr 0x17c16e8-0x17c1714, fs.dis ~134290-134400)

Grouped path (items_group, then gu_gs, then dn_g) is taken when ALL of:
- `cmpl $0xc, %r14d; jg` : token count n <= 12
- `testl %ebx,%ebx; jne` : ebx == 0 (the `param_1+0x378` condition of the decompile, meaning not known)
- `cmpl $0x8, %ebp; jne` : split/top width ebp == 8 (first pass called it iVar8 = 8 when topk > 1)
- `movl 0x18fe8d0,%r12d; testl; je` : DEC_GROUP != 0 (default 2, so on by default)
- `cmpl $0x2, 0x154(%rsp); jl` : topk >= 2
- `cmpb $0x0, 0x18fcb08; je` : global flag DAT_019fcb08 set (same flag the decompile tests)
Otherwise falls to the token-indexed `_t`/`_ts` kernels (`imull $0x32, %r14d` at 0x17c1760 then the `ebp` 2/4/8 split ladder). That earlier ladder is the `_t` path, not the grouped one.

Launch sequence, grouped path (vaddrs; kernel names from the stub join):
1. `k_ht_items_group<2>` at 0x17c1966 when r12d == 2 (DEC_GROUP 2), else `<4>` at 0x17c1d5c. Grid 1 x 128 threads (0x17c18f5-0x17c190a: grid 0x100000001, block 0x80). Output arena = three carves (items 16 B per slot, then `shlq $6` 64 B per slot array, then a 0x40 tail), matching 018be390.c:315-321.
2. `k_ht_gemv_gu_gs<bits, 8, S>`: bits 3 or 4 by `cmpl $0x3, 0x12c(%rsp)` (0x17c1d9a), S = r12d (2 -> calls at 0x17c1e0d `<3,8,2>` / 0x17c1e80 `<4,8,2>`, else 0x17c1ee1 `<3,8,4>` / 0x17c1f3b `<4,8,4>`). Grid `imull $0x190 = 400 * n` x 1, block 0x20 = 32 threads (0x17c1d6d-0x17c1d95). 400 = 8 * 50.
3. `k_ht_gemv_dn_g<bits, 8, S, F>`: grid `200*n + 20*rsi` (0x17c1f40 `imull $0xc8`, `addl`), block 0x100 = 256 (0x17c1f57 `leaq 0x100(%rax)`). S = DEC_GROUP (r12d == 2 or 4). F = r13d, computed at 0x17c15ca-0x17c15ec: topk 1 gives 1; topk 2 gives 2; topk >= 3 gives 4 (`cmpl $2; setne; leal 2(,%rax,2)`; `cmpl $1; cmovel`). Calls: F == 4 jumps to the `<..,4>` family (0x17c3659..0x17c3a7f), F == 2 falls to the `<..,2>` calls (0x17c20fa, 0x17c227a, 0x17c23f0, 0x17c2566), otherwise the `<..,0>` family (0x17c3be1..0x17c4007). The F == 0 instances need F == 1, which topk >= 2 never gives, so those four-times-two are unreachable from this launcher (inferred; other launchers not searched).
So: grouped template args are (bits, 8 = split width, S = group size from HALOGEN_HT_DEC_GROUP, F = topk fold). Meaning of F as "fold" is inferred from the HtFold struct name in the `_t` signatures; the F values 1/2/4 equal the topk bucket.

Where `_g` differs from `_t` (signatures, first pass): `_g` consumes `HtItem*` (expert/slice/token-range records) so a workgroup handles one expert slice for a token group; `_t` is indexed by token slot via int arrays. Now backed by the launch order above: items_group builds the records, gu_gs and dn_g consume them.

### k_ht_tg / k_ht_tg_sum

- Launcher is FUN_01928680 (vaddr 0x1828680). The only tg stubs with call sites are `k_ht_tg3<3>` (0x18289c3 -> 0x182bd10) and `k_ht_tg3<4>` (0x1828cac -> 0x182bdf0). `k_ht_tg<3|4>` (0x182bed0, 0x182bfb0) and `k_ht_tg_sum` (0x182c090): `grep -n "182bed0\|182bfb0\|182c090" fs.dis` shows the definitions only, no call or jmp. They are not launched by a direct call in this build (may be launched by an indirect path not searched; label: not found).
- tg3 selection in the launcher: bits at `0x48(obj)` == 3 picks `<3>`, else `<4>` (`cmpl $0x3, %ebx` at 0x1828975 and 0x1828a01).
- Two launch shapes: (A) scratch/M128 configured (`0x18fe6a8` guard set, `0x18fe6a0` nonzero), `(r9 & 1) == 0` and first dim >= `0x18fe6a0`: grid x = ceil(dim/128), y = ceil(r8/256), block 0x200 x 1 (0x18288fb-0x1828970, `sarl $7`, `sarl $8`, `leaq 0x100(%rbp)`); (B) otherwise grid (r9, ebp) block 256 (0x18289cd-0x18289fc). The role of 0x18fe6a0 is inferred as HALOGEN_HT_TG_M128 (FUN_0192c190: unset gives 0x100 = 256, <1 gives 0), because that getter is the only env tied to "M128" and the compare is a lower bound on the first dim; the decompile never names DAT_019fe6a0, so unproven.
- Gate in the caller 01917610.c:61-77 (first pass) unchanged: n < 9, HALOGEN_HT_TG_DEC (default 0 = off), then FUN_01928680.
- "tg" expansion: still open. Role: small-batch split-K trunk GEMM (first pass).

What remains: which dim `edi`/`r8`/`r9` of the tg launcher are K/n/flags (needs the caller's call setup at 0x1917610); where k_ht_tg and k_ht_tg_sum are launched, if anywhere; meaning of `param_1+0x378`.

## U11 per-format bias table and DnfTaps use

Verdict: PARTIAL (no new constants extracted this pass; one new host fact)

- No new bias values this pass (tool budget went to the launch-site problem). Values stay as in moe-host.md: 4-bit symmetric 8 (via 1024 / -1032.0 f16 trick, arithmetic re-checked: 0xe408 = -(1024 + 8) = -1032), q6k 32, q4a asymmetric (zero point from group data, inferred), q8s32/q5a/iq4nl not extracted.
- DnfTaps (first-pass layout stands: three int32 slots at kernarg 0x48, each -1 or `(index+1)*64`). Template argument 0..12 on `k_dn_fused16` is chosen by `switch(in_stack_00000028)` in 019e2a20.c with all grids 48 x 1 and block 256, so it is a host variant value passed in by the caller of 019e2a20; not traced to its origin.
- Unrelated name clash, noted so it is not confused: the 4th template arg of `k_ht_gemv_dn_g` (U10, F in 0/2/4) is a topk fold, not a DnfTaps count.

What remains: per-format constants for q5a, q8s32, iq4nl; q4a zero-point kernarg; origin of the k_dn_fused16 variant value (caller of 019e2a20.c, argument 7 on the stack).

## U18 k_dv_quant8: block layout, argument roles, host consumer

Verdict: PARTIAL (argument roles and launch geometry CLOSED; output plane offsets and the done-line counters' writers STILL OPEN)

The host launch is in code Ghidra did not give a function (vaddr 0x1792372-0x179251f, Ghidra 0189237x-0189251f, after the 212-byte FUN_01891c50 that reads HALOGEN_DRAFT_VOCAB_FILE). It issues `hipLaunchKernel(&PTR_FUN_019f6430, ...)` inline (0x17924fd `leaq 0x163f2c(%rip), %rdi # 0x18f6430`; 0x18f6430 is the k_dv_quant8 registration pointer, 01968680.c:68). The k_dv_scatter pointer 0x18f6438 is launched at 0x17f4acb inside FUN_018f46b0 (the function with the string `lm_head.weight`). No `callq` to the stubs themselves exists, which is why the first pass found no launch.

What the launch shows (fs.dis ~98100-98170):
- The surrounding code does `hipMemcpy` (0x1792377, 0x1792397, kind 1 = host to device) of two std::vector<int> contents into two device buffers (0x40(%rsp), 0x38(%rsp)). Those are arg1 and arg2 of k_dv_quant8 (the two `int*`).
- Grid: `(bytes*5*4 + 0xff) >> 8` blocks (0x17923a4-0x17923d2), where bytes = byte size of the first int vector; so blocks = ceil(80 * n / 256) with n = vector element count. Block 0x100 = 256 threads (0x17923f0). Total threads 80 * n, which equals the kernel's thread bound `(arg5/32) * arg3` with arg5 = 0xa00 = 2560 (`movl $0xa00, 0x34(%rsp)` at 0x1792457): 2560 / 32 = 80. So one thread per 32-value block and n rows.
- Argument roles (k_dv_quant8(bf16*, int*, int*, int, uint8*, int), args stored at 0x120..0x148(%rsp)):
  - arg0 bf16* = source rows, from 0x48(%rsp) (device pointer; the lm_head weight is the likely source given the surrounding strings, inferred)
  - arg1 int* = 0x40(%rsp), uploaded vector
  - arg2 int* = 0x38(%rsp), uploaded vector (which of arg1/arg2 is source map vs destination map is still not resolved)
  - arg3 int = `(end - begin) / 4` of the int vector = row count n (0x179242b `shrq $2, %rsi`, stored 0x78(%rsp))
  - arg4 uint8* = `[rbp + 0x48]` (output buffer pointer inside the draft-vocab object)
  - arg5 int = 2560 (row length in elements)
- Dispatch on width: `cmpl $0x8, 0x58(%rbp); jne 0x1792530` (0x17923e1). Value 8 goes to k_dv_quant8 (this launch), anything else to the other path at 0x1792530, which by the first pass is `k_dv_quant` (4-bit codebook). So `0x58(rbp)` is the draft-vocab bit width (inferred). 0x1792530 was not read.
- This answers the first pass's open point: k_dv_quant8 IS launched in 0.17.3, once, at draft-vocab setup (not per token), from rows of 2560 elements.

Block layout (from the first pass, unchanged): 32 bf16 in, 32 int8 quants plus an f16 scale (`absmax/127`), scale plane at 2 * block index and quant plane at 32 B per block, with an int row map choosing the destination row. Exact plane base inside a destination row still undecoded.

Host consumer of the kept q rows, strings:
- `q_keep: row %d outside 0..%d` at 018f9fb0.c:5-18 and `was not kept` at 019098d0.c:20-25 concern the f32 logit rows kept for draft_sample (first pass). They are not consumers of k_dv_quant8 output. The quantized draft-vocab is a weight-side structure (built once from lm_head rows), whereas q rows are logits; the earlier doc text pairing "q rows" with dv_quant8 is not supported (first pass said the same).
- Done-line fields `dv_tokens, dv_misses, dv_switches`: unchanged from first pass (0184c9a0.c:90-103, stats offsets +0xa0, +0xa8, +0xb0). Writers not found; not searched this pass. Given the structure above, dv_tokens plausibly counts draft tokens scored against the reduced vocab and dv_misses/dv_switches counts of leaving or switching it (pure guess, not evidence).

What remains: plane offsets inside the output row (arg4 layout, read `k_dv_quant8` s2 stride and v[3:4] address and the host buffer size at `[rbp+0x48]` allocation); which of arg1/arg2 is the source map; the other width path at 0x1792530; writers of stats +0xa0/+0xa8/+0xb0 (search for stores to `0xa0(%reg)` near the draft-vocab code).

## Corrections to first-pass files

- analysis/closure/moe-host.md U10: `FUN_01940b50` is HALOGEN_HT_DEC_GROUP (default 2), not HALOGEN_HT_DN_FOLD; the grouped path is on by default for n <= 12, topk >= 2. The "grid grows with split factor" branch at 018be390.c:284-306 is the token-indexed `_t` ladder.
- analysis/closure/moe-host.md U10: grid for gu_gs is 400 * n x 32 threads and for dn_g is (200 * n + 20 * x) x 256, from the disassembly.
- analysis/closure/ngram-dv-host.md U18: "launch of k_dv_quant8 untraced" is now traced (0x17924fd); launches use inline hipLaunchKernel with the registration pointer, so look for `leaq <ptr>` not `callq <stub>` for dv, tg, tg_sum.
