# U12 and U14, second pass: host launch sites via x86 disassembly (flash_serve 0.17.3)

Checked: analysis/closure/indexer-attn-host.md (pass 1), analysis/kernels-attention.md, analysis/kernels-indexer.md (correction note), analysis/function-map-kernel-stubs.tsv, work/decomp/flash_serve/{019e9920,019eb380,019dd130,019f29e0,019f1b30,019f25b0,018b86d0}.c, `objdump -d` of binaries/flash_serve.

## Method note (why grep of stub addresses in the decompile finds nothing)

Pass 1 blamed the noreturn truncation for the lost stub calls. There is a second reason, and it gives a way around both:

- Ghidra address = ELF VMA + 0x100000 (`.text` VMA 0x172b880..0x18f4280; Ghidra 019e9920 is VMA 0x18e9920). `objdump -d --start-address=<VMA>` works on the stripped ELF; the full dump is 398,730 lines.
- Most launch sites do NOT call the stub. The compiler inlined the stub body: `leaq <handle>(%rip), %rdi` then `hipLaunchKernel@plt`. The handle is an 8-byte slot in `.data.rel.ro` (0x18fa000..0x18fa2e8 for the attention family, 0x18f6xxx for others) whose `R_X86_64_RELATIVE` addend is the stub VMA. `objdump -R binaries/flash_serve | grep RELATIVE` gives handle to stub.
- Other sites (the dn_chunk att launcher, `k_attn`) call or jump to the stub address directly (`callq 0x18e0120`), so a grep of the stub VMA in the disassembly works there.
- Example proof of the handle route: `18e9c27: leaq 0x1060a(%rip), %rdi # 0x18fa238` and rela `0x18fa238 -> 0x18f1b60` = Ghidra 019f1b60 = `k_attn_fa2<1,1,true,5,1,true>` (stubs tsv).
- All instruction addresses below are VMA (Ghidra minus 0x100000) unless written as 019xxxxx.

---

## U14

### Verdict: PARTIAL (fa2 args and legacy dispatch closed; FaSel byte layout proven, slot meanings open)

### 1. fa2 template args 4-6: CLOSED for OPT and Y, strong inference for X

Signature order (stubs tsv, 019f2250): `k_attn_fa2<A, B, bool, OPT, X, Y>`.

- **OPT (arg 4) = HALOGEN_FA_OPT & 7. CLOSED, now with case-to-stub mapping.** `019e9920.c:99-217` `switch(DAT_019ffa08 & 7)`; each case calls a one-time-init helper placed 0x30 bytes before the stub it launches: case 0 `FUN_019f25b0` -> stub 019f25e0 (`<1,1,true,0,0,false>`); case 1 `FUN_019f1d60` -> 019f1d90 (OPT 1); case 5 `FUN_019f2220` -> 019f2250 (`<1,1,true,5,0,false>`); case 7 `FUN_019f2480` -> 019f24b0 (OPT 7). Dense launcher `019eb380.c:17-96` does the same: `FUN_019f3460` (case 0) -> 019f3490 `<2,1,true,0,0,false>`, `FUN_019f2c10` (case 1) -> 019f2c40, `FUN_019f30d0` (case 5) -> 019f3100. (The helper/stub adjacency is inferred from addresses; the 8-case count and OPT range match exactly.) Each helper only does `getenv("HALOGEN_FA_LPT")` (see `019f25b0.c`): it is a per-instantiation copy of the same static-local reader.
- **X (arg 5): strong inference, control flow proven.** Special path condition in disassembly: `18e9a7e: cmpl $0x5,%eax` then `18e9a87: testb $0x1, 0x18ffa18(%rip)` (DAT_019ffa18 = FUN_019eb350 = HALOGEN_FA_S21, default 1), also requiring VB and RS nonzero (`18e9a70`, `18e9a76`). Only the OPT=5 variants carry X=1 (stubs 019f1b60, 019f1c60, 019f2a10, 019f2b10). That the env name S21 is "X" by design is inferred; that the X=1 instantiations are reachable only through this condition is proven.
- **Y (arg 6) = HALOGEN_FA_LPT launch mode. CLOSED.** In the special path: `18e9acb: cmpl $0x0, 0x18ffb10(%rip)` (DAT_019ffb10 = `FUN_019f1b30` = `getenv("HALOGEN_FA_LPT")`, default 1, `019f1b30.c`). Nonzero: `18e9ae0 addl %r13d,%r13d; 18e9ae3 leaq (%rdx,%r13),%rdi; 18e9ae7 addq $-0x180,%rdi` = grid (2*tiles, 1), block 0x180; handle 0x18fa238 = `<1,1,true,5,1,true>` (site `18e9c27`). Zero: grid (tiles, 2), handle 0x18fa240 = `<1,1,true,5,1,false>` (site `18ea25a`). Y=true is the flat 1-D (LPT) grid, Y=false the 2-D grid. The dense A=2 twin does the same (`019eb380.c:100-117`; handle for `<2,1,true,5,1,true>` is 0x18fa2a0, stub 019f2a10).
- **Args 1-3 (A, B, bool): partial.** A=1 launches with block 0x180 (384) in `019e9920`; A=2 with block 0x300 (768) in `019eb380` (`0x100000300`), so A = q-tile multiplicity per block (inferred from block size). The fallback launch when `VB==0 || RS==0` in the sparse path uses handle 0x18fa220 = `<1,1,false,0,0,false>` (site `18ea115`). Handle 0x18fa228 = `<1,0,true,0,0,false>` has one launch site, `18ea3b1`; its selecting condition was not traced. `<1,0,false,...>` (0x18fa230) and `<2,0,*>` have no launch site in these two launchers (only registration and the stub's own body), so B=0 looks unused (inferred from absence in a scan of `# 0x18fa...` leas).
- What remains: which of VB and RS is B and which is the third bool (both default 1 and both must be nonzero for the OPT dispatch, so defaults never reach the B=0 stubs).

### 2. qs_bt2 arg b: CLOSED in pass 1 (HALOGEN_ATTN_QS_BT2_OPT). Not re-opened.

### 3. FaSel layout (4 x u64 at kernarg 0x40): PARTIAL

Evidence is the disassembly of the layer function 018b86d0 (`sparse_attn_block_select`), the single caller of the FA launcher (`17bbcfa: callq 0x18e9920`):

```
17bbc99: movq   %rbx, 0x3b8(%rsp)       ; FaSel +0x00 (u64)
17bbca1: movupd %xmm0, 0x3c0(%rsp)      ; FaSel +0x08 and +0x10 (two u64)
17bbcaa: movl   %r12d, 0x3d0(%rsp)      ; FaSel +0x18 (u32)
17bbcb2: movl   $0x0, 0x3d4(%rsp)       ; FaSel +0x1c (u32), overwritten by the callee
```
- In the dense branch, `17bbbce: xorpd %xmm0,%xmm0; 17bbbd2: xorl %ebx,%ebx; 17bbbd4: xorl %r12d,%r12d` zeroes all three sources: FaSel is all zero in dense mode (proven for that branch).
- The struct goes by value on the stack, 32 bytes. `019e9920` copies it with `18e9a94: movups (%r12),%xmm0; 18e9a99: movups 0x10(%r12),%xmm1` into the kernarg value at 0x110(%rsp), which is the last entry of the `hipLaunchKernel` argument-pointer array (`18e9beb: leaq 0x110(%rsp),%rax`). Kernarg 0x40..0x5f per kernels-attention.md (sel_idx 0x30, n 0x38, m 0x3c) is consistent.
- **+0x1c (u32) = high dword of the indexer budget pair.** `18e996c: movl 0x12c08(%rip),%eax  # 0x18fc574` then `movl %eax, 0x1c(%r12)`. DAT_019fc574 is the high dword of `FUN_01853fb0` (HALOGEN_INDEXER_BUDGET; default pair low 0x800 = 2048, high 0x200 = 512). So the host patches +0x1c with 512 at default. Pass 1's "uStack_5c" is this field. Meaning (inferred): per-query selected-block cap.
- Pass 1's five-slot reading from Ghidra stack names (`local_80 .. uStack_5c`) is withdrawn: the layout is the 32 bytes above; those Ghidra names are source values.
- STILL OPEN: what rbx (+0x00), xmm0 (+0x08, +0x10) and r12d (+0x18) hold in sparse mode; they come from the block-select code in 018b86d0, not traced. Kernel-side hint (selected-index list pointer plus count or stride fields) fits +0x00 = index list pointer, unproven.

### 4. Legacy k_attn and k_attn_fa

- **k_attn_fa: launched, one site, only with HALOGEN_FA_TQ=0.** Handle 0x18fa100 -> stub 0x18ec930 (Ghidra 019ec930). Its only external launch site is `18e9d6f: leaq ... # 0x18fa100` in `019e9920` (other hits: `18ec9f1` inside the stub, `18f35c3` registration). Config at `18e9c3c-18e9c53`: grid `0x1800000000 | tiles` = (tiles, 24), block `0x100000180 - 0x160` = 32, matching pass 1's `019e9920.c:90`. It is the `iVar1 == 0` branch (FA_TQ=0 via `FUN_019e97f0`); default TQ is 2 (dense) or 1 (sparse), so not taken at defaults.
- **k_attn: launched (reachable), gate unresolved.** Pass 1's "no launch site" came from the direct call: `17bc99b: callq 0x17be2b0` (stub 018be2b0) inside 018b86d0, after `__hipPushCallConfiguration` with grid `0x1800000000 | n` = (n, 24) and block `%r14` (`17bc70e-17bc72e`). Reached via `17bc73b: je 0x17bc95e` when the stack byte `0x1e0(%rsp)` is 0 and the push succeeded (`17bc960: testl %eax,%eax; jne`). Args (`17bc96a-17bc99a`): `%rdi=DAT_019fc6d0`, `%rsi=*(ctx+0xb98)`, `%rdx=*(ctx+0xba0)`, `%rcx=0`, `%r8=DAT_019fc6f8`, two stack args.
- What remains: what `0x1e0(%rsp)` is (the same byte gates the FA call at `17bbc30`, so it looks like a "selection path" flag), and the surrounding HALOGEN_QSA_DENSE (`018b86d0.c:118`) and HALOGEN_QSA_FORCE_SEL logic. Whether k_attn runs at default flags is unverified. Test needed: trace `hipLaunchKernel` kernel names at default and at HALOGEN_QSA_DENSE=1 and count `k_attn`.

---

## U12

### Verdict: PARTIAL

Budget note: the tool budget ran out before the scan3 and ut launch sites were traced. Everything below marked proven is proven; the rest is listed as open.

### 1. k_dnc_att* launcher (019dd130, selector HALOGEN_DN_S12): proven stub set

Disassembly of 0x18dd130..0x18ddbf0 shows direct `callq`/`jmp` to these stubs (Ghidra = VMA + 0x100000):
- att6: `<64,F,F,F>` 019e0120 (`18dd354`), `<16,F,F,F>` 019e04e0 (`18dd504`), `<32,F,F,F>` 019e0300 (`18dd612`), `<32,T,F,F>` 019e03f0 (`18dd686`), `<64,T,F,F>` 019e0210 (`18dd75f`).
- att7: `<32,8>` 019e0730 (`18dd6f2`), `<32,2>` 019e05d0 (`18dd7cb`), `<32,4>` 019e0680 (`18dd832`).
- att (v1) 019e07e0 (`18dd897`), att8 019dfb60 (`18dd8fc`).
- att9: `<-8>` (`18dd5a3`), `<0>` (`18dd92b`), `<16>` (`18dd95a`), `<4>` (`18dd989`), `<-4>` (`18dd9b8`), `<8>` (`18dd9e7`), `<12>` (`18dda16`); the ATT_SCALE reader `18dda2f: callq 0x18dfc10` precedes them, matching pass 1's case 10.
- Not reached from this launcher: `k_dnc_att2` (019ddbf0), `att6<32,T,T,T>` (019df980), `att6<32,T,T,F>` (019dfa70). Either launched elsewhere or unused (inferred from this range only).
- The case number to stub mapping was NOT recovered (the switch is a jump table; the table was not read).

### 2. Decode versus prefill: STILL OPEN

No new evidence. The att launcher has no row-count branch beyond nc = ceil(tokens/64). GDN decode has separate step knobs (HALOGEN_GDN_STEP_N, HALOGEN_DN_STEP_SR, `0193ef80`), not traced. Inference only: the default decode step does not use this launcher.

### 3. Host functions that decide each variant

- att variant: `019dd130` (switch on DAT_019ff828 = HALOGEN_DN_S12, `019dcbe0`, default 4; with DAT_019ff7b0 = HALOGEN_DN_XH default 1 for the special default path).
- scan3 variant: `019ca280` (HALOGEN_DN_M10P, M10G, S14; pass 1). Chunk driver: `019d7b70` (dn_chunk_forward). ut variant: `019d7b70` (S3A and the UT knobs; pass 1).
- Attention variant (FA2 family): `019e9920` (sparse, FaSel) and `019eb380` (dense), called from `018b86d0` at `17bbcfa`; flags HALOGEN_FA_OPT, FA_S21, FA_LPT, FA_VB, FA_RS, FA_TQ (this pass).
- qs_bt2 family: `019ee360` (pass 1).

### 4. scan3 args B/C/D/G and ut4/ut5 flag meanings: STILL OPEN

Not traced. Next step with the method above: get handles for the scan3/ut stubs from `objdump -R binaries/flash_serve | grep RELATIVE`, then grep `# 0x18fa...` in the ranges of `019ca280` (VMA 0x18ca280) and `019d7b70` (VMA 0x18d7b70), and grep direct `callq/jmp` to the stub VMAs. Each handle at a launch site fixes the instantiation per switch case, which closes the E/F/A mapping and shows which ut5 bools flip with which env flag.

## Corrections to earlier docs

- FaSel is 32 bytes with +0x1c patched by the host (indexer budget high dword); dense mode passes zeros. Pass 1's five-slot reading is withdrawn.
- k_attn has a launch site (direct stub call at 18bc99b Ghidra / 17bc99b VMA); pass 1's "none found" was a handle-only search.
- k_attn_fa is launched only under HALOGEN_FA_TQ=0 (one site, 18e9d6f VMA).
- fa2 Y = HALOGEN_FA_LPT proven by launch sites; Y=true is the flat 1-D grid.
