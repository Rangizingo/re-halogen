# Closure round 5: U11 matmul-path consumer of q5_0 / q5_1 / q8_0

## Scope

Question left open by `dnftaps-r4.md` section 3 and `dequant-host-r2.md:67`: which code consumes repacked q5_0 / q5_1 / q8_0 tensors on the matmul path. Static only. Binary `binaries/flash_serve` (0.17.3). Addresses: `obj` = ELF vaddr (objdump), Ghidra = obj + 0x100000, decomp files are named by the Ghidra address. Planner disassembly was taken from ai-core: `objdump -d --no-show-raw-insn --start-address=0x18a14a0 --stop-address=0x18aa550 binaries/flash_serve` (the decompile `work/decomp/flash_serve/019a14a0.c` is 8 lines because Ghidra cut the function at a no-return `_Znwm`; the real body runs obj 0x18a14a0 to 0x18aa550). Jump tables were read straight from the file bytes with `struct.unpack_from('<i', ...)`. "Inferred" marks anything not read directly.

Naming used below: ggml types are the GGUF enum (Q4_0 2, Q4_1 3, Q5_0 6, Q5_1 7, Q8_0 8, Q4_K 12, Q5_K 13, Q6_K 14, IQ4_NL 20, IQ4_XS 23). hgn dtypes are `analysis/hgn-format.md` section 2 (q8s32 11, q4a 14, q5a 15, q8_0 20, q5_0 21, q5_1 22).

## Verdict: CLOSED

The premise of the open item is wrong in a useful way. hgn dtypes 20 (q8_0), 21 (q5_0) and 22 (q5_1) are produced only for the n-gram table, and their only consumer is `k_gather_ggml<2|3|4>`. The matmul tensors that arrive as ggml Q5_0 / Q5_1 / Q8_0 are re-labelled at repack time into other dtypes that have real kernels: Q5_0 and Q5_1 become q5a (15), Q8_0 becomes q8s32 (11). Dtypes 20/21/22 on a Linear weight hit a fatal reject in the binder's fallback converter.

## 1. ggml type to hgn dtype in the GGUF repack (item 1)

The planner (obj 0x18a14a0) rejects unsupported ggml types with one bitmask test, then maps each accepted type to a (role, hgn dtype) pair stored on the stack as a qword `role | dtype << 32` at `0x148(%rsp)`. Role is the case selector in the filler `019bc480.c` (`switch(uVar37)` at :161, `uVar37 = *(uint *)(... + 0x20 + ...)` at :158).

### 1.1 Reject mask

obj 0x18a1908 `movabs $0x8c204f0c08,%rdx`, 0x18a1929 `mov 0x38(%rbx),%esi` (ggml type), 0x18a192c `cmp $0x27,%rsi`, 0x18a1932 `bt %rsi,%rdx`. Bits set in 0x8c204f0c08: types 3, 10, 11, 16, 17, 18, 19, 22, 29, 34, 35, 39 (python). The message at 0x18a1964 (`lea ... # 2c5a3`) is the string at file 0x2c5a3: ` is not read by this engine (Q4_1, Q2_K, Q3_K and the IQ2/IQ1 families are not repacked losslessly; no lossy fallback is offered). Use a build whose tensors are IQ4_NL, IQ4_XS, IQ3_S, Q4_0, Q4_K, Q5_K, Q5_0, Q5_1, Q6_K, Q8_0 or wider ...`. So 6, 7, 8 pass the gate.

### 1.2 General tensors (Linear weights, routed experts)

Jump table at file 0x478ec (31 entries, indexed by ggml type; jump at obj 0x18a35b9 `jmp *%r9`). Decoded groups: types {0, 1, 30} to 0x18a368b (f32/f16/bf16 path); {2, 20, 21, 23} to 0x18a35bc; {6, 7, 12, 13} to 0x18a3618; {8} to 0x18a40ae; {14} to 0x18a40f0; the rest to 0x18a8d9f (reject).

| ggml type | handler | record written | hgn dtype | role (019bc480 case) |
|---|---|---|---|---|
| Q5_0 (6), Q5_1 (7) | 0x18a3618 | 0x18a3621 `xor $0xf,%eax` after `cmp $0xc,%r8d; sete %al` gives 0xf for type != 12; 0x18a3776 `movl $0xa,0x148(%rsp)` | 15 q5a | 10 (:1724), 0xb for expert 3D (:2010) |
| Q4_K (12) | same handler | same `xor $0xf` gives 0xe | 14 q4a | 10 / 0xb |
| Q5_K (13) | same handler | 0xf | 15 q5a | 10 / 0xb |
| Q8_0 (8) | 0x18a40ae | 0x18a40b8 `movabs $0xb00000002,%rax` / 0x18a40c2 `mov %rax,0x148(%rsp)` | 11 q8s32 | 2 (:398) |
| Q4_0/IQ4_NL/IQ3_S/IQ4_XS (2, 20, 21, 23) | 0x18a35bc | `movabs $0x500000001,%rax` | 5 q4c | 1 (:194) |
| Q6_K (14) | 0x18a40f0 | `movabs $0xc00000005,%rax` | 12 q6k | 5 (:1436) |

That the high dword at `0x14c(%rsp)` is the hgn dtype is inferred from agreement, not read from a named field: Q8_0 role 2 builds a row stride `(K & ~31) + (K >> 5) * 2` (`019bc480.c:430-431` `uVar52 = uVar40 >> 5; uVar32 = (uVar40 & 0xffffffffffffffe0) + uVar52 * 2`), which is the q8s32 size rule (N*K + 2*N*K/32, `hgn-format.md` section 3); role 10/0xb pass `uVar37 == 0xc ^ 5` as the bit count (4 or 5) to `019c3610` / `019c37a0`, matching q4a/q5a "bits = 4 or 5" in `dequant-dispatch-r3.md:54`; Q6_K gets role 5 and dtype 12 (q6k).

Q5_0 payload: `019c37a0.c:318` `if ((param_5 & 0xfffffffe) == 6) {` (types 6 and 7), `:320` `if (param_5 == 6) {` (Q5_0 only), error text at `:533` ` have a scale d whose -16 * d is not an fp16 ...`. So Q5_0 rows are rewritten as an affine pair (d, -16*d) = the q5a layout (hgn-format section 3: q5a 24,576 B for [128,256] = 20,480 + 4,096, scale and min per 32). Q5_1 already has (d, m).

Host checks agree: `018897f0.c:1739` `"%sexperts.down_proj: %s qparam %u has no down kernel (Q5_1 or Q5_0, the 5-bit PAIR form, is what is read)\n"` is printed when the down tensor is q4a/q5a (`& 0xfffffffe == 0xe`, `018897f0.c:1670`) but not in the 5-bit pair form. The "PAIR form" in that message is dtype 15.

### 1.3 n-gram table (the only place dtypes 20/21/22 are written)

Jump table at file 0x47968 (ggml types 6 to 20, jump at obj 0x18a3752 `jmp *%rdx` after `add $0xfffffffa,%edx; cmp $0xe,%edx`). Decoded: 6 to 0x18a3754, 7 to 0x18a3d87, 8 to 0x18a3d6d, 20 to 0x18a3d7a, everything else in 6..20 to 0x18a8a1a (reject).

- 0x18a3754 `mov $0x15,%ebx` / `mov $0x6e,%r14d`: Q5_0 to dtype 21, 110 B per 160-element row.
- 0x18a3d87 `mov $0x16,%ebx` / `mov $0x78,%r14d`: Q5_1 to dtype 22, 120 B.
- 0x18a3d6d `mov $0x14,%ebx` / `mov $0xaa,%r14d`: Q8_0 to dtype 20, 170 B.
- 0x18a3d7a `mov $0xd,%ebx` / `mov $0x5a,%r14d`: IQ4_NL to dtype 13, 90 B.
- Then 0x18a3d96 `cmp $0x10,%rcx; jne 0x18a8a1a`, 0x18a3da7 `cmpq $0xa0,(%rax)` (160 elements), and 0x18a3e4c `movl $0x8,0x148(%rsp)` / 0x18a3e57 `mov %ebx,0x14c(%rsp)`: role 8 (a verbatim copy, `019bc480.c:1676`), dtype taken from `%ebx`.
- Reject target 0x18a8a1a: `lea ... # 2b099`, file 0x2b099 `: expected IQ4_NL, Q8_0, Q5_0 or Q5_1 [160, 128*R], got `.

So Q5_0, Q5_1, Q8_0 become dtypes 21, 22, 20 only for the 160-wide n-gram rows ([160, 128*R] shape check at 0x18a3da7). These values match the loader's own accept list at obj 0x178e2e4 (`lea -0x14(%rdi),%ecx; cmp $0x3,%ecx`), string at file 0x314f2 `ngram_embedding: expected fp8g, iq4nl, q8_0, q5_0 or q5_1, got %s`.

## 2. Consumers of dtypes 20 / 21 / 22 (item 2)

| dtype | function and site | kernel |
|---|---|---|
| 20 q8_0 | loader stores gather code 2: obj 0x178e337 `cmp $0x14,%edi`, 0x178e354 `cmove %edx,%eax` (edx = 2), then `mov %eax,0x79c(%rcx)` (r4 cites 0x178e6b0) | `k_gather_ggml<2>` (`kernels-disasm/codeobj_53000.s` 2405553) |
| 21 q5_0 | same site: `cmp $0x15,%edi; setne %al; add $0x3,%eax` gives 3 | `k_gather_ggml<3>` (2405809) |
| 22 q5_1 | same site: 4 | `k_gather_ggml<4>` (2405945) |

Launch: `work/decomp/flash_serve/018ede30.c:165` `switch(*(undefined4 *)(param_1 + 0x79c)) {`; this is the only decompiled file that reads `+0x79c` (`grep -ln 0x79c work/decomp/flash_serve/*.c`). Code to handle mapping is in `dnftaps-r4.md` section 3.1 (table 0x468bc, handle table 0x18f6e00). Decode constants (q5_0 bias -16, q5_1 per-block min, q8_0 none) are in `dnftaps-r4.md` section 3.2 and are unchanged.

Any other reader of dtypes 20 to 22: none found. Checked the Linear binder `FUN_0190e330` (switch `0190e330.c:196` has only cases 0, 7, 0xb, 0xc, 0x10, 0x18) and the fallback converter below; neither accepts them.

## 3. Proof that 20 / 21 / 22 have no matmul consumer (item 3)

1. Binder: `0190e330.c:196` `switch(*(int *)(param_2 + 0x10)) {` with `case 0:` (:197), `case 7:` (:201), `case 0xb:` (:255), `case 0xc:` (:306), `case 0x10:` (:348), `case 0x18:` (:375) fills a descriptor slot only for bf16, q8g64, q8s32, q6k, ht, q6g64. Everything else falls to `0190e330.c:429` `lVar13 = FUN_0190d290(param_2,param_3);` (the dense bf16 converter, real body obj 0x180d290 to 0x180dfc0, `dequant-dispatch-r3.md` section 3).
2. Converter dispatch: obj 0x180d34f `mov 0x10(%r12),%edi` (dtype), 0x180d354 `cmp $0x18,%rdi`, 0x180d358 `ja 0x180dc84`, then table at file 0x46904 (`lea ... # 46904`, `movslq (%rax,%rdi,4),%rcx`, `jmp *%rcx` at 0x180d36f). Decoded (25 entries): 0 to 0x180d9c8, 5 to 0x180d89d, 7 to 0x180d6ce, 11 to 0x180d705, 12 to 0x180d397, 14 and 15 to 0x180d371, 24 to 0x180d52a, and dtypes 1, 2, 3, 4, 6, 8, 9, 10, 13, 16, 17, 18, 19, **20, 21, 22**, 23 to 0x180dc84.
3. Reject branch: obj 0x180dc84 loads stderr, 0x180dc98 `lea ... # 27f93` (file 0x27f93 `%s: expected bf16, Q4C-P, Q8G64, Q8S32, Q6K, Q4A or Q5A, got %s qparam %u`), 0x180dcaa `call fprintf@plt`, 0x180dcaf `mov $0x1,%edi`, 0x180dcb4 `call exit@plt` (exit identity from the call to PLT 0x18f4540 with edi = 1, as in `dequant-dispatch-r3.md:57`). A dtype-20/21/22 Linear weight is a fatal startup error. (16 ht and 23 i4r also land here but are handled before the converter by their own slots/expert code, `dequant-dispatch-r3.md` table; not relevant to this question.)
4. Positive consumers of what Q5/Q8 GGUF tensors become:
   - q5a / q4a (15 / 14): converter entry 0x180d371 calls `FUN_0189a700` (builds AffDev) then the `k_deq_aff<4|5,bool>` launcher at obj 0x17e9180 (call at 0x180da58, `dequant-dispatch-r3.md:72`), then the bf16 GEMM (`k_lbfw`). Routed experts: `k_exp_gu_aff` / `k_exp_dn_aff` / `k_expert_*_aff` (`expert-dispatch-r4.md:40`; `018897f0.c:1670-1740` sets `lay+0x721/0x722` for dtype 14/15).
   - q8s32 (11): binder slot +0x18 (`0190e330.c:255` `case 0xb:`), dense `k_lq8s32<M,R,W,false>` via `FUN_01920310` (`018a7520.c:41`), prefill `k_deq_q8s32` (obj 0x17eaf95) then GEMM, routed-expert down `k_exp_dn_w_q8` (`expert-dispatch-r4.md:39`).

## Outcome for the tensors the question named

| GGUF tensor class | ggml | hgn dtype stored | consumer |
|---|---|---|---|
| Linear, experts | Q5_0, Q5_1 | 15 q5a (PAIR form) | `k_deq_aff<5,*>` then `k_lbfw`; expert `k_exp_*_aff` |
| Linear, experts down | Q8_0 | 11 q8s32 | `k_lq8s32`, `k_deq_q8s32`, `k_exp_dn_w_q8` |
| n-gram table ([160, 128*R]) | Q5_0 / Q5_1 / Q8_0 | 21 / 22 / 20 | `k_gather_ggml<3|4|2>` only |
| any Linear carrying dtype 20/21/22 (e.g. a hand-built .hgn) | n/a | 20/21/22 | none: `exit(1)` at obj 0x180dc84 |

## Residuals

1. The mapping `0x14c(%rsp)` = hgn dtype is inferred from the cross-checks in section 1.2, not from a named field. A runtime check would be: repack a small Q5_0 GGUF with `flash_serve --repack IN.gguf --out OUT.hgn` on ai-core and run `halogen-tools inspect` on OUT.hgn to read the dtype column. Not done (no GGUF in the workspace; `flash_serve` must not be run per CLAUDE.md).
2. Which tensors a particular published GGUF (unsloth, bartowski) stores as Q5_0 / Q5_1 / Q8_0 is build-dependent and unobserved here; this note proves the mapping, not a census.
3. The planner has 7,600 lines; only the type-keyed paths for ggml 6, 7, 8, 12 to 14, 20, and the n-gram path were decoded. The 3D expert block (obj 0x18a42e3 to 0x18a4394: `mov $0x30c0,%eax; bt %ebx,%eax` accepts 6, 7, 12, 13 to role 0xb, dtype 0xe/0xf) was read but its per-name conditions were not enumerated. Expert-down Q8_0 reaching q8s32 is inferred from the string at file 0x398ce (`the engine's down kernels take IQ4_NL, IQ4_XS, IQ3_S, Q4_0, Q8_0, Q5_0 or Q5_1`, reject at 0x18a8e22, entered from 0x18a4146 for types outside 6..8) plus `expert-dispatch-r4.md:39`; the per-name routing was not traced.
4. The bit order of the q5a payload produced from Q5_0 and the `k_deq_aff` decode were not re-derived (`019c37a0.c` is vectorized). The U11 bias question is closed: q5_0 -16 (gather, `v_add3_u32 v3, v3, v4, -16`), q8_0 0, q5_1 per-block min; on the matmul path the -16 is folded into the f16 min by the repacker (`019c37a0.c:533`), so no bias constant exists in the q5a dequant (inferred from the repacker text and the q5a size rule).
5. A host-side (CPU) decode of dtypes 20 to 22 in the n-gram lookup threads (`HALOGEN_NGRAM_GATHER_THREADS`, `01903710.c`) was not ruled out beyond the `+0x79c` search.

Reproduce:

```bash
cd /Users/peterb/Documents/AI/re-halogen
# planner type map, reject mask, n-gram dtype assignment
ssh ai-core 'cd ~/re-halogen && objdump -d --no-show-raw-insn --start-address=0x18a14a0 --stop-address=0x18aa550 binaries/flash_serve' | sed -E 's/<[^>]*>//g' | grep -E '8c204f0c08|movabs \$0x(b00000002|500000001|c00000005)|mov    \$0x1[456],%ebx|xor    \$0xf,%eax'
# jump tables
python3 -I -c "
import struct;d=open('binaries/flash_serve','rb').read()
for base,n in((0x47968,15),(0x478ec,31),(0x46904,25)):
    print(hex(base),[hex(base+struct.unpack_from('<i',d,base+4*i)[0]) for i in range(n)])"
# converter reject
ssh ai-core 'cd ~/re-halogen && objdump -d --no-show-raw-insn --start-address=0x180dc84 --stop-address=0x180dcc0 binaries/flash_serve'
```
