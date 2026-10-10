# Closure round 4: U21 routed-expert dispatch (token-count ranges and the expert dtype binder)

Inputs: `work/decomp/flash_serve/*.c` (Ghidra address = ELF vaddr + 0x100000), `llvm-objdump -d` of `binaries/flash_serve` (ai-core `/tmp/r3dd/text.s`, same listing as R3; addresses written `obj 0x17...` are ELF vaddrs), `analysis/function-map-kernel-stubs.tsv`, `analysis/closure/dequant-dispatch-r3.md` ("R3"). Static only, nothing was run. "Inferred" marks anything not read directly from code.

Method note: Ghidra cut three functions at their first no-return `__hipPushCallConfiguration`, so the decomp shows a fraction of each. Real extents, read from the listing: `moe_layer_forward` FUN_018d7510 = obj 0x17d7510 to 0x17e43d0 (next function starts 0x17e43e0 `pushq %rax`), `FUN_018d2430` (i4r plan) = obj 0x17d2430 to 0x17d59b1 (the `nop` after the last `exit` call), `moe_forward` FUN_018be390 (HT) = obj 0x17be390 to about 0x17ca800. Kernel launches were attributed two ways: a direct `callq` to a host stub (stub table rows), or `leaq <handle>(%rip), %rdi` before `hipLaunchKernel`, with handle to stub resolved by the first `leaq ... %rdi` inside each stub (1738 of 1738 stubs resolved).

## 0. What R3 left and what changed

R3 section 5 listed: (1) routed-expert kernels by token count, (2) the binder path for dtype 23 and plain 4-bit experts. Both are answered below. New facts:

1. The expert dtype switch is in the layer binder `FUN_018897f0` (not `FUN_0190e330`, which is the dense binder). It stores the two tensor descriptors in place and sets flag bytes in the layer record (section 1).
2. `k_q4moe` is the prefill grouped GEMM for BOTH the plain Q4C-P experts and the i4r experts. R3 had it as decode-side only; it is the n >= 64 (plain) and n >= 9 (i4r) kernel.
3. The decode flag `ctx[1]` is written once per forward at 018fa150.c:50 (`param_1[1] = bVar8`), the same value as `DAT_019fe608` (018fa150.c:53). 1 = decode (n < 9 with a non-empty cache), 0 = prefill.
4. Slots per token are 10 (topk): obj 0x17d94d5 `addl %eax, %eax`, 0x17d94d7 `leal (%rax,%rax,4), %eax`, stored to 0xa8(%rsp) (= 10 * n); same as 018d2430.c:64 (`iVar4 = iVar4 * 10`).

## 1. Binder: where tensor dtype becomes an expert kernel family

Function `FUN_018897f0` (layer binder). Gate-up and down tensors are looked up by name and the descriptor pointer is stored unconverted (no dequant, no repack). The descriptor dtype is `desc+0x10`, qparam `desc+0x44` (obj 0x17d765b `cmpl $0x5, 0x10(%rax)`, 0x17d766e `cmpl $0x2, 0x44(%rax)`).

| line | action |
|---|---|
| 018897f0.c:1347-1351 | `"experts.gate_up_proj.weight"` through `FUN_01894600` / `FUN_01997e40`, stored `*(long*)(layer+0x700)` |
| 018897f0.c:1369-1373 | `"experts.down_proj.weight"`, stored `*(long*)(layer+0x708)` |
| 018897f0.c:1417-1420 | `iVar6 = *(int*)(desc700+0x10)`; `if (iVar6 == 0x10) { *(layer+0x723) = 1; ...` HT branch: both must be 0x10, slot +8 non-null, shapes `[0x200,0x500,0xa00]` gate-up and `[0x200,0xa00,0x280]` down (:1422-1440) |
| 018897f0.c:1533-1549 | not HT: `if (*(int*)(desc700+0x10) == 0x17) goto LAB_0188c6b9` (:1534); else `bVar27 = dtype708 == 0x17; *(layer+0x748) = bVar27` (:1536-1537). `LAB_0188c6b9` sets `*(layer+0x748) = 1` (:1549) |
| 018897f0.c:1550-1558 | i4r validation: both dtypes 0x17, both slot +8 non-null, `desc700+0x20 == 0x500`, `desc708+0x20 == 0xa00`, both slot +8 pointers 64-byte aligned (`& 0x3f`). Error text: "I4R needs both projections in I4R ([512, 1280, 2560] gate_up, [512, 2560, 640] down), 64-B aligned, on the pinned path" |
| 018897f0.c:1560-1660 | i4r side tensors: `experts.su` (size 0xa00) to layer+0x750, `experts.gate_up_proj.svh` (0x500) to +0x758, `experts.down_proj.suh` (0x280) to +0x760, `experts.down_proj.svh` (0xa00) to +0x768. HT uses +0x728..+0x740 with other sizes (:1459-1500) |
| 018897f0.c:1659-1663 | `if (!HT && !i4r) *(layer+0x720) = (dtype708 == 0xb)` (down is Q8S32; nbytes must be 0x35c00000) |
| 018897f0.c:1672-1675 | `*(layer+0x721) = (dtype700 & ~1) == 0xe` (gate-up is Q4A or Q5A, dtype 14 or 15); `*(layer+0x722) = (dtype708 & ~1) == 0xe` |
| 018897f0.c:1752-1762 | reject unless HT, i4r, or (gate-up dtype 5 or q4a/q5a) and (down dtype 5 or flag 0x720/0x722): "experts: gate_up must be Q4C-P, Q4A, Q5A or HT and down Q4C-P, Q8S32, Q5A or HT" |

Expert dtype to family map, by the flag bytes of the layer record (`lay = *(ctx+0x5e0) + layer * 0xc68`, 018d7510.c:68, obj 0x17d753a `imulq $0xc68`):

| expert tensors | flag set | family |
|---|---|---|
| dtype 16 (ht) both | `lay+0x723` | HT MoE (`FUN_018be390`) |
| dtype 23 (i4r) both | `lay+0x748` | I4R MoE (`FUN_018d2430`) |
| dtype 5 (q4c planar, qparam 2) both | none | plain Q4C-P: q4moe prefill GEMM or older per-expert kernels |
| gate-up 5, down 11 (q8s32) | `lay+0x720` | plain family, Q8S32 down kernels (`k_exp_dn_w_q8`) |
| dtype 14 / 15 on gate-up or down | `lay+0x721` / `lay+0x722` | plain family, affine kernels (`k_exp_gu_aff`, `k_exp_dn_aff`, `k_expert_*_aff`; AffDev from `FUN_0189a700`) |
| anything else | none | fatal at bind (:1762) |

Plain 4-bit qparam: moe_layer_forward re-checks `dtype == 5` and `qparam == 2` for both tensors before the V2 route (obj 0x17d765b, 0x17d766e, 0x17d7668, 0x17d7674; 018d7510.c:131-134). The binder does not check qparam for experts (:1752-1757 tests only dtype 5).

Answer to open item (2): dtype 23 is bound in place by the layer binder at 018897f0.c:1533-1549, not by `FUN_0190e330` or the dense converter `FUN_0190d290` (no case 23, R3 section 3). Plain 4-bit is dtype 5 with no flag byte set; it is bound at :1347-1373 and accepted at :1752-1757.

## 2. Layer-kind dispatch in moe_layer_forward (FUN_018d7510)

After routing, expert compute is chosen by the flag bytes (obj addresses):

| obj | instruction | meaning |
|---|---|---|
| 0x17d94a9 | `cmpb $0x1, 0x723(%rcx)` then `callq 0x17be390` at 0x17d94c3 | HT layer: `FUN_018be390`, then jump to the tail 0x17e1c11 |
| 0x17d9631 | `cmpb $0x1, 0x748(%rax)` then `callq 0x17d2430` at 0x17d9650 | i4r layer: `FUN_018d2430`, then tail 0x17e1c11 |
| 0x17d965a onward | neither flag | plain family (section 5) |

(The block 0x17d94e2 to 0x17d9629 handles `ctx+0x378 != 0`, a per-layer callback hook, inferred and not traced; both arms converge on the same two callees.)

## 3. I4R experts (dtype 23): FUN_018d2430, by token count

Entry gate, 018d2430.c:67 and obj 0x17d24e5 `cmpq $0x9, %r15; jl 0x17d2890`, 0x17d24ef `cmpb $0x0, 0x1(%rbp); jne 0x17d2890`:

| range | condition | kernels (cite) |
|---|---|---|
| decode: n < 9, or `ctx[1] != 0` | branch to obj 0x17d2890 | `k_i4r_rows<2560>` (launch 0x17d3bd3), `k_i4r_gu_epi` (0x17d3cc0), `k_i4r_rows<640>` (0x17d3dcf), `k_i4r_fold` (0x17d3f00), `k_i4r_rows_rot` / `_epi` / `_fold` (0x17d2a81, 0x17d2bba, 0x17d2ce3), the `_gl/_fl/_o/_g` family (0x17d49f9 to 0x17d57cb). Selection among them is by `HALOGEN_I4R_FUSE` (018d2430.c:70, cell 0x18fccb0), `HALOGEN_I4R_LAST` (:149); not decoded further here |
| prefill: n >= 9 and `ctx[1] == 0` | fall through at 0x17d24f9 | `k_ht_rot_rows` (handle 0x18f6bb8, launch obj 0x17d2610; grid `(20n+4)>>3` = obj 0x17d2545 `leaq 0x4(,%rax,4)`, 0x17d254d `shrq $0x3`, same as R3), then `k_q4moe_items<W>`, then two `k_q4moe` launches (gate-up, down), then `k_i4r_fold_rows` or `k_slot_moe_sum_rows` |

Prefill arena: `n * 0x1400` bytes (obj 0x17d24fc `shlq $0xa, %rax`, 0x17d2500 `leaq (%rax,%rax,4)`; compared with the arena size at 0x17d2504, overflow goes to 0x17d57d9, the "internal sizing error in the per-forward arena" exit).

Tile width W (rows per item), the only token-count threshold inside the prefill path:

- obj 0x17d27ac `cmpl $0x1366, 0x150(%rsp); jg 0x17d2db4`, and again at 0x17d2f22 and 0x17d31a4. 0x150(%rsp) = n (saved at 0x17d2459 `movq %rdx, 0x150(%rsp)`). So `n <= 4966` uses W = 64, `n >= 4967` uses W = 128.
- Item-count sizing agrees: obj 0x17d2653 `cmpl $0x1367, %ebx; setge %cl; shll $0x6, %ecx; addl $0x40, %ecx` = 64 + 64 * (n >= 4967), then `idivl` of 10n (0x17d2662-0x17d2666) by that width.
- No env override of W on the i4r path (the plain path has one, section 5).

Items kernel: W = 64 launches `q4moe::k_q4moe_items<64>` (handle 0x18f92a8, obj 0x17d2884); W = 128 launches `k_q4moe_items<128>` (handle 0x18f92b0, obj 0x17d2e7b).

GEMM kernels, `q4moe::k_q4moe<A, B, W, Tune<0,128>, E>`. A = V2 mode `DAT_019fcca0` = `FUN_018d59c0` = `getenv("HALOGEN_FLASH_MOE_V2")` returning `(strtol == 2) + 1`, default 1 (obj 0x17d5817 `callq 0x17d59c0`, stored 0x17d581c to 0x18fcca0; tested at 0x17d2f18 `cmpl $0x1, %r14d`). E on the down launch is 2 when `DAT_019fccd8` (= `HALOGEN_I4R_DN_FOLD`, default true: obj 0x17d5950 `movb $0x1, %al` when unset, 0x17d5890 `cmpb $0x30, (%rax); setne %al` when set, string at ELF 0x233af; tested at 0x17d308a `cmpb $0x1, %bl`) else 1.

| A | W | stage 1, gate-up (B=0, E=1) | stage 2, down, DN_FOLD on (B=2, E=2) | stage 2, DN_FOLD off (B=2, E=1) |
|---|---|---|---|---|
| 1 | 64 | `<1,0,64,1>` stub 0x1879810, launch 0x17d303e, handle 0x18f92b8 | `<1,2,64,2>` stub 0x1879870, 0x17d3198, 0x18f92c0 | `<1,2,64,1>` stub 0x18798d0, 0x17d405e, 0x18f92c8 |
| 1 | 128 | `<1,0,128,1>` 0x1879930, 0x17d3531, 0x18f92d0 | `<1,2,128,2>` 0x1879990, 0x17d368b, 0x18f92d8 | `<1,2,128,1>` 0x18799f0, 0x17d4280, 0x18f92e0 |
| 2 | 64 | `<2,0,64,1>` 0x1879a50, 0x17d32c0, 0x18f92e8 | `<2,2,64,2>` 0x1879ab0, 0x17d341a, 0x18f92f0 | `<2,2,64,1>` 0x1879b10, 0x17d416f, 0x18f92f8 |
| 2 | 128 | `<2,0,128,1>` 0x1879b70, 0x17d37a2, 0x18f9300 | `<2,2,128,2>` 0x1879bd0, 0x17d38fc, 0x18f9308 | `<2,2,128,1>` 0x1879c30, 0x17d4391, 0x18f9310 |

(Stub addresses are Ghidra addresses from `function-map-kernel-stubs.tsv`; obj = Ghidra - 0x100000. Branch targets for the fold-off column: A=1 W=64 `jne 0x17d3f59` at 0x17d308d reaches 0x17d405e; A=1 W=128 `jne 0x17d417b` at 0x17d3580 reaches 0x17d4280; A=2 W=64 `jne 0x17d406a` at 0x17d330f reaches 0x17d416f; A=2 W=128 `jne 0x17d428c` at 0x17d37f1 reaches 0x17d4391.)

Tail: `cmpb $0x1, 0x18fccd8` at obj 0x17d43b2; fold on launches `k_i4r_fold_rows` (handle 0x18f6bc0, 0x17d44fb), off launches `k_slot_moe_sum_rows` (handle 0x18f6718, 0x17d4610).

Reading of the template arguments (inferred): A follows `HALOGEN_FLASH_MOE_V2` (1 or 2), B 0/2 = gate-up versus down stage, E 1/2 = fold bit. The weights these kernels read for i4r are the slot-+8 pointers of the two descriptors (`FUN_018d2430` loads `*(desc700+8)` and `*(desc708+8)` into the Args, 018d2430.c:52-53), so the i4r nibble-plane data is consumed by the same kernel family as Q4C-P. The i4r input is first rotated by `k_ht_rot_rows` with the `su` vector (layer+0x750), which is the visible host-side difference.

## 4. HT experts (dtype 16): FUN_018be390, by token count

Early gate, 018be390.c:170: `if (HT_MOE_GEMM == 0 || n < 9 || ctx[1] != 0)` the decode path (token-indexed or grouped GEMV); otherwise the prefill GEMM path. `HT_MOE_GEMM` is `DAT_019fca88 = FUN_018ca640` (`HALOGEN_HT_MOE_GEMM`, default 1; asm obj 0x17be814 `cmpl $0x0, 0x13e26d(%rip) # 0x18fca88`).

The decode branch (`HT_DEC_FUSE`, `n < 9`, `topk < 0x19a`, 018be390.c:181) is existing U10 material (`closure/moe-host.md`) and is not repeated.

Prefill tile rows, 018be390.c:623-633: `iVar7 = (HT_MOE_V2 == 0) * 0x40 + 0x40`; when `HT_MOE_V2 == 1`: gate-up tile 128 rows if `HT_MOE_M128_GU <= n` and bit 0 of `HT_MOE_M128` (`bVar16 = DAT_019fcaf8 <= n & DAT_019fcad8`, :625); down tile 128 rows if bit 1 of `HT_MOE_M128` and `HT_MOE_M128_DN <= n` (:626-629). Defaults (decomp): `HT_MOE_V2` 1 (018ca6b0), `HT_MOE_DV` 1 (018ca6f0), `HT_MOE_DNX` 1 (018ca730), `HT_MOE_M128` 3 (018ca770), `HT_MOE_M128_DN` 0x1000 = 4096 (018ca7b0), `HT_MOE_M128_GU` 0x4000 = 16384 (018ca7e0), `HT_MOE_WPE` 8 (018ca670). With defaults: n < 4096 both 64-row tiles; 4096 <= n < 16384 down 128-row; n >= 16384 both 128-row.

Kernel by `HT_MOE_V2` mode (`DAT_019fcaa8`), switch at obj 0x17bfc33 `movl 0x13ce6f(%rip), %ecx # 0x18fcaa8`, 0x17bfc39 `testl %ecx,%ecx; je 0x17bffa5`, 0x17bfc48 `cmpl $0x3,%ecx; je 0x17bfe2b`, 0x17bfc51 `cmpl $0x1,%ecx`:

| mode | target | kernels (cite) |
|---|---|---|
| 0 | 0x17bffa5 (`WPE == 1` test at 0x17bfff0 `cmpl $0x1, %ebp`) | `k_ht_moe_gu<bits,8,WPE>` and `k_ht_moe_dn<bits,8,WPE>`, WPE 1 or 8 (calls 0x17c003b to stub 0x17d1e30, 0x17c14f5 to 0x17d2130, 0x17c2609, 0x17c26ac, 0x17c288b, 0x17c2920) |
| 3 | 0x17bfe2b, launch 0x17bff99, handle 0x18f6ac8 | `k_ht_moe_gu2w<bits,64,6,0>` and `k_ht_moe_dn2w<bits,64,6,E>` (0x17c0d8b, 0x17c0f9d, 0x17c5135 and siblings) |
| 1 (default), DV, DNX, M128 mask all 0 | falls through 0x17bfc56 to 0x17bfc78 | `k_ht_moe_gu2<bits,64,0>` (launch 0x17bfdec handle 0x18f69b8; bits 4: 0x17c017a) and `k_ht_moe_dn2<bits,64,0>` (0x17c0c74, 0x17c0e94) |
| 1 with DV or DNX or M128 nonzero (all defaults) | `jne 0x17c13cf` at 0x17bfc5e, 0x17bfc6b, 0x17bfc78 | gate-up `k_ht_moe_gu3<bits,1,E>`, E = `HT_MOE_DV` 0 / 1 / 2 (calls 0x17c1461 to stub 0x17d00f0, 0x17c1bf2 to 0x17d09b0, 0x17c1c6e to 0x17d1270; dispatch `cmpl $0x1,%ecx; je 0x17c1b80` at 0x17c13de); down `k_ht_moe_dn3<bits,1,E>` or `k_ht_moe_dn2x<bits,64,6\|1,E>` per `HT_MOE_DNX` (0x17c42a1, 0x17c432a, 0x17c44d9, 0x17c50b4, 0x17c5b3f and siblings); bits 3 or 4 from `cmpl $0x3, 0x12c(%rsp)` (0x17bfca9) |

Mode 2 (the `jne 0x17bfc7e` arm) and the exact DNX/M128 cases for the down kernel were not decoded. The tie of the 64/128 tile widths to specific kernel instances in the last two rows comes from the host arithmetic above and is inferred.

## 5. Plain Q4C-P experts (dtype 5, no flag byte): moe_layer_forward, by token count

All obj addresses are inside `FUN_018d7510`. Environment cells: `DAT_019fcce8` = `FUN_018e43e0` = `HALOGEN_FLASH_MOE_V2` (1 or 2, else 0; default 1); `DAT_019fccf8` = `FUN_018e4420` = `HALOGEN_FLASH_MOE_V2_MINN` (default 0x40 = 64; a set value is clamped to >= 9); `DAT_019fcd08` = `FUN_018e4460` = `HALOGEN_FLASH_MOE_V2_BN` (64 or 128, else 0); `DAT_019fcd18` = `FUN_018e44a0` = `HALOGEN_FLASH_MOE_V2_CONC` (default 0); `DAT_019fcd28` = `FUN_018e44d0` and `DAT_019fcdb4` = `FUN_018e8870` = `HALOGEN_FLASH_MOE_FUSED_N` (default 0).

Two gates are involved.

Gate G1 (V2 route fusion and side stream), 018d7510.c:103-106, obj 0x17d7593 to 0x17d75e2: `MOE_V2 != 0`, `n >= 2` (`cmpl $0x2, %eax; jl` at 0x17d75b7), `n > FUSED_N` (`cmpl 0x18fcd28, %eax; jle` at 0x17d75c0), `ctx[1] == 0` (`cmpb $0x0, 0x1(%rdi)` at 0x17d75cc), `n >= MINN` (`cmpl 0x18fccf8, %eax; jl` at 0x17d75d6), `ctx+0x378 == 0`, `DAT_019fe3e0 == 0`, `DAT_019fe920 != 2` (`HALOGEN_MOE_DN_ROWS`, obj 0x17d761b), flags `lay+0x720..0x723 == 0`, both dtype 5 with qparam 2, slot +0 non-null on both. G1 yields `0xdf(%rsp)` (slot +8 also non-null on both, obj 0x17d76a9, 0x17d76b0). With `0xdf` set, `FUN_018e4580` (route-decode-fused) is skipped (obj 0x17d77a0 `cmpb $0x0, 0xdf(%rsp); jne 0x17d77b7`) and a side stream is created (obj 0x17df94a to 0x17df9ff: `hipStreamCreateWithFlags`, `hipEventCreateWithFlags`). G1 chooses route and stream options; it does not itself select the GEMM.

Gate G2 (the q4moe path), obj 0x17d96a1 to 0x17d96eb (`DAT_019fcdb4` is initialised from `FUN_018e8870` at obj 0x17e308e):

```
17d96a8  cmpl $0x2, %edx            ; n >= 2
17d96b1  cmpl 0x18fcdb4, %edx       ; n > FUSED_N (default 0)
17d96bd  cmpl $0x40, %edx ; setb %dl ; orb 0x1(%r14), %dl ; jne 0x17d9922   ; n < 64 or decode: leave
17d96cd  cmpl $0x5, 0x10(%rax)      ; gate-up dtype 5
17d96d7  cmpl $0x5, 0x10(%rcx)      ; down dtype 5
17d96e1  cmpq $0x0, (%rax) ; 17d96eb cmpq $0x0, (%rcx)   ; planar slot present
```

The `$0x40` is a hard-coded 64, independent of `MOE_V2_MINN`. If both slot-+8 pointers are null the code stages a copy of the planar weights once: `ensure_device_buffer_grow` (`callq 0x17e88a0`, obj 0x17d975f and 0x17d978c, buffers at ctx+0x478 and ctx+0x480) and `hipMemcpy` wrappers (`callq 0x179a920`, 0x17d97a7 and 0x17d97c2); the staged pointers then feed the checks below (0x17d986d-0x17d9870). Inferred: a one-time layout copy for the q4moe kernel.

Layout checks (obj 0x17d98c1 to 0x17d9915 and 0x17df8e7 to 0x17df944), through `check_q4cp_tensor_layout` (`callq 0x17e8950`): `(0x110(%rsp) + r13) & 0xf == 0`, `& 0x7 == 0`, gate-up `0x100(%rsp) == 0xa0000` and `0x108(%rsp) == 0xa00`, down `0x1f0(%rsp) == 0x140000` and `0x1f8(%rsp) == 0x280`. Failure jumps to 0x17d991b, which sets `bl = r14b = 1`; the later `andb %bl, %r14b; cmpb $0x1; jne 0x17d99d1` (obj 0x17d994d-0x17d9954) then leaves the q4moe route. Pass reaches 0x17df94a and the q4moe launch.

### 5.1 q4moe path (n >= 64, prefill, shapes match)

| step | cite |
|---|---|
| tile width W | obj 0x17dfa4f `movl 0x18fcd08, %eax`; 0x17dfa58 `cmpl $0x1367, 0xf0(%rsp)`; `setge %r12b; shll $0x6; addl $0x40` = 64 + 64 * (n >= 4967); 0x17dfa6f `testl %eax,%eax; cmovnel %eax, %r12d` (`_BN` 64 or 128 overrides). 0x17dfbd3 `cmpl $0x40, %r12d; jne 0x17e06c5` splits W = 64 versus 128 |
| items kernel | W = 64: `k_q4moe_items<64>` `callq 0x1879670` at 0x17dfc1c; W = 128: `k_q4moe_items<128>` `callq 0x1879740` at 0x17e0700 |
| A | `movl 0x18fcce8, %r13d; cmpl $0x1, %r13d` at 0x17e0801 / 0x17e0837 (V2 mode 1 versus 2) |
| gate-up and down, W=64 | A=1: `k_q4moe<1,0,64,Tune<0,128>,0>` `callq 0x1879c90` at 0x17e0935, then `<1,2,64,..,0>` `0x1879cf0` at 0x17e0a59; A=2: `<2,0,64,..,0>` `0x1879e10` at 0x17e0b57, then `<2,2,64,..,0>` `0x1879e70` at 0x17e0c7b |
| gate-up and down, W=128 | A=1: `<1,0,128,..,0>` `0x1879d50` (0x17e1cfd), `<1,2,128,..,0>` `0x1879db0` (0x17e1e21); A=2: `<2,0,128,..,0>` `0x1879ed0` (0x17e1f05), `<2,2,128,..,0>` `0x1879f30` (0x17e2029) |
| tail | `k_slot_sum_rows` (0x17e20a9), `k_slot_moe_sum_rows` (0x17e215f), `k_moe_sum` (0x17e21a8) |

Observed: every launch on the plain path has E = 0; the i4r path has E = 1 or 2 (section 3). Inferred: E encodes the weight/epilogue variant. The stub table has family `q4moe` = 2 items stubs + 20 GEMM stubs: 12 with E in {1,2} (Ghidra 0x1979810 to 0x1979c30, the i4r set above) and 8 with E = 0 (0x1979c90 to 0x1979f30, the plain set above). All 20 are accounted for.

### 5.2 Non-q4moe plain path (n < 64, decode, or shapes not matching)

Only the attributable gate, obj 0x17da5fd to 0x17da61d: `cmpl $0x2, %eax; jl 0x17da6a4` (n >= 2), `cmpl 0x18fcdb4, %eax; jle 0x17da6a4` (n > FUSED_N), `cmpb $0x0, 0x1(%r14); jne 0x17da6a4` (not decode). Passing, with `HALOGEN_MOE_GPU_ROUTE` (`FUN_018e8b30`, default 1; cell 0x18fcde8, tested at obj 0x17da64a): grouped kernels `k_expert_h_grouped_warp` and `_aff<4|5,true>` (calls 0x17ddcd5, 0x17ddd51, 0x17de2b1), `k_expert_out_grouped_warp` and `_aff<5,false>` (0x17de517, 0x17de57f), `k_glu_slots` / `k_glu_rows` (0x17ddd7e, 0x17ddbf2), `k_gather_rows` (0x17dd27e, 0x17dd9a8). Otherwise (jump to 0x17da6a4) per-token kernels `k_exp_gu_w<R,K,E>` and `k_exp_gu_aff<R,K,E>` with R in {8, 16, 32} (`HALOGEN_EXP_LPR`, `FUN_018eb360`, default 16, cell 0x18fe060; compares `cmpl $0x8` at 0x17da6ff and `cmpl $0x20` at 0x17da708), `k_exp_dn_w<0|1|2>`, `k_exp_dn_w2`, `k_exp_dn_w_q8`, `k_exp_dn_aff` (calls 0x17e1861 to 0x17e1bee), `k_expert_h` / `k_expert_out` (0x17da12e, 0x17da265), `k_expert_h_q4` / `k_expert_out_q4` (0x17da865, 0x17da90e). The pairing of each of these with an exact token interval was not read.

## 6. Result table: token count to kernel family, routed experts

| expert dtype | n (tokens this forward) | first-launched GEMM family | cite |
|---|---|---|---|
| any | decode (`ctx[1]=1`, n < 9 with cache) | per-dtype decode family (sections 3, 4, 5.2) | 018fa150.c:50 |
| 23 i4r | n < 9, or `ctx[1] != 0` | `k_i4r_rows<2560>`, `k_i4r_rows<640>`, `_gu_epi`, `_fold` and the `_gl/_fl/_o/_g` set | 018d2430.c:67; obj 0x17d24e5 |
| 23 i4r | 9 <= n <= 4966, prefill | `k_ht_rot_rows`, `k_q4moe_items<64>`, `k_q4moe<A,0,64,.,1>` then `<A,2,64,.,2 or 1>` | obj 0x17d27ac, 0x17d2884, table in section 3 |
| 23 i4r | n >= 4967, prefill | same with `_items<128>` and W = 128 | obj 0x17d27ac `jg 0x17d2db4` |
| 16 ht | n < 9, or decode, or `HT_MOE_GEMM == 0` | `k_ht_gemv_*` / `k_ht_moe_gu3,dn3` decode set (U10) | 018be390.c:170-181 |
| 16 ht | n >= 9, prefill | `k_ht_moe_gu3<bits,1,DV>` + `dn3` / `dn2x` by default (mode 1); 64-row tiles for n < 4096, 128-row down tile from n >= 4096, 128-row gate-up tile from n >= 16384 | 018be390.c:170, :623-629 |
| 5 plain | 2 <= n < 64, or decode | grouped warp or per-token kernels (5.2) | obj 0x17d96bd, 0x17da5fd |
| 5 plain | n >= 64, prefill, shapes ok | `k_q4moe_items<W>`, `k_q4moe<A,0,W,.,0>` then `<A,2,W,.,0>`; W = 64 for n <= 4966, 128 for n >= 4967 (or `_BN`) | obj 0x17d96bd, 0x17dfa58, 0x17dfbd3 |
| 5 plain | n >= 64 but a layout check fails | non-q4moe path (5.2) | obj 0x17d991b |

The constant 4966 / 4967 (`0x1366` / `0x1367`) appears in both q4moe paths and is the only token-count constant that selects between q4moe kernel instances. The lower thresholds are `9` (i4r and HT prefill) and `64` (plain q4moe).

## 7. What is still open

1. `HT_MOE_V2` mode 2 (the `jne 0x17bfc7e` arm at obj 0x17bfc54) and which down kernel the default `DNX = 1` picks at which tile width were not decoded (section 4).
2. In the non-q4moe plain path (5.2) only the gate was read. The choice of `k_exp_gu_w<R,K,E>` instantiation (K in {1,2,3}, E in {0,1,4,5}) by `HALOGEN_EXP_PF` (`FUN_018eb390`, default 2) and the `FUN_0183ef20` cell at 0x18fe7c0 was not tied to a token interval.
3. The meaning of the `k_q4moe` template arguments (A = V2 mode, B = stage, E = variant), and that i4r nibble planes are consumed by the same kernel as Q4C-P, are inferred from launch pairing and from the Args carrying the slot-+8 pointers. `k_q4moe` bodies are not in `work/kernels*` and were not read.
4. Whether E is a format id or an epilogue id (plain path E=0, i4r path E=1/2) is not proven.
5. `HALOGEN_MOE_DN_ROWS` (`FUN_01940e10`) gating (`DAT_019fe920 == 2` disables G1) was not traced beyond the compare.

## Verdict: PARTIAL

Closed from code: the expert dtype binder (018897f0.c:1347-1373, :1417-1420, :1533-1558, :1659-1762: HT 16, i4r 23, plain 5, Q8S32 down, Q4A/Q5A, all others fatal); the layer-kind dispatch (obj 0x17d94a9, 0x17d9631); the i4r prefill path (n >= 9, W = 64 for n <= 4966 and 128 above, full A x W x fold kernel table); the plain Q4C-P q4moe path (n >= 64 hard-coded, same 4966 width split, `_BN` override, E=0 stub set); HT prefill tile thresholds (4096 and 16384). Remaining: items 1 to 5 above (HT mode 2, non-q4moe plain interval mapping, `k_q4moe` template meaning), none of which changes the dtype-to-family map.

## Reproduce

```bash
# listing (ai-core, read-only); obj = Ghidra - 0x100000
ssh ai-core 'sed -n "/^ 17d7510:/,/^ 17e43d0:/p" /tmp/r3dd/text.s' > moe_layer_forward.s
# handle to stub map: for each stub in analysis/function-map-kernel-stubs.tsv (addr - 0x100000), the first
# "leaq X(%rip), %rdi # 0xH" in the next 80 lines is the hipLaunchKernel handle H.
# Launches in a function: "leaq ..., %rdi # 0xH" with H in that map, or "callq 0x<stub>".
```
