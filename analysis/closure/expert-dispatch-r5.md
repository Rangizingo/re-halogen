# Closure round 5: U21 routed-expert dispatch (HT mode 2, plain Q4C-P non-q4moe path, k_q4moe template args)

Inputs: `analysis/closure/expert-dispatch-r4.md` ("R4"), `work/decomp/flash_serve/*.c` (Ghidra = ELF vaddr + 0x100000), `objdump -d` of `binaries/flash_serve` decoded from function starts on ai-core (`moe_layer_forward` obj 0x17d7510 to 0x17e43d0, `moe_forward` (HT) obj 0x17be390 to 0x17ca800), `analysis/function-map-kernel-stubs.tsv`, `kernels-disasm/codeobj_53000.s` (the `q4moe` kernels). Static only, nothing was run. Addresses written `0x17...` are ELF vaddrs (obj). "Inferred" marks anything not read directly from code.

Method: launch sites were attributed by `leaq <handle>(%rip), %rdi` before `hipLaunchKernel` (handle to stub from the first `leaq ... %rdi` inside each stub; 1737 of 1738 stubs resolved) or a direct `callq <stub>`. Branch conditions were derived by a dominator pass over the basic blocks of `moe_layer_forward` (a condition is listed for a launch only if it dominates the launch block and only one successor side reaches it), then each claim below was re-read in the listing.

## Scope

R4 section 7 open items 1 to 3: (1) `HT_MOE_V2` mode 2 and the down kernel `HT_MOE_DNX` picks per tile width; (2) which instance runs for which token count on the plain Q4C-P path when the q4moe GEMM is not taken (`k_exp_gu_w<R,K,E>`, `HALOGEN_EXP_LPR`, `HALOGEN_EXP_PF`, cell 0x18fe7c0, grouped versus per-token, `HALOGEN_MOE_GPU_ROUTE`, down kernel); (3) meaning of `k_q4moe` template args A, B, E.

## Verdict: PARTIAL (residuals 1, 2 and 4 closed in `expert-dispatch-r5b.md`; the row is CLOSED with the remaining residuals bounded as naming or untested-equality items)

Items 1, 2 and 3 are answered from code. R4 section 5 needed one structural correction: the q4moe condition lives in gate G1, not in the `$0x40` compare (section 2.1). What stays open is bounded: the library-GEMM arm internals, the `CSORT=0` counting arm, and the numeric meaning of `E` (section 6).

## 1. HT mode 2 and the down kernel by tile width (FUN_018be390, obj 0x17be390)

Cells: `HT_MOE_V2` 0x18fcaa8 (`getenv("HALOGEN_HT_MOE_V2")`, values 0 to 3, else 1, 018ca6b0.c), `HT_MOE_DV` 0x18fcab8 (0 to 2, else 1, 018ca6f0.c), `HT_MOE_DNX` 0x18fcac8 (0 to 2, else 1, 018ca730.c), `HT_MOE_M128` 0x18fcad8 (default 3, 018ca770.c), `_M128_DN` 0x18fcae8 (4096), `_M128_GU` 0x18fcaf8 (16384). Bits per projection: gate-up `(qparam >> 1) & 0x7f` stored at 0x12c(%rsp) (obj 0x17be3e3 `mov 0x44(%rax),%eax`, 0x17be7ea `shr $1,%eax`, 0x17be7f2 `and $0x7f,%eax`, 0x17be7f5 `mov %eax,0x12c(%rsp)`), down the same from `0x44(%rcx)` into 0xb4(%rsp) (0x17be7fc, 0x17be800). A kernel name below has `bits` 3 when that projection's value is 3 (`cmpl $0x3,0x12c(%rsp)` at 0x17bfca9, `cmpl $0x3,0xb4(%rsp)` at 0x17c0b69), else 4. The two projections pick their bits separately.

Tile flags (mode 1 only): obj 0x17bf02e `cmp $0x1,%eax; jne 0x17bf07e` (other modes: both tiles 64 rows, flags 0). Gate-up 128-row flag 0xe8(%rsp) = `n >= 0x18fcaf8 && (M128 & 1)` (0x17bf03c `cmp 0x13dab6(%rip),%eax`, 0x17bf04b `and %cl,%dl`). Down 128-row flag 0xd8(%rsp) = `(M128 & 2) && n >= 0x18fcae8` (0x17bf056 `test $0x2,%cl`, 0x17bfdf8 `cmp 0x13ccea(%rip),%eax; setge %al; mov %eax,0xd8(%rsp)`). n is `param_3`, stored at 0x154(%rsp) (0x17be3a3; 018be390.c:63 `local_1e24[0] = param_3`).

### 1.1 Mode switch, all modes

Switch at 0x17bfc33 to 0x17bfc54 (`test %ecx,%ecx; je 0x17bffa5`, `cmp $0x3,%ecx; je 0x17bfe2b`, `cmp $0x1,%ecx; jne 0x17bfc7e`). Mode 1 then reloads ecx from `HT_MOE_DV` (0x17bfc56 `mov 0x13ce5c(%rip),%ecx # 0x18fcab8`) and jumps to 0x17c13cf if DV, DNX or M128 is nonzero (0x17bfc5e, 0x17bfc6b, 0x17bfc78). Otherwise mode 1 falls into 0x17bfc7e. Mode 2 (the `jne 0x17bfc7e` arm, R4 open item 1) jumps straight to the same 0x17bfc7e.

| mode | gate-up (launch, handle) | down (launch, handle) |
|---|---|---|
| 0 | `k_ht_moe_gu<bits,8,WPE>` (R4 table, unchanged) | `k_ht_moe_dn<bits,8,WPE>` |
| 1, DV=DNX=M128=0 | `k_ht_moe_gu2<bits,64,0>`: bits 3 at 0x17bfdec (`lea ... # 0x18f69b8`), bits 4 at 0x17c017a (0x18f69c0) | `k_ht_moe_dn2w<bits,64,6,0>`: 0x17c0d8b (`# 0x18f69f8`), 0x17c0f9d (`# 0x18f6a00`) |
| 2 | same `k_ht_moe_gu2<bits,64,0>` (0x17bfc7e shared) | `k_ht_moe_dn2<bits,64,0>`: 0x17c0c74 (`# 0x18f6ad8`), 0x17c0e94 (`# 0x18f6ae0`) |
| 3 | `k_ht_moe_gu2w<bits,64,6,0>`: 0x17bff99 (`# 0x18f6ac8`), 0x17c0b1a (`# 0x18f6ad0`) | `k_ht_moe_dn2w<bits,64,6,0>` (same sites as mode 1) |

Down selection for modes 1 to 3 is at obj 0x17c0b2e `mov 0x13bf74(%rip),%ebx # 0x18fcaa8`, 0x17c0b60 `cmp $0x2,%ebx; jne 0x17c0c80`: mode 2 stays on the `dn2` arm (0x17c0b69 onward), every other mode takes 0x17c0c80 and its `dn2w` launches. R4 section 4 listed `dn2<bits,64,0>` under mode 1; that is mode 2 only. Mode 1 with all three knobs zero launches `dn2w`.

### 1.2 Default mode 1 (route 0x17c13cf), kernel by DV, tile width and DNX

DV is E in every name here (`HT_MOE_DV` 0, 1 or 2). Gate-up (ecx = DV): 0x17c13de `cmp $0x1,%ecx; je 0x17c1b80`, 0x17c13e7 `test %ecx,%ecx; jne 0x17c1bfc`; each of the three arms tests 0xe8: `cmpb $0x0,0xe8(%rsp)` at 0x17c13ef (DV 0), 0x17c1b80 (DV 1), 0x17c1bfc (DV 2). 0xe8 set (block 0x200 = `lea 0x200(%rax),%rdx`) launches `k_ht_moe_gu3<bits,1,E>` (calls 0x17c1461, 0x17c1bf2, 0x17c1c6e for bits 3; 0x17c41f0, 0x17c439f, 0x17c454e for bits 4). 0xe8 clear launches `k_ht_moe_gu2<bits,64,E>` (calls 0x17c271a, 0x17c2788, 0x17c27f6 for bits 3; 0x17c422d, 0x17c43dc, 0x17c458b for bits 4).

Down (same E as gate-up): 0x17c4232 `cmpb $0x0,0xd8(%rsp); je 0x17c42ab` (E=0 chain; E=1 chain starts 0x17c43e1; E=2 chain 0x17c4590). 0xd8 set (block 0x200): `k_ht_moe_dn3<bits,1,E>` (0x17c42a1, 0x17c4450, 0x17c45ff for bits 3; 0x17c435f, 0x17c450e, 0x17c46bd for bits 4), no DNX test. 0xd8 clear (block 0x100) loads DNX (0x17c42ab `mov 0x138817(%rip),%ebx # 0x18fcac8`, 0x17c42dd `cmp $0x2,%ebx; je 0x17c507b`, 0x17c42e6 `cmp $0x1,%ebx; jne 0x17c5101`):

| DNX | down kernel at the 64-row down tile | sites for E=0, 1, 2 (bits 3) |
|---|---|---|
| 1 (default) | `k_ht_moe_dn2x<bits,64,6,E>` | 0x17c432a, 0x17c44d9, 0x17c4688 |
| 2 | `k_ht_moe_dn2x<bits,64,1,E>` | 0x17c50b4, 0x17c50f7, 0x17c51b6 |
| 0 | `k_ht_moe_dn2w<bits,64,6,E>` | 0x17c5135, 0x17c5173, 0x17c51f4 |

With defaults (V2=1, DV=1, DNX=1, M128=3, thresholds 16384 and 4096), per n and per projection bits:

| n (prefill, n >= 9) | gate-up | down |
|---|---|---|
| 9 to 4095 | `k_ht_moe_gu2<bits,64,1>` | `k_ht_moe_dn2x<bits,64,6,1>` |
| 4096 to 16383 | `k_ht_moe_gu2<bits,64,1>` | `k_ht_moe_dn3<bits,1,1>` |
| 16384 and up | `k_ht_moe_gu3<bits,1,1>` | `k_ht_moe_dn3<bits,1,1>` |

(Thresholds 0x18fcaf8 and 0x18fcae8, R4 section 4; n >= 9 and decode gate at 018be390.c:170.) R4 had gate-up as `gu3` for all n under defaults; `gu3` runs only from n = 16384.

## 2. Plain Q4C-P experts (dtype 5, qparam 2, flags 0x720 to 0x723 clear): moe_layer_forward

Defaults used below: `HALOGEN_FLASH_MOE_V2` 1, `_MINN` 64, `_FUSED_N` 0, `_BN` 0, `HALOGEN_LIN_WAVE` 1, `HALOGEN_EXP_LPR` 16, `HALOGEN_EXP_PF` 2, `HALOGEN_LQ4W_SH` 0, `HALOGEN_EXP_DN_R2` 0, `HALOGEN_DEC_PROBE` 0, `HALOGEN_FLASH_MOE_GEMM` 8, `_GEMM_DN` 8, `HALOGEN_MOE_H_ORDER` 1, `HALOGEN_FLASH_MOE_BATCH_GU` 0, `HALOGEN_MOE_DN_ROWS` 1, `HALOGEN_MOE_GPU_ROUTE` 1, `HALOGEN_FLASH_MOE_V2_SUM` 1, `HALOGEN_MOE_CSORT` 1. Each is a `getenv` plus a default return in 018e43e0.c, 018e4420.c, 018e4460.c, 018e44d0.c, 0193ef20.c, 018eb360.c, 018eb390.c, 019406c0.c, 01946870.c, 019468a0.c, 018e8fe0.c, 018e9010.c, 018e9040.c, 018e90c0.c, 01940e10.c, 018e8b30.c, 018e8b00.c, 018e8b60.c.

### 2.1 Where q4moe is chosen (correction to R4 section 5)

Gate G1 (obj 0x17d7593 to 0x17d7652) does not only "choose route and stream options". Every failing test jumps to obj 0x17d76ba, which does `movl $0x0,0x160(%rsp)` (0x17d76ba), `xor %edx,%edx` (0x17d76c5) and `mov %dl,0xdf(%rsp)` (0x17d76c7). 0x160(%rsp) is later ANDed into the q4moe condition (`mov 0x160(%rsp),%eax; and %r14b,%al; cmp $0x1,%al; jne 0x17d9936`, obj 0x17d987a to 0x17d9894), so a failing G1 test removes the q4moe route. The tests, in order:

- `HALOGEN_FLASH_MOE_V2 != 0` (0x17d7593 `cmpl $0x0,0x12574e(%rip) # 0x18fcce8; je 0x17d76ba`)
- `n >= 2` (0x17d75b7)
- `n > FUSED_N` (0x17d75c0 `cmp 0x125762(%rip),%eax # 0x18fcd28; jle`)
- prefill (0x17d75cc `cmpb $0x0,0x1(%rdi); jne`)
- `n >= MINN` (0x17d75d6 `cmp 0x12571c(%rip),%eax # 0x18fccf8; jl`)
- `ctx+0x378 == 0` (0x17d75e2)
- the byte at 0x18fe3e0 clear (0x17d75ff `cmpb $0x0,0x126dda(%rip)`; a global profile-table flag zeroed by FUN_0190c8b0, 0190c8b0.c:6 `DAT_019fe3e0 = 0`)
- `HALOGEN_MOE_DN_ROWS != 2` (0x17d761b)
- layer flags 0x723, 0x721, 0x722, 0x720 all zero (0x17d7628 to 0x17d764a)
- both tensors dtype 5 and qparam 2 (0x17d765b to 0x17d7674), gate-up slot +0 non-null (0x17d767a)
- then 0x160 = (down slot +0 non-null) (0x17d7683 `test %rsi,%rsi; setne %dl`, 0x17d7689 `mov %edx,0x160(%rsp)`).

So the effective threshold for q4moe is `n >= max(2, MINN, FUSED_N + 1)` in prefill; MINN default 64 (`FUN_018e4420`: a set value is clamped to >= 9, unset gives 0x40). The hard-coded `$0x40` at 0x17d96bd (`cmp $0x40,%edx; setb %dl; or 0x1(%r14),%dl; jne 0x17d9922`) is reached only when both slot +8 pointers are null (0x17d9692 `test %r13,%r13; jne 0x17d9873` and 0x17d969b `test %rsi,%rsi; jne 0x17d9873` skip it). It gates the one-time device staging copy, not the route.

After G1 the route needs device pointers and a layout match:

- slot +8 is the pinned device pointer. The checkpoint loader stores host `local_118 = piVar6 + off` and device `lStack_110 = off + local_140`, zero when not pinned (01996d90.c:181-184); `local_140` comes from `hipHostGetDevicePointer` (01996d90.c:334). In the pinned case r13 and rsi are the two slot +8 pointers (0x17d9687 `mov 0x8(%rax),%r13`, 0x17d968b `mov 0x8(%rcx),%rsi`) and the code jumps to 0x17d9873.
- 0x17d9876 `test %r13,%r13; setne %r14b`, 0x17d9887 `setne %bl`. Layout check through `call 0x17e8950` (0x17d98c1, 0x17d98d4), alignment tests 0x17d98ed, 0x17d9903, 0x17d9913, shapes `cmpq $0xa0000,0x100(%rsp)`, `$0xa00,0x108`, `$0x140000,0x1f0`, `$0x280,0x1f8` (0x17df902 to 0x17df938). Pass reaches 0x17df94a (q4moe). Any failure jumps to 0x17d991b (`mov $0x1,%bl; mov $0x1,%r14b`), which keeps the pinned pointers valid and continues at 0x17d9936.
- 0x17d994d `and %bl,%r14b; cmp $0x1,%r14b; jne 0x17d99d1`: both device pointers present (fallthrough, "F path" below) versus not (jump, "T path", host-pointer generic kernels, section 2.4).

Whether the shipped config pins the model file (slot +8 non-null) is not shown by static code: the loader registers the mapping with `hipHostRegister` unless its `param_3` is nonzero ("mapped ..., NOT pinned (Pin::None)", 01996d90.c:79-84, caller not traced). **Unverified**: pinned in the shipped config.

### 2.2 F path (device pointers present, q4moe not taken): top-level split by n

Gate at obj 0x17da5f6 to 0x17da617: `mov 0xa4(%rsp),%eax` (n), `cmp $0x2,%eax; jl 0x17da6a4`, `cmp 0x1227a8(%rip),%eax # 0x18fcdb4; jle 0x17da6a4`, `cmpb $0x0,0x1(%r14); jne 0x17da6a4`. Pass (n >= 2, n > FUSED_N, prefill): grouped path 2.3. Fail (n = 1, n <= FUSED_N, or decode): per-token path 2.2.1.

`HALOGEN_MOE_GPU_ROUTE` is not part of this split. `cmpl $0x0,0x1227b3(%rip) # 0x18fcde8; je 0x17daa60` (0x17da62e to 0x17da64a) skips the grouping-table arena call and both arms rejoin at 0x17dab0d. GPU_ROUTE nonzero with DN_ROWS != 2 and no 0x378 hook calls `FUN_018d6f60` at 0x17daab3 to allocate device grouping tables (018d6f60.c, arena sizes 0x800, 0x804, ...). No grouped or per-token kernel launch has a GPU_ROUTE test among its dominating conditions.

#### 2.2.1 Per-token path (n = 1, or decode with n < 9)

Cell 0x18fe7c0 = `HALOGEN_LIN_WAVE` (init obj 0x17e3377 `call 0x183ef20` stored to 0x18fe7c0; 0193ef20.c `getenv("HALOGEN_LIN_WAVE")`, default `return 1`; Ghidra name FUN_0193ef20, R4 wrote the obj address). Test at 0x17da6b3 `cmpl $0x0,0x124106(%rip) # 0x18fe7c0; je 0x17da7fb`.

| LIN_WAVE | kernels | cite |
|---|---|---|
| 0 | `k_expert_h_q4`, `k_expert_out_q4`, `k_moe_sum` | calls 0x17da865, 0x17da90e, 0x17da975. Flagged experts are fatal: 0x17da86a to 0x17da88c jump to 0x17e32ed, string at file 0x29b4f "layer %d: Q8S32 / Q4A / Q5A experts have no HALOGEN_LIN_WAVE=0 arm" |
| nonzero (default) | gate-up `k_exp_gu_w<R,K,E>` plus a down kernel, below | 0x17da6de, 0x17da6e4 |

Plain gate-up (flag 0x721 clear: `cmpb $0x1,0x721(%rcx); jne 0x17da9b3` at 0x17da6f2): `eax` = cell 0x18fe060 = `HALOGEN_EXP_LPR` (init 0x17e37f0 `call 0x17eb360`, 018eb360.c, default 0x10), `ebp` = cell 0x18fe070 = `HALOGEN_EXP_PF` (init 0x17e3820 `call 0x17eb390`, 018eb390.c, default 2). Neither value is clamped.

- R: 0x17da9b3 `cmp $0x8,%eax; je 0x17db5df` gives R=8; 0x17da9bc `cmp $0x20,%eax; jne 0x17db764` gives R=16 for every value except 8 and 32; 32 stays here as R=32.
- K: `cmp $0x3,%ebp; jl` (0x17da9c5, 0x17db5df, 0x17db764): K=3 when PF >= 3. Below 3, `cmp $0x2,%ebp` (0x17dfd4c, 0x17dfed1, 0x17e0056): equal gives K=2, else K=1. PF=2 (default) gives K=2.
- E: `call 0x17b2900` (0x17da9ce, 0x17db5e8, 0x17dfd10, ...) is `FUN_018b2900` returning `HALOGEN_LQ4W_SH` (018b2900.c, 019406c0.c, default 0); `test %ebx,%ebx; je` (0x17daa0c): zero gives E=0, nonzero gives E=1.
- Grid: `imul $0x320,0xf0(%rsp),%eax` (0x17da721): 800 blocks of 256 threads per token; n only scales the grid.

Instance call sites (default instance is `k_exp_gu_w<16, 2, 0>`, 0x17e178a):

| R | K=3, E=1 | K=3, E=0 | K=2, E=1 | K=2, E=0 | K=1, E=1 | K=1, E=0 |
|---|---|---|---|---|---|---|
| 8 | 0x17db670 | 0x17e066d | 0x17dff26 | 0x17e16ee | 0x17e043a | 0x17e173c |
| 16 | 0x17db7f5 | 0x17e06bb | 0x17e00ab | 0x17e178a | 0x17e05d1 | 0x17e17d1 |
| 32 | 0x17daa56 | 0x17e061f | 0x17dfda1 | 0x17e1652 | 0x17e02a3 | 0x17e16a0 |

Affine gate-up (flag 0x721 set, Q4A or Q5A): `k_exp_gu_aff<R,K,B>` with the same R and K rules; B is 5 when the value at 0x140(%rsp) equals 5 (`cmp $0x5,%ebx` at 0x17dcfdc, 0x17db53a, 0x17dfc6b, 0x17dfdf0, 0x17dff75), else 4. Inferred: bit width (Q5A versus Q4A); the stack slot is read at 0x17da71a and its writer was not located.

Down kernel (after gate-up, common tail 0x17e17d6): `cmpb $0x1,0x720(%rax); jne 0x17e186b` (0x17e17de): Q8S32 gives `k_exp_dn_w_q8` (call 0x17e1861). 0x17e186b `cmpb $0x1,0x722(%rax)`: affine gives `k_exp_dn_aff` (0x17e1959). Plain: `call 0x17ecf70` (0x17e1963) is `FUN_018ecf70` returning `HALOGEN_EXP_DN_R2` (01946870.c, default 0); nonzero gives `k_exp_dn_w2` (0x17e1a0d, grid x 0xa0 at 0x17e197b). Else `call 0x17ed0d0` (0x17e1a17) is `FUN_018ed0d0` returning `HALOGEN_DEC_PROBE` (019468a0.c, default 0): value 1 gives `k_exp_dn_w<1>` (0x17e1ac2), value 2 gives `k_exp_dn_w<2>` (0x17e1b7a; `cmp $0x2,%ebx` at 0x17e1b04), anything else gives `k_exp_dn_w<0>` (0x17e1bee). Grid x is 0x140 (`or $0x140,%rdi`, e.g. 0x17e1a30) with n in the high half. No separate sum launch follows (0x17e1bf3 goes to the function tail 0x17e1c11).

Default per-token instances (decode, and prefill n = 1): `k_exp_gu_w<16, 2, 0>` (0x17e178a) then `k_exp_dn_w<0>` (0x17e1bee).

### 2.3 Grouped path (n >= 2, prefill, n > FUSED_N, q4moe not taken)

The split is by the mean number of slots per active expert, not by n directly. Host counting sort over 512 experts (`HALOGEN_MOE_CSORT` default 1: obj 0x17db0b0 to 0x17db0c0 `cmp $0x200,%rdx` / `incl (%r12,%rdx,4)`, and 0x17db10f to 0x17db143 pushes every expert with a nonzero count onto the vector at 0x18fce38). A = that vector's size, `r12d` (0x17dc3d0 `mov 0x120a69(%rip),%r12 # 0x18fce40`, 0x17dc3d7 `sub 0x120a5a(%rip),%r12 # 0x18fce38`, 0x17dc3e6 `shr $0x2,%r12`, stored to 0x358(%rsp) at 0x17dc914, reloaded at 0x17dccbf). S = 10n (0xa8(%rsp), 0x17d94da, R4 section 0 item 4). avg = S / A, integer (`idiv %r12d`, 0x17dcde1).

Gate-up arm:

| condition | gate-up kernels | cite |
|---|---|---|
| BLAS handle cell 0x18fc898 != 0, `FLASH_MOE_GEMM` (0x18fe020, default 8) > 0, avg >= GEMM | GEMM arm: dequant (`FUN_018e95d0` call 0x17dd5fa for dtype 5, `FUN_018e9180` call 0x17dd5a8 for affine), `k_gather_rows` (0x17dd27e, 0x17dd9a8), library GEMM (`call 0x18e7e60`, 0x17dd74c), `k_glu_rows` / `k_glu_slots` (0x17dd399, 0x17dd8af, 0x17ddbf2; chosen by H_ORDER, BATCH_GU, 0xe8) | 0x17dcdb9 to 0x17dcde6, `test %bl,%bl; je 0x17dcf92` (0x17dce9e) |
| otherwise, S >= 4*A (avg >= 4) | `k_expert_h_grouped_warp` (call 0x17ddd51); affine `k_expert_h_grouped_warp_aff<5,true>` (0x17ddcd5) or `<4,true>` (0x17de2b1) | 0x17dcf92 `lea 0x0(,%r12,4),%eax`; 0x17dcfcf and 0x17dd0c8 `cmp %eax,0xa8(%rsp)` then `jge` |
| otherwise (avg < 4) | `k_expert_h_warp` (call 0x17dd144); affine `k_expert_h_warp_aff<5,true>` (0x17dd0a5), `<4,true>` (0x17de20a) | same |

The dequant instance inside `FUN_018e95d0` is picked by `HALOGEN_FLASH_DEQ_V8`, `HALOGEN_DEQ_KERN`, `HALOGEN_DEQ_PROBE`, `HALOGEN_DEQ_CB` (01945430.c, 019466e0.c, 01946730.c, 019467a0.c), not by n; R3 (`dequant-dispatch-r3.md`) lists the `k_deq_q4cp*` instances launched at 0x17e98f6 to 0x17ea5b9.

Down arm, flag 0x248(%rsp): set to 1 when `FLASH_MOE_GEMM_DN` (0x18fe030, default 8) > 0 and avg >= GEMM_DN (0x17dce0c to 0x17dce2d), otherwise to the Q8S32 flag `0x720(%rax)` (0x17dce37). Test: 0x17de394 `cmpb $0x0,0x248(%rsp); je 0x17de3fe`.

| 0x248 | down kernels | cite |
|---|---|---|
| set | dequant (`k_deq_q8s32` 0x17dec14 for Q8S32, else the q4cp or affine dequant), `k_gather_slots_h` (0x17dee70), library GEMM, `k_scatter_slots_dn` (0x17df0d4), `k_slot_sum` (0x17df31d, 0x17df53e), then `k_slot_sum_rows` (0x17df3bb) / `k_moe_sum` (0x17df58d) | 0x17de6c6 to 0x17df2ad |
| clear, flag 0x722 clear | `k_expert_out_grouped_warp` (call 0x17de57f) | 0x17de3fe `cmp $0x1,%bl`, bl = `0x722` (0x17de406) |
| clear, flag 0x722 set | `k_expert_out_grouped_warp_aff<5,false>` (0x17de517) | same |

Uniform-routing estimate (an estimate, not measured: 512 experts, 10 independent picks per token, `A = 512 * (1 - (1 - 10/512)^n)`, computed with python): avg is 1.01 at n = 2, 1.74 at n = 64, 3.98 at n = 200, 4.08 at n = 205, 8.01 at n = 410. With random routing `k_expert_h_warp` covers n up to about 200, `k_expert_h_grouped_warp` about 205 to 410, the GEMM arm above that. Concentrated routing (many tokens on the same experts) raises avg at smaller n.

### 2.4 T path (device pointers absent: not pinned, and n < 64 or decode)

0x17d99d1 onward: `k_expert_h` (call 0x17da12e), `k_expert_out` (0x17da265), `k_moe_sum` (0x17da39d), weights from slot +0 (host pointers). When both slot +8 are null and n >= 64 in prefill, 0x17d974a to 0x17d97c2 stage both tensors into device buffers (`call 0x17e88a0` x2, `call 0x179a920` x2) and r13/rsi become the staged pointers, so the F path applies from then on.

### 2.5 q4moe launch details (R4 section 5.1, restated, plus the tail)

W = 64 + 64 * (n >= 4967) unless `_BN` is 64 or 128 (0x17dfa58 `cmpl $0x1367,0xf0(%rsp)`, 0x17dfa6b to 0x17dfa71). A = 1 when `HALOGEN_FLASH_MOE_V2` is 1 (default), A = 2 for every other value that reaches here (0x17e0837 `cmp $0x1,%r13d; jne 0x17e0a63`; value 0 is already excluded by G1, so A = 2 means V2 = 2). Kernels: `k_q4moe_items<W>` (0x17dfc1c, 0x17e0700), `k_q4moe<A,0,W,Tune<0,128>,0>` then `<A,2,W,Tune<0,128>,0>` (calls in R4 section 5.1). Tail: `cmpl $0x0,0x11ad7f(%rip) # 0x18fcdd8` (0x17e2052; the cell is `HALOGEN_FLASH_MOE_V2_SUM`, 018e8b00.c, default 1): nonzero launches `k_slot_moe_sum_rows` (call 0x17e215f); zero launches `k_slot_sum_rows` (0x17e20a9) then `k_moe_sum` (0x17e21a8).

## 3. k_q4moe template arguments (kernels-disasm/codeobj_53000.s)

Symbol `_ZN5q4moe7k_q4moeILi<A>ELi<B>ELi<W>ENS_4TuneILi0ELi128EEELi<E>EEEvNS_4ArgsE`. All 20 GEMM instances and `k_q4moe_items<64>`, `<128>` are in `codeobj_53000.s` (symbol lines 1995557 to 2377097). Per-instance counts come from awk over each symbol's line range (`v_wmma_f32_16x16x16_bf16`, `v_exp`, `v_rcp`, `v_cvt`, `v_fma_mix`, `ds_*`, `global_*`, `s_barrier`, and the kernarg `s_load` offsets).

| arg | values | what changes in the object code | verdict |
|---|---|---|---|
| W | 64, 128 | `v_wmma` 160 versus 576 in every A, B, E combination; total instructions 2.8x to 3.2x | GEMM row tile (matches `k_q4moe_items<W>`) |
| B | 0, 2 | B=0 has `v_exp_f32` x80 and `v_rcp_f32` x80 at W=64 (x288 each at W=128); B=2 has none (e.g. line 2272289 `v_exp_f32`). B=0 is launched first (gate-up), B=2 second (down) | B=0 gate-up with the SiLU (sigmoid) epilogue; B=2 down projection, no activation |
| A | 1, 2 | same WMMA flavor in both; A=1 has `v_fma_mix_f32` x256, `v_bfe_u32` x500, `v_add3_u32` x500 (B0 E0 W64); A=2 has no `v_fma_mix`, adds `v_cvt_pk_rtz_f16_f32` x80 and `v_mul_i32_i24` x128, and roughly halves `v_bfe`/`v_add3` (244 and 246) | A picks the dequant arithmetic; A = `HALOGEN_FLASH_MOE_V2` (1 or 2); both variants exist for E = 0, 1, 2 |
| E | 0, 1, 2 | table below | E=0 Q4C-P layout; E=1, E=2 i4r layout (inferred) |

E detail (A=1, W=64; other A and W have the same shape):

| instance | kernarg slots read (Args) | small global loads | other deltas |
|---|---|---|---|
| B0 E0 | 0x00, 0x20, 0x70 (b128), 0x80, 0x88 | `global_load_b32` x10, `global_load_b64` x8 | `ds_bpermute_b32` x20, `global_store_b128` x7, barriers 8, 7774 instructions |
| B0 E1 | 0x00, 0x20, 0x60 (b256, i.e. two more pointers than E0's 0x70 b128), 0x80, 0x88 | `global_load_u16` x4 and `global_load_d16_b16` x4 (offset:2 pairs), `global_load_b64` x60, `global_load_b32` x10 | `ds_store_2addr_b32` x80, `global_store_b64` x20, barriers 28, 12125 instructions |
| B2 E0 | 0x00, 0x20, 0x70 (b64), 0x80, 0x88 | `global_load_b64` x8 | `ds_bpermute_b32` x40, `global_store_b128` x17, 5981 instructions |
| B2 E2 | same as B2 E0 | `global_load_u16` x4 and `d16_b16` x4, no `global_load_b64` | `ds_bpermute_b32` x40, `global_store_b128` x17, fp-class instructions 118 versus 310 in E0, 5237 instructions |
| B2 E1 | adds 0x60 (b64, one pointer) to B2 E0's slots | `global_load_u16` x4 and `d16_b16` x4, `global_load_b64` x40 | `ds_store_2addr_b32` x80, `global_store_b64` x40, barriers 28, 9070 instructions |

Reading (inferred from the deltas plus the launch pairing in R4 section 3, where the i4r path launches E=1 for gate-up and E=2 or E=1 for down depending on `HALOGEN_I4R_DN_FOLD`, and the plain path launches E=0):

- E=0 is the Q4C-P path: no 16-bit scale loads.
- E=1 and E=2 read 16-bit values in pairs (`u16` plus `d16_b16` at `offset:2`), which fits the i4r layout (one fp16 scale per 128 along K, `closure/ht-decode.md`).
- E=1 also reads another kernarg block (0x60) and stores through `ds_store_2addr_b32` and 64-bit global stores: an in-kernel output transform. E=2 (down only) has no extra kernarg and stores like E=0: no transform in the kernel. That matches the host tail: E=2 is followed by `k_i4r_fold_rows`, E=1 by `k_slot_moe_sum_rows` (R4 section 3).
- `Tune<0,128>` is the same type in all 20 instances; no instance varies it.

What the 0x60 kernarg block points to was not decoded; see section 6.

## 4. Result table: n range and env defaults to launched instances (plain Q4C-P, dtype 5, qparam 2)

Defaults as listed in section 2; model pinned (slot +8 non-null), layout check passes, 10 slots per token.

| path | n / mode | kernels launched (first to last) |
|---|---|---|
| decode (`ctx[1]` = 1) | any n < 9 | `k_exp_gu_w<16,2,0>`, `k_exp_dn_w<0>` (2.2.1) |
| prefill | n = 1 | same |
| prefill | 2 <= n < 64, avg < 4 | `k_expert_h_warp`, `k_expert_out_grouped_warp` (2.3) |
| prefill | 2 <= n < 64, 4 <= avg < 8 | `k_expert_h_grouped_warp`, `k_expert_out_grouped_warp` |
| prefill | 2 <= n < 64, avg >= 8 (concentrated routing only) | dequant + `k_gather_rows` + GEMM + `k_glu_*`; down dequant + `k_gather_slots_h` + GEMM + `k_scatter_slots_dn` + `k_slot_sum` |
| prefill | 64 <= n <= 4966 | `k_q4moe_items<64>`, `k_q4moe<1,0,64,.,0>`, `k_q4moe<1,2,64,.,0>`, `k_slot_moe_sum_rows` |
| prefill | n >= 4967 | `k_q4moe_items<128>`, `k_q4moe<1,0,128,.,0>`, `k_q4moe<1,2,128,.,0>`, `k_slot_moe_sum_rows` |
| prefill, layout check fails | n >= 2 | the three grouped rows above; n = 1 per-token |
| env: `HALOGEN_FLASH_MOE_V2=2` | prefill n >= MINN | A = 2 instances; `=0` removes q4moe |
| env: `_MINN=m` (m >= 9) | pinned | q4moe from n >= m; unpinned staging still needs n >= 64 |
| env: `_BN=64` or `128` | q4moe | forces W |
| env: `HALOGEN_LIN_WAVE=0` | per-token | `k_expert_h_q4`, `k_expert_out_q4`, `k_moe_sum` |
| env: `HALOGEN_EXP_LPR` 8 or 32, `HALOGEN_EXP_PF` >= 3 or <= 1, `HALOGEN_LQ4W_SH` != 0 | per-token | R, K, E per the 18-cell table (2.2.1) |
| not pinned (slot +8 null) | n < 64 or decode | `k_expert_h`, `k_expert_out`, `k_moe_sum` (2.4) |
| not pinned | n >= 64 prefill | staging copy, then the pinned rows |

HT (dtype 16) rows are in section 1.2; i4r (dtype 23) rows in R4 section 6.

## 5. Corrections to R4

1. HT `HT_MOE_V2` mode 1 (all knobs zero) and mode 3 use `k_ht_moe_dn2w`, not `dn2`; `dn2<bits,64,0>` is mode 2 only (section 1.1).
2. HT defaults: gate-up is `gu2<bits,64,1>` below n = 16384 and `gu3<bits,1,1>` from n = 16384; down is `dn2x<bits,64,6,1>` below n = 4096 and `dn3<bits,1,1>` from n = 4096 (section 1.2).
3. Plain path: the q4moe route is gated by G1 (V2, `n >= MINN`, prefill, ...), which clears 0x160(%rsp) on failure. The `$0x40` compare at 0x17d96bd gates only staging when slot +8 is null. R4 section 5 said G1 "does not itself select the GEMM"; it does, through 0x160.
4. Non-q4moe plain path: the per-token versus grouped split is the gate at 0x17da5f6 (n >= 2, n > FUSED_N, prefill), not `HALOGEN_MOE_GPU_ROUTE`. Inside the grouped path the arm is chosen by slots per active expert (4 and 8 by default), not by n.
5. `k_exp_gu_w<R,K,E>`: E is `HALOGEN_LQ4W_SH` (zero or nonzero), K is `HALOGEN_EXP_PF` (>= 3, 2, else 1), R is `HALOGEN_EXP_LPR` (8, 32, else 16). The down kernel is chosen by `HALOGEN_EXP_DN_R2` and `HALOGEN_DEC_PROBE`.

## 6. Residuals

1. Library GEMM arm of the grouped path (`call 0x18e7e60`, obj 0x17dd74c, and the down counterpart): its device kernels were not tied to this call site. The dequant instance inside `FUN_018e95d0` depends on the env cells named in 2.3 (R3 lists the launches).
2. The `HALOGEN_MOE_CSORT=0` arm (0x17db352) and the active-expert count it produces were not read; 2.3 holds for the default.
3. The source of the stack slot 0x140(%rsp) (`k_exp_gu_aff` third argument 4 or 5) was not located; "bit width" is inferred. The affine kernels are outside plain Q4C-P.
4. `E` of `k_q4moe` is characterised by instruction and kernarg deltas only; what the 0x60 kernarg block holds is unproven (the i4r side tensors at layer+0x750 to +0x768, R4 section 1, are the candidates). Next test: decode the `Args` fill in `FUN_018d2430` around the `k_q4moe` launches (obj 0x17d303e, 0x17d3198) and compare slot 0x60 with layer+0x758 and +0x760.
5. Whether the shipped container pins the checkpoint (slot +8 non-null, so the F path and q4moe are live at n < 64 and at n >= 64 respectively) is a runtime fact. Test: the startup line "checkpoint: registered ... GB" versus "NOT pinned (Pin::None)" (01996d90.c:79, :344), needs a run on the analysis host; not run here.
6. `HALOGEN_DEC_PROBE` reads like a diagnostic knob, but it selects the template argument of `k_exp_dn_w` on the default decode path; its intended meaning is unproven.
7. The HT item builder loop (018be390.c:825-863, `bVar30 = DAT_019fcac8 != 0`) also reads DNX; its effect on the `HtItem` list was not decoded (kernel selection is unaffected).

## Reproduce

```bash
# listings (ai-core, read-only); decode from function starts
ssh ai-core 'cd ~/re-halogen && objdump -d --no-show-raw-insn --start-address=0x17d7510 --stop-address=0x17e43d0 binaries/flash_serve > /tmp/r5/mlf.s'
ssh ai-core 'cd ~/re-halogen && objdump -d --no-show-raw-insn --start-address=0x17be390 --stop-address=0x17ca800 binaries/flash_serve > /tmp/r5/ht.s'
# handle to stub map: for each stub in analysis/function-map-kernel-stubs.tsv (addr - 0x100000) the first
# "leaq X(%rip), %rdi # 0xH" in its body is the hipLaunchKernel handle H (1737 of 1738 resolved).
# q4moe stats: symbol line ranges in kernels-disasm/codeobj_53000.s (grep -n '^[0-9a-f]* <.*q4moe'), awk class counts per range.
```
