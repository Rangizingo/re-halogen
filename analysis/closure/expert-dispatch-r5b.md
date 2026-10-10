# Closure round 5b: U21 residuals 1, 2 and 4 of expert-dispatch-r5

Inputs: `analysis/closure/expert-dispatch-r5.md` ("R5", sections 2.3, 3, 6), `binaries/flash_serve` decoded with `objdump -d` from function starts on ai-core (`moe_layer_forward` obj 0x17d7510, `FUN_018d2430` obj 0x17d2430, the hipBLASLt wrapper TU obj 0x18e5000 to 0x18e9400), `work/decomp/flash_serve/*.c` (Ghidra = obj + 0x100000), `analysis/function-map-kernel-stubs.tsv`, `kernels-disasm/codeobj_53000.s`. Static only, nothing was run. Addresses `0x17...`/`0x18...` are ELF vaddrs (obj). "Inferred" marks anything not read directly from code.

## Scope

R5 section 6 residuals 1, 2 and 4:

1. Which device kernels the library GEMM (`call 0x18e7e60`, obj 0x17dd74c, and the down counterpart) launches, for which shapes.
2. What the `HALOGEN_MOE_CSORT=0` arm (obj 0x17db352) computes and which kernels it launches; default of `HALOGEN_MOE_CSORT` and its getter.
3. What `k_q4moe` kernarg slot 0x60 holds for E=1 and E=2.

## Verdict: CLOSED

All three are answered from code. Residual 1 is closed with a bound: the kernels are Tensile kernels inside `libhipblaslt.so.1`, chosen at run time, so their names are not in `flash_serve` and cannot be named statically (section 1.5).

| residual | verdict | one line |
|---|---|---|
| 1 library GEMM arm | CLOSED (bounded) | hipBLASLt `hipblasLtMatmul` wrapper; launches zero `flash_serve` stubs; shapes M x 1280 x 2560 (gate-up) and M x 2560 x 640 (down), bf16 in and out |
| 2 CSORT=0 arm | CLOSED | host-only `std::stable_sort` of slot indices by expert id; same outputs as the counting sort; launches nothing; default 1 |
| 4 slot 0x60 | CLOSED | gate-up E=1: `layer+0x758` (gate_up svh), next slot `layer+0x760` (down suh); down E=1: `layer+0x768` (down svh); down E=2: filled with `layer+0x768` but never read |

## 1. Residual 1: the library GEMM arm

### 1.1 Call sites and shapes

`0x18e7e60` is not a GEMM body. It is a thin wrapper:

```
18e7e75: mov 0x30(%rsp),%rcx     ; 7th arg (K)
18e7e7a: mov %r8,%rsi            ; arg5 (M)
18e7e7d: mov %r9,%rdx            ; arg6 (N)
18e7e80: xor %r8d,%r8d           ; mode 0
18e7e83: call 18e6240            ; plan lookup/build
...
18e7ec1: jmp 18e7b30             ; execute
```

Siblings `0x18e7ed0` (`mov $0x1,%r8d`, mode 1) and `0x18e7f40` (`mov $0x2,%r8d`, mode 2) are the same wrapper with other modes. In `moe_layer_forward` the three entry points are called at obj 0x17dd74c, 0x17def1a, 0x17def96 (mode 0), 0x17d7cbe (mode 1) and 0x17d7837 (mode 2).

Args at the expert call sites (rdi = `*0x18fc898`, the hipBLASLt context):

| site | role | M | N | K | cite |
|---|---|---|---|---|---|
| 0x17dd74c | gate-up, per expert chunk | `r15` (rows of the chunk) | `0x500` = 1280 = 2 x 640 | `0xa00` = 2560 (stack arg) | `17dd740: mov $0x500,%r9d`, `17dd738: movq $0xa00,(%rsp)`, `17dd749: mov %r15,%r8` |
| 0x17def1a, 0x17def96 | down | `rbx` or `movslq %ebx,%r8` | `0xa00` = 2560 | `0x280` = 640 | `17def08: mov $0xa00,%r9d`, `17def00: movq $0x280,(%rsp)` |
| 0x17d7cbe | mode 1 (f32 operands), router side | | `0x17d7cb8: mov $0x200,%r9d` = 512 | `0xa00` | `17d7cb0: movq $0xa00,(%rsp)` |
| 0x17d7837 | mode 2 (bf16 in, f32 out), router side | | 512 | `0x1400` = 5120 | `17d7829: movq $0x1400,(%rsp)`, `17d7831: mov $0x200,%r9d` |

The last two rows are outside the expert path and listed only to place modes 1 and 2. The matching dequant call in the same arm confirms the gate-up operand: `17dd5dc: mov $0x500,%edx` and `17dd5e1: mov $0xa00,%ecx` before `call 17e95d0` (a 1280 x 2560 weight tile).

### 1.2 What the wrapper builds (obj 0x18e6240)

```
18e634b: call hipblasLtMatmulDescCreate     ; esi = 2 (compute type), edx = 0 (scale type)
18e6358: movl $0x70,0x5c(%rsp)  ; TRANSA = 112
18e6360: movl $0x6f,0x58(%rsp)  ; TRANSB = 111
18e63d5/18e63f1/18e6411: hipblasLtMatrixLayoutCreate x3
```

Layouts (register moves at 0x18e63c5 to 0x18e6411): A = (K rows, N cols, ld K), B = (K rows, M cols, ld K), C = (N rows, M cols, ld N). Types: mode 0 gives all three `0xe`; mode 1 gives `0` for all three; mode 2 gives A and B `0xe`, C `0` (`18e63a6: cmp $0x1,%ebx`, `cmove %r13d,%ebp`; `test %ebx,%ebx; cmove %eax,%r13d`). Inferred from the public hipBLASLt enums: `0x70`/`0x6f` are `HIPBLAS_OP_T`/`HIPBLAS_OP_N`, `0xe` is `HIP_R_16BF`, `0` is `HIP_R_32F`, compute type 2 is fp32. So the expert GEMM is C(N x M) = op_T(A)(N x K) x B(K x M), bf16 in and out, fp32 accumulate.

Evidence the library is hipBLASLt and not a bundled kernel: `readelf -d binaries/flash_serve` lists `NEEDED libhipblaslt.so.1`; the wrapper imports `hipblasLtMatmul`, `hipblasLtMatmulAlgoGetHeuristic` and `hipblaslt_ext::getAlgosFromIndex` through the plt; `.rodata` has `hipBLASLt matmul failed (status %d) for M=%ld N=%ld K=%ld %s`.

### 1.3 Execute step (obj 0x18e7b30)

```
18e7b54: movl $0x3f800000,0xc(%rsp)   ; alpha = 1.0f
18e7b5c: movl $0x0,0x8(%rsp)          ; beta = 0
18e7bb0: push $0x8000000              ; workspace bytes = 128 MiB
18e7bc7: call hipblasLtMatmul
```

If `hipblasLtMatmul` fails and the plan has no cached algo (`18e7bd8: cmpb $0x1,0x60(%rbx)`, `18e7be2: cmpb $0x0,0x61(%rbx)`), it calls `hipblasLtMatmulAlgoGetHeuristic` with requested count 1 (`18e7c70: push $0x1`) and workspace 128 MiB (`18e7bfe: movq $0x8000000,0x18(%rsp)`), then retries (second `hipblasLtMatmul` at 0x18e7cf8). Failure exits (`18e7e54: call exit`).

### 1.4 Plan selection and tuning knobs

The plan cache is a tree keyed on (mode, K, N, M-bucket). The M key is rounded up at 0x18e6480 to 0x18e64a6 (`bsr`, shift of `0x0800000000000000`, mask `0xf800000000000000`) when M >= 32 (`18e6473: cmp $0x20,%r12`) and the bucket flag allows it (inferred: rounds M up to a multiple of 2^(p-4), p = index of the top bit of M). Env strings read from `.rodata` with `objdump -s`: `HALOGEN_MATMUL_TUNING_FILE` (0x297e0, getenv call 0x18e5748), `HALOGEN_MATMUL_STATS` (0x2609a, getter 0x18e6218), `HALOGEN_MATMUL_ALGOS` (0x32903, getter 0x18e7508: value if >= 2 else 1, default 1), `HALOGEN_MATMUL_BUCKET` (0x16644, getter 0x18e7548: default on, `18e7568: mov $0x1,%al`). With `HALOGEN_MATMUL_ALGOS` >= 2 the tuner (obj 0x18e7570) times candidates with `hipEventCreate`/`hipEventRecord`/`hipStreamSynchronize` around `hipblasLtMatmul` (calls at 0x18e769f, 0x18e76e5) and can persist the winner by index (`hipblaslt_ext::getIndexFromAlgo`, call 0x18e5fa2) in the tuning file. Default: heuristic top-1 solution.

### 1.5 Which device kernels

- `flash_serve` stubs reached: none. The only `hipLaunchKernel` calls in the TU (0x18e5084, 0x18e5194, 0x18e52a4) belong to three stubs for `k_dn_fused16<0,true,true,false>`, `<0,true,false,false>`, `<0,false,true,false>` (TSV rows 019e4f90, 019e50a0, 019e51b0; handles 0x18fa0c0, 0x18fa0c8, 0x18fa0d0). Every direct call target in 0x18e6240 to 0x18e8960 is a `@plt` import or an address inside the TU's own range; none targets those stubs.
- Device kernels that do run: the Tensile kernels `libhipblaslt.so.1` selects for (bf16, bf16, bf16, fp32 compute, TN, M x 1280 x 2560 and M x 2560 x 640) on gfx1151. Their symbol names are not recoverable from `flash_serve`.
- Inferred candidate files (ai-core, container image layer under `_rocm_sdk_libraries/lib/hipblaslt/library/gfx1151/`): `ls` shows `TensileLibrary_BB_BB_HA_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1151.co` (no bias) and the `..._Bias_..._Alik_Bljk_...` variants. Which solution is chosen depends on the library build and the tuning file; not decided here.

Next test if names are wanted: a kernel trace on the analysis host (needs a run; not done).

## 2. Residual 2: `HALOGEN_MOE_CSORT`

### 2.1 Getter and default

```
FUN_018e8b60 (obj 0x17e8b60, work/decomp/flash_serve/018e8b60.c):
  pcVar1 = getenv("HALOGEN_MOE_CSORT");
  if (pcVar1 != (char *)0x0) { uVar2 = __isoc23_strtol(pcVar1,0,10); return uVar2; }
  return 1;
```

Cached in a function-local static: `17e376c: lea ...(%rip),%rdi # 0x18fcec8` (guard), `17e3780: call 17e8b60`, `17e3785: mov %eax,0x119735(%rip) # 0x18fcec0`. Consumer: `17db05b: cmpl $0x0,0x121e5e(%rip) # 0x18fcec0` then `17db062: je 17db352`. Default 1 (counting-sort arm). Only the integer value 0 selects the other arm; a non-numeric string also parses to 0.

### 2.2 What the CSORT=0 arm does

Inputs (both arms): expert id per slot in the vector at `0x18fcdf8`; S = max(10n, 1) (`17db376` to `17db386`: `mov 0xa8(%rsp),%rcx; cmp $0x2,%ecx; cmovge`). Outputs: `order[]` at `0x18fce78`, `inv[]` at `0x18fce98`, active-expert list at `0x18fce38`, group offsets at `0x18fce58`.

CSORT=0 (obj 0x17db352):

1. `17db352: lea 0x121abf(%rip),%rdi # 0x18fce18`, `17db35c: call 18795f0`: resize the int vector `0x18fce18` to the slot count (Ghidra `019795f0.c`, a `vector<int>` resize).
2. 0x17db392 to 0x17dc189 fills it with 0..S-1 (`movdqu`/`paddd` loop; scalar tail `17dc180: mov %ecx,(%rdi,%rcx,4)`).
3. `17dc199: call 17e8b90` (Ghidra `018e8b90.c`): a `std::stable_sort` of that int range (temporary buffer, merge helpers `FUN_019665a0`, `FUN_01967070`). The comparator reads the global `DAT_019fcdf8` (019665a0.c:40 `lVar4 = DAT_019fcdf8;`, :41-42 `if (*(int *)(lVar4 + (long)iVar9 * 4) < *(int *)(lVar4 + (long)*piVar22 * 4))`), i.e. `eid[a] < eid[b]`.
4. Scan. Element 0: `17dc1b8` to `17dc20d` (`order[0]=sorted[0]`, `inv[sorted[0]]=0`, push its expert id on the active list). Loop i = 1..S-1 at `17ddda0` to `17dddc5`: `order[i]=sorted[i]` (`17dddd7: mov %eax,(%rcx,%rbp,4)`), `inv[sorted[i]]=i` (`17ddde1: mov %ebp,(%rcx,%rax,4)`); when `eid[sorted[i]] != eid[sorted[i-1]]` (`17dddf9: cmp (%r12,%rcx,4),%eax`, `17dddfd: je`) push i on the offsets vector (`17dde13: mov %ebp,(%rbx)`) and the expert id on the active list (`17dddae`). `17dc2d3` to `17dc2eb` pushes S as the final offset (`mov 0xa8(%rsp),%rax; mov %eax,(%rbx)`).

CSORT=1 (default) builds the same four outputs without a sort: histogram over 512 experts (`17db0b0` to `17db0c0`), exclusive prefix into `0x18fd6d0` (`17db10f`), then the scatter loop `17db480` to `17db4bd` (`17db490: mov %esi,(%rax,%r10,4)` = `order[pos]=i`, `17db494: mov %r10d,(%rcx,%rsi,4)` = `inv[i]=pos`). Both orders are stable by slot index, so the outputs match (inferred from the two loops; not diffed on data). Visible difference: CSORT=1 bounds-checks the expert id against 512 (`17db0b3: cmp $0x200,%rdx`, `17db0ba: jae 17e2c81`; 0x17e2c81 loads stderr and the format at .rodata 0x26245, "moe: router produced expert id %d at slot %d of %d (kExperts=%d); refusing", then jumps to 0x17e2f94; the exit after it is inferred). CSORT=0 has no such check.

### 2.3 Kernels

None. The arm is host code: `resize`, fill, `stable_sort`, scan, `operator new`/`memcpy`/`delete` (`call 18f4380`, `18f4370`, `18f4300`). No `hipLaunchKernel` or stub call lies in 0x17db352 to 0x17dc2d3 or 0x17ddda0 to 0x17dddc5 (read in full). R5 section 2.3 holds for both values: A (active-expert count) is the same, so the avg = S / A split is unchanged.

## 3. Residual 4: kernarg slot 0x60 of `k_q4moe` (i4r path, `FUN_018d2430`, obj 0x17d2430)

### 3.1 Side tensors loaded

```
17d2465: mov 0x700(%rbx,%r13,1),%rax    ; gate_up desc
17d246d: mov 0x708(%rbx,%r13,1),%rcx    ; down desc
17d2475: mov 0x8(%rax),%rax ; 17d2479: mov %rax,0x130(%rsp)   ; gate_up weights ptr
17d2481: mov 0x8(%rcx),%rax ; 17d2485: mov %rax,0x148(%rsp)   ; down weights ptr
17d248d: mov 0x750(%rbx,%r13,1),%r14                           ; su
17d2495: mov 0x758(%rbx,%r13,1),%rax ; 17d249d: mov %rax,0x140(%rsp)  ; gate_up svh
17d24a5: mov 0x760(%rbx,%r13,1),%rax ; 17d24ad: mov %rax,0x138(%rsp)  ; down suh
17d24b5: mov 0x768(%rbx,%r13,1),%rax ; 17d24bd: mov %rax,0x78(%rsp)   ; down svh
```

`rbx + r13` is the layer record (`*(ctx+0x5e0)` + layer * 0xc68, `17d2452: imul $0xc68,%rax,%r13`). What the fields are, from the loader (`018897f0.c`): `:1583` `*(undefined8 *)(local_178 + 0x750) = uVar10;` after `FUN_0189a580(lVar11,0xa00,"experts.su")`; `:1606` `+0x758` from `FUN_0189a580(lVar11,0x500,"experts.gate_up_proj.svh")`; `:1629` `+0x760` from `(lVar11,0x280,"experts.down_proj.suh")`; `:1652` `+0x768` from `(lVar11,0xa00,"experts.down_proj.svh")`. `FUN_0189a580` checks dtype 2 (f16) and element count, then returns the tensor's slot +8 pointer, or uploads it with a host-to-device copy when slot +8 is null (obj 0x179a63f: `mov 0x8(%rbx),%rax`; 0x179a648 to 0x179a66b allocate and `call 18f4a90`; error text "an HT plane must be f16 of %lld values"). So all four are f16 device vectors of 2560, 1280, 640, 2560 elements (matches R4 section 1).

### 3.2 Args fill and launch handles

Args is built at `0x80(%rsp)`; kernarg offset = stack offset - 0x80, so slot 0x60 is `0xe0(%rsp)`.

Gate-up launch (obj 0x17d303e, handle 0x18f92b8):

```
17d2f3b: mov 0x130(%rsp),%rax ; mov %rax,0x80(%rsp)   ; slot 0x00 = gate_up weights
17d2f57: movq $0x32000000,0x90(%rsp)                    ; slot 0x10
17d2f63: movq $0x28,0x98(%rsp)                          ; slot 0x18
17d2fab: mov 0x140(%rsp),%rax ; mov %rax,0xe0(%rsp)     ; slot 0x60 = layer+0x758 (gate_up svh)
17d2fbb: mov 0x138(%rsp),%rax ; mov %rax,0xe8(%rsp)     ; slot 0x68 = layer+0x760 (down suh)
17d2fcb: mov 0x128(%rsp),%rax ; mov %rax,0xf0(%rsp)     ; slot 0x70
17d2fdb: mov 0x120(%rsp),%rax ; mov %rax,0xf8(%rsp)     ; slot 0x78
```

Down launch (obj 0x17d3198, handle 0x18f92c0), taken when `HALOGEN_I4R_DN_FOLD` (cell 0x18fccd8) equals 1 (`17d3085: call 18f4be0` precedes `17d308a: cmp $0x1,%bl`, `17d308d: jne 17d3f59`):

```
17d309b: mov 0x148(%rsp),%rax ; mov %rax,0x80(%rsp)     ; slot 0x00 = down weights
17d30b7: movq $0x19000000,0x90(%rsp) ; 17d30c3: movq $0xa,0x98(%rsp)
17d310b: mov 0x78(%rsp),%rax ; mov %rax,0xe0(%rsp)      ; slot 0x60 = layer+0x768 (down svh)
17d3118: movq $0x0,0xe8(%rsp)                           ; slot 0x68 = 0
17d3124: mov 0x58(%rsp),%rax ; mov %rax,0xf0(%rsp)      ; slot 0x70 = [0x18fc778]
```

The DN_FOLD != 1 path (obj 0x17d3f59, launch handle 0x18f92c8, `17d405e: lea 0x125263(%rip),%rdi # 18f92c8`) fills slot 0x60 identically (`17d3fd1: mov 0x78(%rsp),%rax`, `17d3fd6: mov %rax,0xe0(%rsp)`).

Handle to kernel (first `lea X(%rip),%rdi` of each stub, `objdump` of obj 0x1879810 to 0x1879f90; names from the TSV): 0x18f92b8 = `k_q4moe<1,0,64,Tune<0,128>,1>` (stub 01979810), 0x18f92c0 = `k_q4moe<1,2,64,Tune<0,128>,2>` (01979870), 0x18f92c8 = `k_q4moe<1,2,64,Tune<0,128>,1>` (019798d0). Example: `1879849: lea 0x7fa68(%rip),%rdi # 18f92b8` in the stub at 0x1879810.

### 3.3 Which kernels read slot 0x60

Kernarg `s_load` offsets per instance, counted from `kernels-disasm/codeobj_53000.s` over each symbol's line range (awk + grep):

| instance | `s_load` kernarg offsets |
|---|---|
| B0 E1 (gate-up), lines 1995557 to 2008012 | 0x20 (b512), 0x60 (b256), 0x80 (b64), 0x88 (b128) |
| B2 E2 (down, fold 1), lines 2008013 to 2013488 | 0x20 (b512), 0x70 (b64), 0x80 (b64), 0x88 (b128) |
| B2 E1 (down, fold 0), lines 2013489 to 2019000 | 0x20 (b512), 0x60 (b64), 0x70 (b64), 0x80 (b64), 0x88 (b128) |

The slot-0x00 load is not matched by the pattern; consistent with R5 section 3.

### 3.4 Answer

- E=1 gate-up (B0): slot 0x60 starts a 256-bit load covering slots 0x60, 0x68, 0x70, 0x78. Values: `layer+0x758` (`experts.gate_up_proj.svh`, 1280 f16), `layer+0x760` (`experts.down_proj.suh`, 640 f16), then two scratch pointers (inferred): 0x128(%rsp) is the arena base that `k_ht_rot_rows` fills with the rotated activations (`17d253a: mov 0x12a327(%rip),%r13 # 0x18fc868`, `17d26e1: mov %r13,0x128(%rsp)`), 0x120(%rsp) is `param_1[2]` of `FUN_018d6f60`, an arena slice of 10n ints (018d6f60.c: `param_1[2] = uVar5 + lVar2;` with `uVar8 = (long)param_3 * 4 + uVar5`, `param_3 = r12d = 10n` from `17d24d1: lea (%r15,%r15,4),%r12d`).
- E=1 down (B2, DN_FOLD != 1): slot 0x60 is a 64-bit load of `layer+0x768` (`experts.down_proj.svh`, 2560 f16).
- E=2 down (B2, DN_FOLD == 1): the host still stores `layer+0x768` in slot 0x60, but the kernel never loads offset 0x60 (table above). The vector is applied by the follow-up `k_i4r_fold_rows` per R4 section 3 (inferred pairing).

This closes R5 residual 4 and tightens R5's inference: the E=1 output transform reads the svh/suh side tensors of the layer+0x758 to +0x768 block, not a new buffer. R5 section 3 said B0 E1 reads "two more pointers than E0's 0x70 b128"; the two extra are slots 0x60 and 0x68 = gate_up svh and down suh.

## Residuals

1. Names of the hipBLASLt (Tensile) solutions chosen for 1280 x 2560 and 2560 x 640 bf16 TN on gfx1151: need a kernel trace on the analysis host. Not run here.
2. The exact M-bucket rounding conditions at 0x18e6452 to 0x18e64a6 (flag cells 0x18ff950, 0x18ff958, 0x18ff960, 0x18ff968) are read but tied to env names only for `HALOGEN_MATMUL_BUCKET`; they do not change which kernel class runs.
3. CSORT=0 versus CSORT=1 output equality is argued from the two loops, not tested on routed data.
4. Roles of slots 0x70 and 0x78 in the E=1 gate-up launch (arena base, 10n-int slice) are inferred from allocation order; no consumer decode in the kernel body was done.
5. Carried from R5 and untouched: items 3, 5, 6, 7 (affine bit-width slot, run-time pinning, `HALOGEN_DEC_PROBE`, HT item builder DNX read).

## Reproduce

```bash
ssh ai-core 'cd ~/re-halogen && mkdir -p /tmp/r5b &&
 objdump -d --no-show-raw-insn --start-address=0x17d7510 --stop-address=0x17e43d0 binaries/flash_serve > /tmp/r5b/mlf.s &&
 objdump -d --no-show-raw-insn --start-address=0x17d2430 --stop-address=0x17d59b1 binaries/flash_serve > /tmp/r5b/i4r.s &&
 objdump -d --no-show-raw-insn --start-address=0x18e5000 --stop-address=0x18e7e60 binaries/flash_serve > /tmp/r5b/gemm0.s &&
 objdump -d --no-show-raw-insn --start-address=0x18e7e60 --stop-address=0x18e9400 binaries/flash_serve > /tmp/r5b/gemm.s &&
 objdump -d --no-show-raw-insn --start-address=0x1879670 --stop-address=0x1879f90 binaries/flash_serve > /tmp/r5b/q4stubs.s &&
 readelf -d binaries/flash_serve | grep NEEDED'
# env strings: objdump -s --start-address=0x297e0 --stop-address=0x29808 binaries/flash_serve (also 0x2609a, 0x32903, 0x16644)
# kernarg loads: awk 'NR>=A && NR<=B' kernels-disasm/codeobj_53000.s | grep -o 's_load_b[0-9]* s\[[0-9:]*\], s\[[0-9:]*\], 0x[0-9a-f]*'
```
