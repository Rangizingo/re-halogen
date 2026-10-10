# Closure round 3: U21 dtype-to-kernel dispatch (prefill, helper reach, lq args)

Inputs: `work/decomp/flash_serve/*.c` (Ghidra address = ELF vaddr + 0x100000), `llvm-objdump -d` of `binaries/flash_serve` (addresses below prefixed `obj` are ELF vaddrs), `work/kernels-dequant/*.s`, `analysis/function-map-kernel-stubs.tsv`, `analysis/closure/dequant-host.md` (round 1, "R1") and `dequant-host-r2.md` ("R2"). Static only, nothing was run. "Inferred" marks anything not read directly from code.

## 0. What round 2 left and what changed

R2 listed three gaps: (a) which dtype slot reaches which launch helper, (b) the prefill choice, (c) the converter for the dtypes that have no slot. All three are answered by two functions the decompiler cut short:

1. `FUN_018a7520` (bf16-out linear, 5 callers: 018a5b30, 018ab5c0, 0189abe0, 018b86d0, 018e4580) is the master dispatcher. It calls every helper in a fixed order. `FUN_018e6f20` (`linear_f32`, f32-out) is its twin.
2. `FUN_0190d290` is not `tensor_numel`. Ghidra cut it at 191 bytes because `FUN_01892e30` is `dmalloc` (strings "dmalloc: %10.3f MiB") and was marked no-return. The real body runs obj 0x180d290 to 0x180dfc0. It is the dense-bf16 converter, a jump table on the dtype (section 3).

Also new: the decode/prefill phase flag (section 1) and the legacy `k_linear_q4*` family, which R2 said did not exist for q4c (it is in the stub table, `analysis/function-map-kernel-stubs.tsv` rows 459+).

## 1. Dispatch order in FUN_018a7520 and the phase flag

Descriptor `param_1` (long array), slots as written by the binder `FUN_0190e330` (0190e330.c:185-199 q4c/bf16, :201 q8g64, :255 q8s32, :306 q6k, :348 ht, :375 q6g64): `[0]` +0x00 q4cp, `[1]` +0x08 bf16, `[2]` +0x10 q8g64, `[3]` +0x18 q8s32, `[4]` +0x20 q6k, `[5]` +0x28 q6g64, `[6]` +0x30 ht, `[0xb..0xd]` q4cp planes. Args: `param_2` x, `param_3` y, `param_4` N, `param_5` M, `param_6` K.

Order of attempts (018a7520.c line numbers):

| line | call | takes the call when |
|---|---|---|
| 17 | `FUN_01917610` (HT linear) | `*(param_1+0x30) != 0` (01917610.c:28-29), any M |
| 29 | `FUN_01918180` | q4cp slot set, `M-1 < 8`, `K%32==0`, `HALOGEN_LIN_WAVE != 0` (01918180.c:13) |
| 33 | `FUN_0191cd30` | q8g64 slot (+0x10) set, `M-1 < 8` (0191cd30.c:11-12) |
| 37 | `FUN_0191e8f0` | q6g64 slot set, `M-1 < 8` (0191e8f0.c:9-10) |
| 41 | `FUN_01920310` | q8s32 slot (+0x18) set, `M-1 < 8` (01920310.c:8-10) |
| 45 | `FUN_01921ed0` | q6k slot (+0x20) set, `M-1 < 8` (01921ed0.c:10-11) |
| 49-70 | legacy multi-token q4 row GEMV | `HALOGEN_ROW_GEMV` (default 0, 0193caf0.c), `2 <= M <= 8`, `N%8==0`; stubs `k_linear_q4_mt<...>` |
| 80-85 | q4cp dequant plus GEMM | next table |
| 96-299 | bf16 GEMM, `FUN_01927700` (k_lbfw), 8-token tile loops, other-slot dequant plus GEMM | next table |

Consequence: for every quantized dense dtype, M from 1 to 8 goes to the matching `k_lq*` helper on any step. The dequant plus GEMM prefill is only reached for M >= 9 (or when the helper declines).

Phase flag: `DAT_019fe608` is written once per forward in `FUN_018fa150` (018fa150.c:31-53): `bVar8 = 1; if (param_1[0xcc]==0) { ...; if (param_3 < 9 && 0 < iVar5) bVar8 = param_1[0x408]^1; else bVar8 = 0; }` and `DAT_019fe608 = bVar8` (line 53). `param_3` is the new token count, `iVar5` the cached position (`*(int*)(param_1+0x2e8)`). So `DAT_019fe608 == 1` means decode (n <= 8 with a non-empty cache), `0` means prefill (n >= 9, or an empty cache). `param_1[0xcc] != 0` keeps it 1 (mode not decoded; inferred a fixed-shape replay mode).

Prefill scratch: engine init sets `DAT_019fc888 = dmalloc(0x10000000)` (256 MiB), `DAT_019fc890 = 1`, `DAT_019fc898 = &hipBLASLt context (obj 0x18fe2a0)`, only if `max_tok (cfg+0x2e0) >= 2` (obj 0x178710b `cmpl $0x2, 0x2e0(%rax)`, 0x1787110 `dmalloc`, 0x1787115 store to 0x18fc888, 0x178711c `movb $1, 0x18fc890`, 0x1787139 store to 0x18fc898). 256 MiB is 2^27 bf16 elements, matching the size test in 01917610.c:109-112 (`(uint)DAT_019fc890 << 0x1b`). The GEMM is hipBLASLt (strings `hipblasLtCreate`, `hipblasLtMatmulDescCreate` in the binary); `FUN_019e7e60` and `FUN_019e7f40` are the wrapper with last argument 0 versus 2 (019e7e60.c: `FUN_019e6240(...,0)`, 019e7f40.c: `...,2`; reading: bf16 out versus f32 out, inferred).

Prefill gate (018a7520.c:71-80): `HALOGEN_FLASH_PREFILL_GEMM` (default on) `& DAT_019fc898 != 0 & 1 < M` = `bVar2`. Dequant branches additionally need `DAT_019fc890 == 1` and `DAT_019fe608 == 0`. No `N*K <= scratch` check appears in 018a7520 (it exists only in the HT path, 01917610.c:109-112); inferred that the scratch is sized for the largest dense matrix.

## 2. Result table: dtype id to decode kernel to prefill kernel

Decode = M 1..8. Prefill = M >= 9 on a prefill step, `HALOGEN_FLASH_PREFILL_GEMM` on. "GEMM" = hipBLASLt via `FUN_019e7e60` on the 256 MiB scratch.

| dtype id | slot | decode kernel (cite) | prefill kernel (cite) | fallback when the above declines |
|---|---|---|---|---|
| 0 bf16 | +0x08 | `k_lbfw<M,2\|4>`, `k_lbfw_lds<1\|2,4>` via `FUN_01927700` (018a7520.c:112; stub sites in obj 0x1827700) | GEMM straight on the bf16 weight, M >= 2 on a prefill step, `HALOGEN_FLASH_BF16_GEMM` on: `FUN_019e7e60(DAT_019fc898,param_1[1],...)` (018a7520.c:96-114, 249) | legacy `k_linear`, `k_linear_m`, `k_linear_mt` (inferred from stub names; launched after line 112 via the `FUN_01923a20` ladders, 018a7520.c:123-170) |
| 5, qparam 2 (q4c planar) | +0x00 | `k_lq4w<M,R,W,false,E=0,F>`, helper `FUN_01918180` (018a7520.c:29; kernel names from handle sites in R2 section "Launch helper") | `FUN_018e95d0` then GEMM (018a7520.c:80-85). `FUN_018e95d0` launches `k_deq_q4cp_v8` (obj 0x17e98f6, handle 0x18f8378) or `k_deq_q4cp` (obj 0x17e9a50, handle 0x18f8380); `k_deq_q4cp_v32<bool>` at obj 0x17ea100 and 0x17ea41d | legacy `k_linear_q4_mt<R,M,S,T>` (63 instantiations), `k_linear_q4_m<...>` (8), `k_linear_q4` (stub callq sites obj 0x1823b10..0x1827c40 inside 018a7520; `HALOGEN_LIN_WAVE=0` or `K%32 != 0`) |
| 5, qparam != 2 | none | fatal at bind: 0190e330.c:185 accepts only `qparam == 2`; the converter repeats the test (`cmpl $0x2, 0x44(%r12)` at obj 0x180d89d, `jne 0x180dc84`) | same | none |
| 7 q8g64 | +0x10 | `k_lq8w<M,R,W>` via `FUN_0191cd30` (018a7520.c:33) | `FUN_0190dfc0` launches `k_deq_q8g64` (obj 0x180e0b5, handle 0x18f6f58), then GEMM (018a7520.c:290, 299) | `FUN_018e6f20` has no f32 path: fatal "Q8G64 / Q6G64 and has no f32 path" (018e6f20.c:29-37) |
| 11 q8s32 | +0x18 | `k_lq8s32<M,R,W,false>` via `FUN_01920310` (018a7520.c:41); `HALOGEN_LQ8_WAVE=0` (`DAT_019fe598 == 0`, 01920310.c:15-18) takes the legacy `k_linear_q8s32` (inferred) | `FUN_018eaea0` launches `k_deq_q8s32` (obj 0x17eaf95, handle 0x18f8388), then GEMM (018a7520.c:268, 299) | f32 out: `k_lq8s32<M,R,W,true>` via `FUN_018f5c80` (R2) |
| 12 q6k | +0x20 | `k_lq6k<M,R,W,false>` via `FUN_01921ed0` (018a7520.c:45) | `FUN_018fd840` launches `k_deq_q6k` (obj 0x17fd935, handle 0x18f83f0), then GEMM (018a7520.c:258, 299) | f32 out: `k_lq6k<M,R,W,true>` in `FUN_018e6f20`, 8-token tiles for any M (018e6f20.c:39-107) |
| 24 q6g64 | +0x28 | `k_lq6w<M,R,W>` via `FUN_0191e8f0(param_1[5],...)` (018a7520.c:37) | `FUN_0190e0f0` launches `k_deq_q6g64` (obj 0x180e1cb, handle 0x18f6f60), then GEMM (018a7520.c:279, 299) | no f32 path (same fatal as q8g64) |
| 16 ht (bits 3 or 4 from `(qparam>>1)&0x7f`, 0190e330.c:357) | +0x30 | `FUN_01917610` M <= 8: if bf16-out and `M >= HALOGEN_HT_TG_DEC` (clamp 1..8, `FUN_0192bc30`, `DAT_019fe680`): `FUN_01928680` launches `k_ht_tg3<bits>` (obj 0x18289c3, 0x1828cac via stubs 0192bd10/0192bdf0), `k_ht_tg<bits>` (obj 0x1828b21, 0x1828c44) and `k_ht_tg_sum` (obj 0x1828dc6); else `FUN_01928f40` = `k_ht_lin<bits,S,A,B,C,D>` (01917610.c:60-76) | M > 8: (1) if bf16-out, prefill step and `M <= HALOGEN_HT_TRUNK_GEMM` (`FUN_019330f0`, `DAT_019fe710`): the same `FUN_01928680` tg kernels (01917610.c:90-100); (2) else if scratch and `HALOGEN_HT_TRUNK_ORIG` (default on) and `N*K <= 2^27`: `FUN_018f5480` launches `k_ht_deq_orig<bits,PRE,W16?16:8>` into scratch with su/sv passed (`*(param_1+0x38)`, `*(param_1+0x40)`), then GEMM (01917610.c:109-136); (3) else if the trunk-size test passes: rotate-GEMM-rotate with `k_ht_rot_rows` (obj 0x1817bdd), two hipBLASLt wrappers (0x1817c47, 0x1817c84, 0x1817db7) and `k_ht_rot_out_f32` / `k_ht_rot_out` (obj 0x1817d7a, 0x1817e94) around `k_ht_deq_t<bits>` (inferred pairing, 01917610.c:137-163); (4) else 8-token tiles of `FUN_01928f40` `k_ht_lin` (01917610.c:166-181) | none; missing planes or a non-128 grid are fatal (01917610.c:36-47) |
| 14 q4a, 15 q5a | none; bf16 copy at +0x08 | bf16 row (`k_lbfw`) | bf16 row (GEMM on the copy) | the copy is made once at bind by the converter (section 3) with `k_deq_aff<bits,bool>`, bits = 4 for q4a and 5 for q5a, bool = `qparam == 1` (0189a700.c:17,21: `*(uint*)(param_1+8) = iVar2 == 0xf \| 4; ...+0x45 = iVar3 == 1`) |
| 23 i4r | routed experts only | `k_i4r_rows<2560\|640>` and the `_gl/_fl/_o/_epi/_rot` variants, obj 0x17d2890 onward in `FUN_018d2430`; taken when `n < 9` or the decode flag byte `param_1[1] != 0` (018d2430.c:67) | `FUN_018d2430` itself launches only `k_ht_rot_rows` for `n >= 9` and not decode (obj 0x17d2610, grid `(n*20+4)>>3`, 018d2430.c:224-232); the expert GEMM kernel for n >= 9 is not in this function (open, section 5) | n/a |
| plain 4-bit routed experts | n/a | `k_q4moe<A,B,64\|128,Tune<0,128>,E>` from `FUN_018d2430` / `FUN_018d7510` (R2) | not isolated (open, section 5) | n/a |
| 1 f32, 2 f16, 3 i32, 4 i64, 6 fp8r, 8 i4l, 10 fp8g, 13 iq4nl, 20 q8_0, 21 q5_0, 22 q5_1 (9 and 17 to 19 are unassigned codes) | none | not a Linear weight format for flash_serve. The converter jump table sends them to the error branch: `%s: expected bf16, Q4C-P, Q8G64, Q8S32, Q6K, Q4A or Q5A, got %s qparam %u` (string at ELF 0x27f93, printed at obj 0x180dc84, then `exit(1)`) | same | none |

Note: R2's "likely host-side or loader" for i4l, iq4nl, q5_0, q5_1, q8_0, fp8r, fp8g is closed negatively. There is no converter for them in `flash_serve`; a checkpoint must be repacked by `halogen-tools` into one of the seven accepted formats first.

## 3. The converter (FUN_0190d290, real body obj 0x180d290 to 0x180dfc0)

Call chain: `FUN_0190e330` default path (0190e330.c:429 `lVar13 = FUN_0190d290(param_2,param_3)`) and 24 call sites in the layer binders `FUN_0188f710` (10) and `FUN_018897f0` (14) (grep `FUN_0190d290` in `work/decomp/flash_serve`). It sizes `numel*2` (obj 0x180d340 `imulq %r15,%rbp; addq %rbp,%rbp`), calls `dmalloc` (0x180d34a), then `movl 0x10(%r12),%edi; cmpq $0x18,%rdi; ja 0x180dc84; jmpq *%rcx` (0x180d34f-0x180d36f). Jump table at ELF 0x46904, 25 entries of `s32` relative to the table base (read with `r2 pxw`):

| dtype | target | action |
|---|---|---|
| 0 | 0x180d9c8 | plain `hipMemcpy` of the payload into the new buffer (`callq 0x179a920`) |
| 5 | 0x180d89d | requires `qparam == 2`; `callq 0x17e95d0` (`FUN_018e95d0`, k_deq_q4cp family) |
| 7 | 0x180d6ce | K%64 test, then `k_deq_q8g64` (launch at 0x180dc28, handle 0x18f6f58) |
| 11 | 0x180d705 | K%64 test, then `k_deq_q8s32` (launch at 0x180d82a, handle 0x18f8388) |
| 12 | 0x180d397 | `nbytes == N*K/256*210` test, then `k_deq_q6k` (launch at 0x180d4b7, handle 0x18f83f0) |
| 14, 15 | 0x180d371 | `FUN_0189a700` builds `AffDev`, then `callq 0x17e9180` (launcher of `k_deq_aff<4\|5,bool>`; call at 0x180da58) |
| 24 | 0x180d52a | K%64 test, then `k_deq_q6g64` (launch at 0x180d65b, handle 0x18f6f60) |
| every other value 1 to 23 | 0x180dc84 | error text above |

After each kernel: `hipGetLastError`, `hipDeviceSynchronize`, `hipFree` of the temporary raw copy. The error text omits Q6G64 but the table handles it (stale string, inferred).

Correction to R2: the "boundary-attribution artefact" around `FUN_0190d290` was this real function; the four `k_deq_q6g64/q6k/q8g64/q8s32` handle references in that range are its launches, and `k_deq_aff` has 3 callers (obj 0x17dd5a8, 0x17dec6c, 0x180da58), the last being this converter.

## 4. lq* template arguments

`k_lq4w<M,R,W,bool,E,F>(u8 const* w, u16 const* x, void* y, long x5, void* p, float s)`.

| arg | reading | evidence |
|---|---|---|
| M | tokens, 1..8 | R2/R1 `switch(param_5)`; unchanged |
| R, W | lanes per row and waves per row; `(R,W)` in {(32,8),(32,2),(32,1),(16,1)}; rows per block = (8/W)*(32/R) | grid math 01918180.c:20-53 (R2 item 3); inferred from rows per block, not from a lea order |
| bool | `true` = helpers without a K argument (`018f6d80`: N only; `018e5940`: neither), `false` = helpers with K (`01918180`) | handle sites; every `K`-taking helper instantiates `false` (R2 item 4); the kernel difference is one `v_bfe_u32` (3 vs 2) in `<1,32,8,*,0,0>`, not decoded further |
| E | epilogue id, now decoded below | slices `work/kernels-dequant/_ZN7halogen12_GLOBAL__N_16k_lq4wILi1ELi32ELi8ELb*ELi*ELi0EEEvPKhPKtPvlllllS6_f.s` |
| F | `HALOGEN_LQ4W_SH` (01918180 and 018f6d80 pick F from `DAT_019fe850`, R2 item 2). Observed effect (`<1,32,8,false,0,F>`, opcode histogram diff): F=1 has 10 `global_load_d16_b16` and 10 `v_cvt_f32_f16` versus 2 and 2, 14 `s_cbranch_execz` versus 6, same LDS ops (2 `ds_store`, 2 `s_barrier`). So F=1 loads the fp16 scales per lane group instead of once; the LDS use is unchanged, so "SH" is not shared-memory staging of the weights | opcode counts by `awk` over the two `.s` slices |

E values, read from the epilogue tails (slice, F=0, M=1, R=32, W=8). Sum is rounded to bf16 first (`v_bfe_u32 16,1; +0x7fff; & 0xffff0000`), `s` is the trailing `float` kernarg at offset 0x48, `p` the `void*` at 0x40:

| E | helper call sites (R2 item 5) | epilogue | evidence |
|---|---|---|---|
| 0 | 018f6d80, 01918180 | `y = bf16(sum)`, plain store | no `v_mul`, `v_exp`; one `global_store_d16_hi_b16`; the `0x48` load is absent |
| 1 | 018a5f40 | `y = silu(s * sum)` = `x/(1+exp(-x))` | slice b0 E1 lines 305, 327 (`v_mul_f32 v0,s0,v0`), 331 (`v_mul 0xbfb8aa3b`), 334 `v_exp_f32`, 353 `v_div_fixup_f32 v0,v2,v1,v0`; mul count 3 |
| 2 | 018a97a0 | `y = sigmoid(sum)` = `1/(1+exp(-x))`, no scale | slice b0 E2 tail: `v_mul 0xbfb8aa3b`, `v_exp`, `v_div_fixup_f32 v0,v1,v0,1.0`, no `s_load 0x48`; mul count 2 |
| 3 | 0189abe0, bool true | `y = sigmoid(s * sum)` | slice b1 E3: line 305 `s_load_b32 ...0x48`, 327 scale mul, 353 `v_div_fixup_f32 v0,v1,v0,1.0` (numerator 1.0); mul count 3 |
| 4 | 018e5940, bool true | `y = sigmoid(sum)`, same op counts as E2 | slice b1 E4: `v_div_fixup_f32 ...,1.0` at line 344, no `0x48` load |
| 5 | 018b86d0 | plain bf16 sum stored to one of two destinations: `s_load_b64 s[0:1], s[0:1], 0x40` loads `p`; the output index is split per 512 outputs, `(n%512 >= 256)` selects `p` or `y`, offset `(n/512)*256 + (n%256)` elements (reading of the `s_and 0xfffffe00`, `s_cselect` sequence, tail of the b0 E5 slice; the exact destination order is inferred) | no `v_exp`; extra pointer load |

The sigmoid (E2, E3, E4) and SiLU (E1) are what the earlier "power-of-two group scale" reading was (R1 part 4). Re-checked here: `v_exp_f32` appears in the E1-E4 tails only, always preceded by `v_mul_f32 ..., 0xbfb8aa3b` (-log2 e), never feeding a weight decode. Power-of-two group scale: CLOSED negatively, unchanged. Scales in the lq decode are fp16 (`v_cvt_f32_f16`).

## 5. What is still open

1. Routed-expert dispatch by token count. Dense linears are fully read. For i4r (23) and the plain 4-bit `k_q4moe` experts only the decode side is attributed (kernel families per launcher function). `FUN_018d2430` for `n >= 9` launches `k_ht_rot_rows` and returns to code that the decompile cut; the expert GEMM kernels used for prefill (and the `HALOGEN_I4R_SORTED`, `HALOGEN_HT_MOE_M128*` thresholds) are not tied to a branch. `FUN_018d2430`, `FUN_018be390` have no decompiled caller (index `callers` 0), so the call path from `moe_layer_forward` (018d7510) is a vtable or tail jump.
2. How i4r (23) and plain 4-bit expert tensors are bound: `FUN_0190e330` and the converter table have no case for 23, so those tensors use another binder (not traced).
3. `k_lq4w` bool and R/W stay inferred (rows-per-block arithmetic and the K-argument pattern), E5 destination order is inferred from the address math.
4. `HALOGEN_HT_TG_DEC` default and the S, B, C, D arguments of `k_ht_lin` (R2 item 7) are unchanged.

## Verdict

PARTIAL. Dense Linear dispatch (dtype to decode kernel to prefill kernel, the converter for q4a/q5a, the rejection of all other dtypes, and the lq epilogue ids E0-E5) is CLOSED from code. Remaining: the routed-expert prefill kernels by token count and the binder path for dtype 23 and plain 4-bit experts (section 5, items 1 and 2), plus the inferred-only items in section 4 (R, W, bool, E5 order).

## Reproduce

```bash
# object listing (ai-core, read-only); ELF vaddr = Ghidra - 0x100000
ssh ai-core 'cd ~/re-halogen && llvm-objdump -d --no-show-raw-insn --start-address=0x1700000 --stop-address=0x19f0000 binaries/flash_serve > /tmp/r3dd/text.s'
# handle -> kernel map: PTR_FUN_019f.... in work/decomp/flash_serve/01968680.c, demangle with c++filt; stub -> kernel: analysis/function-map-kernel-stubs.tsv
# converter jump table and error text
ssh ai-core 'cd ~/re-halogen && r2 -q -c "pxw 100 @ 0x46904; psz @ 0x27f93" binaries/flash_serve'   # target = 0x46904 + s32
grep -n "leaq.*# 0x18f6f58\|# 0x18f6f60\|# 0x18f8388\|# 0x18f83f0" /tmp/r3dd/text.s              # deq launches in 0x180d290..0x180dfc0
# scratch arming
grep -n "# 0x18fc888\|# 0x18fc890\|# 0x18fc898" /tmp/r3dd/text.s | grep -v "cmp\|test"           # stores at obj 0x1787115/0x178711c/0x1787139
# dispatch order and gates
sed -n 17,50p work/decomp/flash_serve/018a7520.c ; sed -n 71,114p work/decomp/flash_serve/018a7520.c
sed -n 25,53p work/decomp/flash_serve/018fa150.c
# E decode
cd work/kernels-dequant; for e in 0 1 2 5; do grep -c "v_exp\|v_mul_f32" _ZN7halogen12_GLOBAL__N_16k_lq4wILi1ELi32ELi8ELb0ELi${e}ELi0EEEvPKhPKtPvlllllS6_f.s; done
```
