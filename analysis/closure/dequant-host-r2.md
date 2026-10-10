# Closure round 2: U21 (dtype to kernel dispatch) and U19 static part

Method and what changed since round 1. Round 1 (`analysis/closure/dequant-host.md`) said the kernel identity per launch is lost because the decompiler drops the stub call after the no-return `__hipPushCallConfiguration`. That is true for the `.c` files, but the machine code still has it. In the launch helpers the kernel is not called by name: the code does `leaq <handle>(%rip), %rdi` (a `PTR_FUN_019f....` data slot) and passes it to `hipLaunchKernel`. The same handles are registered with their mangled names in `01968680.c` (`__hipRegisterFunction(param_1,&PTR_FUN_019f63c0,"_ZN7halogen12k_sample_f32...`). So: handle address to mangled name (1,419 registrations, demangled with llvm-cxxfilt), then every `leaq` of a handle in the host text, attributed to the nearest preceding function in `work/decomp/flash_serve/index.tsv`. Disassembly: `llvm-objdump -d` on `binaries/flash_serve`; objdump vaddr = Ghidra address minus 0x100000 (checked: objdump 17f5c80 is the same code as `018f5c80.c`). 3,554 handle references found; 1,588 are in the dequant, lq, ht, i4r, hc, q4moe families outside the stub bodies. Nearest-preceding attribution can mislabel a site that sits in a function missing from the index; the per-function counts below were cross-checked against the decompiles where noted.

Env helper functions (each is a tiny cached `getenv`): `FUN_0193ef20` = `HALOGEN_LIN_WAVE` (default 1), `FUN_019406c0` = `HALOGEN_LQ4W_SH` (default 0), `FUN_0190fc50` = `HALOGEN_LQ8_WAVE` (default 1), `FUN_01940f00` = `HALOGEN_HT_LIN_DIRECT`.

## U21 part 1: dtype to kernel dispatch table

Verdict: CLOSED for the decode and small-batch launch (kernel identity now read from the binary, not inferred); PARTIAL for prefill and for which helper each dtype slot reaches (the callers of the K-aware selectors were not traced).

### Launch helper to kernel family (read from handle `leaq` sites)

| helper (addr, function-map name) | kernel instantiations launched | gate on entry |
|---|---|---|
| 018f6d80 `launch_gemm_by_format_b` | `k_lq4w<M, R, W, true, 0, F>`, M 1..8; (R,W) in {(32,8),(16,1)}; F in {0,1} (32 sites) | `HALOGEN_LIN_WAVE != 0`, `*param_1 != 0` (+0x00 q4cp slot), tokens 1..8 (018f6d80.c:10-12, 15) |
| 01918180 `launch_gemm_select_c` | `k_lq4w<M, R, W, false, 0, F>`; (R,W) in {(32,8),(32,2),(16,1),(32,1)} (48 sites) | same, plus `(K & 0x1f) == 0` (01918180.c:13, `param_6` = K) |
| 018f5c80 | `k_lq8s32<M,16,1,true>` and `<M,32,8,true>` | q8s32 slot `+0x18 != 0`, tokens 1..8 (018f5c80.c:8) |
| 01920310 `launch_gemm_select_f` | `k_lq8s32<M,{16,1 / 32,1 / 32,2 / 32,8},false>` | K-aware twin of 018f5c80 |
| 018e6f20 `linear_f32` | `k_lq6k<M,16,1,true>`, `<M,32,8,true>` | q6k slot `+0x20` (018e6f20.c:39-107) |
| 01921ed0 `launch_gemm_select_g` | `k_lq6k<M,{16,1 / 32,1 / 32,2 / 32,8},false>` | K-aware twin |
| 0191cd30 `launch_gemm_select_d` | `k_lq8w<M,{16,1 / 32,1 / 32,2 / 32,8}>` | q8g64 (inferred from kernel name) |
| 0191e8f0 `launch_gemm_select_e` | `k_lq6w<M,{16,1 / 32,1 / 32,2 / 32,8}>` | q6g64 (inferred from kernel name) |
| 01927700 `launch_gate_guarded_kernel` | `k_lbfw<M,2>` M 1..8, `k_lbfw<{1,2},4>`, `k_lbfw_lds<1,4>`, `<2,4>` | bf16 slot (+0x08); called from 018a7520 |
| 018e5940 `launch_kernel_by_dtype_switch` | `k_lq4w<M,32,8,true,4,F>` | tokens 1..8 |
| 019125b0 to 01913e30 (8 helpers) | `k_lq4w_grp<M,...>` x4 each | the batched `LqGrp` launches |
| 019155b0, 01915a00 | `k_lq6w_grp` x8 each | |
| 018e4580 `moe_route_decode_fused` | `k_lq4w_pair<1..4,16,1>` | M 1..4 only (018e46b6, 018e4cc8, 018e4d69, 018e4e06) |

The previous report's inference that the q4c-planar slot launches the `k_lq4w` family is confirmed (018f6d80 launches only `k_lq4w`). Confirmed too: q8s32 launches `k_lq8s32` (018f5c80), q6k launches `k_lq6k` (018e6f20), bf16 launches `k_lbfw` (01927700).

### Dispatch table (decode = 1 to 8 tokens)

| .hgn dtype (code) | decode kernel and helper | prefill (more than 8 tokens, GEMM) | MoE expert path |
|---|---|---|---|
| bf16 (0) | `k_lbfw<M,2>` / `k_lbfw_lds`, helper 01927700 | bf16 GEMM `FUN_019e7e60` (round 1, 018fdaf0) | n/a |
| q4c planar, qparam 2 (5) | `k_lq4w`, helpers 018f6d80 (no K) and 01918180 (K-aware) | dequant `k_deq_q4cp` / `_v8` / `_v32` (018e95d0), `_sel` (018ed600), then bf16 GEMM (inferred pairing; callers of the deq launchers not traced) | `k_q4moe` (below); `k_lq4w_pair` for M<=4 in 018e4580 |
| q8g64 (7) | `k_lq8w` (0191cd30); no f32 path (round 1, 018e6f20.c:30-38) | `k_deq_q8g64` (018a5730, 0190dfc0, 0190d290 region), then bf16 GEMM (inferred) | n/a |
| q8s32 (11) | `k_lq8s32` (018f5c80, 01920310) | `k_deq_q8s32` (018eaea0, 018fc7e0, 018fdaf0) | `q8s32_expert_forward_checked` 018ed600, "Q8S32 experts read in place" (function map); kernel not isolated |
| q6k (12) | `k_lq6k` (018e6f20, 01921ed0) | `k_deq_q6k` (018fc7e0, 018fd840, 018fdaf0) | n/a |
| q6g64 (24) | `k_lq6w` (0191e8f0); no f32 path | `k_deq_q6g64` (018a5730, 0190e0f0) | n/a |
| ht (16), bits 3 or 4 | `k_ht_lin<bits,S,A,B,C,D>` via 01928f40 (A true), 0192c1d0 and 0192d700 (A false); `k_ht_lin_grp` x33 in 018b0e40 (`linear_attn_ht_forward`, `HALOGEN_HT_LIN_GRP`); rotation `k_ht_rot_in/out/rows` in 01917610 and 018be390; `k_ht_tg3<3|4>` in 01928680 and `k_ht_tg` | `k_ht_deq_t<3|4>` (01917610) and `k_ht_deq_orig` x8 (018f5480) dequant path (inferred: dequant then GEMM; `HALOGEN_FLASH_PREFILL_GEMM` string) | `k_ht_gemv_gu<3|4,8,1>`, `k_ht_gemv_dn<3|4,8,1>`, `k_ht_moe_gu2<3|4,64,0>`, `k_ht_items_slots`, plus `_t`, `_ts`, `_g`, `_gs` variants, all launched from 018be390 |
| i4r (23, rank 3 experts) | n/a (routed experts only) | n/a | `k_i4r_rows<2560>` and `<640>`, `k_i4r_fold`, `_fold_rows`, `_gu_epi`, `_rows_epi`, `_rows_rot`, `k_ht_rot_in`, `k_ht_rot_rows`, all from 018d2430; `_gl`, `_fl` and `_o` variants (x2 to x4) from 018d2430 (function map name `moe_i4r_deferred_shared_down` is 018d5a00) |
| 4-bit routed experts (plain) | n/a | n/a | `q4moe::k_q4moe<A,B,64\|128,Tune<0,128>,E>`: (A,B,E) in {(1,0,1),(1,2,1),(1,2,2),(2,0,1),(2,2,1),(2,2,2)} x width {64,128}, plus `k_q4moe_items<64\|128>`; all from 018d2430 (14 instantiations listed there, the other 6 from 018d7510 which also has `k_q4moe_items`) |
| q4a (14), q5a (15) | no `k_lq*` kernel; `k_deq_aff<4\|5, bool>` (018e9180, name `launch_cfg_grid_rows_div32`, grid = rows*cols/32/256) with the 4/5 equal to the bit width; signature `(u8 const*, u16*, AffDev, uint, uint, long)`: a dequant to a 16-bit (bf16) buffer; then `k_lbfw`. Inferred pairing of q4a to 4 and q5a to 5; the dtype-to-instantiation link is not read from a call site | same | n/a |
| f32 (1), f16 (2), fp8r (6), fp8g (10), i4l (8), iq4nl (13), q5_0 (21), q5_1 (22), q8_0 (20), q4c interleaved | no dedicated GEMV kernel exists in the 1,738-stub list for these (grep of the stub table: no `k_lq*` or `k_deq_*` name carries them). Round 1's reading holds (bf16 copy, then `k_lbfw`); the converter is not among `k_deq_*` except `k_deq_aff` (maybe i4l, unverified) and `k_bf16_to_f32`/`k_deq_q4cp_f32` (01899dc0, an f32 view) | same | n/a |
| i32 (3), i64 (4) | not linear weights | | |

Gap in the table: the prefill column names the dequant kernels and their launch helpers, but I did not trace the GEMM-side branch that chooses dequant-then-GEMM over a fused kernel (0x1400 and 0xa1f thresholds below are the only shape tests seen). Round 1 also noted `FUN_0190d290` as the bf16 fallback allocator; the function map calls it `tensor_numel`, and the four `k_deq_q6g64/q6k/q8g64/q8s32` handles are referenced in the index range of 0190d290. Treat that range as a boundary-attribution artefact until the index entry is split.

### Selection conditions read from the helpers

M = first argument = token count 1..8 (a `switch(param_5)` or `param_5 - 1U < 8` guard in every helper). N = output rows (`param_4`), K = input width (`param_6`).

1. Global gate: `FUN_0193ef20` = `HALOGEN_LIN_WAVE` (default 1); 0 disables every `k_lq*` launch in 018f6d80, 01918180 and 018e5940 (they return 0 and the caller falls back; fallback not traced).
2. `F` (last `k_lq4w` argument) = `HALOGEN_LQ4W_SH != 0` (019406c0.c returns `strtol` of the env, default 0; the cached value `DAT_019fe850` is tested `testl %ebp,%ebp` at objdump 17f6e41 and picks the F=0 or F=1 instantiation). Meaning of F: not decided; the env name suggests a shared-memory variant (inferred).
3. Variant by shape in 01918180.c (K-aware, bool false): N < 9: grid N blocks, 256 threads (line 20); else if `N > 0x1407` or `K < 0x1400`: if `K > 0xa1f` grid `(N+7)>>3` (line 29-37) else grid `(N+15)>>4` (line 45); else grid `(N+3)>>2` (line 53). Reading that fits all four instantiations (inferred from rows per block, not from a lea order): R = lanes per output row, W = waves per row, block = 8 waves of 32. Rows per block = (8/W) x (32/R): (32,8) = 1 row (grid N), (32,2) = 4 rows (grid (N+3)>>2), (32,1) = 8 rows (grid (N+7)>>3), (16,1) = 16 rows (grid (N+15)>>4). So: N < 9 uses (32,8); narrow K (`0xa20 <= K`, below 5120) or large N (> 5127) uses (32,1) when K > 2591 else (16,1); the middle case (N <= 5127 and K >= 5120) uses (32,2).
4. Descriptor-based helpers without a K argument (018f6d80, 018e6f20, 018f5c80, 018e5940) only instantiate (32,8) and (16,1) with bool true: N <= 8 gives (32,8), N > 8 gives (16,1) (grid `(N+15)>>4`, 018f5c80.c:20). Same bool split in all families: `true` is the no-K variant, `false` the K-aware variant (observed: every helper with a K parameter instantiates `false`, none with `true`). What the bool changes in the code (`v_bfe_u32` 3 vs 2, round 1) is probably a K tail/alignment branch removed at compile time (inferred).
5. `E` (fifth `k_lq4w` argument, epilogue) is a per-call-site constant: plain GEMM helpers 018f6d80 and 01918180 use E=0; E=1 from 018a5f40 (hc group helper), E=2 from 018a97a0 (hc group helper), E=3 from 0189abe0 (hc layer function), E=4 from 018e5940 (dtype switch), E=5 from 018b86d0 (`sparse_attn_block_select`, QSA block scoring). Which numeric E is which epilogue stays open; the caller list is the new fact. Inference: E=0 is plain store.
6. lq template arguments, now closed: `k_lq4w<M,R,W,bool,E,F>`: M tokens, R lanes per row (16 or 32), W waves per row (1, 2 or 8), bool = K-aware, E epilogue id, F = `HALOGEN_LQ4W_SH`. `k_lq6w/k_lq8w<M,R,W>`: same R, W, with (16,1), (32,1), (32,2), (32,8). `k_lq6k/k_lq8s32<M,R,W,bool>`: bool true is the no-K helper, false the K-aware one.
7. HT linear (01928f40.c:23-27): `*(int*)(desc+0x48)` is the bit width (3 or 4 as the binder stores it); `*(char*)param_1[1] == '\0'` hands the call to 0192c1d0 (small). Otherwise the big path checks `HALOGEN_HT_LIN_DIRECT` (default on; `DAT_019fe6b0`), then `HALOGEN_HT_LIN_PIN == "1"` (default off) and `HALOGEN_HT_U24 == "1"` (default off). Instantiation `k_ht_lin<bits,S,A,B,C,D>`: bits 3 or 4; S in {1,2} (selected with `param_4`, token count 1 vs 2; larger counts use the same two, so S is not the token count itself, meaning not decided); A = true in 01928f40 (big), false in 0192c1d0 (3-bit) and 0192d700 (4-bit); B, C, D are the booleans (direct, pin, u24 in some order, inferred) with the four seen combinations FFF, TFF, TFT, TTF, i.e. pin and u24 are mutually exclusive and both require direct. Entry condition on size: `rows * K_blocks * 8 <= DAT_019fcb18 << 22` and `(rows >> 7) <= DAT_019fcb1c * 0x1000` (01928f40.c, lines 29-35); failing it falls to split-scratch code with the message `HT linear %d x %lld needs more split scratch than armed` (line 423).
8. HT MoE: env helpers in the 018ca6xx-018ca7xx range with defaults: `HALOGEN_HT_MOE_GEMM` 1, `_DV` 1, `_V2` 1, `_DNX` 1, `_WPE` 8, `_M128` 3, `_M128_GU` 0x4000, `_M128_DN` 0x1000; `HALOGEN_HT_TG_M128` default 0x100, `HALOGEN_HT_GU_SPLIT`, `HALOGEN_HT_DEC_GROUP`, `HALOGEN_HT_GEMV_KL` 0. These are read by 018be390/018d2430 paths (not each tied to a branch). I4R: `HALOGEN_I4R_SORTED` default 3, `_HFUSE`, `_SDFUSE`, `_LAST`, `_DEC_GROUP`, and `HALOGEN_MOE_*` (GPU_ROUTE 1, CSORT 1, H_ORDER 1, DN_ROWS 1, SIZE_HIST 0).
9. `k_q4moe<A,B,W,Tune<0,128>,E>`: W = 64 or 128 (expert tile width, inferred from the paired `k_q4moe_items<64|128>` index kernels); A in {1,2} and B in {0,2} and E in {1,2} are used only in the six combinations above (E=2 only with B=2). Meaning of A, B, E not decided.

What remains for part 1:
- Which dtype slot reaches 01918180 vs 018f6d80 (the callers of the K-aware `select_c/d/e/f/g` helpers, expected in the layer-forward functions) and the GEMM-side choice for prefill.
- The meaning of `k_lq4w` F and E and `k_ht_lin` S, B, C, D (needs the argument fill in a launch frame, or a runtime `hipLaunchKernel` trace).
- Which kernel converts i4l, iq4nl, q5_0, q5_1, q8_0, fp8r, fp8g to bf16 (not in the handle list; likely host-side or the loader).

## U21 part 2 (template arguments): CLOSED, with the exceptions above

R, W, M, bool, F are decided by section 6 and the grid arithmetic (R, W are inferred from rows per block; the grids and instantiation sets are read directly). E, S, B, C, D remain open as stated.

## U19 static part: what round 1 left, and what I added

Verdict: PARTIAL. Round 1's decode recovery stands (16-bit sliding windows, multiply by `0x83dcd12d`, `v_sad_u8` byte sum plus `0x6400`, fp16 affine `0xc931`). The host side adds the following concrete facts.

1. Bits (K) reach the kernel as template argument `bits` (3 or 4), chosen from the descriptor field at `+0x48`, which the binder fills with `(qparam >> 1) & 0x7f` (round 1: 0190e330.c:357). So qparam values 6 and 8 mean 3 and 4 bit, and bit 0 of qparam is not read at bind time (qparam bit 0 meaning still open).
2. The weight stream is read in place from pinned memory (round 1: 0190e330.c:351). `k_ht_lin` has a dedicated direct path (`HALOGEN_HT_LIN_DIRECT`, default on) and a pin path; the dequant to a bf16 buffer (`k_ht_deq_t<3|4>` 01917610, `k_ht_deq_orig` x8 018f5480) is the alternative route. A host CPU decoder built from the round 1 recipe can be checked against `k_ht_deq_orig` output on a real tensor.
3. su and sv: the rotation kernels `k_ht_rot_in` (su side, input rotation), `k_ht_rot_out`/`_f32` (sv side), `k_ht_rot_rows`/`_h` are launched from 01917610 (`ht_linear_prepare`), 018be390 (HT MoE) and 018d2430 (I4R MoE) and the same rotation kernels serve the i4r experts (018d2430 launches `k_ht_rot_in` and `k_ht_rot_rows`). So i4r shares the HT rotation planes; its `k_i4r_rows_rot` is a fused variant. The descriptor offsets `+0x38` and `+0x40` copied into the launch frame in 01928f40.c (`local_128 = *(lVar2 + 0x38)`, `local_130 = *(lVar2 + 0x40)`) are the su and sv device pointers (inferred from the two adjacent pointer slots next to the weight at `+0x30`; not proved by a name).
4. Dimension input to the HT kernels: `uVar1 = *(uint*)param_1[2]`, `uVar3 = *param_1[3]` are the two dims; the block count is `rows >> 7` (128-row granule), consistent with the 128-point Walsh-Hadamard.

Needs a real HT checkpoint (unchanged from round 1, restated with the host facts now removed from the list):
- stream bit order and tile order (the `.hgn` error text says "codebook, tile order or bits"),
- the exact scale `s6` and whether row or group scales fold in (which kernel argument feeds it),
- the content of su and sv (signs, scales or both; a fp16 vector at `+0x38/+0x40`),
- any additive term outside the multiply,
- qparam bit 0.

Test: take one HT tensor with its tensor-entry (qparam, dims) and its su/sv from a real checkpoint, decode a handful of words on the CPU with the round 1 recipe, and compare with the same tensor dequantised in bf16 (or with `k_ht_deq_t` output from a real engine run).

## Reproduction

Handle table: `01968680.c` (`__hipRegisterFunction(param_1,&PTR_FUN_019f....,"_ZN7halogen..."`); handle uses: `llvm-objdump -d --no-show-raw-insn --start-address=0x1700000 --stop-address=0x1a00000 binaries/flash_serve`, then every `leaq ... # 0x18f....` whose address plus 0x100000 equals a registered handle. Unfinished: the direct `callq` references to stub addresses are only 347 and are all inside the `k_hc*` and `k_ht_*` paths; the main launches use the handle form above.
