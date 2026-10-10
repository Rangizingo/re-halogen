# Closure: U21 (dtype-to-kernel dispatch) and U19 static part (HT weight decode)

Inputs: `work/decomp/flash_serve/*.c`, `work/kernels-dequant/*.s`, `analysis/kernels-dequant.md`, `analysis/hgn-format.md`. Slice names are shortened; line numbers are `.s` line numbers. Decompile line numbers are `.c` lines. Anything not marked "inferred" is read directly from the quoted line.

## U21 part 1: dtype-to-kernel dispatch

Verdict: PARTIAL (binder and slot map closed; per-batch kernel choice inside the launch helpers only partly recoverable).

### The weight descriptor (the key finding)

`FUN_0190e330` (function map: `upload_weight_tensor`) is the per-tensor binder. It zeroes a 128-byte descriptor (8 x 16 B) and fills one slot depending on the dtype code at tensor-entry `+0x10`. The slot that is non-null IS the dispatch key used later. Slots (offsets into the descriptor):

| offset | dtype (code) | filled at | notes |
|---|---|---|---|
| +0x00 | q4c planar "Q4C-P" (5, qparam 2) | 0190e330.c:185-192 `if (uVar1 == 2 && *(int *)(param_2 + 0x10) == 5)` then `*(undefined8 *)*param_1 = *(undefined8 *)(param_2 + 8)` | band/pad info from `FUN_018e8950` (018e8950.c:19 `expected Q4C-P, got %s qparam %u`, :64-76 size check) |
| +0x08 | bf16 (0), and the fallback for everything else | 0190e330.c:198-199 `case 0: ... goto LAB_0190e955`; :431 `*(long *)(*param_1 + 8) = lVar13` | fallback calls `FUN_0190d290` (0190d290.c:38, allocates `numel*2` bytes) |
| +0x10 | q8g64 (7) | 0190e330.c:241 `*(ulong *)param_1[1] = uVar15`; size rule `numel*17/16`, K%64 (:207,:238,:247) | |
| +0x18 | q8s32 (0xb) | :295 `*(ulong *)(param_1[1] + 8) = uVar15`; size `N*pad16(17K/16)` (:294,:301) | |
| +0x20 | q6k (0xc) | :340 `*(long *)param_1[2] = ...`; size `N*K/256*210`, K%256 (:339,:344) | |
| +0x28 | q6g64 (0x18) | :416 `*(ulong *)(param_1[2] + 8) = uVar19`; qparam must be 0x40 (:408), size `N*pad16(13K/16)` (:423) | |
| +0x30 | ht (0x10) | :356 `*(long *)param_1[3] = *(long *)(param_2 + 8)` | pinned-memory only, :351 `HT weights are read in place and need the pinned path (HALOGEN_FLASH_PIN_TRUNK must not be 0)` |
| +0x48 | HT bit width | :357 `*(uint *)(param_1[4] + 8) = uVar1 >> 1 & 0x7f` | `uVar1` is the entry's qparam (`param_2 + 0x44`): bits = (qparam >> 1) & 0x7f; bit 0 of qparam not used here (inferred: a flag consumed elsewhere) |
| +0x4c / +0x50 | HT dims | :358-368 | rank-2 only: `+0x4c` = dim0 (N), `+0x50` = dim1 (K); otherwise 0 |

The dtype codes handled by the switch are exactly 0, 7, 0xb, 0xc, 0x10, 0x18 (plus the q4c/qparam 2 case before it). All other dtypes take no slot: they go to `FUN_0190d290`, which sizes a bf16 buffer of `numel*2` bytes (0190d290.c:38). The decompile loses the fill (the call is marked no-return, 0190e330.c:429). So for q4c (interleaved, qparam != 2), f32, f16, fp8r, fp8g, i4l, iq4nl, q4a, q5a, q5_0, q5_1, q8_0 the host holds a bf16 copy and the kernel is the bf16 one (k_lbfw). That the conversion to bf16 happens in the loader is inferred from the `numel*2` allocation and the `.bf16` sidecar path (:105, `fopen(..., "rb")`, size check `sVar12 == sVar8` with `sVar8 = numel*2`, :157-162), not seen directly. `i4r` (23, rank 3 experts) is not in this binder; it is bound by the MoE paths (function map: `moe_i4r_deferred_shared_down` 018d5a00, high confidence).

### What launches from each slot (decode, batch 1 to 8 tokens)

`FUN_018e6f20` (`linear_f32`, function map: high) takes the descriptor as `param_1` (`long*` indexing, so `[2]` = +0x10, `[3]` = +0x18, `[4]` = +0x20, `[5]` = +0x28):

- 018e6f20.c:30-38: `if (param_1[5] != 0 || param_1[2] != 0)` print `linear_f32: %s is Q8G64 / Q6G64 and has no f32 path`, `exit(1)`. This answers the string: q8g64 and q6g64 have no f32-output path at all.
- :39-107: `if (param_1[3] != 0 || param_1[4] != 0)`: loop in chunks of 8 tokens (`iVar6 = 8; if (iVar2 < 8) iVar6 = iVar2`), `if (param_1[4] == 0) FUN_018f5c80(...)` (q8s32 only), else a `switch(iVar6)` of 7 cases plus default, each pushing a launch config `grid = N` if `param_4 < 9` else `(N+15)>>4`, block `0x100` = 256 threads. So q8s32 goes to `FUN_018f5c80`, q6k to the inline switch.
- `FUN_018f5c80` (018f5c80.c:8-67): requires `*(long *)(param_1 + 0x18) != 0` (q8s32 slot) and `param_5 - 1U < 8` (1 to 8 tokens), `switch(param_5 - 1U)` 8 arms, grid `N` or `(N+15)>>4`, block 256. This is the q8s32 GEMV with token count as the first template argument.
- :108 onward: `FUN_018f6d80` (018f6d80.c:6-60, function map `launch_gemm_by_format_b`): requires `*param_1 != 0` (the +0x00 q4cp slot) and token count in 1..8 (`0xfffffff7 < param_5 - 9U`, i.e. `param_5 <= 8`), `switch(param_5)` with the same grid rule. Inferred: this is the k_lq4w / k_lq4w_pair path for q4c planar. The fallthrough after it (018e6f20.c:~110-150) uses the bf16 slot and `__hipPushCallConfiguration` with `param_5 < 9`-style grids: inferred k_lbfw / k_lbfw_lds.
- The kernel identity per launch is not in the decompile: every launch is a no-return `__hipPushCallConfiguration` with the stub call dropped. The kernel handles are only visible in the registration table `01968680.c` (mangled names, e.g. :128 `k_deq_q4cp_f32`, :3468-3562 the `k_deq_q4cp*` family). So the mapping slot to kernel family below is by slot plus kernel signature, labelled inferred where the stub is not shown.

### Table: .hgn dtype to kernel family

"Decode" = 1 to 8 tokens (`param_5 <= 8`). "Prefill" = more than 8 tokens, or the lm-head / trunk GEMM path.

| dtype (code) | slot | decode kernel | prefill kernel |
|---|---|---|---|
| bf16 (0) | +0x08 direct | k_lbfw<M,2\|4> / k_lbfw_lds (inferred, bf16 slot) | `FUN_019e7e60` bf16 GEMM (018fdaf0.c: `FUN_019e7e60(DAT_019fc898, lVar14, lVar19, ...)` with `param_1 + 0x610` = descriptor +0x08) |
| q4c planar, qparam 2 (5) | +0x00 | k_lq4w / k_lq4w_pair / k_lq4w_grp via 018f6d80 (inferred from slot and signature) | inferred: `k_deq_q4cp*` dequant to bf16 scratch, then bf16 GEMM (the registration shows the full `k_deq_q4cp*` family, 01968680.c:3468-3562; the call is not traced) |
| q4c interleaved (5, other qparam), q4a (14), iq4nl (13), q5a (15), q5_0 (21), q5_1 (22), q8_0 (20), i4l (8), fp8r (6), fp8g (10), f16 (2), f32 (1) | none; bf16 copy at +0x08 | k_lbfw | bf16 GEMM |
| q8g64 (7) | +0x10 | k_lq8w / k_lq8w_grp (inferred); never the f32 path | no f32 path (018e6f20.c:34); inferred `k_deq_q8g64` then bf16 GEMM |
| q8s32 (11) | +0x18 | `FUN_018f5c80`, k_lq8s32<M> (inferred) | `linear_f32` handles it in 8-token chunks; larger batches: inferred `k_deq_q8s32` |
| q6k (12) | +0x20 | inline switch in 018e6f20.c:50-106, k_lq6k<M> (inferred) | same chunked loop; inferred `k_deq_q6k` for GEMM |
| q6g64 (24) | +0x28 | k_lq6w / k_lq6w_grp (inferred); never the f32 path | no f32 path; inferred `k_deq_q6g64` |
| ht (16) | +0x30, bits at +0x48 | k_ht_lin / k_ht_lin_grp / k_ht_gemv_gu; `linear_attn_ht_forward` 018b0e40 and `ht_linear_prepare` 01917610 (strings `HALOGEN_HT_LIN_GRP/DIRECT`, `HALOGEN_FLASH_PREFILL_GEMM`) | k_ht_* with the rotation planes; MoE: k_ht_moe_* (WMMA) |
| i4r (23) | not in this binder (rank-3 experts) | k_i4r_rows / _gl / _rot / _epi (function map: `moe_i4r_deferred_shared_down`) | same family; `k_q4moe` is the plain 4-bit routed-expert WMMA path |
| i32 (3), i64 (4) | none | not linear weights | none |

What remains for U21 part 1: the per-slot kernel names are inferred from the slot plus signatures because the launch stubs are dropped by the decompiler. Closing it needs either a decompile of the stub call sites (the `PTR_FUN_019f....` handles registered in 01968680.c, matched against each launch helper's callee list) or a runtime trace of `hipLaunchKernel` names with one tensor of each dtype.

## U21 part 2: template arguments

Verdict: PARTIAL.

Instantiation inventory (from the registration table in 01968680.c, grep of mangled names):

- `k_lq4w<M,R,W,bool,E,F>`: M = 1..8, R in {16,32}, W in {1,2}, bool in {0,1}, E in {0,1,2,3,4,5}, F in {0,1}; signature `(uint8*, bf16*, void*, long x5, void*, float)`.
- `k_lq4w_grp<M,R,W,E>` with (R,W) = (16,1) or (32,2); `k_lq4w_pair<M,16,E>` for M = 1..4 only.
- `k_lq6w_grp<M,32,{2,4,8}>`, `k_lq8w_grp<M,32,2>`, `k_lbfw<M,{2,4}>` and `k_lbfw_lds<{1,2},4>`.

Readings:

1. First argument = number of tokens in the batch, 1 to 8. Evidence: every family is instantiated for M = 1..8 and the host picks the instantiation with `switch(param_5)` over 1..8 (018f5c80.c:14, 018f6d80.c `switch(param_5)`, 018e6f20.c `switch(iVar6)` where `iVar6 = min(8, remaining tokens)`). `k_lq4w_pair` stops at 4 (inferred: two weight streams double the register cost).
2. Second argument in `k_lq4w<1,16,1,...>` and `k_lq8w<1,16,1>` = output rows per workgroup (16 or 32). Evidence (partly inferred): the host grid is `(N+15)>>4` blocks of 256 threads for N > 8 (018e6f20.c:~46 `param_4 + 0xf >> 4`, 018f5c80.c:20), which is 16 rows per block; the 32-row variants are not selected in the visible grid math (their selection is in the dropped stub call sites). Eight wave32 waves per block, so 16 rows = 2 rows per wave (inferred).
3. Third argument (1 or 2 in lq4w, 2/4/8 in lq6w, 2/4 in lbfw) = a per-row parallelism or unroll factor. Not decided. Measured in slices: `k_lq4w<1,16,1,..>` 301 lines, `<1,32,1,..>` 319, `<1,32,2,..>` 322 lines with 2 `s_barrier` instead of 1 and `v_bfe_u32` 3 to 4; `v_dot2_f32_bf16` stays 16 in all. A second barrier at W = 2 suggests a second LDS reduction stage (inferred); nothing proves "k-split" versus "unroll".
4. In `k_lq4w`, the `bool` (0/1) changes `v_bfe_u32` count 3 to 2 (slice `<1,16,1,b1,0,0>`: 298 lines, bfe 2, vs 301 lines and 3 for b0): a decode-variant switch (inferred, not decided what).
5. Fifth argument E (0..5) = epilogue mode; the `_grp` kernels read it from the `LqGrp` struct at runtime (see part 4, `s_cmp ... s7, 3/4/5` compares at lines 436-445 of `k_lq4w_grp<1,16,1,0>`). Modes differ in which output is written (bf16 store, f32 store, sigmoid, scaled sigmoid). Which numeric E maps to which epilogue is not decided for plain `k_lq4w`.
6. `k_lbfw<M,2|4>` has no unpack (8 `v_dot2`), so its second argument cannot be a decode choice; reading: unroll depth of the K loop (inferred). `k_lbfw_lds` is the LDS-staged variant.

What remains: map E values 0..5 to epilogues for plain `k_lq4w`, and confirm the 3rd-argument meaning, by reading the argument fill at each stub call site or tracing real launches.

## U21 part 3: the u64 outputs of k_hc6_fused / k_hc8_fused

Verdict: CLOSED for the u64 output (it is a progress counter, not data). The earlier guess "argmax/top-k key" in `kernels-dequant.md` s1 is wrong.

Signature: `k_hc8_fused<M>(LqGrp, bf16*, long, int, uint8*, uint8*, bf16*, float*, unsigned long long* P, unsigned long long y)`. Kernarg layout in the slice: `s_load_b128 s[12:15], s[0:1], 0x160` (slice `k_hc8_fused<1>`, line 6) loads P into `s[12:13]` and `y` into `s[14:15]`.

Evidence in `k_hc8_fused<1>`:
- Lines 92-108: thread 0 only (`v_cmpx_eq_u32 0, v0`) spins: `global_load_b64 v[18:19], v17, s[12:13] glc` (96), `v_cmp_le_u64 vcc_lo, s[14:15], v[18:19]` (100), `s_sleep 1` (102), reload, `v_cmp_gt_u64 ... s[14:15]` (107), `s_cbranch_vccnz` back (108). I.e. wait until `*P >= y`, then `s_barrier` (112) for the block.
- Lines 1715-1730: at the end thread 0 does `global_atomic_add_u64 v1, v[0:1], s[12:13]` (1730) with the value `s_bcnt1_i32_b32` (1727), i.e. adds 1 per finishing block.
- Same pattern in `k_hc6_fused<1>` (spin compares at lines 99-106, atomic add at 2436) and in `k_hc6_fused3<1>`, which has two counters (spin on `s[18:19]` at 103-110 and `s[14:15]` at 1728-1735; atomic adds at 2441 and 2771 on `s[12:13]` and `s[16:17]`).

Meaning: P is a monotonically increasing u64 completion counter in device memory and `y` is the value it must have reached before this kernel's blocks start (a device-side dependency barrier between fused launches, a timeline semaphore); every block adds 1 when done, so the next kernel waits for `y = previous + number of blocks`. The data outputs of these kernels are the `bf16*` and `float*` arguments. Host context (inferred): "hc" is a per-layer residual-mix read ("hc_read attn" / "hc_read mlp" in 01946cf0.c:23,29; `hc_read: the deferred scatter's fused launch did not run` at 0189abe0.c:283); gating env vars `HALOGEN_HC_Q8_FUSE`, `HALOGEN_HC_Q8_DEC`, `HALOGEN_HC6_FUSE`, `HALOGEN_HC8_FUSE`. The 6 and 8 correspond to q6g64 and q8g64 hc weights (0189abe0.c:~165-180 tests per-layer slots at `+0x780`/`+0x790`/`+0x7a8` and `+0x960`/`+0x970`/`+0x988` under `HALOGEN_HC_Q8_FUSE`; slot-to-dtype match is inferred). Where the host allocates P and computes `y` is not traced.

## U21 part 4: where the power-of-two group scale comes from

Verdict: CLOSED, negatively. There is no power-of-two group scale decoded by `v_exp_f32` in the lq/hc kernels. The earlier inference in `kernels-dequant.md` s1 is wrong.

Evidence:
- `k_lq4w_grp<1,16,1,0>` lines 486-513: `s_mul_f32 s13, s10, 0xbfb8aa3b` (value times -log2(e)), `v_exp_f32 v0, s13` (490), `v_add_f32 v0, 1.0, v0` (493), then a full reciprocal sequence (`v_div_scale/v_rcp/v_fma/v_div_fixup`, 494-511) and `global_store_b32` (513). That is `1/(1+exp(-x))`, a sigmoid epilogue, with a scaled variant at lines 518-529 (`s_mul_f32 s13, s6, s13` then the same exp). The epilogue is chosen at runtime from the `LqGrp` mode integer (`s_cmp_lt_i32 s7, 3`, `s_cmp_gt_i32 s7, 3/4`, `s_cmp_eq_u32 s7, 5`, lines 436-445).
- Across all 573 slices that contain `v_exp_f32`, each also contains the constant `0xbfb8aa3b` or `0x3fb8aa3b` (grep loop, no file lacks it). `k_hcup4w` (64 `v_exp_f32`): each is preceded by `v_mul_f32 ... 0xbfb8aa3b` (slice lines 322-341), the same sigmoid/SiLU form.
- `v_ldexp_f32`: 0 slices. `v_lshlrev_b32 ..., 23` (exponent-field construction): 0 slices.
- Weight scales are fp16: `v_cvt_f32_f16` (lq4w_grp lines 172, 318) and `v_fma_mix_f32` in lq6w/lq8w; the size rules in `analysis/hgn-format.md` section 3 store fp16 scales.
- `_grp` does not mean "quantization group". `LqGrp` is passed by value (kernarg offsets 0x88, 0xd0, 0x118, 0x120 are 0x48 apart: repeating member entries) and the kernel loads several pointers from it (`s_load_b256 s[4:11], s[16:17], 0x18`); reading (inferred): a batch of linear members processed in one launch.
- The `v_div_scale/v_rcp` pairs are the sigmoid division. A separate `v_rcp_iflag_f32` after `v_cvt_f32_ubyte0` at lines 71-73 is an integer-to-reciprocal (likely a count normalisation; not decided).

Remaining: none for this claim. The open item "what does the mode integer encode" is part 2, point 5.

## U19 static part: the v_sad_u8 decode in k_ht_lin, k_ht_gemv_gu, k_i4r_rows_gl

Verdict: PARTIAL. The decode algorithm is recovered statically and is a computed code, not a stored codebook. The exact scale and sign conventions and the stream bit order need a real HT checkpoint.

### Decode, step by step (slice `k_ht_lin<3,1,b0,b0,b0,b0>`, lines 520-595)

1. Each lane loads 32-bit words from the weight stream. The row-block stride is `0x300` bytes for 3-bit (`v_mad_i64_i32 ... 0x300`, `s_mul_i32 ... 0x300` in the `_grp` slice, lines 505-534; 768 B = 2048 weights x 3 bits), and six `global_load_b32` use offsets 0,128,...,640 (one word per lane per 128 B). For 4-bit the matching stride would be 1024 B (2048 x 4 bits); I saw the constant `0x400` in the 4-bit slice but did not tie it to the stride (inferred).
2. A 64-bit bit-stream is formed per lane with `v_lshrrev_b64` (opcode D73D, `.long 0xd73d002f` at line 520; verified byte-exact with `llvm-mc`: `v_lshrrev_b64 v[47:48], v25, v[40:41]` encodes `2f 00 3d d7 19 51 02 02`). Reason (inferred): 3-bit codes do not align to 32-bit words.
3. 16-bit windows are cut with `v_bfe_u32 x, w, s, 16`. Counts per `k_ht_lin` slice (grep): 3-bit uses shifts 3, 6, 9 (16 each) plus the unshifted low 16 (`v_and 0xffff`); 4-bit uses shifts 4, 8, 12 (16 each) plus the unshifted low 16. The window step equals the bit width: a sliding-window (bitshift trellis) read with window length L = 16 bits and step K = 3 or 4 bits per weight. K is the first template argument, which confirms the 3/4 reading in `kernels-dequant.md` 3c by the window step, not only by the split of instantiations.
4. Each 16-bit window x is multiplied by the literal `0x83dcd12d`: slice lines 533-534 show `.long 0xd72c002c` followed by a literal `0x83dcd12d`. Verified with llvm-mc: `v_mul_lo_u32 v44, 0x83dcd12d, v44` encodes `2c 00 2c d7 ff 58 02 02 2d d1 dc 83`, matching the slice bytes (D72C002C, 020258FF, 83DCD12D). llvm-objdump 21 prints this as `.long` plus a mis-split second dword. The literal appears 64 times in the 4-bit `k_ht_lin` slice and 64 times in `k_i4r_rows_gl<1>` and `<2>`.
5. `v_sad_u8 v, v, 0, 0x6400` (line 543 etc.) sums the four bytes of the product and adds `0x6400`. `0x6400` is the fp16 value 1024.0, so the low bits become an exact fp16 `1024 + S` with `S` = byte0+byte1+byte2+byte3 of `(x * 0x83dcd12d) mod 2^32`, range 0..1020.
6. Two results are packed into one fp16x2 (`v_and 0xffff`, `v_lshl_or_b32 ..., 16`, lines 573-574) and converted with `v_pk_fma_f16 v37, v37, s6, 0xc931 op_sel_hi:[1,0,0]` (line 577): `w = (1024+S) * s6 + c`, `c` = fp16 `0xc931` = -10.383. The constant appears 32 times in each of the 3-bit and 4-bit slices.
7. `v_dot2acc_f32_f16` (line 582) then dots the fp16x2 weights with fp16 activations from LDS (`ds_load_b128`, line 596).

### What this is (inferred, with arithmetic)

The sum of the four bytes of a 32-bit value has mean 4 x 127.5 = 510 and standard deviation sqrt(4 x (256^2 - 1)/12) = 147.8 (arithmetic, not measured). `c = -10.383` equals `-(1024 + 510) / 147.8` = -10.379 within fp16 rounding (step 2^-7 near 10). So if `s6 = 1/147.8`, then `w = (S - 510)/147.8`: an approximately unit-Gaussian value computed from the 16-bit trellis state with one multiply and a byte sum. This is the structure of the QTIP "1MAD" computed code (multiply-hash of the state, byte sum as a Gaussian), with a different multiplier and no additive term visible; that comparison is from memory of the QTIP method, not from anything in this workspace. No codebook table exists in the kernel: between steps 3 and 6 there are no table loads; the only data loads are the weight words and the activations. The 3-bit and 4-bit variants share the same multiplier and bias; they differ only in window step (K) and stream stride.

Whether `s6` is exactly 1/147.8 or also folds in a per-tensor or per-row scale is not decided: `c` is a literal, so a different `s6` would shift the mean. This matters for a rebuild.

### su / sv fp16 vectors

From `kernels-dequant.md` 3a-3b (unchanged): `su` multiplies the activation before the 128-point Walsh-Hadamard butterfly (`v_fma_mix_f32` elementwise, `1/sqrt(128)` = `0x3db504f3`), and `sv` is applied on the output side (`k_ht_rot_out`, or inline in the fused GEMVs). Not decided: whether the stored values are +-1 random signs, learned scales, or sign-times-scale. A fp16 vector is stored either way, and no checkpoint is in the workspace.

### k_ht_gemv_gu and k_i4r_rows_gl

- `k_i4r_rows_gl<1>` and `<2>`: 64 `v_sad_u8 ..., 0, 0x6400` each and 64 occurrences of `0x83dcd12d` (grep). The sad pattern is identical to `k_ht_lin`, so the same multiply-and-byte-sum decode is used (inferred; the multiply is an undecoded `.long` in the dump, so a `v_mul_lo_u32` text grep gave 0). The window step for i4r was not read.
- `k_ht_gemv_gu<3|4,8,1>`: 128 `v_sad_u8` per `kernels-dequant.md`. I did not read its windows separately; same family, identical decode is inferred.

### What cannot be decided without a real HT checkpoint

1. Bit order inside the stream (increasing `v_bfe_u32` shift gives window i = bits [K*i, K*i+16) in the word, but the word order across lanes, the lane-to-column permutation and the tile order are not decided; the `.hgn` error text names "codebook, tile order or bits").
2. Exact `s6` (scale source) and whether row or group scales are folded in; which kernel argument feeds `s6`.
3. Sign and scale content of `su` and `sv`.
4. Whether the multiplier needs an additive term elsewhere (the slice shows only the multiply).
5. The meaning of qparam bit 0 for HT (the binder uses only `(qparam >> 1) & 0x7f`).

A test that closes all five: take one HT tensor from a real checkpoint, run the decode above on a few words on the CPU, and compare the rows with a known bf16 reference of the same tensor.
