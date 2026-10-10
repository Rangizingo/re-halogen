# Dequant / quantized-GEMM kernel family: objdump-level analysis

Phase 2, family 4 of 5. Evidence: 1141 per-kernel slices in
`work/kernels-dequant/` (llvm-objdump 21, `--mattr=+gfx1151`), instruction
classes in `tmp/stats-dequant.csv`, demangled signatures (`c++filt`), format
table in `analysis/hgn-format.md`. Method: group by family from the stats,
census opcodes, read targeted windows of the rotation kernels. Slice line
numbers refer to the `.s` files. Slice counts are template instantiations, not
distinct algorithms.

## 0. Family map

| family | slices | role | inner product |
|---|---|---|---|
| `k_lq4w` (+`_grp`, `_pair`) | 448 + 32 + 8 | 4-bit weight GEMV/GEMM | `v_dot2_f32_bf16` |
| `k_lq6k`, `k_lq6w` (+`_grp`) | 64, 32, 24 | 6-bit (K-block / wide) | `v_dot2_f32_bf16` |
| `k_lq8s32`, `k_lq8w` (+`_grp`) | 64, 32, 8 | 8-bit, scale per 32 | `v_dot2_f32_bf16` |
| `k_lbfw`, `k_lbfw_lds` | 16, 2 | bf16 weights, no decode | `v_dot2_f32_bf16` |
| `k_hcup4/6/8(w)`, `k_hc6/8_fused(3)` | 12+4+4+4+4+1, 8x4 | 4/6/8-bit hybrid-code weights, fused epilogue | `v_dot2_f32_bf16` |
| `k_i4r_*` | 19 | i4r rows: 4-bit rows with rotation plane (`_rot`, `_fold`, `_epi`, `_gu_epi`) | `v_dot2acc_f32_f16` |
| `k_ht_*` | 262 | Hadamard-transform (HT) weights and activations | `v_dot2acc_f32_f16`, `v_wmma` (MoE) |
| `k_q4moe`, `k_q4moe_items` | 20, 2 | 4-bit routed-expert GEMM | `v_wmma_f32_16x16x16_bf16` |
| `k_deq_*` (`q4cp`, `q4cp_v8/v8u/v32/probe/sel/f32`, `q6k`, `q6g64`, `q8g64`, `q8s32`, `aff`) | ~30 | standalone dequant to bf16 | none (convert + store) |

No `v_mfma` anywhere (gfx1151 has none). WMMA appears in `k_q4moe` (368 per
kernel on average), the `k_ht_moe_*` expert kernels, `k_ht_tg`/`k_ht_tg3` (4
slices), `k_dn_fused16` (17 slices, covered in `kernels-moe.md`) and
`k_hcup4w` (one slice). All other lq/hc/lbf dense and decode kernels use
packed dot2 on the VALU.

Mapping to `.hgn` dtypes (`analysis/hgn-format.md`): `q4c`/`q4a`/`iq4nl` ->
`k_lq4w`/`k_deq_q4cp*`; `q6k`/`q6g64` -> `k_lq6k`/`k_lq6w`/`k_deq_q6*`;
`q8s32`/`q8g64`/`q8_0` -> `k_lq8*`/`k_deq_q8*`; `bf16` -> `k_lbfw`; `i4r` ->
`k_i4r_*`; `ht` -> `k_ht_*`. `k_deq_aff` is an affine (scale + zero) 4-bit
decode. These pairings come from kernel names and signatures, not from a
traced host dispatch table (bounded open).

## 1. The common shape of an `lq*` kernel

Demangled, `k_lq8w<1,16,1>(uint8* W, bf16* x, bf16* y, long, long)` and
`k_lq4w<1,16,1,false,0,0>(uint8*, bf16*, void*, long x5, void*, float)`.
Weights are raw bytes, activations are bf16, output bf16 (or a `void*`
epilogue target). Per kernel:

- Unpack with `v_bfe_u32` / `v_and_or_b32` / `v_lshrrev_b32` (4-bit and
  6-bit), or `v_cvt_f32_ubyte0..3` (8-bit and 6-bit after widening).
- Scale with `v_fma_mix_f32` (fp16 scale times f32) or `v_cvt_f32_f16` then
  `v_mul_f32`.
- Accumulate with `v_dot2_f32_bf16`: two bf16 products per lane per
  instruction against the activation row.
- Reduce across lanes with `ds_*`/DPP (LDS 23 to 53 ops per kernel).

Op census (top, per representative kernel): `k_lq8w` 8 `dot2`, 17 `bfe`,
16 `fma_mix`; `k_lq6w` 16 `dot2`, 33 `bfe`, 32 `cvt_ubyte0`, 32 `fma_mix`;
`k_lq4w_pair` 32 `dot2` (two weight tensors per pass: gate and up);
`k_lbfw` 8 `dot2` and nothing else (no unpack). The `_grp` variants take an
`LqGrp` struct by value and add `v_div_scale`/`v_readfirstlane` (16 and 18
in `k_lq4w_grp`) for per-group scale setup. `k_hc6_fused`/`k_hc8_fused`
carry `unsigned long long*` outputs plus a `u64` argument: a fused epilogue
that writes a packed result (inferred: argmax/top-k key, not traced).

The `_grp` kernels, `k_hcup4w` and `hc*_fused` also run `v_exp_f32`
(8 in a `k_lq4w_grp` slice, 64 in `k_hcup4w`, grep), plus
`v_div_scale/v_rcp` pairs. A dequant kernel has no use for a softmax
exponential, so the likely reading is a power-of-two group scale decoded as
2^e (inferred, the exponent source is not traced). Stats CSV note: its `exp`
column counts `v_exp` plus `v_log`; `rcp` counts `v_rcp`.

## 2. `k_q4moe`: the routed-expert tensor-core path

`k_q4moe<1,0,128,Tune<0,128>,0>(Args)` is the largest family member (5,472 to
41,495 instrs). Half of the 20 slices use bf16 WMMA and half use f16 WMMA
(10 and 10 by grep for `v_wmma_f32_16x16x16_bf16` vs `_f16`). The 41,495-
instruction variant is an f16 one: 576 `v_wmma_f32_16x16x16_f16`, about 2,030
`v_perm_b32`, about 1,380 `v_permlanex16_b32` (counts from the independent
review; my first pass quoted a bf16 census that does not match that variant).
Reading: 4-bit expert weights are expanded with byte permutes (`v_perm_b32` is
the nibble expansion trick) and fed to 16x16x16 WMMA tiles; `permlanex16`
shuffles tiles between the two 16-lane halves of a wave32.
`k_q4moe_items<128>(int*, int*, int*, int, Item*, Item*, int*)` (309 instrs,
no WMMA) builds the per-expert work list (token counts to `Item` records), so
tokens are grouped by expert before the GEMM. The tile width 128 matches the
128-element Hadamard block in section 3. This is the dense-MoE counterpart of
`k_dn_fused16` in `analysis/kernels-moe.md`.

## 3. HT tensors carry a rotation plane (claim closed)

Claim: HT-quantized tensors are stored in a Hadamard-rotated basis and the
engine applies the rotation to activations on the fly. Evidence, in four
parts.

**3a. Rotation kernels exist and are exactly a signed Hadamard.** Six small
kernels, all with the same skeleton:

```
k_ht_rot_in     (bf16* x, half* s, half* y, int n)
k_ht_rot_out    (bf16* y, half* s, int n, int m)
k_ht_rot_rows   (bf16* x, half* s, bf16* y, int n, int m)
k_ht_rot_rows_h (bf16* x, half* s, half* y, int n, int m)
k_ht_rot_out_f32(float* x, half* s, bf16* y, int n, int m)
k_ht_tg_sum     (float* x, int, half* s, bf16* y, int, int)
```

In `k_ht_rot_in` (168 lines, slice `..k_ht_rot_in..s`):

1. Two `global_load_b64`: four bf16 of `x` (`v_and_b32 0xffff0000` /
   `v_lshlrev_b32 16` widen bf16 to f32) and four fp16 of the vector `s`,
   indexed `v1 << 8 | lane << 3`, i.e. one 128-element fp16 vector per
   block, 256 bytes (slice lines 58-59).
2. `v_fma_mix_f32 ... op_sel` multiplies x by s elementwise (the diagonal
   sign/scale), with the `+/-` pair formed in the same chain: the first
   butterfly stage inside each lane (`v_add_f32` / `v_sub_f32`).
3. Four stages of `v_mov_b32_dpp ... row_xmask:1/2/4/8` with
   `v_cndmask_b32 -v, v` (negate by lane bit) then `v_add_f32`. That is
   the lane-exchange butterfly of a fast Walsh-Hadamard transform: 4 stages
   across 16 lanes, plus `v_permlanex16_b32` (4 per kernel, lines 141-153) for the 5th
   stage across the two lane halves. Stages in-lane (4 values) + 5 across
   lanes = log2(128) = 7 stages, a 128-point WHT.
4. Constant `0x3db504f3` = 0.08838835 = 1/sqrt(128) scales the result
   (one use in `rot_in`, line 30; 3 to 4 in the others), so the transform is
   orthonormal.
5. Row addressing `n * 20` and stride `0x1400` bytes: 20 blocks x 128 = 2560
   elements per row, equal to the template parameter of
   `k_i4r_rows<2560>`, i.e. the model hidden size 2560.

So `y = H128 (x (.) s) / sqrt(128)` per 128-block: a randomized Hadamard
rotation (random-sign diagonal times Hadamard). `rot_out` is the same
butterfly with the diagonal applied after (it takes `bf16* y` as the in-place
target); `_f32`/`_h`/`_rows` differ only in input/output precision and row
count. This is the QuIP#-style incoherence-processing pattern.

**Reconciliation with `kernels-moe.md`.** That older writeup (section 1.1) reads
the same constant `0x3db504f3` as 1/sqrt(512) and the rotation as a 32-point
Hadamard per 32-element group. The constant is 0.08838835, which is 1/sqrt(128)
(and equals 1/16 times sqrt(2), the parenthetical it quotes). The butterfly
adds 2 in-lane stages (each lane holds 4 elements), 4 DPP stages and 1
`permlanex16` stage: 2 + 4 + 1 = 7 stages = a 128-point transform, and the
orthonormal normalization of a 128-point WHT is 1/sqrt(128). I take 128-point
as correct; a correction note is added to `kernels-moe.md`.

**3b. The GEMV kernels consume the rotated basis.** `k_ht_lin<3|4,...>`
(`uint32* Wq, bf16* x, int x4, half* su, half* sv, float* out, uint32*, void*,
bf16*`) and `k_ht_gemv_gu<3|4,8,1>(uint32*, half*, HtItem*, int*, half*,
half*, half*)` take **two fp16 vectors**, which are the per-block vectors of
section 3a (`su`/`sv`: input side and output side), and inline the same WHT:
`k_ht_lin` has 48 `dpp` in the first slice, 12 to 68 `permlane` across slices,
7 uses of `0x3db504f3` and 16 `fma_mix`; `k_ht_gemv_gu` 96 `dpp`, 16
`permlane`, 16 uses of the constant; the expert kernels `k_ht_moe_gu`/`_dn`
768 and 256 `dpp`, 125 and 44 uses. Weights arrive as packed `uint32` and are
decoded with `v_sad_u8` (64 in `k_ht_lin`, 128 in `k_ht_gemv_gu`) and dotted
with `v_dot2acc_f32_f16` (21 to 137 per `k_ht_lin` slice, 50 to 53 per
`k_ht_gemv_gu` slice; ranges per the independent review, they vary with the
instantiation). Rotating the activations by H then taking the dot product with
a stored weight that was quantized in the rotated basis equals the original
product, which is the reason to store the plane.

**3c. The first template argument is 3 or 4** across the GEMV and MoE `k_ht_*`
families, mostly in an even split (`k_ht_lin` 32/32, `k_ht_gemv_gu_t` 20/20,
`k_ht_moe_*` 3/3), but not everywhere: `k_ht_lin_grp` is 16/17, `k_ht_items_group`
is 2/4 and the `k_ht_rot_*` kernels have no template arguments. Reading: bits
per weight, so HT is a 3-bit and a 4-bit format sharing one kernel set.
Inferred from the split and the `.hgn` error string "form the engine does not
read (codebook, tile order or bits)" in `analysis/hgn-format.md`; the decode
through `v_sad_u8` is not reversed (bounded open).

**3d. Companion tensors.** The `.suh` / `.svh` companion rule in
`analysis/hgn-format.md` (HT tensors travel with two fp16 vectors) matches
the two `_Float16 const*` arguments of every `k_ht_lin`/`gemv`/`moe` kernel
and the one in each `rot` kernel. The rotation plane is those vectors plus
the fixed Hadamard; only the signs/scales are stored, the matrix is code.

Verdict: **closed.** HT = randomized 128-point Hadamard rotation of
activations (`rot_in`/`rot_out`, or inlined in the fused GEMVs) against weights
stored rotated and 3/4-bit quantized, with per-block fp16 vectors as the
only stored rotation data. Not proven: that `su`/`sv` are random signs
rather than learned scales (a fp16 vector is stored either way).

## 4. `k_i4r_*`: 4-bit rows with a rotation plane

`k_i4r_rows<2560>(uint8* W, long stride, int, half* x, int, int* idx, float*
out)` is specialised on the 2560 hidden size. Op census: 321 `v_or_b32`, 320
`v_perm_b32`, 320 `v_mul_i32_i24`, 269 `v_dot2acc_f32_f16`. Nibble expansion
by `v_perm_b32`, integer pre-multiply by a 24-bit scale (`v_mul_i32_i24`),
packed fp16 dot. The `_gl`/`_fl` variants add `I4rSd` (a scale/seed struct)
and `v_sad_u8` (64, same decode primitive as HT). `k_i4r_fold` (348 instrs,
4 `permlanex16`) folds partial sums across the two lane halves, `k_i4r_gu_epi`
and `k_i4r_rows_epi` are gate/up epilogues (4 `exp`+`rcp` pairs: a SiLU or
sigmoid, `v_exp` + `v_rcp` is the standard `1/(1+exp(-x))`). `k_i4r_rows_rot`
(2,325 instrs, 81 LDS ops) applies the rotation inline. `0x3db504f3`
(1/sqrt(128)) appears in 17 of the 19 `k_i4r_*` slices (grep), so i4r uses
the same 128-point orthonormal Hadamard as HT: both formats carry a rotation
plane. HT adds the 3/4-bit `v_sad_u8` code path and the MoE tiles; i4r is the
4-bit-rows variant specialised to hidden size 2560.

## 5. Claims closed and bounded open

| claim | verdict | evidence |
|---|---|---|
| HT tensors carry a rotation plane | closed | section 3a-d |
| lq/hc/lbf decode kernels dot in bf16 on the VALU, no tensor cores | holds with one exception: `k_hcup4w` (1 slice, 32 WMMA) | grep: 1 of 791 `k_lq*`/`k_hc*`/`k_lbf*` slices has `v_wmma` |
| i4r carries the same rotation plane as HT | holds | `0x3db504f3` in 17 of 19 i4r slices, 257 of 262 HT slices |
| MoE expert 4-bit path uses WMMA | holds, in both bf16 and f16 variants | `k_q4moe`: 10 bf16 and 10 f16 slices |

Bounded open:

1. `v_sad_u8` weight decode in `k_ht_*` and `k_i4r_*_gl` (what the byte
   absolute-difference trick looks up; codebook not recovered).
2. Meaning of the remaining template arguments (`<1,16,1,...>` in `lq*`).
3. Dtype-to-kernel dispatch table (host side, not traced).
4. `su`/`sv` are signs or scales (stored values not inspected, no HT
   checkpoint is in this workspace).
5. `hc6/8_fused` `u64` output meaning; per-slice `exp`/`rcp` split for the
   `_grp` variants.
