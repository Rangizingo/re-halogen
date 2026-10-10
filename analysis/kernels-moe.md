# halogen-flash-server — MoE + quantized-GEMM kernel family (objdump-level)

Evidence base: full disassembly `analysis/kernels-disasm/codeobj_53000.s` (primary code
object, 1491 kernels; addresses below are its VAs), `codeobj_14aa000.s` (k_dn_fused16),
symbol inventory `work/fatbin-manifest.json`, slices `work/kernels/*.s`. All instruction
sequences quoted are verbatim from the disassembly. Demangled names use the Itanium
mangling of the manifest.

Conventions used below:
- **4-bit→f16 magic**: `v_sad_u8 v, v, 0, 0x6400` = per-byte add of fp16 1024.0
  (0x6400) into each of 4 nibbles of a u32; `v_pk_fma_f16 v, v, s, 0xc931` = packed
  fp16 FMA with immediate pair encoding −1032 (0xc931 = fp16 −1024, 0x3f81-ish pair),
  i.e. the classic `nibble + 1024 − 1032` trick that yields signed nibble−8 in f16 lanes.
  This pair is the fingerprint of every 4-bit weight decode in the engine.
- **SiLU fingerprint**: `v_mul_f32_e32 v, 0xbfb8aa3b, x` (= −log2e/2, so the
  hardware `v_exp_f32` — an exp2 — evaluates e^{−x/2}), `v_add_f32_e32 v, 1.0, v`
  (1+e^{−x/2}), then IEEE division sequence
  `v_div_scale_f32 / v_rcp_f32_e32 / v_div_fmas_f32 / v_div_fixup_f32` (or
  `v_fma_f32 -d,r,1.0; v_fmac_f32 r,e,r` Newton refinement) →
  **x/(1+e^{−x/2})** — SiLU with the half-factor baked into the exp constant.
- **Hadamard fingerprint**: `v_mov_b32_dpp ..., row_xmask:N row_mask:0xf bank_mask:0xf`
  (N = 1,2,4,8) + `v_add_f32_e32`/`v_sub_f32_e32` (or `v_dual_add_f32 :: v_dual_sub_f32`)
  + `v_cndmask_b32_e64 v, -v, v, s` (sign flip from an SCC bit = Rademacher sign row)
  + `v_permlanex16_b32 v, v, s, 0xfedcba98` (cross-half exchange = the 5th/final
  butterfly stage, pairing lanes l and l+16) + final `v_mul_f32_e32 v, 0x3db504f3, v`
  (= 0.0441942 = 1/√512 = (1/√32)·(1/4), i.e. the 32-point FWHT normalization with a
  1/4 fold; the DF16 plane scales absorb the rest).
  That is a fast Walsh–Hadamard transform over the 32-lane wavefront (4 in-row DPP
  butterfly stages, xmask 1/2/4/8, + 1 cross-half permlanex16 stage), normalized
  in-kernel.

---

## 1. HT family (dtype `ht` — "rotation-plane" quant) — the acceptance claim

### 1.1 The rotation is a real FWHT (Hadamard), applied in-kernel

`k_ht_rot_rows(const unsigned short*, const DF16*, int, int)` @ 0x22ce00 (codeobj_53000):
- Loads 4 f16 pairs: `global_load_b64 v[5:6]/v[7:8]` (0x22d260/0x22d268), widens with
  `v_and_b32_e32 v5, 0xffff0000, v5` + `v_fma_mix_f32 v5, v7, v5, neg(0)` (f16→f32
  multiply by the DF16 plane scale).
- First butterflies are fused into the widening FMA:
  `v_fma_mix_f32 v15, v7, v10, v5` / `v_fma_mix_f32 v5, v7, v10, -v5` (0x22d2ac/0x22d2b4)
  — add and subtract of the same pair = butterfly stage 0 with sign select.
- Stages 1–3: `v_mov_b32_dpp v6, v10 row_xmask:1` → `v_add_f32_e32 v6, v10, v6`
  (0x22d2f4/0x22d32c), then `row_xmask:2` (0x22d344), `row_xmask:4` (0x22d390),
  `row_xmask:8` (0x22d3f4) — the log2 butterfly ladder.
- Stage 4 (lanes 0–15 ↔ 16–31): `v_permlanex16_b32 v7, v7, s2, 0xfedcba98`
  (0x22d488) + `v_add_f32_e32 v0, v0, v12` (0x22d4b0).
- Sign masks: `v_and_b32_e32 v12, 1, v0` / `v14, 4, v0` / `v13, 2, v0` / `v0, 8, v0`
  (0x22d2d0–0x22d350) — lane-index bits feeding `v_cndmask_b32_e64 v, -v, v, s`
  sign flips (Walsh/Rademacher ordering), and `v_xor_b32_e32 v6, s6, v8` (0x22d0d8)
  for the plane sign vector.
- Normalization: `v_mul_f32_e32 v6, 0x3db504f3, v6` (0x22d4f4) = ×1/√512 (1/16·√2).
  **Correction (2026-10-10, see `kernels-dequant.md` s3a):** 0x3db504f3 = 0.0883883 = 1/√128, not 1/√512 (the quoted 1/16·√2 is the right value). With 4 elements per lane the ladder is 2 in-lane + 4 DPP + 1 permlanex16 stages, so the transform is 128-point, not 32-point.

Same ladder in `k_ht_rot_in` @ 0x22ca00 (f16→DF16, `v_fma_mixhi/lo_f16` repack),
`k_ht_rot_out` @ 0x22d600, `k_ht_rot_out_f32` @ 0x22de00, `k_ht_rot_rows_h` @ 0x22ef00.
`k_ht_rot_rows` reads a row of f16 (lane = element index, `v_and_b32_e32 v9, 31, v0`
0x22d1d8) and writes rotated f16 — i.e. **the rotation plane is a 32-point Hadamard
(Walsh) transform applied per 32-element group of the K dimension** (4 DPP stages
inside each 16-lane row + the cross-half permlanex16 stage), with the sign vector
supplied per tensor (the `PKDF16` = DF16 plane/scale operand).

### 1.2 HT dequant kernels

`k_ht_deq_orig<3|4, bool, 8|16>(const unsigned*, int, const DF16*, const DF16*, unsigned short*, int)`
@ 0x49fa00 (`<3,true,8>`):
- 4-bit decode: nibble masks (`v_and_b32_e32 v12, 0xff, v6` 0x49fbcc), then
  `v_sad_u8 v14, v14, 0, 0x6400` (0x4a0654) + `v_pk_fma_f16 v37, v37, s3, 0xc931`
  (0x4a0d9c) → signed nibble−8 f16 pairs; `v_cvt_f32_f16_e32` widens.
- **Hadamard in LDS**: the 16 dequantized f16 lanes are staged with
  `ds_store_b32 v90, v14 … v103, v52` (0x4a0ef0–0x4a0f38), re-read with
  `ds_load_b32 v94…v104` (0x4a142c–0x4a147c), then the butterfly:
  `v_dual_add_f32 v105, v73, v74 :: …` (0x4a1488) followed by
  `v_dual_sub_f32 v73, v73, v74 :: v_dual_add_f32 v74, v75, v76` (0x4a1494) and the
  full add/sub cascade `v_add_f32_e32 v105, v73, v75` / `v_sub_f32_e32 v73, v73, v75`
  (0x4a1550/0x4a1554). The LDS round-trip is the transpose that turns the
  lane-parallel butterfly into a per-row transform.
- Sign flips: `v_xor_b32_e32 v9, v4, v20` (0x49fe08) and the cascade at
  0x49fe60–0x49fef4 (8 lanes × 3 sign vectors v20/v5/v8) — the rotation-plane sign
  pattern applied to the nibble stream.
- Output: `global_store_d16_hi_b16 v[26:27], v21, off` (0x4a3184) — f16.

`k_ht_deq_t<3>(const unsigned*, int, int, unsigned short*)` @ 0x548700:
transposed (tile) variant; ends with the bf16→f16 rounding trick
`v_add3_u32 v6, v6, v0, 0x7fff` (0x548f6c, ×8 lanes) and a single
`global_store_b128 v[4:5], v[0:3], off` (0x548fe4) — 8 f16 per lane.
`k_ht_deq_t<4>` @ 0x549000 adds `v_and_b32_e32 v8, 0xf8, v2` / `0x7c` masks
(0x5493e4/0x549454) = e5m2/e4m3 (fp8) field extraction — template arg 4 = fp8-plane
dequant, 3 = 4-bit-plane dequant.

### 1.3 HT MoE GEMMs

`k_ht_moe_gu3<3,1,0>(const unsigned*, const DF16*, const HtItem*, const int*,
const DF16*, const DF16*, DF16*)` @ 0xa86900 (15718 instrs):
- Tile setup: `s_min_i32 s27, s0, 0x80` (0xa8693c) → tile width 128;
  `v_mul_u32_u24_e32 v88, 0x48, v5` (0xa86988) → **HtItem stride = 72 B**;
  row addressing `v_mad_i64_i32 v[1:2], null, 0x1400, v1, v[78:79]` (0xa86a34)
  → HT weight row stride 5120 B (= 128 lanes × 40 B/lane: 32 B of 4-bit quads +
  8 B DF16 scales + 0.5 B/lane sign-plane, rounded to the 72-B HtItem descriptor).
- LDS staging: `ds_load_b128 v[100:103], v160` at offsets 0/16/2304/2320/4608/…
  (0xa86e84–0xa86ebc), 88 × `s_barrier`.
- Matrix cores: `v_wmma_f32_16x16x16_f16 v[58:65], v[164:171], v[100:107], v[58:65]`
  (0xa8717c) — 36 WMMA 16×16×16 f16 ops.
- **Rotation applied in-kernel, fused into the accumulator epilogue**:
  `v_mov_b32_dpp v68, v70 row_xmask:1` + `v_add_f32_e32 v65, v67, v69` +
  `v_sub_f32_e32 v67, v67, v69` + `v_cndmask_b32_e64 v60, -v60, v60, s4`
  (0xa873b8–0xa873f8), `row_xmask:4` at 0xa87668, `row_xmask:8` at 0xa8769c,
 `v_permlanex16_b32 v70, v70, s33, 0xfedcba98` (0xa87538), and the
 1/√512 normalization `v_mul_f32_e32 v59, 0x3db504f3, v59` (0xa877ac).
  1728 DPP + 464 permlanex16 + 2232 sign-cndmask = the full FWHT over the
  gate and up accumulators.
- **Fused gate/up + SiLU in the same kernel**: after the rotation,
  `v_mul_f32_e32 v68, 0xbfb8aa3b, v69` → `v_exp_f32_e32 v68` →
  `v_dual_add_f32 v68, 1.0, v68` (0xa876d8/0xa876fc/0xa87728) →
  `v_rcp_f32_e32 v83, v80` + Newton (`v_fma_f32 v90, -v80, v83, 1.0`,
  `v_fmac_f32_e32 v83, v90, v83`, 0xa877b8/0xa877d0) → gate·silu(up) product,
  repacked `v_fma_mixhi_f16 v62, v62, s5, 0` / `v_fma_mixlo_f16 v64, s5` and
  `global_store_b64 v[59:60], v[61:62], off` (0xa87c4c). One pass: decode →
  WMMA → rotate → SiLU → store.

`k_ht_moe_dn3<3,1,0>` @ 0xae6d00 (down proj, 11178 instrs): same structure minus
SiLU — 1152 DPP, 320 permlanex16, 36 WMMA f16, 88 barriers, `global_store_b64`
0xae7c88. `k_ht_moe_dn2x<3,64,1,0>` @ 0xb33a00 (13055 instrs): 160 WMMA f16,
512 `v_sad_u8` + 256 `v_pk_fma_f16 … 0xc931` (4-bit decode), 640 DPP,
416 permlanex16, 48 barriers; store `global_store_b64 v[61:62], v[63:64], off`
(0xb37814). `k_ht_moe_gu2<3,64,0>` @ 0xab8400: 160 WMMA, 512 sad_u8, 80 SiLU
exp/rcp pairs. `k_ht_moe_dn2<3,64,0>` @ 0xd75d00, `k_ht_moe_dn2w<3,64,6,0>` @
0xb5c900, `k_ht_moe_gu2w<3,64,6,0>` @ 0xd47000, `k_ht_moe_dn<3,8,1>` @ 0xd94c00
(16 WMMA, 2 barriers — 8-wide tile), `k_ht_moe_gu<3,8,1>` @ 0xd8c300 (64 SiLU exps).

`k_ht_gemv_gu_t<3,8,4,3>(const unsigned*, const DF16*, const int*, const DF16*,
const DF16*, DF16*, HtSd, int)` @ 0x9e0200 (GEMV = decode-side transposed, token
batch 1): 320 `v_sad_u8 v14, v14, 0, 0x6400` (0x9e0eb0) + 160
`v_pk_fma_f16 v95, v95, s5, 0xc931` (0x9e0f64) = 4-bit decode; 176+55
`v_dot2acc_f32_f16` (0x9e0408 region) = dot-product accumulation (no WMMA —
GEMV path); butterfly `v_mov_b32_dpp v13, v2 row_xmask:1` (0x9e0434),
`v_cndmask_b32_e64 v6, -v6, v6, s1` (0x9e0444),
`v_permlanex16_b32 v11, v11, s54, 0xfedcba98` (0x9e05d0). `HtSd` passed by value
(16 B struct: `s_load_b64 s[8:9], s[0:1], 0x7c` 0x9e022c).
Siblings: `k_ht_gemv_dn_t<3,8,3,b>` @ 0x9ed900 (HtFold struct),
`k_ht_gemv_dn_ts<3,8,1,F>` @ 0x94e200, `k_ht_gemv_gu_ts<3,8>` @ 0x93f500,
`k_ht_gemv_gu<3,8,1>` @ 0xa7c100 (64 SiLU exps = fused gate/up+SiLU GEMV),
`k_ht_gemv_dn<3,8,1>` @ 0xa83300, `k_ht_gemv_dn_g<3,8,4,0>` @ 0x91ff00,
`k_ht_gemv_gu_gs<3,8,4>` @ 0x8fe800.

`k_ht_lin<3,8,T,T,F,F>(const unsigned*, const unsigned*, int,int,int,int,
const DF16*, const DF16*, float*, unsigned*, void*, const unsigned*)` @ 0x519d00
(trunk linear on HT tensors, f32 out): 112+84 `v_dot2acc_f32_f16`, 64 sad_u8,
48 DPP butterfly (0x519ea8), 68 permlanex16, LDS tiles at 5120-B stride
(`ds_load_b128 … offset:5120/10240/…/35840` 0x51ab44–0x51ae04 = 8×5120 = 40960 B
LDS window), `v_mad_i64_i32 … 0x300 …` (0x51a628) row stride 768.
`k_ht_lin_grp<3,8,T,T,F,F>(HtGrp, const unsigned*, int, int, float*, unsigned*)`
@ 0x408600: HtGrp by value (pointers at +0x130, ints at +0x138, +0x198 —
struct ≥ 0x1a0 B); 94+92 dot2acc, 64 sad_u8, butterfly at 0x40879c.

`k_ht_tg<3>` @ 0x4dd800 / `k_ht_tg3<3>` @ 0x4b3b00: 80 WMMA f16 + 256 sad_u8 +
320 DPP — HT "token-gate" (tied gate) GEMM. `k_ht_tg_sum` @ 0x22e600: pure
f32 accumulation (`global_load_b128`, `v_add_f32`).

### 1.4 HT bookkeeping

`k_ht_items_group<2>(const int*, int, HtItem*, const int*, int*, int*, int*)`
@ 0x8f8c00: builds per-group HtItem lists; `v_mul_u32_u24_e32 v1, 0x199a, v0`
(0x8f8f78) = index × 6554 (6554 = 0x199a; consistent with a 6554-B group record /
magic-multiplied division, exact form open). `global_store_b128 … offset:32`
(0x8f9500) writes 32-B item headers. `k_ht_items_slots` @ 0x22f700.
`k_route_topk_w_grp<2|4>(const float*, int*, float*, int, HtItem*, HtItem*,
const int*×4)` @ 0xfb9b00: router top-k that *emits HT item lists* — 40
`v_bfrev_b32_e32` (bit-reversal rank), bitonic `v_cmpx_eq_f32` ladder,
softmax exp/rcp, 27 `global_store_b128`.

**Acceptance verdict**: HT tensors carry a rotation plane — confirmed. The plane is
a 32-point fast Walsh–Hadamard transform (DPP butterfly ladder + permlanex16 stage +
Rademacher sign flips via `v_cndmask_b32_e64 v,-v,v,s` / `v_xor_b32_e32` + 1/√512
normalization), applied **in-kernel** in every `k_ht_*` consumer (dequant, lin,
moe gu/dn, gemv), with per-32-element sign vectors and DF16 scales supplied per
tensor via the `PKDF16` args and the 72-B `HtItem` descriptors.

---

## 2. Quantized linear kernels (trunk/draft linears)

### k_lq8w<N,16|32,M>(const unsigned char*, const unsigned short*, unsigned short*, long, long)
@ 0x2e0300 (`<1,32,1>`, 293 instrs). **fp8r** (fp8 row-scale):
`global_load_b128` weight bytes → `v_cvt_f32_ubyte0..3_e32` (u8→f32, 4/lane) →
`v_fma_mix_f32` with row scale → `v_dot2_f32_bf16` ×8 against bf16 activations
(packed via `v_add3_u32 v, v, b, 0x7fff` bf16 rounding) → `ds_bpermute_b32` ×5
cross-lane reduction. No matrix cores: 8 dot2 = memory/decode-bound.

### k_lq4w<N,16|32,M,b,b,b>(const unsigned char*, const unsigned short*, void*, long×5, const unsigned short*, float)
@ 0x54d600 (`<1,32,1,F,F,F>`, 315 instrs). **q4c/q4a/q5a family** (4-bit,
interleaved; the 5–6 `long` args = M/N/K/lda/ldb + group size; trailing float =
global scale; `S6` = 6th pointer = fp16 scale array):
- nibble unpack: `v_and_b32_e32 v38, 0xff, v1` / `v_lshrrev_b32_e32 v39, 6, v1` /
  `14` / `22` then `v_and_b32_e32 v39, 0x3fc, v39` (0x54d864–0x54d934) —
  8 nibbles per u32 gathered to LDS byte addresses (`×4` stride = f32 LDS rows);
- `ds_load_b32` ×16 = LDS-resident 4-bit block (the "interleaved" repack:
  nibbles pre-gathered so one LDS read feeds 8 MACs);
- `v_dot2_f32_bf16` ×16 accumulate; `v_cvt_f32_f16_e32` (0x54d7cc) for the
  fp16 group scale; `v_xor_b32_e32` ×5 = sign nibble (q4a asym zero-point).
`k_lq4w_grp<N,W,M,b>(LqGrp, const unsigned char*, long)`: same decode with
group descriptor struct; `k_lq4w_pair<N,16,b>` = paired-K variant.

### k_lq6k<N,16|32,M,b>(const unsigned char*, const unsigned short*, void*, long, long)
@ 0x597900 (`<1,32,1,F>`, 339 instrs). **q6k** (6-bit K32-packed):
`v_bfe_u32 v53, v26, v9, 4` with *variable* bit offsets (0x597b3c–0x597b54) —
6-bit fields straddling byte boundaries extracted by dynamic `v_bfe` (34 bfe
total), `v_cvt_f32_i32_e32` ×17, `v_subrev_nc_u32_e32` ×16 = −32 zero-point,
`v_mul_f32_e32` ×14 scale, `v_dot2_f32_bf16` ×8.

### k_lq8s32<N,16|32,M,b>(const unsigned char*, const unsigned short*, void*, long, long)
@ 0x581000 (`<1,32,1,F>`, 310 instrs). **q8s32** (8-bit group32, fp16 scales):
`v_bfe_i32 v11, v1, 0, 8` / `8, 8` / `16, 8` (0x581198–0x5811a8) = sign-extended
s8 lanes, `v_cvt_f32_i32_e32` ×16, `v_fma_mix_f32` ×16 with fp16 group scale,
`v_dot2_f32_bf16` ×8.

### k_linear_q4_m<16,1,256>(const unsigned char*, const unsigned short*, unsigned short*, long×5)
@ 0x700100 (966 instrs): tiled q4 GEMM, f16 out. `ds_load_2addr_b32` ×96 +
`ds_store_2addr_stride64_b32` ×72 (double-buffered LDS tiles),
`global_load_u8` ×16 + `global_load_d16_b16` ×16 (weights + fp16 scales),
`v_bfe_u32`/`v_and`/`v_lshrrev` nibble decode, `v_fma_mix_f32` ×16.
`k_linear_f32_q4_mt<8,2,4,256>` @ 0x76db00 (1724 instrs): same, f32 out, 4-tile M.
`k_linear_f32_q4`, `k_linear_f32_m<8>`: bf16 dense linears.

### k_deq_q4cp*(const unsigned char*, unsigned short*, long×5[, DeqCb])
`k_deq_q4cp` @ 0x226b00 (103 instrs, scalar tail), `k_deq_q4cp_v8/_v8u<N,b>`,
`k_deq_q4cp_v32<b>` @ 0x274a00 (759 instrs), `k_deq_q4cp_f32`, `k_deq_q4cp_sel`.
**q4c planar** (K%32): `ds_store_b128` staging of the 8-arg `DeqCb` struct
(0x274a5c), nibble gather `v_lshrrev_b32_e32 v10, 6, v1` + `v_and_b32_e32 v10, 60, v10`
(0x274c20–0x274c64: ×4 byte stride = 4-bit→LDS f32 addressing),
`v_bfe_u32` ×64, per-lane `v_mul_f32_e32 v12, v7, v12` (0x274cb0) by fp16 scale,
`global_store_b128` ×8 (8 f16 out per lane). Pure dequant (no MACs) — feeds
hipBLASLt bf16 GEMMs; the v32 kernel is the decode-bandwidth workhorse.

---

## 3. i4r (routed-expert 4-bit, group 128)

`k_i4r_rows<640|2560>(const unsigned char*, long, int, const DF16*, int, const int*, float*)`
@ 0xdff100 (`<640>`, 639 instrs):
- nibble→f16: `v_and_b32_e32 v92, 0xf0f0f0f, v45` + `v_lshrrev_b32_e32 v93, 4, v45`
  (0xdff41c/0xdff424) = even/odd nibble split;
  `v_perm_b32 v26, v39, v77, 0xc040c00` (0xdff428) = interleave nibbles into
  fp16 exponent slots; `v_or_b32_e32 v42, 0x64006400, v26` (0xdff44c) = +1024.0
  pair; `v_mul_i32_i24_e32 v1, 0xe408, v42` (0xdff47c) = packed f16
  (1024+n)·scale−bias fold; 80 perm + 80 or + 80 mul_i24 = 320 nibbles/lane.
- `global_load_d16_b16 … offset:8` (0xdff458) = fp16 group-128 scales;
  `ds_load_b128` ×20 (0xdff2c4, offsets 0/16/256/272/288/304 = 256-B row stride)
  = activation tile; `v_dot2acc_f32_f16` ×46 + `v_dual_dot2acc_f32_f16` ×21.
- `const int*` arg = expert row indirection (routed experts: rows are gathered).
Variants: `k_i4r_rows_epi` @ 0x234300 (adds 50 DPP + 12 permlanex16 +
div/rcp = fused SiLU epilogue), `k_i4r_rows_rot` @ 0x230400 (adds the full
DPP butterfly = **HT-rotated i4r rows** — the i4r↔HT bridge kernel),
`k_i4r_rows_fl` (fused-linear), `k_i4r_rows_gl<1|2>(…, I4rSd, int)` (gate/up).
Shapes match the manifest: `<2560>` = gate_up [512,1280,2560] rows, `<640>` =
down [512,2560,640]; 865075200 B over 512·1280·2560 elements = 4.225 bpw
(4-bit payload + fp16 group-128 scales + sign plane).

## 4. q4moe namespace (standalone fused MoE GEMM)

`q4moe::k_q4moe_items<64|128>(const int*, const int*, const int*, int, q4moe::Item*, q4moe::Item*, int*)`
@ 0xe00200 (`<64>`, 309 instrs): work-list builder — `ds_load_b32` ×20, 19 × `s_barrier`,
`v_cmpx_lt_u32` bucketing, `global_store_b128` ×15 (Item = 16 B records).
`q4moe::k_q4moe<1|2, 0|2, 64|128, Tune<0,128>, 0|1|2>(q4moe::Args)`
@ 0xe31000 (`<1,0,128,Tune<0,128>,1>`, 39265 instrs): the big fused MoE GEMM.
- 4-bit decode: `v_bfe_u32 v163, v162, 16, 1` cascade +
  `v_perm_b32 v162, v163, v162, 0x7060302` / `0x6040200` / `0x7050301`
  (0xe31858–0xe319b8) = 8-nibble→bf16 byte-shuffle unpack (2160 v_perm total);
- `v_wmma_f32_16x16x16_bf16` ×576 (0xe31e4c) — bf16 matrix cores (weights
  decoded to bf16, not f16, unlike the HT family);
- 720 ds_load_b128, 19 barriers, 3456 DPP + 1376 permlanex16 (rotation/reduce
  epilogue), 576 div_scale/rcp (SiLU), 512 `v_bfi_b32` bit-field insert
  (asym zero-point merge).
This is the "4-bit experts = memory-bound decode bandwidth" kernel: 2160 perm +
424 bfe + 452 lshr decode ops vs 576 WMMA — decode ALU ≈ 5× the MAC count, and
the whole point of the fused form is that the 4-bit stream is read once, decoded
in registers, and consumed by WMMA without ever materializing bf16 weights.

## 5. Expert gate/up + activation kernels

`k_exp_gu_w<8|16|32, 1|2|3, 0|1>(const unsigned short*, const unsigned char*,
const long*, long, long, long, unsigned short*)` @ 0x109eb00 (`<16,1,0>`,
459 instrs): fused gate/up for one expert, W8/W16/W32 = output width class,
arg2 = experts-per-block, arg3 = scale mode. Weights: `ds_load_b32` ×32
(LDS-staged nibble table, staged by `ds_bpermute_b32` ×8 + `global_load_b128` ×8),
`v_dot2_f32_bf16` ×32, fp16 scales via `v_cvt_f32_f16_e32` (0x109ee28),
bf16 rounding `v_add3_u32 … 0x7fff`. SiLU fingerprint present (exp/rcp/div).
`k_expert_h_q4(const unsigned short*, const unsigned char*, const long*, long,
long, long, unsigned short*)` @ 0x251f00 (349 instrs): hidden-state q4 expert —
`global_load_u8` (weights) + `global_load_d16_b16` (fp16 scales) with
**1280-B strides** (`offset:3840` = 3×1280, `offset:1280`) = q4a/q4c row layout
(N·K/2 bytes + 2·N/32·2 scale bytes at K=2560); `v_fma_mix_f32`/`v_fmac_f32`
×16, `ds_load_2addr_stride64_b32` ×9.

## 6. Router + activation epilogues

- `k_route_logits_rows<4,5..8|8,2..4>(const unsigned short*, const float*, float*, long, long)`
  and `k_route_logits_rows_sg<…,b>(…, SgArg)` (e.g. `<4,5,0>` @ 0x79cf00): router logits =
  `v_dot2_f32_bf16` ×80 + `ds_bpermute_b32` ×25 + `ds_load/store_2addr_stride64_b32`
  (LDS butterfly reduction, 798 instrs for <4,5>). `k_route_logits_w32` @
  0x229900: f32 weights, `v_fmac_f32` + ds_store/load pairwise tree
  (0x229a58–0x229b0c). `k_route_logits_x32` @ 0x247800: bf16 weights
  (`global_load_u16` + fmac).
- `k_route_topk(const float*, int*, float*)` @ 0x245d00 (427 instrs): bitonic
  top-k in LDS — `ds_store_b32 … offset:4096` then compare-exchange at
  stride 1024/512/256 (`ds_load_2addr_stride64_b32 offset1:4/2/1`) with
  `s_barrier` ×13; softmax tail: `v_exp_f32` ×10 + full div sequence.
  `k_route_topk_w` = weighted; `k_route_hist/scan/scatter` = histogram routing.
- `k_glu_rows(const unsigned short*, const int, long, unsigned short*)` @
  0x257400: `global_load_u16 v5, … offset:-1280` (gate half) + `offset:0`
  (up half) → `v_exp_f32` + `1.0+` + IEEE div → gate·σ(up) (85 instrs).
- `k_scale_silu(const unsigned short*, unsigned short*, long, float)` @ 0x23fa00:
  y = s·silu(x) (55 instrs, exp/rcp/div).
- `k_act_rows(const unsigned short*, int, const int*, const int*, const int*, int, float*, float*)`
  @ 0x22b000: indexed row add (`global_load/store_b32` ×2) + `v_fmac_f32 v3, v6, v6`
  (square → sumsq) = fused residual-add + RMSNorm statistics for gathered rows.
- `k_act_dense(const unsigned short*, int, int, float*, float*)` @ 0x22ae00:
  dense version (64 instrs).
- `k_conv1d_silu<b>(const unsigned short*, ×3, unsigned short*, int)` @
  0x8aa000: 4-tap causal depthwise conv (`v_fma_f32` + 3× `v_fmac_f32` over
  `global_load_u16` taps, 0x8aa204–0x8aa22c) fused with SiLU
  (`v_mul_f32 0xbfb8aa3b` → `v_exp_f32` → `1.0+` → div, 0x8aa234–0x8aa260).
- `k_hcup4<N,0|1|2>(const unsigned char*, long, long, long, const unsigned short*,
  const unsigned short*, unsigned short*, float*)` @ 0x2cbe00 (`<1,0>`, 610 instrs):
  4-bit head-count-update linear — `ds_load_b32` ×64 + `v_dot2_f32_bf16` ×64 +
  `v_rsq_f32_e32` (RMSNorm) + SiLU div sequence. `k_hcup4w` = wide variant.

## 7. k_dn_fused16 (down-projection fused, 16-wide)

`k_dn_fused16<0..12, b, b, b>(const unsigned short*, const float*, const float*,
float*, const int*, long, unsigned short*, int, const unsigned long*, DnfTaps, DnfNorm)`
@ codeobj_14aa000 0x20300 (`<0,0,0,0>`, 1090 instrs; 17 variants in the manifest):
- `v_dual_lshlrev_b32 v64, 4, v0` (0x20314) = 16-wide tile (lane ×16);
- LDS tile pipeline: `ds_load_b128` ×35 + `ds_load_u16` ×10 +
  `ds_load_2addr_b32` ×9, `ds_store_b32 … offset:24064` (0x21614);
- RMSNorm: `v_mul_f32_e32 v11, v14, v14` + `v_add_f32` sumsq +
  `v_rsq_f32_e32 v4, v4` (0x21170) — the `DnfNorm` struct carries the norm
  weight/eps; `v_mul_f32_e32 v7, 0x3fb8aa3b, v3` (1/√256·…) scaling;
- matrix core: `v_wmma_f32_16x16x16_bf16 v[2:9], v[18:25], v[10:17], v[2:9]`
  (0x21750) — bf16 WMMA tiles fed from LDS at offsets 6160/6240/6256/6272/6288
  (0x21674–0x21760) and 96/112/128/144;
- `DnfTaps` = tap offsets (the `v_perm_b32` ×32 + `v_fma_mixlo/hi_f16` ×28
  gather the tap-shifted bf16 pairs); `Py` (u64 ptr) = per-layer accumulator;
  `PKil` = int64 stride; epilogue `v_cvt_f16_f32` ×13 + `v_perm_b32` repack,
  `global_store_b32` ×66.
So: fused (residual taps) → RMSNorm → bf16 WMMA down-projection → f16 store,
one kernel, one LDS residency — the "fused = one pass" claim for the down side.

## 8. Perf-claim mapping

| Claim | Evidence |
|---|---|
| 4-bit experts = memory-bound decode bandwidth | q4moe: 2160 `v_perm_b32` + 424 `v_bfe` decode vs 576 WMMA (0xe31430–0xe319b8); lq4w: 16 `v_dot2` vs 16 LDS loads + 31 shift/mask ops; deq_q4cp_v32: 64 bfe + 61 and + 57 lshr, zero MACs. Decode ALU dominates; DRAM traffic is K/2 bytes/row + 2·K/32 scales. |
| fused gate/up + SiLU = one pass | k_ht_moe_gu3: gate+up in one WMMA tile set (36 WMMA), SiLU (exp/rcp/div, 0xa876d8–0xa8783c) applied before store (0xa87c4c) — no intermediate global round-trip; k_exp_gu_w, k_ht_moe_gu (64 exps), k_ht_gemv_gu (64 exps) same pattern. |
| HT = rotation applied in-kernel | §1: DPP butterfly ladders + permlanex16 + sign-cndmask + 1/16 norm inside k_ht_rot_*, k_ht_deq_orig (LDS-transposed butterfly 0x4a142c–0x4a155c), and fused into the accumulator epilogues of k_ht_moe_gu3/dn3/dn2x/gu2 and k_ht_lin/lin_grp/gemv_* (e.g. 0xa873b8, 0x9e0434, 0x40879c, 0x519ea8). |
| matrix-core tiles for the big GEMMs | 8756 `v_wmma_f32_16x16x16_f16` + 4004 `…_bf16` across the code objects; zero `v_mfma_*` — the engine standardizes on WMMA 16×16×16 (gfx1151 wave32 path), with `v_dot2acc_f32_f16`/`v_dot2_f32_bf16` for GEMV/row kernels. |

## 9. Open items

- Exact `HtItem` field layout (stride 72 B confirmed via `v_mul_u32_u24_e32 …, 0x48`;
  0x199a=6554 in k_ht_items_group and the 0x1400/0x500 strides in gu3 need a
  byte-level layout decode).
- `k_ht_gemv_dn_g` / `k_ht_gemv_gu_gs` (grouped-GEMV variants, @ 0x90c100–0x935a00 /
  0x8fa600–0x907600) indexed but not deep-dived; their butterfly/decode mix matches
  the _t/_ts siblings.
- `k_ht_tg` semantics ("token gate"? tied-gate?) inferred from shapes only.
- k_dn_fused16 `DnfTaps` tap count per template arg (0..12) not enumerated.
- The 0x6400/0xc931 bias is −8 (symmetric); q4a asym zero-points ride the
  `v_xor`/`v_bfi` paths — per-format bias table not extracted.
