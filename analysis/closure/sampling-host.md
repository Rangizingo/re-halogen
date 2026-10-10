# Sampling host side: U15, U16, U20, U23

Method: host decompiles in work/decomp/flash_serve, kernel slices in work/kernels-spec. Ghidra image base is 0x100000 (constants read from binaries/flash_serve at file offset = vaddr - 0x100000: DAT_00113b10 = 1.0f, DAT_00113b34 = 0x33800000 = 2^-24, DAT_00113bb8 = 1.0 double). Line numbers are lines in work/decomp/flash_serve/<addr>.c unless a kernel slice is named. "inferred" is marked.

## U15 SampleRow (32 bytes) field map

Verdict: CLOSED for layout and sources; PARTIAL for the kernel-side meaning of +4 and +0xc.

Four host writers build the same 32-byte row and hipMemcpy it to the device row buffer at engine+0x9a0:

- sample_rows 019085c0 (n rows, 1..8, `n << 5` bytes): 019085c0.c:205
- draft_sample 019098d0 (1 row, 0x20 bytes): 019098d0.c:55-62
- accept_probs 01909e50 (n rows): 01909e50.c:240
- residual_sample 0190aa50 (1 row, 0x20 bytes): 0190aa50.c, `hipMemcpy(...,&local_68,0x20,1)`

The host input is a 16-float (64-byte) per-request parameter block, stride 0x10 floats (019085c0.c: row 1 reads param_3[0..5], row 2 reads param_3[0x10..0x15], row 3 param_3[0x20..]). Its fields follow the wire order `SAMPLE <temp> <top_k> <top_p> <min_p> <seed>` (analysis/wire-protocol.md:53,69). The device row copies it as follows.

| off | size | type | host source | meaning |
|---|---|---|---|---|
| 0x00 | 4 | float | `*param_3`; must be > 0 or exit "temperature 0 is the GREEDY path and must stay on argmax()" (019085c0.c:76-80) | temperature T. The kernels use 1/T (`v_div_fixup_f32 v6, v0, s38, 1.0` in k_smpg_pick, slice lines 472-491) |
| 0x04 | 4 | int bits | `param_3[1]` copied raw (`local_134 = param_3[1]`, 019085c0.c:~84) | top_k, 0 = off (wire-protocol.md:69; the parser rejects top_k < 0 at 0182ea80.c:2530). Kernel use not traced |
| 0x08 | 4 | float | `param_3[2]` raw | top_p (nucleus). draft_sample writes HALOGEN_S2_NUCLEUS here (019098d0.c, `local_60 = DAT_019fe1c8`) |
| 0x0c | 4 | float | `param_3[3]` clamped to [0.0, 1.0f] (019085c0.c:~85-88); accept_probs maps a negative value to 0 (01909e50.c:~86-92) | min_p (inferred from wire order and the [0,1] clamp). k_smpg_accept_final compares it with the exp numerator and rejects when it is larger (`v_cmp_nle_f32 vcc_lo, s15, v0`, accept_final slice line 306/`0x219F40`), which is the min_p test exp((l - lmax)/T) >= min_p (inferred) |
| 0x10 | 8 | u64 | sample_rows: `*(u64*)(param_3+4)` (019085c0.c, `local_128`); draft_sample 3rd arg (`local_58 = param_3`); spec_round_run passes `*(param_1[0x32]+0x10)` (01851dc0.c:67) | request seed (see U23) |
| 0x18 | 4 | u32 | sample_rows `param_4[i]` per row (`local_120 = *param_4`); draft_sample and residual_sample `param_4`; accept_probs 0 (01909e50.c:~237 `local_200 = 0`) | per-row counter A: the position of the drawn token (spec_round_run passes `iVar11`, the base position, then `iVar11 + k`, 01851dc0.c:67,73,86) |
| 0x1c | 4 | u32 | sample_rows `param_5`; draft_sample `param_5` (spec_round_run passes 1, 01851dc0.c:67); residual_sample literal 3 (`local_4c = 3`); accept_probs 0 | stream tag B, a domain separator between draw purposes |

Kernel loads agree. k_smpg_pick loads row+0x00 (s38), +0x0c (s33) and a 16-byte block at +0x10 (s28..s31 = seed lo, seed hi, A, B) (pick slice lines 13-15). k_smpg_rpick loads +0x00, +0x0c, +0x10..0x1f the same way (rpick slice lines 10-12). k_smpg_accept_final loads +0x00 and +0x0c only (accept_final slice lines 14-15), so the accept kernel never draws random numbers, which matches accept_probs writing 0 in +0x18/+0x1c. Row index scale in the kernels is `<< 5` (pick slice line 5, `s_lshl_b64 s[0:1], s[10:11], 5`), consistent with 32 bytes.

Remains: what pick does with +4 and +8 (top_k and top_p are from wire order only; no kernel instruction was traced to them). In pick, +0xc is the value chosen for tokens whose sortable key is below a scratch threshold (`v_cndmask_b32 v1, s33, v10, vcc_lo` at pick slice line 551, vcc from `v_cmp_le_u32 s30, v11` with s30 = SmpScratch+0x4040). That is not a plain "weight 0 outside the filter", so min_p is not proven for pick.

## U16 which mode value of k_smpg_accept_final is greedy

Verdict: STILL OPEN (no host call site in the decompile). Kernel side is bounded.

Kernel: mode is the second of the three trailing ints (args: six pointers to 0x30, ints at 0x30, 0x34, 0x38; s13 = arg at 0x34, loaded by the `s_load_b128 s[12:15], s[0:1], 0x30` at accept_final slice line 6). The only use of s13 is `s_cmp_lg_u32 s13, 3` at slice line 318, in the fall-through block (rows with an out-of-range id, or an early exit). There the accept value is 1.0 when mode == 3, else 0 (`v_cndmask_b32_e32 v2, 1.0, v3, vcc_lo`, v3 = 0). So mode 3 means "force-accept the fall-through row", other values mean "reject it". The in-range path does not read s13.

Host: the registration entry is PTR_FUN_019f6408 (01968680.c:44), a relative relocation (`objdump -R`: 0x18f6408 R_X86_64_RELATIVE, addend 0x1783e60) to the stub FUN_01883e60 (174 bytes, 0 callers in index.tsv). `grep -l 019f6408` over every decompile hits only 01968680.c; PTR_FUN_019f63f0..019f6418 (the pick, rpick and accept_final descriptors) are likewise referenced only by the registration function. So no decompiled host function launches k_smpg_pick, k_smpg_rpick or k_smpg_accept_final in 0.17.3 (inferred: the launch is through a path Ghidra did not resolve, or the engine does not drive the SMPG graph in this build). The mode argument was not found at a call site.

Nearest host facts, not proof for this kernel: the host accept test in spec_round_run treats WALK_FALSIFY == 3 as a special case (U20), and residual_sample writes the literal 3 into SampleRow+0x1c. No link to s13 was shown.

Needed: a runtime trace that records the second trailing int of k_smpg_accept_final during a sampled MTP run, or the stub's real caller.

## U20 what the falsifier flags change in host data fed to the accept kernels

Verdict: CLOSED for WALK_FALSIFY, the S2 flags and SAMPLED_*; STILL OPEN for S3_FALSIFY (no consumer found).

The flags change host inputs only; none touches device code (consistent with kernels-speculation.md 4b).

HALOGEN_WALK_FALSIFY: getter 01883020.c:8-12 (`getenv`, `strtol` base 10, default 0), cached in static DAT_019fc588 (01851dc0.c:121-124) inside spec_round_run 01851dc0, the only consumer. In its accept loop (01851dc0.c:130-235), with m = DAT_019fc588:

- m == 3: the position counter in the host accept draw is forced to 0: `iVar6 = iVar18; if (iVar14 == 3) iVar6 = 0;` (01851dc0.c:144-146). Every draft position in a round then draws the same uniform. This is the change closest to the kernels' SampleRow+0x18 counter.
- m == 4: the residual resample at a reject gets row 0 instead of the failing row i: `uVar8 = uVar19; if (iVar14 == 4) uVar8 = 0;` (01851dc0.c:~177-178), passed as param_2 of FUN_0190aa50 (residual_sample), so the wrong target-logit row is used.
- m == 1: the last argument of FUN_0190aa50 (q row, param_10) is forced to 0 instead of i: `if (iVar14 != 1) uVar12 = uVar19;` (01851dc0.c:~181-182), so the drafter's q row 0 is used for the residual.
- m == 2: no row data change. It defers the per-token sampler-state step (FUN_0184ad90 plus the conditional FUN_0184a9b0) so accepted tokens are stepped after the loop instead of one by one; on a reject all earlier tokens are stepped first (01851dc0.c:150-165, 195, 218-232). History update order only.
- m == 0: normal path.

Purpose of 1, 3 and 4 (inferred): deliberately wrong inputs so a test can prove the output check notices them.

The host accept test itself (not a kernel), 01851dc0.c:136-155: `dVar21 = local_10c / exp(local_78[i])`, where local_10c is the p output of accept_probs (target probability of the draft token) and local_78[i] is the drafter logprob from draft_sample. dVar21 is capped at 1.0 (DAT_00113bb8) and compared with u = (top 24 bits of the splitmix64 output) * 2^-24. `dVar21 <= u` is the reject branch (calls residual_sample); otherwise the draft token is kept (`local_e8[i] = local_b8[i]`). So the host-path acceptance is min(1, p/q) against a uniform draw; this confirms the shape that kernels-speculation.md only inferred.

HALOGEN_S2_NUCLEUS, HALOGEN_S2_QOFF, HALOGEN_S2_TSKEW (getters 01909da0, 01909dd0, 01909e00: getenv plus strtod): used only in draft_sample 019098d0.c:44-62 and 85-105. Row written: T = temp * tskew (+0x00), top_k 0 (+0x04), top_p = nucleus (+0x08), min_p 0 (+0x0c), seed, position, tag. qoff is added to the returned drafter logprob (`*param_7 = DAT_019fe1d8 + *param_7`), which skews the p/q ratio in the host accept test. When any is non-neutral, the first call prints "draft_sample: FALSIFIER ACTIVE: nucleus %.4f, qoff %+.4f, tskew %.4f. Every proposal from this process is deliberately wrong (METHOD 37)." once (guard byte at engine+0xa88). Defaults when unset are not visible (the decompiled getter returns an unseen register); the gate `tskew != 1.0f || nucleus > 0 || qoff != 0` implies neutral = 1.0 / 0 / 0 (inferred).

HALOGEN_SAMPLED_DEPTH: parsed in main_init_config 0182ba00.c:1010-1019 (strtol; must be >= 1 and, when not 1, below HALOGEN_SPEC_N, else exit). Sets the sampled-speculation draft depth in config. No row change.

HALOGEN_SAMPLED_GATE=t1[,t2]: parser FUN_01853890 (01853890.c:15-33). Values must lie in [0, 1.0] (compare with DAT_00113b10). Stored at config +0x34 and +0x38, only when config+0x30 > 1; default is the 8-byte constant DAT_001143d0. Consumer (inferred from use): spec_round_run 01851dc0.c:62-85 stops drafting when `exp(q logprob) < t` (first threshold at `*(float*)(param_1+0x144)`, the second at `param_1[0x29]`). It changes how many drafts reach accept_probs, not row contents. The mapping from config +0x34/+0x38 to engine +0x144/+0x148 was not traced.

HALOGEN_S3_FALSIFY: getter FUN_0194bd90 (strtol, default 0, 25 bytes) has no caller in index.tsv and no `FUN_0194bd90` text in any decompile. What it changes is STILL OPEN.

## U23 seed and counter source of the splitmix64 draw

Verdict: CLOSED for the kernel-side derivation and the host row fill; PARTIAL for the link from the wire seed to params+0x10 (parse store not located).

k_smpg_pick, slice lines 182-217 (rpick has the same pattern at 234-264 per the earlier notes; not re-read here):

- seed = u64 at SampleRow+0x10 (s[28:29]); A = u32 at +0x18 (s30); B = u32 at +0x1c (s31) (loads at slice lines 13-15).
- x = (A * 0xd1b54a32d192ed03) ^ seed ^ (B * 0x9e3779b97f4a7c15) (lines 182-193: `s_mul_i32 s37,s30,0xd1b54a32`, `s_mul_hi_u32 s39,s30,0xd192ed03`, `s_mul_i32 s30,s30,0xd192ed03`, `s_mul_i32 s40,s31,0x9e3779b9`, `s_mul_hi_u32 s41,s31,0x7f4a7c15`, `s_mul_i32 s36,s31,0x7f4a7c15`, then two `s_xor_b64`). A and B are 32-bit multiplicands of the 64-bit constants.
- x += 0x9e3779b97f4a7c15 (lines 194-195).
- z = x ^ (x >> 30); z *= 0xbf58476d1ce4e5b9; z ^= z >> 27; z *= 0x94d049bb133111eb (lines 196-213).
- u = top 24 bits of z (`s_lshr_b32 s28, s28, 8` at line 217, on the high word), converted to float and scaled by 2^-24 (`v_mul_f32_e32 v9, 0x33800000, v1`, line 227). The `s_clz_i32_u32`/`s_min_u32 32` pair (lines 214-216, 222-225) is the integer-to-float normalization (inferred).

Host mirror, which proves the field roles: spec_round_run draws the accept uniform on the CPU with the same formula: `((pos + base) * 0xd1b54a32d192ed03 ^ *(u64*)(params+0x10) ^ 0x3c6ef372fe94f82a) + 0x9e3779b97f4a7c15`, then `(z ^ z>>30) * 0xbf58476d1ce4e5b9`, `(z ^ z>>27) * 0x94d049bb133111eb`, `>> 0x28`, `* 2^-24` (01851dc0.c:147-155; Ghidra prints -0x2e4ab5cd2e6d12fd = 0xd1b54a32d192ed03, -0x40a7b892e31b1a47 = 0xbf58476d1ce4e5b9, -0x6b2fb644ecceee15 = 0x94d049bb133111eb). 0x3c6ef372fe94f82a is 2 * 0x9e3779b97f4a7c15 mod 2^64, so the host accept draw is the kernel formula with stream tag B = 2. Tags seen: 1 draft_sample (01851dc0.c:67), 2 host accept draw, 3 residual_sample (literal in 0190aa50.c). The target sample_rows tag is its caller's param_5 (not traced). Counter A is the token position (base position + draft index), except under WALK_FALSIFY=3 (U20).

Seed source: the u64 at (request sampler params)+0x10, copied to SampleRow+0x10 (019085c0.c, `local_128 = *(u64*)(param_3+4)`; spec_round_run reads `*(param_1[0x32]+0x10)` at 01851dc0.c:67). Wire: the GEN line `SAMPLE <temp> <top_k> <top_p> <min_p> <seed>` carries `req.seed` or a random 63-bit int as the fifth field (wire-protocol.md:53,69), which lands at float index 4..5, byte +0x10, of the 16-float block. The layout match is the evidence. The statement that stores the fifth SAMPLE field was not located (the SAMPLE validation near 0182ea80.c:2500-2540 checks temp in (0, DAT_00113b24] and top_k >= 0; its seed store was not read). No transform of the seed other than the formula above appears in the three row writers (inferred: the seed is used as is).

Consequence for the earlier open item: `s15` in k_smpg_accept_final is not an RNG value. It is SampleRow+0xc (min_p, U15), and the accept kernel has no RNG. The uniform for the accept decision is drawn on the host (above); the in-kernel splitmix64 is for the token pick in pick and rpick.

Remains: (1) the statement that stores the wire seed at params+0x10; (2) whether pick and rpick are launched at all by this binary (U16: no host reference to their descriptors); (3) a fixed-seed runtime run to confirm the host formula reproduces the kernel's u (not run).
