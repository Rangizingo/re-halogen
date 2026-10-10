# Sampler math, round 5: top_k ties, top_p inclusion, min_p, penalties and bias, order

## Scope

Close the five items listed in `analysis/rebuild-guide.md` section 11 ("Sampler tie handling, top_p inclusion, penalty and bias math"): (1) top_k tie handling, (2) top_p inclusion rule, (3) min_p rule, (4) presence, frequency, repetition penalty and logit_bias math and order, (5) temperature order relative to the filters. Static only, nothing run.

Conventions. Ghidra address = ELF vaddr + 0x100000. Decomp files: `work/decomp/flash_serve/<addr>.c`. Host disassembly: `objdump -d binaries/flash_serve` (cited `vaddr: instruction`). Kernel listings are the slices in `work/kernels-spec/`, cited `<short>:<line>` with these short names (every file is `_ZN7halogen...<mangled>.s`):

| short | file |
|---|---|
| max | `_ZN7halogen10k_smpg_maxEPKfPNS_10SmpScratchEii.s` |
| L0..L3 | `_ZN7halogen12k_smpg_levelILi<N>EEEvPKfPKNS_9SampleRowEPNS_10SmpScratchEii.s` |
| mass | `_ZN7halogen11k_smpg_massEPKfPKNS_9SampleRowEPNS_10SmpScratchEPfS7_ii.s` |
| pick | `_ZN7halogen11k_smpg_pickEPKfPKNS_9SampleRowEPKNS_10SmpScratchES1_S1_PiPfS9_ii.s` |
| acc | `_ZN7halogen19k_smpg_accept_finalEPKfPKiPKNS_9SampleRowEPKNS_10SmpScratchEPfSA_iii.s` |
| rmass, rpick | the `k_smpg_rmass`, `k_smpg_rpick` slices |
| sample | `_ZN7halogen12k_sample_f32EPKfPKNS_9SampleRowEPiPfS6_i.s` |
| penrep | `_ZN7halogen18k_pen_rep_rows_f32EPfPKiPKfS4_S4_PKhS4_S2_iii.s` |
| penscat | `_ZN7halogen22k_pen_scatter_rows_f32EPfPKiPKfS2_iii.s` |

"Hand-decoded" marks a VOP3 pair that the slice printed as `.long` plus a mis-decoded second dword (src fields read out of the second dword by hand). Opcode check: the slices print `v_cmp_ge_u64_e32 ... // 7CBC0303` elsewhere, so VOP3 op 0x5e (`D45E....`) is `v_cmp_ge_u64`; `D449` = `v_cmp_lt_u32` (0x49), `D41C` = `v_cmp_nle_f32` (0x1c), `D413` = `v_cmp_le_f32` (0x13), `D728` = `v_mul_f64`, `D72B` = `v_ldexp_f64`.

## Verdict

| # | item | verdict |
|---|---|---|
| 1 | top_k tie handling | CLOSED |
| 2 | top_p inclusion rule | CLOSED |
| 3 | min_p rule | CLOSED |
| 4 | penalty and logit_bias math, order, ids | CLOSED |
| 5 | temperature order vs filters | CLOSED |

The rule set as built. Default path is `HALOGEN_SAMPLE_BLOCKS` unset = 64 blocks, so the `k_smpg_*` graph runs, not `k_sample_f32`:

1. The logit row copy gets bias, then repetition, then presence and frequency (all on raw logits).
2. `w_i = exp((x_i - x_max) / T)`. T enters only here. top_k never sees T.
3. Rank threshold `Tkey = max(Tkey_topk, Tkey_topp)` on the 32-bit sortable logit key. Keep `key >= Tkey`. Ties are all kept and the token that crosses p is kept.
4. A token also needs `min_p <= w_i` (inclusive).
5. Draw: target `u * Z_kept`; the first kept token whose inclusive cumulative weight is strictly greater than the target wins.

## Shared evidence for items 1, 2, 3

### Front end, before the engine

`image/serve_api.py` changes the parameters before the wire line:

- `image/serve_api.py:1880-1887` `sample_spec`: `t = server_default(req, "temperature"); if t is None or t <= 0.0: return None` (greedy, no SAMPLE line), else `(float(t), int(filter_default(req, "top_k") or 0), float(filter_default(req, "top_p") or 0.0), float(server_default(req, "min_p") or 0.0), int(seed))`.
- `image/serve_api.py:1433` `MODEL_FILTERS_BUILTIN = {"top_k": 20, "top_p": 0.95}`; `:1448-1458` take `generation_config.json` `top_k` (int >= 0) and `top_p` (0 < p <= 1) over those; `filter_default` (`:1463-1465`) returns the request value if it is not None, else the server env default, else the model filter. A sampled request that omits both therefore runs with top_k 20 and top_p 0.95 active together. An explicit `top_p: 0` or `top_k: 0` is "not None", passes through as 0, and turns that filter off.
- `image/serve_api.py:2140-2150`: temperature must be 0..2, top_p 0..1, min_p 0..1, else 400 ("NOT clamped"); top_k < 0 is a 400.
- `image/serve_api.py:1078`: wire line `" SAMPLE %.9g %d %.9g %.9g %d" % sample` = temp, top_k, top_p, min_p, seed.
- `image/serve_api.py:1853-1877` `penalty_spec`: `" PENALTY %.9g %.9g" % (pp, fp)` only if `pp or fp`; `" REPEAT %.9g"` only if `rp is not None and float(rp) != 1.0`; `" BIAS %d " % len(items)` then `"%d %.9g"` pairs, token id must satisfy `0 <= tid < 248320`, at most 20480 entries. Bias values are not clamped (no OpenAI +-100 clamp). `image/serve_api.py:2181-2197`: `repetition_penalty` must be finite and > 0; presence, frequency, repetition or logit_bias on a greedy request is a 400. Presence and frequency have no range check on the request (the +-2 range is only on the env defaults, `:1322-1325`).

Engine-side parse: PENALTY reads two floats into stack slots 0x560 and 0x564 (`1733ea0: leaq 0x560(%rsp),%rsi`, `1733eb0: leaq 0x564(%rsp),%rsi`), REPEAT one float (`0182ea80.c:3256` "malformed REPEAT (a finite factor > 0)"), BIAS a count then pairs (`0182ea80.c:2929` "BIAS entry malformed or id out of range", ids `< 0x3ca00`).

### Host launch and row

- `0194bd50.c`: `getenv("HALOGEN_SAMPLE_BLOCKS")`, default `0x40` (64). `01908c30.c:14-24`: `if (uVar2 != 0) { FUN_01908ff0(...)`, else the one-block path (`k_sample_f32`).
- `01909630.c:41-47` builds the 32-byte row: `local_4c = param_3[1]` (top_k), `local_48 = param_3[2]` (top_p), min_p clamped to [0,1] (`if ((fVar1 <= DAT_00113b10) && (local_44 = 0.0, 0.0 <= fVar1)) { local_44 = fVar1; }`); top_k and top_p are copied raw.
- Launch order inside `01908ff0` (vaddr 0x1808ff0): `18090d8: leaq 0xed311(%rip),%rdi # 0x18f63f0` (max), `18091ab: # 0x18f9358` (L0), `18092f8: # 0x18f9450` (L1), `18093cb: # 0x18f9458` (L2), `180949e: # 0x18f9460` (L3), `18095b1: # 0x18f63f8` (mass); descriptor to kernel from `01968680.c:3848-3862` and `:7688-7691`. pick is launched inline in `01908c30` (`1808df1: leaq 0xed608(%rip),%rdi # 0x18f6400`). Grid is (blocks, rows) with 256 threads (`movabsq $0x100000100`); the N argument is `1808d30: movl $0x3ca00, 0x6c(%rsp)` (vocab 248320).

### Kernel algorithm (derived)

The filter is a 4-pass radix select over the 32-bit sortable key of the raw logit. No sort is performed.

- Key: `pick:539-543` `v_not_b32 v12, v11; v_or_b32 v13, 0x80000000, v11; v_cmp_gt_i32 vcc_lo, 0, v11; v_cndmask_b32 v11, v13, v12` = `x < 0 ? ~x : x | 0x80000000`, monotone increasing in x.
- Weight: `L0:232` `v_sub_f32_e32 v10, v11, v8` (x minus row max), `L0:173` `v_div_scale_f32 v2, null, s20, s20, 1.0` (1/T, s20 = SampleRow+0), `L0:235` `v_mul_f32_e32 v10, 0x3fb8aa3b, v10`, `L0:237` `v_exp_f32_e32 v10, v10`.
- Gate: `mass:164-169` `s_cmp_gt_i32 s29, 0; s_cmp_lt_i32 s29, s38; ... s_and_b32 s19, s0, s1` = top_k active iff `0 < top_k < N`; `mass:176-178` and `mass:216-222` `s_cmp_gt_f32 s30, 0; s_cmp_lt_f32 s30, 1.0; ... s_and_b32 s1, s0, s1` = top_p active iff `0 < top_p < 1.0`. The two tests are independent (`L1:27 s_or_b32 s3, s19, s21`). Neither active: `mass:184 s_cbranch_vccnz 626` jumps to the Tkey store with `v5` still 0 (`mass:158 v_mov_b32_e32 v5, 0`, hand-traced), so `Tkey = 0` and every key passes.
- Histogram, per level. Digit: `L0:247 v_lshrrev_b32_e32 v11, 24, v11`; `L1:701 v_bfe_u32 v16, v10, 16, 8`; `L2:725 v_bfe_u32 v16, v13, 8, 8`; `L3:711 v_and_b32_e32 v10, 0xff, v10`. Count: `L0:255 ds_add_u32 v14, v9 offset:3072`. Mass: `L0:240 v_fma_f32 v12, 0x56000000, v10, 0.5` (w * 2^45 + 0.5, then trunc and a hi/lo split) and `L0:256 ds_add_u64 v15, v[11:12]`. Later levels count only keys whose higher bytes equal the chosen prefix: `L1:697-699 v_xor_b32 v12, v10, v8 ... v_cmpx_gt_u32_e32 0x1000000, v12` (L2 `0x10000` at L2:723, L3 `0x100` at L3:708).
- Row mass Z: `L0:222 v_add_f32_e32 v1, v1, v10` runs for every element (also when the histogram branch is skipped). Block sums are stored at scratch+0x400 (`L0:394 global_store_b32 v0, v1, s[0:1] offset:1024`) and summed over the B blocks at the start of L1 (`L1:184 s_add_f32 s9, s9, s26`). So Z is the sum of `w` over the whole vocab (248320 ids), before any filter.
- top_p target: `L1:203-208` `v_cvt_f64_f32 v[3:4], s18` (top_p), `v_cvt_f64_f32 v[5:6], v2` (Z), `.long 0xd7280002 / 02020B03` (hand-decoded `v_mul_f64 v[2:3], v[3:4], v[5:6]`), `.long 0xd72b0002 / 02015B02` (hand-decoded `v_ldexp_f64`, src1 field 173 = integer inline constant 45), then `v_trunc_f64`, `v_floor_f64`, `v_fma_f64 ... 0xc1f00000` for the hi/lo split. Target = `trunc(top_p * Z * 2^45)` as a u64. Same code at `mass:216-244`.
- Bin selection (suffix scan from the high bin down, then max over qualifying bins). `L1:483-493`:
  ```
  v_add_nc_u32_e32 v18, v18, v17            ; C_i + count already above
  .long 0xd45e0009 / 02020B0B               ; hand-decoded v_cmp_ge_u64 s9, v[11:12], v[5:6]  (M_i + M0 >= target)
  v_cmp_le_u32_e32 vcc_lo, s17, v18         ; top_k <= C_i'
  s_and_b32 s9, s21, s9                     ; top_p active AND mass test
  s_and_b32 s17, s19, vcc_lo                ; top_k active AND count test
  s_or_b32 vcc_lo, s17, s9
  v_cndmask_b32_e32 v5, 0, v0, vcc_lo       ; tid if it qualifies, else 0
  ```
  followed by a `v_max_u32` tree (`L1:502-578`). The chosen bin is the highest bin whose suffix count reaches top_k OR whose suffix mass reaches the target. The last byte is chosen in `mass` with the same test (`mass:513-520`: `.long 0xd45e0000 / 02020307` hand-decoded `v_cmp_ge_u64 s0, v[7:8], v[1:2]`, `v_cmp_le_u32_e32 vcc_lo, s29, v11`, `s_or_b32 vcc_lo, s19, s0`). The 32-bit threshold is composed at `mass:629 v_or_b32_e32 v6, v8, v0` and stored at `mass:643 global_store_b32 v1, v5, s[22:23] offset:64` (scratch+0x4040). Level state slots: L1 stores at +0x4000/+0x4010/+0x4020 (`L1:618-622`), L2 at +0x4004/+0x4014/+0x4028 (`L2:621-625`), L3 at +0x4008/+0x4018/+0x4030 (`L3:620-624`); mass loads +0x4008, +0x4018, +0x4030 (`mass:249-251`).
- Use of the threshold, same in every kernel: `pick:221 s_load_b32 s30, s[14:15], 0x4040`, `pick:546 v_cmp_le_u32_e32 vcc_lo, s30, v11` (Tkey <= key = keep); `mass:716 .long 0xd4490000 / 02020B07` (hand-decoded `v_cmp_lt_u32 s0, v7, v5`: key < Tkey = excluded); `rmass:260 v_cmp_gt_u32_e32 vcc_lo, s30, v12`; `rpick:822 v_cmp_le_u32_e32 vcc_lo, s39, v13`; monolithic `sample:1206 v_cmp_ge_u32_e32 vcc_lo, v12, v4`; the accept kernel loads it too (`acc:243 s_load_b32 s2, s[2:3], 0x4040`).

## Item 1: top_k tie handling. Verdict: CLOSED

No index wins. The threshold is the sortable key of the k-th largest logit: the largest `Tkey` with `count(key >= Tkey) >= top_k` (`L1:483-492`). Membership is `key >= Tkey` (`pick:546`). Every token whose logit bit pattern equals the k-th largest logit is kept, so more than top_k tokens survive when the k-th value is tied. Tied tokens have equal `w`, so they carry equal probability in the draw.

Consequences of the key transform: `-0.0` (key 0x7FFFFFFF) and `+0.0` (key 0x80000000) are different keys and do not tie. top_k is the only filter that ignores T (the key is built from the raw logit, `pick:539-543`). NaN bit patterns were not analysed.

Correction to `closure/sampling-host-r2.md:52` ("If both are active, top_k wins and top_p is ignored"): wrong. The two tests are ORed per bin (`L1:490-492`, `mass:517-520`), so the stricter threshold applies (intersection). With top_k 20 and top_p 0.95 both active, `Tkey = max(Tkey_k, Tkey_p)`.

## Item 2: top_p inclusion rule. Verdict: CLOSED

- Order: descending by logit key (radix select over keys; no sort).
- Inclusion: the token that crosses p is included. The test is `suffix_mass >= target` (`v_cmp_ge_u64`, `L1:485`, `mass:513`; at `mass:513` objdump prints the raw word `.long 0xd45e0000`, a VOP3-encoded VOPC with opcode 0x5e, which is `v_cmp_ge_u64` in the gfx11 VOPC table, hand-decoded by the orchestrator), so `Tkey_p` is the largest key whose cumulative mass from the top reaches the target, and `key >= Tkey_p` keeps that token and every token tied with it. That is the standard "smallest set whose mass is >= p". The comparison is `>=` on 45-bit fixed point (`w * 2^45 + 0.5`, truncated) against `trunc(top_p * Z * 2^45)`.
- Mass base: Z is the whole-vocab `sum w` at temperature T. It is not renormalised after top_k. With top_k and top_p both active, the p cut is measured against the full distribution, not against the top_k survivors. The vendor text at `image/serve_api.py:3331-3336` ("top_k, top_p and min_p compose as an intersection (the HF/vLLM convention)") holds for the intersection, but the p mass base differs from a top_k, then renormalise, then top_p implementation. That comparison rests on my recollection of HF and vLLM, not on a file here (inference).
- Bounds: top_p active iff `0 < top_p < 1.0` (`mass:176-178`, `mass:216-222`). `top_p >= 1` or `<= 0` is off.

## Item 3: min_p rule. Verdict: CLOSED

Keep a token iff `min_p <= w`, with `w = exp((x - x_max) / T)`, that is `p_i / p_max >= min_p` on the temperature-scaled distribution. Inclusive.

- `pick:550` `.long 0xd4130000 / 02021421` (hand-decoded `v_cmp_le_f32 s0, s33, v10`; s33 = SampleRow+0xc, `pick:14 s_load_b32 s33, s[0:1], 0xc`), `pick:552 s_and_b32 s9, vcc_lo, s0` (rank keep AND min_p keep).
- `mass:720 .long 0xd41c0001 / 0202101F` (hand-decoded `v_cmp_nle_f32 s1, s31, v8` = `!(min_p <= w)`), `mass:723 s_or_b32 vcc_lo, s0, s1` (excluded if rank OR min_p), `mass:722-726` adds `w` to the kept mass only when not excluded.
- `sample:1210 .long 0xd4130000` and `sample:969 .long 0xd41c000c` (monolithic kernel), `acc:306 v_cmp_nle_f32_e32 vcc_lo, s15, v0` (accept).
- It is a conjunction with the rank filter and never feeds the top_k or top_p thresholds: L1 loads it into s19 (`L1:13 s_load_b128 s[16:19]`) then overwrites s19 (`L1:20 s_and_b32 s19, s3, s4`); L0 never reads s23.
- The row maximum always survives (w = 1, min_p clamped to <= 1 by the host at `01909630.c:41-47`). Negative min_p clamps to 0.
- Kept mass for renormalisation = sum of kept `w` only (`mass:722-726`), stored per block at scratch+0x800 (`mass:884 global_store_b32 v0, v1, s[0:1] offset:2048`).

## Item 4: penalties and logit_bias. Verdict: CLOSED

Where it runs: `FUN_01907ff0` (target_rows_, vaddr 0x1807ff0). It copies the logit row to a work buffer (`180820b: callq hipMemcpyAsync`, destination `18081f1: addq 0xa08(%rax),%rdi`), runs a penalty kernel on the copy, and returns the copy (`18081ff: movq %rdi,(%rsp)`, `18084b9: movq (%rsp),%rax`). With no penalty state it returns the original row (`180801f: je 0x180853d`). The sampler consumes that pointer (`1808a4a: callq 0x1807ff0` then `1808a75: callq 0x1808c30`), so items 1-3 filter the penalised logits.

Kernel choice: repetition active on any row (`engine+0xa68[row] != 1.0`, `18081c8: cmpneqss 0xa68(%rax,%rcx,4),%xmm0`) launches `k_pen_rep_rows_f32` (`180839a: leaq 0xee047(%rip),%rdi # 0x18f63e8`), else `k_pen_scatter_rows_f32` (`1808490: leaq 0xedf49(%rip),%rdi # 0x18f63e0`). Grid x = `(max count + 255) >> 8` (`180821b: addl $0xff,%r13d`, `1808222: shrl $0x8,%r13d`), y = rows.

Kernel math, `V = 0x3ca00` (`penscat:27 v_mad_i64_i32 v[0:1], null, 0x3ca00, s0, v[0:1]`):

- Scatter (no repetition), `penscat:36-65`: `v_cmp_lt_i32_e32 vcc_lo, -1, v2` (id >= 0, id from `ids[row*V + i]`), then `v_add_f32_e32 v0, v0, v1` (delta + logit) and `global_store_b32 v[2:3], v0, off`: `logits[y*V + id] += delta[row*V + i]`.
- Repetition kernel, `penrep`, `rep = repfactor[row]` (`penrep:58 s_load_b32 s0, s[2:3], null`):
  ```
  rep == 1.0 : logit += delta                         penrep:64 s_cmp_neq_f32 s0, 1.0 / s_cbranch_scc0 75, :120 v_add_f32_e32 v4, v0, v1
  rep != 1.0 : x = logit + A[idx]                     penrep:78 v_add_f32_e32 v4, v8, v6
               if flag[idx] != 0                      penrep:75 global_load_d16_u8 v5, v[4:5], off ; :80 v_cmpx_ne_u16_e32 0, v5
                   x = x * (x > 0 ? 1/rep : rep)      penrep:98 v_cmp_lt_f32_e32 vcc_lo, 0, v4 ; :101 v_cndmask_b32_e32 v5, s0, v5, vcc_lo ; :103 v_mul_f32_e32 v4, v4, v5
               x = x + B[idx]                         penrep:111 v_add_f32_e32 v4, v4, v5
  ```
  Positive logits are divided by `rep`, non-positive ones multiplied by it (the standard HF/vLLM rule), so `rep < 1` boosts.
- Arrays (`FUN_01907c70` penalties_set, `01907c70.c:56-67`): ids `engine+0xa10`, delta `+0xa18`, A `+0xa48`, B `+0xa50`, flags `+0xa58`, rep `+0xa60`, count `+0xa20`. Kernel argument order `18082a9..18082c7` (dest, +0xa10, +0xa18, +0xa48, +0xa50, +0xa58, +0xa60, +0xa20). The factor is forced to 1.0 when the A/B/flag pointers are null or `rep <= 0` (`01907c70.c:30-35`). A, B and flags are uploaded only when `rep != 1.0` (`01907c70.c:60`).

Per-id values on the request object (`FUN_0184ad90`, called after each committed token; `0184ad90.c:199-217`):

```
n      = ++count[idx]                                           0184ad90.c:199
delta  = (bias[idx] - (n >= 1 ? presence : 0)) - frequency * n  0184ad90.c:203-211
B      = (-frequency) * n - (n > 0 ? presence : 0)              0184ad90.c:213-216  (sign flip with _DAT_001145f0)
flag   = 1                                                      0184ad90.c:217
```

`_DAT_001145f0` is 0x80000000 (read from `binaries/flash_serve` file offset 0x145f0). Request offsets: presence `+0x1e0`, frequency `+0x1e4`, repeat `+0x1e8` (`0184ad90.c:207,211`; `01848190.c:209` reads `+0x3d*8 = 0x1e8`), consistent with the parse order. `A = bias` and never changes, so `delta = A + B`; the two kernel paths agree except for the repetition step.

Which ids carry what (`FUN_01848190`; the arrays are cleared at the start of each sampled request, `01848190.c:51-70`):

- BIAS ids (`param_3[0x42]` ids, `param_3[0x45]` values, `01848190.c:101,121,161`): entry with `A = delta = bias`, `count = 0`, `B = 0`, `flag = 0`. Bias applies from the first sampled token to every listed id, generated or not. No clamp. No dedupe in this path (a duplicate id would get two entries; the front end builds the list from a dict, so it cannot send one: inference).
- Prompt tokens (`01848190.c:209-218`): only when SAMPLE is engaged, `repeat != 1.0` and the prompt count (`+0x54`) > 0, ids `< 0x3ca00`. An existing entry gets `flag = 1`; a new entry gets `A = delta = B = 0`, `count = 0`, `flag = 1` (`:372`). Count 0 means presence and frequency do not apply to prompt tokens, matching `image/serve_api.py:3341-3343` ("presence and frequency count GENERATED tokens only").
- Generated tokens: `0184ad90` after each commit (callers `01848190.c:395`, `01851dc0.c:160,187,206,225,247`, `0182ea80.c:7645,7695,7711,7730`, `018514a0.c:16`, `01852ab0.c:5`).

Order on the logit row: (1) `+ bias`, (2) repetition on the biased value, flagged ids only, (3) `- frequency*n - presence`. All of it before the sampler kernels.

## Item 5: temperature order. Verdict: CLOSED

Sequence on a sampled step: penalties and bias on the raw logit (item 4), row max (`max`), `w = exp((x - max) * (1/T) * log2 e)`, rank threshold from raw-logit keys (top_k, T-free) and from T-scaled mass (top_p), min_p on the T-scaled `w`, renormalise over kept `w`, draw. Temperature therefore acts before top_p and min_p, and top_k is invariant to it. This answers the wire-doc open item `analysis/wire-protocol.md:259` ("Sampler order (top_k, top_p, min_p, temperature)"). The reciprocal is a full-precision fp32 divide (`L0:173`, `pick:472-491 v_div_scale_f32 ... v_div_fixup_f32 v6, v0, s38, 1.0`) multiplied in, not a divide per element.

Draw (partly inferred): seed hash per `closure/sampler-seed-r3.md`; `pick:231 v_mul_f32_e32 v9, 0x33800000, v1` (24-bit integer times 2^-24), `pick:258 v_dual_mul_f32 v2, s29, v9` (target = u * kept mass), then `pick:556 v_add_f32_e32 v1, v1, v10` and `pick:560 v_cmpx_ngt_f32_e32 v1, v9`: advance while `cum <= target`, so the first kept token whose inclusive cumulative weight is strictly greater than the target is drawn. The within-block visiting order (grid-stride per thread, `mass:700 v_add_nc_u32_e32 v1, s14, v1`) was not decoded; it does not change the distribution.

## Corrections to earlier writeups

- `closure/sampling-host-r2.md:52`: "top_k wins and top_p is ignored" is wrong; the stricter of the two applies (item 1).
- `analysis/kernels-speculation.md` section 1 point 5 ("The token-id compare `s_cmp_lt_u32 s0, s2` ... reject when the drafted id is outside the row's accepted range"): the operand is the draft token's sortable logit key against `Tkey`. Evidence: `acc:259 s_not_b32 s3, s1`, `acc:260 s_or_b32 s4, s1, 0x80000000`, `acc:261 s_cmp_lt_i32 s1, 0`, `acc:290 s_cselect_b32 s0, s3, s4`, `acc:293 s_cmp_lt_u32 s0, s2` with `s2` loaded at `acc:243` from scratch+0x4040. So accept_final rejects a draft token the rank filter removed, rejects if `min_p > w` (`acc:306`), and otherwise accepts with probability `w / Z_kept` (`acc:287 v_div_scale_f32 v3, null, v2, v2, v0`; `v2` = kept mass summed from scratch+0x800, `acc:193-214`).

## Residuals

- No runtime probe (static task). Tests that would confirm: (a) a crafted row with 5 equal top logits and top_k 3 should keep all 5; (b) a row where token j brings the cumulative mass to exactly p should keep j; (c) top_k 20 plus top_p 0.95 where the top 20 hold less than 95% of the full mass should show the top_k cut winning, and where they hold more should show the p cut trimming inside the 20 against the full-vocab mass.
- Edge arithmetic, inferred from the code and not exercised: tokens with `w < 2^-46` add 0 to the fixed-point mass; `top_p * Z * 2^45 < 1` gives target 0, every bin qualifies, and the select lands on bin 255 (an empty kept set on ordinary rows). Not reachable with normal p.
- Not decoded: NaN and infinity logits through the key transform and `v_exp`; the pick within-block visiting order; the residual weights in `rmass` and `rpick` beyond the shared `Tkey` and `min_p` tests (`rmass:260,263`, `rpick:822,826`); the penalty state seen by draft positions inside one speculative round (rows use the `0xfffffffe` per-row pen index, `FUN_01907ff0` entry).
- `+0x54` read as "prompt token count" is inferred from `01848190.c:387`, where it is passed as the draw counter.
- Histogram clearing between launches is done by block 0 in `max` (`max:10 s_cmp_lg_u32 s2, 0` guards the zero stores at `max:41-48`) and again inside the level kernels; the ordering between blocks was assumed, not proven.
