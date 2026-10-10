# U12 round 6: what NormFold n1/n2 hold, and what makes a Tap fire (flash_serve 0.17.3)

Update after run 5 (`analysis/runtime-run-0.17.3-r5.md`): the probe recipe below was run; the predicted slots fired in all 36 layers and the Tap payload (residual 1) is observed. Item 2 and residual 1 are closed there; the text below is the static analysis as written before the run.

Checked: CLAUDE.md, analysis/closure/dnc-scan3-r3.md, dnc-scan3-r4.md, analysis/runtime-run-0.17.3-r4.md, the U12 row of analysis/unknowns.md, work/decomp/flash_serve/{01884460,018ab5c0,018897f0,018680b0,01860720,01895170,018fa150,01905ae0,01852ca0,0186a470,0182ea80}.c, `objdump -d` of the whole binaries/flash_serve (via `ssh ai-core`, read only; VMA = Ghidra address minus 0x100000), `objdump -R` joined with analysis/function-map-kernel-stubs.tsv, the `<2,4,1,0,1,1,0,1,0>` scan3x slice in work/kernels (called k101.s below), image/serve_api.py, scripts/engine_probe.py, scripts/engine_probe6.py, analysis/evidence/run-r4-kargs.jsonl.gz (re-read with python, matmul and launch records).

## Scope

1. What the arena regions behind host globals DAT_019fc638 (n1) and DAT_019fc660 (n2) hold; every writer; whether n2 is a weight or scratch; sizes `local_748`, `local_790`, `local_8e8`.
2. Callers of FUN_01895170 (Tap targets) and the exact condition for a Tap to fire.

"Proven" = read in decompile, disassembly, kernel ISA or the run 4 records. "Inferred" is labelled. Nothing was executed on the vendor binaries.

## Verdict: CLOSED (item 1, n1 and n2)

n1 (`DAT_019fc638`) = the per-token output of the layer's `linear_attn.in_proj_z` projection, u16 [tokens][6144]. n2 (`DAT_019fc660`) = per-token gated-RMSNorm output, u16 (bf16 stores) [tokens][6144], the input of `linear_attn.out_proj`. n2 is scratch activation, not a weight. n0 is `linear_attn.norm.weight`.

### 1a. n1 is the z projection output (proven)

- Prefill, ungrouped route, `018ab5c0.c:298`: `FUN_018a7520(lVar19 + 0x80,*(undefined8 *)(param_1 + 0x948),DAT_019fc638,0x1800,uVar17,0xa00);`. Disassembly `17ab9aa: mov 0x150c87(%rip),%rdx # 18fc638`, `17ab9ba: mov $0x1800,%ecx`, `17ab9b1: mov $0xa00,%r9d`, `17ab9c2: call 17a7520`. Argument order (weight, in, out, N, tokens, K). The same call with `DAT_019fc628` and `0x2800` at `018ab5c0.c:297` is the qkv projection, and `0xa00` = 2560 is the hidden width (it is the N of the out projection in 1b).
- Grouped route (HALOGEN_HT_AB_GRP, default on, `018ab5c0.c:214,321`): `local_708 = DAT_019fc638` at `018ab5c0.c:260` and `17ab913: mov 0x150d1e(%rip),%rax # 18fc638` / `17ab91a: mov %rax,0x968(%rsp)`, then `FUN_018a53b0` / `FUN_018b0e40` (`018ab5c0.c:290-298`) with the same 0x2800 / 0x1800 widths.
- Layer weight at `+0x80` is `in_proj_z.weight`: `018897f0.c:797` appends `"in_proj_z.weight"`, `018897f0.c:819` stores `*(undefined4 *)(local_178 + 0x80)`. Siblings: `+0x8` in_proj_qkv (`:754,:776`), `+0xf8` in_proj_a (`:840,:862`), `+0x170` in_proj_b (`:883,:905`), `+0x1e8` out_proj (`:948,:970`), `+0x268` `"norm.weight"` (`:990-996`, prefix `"linear_attn."` appended at `:740`).
- Widths: 0x2800 = 10240 = 2048 + 2048 + 6144 (q, k, v) for qkv; 0x1800 = 6144 = 48 heads x 128 for z; 0x30 = 48 for a and b (`018ab5c0.c:261-296`). 6144 matches the 48 x 128 x 128 f32 state of r4 section 1.
- Run 4 confirms it: in `analysis/evidence/run-r4-kargs.jsonl.gz`, every one of the 108 `k_dnc_scan3x` launch records has `globals.n1_18fc638 = 0x70ce39560c00`, and exactly 108 `hipblasLtMatmul` records have `D = 0x70ce39560c00`. The r4 trace shows `[matmul] hipblasLtMatmul x3` before `k_conv1d_silu_tb<16>` in each DeltaNet layer (one of the three is the z GEMM; which one is hidden by the grouped route, inferred).

### 1b. n2 is the gated-norm output, scratch (proven)

Writers, all with the pointer in hand:

| path | writer | evidence |
|---|---|---|
| prefill, fold on (HALOGEN_DN_NORM_FOLD=1 or 3) | `k_dnc_scan3x<..,i3=1,..>` epilogue, 4 x `global_store_d16_hi_b16` per row | host `17af26f: mov 0x14d3ea(%rip),%rcx # 18fc660` then `17af276: mov %rcx,0x5d8(%rsp)` (NormFold n2); kernel `k101.s:4870 v_mad_u64_u32 v[1:2], null, 0x3000, v30, v[114:115]` then `k101.s:4881-4884 global_store_d16_hi_b16 v[1:2], v4/v3/v5/v6, off offset:0/64/128/192`. `v[114:115]` is n2 + head*256 B: `k101.s:164,167 s_add_u32 s9, s28, s12 / s_addc_u32 s10, s29, s13` (s12:13 = head*128*2 B, `k101.s:139`), `k101.s:210 v_add_co_ci_u32_e64 v115, null, s10, 0, s9`. The low half (v114) is a `.long 0xd7000972` word the objdump does not decode, so v114 is inferred |
| prefill, fold off (default: HALOGEN_DN_NORM_FOLD unset returns 0, `018b4550.c:8-13`) | `k_rmsnorm_gated` (handle 18f65f0 = stub 018b46a0), `k_rmsnorm_gated_w` (handle 18f65e8 = stub 018b45e0) | `17af6b0..17af7ac`: args `fc658` (x), `layer+0x268` (weight), `fc638` (z), `fc660` (out), `$0x80`; second variant `17af80b..17af91f` |
| decode step (rows < 9) | `k_dn_recur_step<N,b>` arg 10 (`unsigned short*`) | `17ac671-17ac69d`: `rbp = row*0x3000`, `fc638+rbp` -> `0x2e0(%rsp)`, `fc660+rbp` -> rbp; `17ac946: mov %rbp,0x290(%rsp)` is slot 10 of the arg array `0x330..0x388`. Stub `0195d6c0` signature (function-map-kernel-stubs.tsv line 1328) has `unsigned short*` at index 10; index 9 = z (`0x268(%rsp)` = `fc638+rbp`), index 8 = `layer+0x268` (`17ac6b4: mov 0x268(%rbx),%rcx`), index 4 `unsigned short*` = `fc658+rbp` |
| prefill, rows variants | `k_dn_recur_rows<false,*,*>` (handles 18f6508, 18f6588, 18f6590) | `fc658, fc638, fc660, fc640, fc648` loaded together at `17ad442-17ad45e`, `17ad9a1-17ad9af`, `17adbe8-17adc05`; per-argument roles not decoded |
| mode 2 (HALOGEN_DN_NORM_FOLD=2, A/B check) | the fold kernel writes `DAT_019fc958` (hipMalloc, `n*0x3000+0x100` B, `17af211-17af22f`) instead; `fc660` still gets the separate-kernel result and `17af99d..17af9b8` calls a compare (`18ca120`) on both | `17af267-17af276` (`cmp $0x2,%eax; je`) |

No GEMM writes n2: 0 of the 2,220 matmul records in run 4 have `D = 0x70ce69b60c00` (n2).

Consumer (proven): `17afa6a: add $0x1e8,%rdi` (out_proj.weight), `17afa71: mov 0x14cbe8(%rip),%rsi # 18fc660`, `17afa80: mov $0x1800,%r9d` (K = 6144), `17afa86: mov $0xa00,%ecx` (N = 2560), call `17a7520`, output `0x938(%r12)`. Run 4: 108 matmul records have `A` or `B = 0x70ce69b60c00`, the first one 7 records after launch index 48.

What the fold kernel computes (proven arithmetic, `k101.s:4660-4880`): per 128-wide head row, `x` (stream from the scan output pointer `s22`) is squared and summed across lanes (`ds_bpermute_b32`), divided by `w` = 128.0 (`k101.s:205 s_cvt_f32_i32 s62, s8`, `k101.s:4797 v_div_fixup_f32 v9, v10, s62, v9`), plus `0x358637bd` (about 1e-6, an epsilon, `k101.s:4799`), `v_rsq_f32`, times the n0 weight (`k101.s:4674` shifts the loaded u16 to f32, `k101.s:4837 v_mul_f32 v5, v5, v25`), times `1/(1+exp(-n1))` from n1 (`v_mul_f32 v4, 0xbfb8aa3b, v4` is -log2(e) x z, `v_exp_f32`, `v_add_f32 1.0`, `v_rcp_f32`/`v_div_fixup_f32`, then `k101.s:4841 v_mul_f32 v4, v4, v5`), rounded to bf16 (`v_bfe_u32 .., 16, 1` and `k101.s:4873 v_add3_u32 .., 0x7fff`) and stored. Inferred: the gate is sigmoid(z) as written; no second multiply by z (which silu would need) appears in `k101.s:4700-4880`, so either this model's gate is sigmoid-only or the z factor sits elsewhere. Not resolved here.

### 1c. Correction to r4

r4 section 3 says the n1 stream is "6144 B per token". The stride constant is `v_mad_i64_i32 .., 0x1800, tok, v[110:111]` (`k101.s:4609`), an element index of u16, so 6144 elements = 12288 B per token. Host evidence: step path `17ac671: shl $0xc,%rax` / `17ac675: lea (%rax,%rax,2),%rbp` = row x 0x3000 B; the fold kernel store stride is `0x3000` (`k101.s:4870`); the mode 2 buffer is `n * 0x3000 + 0x100`.

### 1d. Sizes (01884460.c)

- `ctx+0x170` is `param_3` (`:286`). `:654-657`: `lVar28 = (long)*(int *)(param_1 + 0x170); unaff_R13 = lVar28 + 0x3f` (`+0x7e` when negative). `:805`: `lVar27 = unaff_R13 >> 6`. So lVar27 = ceil(ctx[0x170] / 64), a count of 64-token blocks (inferred for paths where Ghidra shows R13 as "unaffected"; the live run fixes the value, below).
- `local_748` (n1): `:902-910` `local_750 = local_8b8 + (uVar24 & 0xfffffffffffc0100) + 0x2d000; local_748 = local_750 + (uVar24 & 0xfffffffffffc0100)` with `uVar24 = lVar27 * 0x140000` (a qkv-sized block, 0x140000 = 64 x 0x5000 B). `:1276` `DAT_019fc638 = local_748 + local_668`.
- Size of n1: `:911-922` `uVar24 = lVar27 * 0xc0000` (rounded), `local_760 = local_748 + uVar24`. 0xc0000 = 64 tokens x 12288 B. n1 = lVar27 x 0xc0000 B = 64 x lVar27 tokens x 6144 u16.
- `local_790` (n2): `:937` `local_790 = local_758 + uVar24`, `:938` `local_8c8 = local_790 + uVar24`. Same size as n1. Between them: fc640 and fc648 (a, b: lVar27 x 0x1800 B each, `:923-936`) and fc658 (`local_758`, lVar27 x 0xc0000 B; the scan output `x` of 1b, inferred from the `k_rmsnorm_gated` argument order).
- Live check (run 4): n2 - n1 = 0x70ce69b60c00 - 0x70ce39560c00 = 0x30600000 = 0x18000000 + 0x300000 + 0x300000 + 0x18000000, which is exactly `lVar27 x (0xc0000 + 0x1800 + 0x1800 + 0xc0000)` for lVar27 = 512 (0x200). So lVar27 = 512, token capacity 32768 (the `chunk 32768` in the engine banner, `run-r4-engine-0.17.3-r4.log:54`), and n1 and n2 are each 0x18000000 B = 384 MiB (the pointer difference is observed; the formula is read from the decompile).
- `local_8e8` (n3, `DAT_019fc7b8`): `:1038` `local_8e8 = (uVar13 & 0xfffffffffffff900) + 0xf100 + local_878`; `:859` `local_740 = local_878 + (uVar13 & 0xfffffffffffff900)`; so n3 = `DAT_019fc650 + 0xf100` (`:1257` `DAT_019fc650 = local_668 + local_740`, `:1258` `DAT_019fc6c0 = local_740 + local_668 + 0xf000`). The enclosing block is sized `:860` `local_888 = local_740 + (lVar27 * 0xc0 rounded) + 0xf100`, so the n3 sub-block is lVar27 x 0xc0 B = lVar27 x 48 x 4 B (96 KiB at lVar27 = 512). n3 role (inferred): u32 flags or counters, one per (64-token block, head). The fold kernel does `k101.s:4179 global_atomic_inc_u32 v41, v107, v179, s[10:11] glc` with `s[10:11] = n3 + 4*head + (s[24:25] << 2)` (`k101.s:105-106,4177-4178`) and `v179 = 1` (`k101.s:161`), an increment with limit 1. Who zeroes it was not found.

## Verdict: PARTIAL (item 2, Tap targets)

All code paths from the wire to the kernel argument are traced. Two inputs are inferred and nothing was run, so a tap firing is still unobserved.

### 2a. Callers (proven)

- FUN_01895170 has one call site: `018680b0.c:264` `FUN_01895170(lVar14,&local_48);`. A scan of the full objdump for `call 1795170` finds one hit, `176852d` (inside the function at VMA 17680b0).
- `01895170.c:6-27`: `*(int*)(ctx+0x4b0) = max(param_2[0], 0)`, `+0x4b4 = max(param_2[1], 0)`, `+0x4b8 = max(param_2[2], 0)`, and zeroes `+0x4bc/+0x4c0/+0x4c4` and the flags `+0x4c8/+0x4c9/+0x4ca`.
- `int[3]` = `{local_48 low, local_48 high, local_40[0]}` (`018680b0.c:28-29` declares them adjacent). FUN_018680b0 is the prefill step of the prompt cache (cache-host.md), called from `01846b20.c:169,374,399` with `param_2 = request + 0x308` (the "plan").

### 2b. What fills the three slots (018680b0.c, 01860720.c)

Plan fields built by `01860720.c` (the plan constructor; sole call at VMA `173ccda`, args from `173cc9c-173ccbf`: `ecx = req+0x54` n_prompt, `r8d = req+0xb0`, `r9d = req+0xb4`, 7th arg `req+0xb8`):

| plan field | value | condition | source |
|---|---|---|---|
| `+0x14` | `min(param_5, n)` if `param_5 > 0`, else `n` (mode >= 2); `n - n % chunk` (mode < 2) | none | `01860720.c:88-100` |
| `+0x18` | `param_6` | `0 < param_6 < n` and cache mode >= 2 | `01860720.c:74-75` |
| `+0x1c` | `param_7` | `0 < param_7 < n`, mode >= 2, `cache+0xa1` (HALOGEN_CACHE_SNAP3, default on) | `01860720.c:78-79`, `01852ca0.c:95-101` |
| `+0x28,+0x2c,+0x30` | `end-0x1000`, `end-0x400`, `end-0x4000` (tail checkpoints) | mode >= 2, `cache+0xa3` (HALOGEN_CACHE_CKPT, default on), `cache+0xd9` (HALOGEN_CACHE_INPLACE, default on), `cache+0xb0 < 0` (init -1), `cache+0xd8 == 0`, and `max(plan+0x18, done, plan+0x1c) + 0x2000 <= candidate` | `01860720.c:102-126` |

`param_5/6/7` are the wire fields SNAP, SNAP2, SNAP3 (inferred by order: the engine parse stores them in three adjacent stack ints, `1734052` (SNAP, `lea 0x430(%rsp)`), `1734bd3` (SNAP2, `0x434`), `1733e33` (SNAP3, `0x438`), each checked against `0..n_prompt` (`173407f: cmp 0x3d4(%rsp),%eax; jle`) with the strings "SNAP outside 0..n_prompt", "SNAP2 outside 0..n_prompt", "SNAP3 outside 0..n_prompt" at `0182ea80.c:3004,3230,3239`; the copy into `req+0xb0/0xb4/0xb8` was not located).

`018680b0.c:80-264` only runs when `!bVar4 && (bVar1 & 1) == 0`: `bVar4 = cache.mode < 2` (`:61`, mode = HALOGEN_PROMPT_CACHE, `01903680.c`) and `bVar1 = cache+0xd8` (`:79`), the "cache DISABLED" flag (set to 1 at `0186a470.c:362` after a failed snapshot store). So: HALOGEN_PROMPT_CACHE=2 and the cache not disabled.

Slot assignment (piece = `n` tokens from `done`, `iVar16 = done + n`, `chunk = min(HALOGEN_PREFILL_CHUNK, ctx+0x2e0)` via `FUN_01905000`, `:81-96`):

- slot0 (`ctx+0x4b0`) <- `plan+0x18` (`:91`, SNAP2, the end of the system turn); slot1 (`ctx+0x4b4`) <- `plan+0x1c` (`:92`, SNAP3); slot2 (`ctx+0x4b8`) <- `plan+0x14` (`:90`, SNAP, history end). Cross-check: the kind-0 store passes `0 < (int)param_2[3]` (plan+0x18 > 0) as the anchor flag, `:297`.
- Each candidate p must satisfy `done <= p < done + n`, `p != done`, and `p - done >= 64` (`0x3f < d`). The stored target is `done + (d & 0x7fffffc0)` when `n <= chunk` (`:114-146`), or the chunk-relative `p - (r & 63)` with `r = (p - done) % chunk` when `n > chunk` (`:98-113`). Duplicates are dropped.
- Tail checkpoints (`:164-262`) fill the first free slot in the order low, high, `local_40`, flagged `local_4c[k]` so the store uses `FUN_0186f7f0` instead of `FUN_0186a470`.
- A full-prompt end position `p == n_prompt` never taps (`p < done + n` fails); that entry is stored after the forward (`:336`).

### 2c. From target to kernel (018fa150.c, gdn_layer_forward)

- `018fa150.c:43-83`: `c_k = target_k - past` goes to `ctx+0x4bc/4c0/4c4` only if `ctx+0x358 == 0 && ctx+0x338 == 0` (both are `std::function` manager slots, `01893b70.c:103-107`; installer not traced), `c != 0`, `past <= target < past + n`, `(c & 0x3f) == 0`, and the single-token step path is off (`n < 9 && past > 0 && ctx[0x408] == 0` makes `bVar8 = 1`, `:30-33`).
- `01905ae0.c:30-34`: when `n > min(chunk, ctx+0x2e0)` the call is split into sub-forwards; the test above then applies per sub-forward with absolute targets.
- gdn_layer_forward per layer, tap k (`17aefb4..17af047` for k = 0; `17af047..17af0da` and `17af0da..17af16d` for k = 1, 2): `cmpl $0x0,0x4bc(%r12); jle` (c > 0), `hipMemcpyAsync(ctx+0x4e8, layer+0xb88, 0x300000, 3)`, `test $0x3f,%al` (c % 64 == 0), `cmp %r14d,%eax; setg` (c <= n), `cmpb $0x1,0x14d88f(%rip) # 18fc8a8` (HALOGEN_TAP_FROM_SCAN = 1, default per `018951f0.c:13`), then `slot_k = (c + 63)/64 - 1 = c/64 - 1` (`17af023..17af030`) and `p_k = ctx+0x4e8/4f0/4f8` (`17af037..17af03f`, r4 section 1).
- Kernel side: kernarg `+0x80/+0x84/+0x88` = slot0..2, `+0x90/+0x98/+0xa0` = p0..p2 (`k101.s:101 s_load_b256 s[52:59], s[0:1], 0x80`, `k101.s:120 s_load_b64 s[0:1], s[0:1], 0xa0`; r3 section 2). In run 4 the 12 bytes at `+0x80` are `ff` x 12 in `kernarg_hex` of launch record 0.

### 2d. Why run 4 saw no tap

`scripts/engine_probe.py:64-72` builds `GEN <req> <max> 2 151645 151643 <n> <ids> [drafter] [SAMPLE ...]` with no SNAP fields, and run 4 used HALOGEN_PROMPT_CACHE=2 (`run-r4-engine-0.17.3-r4.log:54`, "resume-anywhere"). With SNAP = SNAP2 = SNAP3 = 0: `plan+0x18 = plan+0x1c = 0`, `plan+0x14 = n` (never strictly inside the piece), and the prompts (300 and 1100 tokens) are below the 0x3000 = 12288 tokens a tail checkpoint needs. So `int[3] = {0,0,0}`, FUN_01895170 set all three targets to 0, and every slot stayed -1. (The probe log truncates the GEN lines at 53 words, so the script, not the log, is the evidence.)

### 2e. Who sends non-zero SNAP fields in production

`image/serve_api.py`: `Engine.generate` appends ` SNAP {snap}`, ` SNAP2 {snap2}`, ` SNAP3 {snap3}` when non-zero (`:1080-1082`). `render_prompt` computes them: `:3479` snap = token boundary of the template without the generation prompt; `:3485` snap2 = start of the first `<|im_start|>user` when message 0 is a system turn; `:3495` snap3 = start of the last user turn, only when `HALOGEN_CACHE_SNAP3` is not 0 (`:47`). So a chat request with a system message and at least two user turns sends all three; a single-turn chat sends SNAP (and SNAP2 with a system message). No HALOGEN_* env sets a target directly. The other automatic source is the tail checkpoints (long prompts, table in 2b).

## Probe recipe (item 2)

1. Engine: `HALOGEN_PROMPT_CACHE=2` (as run 4), leave `HALOGEN_TAP_FROM_SCAN` unset or 1 and `HALOGEN_CACHE_SNAP3` default; keep the shim capturing kernarg bytes `0x80..0xa7` of every `k_dnc_scan3x` launch.
2. Send one cold raw GEN with 1100 fresh token ids (new salt, no cache hit) and append after the ids ` SNAP 1000 SNAP2 500 SNAP3 800` (no drafter field); in `scripts/engine_probe.py` add the three fields to `parts` in `gen`.
3. Expect, in each DeltaNet layer's scan3x launch of that forward: i32 at `+0x80/+0x84/+0x88` = 6, 11, 14 (targets 448, 768, 960 = floor-to-64 of 500, 800, 1000 from done = 0, then `/64 - 1`) and pointers at `+0x90/+0x98/+0xa0` = ctx+0x4e8/4f0/4f8, plus three 3 MiB `hipMemcpyAsync` before each scan; run 4 had `-1,-1,-1` and null pointers in all 108 launches. A SNAP-only probe (` SNAP 1000`) should give `-1,-1,14`. Without wire changes: a cold prompt of 12288 or more raw token ids makes tail checkpoints (end-4096 at 12288 tokens, end-1024 as well) fill slots 0 and 1.

## Residuals

1. Payload of the tap: what the kernel writes into p[k] (LDS dump loop at `k101.s:8560-8575`, 0x1f7f bound) versus the 3 MiB state copied by the host; unchanged from r4 residual 2.
2. `ctx+0x338` and `ctx+0x358` (std::function slots): if a hook is installed the forward never taps; which feature installs them is not traced. A text-only GEN is expected to leave them null (inferred; run 4 reached scan3x on the same path).
3. The copy of the parsed SNAP fields into `req+0xb0/0xb4/0xb8` (the plan constructor arguments) is inferred from order and range checks.
4. Gate arithmetic (sigmoid versus silu) in the fold epilogue, and the standalone `k_rmsnorm_gated` body, are not compared.
5. Which of the three grouped-route GEMMs writes n1 per layer (runtime shows one `D = n1` per scan3x launch, 108 of 108).
6. n3 reset and its exact index (`s[24:25]` at `k101.s:4177`).

## Reproduce

```bash
ssh ai-core 'cd ~/re-halogen && objdump -d --no-show-raw-insn binaries/flash_serve' > fs.dis
grep -nE '# (18fc660|18fc638|18fc7b8|18fc958)' fs.dis         # every host use of n2, n1, n3, mode-2 buffer
grep -n 'call.* 1795170' fs.dis                                # sole caller of FUN_01895170 (176852d)
sed -n 262,266p work/decomp/flash_serve/018680b0.c
cd analysis/evidence && python3 -I - <<'E'                     # n1 = matmul D, n2 = matmul A or B
import gzip, json
r = [json.loads(l) for l in gzip.open('run-r4-kargs.jsonl.gz', 'rt')]
mm = [x for x in r if x['kind'] == 'matmul']; la = [x for x in r if x['kind'] == 'launch']
n1 = la[0]['globals']['n1_18fc638']; n2 = la[0]['globals']['n2_18fc660']
print(len([m for m in mm if m['D'] == n1]), len([m for m in mm if n2 in (m['A'], m['B'])]), len([m for m in mm if m['D'] == n2]))
E
```
