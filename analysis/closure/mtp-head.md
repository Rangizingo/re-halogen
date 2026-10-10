# MTP draft head: what it is and how it runs (closes S19)

Target: halogen-flash-server 0.17.3 `binaries/flash_serve`. Decompiles are `work/decomp/flash_serve/<addr>.c` (cited as `D:<addr>.c`). Labels: **proven** = quoted from a file or from the .hgn table; **inferred** = reasoned from names, sizes or call order.

Correction to S19 in `analysis/request-trace.md`: `018f9dc0` and `018f9fb0` are not empty HIP macro stubs. Both have real bodies (below). What is truncated is the kernel launch after `__hipPushCallConfiguration`, which Ghidra marks noreturn. So the host plumbing is proven; the head's layer kernels are still not decoded.

## 1. The head file tensor table (proven)

Source: Ai-core read-only, first 5064 bytes of `/srv/models/halogen-swift/qwen38-flash-next-mtp.hgn` (header and table only, no payload, no engine). Header: `HGN1`, version 2, n_tensors 31, table_off 104, data_off 0x1400, file_bytes 1523566720, model_id `qwen3.8-flash-next-mtp`. Parsed with the 160 B entry layout in `analysis/hgn-format.md` section 1. dtype codes: 0 bf16, 5 q4c (qparam 2), 7 q8g64.

| tensor (prefix `mtp.`) | dtype | dims |
|---|---|---|
| fc_embedding.weight | q8g64 | 2560 x 2560 |
| fc_hidden.weight | q8g64 | 2560 x 2560 |
| pre_fc_norm_embedding.weight | bf16 | 2560 |
| pre_fc_norm_hidden.weight | bf16 | 10240 |
| hyper_connection_mixer.hc_norm.weight | bf16 | 10240 |
| hyper_connection_mixer.input_mix_weight_down.weight | q8g64 | 320 x 10240 |
| hyper_connection_mixer.input_mix_weight_up.weight | q8g64 | 10240 x 320 |
| layers.0.attn_hyper_connection: block_inject_weight 4x10240, hc_norm 10240, input_mix_weight_down 320x10240, input_mix_weight_up 10240x320 | q8g64 (hc_norm bf16) | as listed |
| layers.0.mlp_hyper_connection: same four | q8g64 (hc_norm bf16) | as listed |
| layers.0.self_attn.q_proj.weight | q8g64 | 12288 x 2560 |
| layers.0.self_attn.k_proj.weight, v_proj.weight | q8g64 | 512 x 2560 each |
| layers.0.self_attn.o_proj.weight | q8g64 | 2560 x 6144 |
| layers.0.self_attn.q_norm.weight, k_norm.weight | bf16 | 256 each |
| layers.0.self_attn.indexer.index_qk_proj.weight | q8g64 | 640 x 2560 |
| layers.0.self_attn.indexer.q_layernorm.weight, k_layernorm.weight | bf16 | 128 each |
| layers.0.mlp.gate.weight (router) | bf16 | 512 x 2560 |
| layers.0.mlp.experts.gate_up_proj.weight | q4c | 512 x 1280 x 2560 (943,718,464 B) |
| layers.0.mlp.experts.down_proj.weight | q4c | 512 x 2560 x 640 (482,345,024 B) |
| layers.0.mlp.shared_expert: gate_proj 640x2560, up_proj 640x2560, down_proj 2560x640 | q8g64 | as listed |
| layers.0.mlp.shared_expert_gate.weight | q4c | 1 x 2560 |

Count check: 7 top-level (2 fc, 2 pre_fc_norm, 3 mixer) plus 24 under `layers.0` is 31, matching the engine log "1198 tensors (1166 trunk, 1 table, 31 head)" (`analysis/evidence/engine-0.17.3-run.log:31`). There is no `embed_tokens`, `lm_head` or final-norm tensor in the head: proven absent from the table.

## 2. Architecture

Proven from the table:
- One decoder layer (`layers.0` only), hidden size 2560, MoE FFN with 512 routed experts of intermediate width 640 (gate_up 1280 = 2 x 640) plus one shared expert (640) with its own scalar gate.
- The residual stream is 10240 wide, which is 4 x 2560. Every hyper-connection tensor and `pre_fc_norm_hidden` is 10240; `block_inject_weight` is 4 x 10240; the mixers are low-rank, rank 320. Three hyper-connection modules: the front `hyper_connection_mixer`, `attn_hyper_connection`, `mlp_hyper_connection`.
- Attention: k_proj/v_proj 512 = 2 KV heads x 256, q/k norms 256, o_proj input 6144 = 24 x 256. q_proj output 12288 = 2 x 6144.
- A sparse-attention indexer is present (index_qk_proj 640, layernorms 128), the same family as `analysis/kernels-indexer.md`.
- Two input projections, `fc_embedding` and `fc_hidden`, both 2560 x 2560, each with its own pre-norm. The head takes two inputs: a token embedding (2560) and the trunk hidden state (10240).

Inferred (sizes and names only, no kernel read):
- q_proj emits query plus a per-head output gate (12288 = 2 x 24 x 256).
- Input path: RMSNorm of the embedding (2560) then `fc_embedding`; RMSNorm of the 4-stream trunk hidden (10240), then the front `hyper_connection_mixer` collapse to 2560, then `fc_hidden`; the two are combined (sum or concat-equivalent, not shown) into the layer's 4-stream residual.
- Layer: attn hyper-connection, indexer-selected attention, mlp hyper-connection, top-k MoE (k is not in the head file).
- Embedding table and lm_head are the trunk's, because the head file has neither (section 4).

Engine side, proven: `D:0188f710.c:684` `"mtp.layers.0.experts: the head's experts must be Q4C-P (got %s / %s)"`, and the .hgn has dtype 5 (q4c) for both expert tensors. The loader builds `mtp.layers.0.self_attn.` + `q_proj.weight` and the other names (`:55-100`), plus `mtp.fc_embedding.weight`, `mtp.fc_hidden.weight`, `mtp.pre_fc_norm_*`, `mtp.hyper_connection_mixer.*` (`:954-1134`), and the per-layer hc names from the pointer table `PTR_s_attn_hyper_connection__019f6e60` (`:700`). It ends with `*(undefined1 *)(param_1 + 0xb7c) = 1` (head staged). `HALOGEN_MTP` defaults on; only a value starting with `0` skips it (`:66`, `getenv("HALOGEN_MTP")` then `*pcVar7 != '0'`). Head file path comes from `HALOGEN_MTP_HEAD` (`019aa550`); `019b01a0` refuses a file with no `mtp.*` names (`D:019b01a0.c:75-87`, `strncmp(__s1,"mtp.",4)`).

## 3. How the head consumes trunk state and the last token

`FUN_018f9dc0(model, token, depth, flag)` (`D:018f9dc0.c`) is the head step driver, called as in `analysis/gf_0x1851dc0.c:58-86`.

Proven:
- Guard: `if (*(char *)(param_1 + 0xb7c) != '\x01') return 0xffffffff;` (head not staged).
- Depth 0 with no pending flag: if the model holds a cached draft (`+0x40 >= 0`, `+0x44 == token`, `+0x48 == +0x2e8`, the current position) it returns `+0x40` with no launch (`:17-21`).
- Otherwise it copies 0x5000 bytes (20480 B = 10240 x 2 B, bf16-sized) device to device from the trunk's saved final hidden `model+0x50` into the head input buffer `model+0x930` (`:50-52`, `hipMemcpyAsync(...+0x930, ...+0x50, 0x5000, 3, 0, ...)`). 10240 equals the 4 x 2560 stream width in the table. The head's hidden input is the trunk's full 4-stream residual, not a 2560 projection.
- Depth greater than 0 (`param_3 != 0`): `FUN_018f99c0` syncs a 0x300 B block (`model+0x90` from `*(model+0x5e0)+0x25880`), then row `(rows-1)` of `+0x930` (stride 0x5000) is copied to row 0 (`:26-30`). So the head's own output hidden from the previous depth becomes the next input. The token fed is the previous drafted token (`gf_0x1851dc0.c:71-90`).
- Then `FUN_018f34c0(model, &token, 1, pos)` (`D:018f34c0.c`): memcpy the int token to host staging `+0x7b8`, `hipMemcpyAsync` H2D to device `+0x990`, then `__hipPushCallConfiguration(n + 0x100000000, 1, 0x100000100, 1, 0, 0)`, which is grid (n,1,1), block (256,1,1). Env `HALOGEN_MTP_WIRE` (default char 'D') is read once. This stages the last token id on the device for the head's embedding lookup (inferred use).

Inferred: the kernel launched after that configuration (decode lost) is the embedding gather plus the head layer chain. `k_embed_gather`, the `k_hc_*` kernels (`k_hc_mixed`, `k_hc_fold`, `k_hc_scatter`, `k_hcup4w`), `k_attn*` and the MoE kernels all exist in `analysis/fatbin-inventory.md`, and the trunk uses the same `HALOGEN_HC_*` knobs (`function-map.tsv`). No launch of a named head kernel was read.

## 4. Where the head forward and logits are launched

Proven host chain per draft step (`gf_0x1851dc0.c:58-67`): `018f9dc0` (stage hidden and token), `018f9fb0` (keep q row), `draft_sample` `019098d0`.

- `FUN_018f9fb0(model, k)` (`D:018f9fb0.c`): requires `k <= 7` (`"q_keep: row %d outside 0..%d"`, 7) and copies 0xf2800 B from the head logits buffer `model+0xd80` to `model+0x60 + k*0x3ca00*4`. 0xf2800 B = 995,328 B = 248,320 x 4 B = 0x3ca00 f32. So the head yields a full-vocab fp32 logit row (248,320 entries) and up to 8 are kept as the draft distribution q for acceptance.
- `lm_head_forward` `018f46b0` writes `model+0xd80` (`D:018f46b0.c:167`: `FUN_018e6f20(model+0x608, ..., *(model+0xd80), 0x3ca00, 1)`) using `model+0x608`, the trunk lm_head linear. It also holds a launch `__hipPushCallConfiguration(0x1000003ca,1,0x100000100,...)`, grid 970 x block 256 = 248,320 threads. `use_slot` (`D:019002f0.c:59`) swaps `+0xd80` per slot (`= plVar7[0x17]`). Calling `018f46b0` "the head's logits step" is inferred: the buffer link is proven, but it has no direct caller in the decompile set.
- Host function for the head forward launch: `018f34c0` (launch stub), reached from `018f9dc0`. The head layer compute is not in a decoded host function, so kernel names are unproven.

## 5. How draft tokens are sampled

`draft_sample` `019098d0` (`analysis/gf_0x19098d0.c`), proven:
- Preconditions: q row below 8 and kept (`"draft_sample: q row %d was not kept"`), head staged (`"the MTP head is not staged (HALOGEN_MTP=1)"`), temperature above 0 (`"temp=%g: temperature 0 is the GREEDY path and must stay on argmax()"`).
- It writes a 32 B config (`temp * tskew`, nucleus, args) to device `+0x9a0` (`hipMemcpy(...,0x20,1)`), picks the logit row (`+0xd80` when row is -1, else `+0x60 + row*0xf2800`), calls `sample_aux` `01908c30` (block 256, `D:01908c30.c`), and copies 4 B back from `+0x9a8`. The kernel is probably `k_sample_f32` (`analysis/kernels-speculation.md`), inferred.
- A second float comes back from `+0x9e8` (nucleus on) or `+0x9f0` (off). `spec_round_run` treats it as a log probability and stops the chain when `exp(logp) < threshold` (`gf_0x1851dc0.c:62-63` and `:76-78`, thresholds at `+0x144` for step 1, `+0x148` after). Chain length is capped by `param_2`, which `spec_verify_argcheck` bounds by `HALOGEN_SPEC_N` (`D:018fa0e0.c`, `param_3 <= *(int *)(param_1 + 0x14)`).
- Falsifier: when nucleus, qoff or tskew knobs differ from neutral, "Every proposal from this process is deliberately wrong (METHOD 37)" prints once. A test gate, not production behavior.
- Depth: step 0, step 1, then a loop up to `param_2` (`gf_0x1851dc0.c:71-90`). The banner "MTP depth-2 (up to 3) selectable" fits the loop; the host has no constant 3, the cap is the runtime `HALOGEN_SPEC_N`.
- Greedy: `draft_sample` refuses temp 0, so a greedy draft step must use another argmax path. Not located.

### Shortlist vocab (HALOGEN_DRAFT_VOCAB, HALOGEN_DRAFT_VOCAB_FILE)

Proven:
- Mode parse, `018f51e0` (`D:018f51e0.c`): unset or "1" gives 1; "0" gives 0 (off); "cycle" gives 2; "check" gives 4; any other string gives 3 (a named list, resolved by `018f5380`, error `"HALOGEN_DRAFT_VOCAB=%s: not found in %s"`).
- `01891c50` loads lists from the file named by `HALOGEN_DRAFT_VOCAB_FILE` when mode is not 0 and the model's list slots exist (`D:01891c50.c`). The string `"HALOGEN_DRAFT_VOCAB_FILE=%s: %s; not used"` is in the binary.
- In-memory list object (model `+0x70`): vector of 0x30 B entries at `+0x18`/`+0x20` (name ptr and length at entry +0 and +8, two ints at +0x20 and +0x28), per-entry bitset vector at `+0x30` (stride 0x18), token-id array at `+0x78`. `018f5320` tests bit `token` of the entry's bitset (false when `token > 0x3c9ff`). `018f52d0` returns `entry[+0x20] + entry[+0x28]`, default 0x3ca00.
- Selection inside `018f46b0` (`:30-45`): mode 2 picks `((step * -0x61c8864f) >> 11) % (nlists + 1) - 1` (-1 means full vocab); mode 3 uses fixed index `+0x88`; otherwise index `+0x78`. With a valid list and `+0xb00 == 0`, logits are computed by `FUN_018f6d80` over the list's token-id slice (`*(model+0x70)+0x78 + off*4`, count `entry+0x28`); else the full `FUN_018e6f20` lm_head over 0x3ca00 (`:167`). So the shortlist narrows the lm_head GEMM to listed token ids. `HALOGEN_DRAFT_VOCAB_Q8` exists as a string; its use was not traced.
- Hit tracking: `018715c0` keeps a 64-token window and per-list hit counts via `018f5320`; `01870d40` builds per-list limits and, in modes 1 and 4, scans the prompt for the token pair `0x3c8ed, 0x34e` to choose a list (`:79-85`); `0182ea80.c:7785` counts tokens outside the list as misses.

Open: the list file format. `/opt/halogen/draft-vocab.lists` does not exist on Ai-core (`ls /opt/halogen`: no such directory) and `draft-vocab.lists` is not in the binary strings. Ai-core holds `draft-vocab.txt` files (`/srv/models/ukisai_Swift-1.5-Qwen3.8-Flash-Next-GGUF/MTP/draft-vocab.txt`: 65,536 lines, 383,988 B, one token id per line starting `15 16 11 28 17 2129 198`; same head as `/home/rangizingo/bits-live/draft-vocab-64k-s109.txt`). That fits a 64k shortlist but I did not prove the engine parses it, nor what logits outside the list hold.

## 6. How verify reuses trunk compute

Proven:
- Verify runs the trunk over the chain: `spec_verify_argcheck` `018fa0e0` sets `*model = 1` and calls `018fa150` (`forward(n=%d): outside [1, %d]`), which needs a warm sequence (`+0x2e8 != 0`) and `n <= HALOGEN_SPEC_N`. It is the trunk's own n-token forward (`D:018fa150.c`, small-n path `param_3 < 9`).
- Hidden-state handoff: the trunk leaves one 0x5000 B row per verified position in `model+0x930`. `spec_commit` `018fc360` copies the accepted row `(keep-1)*0x5000 + +0x930` into `model+0x50` (`D:018fc360.c:71`), the same buffer `018f9dc0` reads as the next round's head input (section 3). Batched form: `joint_commit` `D:01902e70.c:55` copies `iVar3*0x5000 + +0x930`; `forward_batch` `D:01902090.c:101` reads rows of it. The head is fed by trunk activations from the verify pass, not recomputed.
- Acceptance compares the kept q rows (`+0x60`) with the trunk logits rows (`accept_probs` `01909e50`; kernels `k_accept_prob_f32`, `k_smpg_accept_final`), and `residual_sample` `0190aa50` draws the correction on a reject (`gf_0x1851dc0.c:137-185`).

## 7. Unproven

1. The kernels the head forward launches. `018f34c0` and `01908c30` end at a noreturn `__hipPushCallConfiguration`. Needed: the disassembly after those calls (`analysis/draft_sample.asm` style) and the device symbol behind each launch stub.
2. The exact fusion of `fc_embedding` and `fc_hidden`, the front mixer's role (collapse of 4 streams to 1 is inferred), router top-k, and whether the head layer has its own KV and indexer cache.
3. That embedding and lm_head are the trunk's (inferred from tensor absence and `model+0x608` at `018f46b0:167`).
4. The greedy draft step, and what draft logits outside a shortlist contain.
5. The draft-vocab list file format, the `HALOGEN_DRAFT_VOCAB_Q8` path, and the meaning of the prompt scan pair `0x3c8ed, 0x34e`.
6. Why the depth is "up to 3": no cap constant found; the cap is the runtime `HALOGEN_SPEC_N`.
7. Who writes the cached first-draft fields (`+0x40`, `+0x44`, `+0x48`); not traced.
