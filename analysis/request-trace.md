# One chat request, end to end (flash_serve 0.17.3)

Scope: `POST /v1/chat/completions` from `image/serve_api.py` to the engine and back. 31 steps. Each is marked `proven` (I read the cited line or decompile in this pass) or `inferred` (with the reason). Addresses are `flash_serve` functions; `work/decomp/flash_serve/<addr>.c` is the Ghidra decompile, `D:` below means that directory. `main` is `FUN_0182ba00` (`analysis/main_pseudo.c`). Names in quotes come from `analysis/function-map*.tsv`; confidence is theirs unless a step says otherwise.

Source note: the tree used for reading was `/Users/peterb/Documents/AI/re-halogen` (the worktree lacks the untracked `work/` and some `analysis/` files). Nothing was run. Live behavior comes from `analysis/runtime-run-0.17.3.md`.

Engine structure learned here, used by every step: the request handler is one very large function `FUN_0182ea80` (52,292 bytes, 9,512 decompiled lines, labelled `parse_request_json` in the map, which is wrong). It holds the verb parser, the admission code, the KV pool logic and the decode tick. Calling it the scheduler is the better label.

## A. Front end (serve_api.py)

### S1. HTTP route. proven
`@app.post("/v1/chat/completions")` at `serve_api.py:3946`, handler `chat_completions` at :3947. NPU models branch off at :3948. `check_sampling`, `tool_choice_mode`, `force_prefill`, `thinking_on`, `chat_kwargs` run at :3950-3960.

### S2. Messages, template, tokenizer. proven
`normalize_messages(req.messages)` (:3964, defined `image/tool_parse.py:350-473`); a `ValueError` becomes HTTP 400. Images are collected (:3970). `render_with_images` (:3978) calls `render_prompt` (:3425), which runs `tok.apply_chat_template(..., tokenize=False, **kw)` (:3427, :3438, :3440) and then `tok(text, add_special_tokens=False, return_offsets_mapping=True)` (:3460). Tools reach the template through `kw` (chat_kwargs). A forced tool choice appends `pre` tokens after the rendered ids (:3982-3983).

### S3. SNAP positions. inferred
`analysis/wire-protocol.md` section 2 cites `:3463-3505` (SNAP = end of the stable prefix, SNAP2 = end of the system turn, SNAP3 = start of the last user turn). I saw the return of `snap3` at :3584-3587 but did not re-read the arithmetic. Image pad expansion shifts the positions (:3584, `expand_image_pads`).

### S4. serve(): think, guard, schema, escape. proven
`serve` (:3589) computes `after` (SCHEMA and thinking), `escape` (SCHEMA and tools), `think` budget (:3600-3608) and `guard` (`guard_for`, :3611). These become the `AFTER`, `ESCAPE`, `THINK` and `GUARD` fields of the GEN line. Tool call syntax is Qwen3-Coder XML, parsed later by `tool_parse.ToolStream` (`analysis/api-layer.md` 1.5).

### S5. GEN line written. proven
`Engine.generate` (:1063) builds `GEN <req> <max_tokens> <n_eos> eos... <n_ids> ids... [drafter] [SAMPLE ...] [SNAP n] [SNAP2 n] ...` (:1074-1097; the SNAP2 fragment is at :1081). Call site `engine.generate(ids, max_tokens, eos_ids, drafter, sample, ...)` at :2366. Field table: `analysis/wire-protocol.md` section 2. Greedy means no `SAMPLE` field.

## B. Engine intake

### S6. Socket accept. proven
`main` binds and listens, then loops `accept(fd,0,0)` (`main_pseudo.c:1070-1135`, accept at :1075, `setsockopt(fd,6,1,...)` = TCP_NODELAY at :1077). Each accepted fd goes to `FUN_0182ea80(&local_1458)` and is `close`d on return (:1111-1112). One connection is therefore served to completion before the next `accept`. That matches the observed behavior that a second connection gets no `GEN`/`CSTAT` reply (`runtime-run-0.17.3.md`, U26). SIGINT and SIGTERM set a stop flag (`sigaction(2)` and `sigaction(0xf)`, :1070-1072, handler `FUN_01844d40`).

### S7. Line parse and queue. proven
`FUN_0182ea80` reads a line (`recv_line_append` `FUN_018456c0`), puts it in an `istringstream`, extracts the first word and switches on its length (`D:0182ea80.c:985`). Length 1 is `X`, 3 is `GEN`, 4 is `PING` (compared to `0x474e4950`, replied with `0x474e4f50` = "PONG", :1313-1330), `INFO` or `VIMG`, 5 is `VOCAB` or `CSTAT` (`bcmp(...,"CSTAT")`, :1568). A `GEN` is not parsed here. The whole line string is pushed on a deque and traced as "GEN queued (%zu bytes), %zu waiting, %zu active" (:1249-1310; trace needs `HALOGEN_SERVE_TRACE=1`). `VIMG` stores the RGB payload and rejects bad sizes with messages at :1499 and :1510 (multiple of a unit, max pixels).

### S8. GEN field parse and validation. proven
When a queued GEN is taken, the same function parses it with `>>`: req id, max_tokens, `n_eos` and the eos ids, `n_ids` and the prompt ids (`D:0182ea80.c:2380-2480`), then optional keyed fields. Rejections found as literals, all sent through `refuse_request` `FUN_018497a0` (:3632 and others): "id out of range" (:2482), "PENALTY/REPEAT/BIAS need SAMPLE with temperature > 0" (:2505), "SAMPLE temperature outside (0, 2] or top_k < 0" (:2528), "SCHEMA with SAMPLE temperature > 0 is not supported yet: structured output decodes greedy" (:2541), "LOGPROBS with SCHEMA is not supported" (:2550), "SNAP outside 0..n_prompt" (:3004), same for SNAP2 (:3230) and SNAP3 (:3239), "AFTER id out of range" (:3136), "malformed THINK" (:3197), "malformed GUARD" (:3216), "ESCAPE count outside 0..16" (:3316), "SCHEMA larger than 1 MiB" (:3346), "unknown field " (:3394). Grammar needs the VOCAB blob: "structured output needs the tokenizer's vocabulary and this connection sent no VOCAB line" (:5244). Valid keys seen: `SAMPLE LOGPROBS TOPLP PENALTY REPEAT BIAS SNAP SNAP2 SNAP3 IMG AFTER THINK GUARD SCHEMA ESCAPE` (this matches the front end's list). One engine rule the front end does not state: SCHEMA plus SAMPLE temperature above 0 is refused, so structured output is greedy only in 0.17.3.

### S9. Admission. proven
After a valid parse the request is admitted with a log line "req %ld admitted: slot %d, region base %d cap %d, prompt %d, drafter %d, %s" (`D:0182ea80.c:5355`, trace only). If the KV pool cannot fit prompt plus max_tokens the log is "req %ld waits: no room for %d positions" (:5411) and the stderr text "kv pool: request %ld waits for %d positions: the pool holds ... it waits for a retirement" (:5602). The wait releases through eviction or region moves (strings at :6019, :6258, :6645, :6733, :6999, :9407, :9467). `max_tokens` is clamped to the room left (:6019: "max_tokens %d -> %d"), which is the `room_from`/`max_tokens` pair the D line reports.

### S10. Slot and KV pool assignment. inferred
Evidence: `slots_alloc` `01900510` and `slots_alloc_pooled` `01900aa0` (high), `use_slot` `019002f0` (high, called at `0182ea80.c:7582`, :7593, :7771, :8238), `slot_region` `01900fc0`, `kv_pool_alloc_region` `0184de40` (med, called at :5957, :6059, :6915), `kv_move_region` `01901ee0`, `kv_copy_region` `01901c50`. The decompile windows I read confirm the names and the log text but I did not read the allocator bodies. Layout (one pool of positions, region per request, base and cap) comes from the admit line above and from INFO `kv_pool`.

## C. Prefix cache and prefill

### S11. Prefix cache lookup and restore. inferred
Chain seen: `prefill_request_with_log` `01846b20` calls `cache_lookup_and_store` `018680b0` (`D:01846b20.c:169`, :374, :399). `prompt_cache_try_restore` `0187fb00` (high) holds the messages "the entry's KV rows are on slot %d and this request is on slot %d; it runs cold (HALOGEN_CACHE_INPLACE=0 copies the rows instead)", "engine at %d below the entry's %d ... the entry is dropped" and "snapshot load failed; this request runs cold" (`D:0187fb00.c:74`, :84, :108). Disk tier: `prompt_cache_disk_load` `01881240` and `prompt_cache_disk_restore` `01882340`. The "ladder: hit %d in region %d, %d rows past it ... victim" log is at `0182ea80.c:6562`. The hit length is the `n_cached` field of the D line (observed: 0 cold, 19 warm in `runtime-run-0.17.3.md`). Why inferred: I did not read the key match (`key_compare` `01868c70` is labelled low) or the hit selection code.

### S12. Prefill in chunks. proven
`FUN_0182ea80` picks a chunk size and calls `FUN_01846b20(engine, request, chunk)` (`D:0182ea80.c:2100-2124`). The chunk comes from `get_prefill_chunk` `01905000`, which reads `HALOGEN_PREFILL_CHUNK` (`D:01905050.c:9`), capped by the engine's max_tok (`D:01846b20.c:129-141`). With active streams the prompt is split in pieces of `HALOGEN_ADMIT_CHUNK` tokens with `HALOGEN_ADMIT_TICKS` decode ticks between pieces (`main_pseudo.c:1010-1050`, messages at :1038-1043). Lone path: `FUN_01905ae0(model, tokens + done, n)` (`D:01846b20.c:161`), and `01905ae0` calls `forward` `018fa150` for each slice (`D:01905ae0.c:37`, :127). Cache path: `018680b0` does the same call at `D:018680b0.c:267`. A pause point exists between slices: `FUN_01847a30` logs "req %ld cancelled during its prefill (%d of %d" (S31). Log on completion: "flash_serve: req %ld prefill %d/%d tokens, %.1f s" (`D:01846b20.c:429`).

### S13. Snapshot store during prefill. proven
Inside prefill, when the prefilled position equals one of three request fields (`param_2[100]`, `+0x31c`, `+0x324`) and the cache is on, it calls `snapshot_store` `0186a470` (`D:01846b20.c:262-270`). `018680b0` stores three entry kinds after the tail is computed (`D:018680b0.c:294-314`, kinds 0, 1, 2). `cache_entry_kind_name` `0186cdc0` lists "full entry, tail checkpoint, anchor". The three request fields are SNAP, SNAP2 and SNAP3 by position and by the "SNAP outside 0..n_prompt" checks at S8 (inferred match; the field names are not in the decompile). `forward` itself only honors a snapshot boundary when its distance from the current position is a multiple of 64 and the boundary is inside the slice (`D:018fa150.c:50-83`, mask `& 0x3f`). Snapshot failure disables the cache ("snapshot store failed cache DISABLED", `0186a470`).

### S14. Per-layer dispatch. proven
`forward` `018fa150` checks `n` in [1, max], checks `sequence would reach %d > max_seq %d` (exit 1), sets the decode-shape flag when `n < 9` and `pos > 0` (`D:018fa150.c:30-52`), copies the token ids to the device (`hipMemcpyAsync`, :121-124) and launches the embedding gather. The layer loop is `FUN_018f2060(model, layer, n)` (labelled `forward_timed_profile`, med; it is the real loop body, its only other use being event timers under `HALOGEN_FLASH_MOE_TIME`). Per layer it runs, in order (`D:018f2060.c:114-240`):
1. `prefill_forward` `0189abe0(..., 0)`: pre-attention stage.
2. If byte 0 of the layer record (stride 0xc68) is 1: `sparse_attn_block_select` `018b86d0`; else `gdn_layer_forward` `018ab5c0`.
3. `018a5150` and `018ab0a0`: output stage (unlabelled).
4. `0189abe0(..., 1)`: post-attention stage.
5. `moe_layer_forward` `018d7510` (calls `018e4580` route-decode-fused and the `i4r` plan functions).
The loop bound is layer index 0x2f (`:229`, `:263`), so 48 layers. This is the trunk the observed run reports.

### S15. Which kernels per layer family. inferred
Hybrid layers: the two branches in step 2 are the two families (a linear-attention layer and an indexer sparse-attention layer). `gdn_layer_forward` carries the `HALOGEN_DN_*` flags and the string "gdn: mixed in-place / ring rows" (map, med). `dn_chunk_forward` `019d7b70` is its prefill kernel launcher (high, "norm_read set without qinv/kinv"). Sparse attention: `sparse_attn_block_select` carries "gives %d blocks > block_topk"; `indexer_select` `018b4750` ("ix_select: no candidate scratch") is reached through `018b5ef0`; attend launchers `attn_qs_launch` `019eca20` (called from `018bddd0`), `attn_fa_launch` `019e9920`. Kernel-side facts: indexer scores 4-token blocks and keeps the top 512 (`kernels-indexer.md` section 0, kernels `k_ix_scores_*`, `k_dnc_scan*`, `k_dnc_att*`); prefill uses `k_attn_fa2` and decode uses `k_attn_qs*` with block-table paging (`kernels-attention.md` section 0); MoE and trunk linears use the `k_lq8w/lq4w/lq6k/lq8s32`, HT (Hadamard rotation plane) and `k_i4r_*` dequant GEMMs (`kernels-moe.md`, `kernels-dequant.md`). Why inferred: no host launch stub was followed from `018b86d0` or `018d7510` to a named kernel, so "this call launches that kernel" is by name and role, not by traced argument.

## D. Decode

### S16. First token. proven
After prefill, `FUN_0182ea80` calls `decode_first_token` `01848190(engine, ctx, state)` (`D:0182ea80.c:2184`). It logs "req %ld first token (prefill work %.1f ms)" and errors on "the penalty history is larger than this build holds (kPenMax" (`D:01848190.c`, strings in index). The result `iVar74 < 0` is an error; otherwise the token is emitted and the stream either ends (reasons "length" or "stop", :2200-2235) or joins the active list.

### S17. Serial decode tick. proven
In the tick loop, for a stream with no speculation depth (`D:0182ea80.c:8405-8530`): `FUN_018f9dc0` (HIP helper, the step that prepares a one-token forward), `draft_sample` `019098d0` (see `analysis/gf_0x19098d0.c`), `spec_verify_argcheck` `018fa0e0(model, tokens, 2)` (the verify launch), `accept_probs` `01909e50`, then `spec_commit` `018fc360` (:8524). The batched form for several streams is `forward_batch` `01902090`, `joint_verify` `01902a00` and `joint_commit` `01902e70` (calls at `0182ea80.c:7604`, :7759); "pair %ld+%ld: %ld joint rounds, %ld batched ticks" is logged by `trace_retire_request` `01845e50`.

### S18. Speculation round with the draft chain. proven
`spec_round_run` `01851dc0` (`analysis/gf_0x1851dc0.c` and `D:01851dc0.c`). Call order in one round: `018f9dc0(slot, last_tok, 0, bVar)` then `018f9fb0(slot, 0)` then `draft_sample` `019098d0` for depth 1 (:58-67); repeat for depth 2 and further, each step feeding the previous drafted token (:71-90); `spec_verify` `018fa0e0` over the chain (:110); `accept_probs` `01909e50` (:137); `residual_sample` `0190aa50` on a reject (:185); `sample_row` `01909630` for the token after the accepted run (:245); `spec_commit` `018fc360` (:263); emit through `01851580` (:298). The caller chooses this path when the stream's depth `> 1` and a draft head is staged (`D:0182ea80.c:8380-8445`, depth clamp `min(depth, room, accepted_limit)`). The depth limit is `HALOGEN_SPEC_N` (`main_pseudo.c:809` text "needs HALOGEN_SPEC_N >= ...").

### S19. The MTP head. inferred
`load_mtp_drafter_weights` `0188f710` loads `mtp.layers.0.self_attn.*` tensor names (map, med); `mtp_head_path_getenv` `019aa550` reads `HALOGEN_MTP_HEAD` (high). `spec_verify_argcheck` prints "spec_verify(n=%d) needs HALOGEN_MTP=1". INFO `mtp`, `draft_head` and drafter id 1 select it (`wire-protocol.md` 3.1). That `018f9dc0` is the head forward is inferred: its body is a HIP macro stub in the decompile (map: "body not decoded") and it is called once per depth just before `draft_sample`. The drafter ids `serial 0, mtp 1, dflash2 2` are front-end names. The engine refuses id 2 on 0.17.3 ("asked for drafter 2 (unknown)", `runtime-run-0.17.3.md`).

### S20. Prompt-lookup drafting. inferred
Config is proven in `main`: `HALOGEN_PLD=N,K` with `HALOGEN_PLD_ADAPT`, `_LOOKBACK`, `_AGREE` (`main_pseudo.c:778-809`; default pair is read from a constant at `_DAT_00114010`). The D line counters `pld_rounds`, `pld_accepted`, `pld_drafted` exist (`wire-protocol.md` 3.3). The code that finds a matching n-gram in the history and builds the chain was not located. The n-gram table path is `locate_ngram_table` `01894790` and the table loader `ngram_lookup_*` `019041a0`-`01904880`, but those serve the lookup-embedding table, not necessarily PLD. Unproven.

### S21. Verify and accept, greedy. inferred
Greedy requests send no `SAMPLE`; the same `spec_round_run` path runs. Observed: greedy MTP and serial give identical 64-token outputs on two prompts, warm and cold (`runtime-run-0.17.3.md`). The accept kernel `k_smpg_accept_final` is a p/q probability test whose greedy mode value is open (`kernels-speculation.md` section 5). So "greedy commits only argmax-equal tokens" is consistent with the run but not shown in code.

### S22. Verify and accept, sampled. proven
Call chain in S18. `penalties_set` `01907c70` and `target_rows_pen` `01907ff0` apply PENALTY/REPEAT/BIAS rows before sampling. Kernels: `k_accept_prob_f32`, `k_smpg_accept_final`, `k_resid_sample_f32`, `k_sample_f32`, splitmix64 in `k_smpg_pick` and `k_smpg_rpick` (`kernels-speculation.md` sections 0 and 5). Observed: MTP seed 111 twice gives identical tokens, seed 222 differs (`runtime-run-0.17.3.md`, one prompt). Open: where the seed enters the RNG (U23), exact `SampleRow` layout, whether the draw matches serial under the same seed (it did not in the one test).

### S23. Grammar mask. proven
Structured output: `grammar_arena_upload` `0184d3a0`, `check_token_legal_grammar` `0184aab0` (called at `D:0182ea80.c:9078`), `spec_verify_grammar_check` `01851770` (called at :8396 before a speculation round), and the error "flash_spec: token %d is illegal under the grammar (the mask was not applied); ending the request" (`:8626`). `compile_json_schema` `018725a0` and the JSON parser `json_parse_top` `018746e0` build the schema from the base64 text.

## E. Output

### S24. T lines. proven
`FUN_01851580(req, token, ...)` (`emit token`) builds the reply with `format_float_list_reply` `0184b4d0` (digit table, then `snprintf(" %.7g")` for logprobs, `D:0184b4d0.c:242`) and sends it with `send_string` `01845940` (`D:01851580.c:18-21`). It counts tokens per request and logs "req %ld generated %d tokens" (:46). The leading `T` literal was not found in a string, so the exact prefix is inferred from the front end's parser (`serve_api.py:1040-1044`, `t_line_logprob` :503).

### S25. D line. proven
`format_stats_D_line` `0184c9a0` writes `"D %ld %s %d %d %.1f %.1f %d %ld %ld %d %ld ..."` (format string at `D:0184c9a0.c:99`, about 29 numeric fields after the reason). Reasons seen as literals: "length", "stop", "cancel" (`D:0182ea80.c:1161`, :1226), "error" (:3487). Field order and the 30-part line were also observed live (`runtime-run-0.17.3.md`, "Behavior measurements").

### S26. Detokenization in the front end. proven
The engine sends only token ids. The front end decodes with `tok.decode(seq, skip_special_tokens=True)` over runs (`serve_api.py:2327-2338`) and converts tool XML through `ToolStream.push` (`tool_parse.py:285-345`). Stop sequences are matched on decoded text; a match makes the front end send `X`.

### S27. SSE framing. inferred
`StreamingResponse` is imported (:22) and `serve` has a `stream` parameter (:3589). I did not read the generator that writes `data: {...}` chunks, the usage chunk or `[DONE]`. `analysis/api-layer.md` and `wire-protocol.md` do not claim this either.

## F. End of request

### S28. Finish. proven
`finish_request_log` `01851210` and `finish_request_log2` `01851580` choose "length" (max_tokens reached) or "stop" (eos matched) (`D:01851210.c:68-72`, `D:01851580.c:71-74`), log "req %ld generated %d tokens, %.1f s" and call the D-line writer. `FUN_018466e0` (`slot_state_release`, low) is called after each terminal path (`D:0182ea80.c:1170`, :2235, :3419, :9140). That it frees the slot and region is inferred from the name and call sites.

### S29. Snapshot store at the end. inferred
Entries are stored during prefill (S13). Whether a final entry covering the generated tokens is stored at retirement was not seen: `0182ea80.c` has no direct call to `0186a470`, and `cache_lookup_wrapper` `0186f7f0` (low) is the only other store caller found. The next turn of a chat resumes from SNAP/SNAP2/SNAP3 entries, not from the previous reply, in that case.

### S30. Cancel (X). proven
Front end: writes `X <req>` once, then drains up to `ABORT_DRAIN_S = 10` s for the `D` (`serve_api.py:1014-1021`, :1108-1115, :1234-1258; `wire-protocol.md` section 4). Engine: case length 1 reads the id (`D:0182ea80.c:986-1000`), clears a pending slot marker, and for a matching active request calls `trace_retire_request` `01845e50(..., "cancel", ...)` (:1010-1015) or the D writer with reason "cancel" (:1161, :1226). Cancel lines that name a queued but not admitted request are handled in the loop at :1020-1060 (not read in full). Cancel is read when the main loop polls the socket, which the entrypoint says happens "between decode rounds, between the layers of a prefill, and while the lookup table is being read" (`entrypoint-flow.md` section 4, vendor note at `entrypoint.sh:856-857`).

### S31. Cancel during prefill. proven
`prefill_cancelled` `01847a30` (high) logs "req %ld cancelled during its prefill (%d of %d" and is consulted after every prefill slice (`D:0182ea80.c:2127`). `HALOGEN_PREFILL_CANCEL=0` makes a lone prefill ignore the cancel until the prompt is in (`main_pseudo.c:1022-1027`).

## Tally

31 steps. Proven 22: S1, S2, S4, S5, S6, S7, S8, S9, S12, S13, S14, S16, S17, S18, S22, S23, S24, S25, S26, S28, S30, S31. Inferred 9: S3, S10, S11, S15, S19, S20, S21, S27, S29.

## Gaps

Ranked by how much a rebuild would need them.

1. Prompt-lookup drafting (S20). No code found that matches history n-grams and builds a draft chain. Next: callers of the PLD constants in `main` (`local_14d8`, `local_14d0` flow into the engine struct) and the `pld` counters written into the D line by `0184c9a0`.
2. Prefix cache key and hit rule (S11). Entry key (`key_compare` `01868c70`), match length, alignment (`cache_align`), disk record format are unread. Needed to reproduce warm-answer equality.
3. KV pool allocator (S10). Region sizing, the move and eviction policy, the clamp that produces `room_from`. Messages are known; bodies of `01900510`, `01900aa0`, `0184de40`, `01901ee0` are not.
4. Kernel launch binding (S15). Host stubs from `018b86d0`, `018ab5c0`, `018d7510`, `0189abe0` to named `k_*` kernels, and which `lq*`/HT/`i4r` variant a given tensor picks.
5. MTP head identity (S19). Decompile of `018f9dc0` and `018f9fb0` is a HIP macro stub; the head's layer compute is not shown.
6. Greedy acceptance (S21) and sampler seed path (S22, U23). Mode value in `SampleRow`, where `seed` becomes the splitmix64 state.
7. End-of-request snapshot (S29) and slot release (S28). Whether a reply-covering entry is stored; what `slot_state_release` frees.
8. SSE framing (S27) in `serve_api.py`.
9. Cancel of a queued, not yet admitted request (S30) was read only to the point of the "cancel" call.
10. Evidence limit: one live run, two prompts, 64 tokens, no sampler-order test. The log line "generated 2 tokens, 0.0 s" per request is unexplained (`runtime-run-0.17.3.md`, Unexplained).
