# Rebuild guide: a compatible halogen-flash-server 0.17.3

Index plus synthesis for an engineer who wants to build a server that talks to the same clients and reads the same files. Each area states what the component does, then points to the writeup and the evidence. This file adds no new analysis. When a source doc has been corrected by a later one, the later one wins and is cited.

## 1. Scope and how to read this

Target: `halogen-flash-server` 0.17.3, engine `binaries/flash_serve` (gfx1151, HIP), tools `binaries/halogen-tools`, NPU engine `binaries/halogen-npu`, vendor Python front end in `image/`. Nothing here comes from running the vendor engine except one live run (section 12); everything else is static reading of decompiles, disassembly and shipped scripts.

Labels (copied from the source docs, never upgraded):
- **proven**: the cited line, decompile text, table or byte was read.
- **inferred**: reasoned from names, sizes, call order or structure.
- **observed**: seen in the live run (`analysis/runtime-run-0.17.3.md`).
- **partial**: the source doc splits the claim; the unproven half is carried forward in section 11.

Evidence conventions:
- `serve_api.py:N` is `image/serve_api.py` line N. `entrypoint.sh:N` is `image/entrypoint.sh`.
- `D:<addr>.c:N` is `work/decomp/flash_serve/<addr>.c` line N (Ghidra export). `main_pseudo.c` is `analysis/main_pseudo.c`.
- Ghidra address = ELF/file/objdump address + `0x100000`. Example: `main` is `0x182ba00` in Ghidra and `0x172ba00` in the file (CLAUDE.md "Gotchas"; `closure/kernel-binding.md` Method step 2).
- `S<n>` is a step in `analysis/request-trace.md`. `U<nn>` is a row in `analysis/unknowns.md`.
- Function maps: `analysis/function-map.tsv` (flash_serve), `-halogen-tools.tsv`, `-halogen-npu.tsv`, `-kernel-stubs.tsv`. Row counts, each `tail -n +2 FILE | wc -l`: 3568, 3222, 1412, 1738. Role counts for flash_serve, from `tail -n +2 function-map.tsv | cut -f2 | sort | uniq -c | sort -rn`: gpu-dispatch 1680, cxx-stl 674, hip-runtime 266, crt-runtime 248, config-env 216, util 83, kv-cache-prefix 73, protocol-net 61, sampling-speculation 60, gguf-repack 58, model-forward 56, hgn-loader 38, scheduler-slots 24, logging-diag 16, unknown 15. Map labels are mostly heuristic; 642 rows are low confidence (`function-map.md`). `parse_request_json` (`0182ea80`) is a misnomer: it is the engine request handler and tick loop (`request-trace.md` header).

Read order for a rebuild: sections 2 and 3 (bytes in and out), 4 and 5 (what happens), 7 and 8 (the parts that decide output and speed), then 11 (what you cannot rely on).

## 2. Formats

### 2.1 `.hgn` container (all little-endian)
Source: `analysis/hgn-format.md`; reference parser `work/hgn/hgnpy.py` (its docstring, lines 1-30, matches the spec).

Layout [proven by 169 crafted files through vendor `verify`, hgn-format.md s8]:
- Header 104 B at offset 0: magic `HGN1`; u32 version (1 and 2 accepted, others "unsupported version"); u64 n_tensors; u64 table_off (must be 104); u64 data_off (64 B aligned, at or after table_off + 160*n); u64 file_bytes (must equal real size); 64 B model_id (UTF-8, NUL padded, up to 63 chars). hgn-format.md s1.
- Tensor table at 104: n entries of 160 B: name[96] (NUL-terminated, up to 95 chars), u32 dtype, u32 rank (up to 4), u64 dims[4], u64 payload_off (absolute), u64 nbytes, u32 xor32, u32 qparam.
- Payloads from data_off, each start 64 B aligned, table order, `payload_off[i+1] >= payload_off[i] + nbytes[i]`.
- Pad regions are zero by builder convention. Vendor `verify` does not check them (`pad-*.hgn` probes PASS); `inspect` reports `pads_zero`. Write zeros.
- xor32 is the XOR of the payload read as LE u32 words, tail zero-padded. Byte xor, sum u32le, sum u32be and xor u32be all FAIL (hgn-format.md s4).
- `header_sha256 = sha256(file[0:data_off])` [proven]; per report.md s2 it feeds the disk prompt-cache lineage (not re-checked in the cache closure reports; cache-host.md lists engine, checkpoint, overlay, n-gram table, head, vision and tuning identities in the fingerprint).

dtype codes [proven from the `inspect` family table and verify messages; 9, 17-19 and 25+ unobserved; 25 is rejected as a version]: 0 bf16, 1 f32, 2 f16, 3 i32, 4 i64, 5 q4c, 6 fp8r, 7 q8g64, 8 i4l, 10 fp8g, 11 q8s32, 12 q6k, 13 iq4nl, 14 q4a, 15 q5a, 16 ht, 20 q8_0, 21 q5_0, 22 q5_1, 23 i4r, 24 q6g64 (hgn-format.md s2; same table as `hgnpy.py` DTYPE_NAMES).

Per-dtype payload size and shape rules are in hgn-format.md s3, measured by crafting `nbytes=0` files and reading "the format implies X B". Examples for dims `[N,K]`: bf16 and f16 N*K*2; q6g64 `[128,256]` is 26,624 B with qparam 64; i4r needs rank 3 with gate_up `[512,1280,2560]` and down `[512,2560,640]`, qparam 128; HT is allowed only on trunk linears and routed experts. The engine binder `0190e330` uses: q8g64 `numel*17/16`, q8s32 `N*pad16(17K/16)`, q6k `N*K/256*210`, q6g64 `N*pad16(13K/16)` (`closure/dequant-host.md`, `0190e330.c:207-423`) [proven]. The q8g64 rule disagrees with hgn-format.md s3 (see the contradiction note in section 11).

Rank-1 tensors are geometry-checked only when model_id matches a known checkpoint (hgn-format.md s3). Codebook and scale errors (non-ascending codebook, non-finite scale) are in s4.

HT payload (dtype 16) [proven on real vendor data, `closure/ht-decode.md`, decoder `scripts/ht_decode.py`]: bits = `(qparam >> 1) & 0x7f` (3 or 4). The [N, K] tensor is 16x16 tiles of 8*bits u32 words, tile id `(row//128)*(8*K//16) + (col//16)*8 + (row%128)//16`. A tile's words in reverse order are one little-endian bit string U; weight j = 128h + 8r + c (tile row r, column 8h+c) reads the 16-bit window at bit `bits*(255-j) mod 256*bits`, wrapping. code = (byte sum of `(window * 0x83dcd12d) mod 2^32` - 510) / 147.8. Weight = `diag(svh) * Hout * codes * Hin * diag(suh)` with H the orthonormal 128-point Sylvester Hadamard per block; `suh` is +/-1 signs, `svh` a signed per-row scale. down_proj experts are expert-major, one [N, K] slice each with its own suh/svh row; gate_up experts share one `mlp.experts.su` and walk tiles per 128-row block, 8 gate tile rows then 8 up tile rows per column tile (`decode_codes_gate_up`, Pearson 0.9922 measured). qparam is 0x1200 | bits << 1; the loader rejects anything else (decomp 01996d90.c:236-238), so 0x1200 is a fixed tag. gate_up experts use their own tile walk (`decode_codes_gate_up`, 0.9922 measured).

i4r payload (dtype 23, v2 checkpoint, routed experts only) [proven on real vendor data, `closure/ht-decode.md` "i4r", `decode_codes_i4r`]: qparam 128. The [E, N, K] tensor holds all experts' nibbles (row-major, low nibble first), then all experts' fp16 scales, one per 128 weights along K (4.125 bits per weight). Wq = (nibble - 8) * scale, then the HT reconstruct (both-side 128-point Hadamard, svh rows, su columns; svh is +/-1 here). Measured: 0.9876 (gate_up) and 0.9885 (down_proj) against q4c.

### 2.2 Head file, n-gram table, GGUF input
- The MTP head is its own `.hgn`, `qwen38-flash-next-mtp.hgn`, named by `HALOGEN_MTP_HEAD`. It must carry `mtp.*` tensors; a llama.cpp MTP GGUF is refused (hgn-format.md s6; entrypoint.sh:544-565). Real table read from the shipped head (version 2, 31 tensors, data_off 0x1400, 1,523,566,720 B): `closure/mtp-head.md` s1 [proven, first 5064 B of the file]. Tensors: `mtp.fc_embedding`, `mtp.fc_hidden`, two `pre_fc_norm`, three hyper-connection modules, one `layers.0` (q8g64 attention, q4c experts 512x1280x2560 and 512x2560x640, bf16 router 512x2560, q8g64 shared expert). No embedding, lm_head or final norm in the head [proven absent].
- `HREF` is NOT an in-.hgn record. It is the magic of the `halogen-tools ppl` reference dump (`--ref-out`, `--ref`): 0x20 B header (`HREF`, u32 version 2, u32 K, u32 vocab, u64 positions, u64 sequence length), then N records of `12*K + 16` bytes (`closure/hgn-href.md` s2; writer `halogen-tools 01825fb0.c:7606-7701`) [proven for layout; field meanings inferred]. hgn-format.md s6 body text still describes HREF the old way; its top banner carries the correction.
- The n-gram table is a second ordinary `HGN1` file named by `HALOGEN_NGRAM_TABLE` (default `qwen38-flash-next-ngram.hgn` beside the checkpoint). Lookup is by tensor name `layers.1.ple.ngram_embedding.weight`, main file first, then the table file (`closure/hgn-href.md` s4; `01894790.c:7-21`) [proven]. The table payload dtype is not established.
- GGUF input: `flash_serve --ck X.gguf` repacks into RAM at start; `flash_serve --repack IN --out OUT --with-table --head H` writes a file (`entrypoint-flow.md` s1 "GGUF to .hgn repack"; entrypoint.sh:1501; hgn-format.md s7; driver `019af350`). It needs the n-gram table tensors or aborts "Nothing was written". Output starts at 0x80, 64 B alignment, parallel fill on `HALOGEN_GGUF_THREADS` (default 8, clamp 1-64). F16 is accepted into bf16 only when every value is bf16-clean.

### 2.3 NPU containers (only if you rebuild section 9)
`devices.hnpm` (text), `consts.hnpb` and `<id>.hnpw` (HNPB v1 binary). Corrected header: 64 B, magic `0x42504e48`, u32 version 1, u32 flavor at +0x08, u64 n_entries +0x10, u64 table_off +0x18, u64 0x1000 at +0x20, u64 file size +0x28; blob record 0xa8 B, data offset 4096 aligned, xor32 at +0xa0 (`closure/npu-closure.md` U01 "Correction", `00167960.c:350-355`, `0016eb50.c:1148-1155`) [proven]. `npu-engine.md` s2.2 carries the old offsets; use the closure report.

### 2.4 Config
Engine config is argv (7 flags) plus environment, not a file. See section 4. The image bakes `/opt/halogen/flash-tune.plan` (hipBLASLt tuning plan per CU count); the engine may rewrite it at exit (entrypoint.sh:442). Its format was not analysed.

## 3. Wire protocol (front end to engine)

Source: `analysis/wire-protocol.md`, written from `image/serve_api.py`, so it describes what this client sends and parses; the engine may accept more. TCP, text lines ending `\n`, whitespace-split fields, blobs base64 on one line. Default `127.0.0.1:8730`.

Handshake [proven from client code, wire-protocol.md s1]: open one stream; `INFO\n`, up to 3 attempts, 5 s each; accept a line starting `I` with at least 6 fields; if INFO says `grammar=1`, send `VOCAB`. A second short connection sends `PING\n` and expects `PONG` (used by `/health` and the healthcheck, hc:8-11).

Engine side [proven, S6, `main_pseudo.c:1070-1135`]: `main` binds and loops `accept`; each connection is served to completion in `FUN_0182ea80` before the next accept. Observed on 0.17.3 (runs 2 and 3, `runtime-run-0.17.3-r2.md`, `runtime-run-0.17.3-r3.md`): a waiting connection's `CSTAT`, `GEN` and every line after them wait while another connection is served, and are answered in order when that connection closes, not when its `GEN` ends; a third connection is released when the second closes [U26 CLOSED]. Leading `INFO\n` and `PING\n` lines on a waiting connection are answered inline by `FUN_01844d50`, which peeks 5 bytes of each queued fd between commands and queues the fd on main's pending vector (`closure/serve-pending-fd-r4.md`) [proven].

Verbs the client sends: `PING`, `INFO`, `VOCAB`, `VIMG`, `GEN`, `X`, `CSTAT`.
- `VOCAB <n> <base64 blob>`: blob is `HGV1`, LE u32 n, then per id a `<BH` record (flag, u16 length) plus bytes. Flag 0 is an ordinary token (GPT-2 byte-level decode), flag 1 a special token (UTF-8), missing id is flag 1 length 0 (wire-protocol.md s2, `serve_api.py:479-501`). Reply `V ok`.
- `VIMG <req> <first_tok> <H> <W> <base64 rgb>`, written before the `GEN` that uses it. H and W are multiples of 32, minimum 256x256, maximum `HALOGEN_VISION_MAX_PIXELS` (default 2560x1440). Each image takes `(H/32)*(W/32)` pad tokens. No reply is parsed.
- `GEN <req> <max_tokens> <n_eos> <eos...> <n_ids> <ids...> [drafter] [SAMPLE t k p minp seed] [LOGPROBS] [TOPLP n] [PENALTY pp fp] [REPEAT r] [BIAS n id bias...] [SNAP n] [SNAP2 n] [SNAP3 n] [IMG k ...] [SCHEMA json|b64] [AFTER tok] [ESCAPE n ids] [THINK budget end n ids] [GUARD a b c d e] [SEG n pos...]` (wire-protocol.md s2, `serve_api.py:1074-1097`). Drafter ids: serial 0, mtp 1, dflash2 2. Greedy is the absence of `SAMPLE`. `SAMPLE` uses `%.9g %d %.9g %.9g %d`: temperature, top_k (0 off), top_p (0 off), min_p, seed.
- `X <req>`: cancel, no direct reply; the engine ends the request with a normal `D` line (inferred; the client only waits). The client drains for up to 10 s.
- `CSTAT`: reply is a `C` line of up to 52 fields gated by line length (wire-protocol.md s3.5).

Engine to client:
- `I <f1..f27>`: field positions in wire-protocol.md s3.1 (mtp, draft_head, ctx, spec_rows, default drafter, drafter_weights, dflash2, cache_mb, cache_align, kv_slots, slot_ctx, cache_mode, sampling, rope_factor, kv_pool, vision, version, pld_n, pld_k, checkpoint_format, grammar, composable, cc_floor, cc_seam, cc_bytes, cc_fingerprint, indexer_budget). Field names are the client's, not the engine's. A live example is in section 12.
- `T <req> <token> [<logprob> [<n> (<id> <logprob>)*n]]`. The engine formats floats with `" %.7g"` (S24, `0184b4d0.c:242`); the leading `T` literal was not found in a string (inferred from the client parser).
- `D <req> <reason> <n_prompt> <n_gen> <prefill_ms> <decode_ms> ...`: 30 whitespace parts on a normal done line, 12 on the error line [observed]. Positions 2-29 are in wire-protocol.md s3.3. Engine formatter `0184c9a0`, format string at `D:0184c9a0.c:99` (S25) [proven]. Reasons seen as literals: length, stop, cancel, error. An `error` line carries four numeric fields then text (`error 0 0 0 0 asked for drafter 2 (unknown)`, observed).
- The client drops unknown line forms silently (`serve_api.py:840-841`).

Engine-side parse (S7, S8) [proven]: the first word is switched on its length; `GEN` is queued as a raw string and parsed later with `>>` into req id, max_tokens, eos list, prompt ids, then keyed fields. Rejection literals include "SAMPLE temperature outside (0, 2] or top_k < 0", "PENALTY/REPEAT/BIAS need SAMPLE with temperature > 0", "SCHEMA with SAMPLE temperature > 0 is not supported yet: structured output decodes greedy", "LOGPROBS with SCHEMA is not supported" (`D:0182ea80.c:2482-2550`).

Left open by wire-protocol.md s6 (some since closed, see sections 7 and 8): the full set of `reason` values; meaning of `spec_rows`, `shared`, `dv_*`, `kept_*`, `think_*`; error shapes for malformed verbs; the JSON schema dialect.

HTTP side (outside the engine): chat completions and sibling routes, the Qwen3-Coder XML tool-call parser (`api-layer.md` s1, `image/tool_parse.py`), the opt-in request recorder (`request_records.py`, api-layer.md s2), the `halogen_tools.py` wrapper (s3). SSE chunk framing was not read (S27, inferred).

## 4. Startup

Source: `analysis/entrypoint-flow.md`. The script proves the supervisor and preflight, not engine internals.

Default mode `all` [proven from the script, lines 1370-1429]:
1. Ports 8730 (engine) and 8731 (API), bind `127.0.0.1`. Slots 4, ctx 262144, KV pool positions `min(2*ctx, 1048576)` and not below ctx, max-tok 32768 (lines 31-41). ctx above 262144 caps max-tok at 16384 and exits 2 unless `HALOGEN_ROPE_YARN` is set (lines 42-62).
2. `need_ckpt`: `host_preflight` needs `/dev/kfd` and `gfx_target_version == 110501` (gfx1151); optional `maybe_download`; the checkpoint must exist; `check_gguf` or `check_sidecar`; n-gram table, vision, KV budget and GTT checks (lines 215-630; GTT can refuse with exit 1).
3. `need_tokenizer` (needs `tokenizer.json`), `serve_api.py --check-defaults`, optional NPU engine on port 8740 (waits for `READY`).
4. Engine launch, then wait until its port accepts (poll every 2 s), then API launch (`--host 0.0.0.0`), then the watchdog.
5. Any component exit tears everything down and the script exits 1, even after a clean engine exit (line 1429).

Engine command line [proven, lines 904-906, 1379-1381]: `flash_serve --ck <ckpt> --port <p> --bind <addr> --slots <n> --ctx <n> --max-tok <n> --kv-pool <n>`. Everything else comes from the environment. 22 front-end-only flags are removed with `env -u` (list in entrypoint-flow.md s2). Exported for the engine: `HALOGEN_CHECKPOINT`, `HALOGEN_CACHE_FILE`, `HALOGEN_MATMUL_TUNING_FILE`, `HALOGEN_NGRAM_TABLE`, `HALOGEN_MTP_HEAD`, `HALOGEN_VISION_TOWER` (the reader being flash_serve is inferred). Full flag table (68 names, 3 retired): entrypoint-flow.md s3.

Engine-read flags found in decompiles: `HALOGEN_PROMPT_CACHE` 0, 1 or 2 (`01903680.c:8-14`; script default 2); `HALOGEN_CACHE_INPLACE`, `_FULL`, `_SNAP3`, `_CKPT`, `_EVICT`, `_BRANCHES`, `_ENTRIES`, `_FILE`, `_DIR`, `_DISK_GIB` (default 64), `_PRUNE_OLD` (`closure/cache-host.md`); `HALOGEN_PLD`, `_ADAPT`, `_LOOKBACK`, `_AGREE` (`main_pseudo.c:753-772`); `HALOGEN_MTP`, `HALOGEN_SPEC_N`, `HALOGEN_SAMPLED_DEPTH`, `HALOGEN_SAMPLED_GATE`, `HALOGEN_S2_*`, `HALOGEN_S3_FALSIFY`, `HALOGEN_WALK_FALSIFY`, `HALOGEN_SAMPLE_BLOCKS` (default 0x40) (`closure/sampling-host*.md`); `HALOGEN_PREFILL_CHUNK`, `HALOGEN_ADMIT_CHUNK`, `HALOGEN_ADMIT_TICKS`, `HALOGEN_PREFILL_CANCEL`; kernel selectors `HALOGEN_LIN_WAVE`, `HALOGEN_LQ4W_SH`, `HALOGEN_HT_LIN_DIRECT`, `HALOGEN_HT_DEC_GROUP`, `HALOGEN_FA_OPT`, `_FA_LPT`, `_FA_S21`, `_FA_TQ`, `HALOGEN_DN_S12`, `HALOGEN_INDEXER_BUDGET`, `HALOGEN_ATTN_QS_BT2_OPT`, `HALOGEN_FLASH_PIN_TRUNK`. The complete set `flash_serve` reads is still open (entrypoint-flow.md s5; `analysis/fals_env.txt`).

Model load as logged by the live run [observed, `analysis/evidence/engine-0.17.3-run.log` lines 25-60]: GGUF repack into a 66.92 GiB arena (35.4 s), 1198 tensors (1166 trunk, 1 table, 31 head), weights pinned, "preparing weights, layer N of 48", KV pool of 524288 positions (about 14.4 GiB), "slots: 1 -> 4 POOLED, 112.0 MiB each of O(1) state", prompt cache banner (24 entries, chunk 32768, resume-anywhere), disk tier, memory summary. The n-gram lookup table is read from disk through the file cache, not held in RAM.

Supervision [proven, entrypoint-flow.md s4]: the healthcheck is `PING` then `PONG` on the engine port (`read -t $HALOGEN_ENGINE_PING_S`, default 30), or `GET /health` on the API. The watchdog probes every 15 s with a silence limit of 180 s (`HALOGEN_ENGINE_WATCHDOG_S`). It defers when any engine thread is in D state or `compact_stall` in `/proc/vmstat` rises, capped by `HALOGEN_ENGINE_WATCHDOG_DEFER_S` (900 s). The script says the engine answers `PING` between decode rounds, between prefill layers and while the lookup table is read (lines 856-857); the engine code path for that was not located.

## 5. Request path

Source: `analysis/request-trace.md`, 31 steps. Tally: 22 proven, 6 proven in part (S10, S11, S15, S19, S21, S22), 3 inferred (S3, S27, S29).

A. Front end (`serve_api.py`)
- S1 route `POST /v1/chat/completions` at :3946 [proven]. S2 `normalize_messages` (`tool_parse.py:350-473`), `apply_chat_template`, tokenize with offsets [proven]. S3 SNAP positions from `render_prompt` (:3463-3505): SNAP is the end of the stable prefix, SNAP2 the end of the system turn, SNAP3 the start of the last user turn [inferred in the trace; arithmetic not re-read]. S4 think budget, guard, schema and escape fields (:3589-3611) [proven]. S5 `Engine.generate` writes the `GEN` line (:1074-1097) [proven].

B. Engine intake
- S6 accept loop, one connection at a time [proven]. S7 verb switch on first-word length [proven]. S8 `GEN` parse and rejection strings [proven]. S9 admission: log "req %ld admitted: slot %d, region base %d cap %d ..."; no room means "waits"; `max_tokens` clamped to the room left [proven]. S10 slot and KV pool assignment [proven in part, section 7].

C. Prefix cache and prefill
- S11 cache lookup and restore [proven in part, section 7]. S12 prefill in chunks: `get_prefill_chunk` `01905000` reads `HALOGEN_PREFILL_CHUNK` (32768 per the banner); while streams are active, prompts are admitted in `HALOGEN_ADMIT_CHUNK` pieces with `HALOGEN_ADMIT_TICKS` decode ticks between [proven]. S13 snapshots stored during prefill when the position equals SNAP, SNAP2 or SNAP3 (`snapshot_store` `0186a470`) [call proven; field-to-SNAP match inferred]. S14 per-layer dispatch [proven, section 6].

D. Decode
- S16 first token, `decode_first_token` `01848190` [proven]. S17 serial tick: head draft `019098d0`, verify `018fa0e0`, `accept_probs` `01909e50`, commit `018fc360`; batched form `forward_batch`, `joint_verify`, `joint_commit` [proven]. S18 speculation round `spec_round_run` `01851dc0` [proven]. S19 MTP head [proven in part, section 8]. S20 PLD [proven, `closure/pld.md`]. S21 greedy accept and S22 sampled accept [proven in part]. S23 grammar mask: `check_token_legal_grammar` `0184aab0`, `compile_json_schema` `018725a0` [proven].

E. Output
- S24 `T` lines via `01851580` and `format_float_list_reply` [proven except the `T` literal]. S25 `D` line via `0184c9a0` [proven]. S26 detokenization in the front end, `tok.decode(..., skip_special_tokens=True)`; stop sequences are matched on decoded text, then `X` is sent [proven]. S27 SSE chunk framing [inferred, not read].

F. End
- S28 finish reasons length or stop, then `slot_state_release` `018466e0` (what it frees is inferred). S29 whether a reply-covering snapshot is stored at retirement [inferred; no direct call to `0186a470` in `0182ea80.c`]. S30 cancel [proven; the queued-not-admitted case was read only to the "cancel" call]. S31 cancel during prefill: `prefill_cancelled` `01847a30`; `HALOGEN_PREFILL_CANCEL=0` makes a lone prefill ignore it [proven].

## 6. Kernels and dispatch

### 6.1 Binary facts
- Fatbin: clang offload bundle v2 at file offset 0x52000, 23,739,040 B, four AMDGCN code objects (`analysis/fatbin-inventory.md`): `codeobj_53000` (2982 symbols), `1146000` (350), `14aa000` (34), `1513000` (124). Disassembly is in `analysis/kernels-disasm/` (299 MB; read by grep windows only).
- Host registration: 1,738 `__hipRegisterFunction` calls in four functions (`01968680` 1,484; `019e0890` 175; `019f3590` 62; `019e52c0` 17), resolved through `.rela.dyn` to 1,738 stub entries, strictly 1:1 with 1,738 kernel names [proven, `closure/kernel-binding.md`; table `function-map-kernel-stubs.tsv`]. Family count 141, from `tail -n +2 function-map-kernel-stubs.tsv | cut -f4 | sort -u | wc -l`; the grouping is a name heuristic [inferred].
- Nothing selects a stub by name at run time. Each template instantiation is its own stub and host code picks by calling it. Host launches use `leaq <handle>` then `hipLaunchKernel`, so grepping for stub addresses finds nothing and Ghidra cuts decompiles at `__hipPushCallConfiguration` (`closure/indexer-attn-host-r2.md` "Method note"; `closure/dequant-host-r2.md`). To find a launch, search the disassembly for the handle (`objdump -R binaries/flash_serve | grep RELATIVE`).
- GPU math: wave32, `v_wmma_f32_16x16x16_bf16/f16` for tiles, `v_dot2_f32_bf16` and `v_dot2acc_f32_f16` for GEMV. No `v_mfma` [proven, kernels-attention.md s0, kernels-moe.md s8].

### 6.2 Per-layer flow [S14, proven, `D:018f2060.c:114-240`]
48 layers (loop bound 0x2f). Layers with `layer % 4 == 3` hold KV (12 layers); the other 36 hold DeltaNet state of 0x300000 B plus conv state of 0x14000 B each (`closure/cache-host.md` point 3, `018fe4f0.c`; 36 x (3 MiB + 80 KiB) = 110.8 MiB, matching the "111 MiB each" banner) [proven]. Per layer, in order: `prefill_forward` `0189abe0(...,0)`; then, if byte 0 of the layer record (stride 0xc68) is 1, `sparse_attn_block_select` `018b86d0`, else `gdn_layer_forward` `018ab5c0`; output stage `018a5150`, `018ab0a0`; `0189abe0(...,1)`; `moe_layer_forward` `018d7510`.

### 6.3 Kernel families: what each computes and when it runs
| family | what it computes | runs when | writeup |
|---|---|---|---|
| `k_attn_fa2<...>` (+ `k_attn_fa`, `k_attn`) | FlashAttention-2 style prefill attention over KV, WMMA | sparse-attention layers, prefill; launchers `019e9920` (sparse, FaSel) and `019eb380` (dense), called from `018b86d0` at `17bbcfa`; `k_attn_fa` only if `HALOGEN_FA_TQ=0` | kernels-attention.md s1; closure/indexer-attn-host-r2.md U14 |
| `k_attn_qs*`, `k_attn_fd*`, `k_attn_sel` | query-stationary decode with block-table paging; flash-decode split-K plus merge; sparse block selection | sparse-attention layers, decode | kernels-attention.md s2-4 |
| `k_ix_scores_*` | score each 4-token block and keep the top 512 (`HALOGEN_INDEXER_BUDGET`, default pair 2048 tokens / 512 blocks) | indexer layers; host `ix_select` `018b4750` | kernels-indexer.md s0-1; closure/indexer-attn-host-r2.md U14 |
| `k_dnc_*` (scan, scan3, prep, ut, att) | DeltaNet chunk kernels over 64-token chunks, NOT the indexer | linear-attention layers; driver `dn_chunk_forward` `019d7b70`, att launcher `019dd130` selected by `HALOGEN_DN_S12` | closure/indexer-attn-host.md "Finding that reframes U12" [inferred naming, strongly supported]; kernels-indexer.md has a correction banner |
| `k_kv_bt`, `k_kv_scatter_rows`, `k_rope*` | write K/V into the pool as bf16 with RoPE applied at write time | KV layers | kernels-attention.md s5-6 |
| `k_lq4w/lq6k/lq6w/lq8s32/lq8w/lbfw` (+`_grp`, `_pair`) | quantized or bf16 GEMV for 1 to 8 tokens | decode linears | kernels-dequant.md s1; closure/dequant-host-r2.md |
| `k_deq_*` | dequantize to bf16, then bf16 GEMM | prefill, more than 8 tokens [pairing inferred] | closure/dequant-host-r2.md |
| `k_ht_*`, `k_i4r_*` | Hadamard-rotated 3 or 4 bit weights; the rotation is a 128-point FWHT scaled by `0x3db504f3` = 1/sqrt(128) (corrected from 1/sqrt(512)) | HT trunk linears and routed experts; i4r experts | kernels-dequant.md s3-4; kernels-moe.md s1 |
| `q4moe::k_q4moe*` | 4-bit routed-expert GEMM, WMMA | MoE layers with plain 4-bit experts | kernels-dequant.md s2; kernels-moe.md s4 |
| `k_exp_gu*`, `k_route_logits*`, `k_dn_fused16` | gate/up with SiLU epilogue, router, fused down-projection | MoE layers | kernels-moe.md s5-7 |
| `k_hc*` | hyper-connection mix kernels (4 x 2560 residual) | trunk; use in the MTP head is inferred | kernels-dequant.md s0; mtp-head.md s3 |
| `k_sample_f32`, `k_resid_sample_f32`, `k_accept_prob_f32`, `k_smpg_*`, `k_pen_*` | sampling, residual sampling, accept test, penalties | sampled decode and speculation | kernels-speculation.md; section 8 |
| `k_ngram_*` | integer hash and modulo producing n-gram embedding row ids | lookup-embedding path; NOT used by PLD | kernels-speculation.md s3; closure/ngram-dv-host.md U17 |
| `k_dv_quant`, `k_dv_quant8`, `k_dv_scatter` | draft-vocab (shortlist) quantization and scatter | draft-vocab setup, once | closure/ngram-dv-host.md; closure/moe-dv-host-r2.md U18 |

### 6.4 Dtype to kernel table (decode, 1 to 8 tokens) [proven for decode and small batch except rows marked inferred, `closure/dequant-host-r2.md`]
The binder `upload_weight_tensor` `0190e330` fills one slot of a 128 B descriptor by dtype; the non-null slot is the dispatch key (`closure/dequant-host.md`): +0x00 q4c planar (qparam 2), +0x08 bf16 and fallback, +0x10 q8g64, +0x18 q8s32, +0x20 q6k, +0x28 q6g64, +0x30 ht (pinned memory only), +0x48 HT bit width = `(qparam >> 1) & 0x7f`. Launch helpers: bf16 `k_lbfw` (`01927700`); q4c planar `k_lq4w` (`018f6d80`, K-aware `01918180`); q8g64 `k_lq8w` (`0191cd30`) [pairing inferred from kernel name]; q8s32 `k_lq8s32` (`018f5c80`, `01920310`); q6k `k_lq6k` (`018e6f20`, `01921ed0`); q6g64 `k_lq6w` (`0191e8f0`) [pairing inferred from kernel name]; ht `k_ht_lin` (`01928f40`, `0192c1d0`) and `k_ht_lin_grp` (`018b0e40`); i4r experts `k_i4r_rows*` from `018d2430`; plain 4-bit experts `k_q4moe` from `018d2430` and `018d7510`. q4a and q5a go through `k_deq_aff<4|5>` then `k_lbfw` [pairing inferred]. f32, f16, fp8r, fp8g, i4l, iq4nl, q5_0, q5_1, q8_0 and interleaved q4c have no dedicated kernel in the stub list and take the bf16 fallback [round-1 reading holds; converter not identified]. Gates: `HALOGEN_LIN_WAVE` (default 1), token count 1 to 8, `(K & 0x1f) == 0` for the K-aware variants. Template meanings: `k_lq4w<M,R,W,bool,E,F>` is M tokens, R lanes per row, W waves per row (R and W inferred from rows per block), bool K-aware, E epilogue id (call-site constants known, meaning open), F = `HALOGEN_LQ4W_SH`.

The prefill choice between dequant-then-GEMM and a fused kernel, and the case-to-stub map of `01920310`, are not extracted (kernel-binding.md "Runtime selection"; section 11).

## 7. Prefix cache and KV pool

Sources: `closure/cache-key-kvpool.md` (S10, S11), `closure/cache-host.md` (U13), `kernels-attention.md` s8.

Prefix cache [proven unless marked]:
- Entry stride 0x160; token vector, side vector at +0x18 (meaning inferred as modality ids), length at +0x48, region base at +0x50, flags +0x88 full, +0x89 tail checkpoint, +0x8a system-prompt anchor, +0x8b shared anchor (`D:01861160.c:80`, `D:0186cdc0.c:23-36`).
- Hit rule `01861160`: scan all entries. A candidate needs `0 < len <= prompt_len`, `len != prompt_len` unless it is a full entry, exact int32 token equality over `len`, side-vector equality, and to be longer than the best so far (ties keep the lowest index). The match length is the entry's `len`; there is no partial match inside an entry (`D:01861160.c:26-72`). The link to the D-line `n_cached` is inferred; 19 was observed on a 19-token prompt.
- `key_compare` `01868c70` is NOT the prefix key. It is the composable chunk-cache lookup (64-bit id, length, `bcmp` of tokens).
- On a hit the engine restores the O(1) DeltaNet state and keeps the KV rows (in place) or copies them region to region, then prefills only tokens from the hit length (`0187fb00`, `01860720.c:51-53`; `018680b0` calls `01905ae0(model, tokens + done*4, n)`). A full-prompt hit makes no forward call. U13 is CLOSED for "no prefill redo on cached positions" (`closure/cache-host.md`).
- Snapshot alignment: a boundary inside a chunk is rounded down to a multiple of 64; a remainder below 64 means no snapshot (`D:018680b0.c:98-101`). That INFO `cache_align` is this value is inferred; the live INFO shows 64 (observed).
- Tail checkpoints at `end - 0x1000`, `end - 0x400`, `end - 0x4000`, when at least 0x2000 past the resume point and `HALOGEN_CACHE_CKPT` is on (`01860720.c:~98-125`, line numbers approximate).
- Modes: `HALOGEN_PROMPT_CACHE` 1 stores on chunk boundaries ("bitwise resume"); 2 stores at the prompt end plus tail checkpoints ("resume-anywhere ... NOT bitwise"). The 1/2 to banner mapping is inferred. A warm answer can differ from a cold one because the resumed prefill starts a chunk at the hit position, which changes the grouping (inferred; size not measured).
- Capacity: entries = `slots * ((branches << SNAP3) + FULL + 1)`; 24 with 4 slots (branches=2 inferred). Rank-mode eviction: full entry or tail checkpoint rank 0, older leaf 0, newest leaf 2, branch point 3, anchor 1 (source formula `3b+1`), shared anchor 4; lowest rank first, ties to the smaller clock. `HALOGEN_CACHE_EVICT=0` is pure LRU. The clock unit and stamper were not found (`D:0186b270.c:29-76`).
- Disk tier [partial] (`HALOGEN_CACHE_DIR`; needs `HALOGEN_CACHE_INPLACE=1` and no `HALOGEN_CACHE_FILE`): a fingerprint text "halogen prompt cache disk tier v1" with hashed engine, checkpoint, overlay, n-gram table, head, vision and tuning-file identities; tree `DIR/<16 hex>/` with a `fingerprint` file; each record is a manifest (magic 0x43443233, version 1, written to a temp name, fsync, rename) plus a `.blob` (O_DIRECT, 64 MiB pieces); budget `HALOGEN_CACHE_DISK_GIB` default 64. Record file-name pattern and manifest field meanings are not decoded. `HALOGEN_CACHE_PRUNE_OLD=1` deletes other configurations' subtrees whose name is 16 hex chars and that hold a `fingerprint` file (CLOSED; the live run did this, section 12).
- Composable context (`SEG`): units are extracted while prefilling and placed at a later offset; the engine itself says "NOT bitwise: about one quantization step". It needs `HALOGEN_PROMPT_CACHE=2`. The success body of the chunk restore is unread.
- Kernel evidence for resume: cached K/V are written once as bf16 with RoPE applied at write time and read back as raw bits into the MMA operands; no kernel in the attention family recomputes them (kernels-attention.md s8, SUPPORTED at kernel scope).

KV pool [proven unless marked]:
- One pool of `P` positions (`engine+0x48`, 524288 in the run). A sorted vector of 12 B records `{base, cap, owner}` at `engine+0x50`; `owner == -2` is a held region (cache rows, no running request). Region ends are rounded up to 256 positions.
- Region size = `round_up_256(min(P, max_tokens + prompt_len))`, capped by P (`D:0182ea80.c:5649-5662`). Admit-time clamp `max_tokens = min(max_tokens, P - prompt_len)` (:5304-5308).
- Placement: try the hit's region if held; else first fit over the sorted table; else wait ("kv pool: request %ld waits for %d positions ...").
- If the hit region is too small: grow in place (`0184e2f0`), move (`0184de40`; `kv_move_region` `01901ee0` copies via `01901c50`; gaps must be whole 4-block multiples), compact held regions, forget longer entries, forget the LRU held region, forget chunks, else wait. Each step exists [proven]; the order is inferred (one 9,500 line function of gotos).
- Short-room clamp: a turn may run in the room left only when `room >= max(max_tokens*15/100, 1024)` (doubled when a second budget is on); it logs "the turn runs in the %d positions it has left (max_tokens %d -> %d)" (`D:0182ea80.c:6000-6030`). The store of the old value into the D-line `room_from` was not found [link inferred].
- `slots_alloc` `01900510` and `slots_alloc_pooled` `01900aa0` size per-slot device buffers, not KV regions; pooled mode sets `model+0xcd = 1`.
- Open: the new-region insert, producers of INFO/CSTAT `pool_used`, `relocated`, `cold_resorts`, ladder order, the default branches dword (closure/cache-key-kvpool.md s3).

## 8. Speculation

### 8.1 Drafters and wiring
Drafter ids: serial 0, mtp 1, dflash2 2. dflash2 is a reserved slot: INFO field 7 is hard-wired 0 in every engine from 0.11.10 to 0.17.3 (dflash2-probe.md s4) [proven]. A raw `GEN ... 2` is refused with `error 0 0 0 0 asked for drafter 2 (unknown)` (U04 CLOSED, observed). PLD runs beside the MTP head and reports through INFO `pld_n`, `pld_k` and the D-line counters, not the drafter slot.

### 8.2 MTP head (`closure/mtp-head.md`)
- Shape [proven from the head file table]: one decoder layer, hidden 2560, 512 routed experts of width 640 plus a shared expert with a scalar gate, residual stream 10240 (4 x 2560), three hyper-connection modules (rank-320 mixers), an indexer; attention with 2 KV heads x 256, q_proj 12288. Inferred: q_proj includes a per-head output gate; the input path is RMSNorm(embedding) then `fc_embedding` and RMSNorm(4-stream hidden) then the front mixer then `fc_hidden`, combined; embedding and lm_head are the trunk's.
- Loader `load_mtp_drafter_weights` `0188f710` requires Q4C-P experts (`:684`) and sets the head-staged flag `model+0xb7c`. `HALOGEN_MTP` defaults on; a value starting with 0 skips it.
- Per draft step [host chain proven, `gf_0x1851dc0.c:58-67`]: `018f9dc0` copies 0x5000 B (10240 x 2 B) from the trunk's saved hidden `model+0x50` to the head input `model+0x930` (depth 0), or feeds the head's own last output row (depth above 0); `018f34c0` stages the last token id on the device; `018f9fb0` keeps the head's full-vocab fp32 logit row (248,320 x 4 B = 0xf2800 B) as q row k (k up to 7); `draft_sample` `019098d0` samples from it. The head layer kernels and the logits launch are NOT read (the decompile ends at a no-return `__hipPushCallConfiguration`).
- Chain length is capped by the runtime `HALOGEN_SPEC_N`; drafting stops when `exp(logp)` falls below the gate thresholds (`+0x144` for step 1, `+0x148` after). "Up to 3" in the banner has no host constant.
- Verify reuses the trunk: `spec_verify` `018fa0e0` runs the trunk's n-token forward (n up to `HALOGEN_SPEC_N`); `spec_commit` `018fc360` copies the accepted hidden row `(keep-1)*0x5000 + model+0x930` back to `model+0x50` for the next round.
- Shortlist vocab (`HALOGEN_DRAFT_VOCAB`, `_FILE`): modes parsed in `018f51e0` (unset or 1 is mode 1, 0 off, cycle 2, check 4, a named list 3); selected in `018f46b0`; it narrows the head's lm_head GEMM to the listed ids. The list file format is NOT established (a 65,536 line, one-id-per-line text file exists on the analysis host; the fit is inferred). `k_dv_quant8` runs once at setup: 32 bf16 in, 32 int8 plus an f16 scale (`absmax/127`) out, rows of 2560. `k_dv_quant` (4-bit, the default unless `HALOGEN_DRAFT_VOCAB_Q8=1`) uses 15 thresholds that equal NF4 midpoints; the 16 NF4 levels are read from rodata 0x46b30 and copied to the table head (U18 CLOSED, `closure/dv-quant-r4.md`).

### 8.3 PLD, prompt lookup (`closure/pld.md`) [proven unless marked]
- It is inline host code in `FUN_0182ea80`; there is no PLD string, kernel or function. `k_ngram_*` and `locate_ngram_table` are not involved.
- Config `HALOGEN_PLD=N,K` (defaults N=3, K=3; window W=32, floor 1.0, backoff base R=64 from `HALOGEN_PLD_ADAPT=W,floor,R`; `main_pseudo.c:744-772`). PLD is forced off when the MTP head is off, and K must be below `HALOGEN_SPEC_N`.
- Per request: a token history vector plus a hash map from the 64-bit hash of an N-gram to the position of the token that followed it (the most recent occurrence wins). Hash: start `0x14650fb0739d0383`, per token `h = (h ^ tok) * 0x100000001b3; h ^= h >> 29` (`D:01870b60.c`). The prompt is bulk-indexed at request start (`018708b0`); every committed token is pushed (`01870b60`).
- Gate (`D:0182ea80.c:8322-8324`): PLD is tried when N >= 1, the sampled-path pointer `pauVar8[0x19]` is null and the backoff counter is zero. That null means "greedy request" is inferred. Each tick is either a PLD verify or a head round, not both.
- Chain: `history[pos .. pos+len)` for the most recent earlier occurrence of the current last-N tokens, `len = min(K, verify width - 1, KV room, tokens left)`. A hit equal to the current size is rejected. No token compare follows the hash hit (collision effect inferred). A grammar cuts the chain at the first illegal token (shape only).
- Verify: one `spec_verify` over `[last_token, chain...]`; accept while the argmax row equals the chain token; commit accepted + 1. Greedy PLD output equals plain greedy by construction [byte equality untested].
- Agree gate (default on, `HALOGEN_PLD_AGREE`): the head's pending draft must equal the first chain token, else the tick falls to the head path.
- Backoff: after W rounds, a mean accepted per round below the floor turns PLD off for `R << min(streak,6)` committed tokens (64 up to 4096).
- Counters `pld_rounds`, `pld_accepted`, `pld_drafted` map to stats longs 8, 9, 0xc; the D-line relations are in wire-protocol.md s3.3. The per-round increment constant `_DAT_001142f0` is presumed 1.

### 8.4 N-gram embedding kernels (not drafting)
`k_ngram_ids` produces 16 ids per token (128 B row). Lanes 0-7 hash a 2-token window and lanes 8-15 a 3-token window: `h = xor_k B[k]*tok[row+2-k]`, `id = (h mod vocab_size[lane]) + offset[lane]`, with out-of-segment tokens replaced by a boundary id (`closure/ngram-dv-host.md` U17, CLOSED at kernel level; the head count per order is inferred). The host gathers table rows (`HALOGEN_NGRAM_GATHER_THREADS`).

### 8.5 Verify, accept and sampler
- Accept test, sampled path [host code proven]: ratio = `p / exp(q_logp)` capped at 1.0, compared with a uniform `u` of 24 bits (top 24 bits of a splitmix64 output times 2^-24); `ratio <= u` rejects and calls `residual_sample` `0190aa50` (`gf_0x1851dc0.c:136-155`, `sampling-host-r2.md`). The uniform is drawn on the host; `k_smpg_accept_final` has no RNG.
- splitmix64 in `k_smpg_pick` [kernel text read]: `x = (A * 0xd1b54a32d192ed03) ^ seed ^ (B * 0x9e3779b97f4a7c15)`, `x += 0x9e3779b97f4a7c15`, `z = x ^ (x>>30); z *= 0xbf58476d1ce4e5b9; z ^= z>>27; z *= 0x94d049bb133111eb`, `u = (z >> 40) * 2^-24`. A is the position, B a stream tag (draft passes 1, residual 3). The host mirror used for accept is `((pos+base) * 0xd1b54a32d192ed03 ^ seed ^ 0x3c6ef372fe94f82a) + 0x9e3779b97f4a7c15` with the same finalizer (sampling-host.md U23; constants hand-matched from Ghidra's signed prints). U23 is PARTIAL: the copy of the parsed seed into the request object at `slot[0x32]+0x10` is not located; the link rests on layout match.
- `SampleRow` is 32 B [proven, sampling-host.md U15 as corrected by sampling-host-r2.md]: +0x00 float temperature (must be above 0), +0x04 int top_k, +0x08 float top_p, +0x0c float min_p (clamped to [0,1]), +0x10 u64 seed, +0x18 u32 position, +0x1c u32 stream tag. The engine parses SAMPLE as temperature, top_k, top_p, min_p, then the seed as an unsigned 64-bit value, stored unmodified.
- Filter semantics [kernel structure read; order inferred]: temperature scales logits (1/T inside every exp); then top_k when `0 < top_k < vocab_width`, else top_p when `0 < top_p < 1` (top_k wins if both); then the weight `w = exp((l - lmax)/T)` must satisfy `min_p <= w`; then the draw. Tie handling at the radix threshold and whether top_p mass includes the threshold token are open.
- Greedy never reaches the sampling kernels: `sample_rows`, `accept_probs` and `residual_sample` exit on "temperature 0 is the GREEDY path and must stay on argmax()". The `mode` argument of `k_smpg_accept_final` is `HALOGEN_S3_FALSIFY` (default 0); mode 3 forces acceptance of the fall-through row (U16 CLOSED). The head's greedy draft step and the non-PLD greedy accept compare are not located.
- Penalties: `penalties_set` `01907c70`, `target_rows_pen` `01907ff0`, kernels `k_pen_*`; `PENALTY`, `REPEAT` and `BIAS` require `SAMPLE`. Penalty and bias math was not analysed.
- Test-only flags: `HALOGEN_S2_NUCLEUS/QOFF/TSKEW` make drafting deliberately wrong ("FALSIFIER ACTIVE (METHOD 37)"); `HALOGEN_WALK_FALSIFY` (U07, U20: device side unchanged); `HALOGEN_SAMPLED_DEPTH` (1, or below `SPEC_N`) and `HALOGEN_SAMPLED_GATE=t1[,t2]` with values in [0,1].
- Grammar: JSON schema compiled by `compile_json_schema` `018725a0` from base64 text, token mask per step (`check_token_legal_grammar`); decodes greedy only. The accepted schema dialect is not characterized.

## 9. NPU path (halogen-npu)

Sources: `analysis/npu-engine.md`, `closure/npu-closure.md`, `closure/npu-closure-r2.md`. A separate 1.1 MB binary on loopback port 8740; the small models (decision, embedding, rerank, moderation, generate, image) run on AMD XDNA through XRT. Function map: 1412 rows; roles from `tail -n +2 function-map-halogen-npu.tsv | cut -f2 | sort | uniq -c`: cxx-stl 779, crt-runtime 243, model-compute 124, util 111, xrt-device 75, hnp-format 28, npu-protocol 25.
- Subcommands: `selftest`, `replay`, `serve` (stdin and stdout), `listen DEVICES_DIR WEIGHTS HOST:PORT` (TCP), `probe`, `convert` (npu-engine.md s5).
- Network: exactly one `getaddrinfo` call site, in `listen` (`133090`); only `127.0.0.0/8` and `::1` are accepted, else "a loopback HOST:PORT only" [proven, npu-engine.md s3].
- XRT flow [proven]: `xrt::device`, `xrt::elf` plus `xrt::hw_context` (`00157940`), `xrt::ext::kernel` named `"main:" + graph name` (`00159980`), run create, set args, start, poll, wait (`00157b70`); BO flags 0x20000000.
- A `READY` banner prints after load; one thread per connection with TCP_NODELAY; line cap and queue backpressure strings such as "ERR 0 busy: ..." (npu-engine.md s4).
- Verb grammars (r2; STORE and HIT in `closure/npu-store-hit-r4.md`, U02 CLOSED): token 0 is `VERB` or `VERB@model`, token 1 the tag. `GEN max_new temp top_k top_p min_p penalty seed K stop1..stopK M id1..idM` replies `OK tag n_prompt n_gen reason t1 t2 t3 t4 text bucket HASH_A,HASH_B,MISMATCH ids...`. Temperature <= 0 is argmax; the seed feeds a `std::mt19937_64` (recurrence constant `0x5851f42d4c957f2d`); the penalty is subtractive for ids in history; prompt plus max_new must fit a bucket of 16 positions. `DEC/EMB/SCO/CLS`: `n m`, then n groups `a c i1..ic`, then n*m ints; DEC replies `OK tag ...` (not `DEC`). Also `ELOAD tag path int`, `EFREE tag int`, `CANCEL`, `QUIT`, `INFO`, `PFX`, `LEDGER`. Errors are `ERR <tag> <text>`.
- Closed since r2: numeric fields of GEN, DEC, EMB, SCO, CLS and the one handler for QHN, QLED, LOOKUP, EDUMP, ELOAD, EFREE (`closure/npu-verbs-r3.md`, `npu-verbs-r3b.md`); STORE, HIT and the IMG side keys (`closure/npu-store-hit-r4.md`); the HNPB header +0x08 pack-kind tag (U01, `closure/hnpb-header-r4.md`); the per-blob tag (4 = f32, 1 = bf16) and the `main:<graph>` names (U03, `closure/npu-shipped-files.md`). Open: how many entry ids a STORE reply lists and which entries a HIT reply lists (U02).

## 10. Tools (halogen-tools)

26 MB ELF. Function map: 3222 rows; roles from `tail -n +2 function-map-halogen-tools.tsv | cut -f2 | sort | uniq -c | sort -rn`: gpu-dispatch 1718, cxx-stl 566, crt-runtime 289, hip-runtime 213, config-env 201, gguf-repack 52, hgn-loader 45, verify-inspect 5, ppl-niah 7. It embeds copies of engine code (for example `spec_verify_argcheck` `018d4f90` has the "same body as flash_serve 018fa0e0").
- `verify FILE.hgn [--threads N]` (main `01847710`): PASS line `PASS %s: %zu tensors, %.2f GiB, v%llu, model_id %s; header, table, layout, payload sizes, checksums, codebooks and scales OK; geometry checked on %zu tensors (%zu with no rule)`, exit 0; FAIL line `FAIL %s: %zu problem%s; the first: %s`, exit 1. Check order: magic and size, header fields, per-entry, quant codebooks and scales, geometry (hgn-format.md s5) [proven by the 169-file matrix].
- `inspect FILE.hgn [--json] [--no-hash] [--threads N]` (main `018441a0`): JSON with path, model_id, version, n_tensors, file_bytes, families[], total_bytes, aux_bytes, params, header_sha256, header_bytes, pads_zero, tensors[] (name, dtype, qparam, dims, nbytes, xor32, sha256).
- `ppl` (main `01825fb0`, chunked perplexity, optional `--ref-out` and `--ref` HREF dumps, `--per-pos` file of 16 B records: f32 kl, dp, top1, nll) and `niah` (needle test from a manifest, writes `NAME.gen` int32 files). The Python wrapper `image/halogen_tools.py` tokenizes, builds the needles `vault7`, `relay`, `ledger`, and scores (api-layer.md s3). Flags beyond what the wrapper passes are known only from the wrapper and strings.
- GGUF repack: `gguf_repack_main` exists in the tools map (`0198cf70`, role gguf-repack, high). The entrypoint only calls `flash_serve --repack`, sends `inspect|verify` to the tools binary directly and `ppl|niah` through the wrapper (entrypoint-flow.md s1). Do not assume the vendor path uses the tools copy.
- A reference implementation to test against: `work/hgn/hgnpy.py` (build, parse, verify-equivalent round trip).

## 11. Proven versus inferred: what a rebuild can rely on

Unknowns ledger counts from `analysis/unknowns.md` (27 rows). Command: `for s in CLOSED PARTIAL OPEN-BOUNDED UNOBSERVABLE; do echo -n "$s "; grep -c "^| U[0-9]* .*| $s |\$" analysis/unknowns.md; done` printed CLOSED 23, PARTIAL 1, OPEN-BOUNDED 2, UNOBSERVABLE 1; `grep -c '^| U[0-9]' analysis/unknowns.md` printed 27.

Can rely on [proven]:
- The `.hgn` container: header, table, checksums, size rules, dtype codes, `verify` and `inspect` behavior (169 vendor-checked files, round trips).
- Wire grammar of `GEN`, `INFO`, `T`, `D`, `X`, `VOCAB`, `VIMG`, `CSTAT` as the vendor front end sends and parses them, the engine rejection strings, and one-connection-at-a-time serving.
- Request flow S1 to S9, S12 to S14, S16 to S18, S20, S23 to S26, S28, S30, S31.
- Prefix-cache hit rule, entry kinds, eviction ranks, snapshot alignment, KV region table and size rule.
- PLD end to end, the SampleRow layout, the accept test, splitmix64.
- Stub to kernel binding (1,738 of 1,738) and the decode dtype to kernel table.
- The MTP head file layout and the host plumbing around it.

Cannot rely on (PARTIAL, OPEN-BOUNDED, UNOBSERVABLE rows; what is missing):
- U01 is CLOSED (moved out of this list): header +0x08 is a pack-kind tag, copied without a check and with no later reader found; vendor weight packs carry 1, vendor device-constants packs 2, the in-binary bulk dump writes 3 (`closure/hnpb-header-r4.md`, `evidence/hnpb-headers-vendor.txt`).
- U02 is CLOSED (moved out of this list): STORE replies list B new entry ids, HIT replies list only new entries and only with f1 set; STORE runs on kind g buckets with xd 3 (the vendor qwen3.5-2b u0..u2), HIT needs kind r, which no shipped manifest has (`closure/npu-store-hit-r4.md`, `npu-store-hit-r5.md`).
- U05 UNOBSERVABLE: a future flip of the dflash2 format literal cannot be seen from the nine pulled releases.
- U06 OPEN-BOUNDED: 0.14.x and 0.15.x patch deltas; those tags were not pulled.
- U07 is CLOSED (moved out of this list): `HALOGEN_WALK_FALSIFY` is a host-only test mode that perturbs the RNG counter and residual-sample inputs, never the accept formula or a kernel argument (`closure/walk-falsify-r3.md`).
- U09 is CLOSED (moved out of this list): HtItem = {expert, tile, begin, end}, 16 B, written on the GPU by `k_ht_items_group` (`closure/htitem-r3.md`).
- U10 is CLOSED (moved out of this list): `k_ht_tg` and `k_ht_tg_sum` are launched by slot in FUN_01928680, by default on the n >= 9 prefill path; tg_sum is split-K sum plus inverse Hadamard plus scales to bf16 (`closure/ht-tg-r3.md`).
- U11 is CLOSED (moved out of this list): the probe variants are dead code; matmul tensors in GGUF Q5_0/Q5_1/Q8_0 are repacked to q5a and q8s32; dtypes 20/21/22 exist only for the n-gram table (`closure/dnftaps-r4.md`, `closure/q5-consumer-r5.md`).
- U12 PARTIAL: ut4/ut5, scan3 A/E/F, case-to-stub maps, Tap and NormFold layouts and the `dnc` name (`closure/dnc-scan3-r3.md`); Tap writers (recurrent-state checkpoints), the n == 1 decode path (k_dn_conv_ab_rows then k_dn_recur_step) and NormFold's inline rsqrt(sum(x^2)/128) (`closure/dnc-scan3-r4.md`) are closed. Open: labels for scan3 B, C, D, G (compile-time constants), NormFold n1/n2 buffer identity, Tap payload size.
- U14 is CLOSED (moved out of this list): FaSel +0x08 ids are ascending 4-token block indices, +0x10 is the per-row id count (`closure/fasel-r3.md`, `closure/fasel-r4.md`).
- U18 is CLOSED (moved out of this list): the draft-vocab table feeds the draft-vocab lm_head GEMV then `k_dv_scatter`; 4-bit NF4 is the default, 8-bit needs `HALOGEN_DRAFT_VOCAB_Q8=1` (`closure/dv-quant-r4.md`). Not decoded: nibble order inside a 16-byte block.
- U19 is CLOSED (moved out of this list): `scripts/ht_decode.py` decodes real 3 and 4 bit HT weights, Pearson 0.98 to 0.99 against q4c references (`closure/ht-decode.md`). qparam is a fixed tag 0x1200 plus bits << 1 (loader check). gate_up experts use their own tile walk (`decode_codes_gate_up`, 0.9922 measured).
- U21 is CLOSED (moved out of this list): token-count to kernel tables for HT, i4r and plain experts, k_q4moe arguments, the hipBLASLt library arm (`closure/expert-dispatch-r4.md`, `-r5.md`, `-r5b.md`).
- U22 OPEN-BOUNDED: Ghidra has no AMDGCN module, so device code stays at disassembly level; rebuilding kernels means writing from instruction listings.
- U23 is CLOSED (moved out of this list): the wire seed reaches SampleRow+0x10 unchanged; the splitmix64 counter is the token's sequence position (`closure/sampler-seed-r3.md`).
- U26 is CLOSED (moved out of this list): one connection is served at a time; a waiting connection gets only exact `INFO\n`/`PING\n` first lines answered inline (FUN_01844d50), everything else waits and is released in order when the served connection closes, not when its GEN ends (`runtime-run-0.17.3-r3.md`, `closure/serve-pending-fd-r4.md`).
- U27 is CLOSED (moved out of this list): i4r (dtype 23) decodes as planar signed nibbles times one fp16 scale per 128, then the HT rotation (`decode_codes_i4r`, 0.9876 and 0.9885 measured, `closure/ht-decode.md`).
- CLOSED rows (U01, U02, U03, U04, U07, U08, U09, U10, U11, U13, U14, U15, U16, U17, U18, U19, U20, U21, U23, U24, U25, U26, U27) can mean kernel level only (U17, U24, U25) or one half (U13, U15); read the row text.

Other gaps a rebuild hits (request-trace.md "Gaps" and the closure reports):
- MTP head: fc_embedding/fc_hidden fusion, router top-k and the greedy draft step are CLOSED (`closure/mtp-head-r5.md`); the head step's ordered kernels (39 launches at decode width 1, 78 over the prompt rows), its Linear kernels and router path are CLOSED from the run 4 launch trace (`closure/mtp-head-r6.md`); each request is prefill, a head step over the prompt, then rounds of [head step, target verify step], then a trailing head step.
- Sampler: top_k keeps every tie of the k-th logit, top_p keeps the crossing token and is measured against the full-vocab mass (top_k and top_p intersect), min_p, penalties, logit_bias and temperature order are CLOSED (`closure/sampler-math-r5.md`); edge cases are untested. Sampled speculation agrees in distribution, not token for token (runs 1 and 2).
- Disk cache record naming and the 32-byte `e<N>.rec` header, the cache clock unit, S29 (no entry covering generated tokens is stored) and SSE framing for all four routes are CLOSED from code (`closure/cache-sse-r5.md`); no on-disk record has been observed live (runs 3 and 4 wrote none).
- Env flags: 256 variables read by flash_serve, one row each in `analysis/env-flags.md`; D-line counters `shared`, `dv_*`, `kept_*`, `think_*` writers are CLOSED (`closure/env-dline-r5.md`).
- Evidence limit: four live runs (runtime-run-0.17.3.md, -r2, -r3, -r4), short prompts, greedy plus a few sampled requests, one engine.

Source disagreements found while writing this guide (later or more specific source wins here):
- q8g64 size: hgn-format.md s3 says `N*K + 2*N`; closure/dequant-host.md says `numel*17/16` (`0190e330.c:207`). The binder rule is read from code; the table row was a formula guess.
- Repack owner: report.md s1 lists repack under `halogen-tools`; entrypoint-flow.md s1 and entrypoint.sh:1501 call `flash_serve --repack`. The tools binary does contain `gguf_repack_main` (`0198cf70`), so both are partly true.
- Acceptance: report.md s2 says "acceptance = trunk-argmax equality"; kernels-speculation.md s1 and sampling-host-r2.md show a probabilistic accept test on the sampled path; equality holds on the PLD greedy path (pld.md s5).
- Registration count: function-map.md says `01968680` lists 1,462 distinct kernel names; kernel-binding.md counts 1,484 calls there (1,738 over four functions).
- HREF: hgn-format.md s6 body versus its own banner and closure/hgn-href.md (HREF is the ppl dump magic).
- `q_keep` guards: version-deltas.md s3 ties them to `k_dv_quant`; moe-dv-host-r2.md says they concern fp32 logit rows kept for `draft_sample`, not dv_quant8 output.
- Dtype pairing: kernels-dequant.md s0 maps q4a and iq4nl to `k_lq4w` or `k_deq_q4cp*`; dequant-host-r2.md maps q4a to `k_deq_aff` and finds no dedicated kernel for iq4nl.
- `cache_mode`: wire-protocol.md s3.1 says 1 is the default; the live INFO shows 2 and entrypoint-flow.md s3 gives `HALOGEN_PROMPT_CACHE` default 2.
- PLD defaults: pld.md s1 gives N=3, K=3; the live INFO shows `pld_n 2`, `pld_k 6` (set by the live service env, not read).
- Lookup table size: entrypoint-flow.md and hgn-format.md s6 say 47.7 GiB; the live engine log says "26.8 GiB lookup table" (different checkpoint).
- Round-trip suite: CLAUDE.md said a 165-file matrix (corrected to 169: `ls work/hgn/files/*.hgn | wc -l` and `tmp/verify_matrix.tsv` both give 169, 9 PASS and 160 FAIL); hgn-format.md s8 says 3/3 round trips and 169 files, and `work/hgn/files` holds 169 `.hgn`.
- NPU header offsets and DEC reply prefix: npu-engine.md versus npu-closure.md and npu-closure-r2.md (the closure reports are the corrected ones).

## 12. Acceptance checks from the live run

Source: `analysis/runtime-run-0.17.3.md`; raw files in `analysis/evidence/`; probes `scripts/engine_probe.py`, `engine_probe2.py`, `engine_probe3.py`. Setup: the extracted `flash_serve` 0.17.3 (sha256 prefix af4f07bb) over the 0.17.2 image userland, mode `engine`, Strix Halo gfx1151, 4 slots, 524,288 position pool. Prompts P (19 tokens) and Q (25 tokens) from the model tokenizer, 64 generated tokens, greedy unless noted. One run, small samples.

Test data and expected outputs [observed]:
1. INFO reply (`probe2-0.17.3.jsonl` line 2; the version field follows `HALOGEN_IMAGE_VERSION`, so it prints 0.17.2): `I 1 0 262144 7 1 1 0 110 64 4 262144 2 1 1 524288 1 0.17.2 2 6 gguf 1 0 0 0 0 - 2048`. Decoded with the section 3 order: mtp 1, draft_head 0, ctx 262144, spec_rows 7, default drafter 1, drafter_weights 1, dflash2 0, cache_mb 110, cache_align 64, kv_slots 4, slot_ctx 262144, cache_mode 2, sampling 1, rope_factor 1, kv_pool 524288, vision 1, version 0.17.2, pld_n 2, pld_k 6, checkpoint_format `gguf`, grammar 1, composable 0, cc fields 0 0 0 `-`, indexer_budget 2048. A rebuild's INFO should parse in the front end and give the same slot and pool numbers.
2. `GEN 900001 64 2 151645 151643 19 <19 ids> 1` (greedy, MTP) streams 64 `T` lines then a 30-part `D` line. In `probe2` the first D line is `D 900001 length 19 64 271.3 1242.7 1 22 63 0 1 4 18 24 ...`: drafter 1, rounds 22, commit 63, n_cached 0 (cold), shared 1, pld_rounds 4, pld_accepted 18, pld_drafted 24 (positions 7-14). Warm repeats show `n_cached` 19 (position 10).
3. Greedy MTP, cold versus warm cache: identical 64 tokens. Greedy MTP versus serial (drafter 0): identical on P and Q. Speculation did not change greedy output.
4. Sampling at temperature 1.2, top_k 0, top_p 1.0, min_p 0: MTP seed 111 twice gives identical tokens; seed 222 differs; MTP seed 111 differs from serial seed 111. Run 2: serial seed 111 twice gives identical tokens (32 of 32), seed 222 differs; MTP seed 111 repeats too. A rebuild that hashes seed and token position the same way (U23) reproduces draws per drafter; serial and MTP draws under one seed differ.
5. A raw `GEN ... 2` is refused: `D <req> error 0 0 0 0 asked for drafter 2 (unknown)`; the engine log says "request refused: asked for drafter 2 (unknown)". The error line has 12 whitespace parts.
6. A waiting connection gets only leading `INFO`/`PING` lines answered; its `CSTAT`, `GEN` and later lines wait until the served connection closes, then are served in order (runs 2 and 3, U26).
7. Startup: GGUF repack 35.6 s, ready about 36 s after container start, 1198 tensors (1166 trunk, 1 table, 31 head), listening on 8730 with 4 slots over one 524,288 position pool.
8. Cache banner to compare: "prompt cache ON, resume-anywhere (24 entries, snapshot every request, chunk 32768, 111 MiB each at ctx 262144): a warm answer is NOT bitwise the cold one".
9. Unexplained: the engine log says `generated 2 tokens, 0.0 s` per request while 64 tokens were delivered.

Static acceptance suites (Linux x86-64 only; they run `halogen-tools verify` and `inspect`): `bash scripts/verify_matrix.sh` expects 9 PASS and 160 designed FAIL over `work/hgn/files/*.hgn` (`ls work/hgn/files | sed 's/.*\.//' | sort | uniq -c` gives 169 hgn and 2 json); `python3 tmp/rt_roundtrip.py` and `work/hgn/roundtrip.py` cover build, verify, parse and byte-identical rebuild.

Version notes (`version-compare-0.17.2-0.17.3.md`, `version-deltas.md`): device code is unchanged between 0.17.2 and 0.17.3, so every delta is host code. 0.17.3 adds `HALOGEN_SAMPLED_DEPTH`, `HALOGEN_SAMPLED_GATE`, `HALOGEN_WALK_FALSIFY`, `spec_round_run` (3,103 B) and `spec_verify_grammar_check`. Function-level match over 3568 functions: 2353 unchanged, 171 changed, 138 changed-shape, 906 added, 902 removed (heuristic; added and removed are upper bounds).

Safety note for anyone repeating the run: give the test container its own `/cache` and set `HALOGEN_CACHE_PRUNE_OLD=0`. The 0.17.3 run deleted a 60.25 GiB on-disk prompt cache because the shared env file set the flag to 1 (runtime-run-0.17.3.md "Incident").
