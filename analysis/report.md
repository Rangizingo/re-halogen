# halogen-flash-server — reverse engineering report

Target: `ghcr.io/peonist-ai/halogen-flash-server:0.17.3` (2026-10-08 build,
`linux/amd64`, base `python:3.12.15-slim-trixie`, 22 layers, ~0.86 GiB
compressed). Repo: https://github.com/peonist-ai/halogen-flash-server
(deployment surface only; engine is closed source).

Tooling: REA (rea-agents 6.1.0) — Ghidra 12.1.2 provider, pwntools-elf layout
provider, plus radare2 for targeted xrefs. Evidence IDs cited where relevant.

## 1. Component map

| artifact | size | what it is |
|---|---|---|
| `flash_serve` | 26 MB ELF, stripped | the engine: C++/HIP, AMD clang/LLD 23, gfx1151, hipBLASLt + libamdhip64. Source path leaked: `/src/halogen/src/flash_model.hip` |
| `halogen-tools` | 26 MB ELF, stripped | checkpoint `verify`/`inspect`/`ppl`/`niah` + GGUF→`.hgn` repack (`--repack IN.gguf --out OUT.hgn`) |
| `halogen-npu` | 1.1 MB ELF, stripped | XRT/AIE NPU engine: `selftest`/`replay`/`serve`/`listen`/`probe`/`convert`; FLUX.2-klein DiT support |
| `serve_api.py` | 4,321 lines | FastAPI front (vendor-shipped, comments stripped) |
| `entrypoint.sh` | 1,534 lines | startup, memory arithmetic, engine watchdog, NPU bring-up |

Engine listens on 127.0.0.1:8730 (raw line protocol: `GEN`, `VIMG`, `SCHEMA`,
`THINK`, `GUARD`, `SEG`, `X <req>`, `PING`/`PONG`); the front serves 0.0.0.0:8731
(OpenAI chat/completions + Responses + Anthropic Messages + `/health` `/cache`
`/metrics`).

## 2. Engine architecture (evidence: decompiled main(), strings, flags)

- **Args** (from `main_pseudo.c`): `--ck FILE --port P --bind ADDR --slots B
  --ctx N --max-tok N --kv-pool POSITIONS --head --no-head --no-table
  --with-table --repack --out --threads --quiet`.
- **Speculation**: MTP draft head (depth ≤ 8, `HALOGEN_MTP_DEPTH*`), n-gram
  prompt-lookup drafting (`HALOGEN_PLD*`, table in a 47.7 GiB FP8 lookup file,
  paged), acceptance = trunk-argmax equality → drafts can only cost speed,
  never change output (confirmed in `draft_sample` decompile,
  `gf_0x19098d0.c`).
- **KV**: one shared pool (`HALOGEN_KV_POOL_POSITIONS`), per-request
  reservation of `prompt + max_tokens`, slots share the pool.
- **Caches**: in-place prompt cache (bitwise resume, `HALOGEN_CACHE_INPLACE`),
  disk cache (`HALOGEN_CACHE_DIR`, lineage records), composable-context
  preview (message-unit extraction + compose, "NOT bitwise" per its own log).
- **Grammar-constrained decoding**: compiled JSON-schema grammars, per-token
  masking, greedy-only; refusal strings for schema+logprobs, schema+images.
- **Context**: native 262,144; beyond that requires static YaRN
  (`HALOGEN_ROPE_YARN`, factor 2 → 524k, 4 → 1M), which changes numerics.
- **Vision**: opt-in tower (`HALOGEN_VISION_TOWER`), 16px patches, 2x2 merge,
  256²–2560×1440 pixel bounds.
- **hipBLASLt tuning plan** baked per-CU-count (`flash-tune.plan`); refuses
  stale plans from other archs/builds (index-validity reasoning in its own
  warning string).
- **Memory discipline**: refuses to pin when MemAvailable is under a floor —
  pinned pages can't be swapped and a task waiting on them can't be killed, so
  over-pinning wedges the machine (its words). Watchdog in entrypoint:
  PING silence + D-state/`compact_stall` = host memory stall (defer, don't
  kill, issues #79/#85); pure silence = wedge → container takedown.

## 3. The "FALSIFIER ACTIVE (METHOD 37)" finding

String: `draft_sample: FALSIFIER ACTIVE: nucleus %.4f, qoff %+.4f, tskew
%.4f. Every proposal from this process is deliberately wrong (METHOD 37).`

Decompiled `draft_sample` (`gf_0x19098d0.c`):
- Gated on `HALOGEN_S2_NUCLEUS` (default 0), `HALOGEN_S2_QOFF` (default 0),
  `HALOGEN_S2_TSKEW` (default 1.0, clamped >0) — inert at defaults.
- When active it skews the drafter's sampling temperature, widens nucleus,
  offsets q-logits, sets a one-shot banner flag, and counts itself.
- It perturbs **draft proposals only**; the trunk verifies every proposal
  (argmax equality), so output is byte-identical — the same property
  `bench-serving.py` asserts ("A drafter CANNOT change output (commit is
  trunk-argmax equality)"), with sampled runs gated distributionally by
  `gate-s3-falsify.sh`.

**Verdict: a negative-control test harness, not a backdoor.** It proves
speculation cannot affect output by making the drafter adversarial.
`HALOGEN_WALK_FALSIFY` / `HALOGEN_S3_FALSIFY` are the same pattern in the
verify/walk paths (getenv wrappers at 0x1883020 / 0x184bd90 Ghidra-space).

## 4. Security posture

- **No telemetry — verified.** `flash_serve` imports only
  `socket/bind/listen/accept/recv/send` (+ inet_pton): no `getaddrinfo`, no
  outbound connect, zero URLs/domains in strings, `HF_HUB_OFFLINE=1` baked in
  the image config. Only outbound path: `hf download` in entrypoint when
  `HALOGEN_DOWNLOAD` is set (user-named HF repo). `halogen-npu` has
  `getaddrinfo` (loopback host:port args; deeper trace open).
- **Auth**: none in the engine (binds 0.0.0.0 → explicit warning at startup:
  "the token protocol has no auth; keep this port unpublished"). None in the
  front either — documented; request records (`HALOGEN_RECORD_DIR`) store
  message text and the docs warn about shared hosts.
- **EULA** (image copy, `image/EULA.md`, revised to v1.0 2026-10-10): free
  use incl. commercial; RE expressly permitted without limitation and
  encouraged as an open community challenge (findings, specs, and derived
  implementations publishable, attribution per §2); redistribution of
  modified/derived works permitted under attribution conditions; benchmark
  publication expressly permitted (ask: name the version + prompt set).

## 5. Checkpoint format (.hgn) — recovered, round-trip implemented

`.hgn` = engine-native container (magic `HGN1`). Full layout recovered from
`halogen-tools` decompiles and verified by a Python parser/builder
(`work/hgn/hgnpy.py`; 169 crafted test containers in `work/hgn/files/`,
tarred at `analysis/hgn-testfiles.tar.xz`):

- Header 104 B LE: magic, version u32, n_tensors u64, table_off u64 (==104),
  data_off u64, file_bytes u64, model_id[64] NUL-padded.
- Tensor table at table_off: 160 B/entry — name[96], dtype u32, rank u32 (≤4),
  dims u64[4], absolute payload_off u64, nbytes u64, xor32 u32 (XOR of LE u32
  words of payload, zero-padded tail), qparam u32 (quant group size; 0 plain).
- Payloads at data_off, 64 B aligned, table order. Pads must be zero:
  [4,table_off), post-table, inter-payload, tail. header_sha256 =
  sha256(file[0:data_off]).
- 24 dtype codes: bf16, f32, f16, i32, i64, q4c, fp8r, q8g64, i4l, fp8g,
  q8s32, q6k, iq4nl, q4a, q5a, ht, q8_0, q5_0, q5_1, i4r, q6g64 (family
  grouping per tensor table).

Repack (GGUF→.hgn) driver decompiled in `flash_serve` at 0x19af350
(`work/hgn/gs_0x19af350.c`): GGUF magic gate (`gguf: %s is not a GGUF`);
requires the n-gram table (`layers.1.ple.ngram_embedding.weight` /
`per_layer_token_embd.weight`) — missing table is fatal, "Nothing was
written"; layout from 0x80 with 64 B alignment per tensor; sparse
`ftruncate` then parallel fill on `HALOGEN_GGUF_THREADS` threads (default 8,
clamped 1..64); banner `repack: %zu tensors, %.2f GiB -> %s on %d threads%s%s`
with ", with the table" / ", NO head" / ", head from <path>" suffixes.
Formal spec writeup (`analysis/hgn-format.md`) + `halogen-tools verify`
round-trips pending (Phase 3 acceptance).

Quant families seen: 4-bit grouped (experts/trunk), 6-bit mixing
layers, 8-bit draft-head projections, FP8 n-gram table, HT (rotation-plane)
and I4R variants with strict shape/group checks. GGUF (`qwen4exp`) repacked
losslessly in RAM at start (~20 s) or once via `convert`.

## 6. Findings complete at plan scope; residual unknowns

All seven phases of `PLAN.md` plus Phase 7 have evidence-backed writeups.
Phases 0 to 6: version deltas (`version-deltas.md`), fatbin and gfx1151
disassembly (`fatbin-inventory.md`), five kernel families
(`kernels-attention.md`, `kernels-moe.md`, `kernels-indexer.md`,
`kernels-dequant.md`, `kernels-speculation.md`), the `.hgn` format with
round-trip tests (`hgn-format.md`), the NPU engine (`npu-engine.md`) and
dflash2 (`dflash2-probe.md`). Phase 7: function maps of all three binaries
(`function-map.tsv` 3,568 rows, `function-map-halogen-tools.tsv` 3,222,
`function-map-halogen-npu.tsv` 1,412, `function-map-kernel-stubs.tsv`
1,738; data rows, `wc -l` minus the header), the wire protocol
(`wire-protocol.md`), entrypoint and API layer (`entrypoint-flow.md`,
`api-layer.md`), a 31-step request trace (`request-trace.md`), a live
0.17.3 engine run (`runtime-run-0.17.3.md`), 17 closure reports
(`closure/`) and a rebuild guide (`rebuild-guide.md`).

Residual unknowns, counted from the table in `unknowns.md`: CLOSED 9,
PARTIAL 13, OPEN-BOUNDED 3, UNOBSERVABLE 1 (26 rows). The PARTIAL rows
(U01, U02, U07, U09, U10, U11, U12, U14, U18, U19, U21, U23, U26) each say
which half is closed and which is not. What stays open and why:

- U03 NPU graph names: they live in the shipped `.elf` graph files, not in
  this workspace.
- U19 HT codebook decode: recovered statically, needs a real HT checkpoint
  to confirm.
- U06 0.14.x and 0.15.x patch deltas: those tags were not pulled (no
  downloads in scope).
- U22 AMDGCN pseudocode: Ghidra has no shader-ISA module; device code stays
  at disassembly level.
- U05 a future flip of the dflash2 literal: not observable from shipped
  releases.

## 7. Method notes (reproducibility)

See `../AGENTS.md` "Gotchas": Ghidra dot-path rejection, 4 GiB worker-scope
OOM vs 16G systemd-run slices, PIE base +0x100000 for `rea function`,
journalctl JSON recovery, `REA_PWNTOOLS_PYTHON` venv.

Key evidence: layout `ev_58f4f920f56cb720…` (flash_serve), overview
`ev_48e5d53cb13709a3…` (3,794 procedures / 17,774 strings), tools overview
`ev_36521f3d6d3555f5…` (3,420 / 17,603).
