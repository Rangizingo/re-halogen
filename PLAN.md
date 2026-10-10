# PLAN.md — Finish reverse engineering halogen (autonomous run)

Standing authorization: Peter approved this plan for **autonomous execution**
(2026-10-09): "so I can have you do all of this autonomously without me next
session." Within this plan only, normal work (installs of open-source tools,
file writes under ~/re-halogen, git commits/pushes to Rangizingo/re-halogen,
read-only analysis, running extracted code objects through disassemblers) is
pre-approved. Anything outside this scope — running flash_serve, touching the
live engine (pid 2918), writes outside ~/re-halogen, new repos, spending money,
deleting anything outside the scratch dirs — still requires asking Peter.

Read first: `AGENTS.md` (layout + gotchas), `analysis/report.md` (state of
play). Tooling: REA CLI `~/.hermes/tools/node-26.7.0-linux-x64/bin/rea`,
Ghidra `~/.hermes/tools/ghidra-12.1.2`, r2, pwntools venv
`~/.hermes/venvs/pwn` (`REA_PWNTOOLS_PYTHON`). Ghidra jobs: `sudo systemd-run
--property=MemoryMax=16G` (scripts/run_fals4.sh pattern), TMPDIR outside
dot-paths, PIE base +0x100000 for `rea function`.

Definition of done: every section below has an evidence-backed writeup in
`analysis/`, the unknowns ledger is closed or explicitly bounded, and
report.md's §6 "Not yet done" is empty.

---

## Progress log (updated 2026-10-10)

Evidence lives in `analysis/` and `work/`; decompiles in `work/hgn/*.c`
(27 Ghidra JSON passes, 17 cleaned pseudocode files).

**Phase 0 — DONE.** Live probes skipped honestly: engine pid 2918 gone,
127.0.0.1:8730 refuses; 8731 is the bits-live llama stack (not the target).
Protocol grammar corroborated statically (client-side) in
`analysis/runtime-findings.md`. **Version pull rerun 2026-10-10**: all 8
versions (0.11.10/0.12.0/0.13.3/0.13.4/0.15.0/0.16.4/0.17.0/0.17.2) pulled
to `work/versions/`, carved per-version (`carve_fatbin.py`), symbol-set +
string-set diffs run (`scripts/version_deltas.py` → `tmp/deltas_raw.txt`)
→ **`analysis/version-deltas.md` written** (code-object sha256 lineage
across 9 releases; +318 kernels at 0.16.4; k_dv_* at 0.17.2;
HALOGEN_SAMPLED_*/WALK_FALSIFY host strings at 0.17.3; 0.17.2→0.17.3 ships
zero device-code change).

**Phase 1 — DONE at disassembly level.** Clang offload bundle v2 walked
(`scripts/carve_fatbin.py`): 4 code objects in `work/fatbin/` (largest
17.7 MB, 2,982 symbols; `work/fatbin-manifest.json` has offsets/sha256s).
llvm-objdump 21 `--mattr=+gfx1151` disassembled all four →
`analysis/kernels-disasm/*.s` (3.34M lines). 214 per-kernel slices in
`work/kernels/` (k_attn_fa2/qs*, k_dnc_att*/scan3*/prep, k_dn_fused16,
k_lq8w/k_lq4w, k_i4r_rows, k_linear_f32_q4_mt, k_smpg_accept_final,
k_ngram_*). Ghidra AMDGCN: **no module exists** for shader ISA — probed and
documented in `analysis/ghidra-module-writeup.md` (only the F32 management-CPU
module, installed + tested, irrelevant to kernels). Per plan's fallback:
kernel analysis proceeds at objdump level; Ghidra pseudocode for AMDGCN is a
bounded, documented open item. `analysis/fatbin-inventory.md` written
from the manifest (2026-10-10).

**Phase 2 — IN PROGRESS (objdump-level path), 3/5 families written.**
`analysis/kernels-attention.md` (k_attn_fa2/qs*, 264 L),
`analysis/kernels-moe.md` (k_dn_fused16/k_exp*/router, 370 L),
`analysis/kernels-indexer.md` (k_dnc_* sparse-attention, 194 L).
Slices for the remaining families cut 2026-10-10:
`work/kernels-dequant/` (1141: k_lq4w/lq6k/lq8s32/lq8w/lq6w/lbfw/hcup*/
hc6_fused/hc8_fused/ht_*/i4r_*/q4moe) + `work/kernels-spec/` (43:
k_smpg_*/accept_prob/pen_*/sample/ngram_*/dv_quant/dv_scatter), stats in
`tmp/stats-dequant.csv` / `tmp/stats-spec.csv`. Two subagent writeup runs
on these died from model context overflow (transcripts under
`~/.hermes/cache/delegation/live/`); the reads are cheap, the writeup
needs a grep-driven budget — see start-here.md. Acceptance claims:
attention/MoE/indexer claims closed in their writeups; HT rotation-plane
and speculation-cannot-change-output claims need the two missing writeups.

**Phase 3 — DONE.** `.hgn` format fully recovered and
implemented round-trip in `work/hgn/hgnpy.py` (docstring = the spec draft):
104-B header (magic HGN1, version, n_tensors, table_off, data_off, file_bytes,
model_id[64]); 160-B table entries (name[96], dtype, rank≤4, dims u64[4],
absolute payload_off, nbytes, xor32 = XOR of LE u32 words, qparam); payloads
64-B aligned in table order; zero pads at [4,table_off), post-table, inter-
payload, tail; header_sha256 = sha256(file[0:data_off]). 24 dtype codes
(bf16/f32/f16/q4c/fp8r/q8g64/i4l/fp8g/q8s32/q6k/iq4nl/q4a/q5a/ht/q8_0/q5_0/
q5_1/i4r/q6g64…). 165 crafted test containers in `work/hgn/files/`
(valid/bad-xor/truncated/bad-pad/bad-header variants). Repack driver
decompiled: `flash_serve` 0x19af350 (`work/hgn/gs_0x19af350.c`) — GGUF magic
gate, n-gram table requirement (`layers.1.ple.ngram_embedding.weight` /
`per_layer_token_embd.weight`, fatal "Nothing was written" path), 64-B
aligned layout from 0x80, sparse `ftruncate`, `HALOGEN_GGUF_THREADS`
(default 8, clamp 1..64), thread-pool spawn. **`analysis/hgn-format.md`
written; acceptance matrix ran (`tmp/verify_matrix.tsv`, 165 files: 9 PASS
/ 160 FAIL as designed) and the round-trip suite
(`tmp/rt_roundtrip.py` → `work/hgn/rt/`) PASSES all 5: build→verify PASS,
corrupt-payload→checksum FAIL, parse-back field-exact, truncate→layout
FAIL, HT negative matrix (9 qparam forms, all rejected as
"form the engine does not read" + companion-plane rule).**

**Phase 4 — DONE.** `analysis/npu-engine.md` written. getaddrinfo: single call
site 0x35042 in FUN_00133090 (listen), loopback gate proven (sa_data[2]==0x7f /
::1 checks; no connect import) — verdict loopback-only. XRT call graph documented
(device→elf→hw_context→ext::kernel "main:<graph>"→run start/wait→bo
host-only 0x20000000 = XCL_BO_FLAGS_HOST_ONLY, not CACHEABLE). Containers:
devices.hnpm = text key=value manifest (parser FUN_00164090, xor32 FUN_00162fd0);
consts.hnpb = HNPB v1 binary blob pack (FUN_00167960, 0xa8-B table, 4K-aligned
blobs); `<id>.hnpw` = same HNPB v1 reader (no HNPW magic exists). serve/listen
protocol verbs recovered (INFO/GEN/DEC/IMG/EMB/SCO/CLS/PFX/LEDGER/CANCEL/QUIT,
READY banners, ERR/OK replies); worker FUN_001362d0 + dispatcher FUN_0012b110
decompiled (work/npu/*.pseudo.c). Bounded-open: HNPB +0x08 header u32, per-blob
u32 pair @+0x50/+0xa0, per-verb argument grammars.

**Phase 5 — DONE.** `analysis/dflash2-probe.md` written. Verdict: dflash2
is a reserved wire-protocol drafter slot (id 2), hard-wired to 0 in the
INFO format string of every engine 0.11.10→0.17.3 (zero `dflash` strings
in any engine binary); the client 400s it before the engine sees it. The
shipped second drafter is the n-gram prompt-lookup (HALOGEN_PLD,
k_ngram_* kernels) running beside the MTP head. Bounded-open: engine
behavior for a raw `GEN ... 2` on the socket.

**Phase 6 — NOT STARTED.** Ledger, coverage, evidence bundle, report
rewrite. Blocked on Phase 2's last two writeups (see start-here.md).

## Phase 0 — Runtime corroboration (cheapest, do first)

Goal: turn inference into observation on the live engine family.

1. `capture_process_scenario` / plain `curl` against the **local** engine
   front (bits-live llama-server stack is NOT the target; the halogen-family
   engine on this host is pid 2918, engine port 8730 loopback). Read-only
   probes only: `PING`, one tiny `GEN` at temperature 0, `/health` on 8731.
   Do not send SCHEMA/THINK/GUARD variants that mutate state; do not restart,
   signal, or attach to pid 2918. If 8730 refuses PING, skip the live probes
   and note it — do not start a new engine instance.
2. `compare_application_versions`: pull 0.17.2, 0.17.0, 0.16.4, 0.15.0 image
   layers (registry pattern in report.md §intro; token flow: ghcr.io/token +
   manifest + layer blobs). `compare_functions` / `find_changed_behavior` on
   flash_serve across versions; verify 3 changelog claims (0.17.3 sampled
   decode, 0.12.0 indexer rewrite "byte-identical", 0.13.4 end-of-turn guard)
   against binary deltas.
3. Output: `analysis/runtime-findings.md`, `analysis/version-deltas.md`.

Acceptance: protocol grammar claims marked observed vs inferred; ≥3 changelog
claims verified/refuted with evidence IDs.

## Phase 1 — fatbin extraction + AMDGCN toolchain

Goal: disassemble the 23 MB device code with AMD's own LLVM (no Hopper/IDA).

1. Parse clang offload bundle **v2** at file 0x52000 (magic
   `__CLANG_OFFLOAD_BUNDLE__`, version u16 at +22 = 2, offset-table-size u64
   at +32; entries: target triple, encoding, content offset; content region
   precedes the table). Write `scripts/extract_fatbin.py`; verify every
   extracted blob is a valid ELF with `e_machine == 0xe6` (EM_AMDGPU) or LLVM
   bitcode; record counts + sha256s in `analysis/fatbin-inventory.md`.
2. Install AMD LLVM objdump free: `sudo apt-get install -y llvm` (upstream
   LLVM ≥ 19 supports gfx1151) and/or extract `amd-smi`-adjacent LLVM tools
   from the image's `_rocm_sdk_devel` layer (layer 4, 773 MB — it ships
   llvm-objdump with AMDGPU targets). Prefer the ROCm one: exact ISA match.
3. `llvm-objdump -d --mattr=+gfx1151` every code object →
   `analysis/kernels-disasm/*.s`.
4. Ghidra AMDGCN: try community processor modules
   (e.g. `protondfx/ghidra-amdgpu`, GSoC AMDGPU modules). Install into
   `ghidra-12.1.2/Ghidra/Processors/`, run `rea analyze` on one code object.
   If no module handles RDNA3.5 ISA cleanly, fall back: import the
   llvm-objdump listing into Ghidra as binary + apply a hand-written loader
   script, or keep analysis at objdump level with r2 `gcn` plugin as the
   disassembler and do xref/data recovery manually. Record which path worked.

Acceptance: every kernel entry point (the `k_*` symbols in the fatbin symbol
tables) has disassembly; ≥10 core kernels have Ghidra pseudocode.

## Phase 2 — Core kernel analysis

Goal: explain, with pseudocode, what the engine's numbers claim.

Priority kernels (from `analysis/functions.txt` + fatbin symbol tables):
1. Attention: `k_attn_qs*` (block-tiled variants), `k_dnc_att*`, FA variants
   (`HALOGEN_ATTN_FA/FD/QS*` flags select them).
2. MoE: `k_exp_gu_w` (gate/up fused, W8/W16/W32 widths), `k_expert_h_q4`,
   router (`HALOGEN_FLASH_ROUTE_*`).
3. Dequant: `k_deq_q4cp_sel`, HT (`HALOGEN_HT_*`) and I4R (`HALOGEN_I4R_*`)
   families — tie each to the quant families in report.md §5.
4. Speculation: `accept_probs` / `k_accept_prob_f32`, n-gram
   (`k_ngram_ids/hist/segstart/ctx_update`) — confirm trunk-argmax commit
   inside the verify kernel, closing the falsifier argument at kernel level.
5. Indexer (sparse attention): the 0.12.0 rewrite targets.

Method: `open_binary` (Ghidra) per code object, `analyze_function` per
kernel, `annotate_native_function` to name/rename as understanding firms;
`build_call_path` where host-side dispatch is reachable. One writeup per
family: `analysis/kernels-<family>.md` — algorithm, quant formats handled,
SIMD/LDS structure, and which README perf claims it explains.

Acceptance: the four claims "speculation can't change output", "prompt cache
resume is bitwise", "HT tensors carry a rotation plane", "n-gram table is a
pure lookup" each have kernel-level evidence.

## Phase 3 — halogen-tools: .hgn format spec

Goal: written format spec, verified by parsing the shipped checkpoints'
metadata (not the 118 GiB weights — parse headers/tables only).

1. Decompile: `verify`, `inspect`, `convert`/repack paths (entry via
   `rea analyze halogen-tools` overview → main → dispatch).
2. Recover: header layout, tensor-family table, per-tensor
   name/dtype/qparam/dims/nbytes/xor32/sha256 encoding, pads_zero semantics,
   HREF ref-file format (already documented from strings in AGENTS.md),
   GGUF→hgn repack transform (lossless claim).
3. Write `analysis/hgn-format.md`; validate by round-tripping a tiny synthetic
   file through the decompiled logic (python reimplementation vs binary's
   `verify` on a crafted file — running halogen-tools `verify`/`inspect` on a
   file you create in ~/re-halogen is pre-approved; it loads no model).

Acceptance: spec complete enough that a third party writes a reader; 3
round-trip tests pass.

## Phase 4 — halogen-npu

1. Decompile XRT call graph (`xrt::device`, `hw_context`, `bo` map/sync),
   `devices.hnpm` manifest reader, `consts.hnpb`, `hnpw` weight reader.
2. Resolve the `getaddrinfo` question: enumerate call sites, prove loopback
   only (HALOGEN_NPU_* host flags) or report the exception.
3. Write `analysis/npu-engine.md`.

Acceptance: network claim closed; container formats documented.

## Phase 5 — dflash2 drafter

1. Trace probe: serve_api `_probe`/`_parse_info` fields ↔ engine probe
   emitter (strings `drafter_weights`, `dflash2` in the probe line).
2. Find the implementation: candidate = a drafter path in flash_serve with no
   distinctive strings (compare `draft_sample` siblings in the same object
   file region 0x1809xxx-0x180bxxx).
3. Write `analysis/dflash2.md` (or a bounded negative: searched X, not
   present in 0.17.3, evidence IDs).

## Phase 6 — Closure

1. `build_reconstruction_obligation_ledger` → `record_unknown`/
   `update_unknown` for every open question → `verify_unknown_resolution` /
   `verify_reconstruction` where applicable.
2. `evaluate_reconstruction_coverage` → coverage table in
   `analysis/coverage.md`.
3. `export_evidence_bundle` → `analysis/evidence/`.
4. Rewrite `analysis/report.md` §6 → "Findings complete; residual unknowns:
   <list>". Update AGENTS.md status line.
5. Commit each phase (pushy-style: detailed commit bodies). Push at the end.

## Budget & stops

- Phases 0,3,4,5: hours each. Phase 1: ~1 day (module risk). Phase 2: the
  long pole, 2-4 days of compute; if a single Ghidra batch exceeds ~30 min,
  split per code object.
- Stop conditions requiring Peter: any need to run the engine, attach a
  debugger to a live process, install paid software, or write outside
  ~/re-halogen.
- If Ghidra AMDGCN fails entirely: deliver objdump-level analysis + the
  loader-script groundwork, mark Phase 2 pseudocode as the open item, and say
  so plainly in the final report. No silent scope shrink.

## Report-back format (final message to Peter)

Per phase: one line status + artifact paths + the one most interesting
finding. Total: ≤15 lines.
