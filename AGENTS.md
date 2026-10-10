# re-halogen

Reverse-engineering workspace for **halogen-flash-server** (Peonist, LLC) —
the OpenAI-compatible inference server for Qwen3.8-Flash-Next on AMD Strix
Halo (gfx1151). Target image: `ghcr.io/peonist-ai/halogen-flash-server:0.17.3`
(manifest sha256:3bca0132db3c…, built 2026-10-08).

Status: **~85% complete (see `start-here.md` + PLAN.md "Progress log").**
Settled with evidence: architecture, network surface, falsifier question,
`main()`, `draft_sample` + falsifier callers, the GGUF→.hgn repack driver
(0x19af350), fatbin v2 extraction + full gfx1151 disassembly (llvm-objdump),
a working Python parser/builder for the `.hgn` container (`work/hgn/hgnpy.py`)
with the format spec AND acceptance round-trips passing
(`analysis/hgn-format.md`, `tmp/rt_roundtrip.py`), NPU engine
(`analysis/npu-engine.md`), version deltas across 9 releases
(`analysis/version-deltas.md`), dflash2 (`analysis/dflash2-probe.md`), and
3 of 5 kernel-family writeups (attention, MoE, indexer). Open: the
dequant + speculation kernel-family writeups (slices and stats prepped,
`work/kernels-dequant/`, `work/kernels-spec/`) and Phase 6 closure
(ledger, coverage, evidence bundle, report.md rewrite). `PLAN.md` = the
approved autonomous completion plan (Phases 0-6); execute it when Peter says
"run the halogen plan".

## Layout

- `binaries/` — the three shipped ELFs, verbatim from image layers:
  - `flash_serve` (26 MB, sha256 af4f07bb…) — the closed engine (C++/HIP, AMD clang 23, hipBLASLt)
  - `halogen-tools` (26 MB, sha256 a79ad40c…) — checkpoint verify/inspect/ppl/niah + GGUF→.hgn repacker
  - `halogen-npu` (1.1 MB) — XRT/AIE NPU engine (source-extracted copy)
- `image/` — deployment-surface files extracted from image layers (entrypoint,
  healthcheck, npu models manifest, `serve_api.py`, `tool_parse.py`,
  `halogen_tools.py`, request_records, bench tools, EULA). These are the
  vendor's own files, comments stripped, shipped in the image.
- `analysis/` — REA output:
  - `report.md` — the findings report (start here)
  - `main_pseudo.c` — Ghidra decompile of engine `main()` (0x182ba00 in Ghidra's PIE space)
  - `gf_0x19098d0.c` — `draft_sample` (MTP drafter + falsifier gate), decompiled
  - `gf_0x1909e50.c`, `gf_0x1851dc0.c` — falsifier-caller functions, decompiled
  - `*_analysis.json` — REA binary_overview evidence (procedure/string counts)
  - `layout_flash_serve.json` — REA pwntools ELF layout evidence (ev_58f4f920…)
  - `functions.txt` — r2 function inventory (2,112 fns at file-vaddr; Ghidra base = +0x100000)
  - `*.asm`, `fals_*.txt` — targeted disassembly/xref passes
- `scripts/` — the systemd-run launcher pattern used for Ghidra (see Gotchas).

## Reproducing analysis runs

Ghidra 12.1.2 lives at `~/.hermes/tools/ghidra-12.1.2` (JAVA_HOME=
/usr/lib/jvm/java-21-openjdk-amd64). REA CLI:
`~/.hermes/tools/node-26.7.0-linux-x64/bin/rea`.

### Gotchas (all learned the hard way)

1. **Ghidra rejects paths with dot-prefixed elements** (`Path element starting
   with '.' is not permitted`) — never run REA's Ghidra provider with TMPDIR
   under `~/.hermes/...`. Use `TMPDIR=/home/rangizingo/re-halogen/tmp`.
2. **Hermes background worker scopes are capped at 4 GiB** (systemd
   MemoryMax); Ghidra peaks ~4.8 GiB and gets OOM-killed (exit 143). Run Ghidra
   jobs via `sudo systemd-run --property=MemoryMax=16G` (see
   `scripts/run_fals4.sh`).
3. **Ghidra loads this PIE at base 0x100000**: r2/file address + 0x100000 =
   the address `rea function` accepts. `main` = 0x172ba00 (file) = 0x182ba00
   (Ghidra). Stripped binary: no symbol names, use addresses.
4. `rea function` output redirected through journalctl needs
   `json.JSONDecoder().raw_decode` from the first `{` (journal prefix lines).
5. pwntools provider (inspect-binary-layout) needs
   `REA_PWNTOOLS_PYTHON=/home/rangizingo/.hermes/venvs/pwn/bin/python`.

## Open questions / next targets

- Kernel algorithm writeups from `work/kernels/*.s` slices (Phase 2):
  attention (k_attn_fa2/qs*), MoE (k_dn_fused16, k_lq8w/k_lq4w, k_i4r_rows,
  k_linear_f32_q4_mt), dnc indexer (k_dnc_att*/scan3*/prep), speculation
  (k_smpg_accept_final, k_ngram_*). Ghidra cannot decompile AMDGCN (no
  module; see analysis/ghidra-module-writeup.md) — objdump-level analysis.
- `halogen-npu`'s getaddrinfo: which host:port pairs, any non-loopback use.
- `.hgn` format: recovered + implemented (`work/hgn/hgnpy.py`); formal
  `analysis/hgn-format.md` + verify round-trips pending.
- The `dflash2` drafter (id 2 in serve_api) — no matching strings in
  flash_serve 0.17.3; engine reports capability via probe flags.
- Version deltas 0.17.2/0.17.0/0.16.4/0.15.0: pull stalled at token step,
  rerun `scripts/pull_versions.sh`.

## Ground rules

- Read-only analysis of vendor binaries; do not run `flash_serve` from this
  folder (a production engine from the same family runs on this host).
- EULA v1.0 §3 permits reverse engineering without limitation and
  encourages it as an open community challenge: findings, specs,
  disassemblies, and derived implementations may be published and shared
  freely (attribution per §2). Our own repo material is CC0
  (`LICENSE`).
- Every mutation on this host needs Peter's explicit go first.
