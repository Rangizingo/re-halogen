# re-halogen

Reverse-engineering workspace for **halogen-flash-server** (Peonist, LLC) —
the OpenAI-compatible inference server for Qwen3.8-Flash-Next on AMD Strix
Halo (gfx1151). Target image: `ghcr.io/peonist-ai/halogen-flash-server:0.17.3`
(manifest sha256:3bca0132db3c…, built 2026-10-08).

Status: **in progress.** Architecture, network surface, and the falsifier
question are settled with evidence; kernel-level decompilation has not started.

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

- gfx1151 device code: 23 MB `.hip_fatbin` section (file 0x52000,
  0x16a3aa0 bytes). The clang offload bundle v2 header parses (magic at
  0x52000, version 2) but the offset-table walk needs the v2 layout
  (offset-table-size field at +32, entries follow content). Extract per-target
  code objects, then Ghidra AMDGCN analysis. This is the kernel-level work.
- `halogen-npu`'s getaddrinfo: which host:port pairs, any non-loopback use.
- `.hgn` container format spec from `halogen-tools` (magic `HGN1` strings,
  tensor families, xor32/sha256 fields per `inspect --json`).
- The `dflash2` drafter (id 2 in serve_api) — no matching strings in
  flash_serve 0.17.3; engine reports capability via probe flags.

## Ground rules

- Read-only analysis of vendor binaries; do not run `flash_serve` from this
  folder (a production engine from the same family runs on this host).
- EULA §3 permits RE for interoperability/security research; do not
  redistribute derived source as our own work.
- Every mutation on this host needs Peter's explicit go first.
