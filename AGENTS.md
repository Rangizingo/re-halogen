# re-halogen

Reverse-engineering workspace for **halogen-flash-server** (Peonist, LLC) —
the OpenAI-compatible inference server for Qwen3.8-Flash-Next on AMD Strix
Halo (gfx1151). Target image: `ghcr.io/peonist-ai/halogen-flash-server:0.17.3`
(manifest sha256:3bca0132db3c…, built 2026-10-08).

Status: **Phase 7 (complete reverse engineering) done at plan scope,
2026-10-10; see `start-here.md` and `analysis/rebuild-guide.md`.** Settled
with evidence: architecture, network surface, falsifier question, `main()`,
the GGUF to .hgn repack driver, fatbin extraction and gfx1151 disassembly,
the `.hgn` format with passing round trips, the NPU engine, version deltas,
dflash2, five kernel-family writeups, function maps of all three binaries,
the wire protocol, a 31-step request trace and a live 0.17.3 engine run.
Unknowns (`analysis/unknowns.md`, counted from its table): CLOSED 25,
PARTIAL 0, OPEN-BOUNDED 1 (U06), UNOBSERVABLE 1 (U05).

## Layout

- `binaries/` — the three shipped ELFs, verbatim from image layers:
  - `flash_serve` (26 MB, sha256 af4f07bb…) — the closed engine (C++/HIP, AMD clang 23, hipBLASLt)
  - `halogen-tools` (26 MB, sha256 a79ad40c…) — checkpoint verify/inspect/ppl/niah + GGUF→.hgn repacker
  - `halogen-npu` (1.1 MB) — XRT/AIE NPU engine (source-extracted copy)
- `image/` — deployment-surface files extracted from image layers (entrypoint,
  healthcheck, npu models manifest, `serve_api.py`, `tool_parse.py`,
  `halogen_tools.py`, request_records, bench tools) and the vendor's
  `licenses/` (EULA v0.1, trademark policy, third-party and XRT/NPU notices).
  These are the vendor's own files, comments stripped, shipped in the image.
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

- Peter lifted (2026-10-10): running `flash_serve` and `halogen-tools` is
  allowed, and full access to the Ai-core box (10.0.10.84, `ssh ai-core`)
  is granted for Phase 7. Record every run (command, inputs, outputs) under
  `analysis/evidence/`. Hermes on Ai-core may be taken down for the work.
- Still binding: no money spent without Peter's yes, nothing destructive or
  outside Ai-core and this repo, no real user data in runs.
- The vendor EULA in force is **v0.1, effective 2026-08-25**
  (`image/licenses/EULA.md`, sha256 `58f970ff94fb67d4…`, re-extracted from the
  0.17.3 image; v0.1 also in 0.17.2, 0.15.0, 0.13.1). It licenses use and
  running for any purpose including commercial, and redistribution of the
  Software **unmodified** with the Agreement and `THIRD-PARTY-NOTICES.md`
  intact. On reverse engineering it does not prohibit it and asks that
  derived source or kernels not be redistributed as your own work. It bars
  modified builds passed off as halogen and bars removing notices. So:
  the analysis is publishable, `rewrite/` names the vendor kernel each
  rewrite reproduces, and the vendor binaries travel with their notices.
- Do not treat any other EULA text as the vendor's word. The "v1.0, effective
  2026-10-10" text granting unrestricted RE as an open community challenge
  was committed at 7af991f by this repo's git author and appears in no
  published image. If a future release ships a v1.0, re-extract it and cite
  the image digest.
- `LICENSE` is CC0 1.0 for material authored in this repo; `NOTICE` gives
  provenance for everything else. The repo is public (Peter, 2026-10-15).
