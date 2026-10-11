# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Reverse-engineering workspace for `halogen-flash-server` 0.17.3 (Peonist, LLC), an OpenAI-compatible inference server for Qwen3.8-Flash-Next on AMD Strix Halo (gfx1151). There is no application code to build. The repo holds vendor binaries, analysis output, and small Python/shell tooling. Deliverables are evidence-backed markdown writeups in `analysis/`.

Read order: `start-here.md` (state of play, what remains) > `PLAN.md` (7-phase plan + progress log) > `analysis/report.md` (findings). `AGENTS.md` is the original agent brief; it is protected, so each write to it needs explicit approval.

## Layout that needs several files to understand

- `binaries/`: the three shipped ELFs (`flash_serve` engine, `halogen-tools` verify/inspect/repack, `halogen-npu` XRT engine). Stripped, so work by address, not symbol.
- `image/`: vendor files extracted from the container image layers (entrypoint, `serve_api.py`, EULA).
- `work/fatbin/` holds the 4 AMDGCN code objects carved from `flash_serve`'s clang offload bundle v2 (`scripts/carve_fatbin.py`). `analysis/kernels-disasm/` (299 MB, 3.34M lines) is their `llvm-objdump --mattr=+gfx1151` output. `work/kernels*/` holds per-kernel `.s` slices cut by `scripts/slice_kernels.py`; `tmp/stats-*.csv` has per-kernel instruction-class counts from `scripts/kernel_stats.py`.
- `work/hgn/hgnpy.py` is the `.hgn` container parser/builder; its docstring is the spec draft, formalized in `analysis/hgn-format.md`. `work/hgn/files/` has 169 crafted test containers.
- `work/versions/<ver>/` has the 9 pulled releases (0.11.10 to 0.17.3), diffed by `scripts/version_deltas.py`.
- `rewrite/` (U22): one HIP rewrite per kernel family in `rewrite/kernels/`, specs in `rewrite/specs/`, per-group result tables in `rewrite/tables/` merged into `rewrite/match-table.md`. A fixed kernel is a copy (`<family>_c.hip`, `_d`), never an edit of the original; `rewrite/superseded.tsv` lists original > replacement, and the constant audit (`rewrite/harness/const_audit.py`, output `rewrite/const-audit.md`) and coverage checks use the end of each chain. Per-instance audit exceptions live in `rewrite/const-audit-exceptions.d/<group>.tsv` and must cite the vendor listing line and the rewrite line.
- `analysis/*.md` are the writeups. `analysis/kernels-indexer.md` is the format template for kernel-family writeups.

## Commands

```bash
# Reproduce the .hgn acceptance suites (these run halogen-tools verify/inspect, Linux x86-64 only)
bash scripts/verify_matrix.sh          # 169-file matrix, expect 9 PASS / 160 designed FAIL
python3 tmp/rt_roundtrip.py            # 5-test round-trip suite, expect all PASS

# Kernel work: rank from stats, then grep instruction classes, then read 50-150 line windows
sort -t, -k2 -n -r tmp/stats-spec.csv | head
grep -c 'v_exp\|v_rcp\|v_mfma\|v_permlane\|v_bfe' work/kernels-spec/<slice>.s

# REA (MCP server "rea" is registered at user scope; CLI equivalent)
npx -y rea-agents@6.3.0 doctor --provider ghidra --json

# Tests (pytest; 3 real-data HT tests skip unless HT_TESTDATA is set)
python3 -m pytest -q tests                                   # 145 pass, 4 skip (s7)
bash scripts/fetch_ht_testdata.sh <dir>                      # ~496 MB of vendor slices by HTTP range read
HT_TESTDATA=<dir> python3 -m pytest -q tests                 # 149 pass (derived: 145 + the 4 data skips; not re-run in s7)
python3 -I scripts/ht_decode.py <dir>                        # HT and i4r decode scores vs q4c references

# Kernel rewrite harness (U22): on Ai-core, after rsync -a --exclude __pycache__ rewrite/ ai-core:re-halogen/rewrite/
bash rewrite/harness/build.sh && python3 -I rewrite/harness/compare.py rewrite/specs/<spec>.json --kernel <mangled> --seeds 1,2,3 --sizes 4096,100003
```

There is no CI config. Do not add a CLAUDE.md "Stack" or web conventions here; none apply.

## Gotchas

- Docs written on the Linux analysis host reference `/home/rangizingo/...` and `~/.hermes/tools/...`. This checkout is on macOS. Binaries are x86-64 Linux ELFs, so `halogen-tools`/`flash_serve` cannot run here; analysis is static only.
- Ghidra rejects any path with a dot-prefixed element: keep `TMPDIR` out of dot-directories. Ghidra loads the PIE at base `0x100000`, so Ghidra address = r2/file address + `0x100000` (`main` = 0x172ba00 file = 0x182ba00 Ghidra). On Linux, run Ghidra under `systemd-run --property=MemoryMax=16G`.
- Ghidra has no AMDGCN (shader ISA) module (`analysis/ghidra-module-writeup.md`). Kernel analysis is objdump-level.
- Subagent kernel writeups have died from context overflow reading big slices (`k_dv_quant` is 6068 lines). Read grep-driven windows only, about 15 read calls per writeup.
- Big listings stay out of git (tarballs; raw listings gitignored).

## Ground rules

- Never run `flash_serve` or run `halogen-tools` beyond `verify`/`inspect`, and only on files created in this workspace.
- EULA v1.0 section 3 permits unrestricted reverse engineering; original repo material is CC0 (`LICENSE`). Vendor binaries and image files keep their own licenses. The repo stays private for now.
- Mutations outside this repo need the user's go first. Commit with detailed bodies; push to `https://github.com/Rangizingo/re-halogen` via `gh auth setup-git` (the SSH key is deploy-key-bound, SSH push fails).
- Done means: every section in `PLAN.md` has an evidence-backed writeup, the unknowns ledger is closed or explicitly bounded, and `analysis/report.md` section 6 is empty.
