# START HERE: halogen RE, state of play (2026-10-10)

Read this first, then `PLAN.md` (the phase plan and progress log), then
`analysis/report.md` (findings) and `analysis/rebuild-guide.md` (how to
rebuild a compatible server from the writeups).

**Overall: Phase 7 done at plan scope (2026-10-10).** All phases have
evidence-backed writeups. Unknowns, counted from the table in
`analysis/unknowns.md`: CLOSED 19, PARTIAL 5, OPEN-BOUNDED 2 (U06,
U22), UNOBSERVABLE 1 (U05). PARTIAL rows say which half is closed. What is
left needs data that is not in this workspace (see "Open" below).

## Done (with artifacts)

| Phase | What | Artifacts |
|---|---|---|
| 0 | Runtime corroboration and version deltas over 9 releases | `analysis/runtime-findings.md`, `analysis/version-deltas.md` |
| 1 | Fatbin carve and gfx1151 disassembly (4 code objects, 3.34M lines); no Ghidra AMDGCN module exists | `analysis/fatbin-inventory.md`, `analysis/ghidra-module-writeup.md` |
| 2 | Five kernel families: attention, MoE, indexer, dequant, speculation | `analysis/kernels-*.md` |
| 3 | `.hgn` container spec, parser/builder, 169-file verify matrix, 5-test round trip | `analysis/hgn-format.md`, `work/hgn/hgnpy.py` |
| 4 | NPU engine: loopback gate, XRT call graph, container formats, serve verbs | `analysis/npu-engine.md` |
| 5 | dflash2: reserved drafter slot, refused by the 0.17.3 engine | `analysis/dflash2-probe.md` |
| 6 | Unknowns ledger, coverage, report section 6 | `analysis/unknowns.md`, `analysis/coverage.md`, `analysis/report.md` |
| 7 | Function maps of all three binaries (3,568 / 3,222 / 1,412 functions, 1,738 kernel stubs) | `analysis/function-map*.tsv`, `analysis/function-map.md` |
| 7 | Wire protocol, entrypoint, API layer, 31-step request trace | `analysis/wire-protocol.md`, `entrypoint-flow.md`, `api-layer.md`, `request-trace.md` |
| 7 | Live 0.17.3 engine run on Ai-core, version compare 0.17.2 to 0.17.3 | `analysis/runtime-run-0.17.3.md`, `analysis/version-compare-0.17.2-0.17.3.md` |
| 7 | closure reports merged into the ledger in three rounds (`analysis/closure/`, 31 files); rebuild guide; i4r decode; engine run 2 | `analysis/closure/`, `analysis/rebuild-guide.md` |

## Open

- PARTIAL rows U02, U11, U12, U21, U26: each row says
  which half is closed and what is left (most need a runtime probe or a
  vendor file not shipped).
- U06 old patch releases, U05 future releases, U22 no AMDGCN decompiler:
  named limits by decision.
- The inferred steps of the request trace are listed in its Gaps section.

Peter-only steps are in `.claude/needs-human.md`.

## Hazard

A test engine run deleted the live service's 60 GiB disk prompt cache
(`analysis/runtime-run-0.17.3.md`). Test engines never mount
`/srv/halogen-cache`; they get their own cache dir and
`HALOGEN_CACHE_PRUNE_OLD=0`.

## Licensing (2026-10-10, commit 7af991f)

- `image/EULA.md` v1.0: section 3 grants reverse engineering without
  limitation; findings and derived implementations are publishable
  (attribution per section 2).
- `LICENSE`: CC0 1.0 for all original repo material. Vendor binaries and
  image files keep their own licenses.
- The repo stays private (Peter's call, 2026-10-10).

## Ground rules

- Running `flash_serve` and `halogen-tools` was granted by Peter on
  2026-10-10 for this work; only on files created in this workspace, and
  never against the live service's cache.
- Ghidra jobs: TMPDIR outside dot-paths, PIE base +0x100000; on Linux run
  under `systemd-run --property=MemoryMax=16G`.
- Big disassembly listings stay out of git (tarballs; raw listings
  gitignored).
- Mutations outside this repo need Peter first.
