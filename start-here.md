# START HERE: halogen RE, state of play (2026-10-10)

Read this first, then `PLAN.md` (the phase plan and progress log), then
`analysis/report.md` (findings) and `analysis/rebuild-guide.md` (how to
rebuild a compatible server from the writeups).

**Overall: Phase 7 done at plan scope (2026-10-10).** All phases have
evidence-backed writeups. Unknowns, counted from the table in
`analysis/unknowns.md`: CLOSED 25, PARTIAL 0, OPEN-BOUNDED 1 (U06),
UNOBSERVABLE 1 (U05). What is
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
| 7 | closure reports merged into the ledger in five rounds (`analysis/closure/`, 48 files); kernel rewrite harness (`rewrite/`, first bitwise match); rebuild guide; i4r decode; engine runs 2 to 5 | `analysis/closure/`, `analysis/rebuild-guide.md` |

## Open

- U12 CLOSED in s6 (run 5 Tap payload, dnc-scan3-r6 n1/n2, scan3 B, C, D, G value-neutral by the rewrites).
- U22 CLOSED in s6, merged in s7 (PR #24, e662a3f): all 1,738 kernel instances rewritten in HIP and matched bitwise (`rewrite/match-table.md`); constant audit with integer bounds 1,738 AUDIT OK (`rewrite/const-audit.md`); replaced rewrites in `rewrite/superseded.tsv`.
- U06 old patch releases, U05 future releases: named limits by decision.
- The inferred steps of the request trace are listed in its Gaps section.

Peter-only steps are in `.claude/needs-human.md`.

## Hazard

A test engine run deleted the live service's 60 GiB disk prompt cache
(`analysis/runtime-run-0.17.3.md`). Test engines never mount
`/srv/halogen-cache`; they get their own cache dir and
`HALOGEN_CACHE_PRUNE_OLD=0`.

## Licensing (vendor texts re-extracted from the image, 2026-10-15)

**The vendor ships EULA v0.1, effective 25 August 2026.** Verified by creating
a container from `ghcr.io/peonist-ai/halogen-flash-server:0.17.3` and reading
`/licenses/` (sha256 `58f970ff94fb67d4…`), and the same version in 0.17.2,
0.15.0 and 0.13.1. Copies of all six vendor license files are in
`image/licenses/` (`EULA.md`, `THIRD-PARTY-NOTICES.md`, `TRADEMARKS.md`,
`npu/NPU-NOTICES.md`, `npu/XRT-LICENSE`, `npu/XRT-NOTICE`).

What v0.1 grants: use and run for any purpose including commercial, unlimited
machines, no key. Redistribute the Software **unmodified**, with this
Agreement and `THIRD-PARTY-NOTICES.md` travelling intact. On reverse
engineering it does not prohibit it and asks that derived source or kernels
not be redistributed as your own work. It bars distributing a modified image
as halogen, removing notices, and endorsing with the Peonist/halogen marks.

**There is no vendor EULA v1.0.** A text headed "Version 1.0 · Effective 10
October 2026" was committed here at 7af991f by this repo's own account, with
an unconditional RE grant, an "open community challenge" framing, and
publish-findings-freely terms. No image contains it. It is not a Peonist
statement. The text stays in git history as this repo's file; it is no longer
in the working tree, and no doc cites it as the vendor's word.

`LICENSE`: CC0 1.0 over all original material authored in this repo. Vendor
binaries and extracted image files keep their own licenses. The repo is
public (Peter's call, 2026-10-15).

## Ground rules

- Running `flash_serve` and `halogen-tools` was granted by Peter on
  2026-10-10 for this work; only on files created in this workspace, and
  never against the live service's cache.
- Ghidra jobs: TMPDIR outside dot-paths, PIE base +0x100000; on Linux run
  under `systemd-run --property=MemoryMax=16G`.
- Big disassembly listings stay out of git (tarballs; raw listings
  gitignored).
- Mutations outside this repo need Peter first.
