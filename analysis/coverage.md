# Coverage table (Phases 0 to 7)

Scope is PLAN.md's definition of done: every section has an evidence-backed
writeup, the unknowns ledger is closed or bounded, report.md section 6 lists
residual unknowns only. Produced by hand; the REA coverage evaluator needs
authenticated application Evidence (see `unknowns.md` header).

| phase | deliverable | artifact | verified by | state |
|---|---|---|---|---|
| 0 | runtime corroboration, version deltas | runtime-findings.md, version-deltas.md | 9 releases pulled, sha256 lineage | closed |
| 1 | fatbin carve, gfx1151 disassembly | fatbin-inventory.md, kernels-disasm/, ghidra-module-writeup.md | 4 code objects, 3.34M lines | closed (objdump level) |
| 2 | 5 kernel families | kernels-attention.md, -moe.md, -indexer.md, -dequant.md, -speculation.md | grep and line-window evidence per claim | closed |
| 3 | `.hgn` format | hgn-format.md, work/hgn/hgnpy.py | 169-file matrix, 5-test round trip | closed |
| 4 | NPU engine | npu-engine.md | loopback gate at 0x35042 | closed |
| 5 | dflash2 | dflash2-probe.md | string absence across 9 releases | closed |
| 6 | ledger, coverage, report | unknowns.md, coverage.md, report.md s6 | this file | closed |
| 7 | function maps of all three binaries | function-map.tsv, function-map-halogen-tools.tsv, function-map-halogen-npu.tsv, function-map-kernel-stubs.tsv | 3,568 / 3,222 / 1,412 / 1,738 data rows (`wc -l` minus header) | closed |
| 7 | protocol, front end, request path | wire-protocol.md, entrypoint-flow.md, api-layer.md, request-trace.md | 31 steps, each tagged proven or inferred with file:line | closed (inferred steps listed in its Gaps) |
| 7 | live engine run | runtime-run-0.17.3.md, evidence/ | 0.17.3 engine on Ai-core, U04 closed, U26 partial | closed |
| 7 | closure rounds and rebuild guide | closure/ (17 reports), unknowns.md, rebuild-guide.md | two rounds merged into the ledger | closed at plan scope |

Acceptance claims from PLAN.md Phase 2:

| claim | verdict | where |
|---|---|---|
| speculation can't change output | observed on the live engine: greedy MTP vs serial identical (2 prompts, 64 tokens), sampled MTP vs serial with the same seed differ (valid draws, not the same sequence). Kernels alone do not prove it. Greedy: report.md s3. Sampled: accept-by-probability plus residual resample is the standard scheme (inferred); the draw comes from an on-GPU splitmix64 whose seed path is half traced (U23 PARTIAL: the wire seed parse is closed, the copy into the per-request params is not located, `closure/sampling-host-r2.md`) | kernels-speculation.md s1 |
| prompt cache resume is bitwise | contradicted for the in-place cache: the engine's own startup text says "a warm answer is NOT bitwise the cold one" (HALOGEN_PROMPT_CACHE=2, resume-anywhere). Not yet tested for other cache modes (U13) | analysis/evidence/live-0.17.2-startup-and-info.txt |
| HT tensors carry a rotation plane | closed, 128-point Hadamard; kernels-moe.md corrected | kernels-dequant.md s3 |
| n-gram table is a pure lookup | holds at kernel level (integer hash plus modulo) | kernels-speculation.md s3 |

Residual (counted from the table in `unknowns.md`): CLOSED 9, PARTIAL 13,
OPEN-BOUNDED 3 (U03, U06, U22), UNOBSERVABLE 1 (U05), 26 rows. U03 needs the
shipped NPU `.elf` graph files and U19 (PARTIAL) needs a real HT checkpoint;
neither is in this workspace. The prompt cache claim stays contradicted by
the engine itself; U13 closed the host half (no prefill redo on cached
positions, `closure/cache-host.md`).
