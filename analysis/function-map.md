# Function map: flash_serve 0.17.3 (all 3568 functions)

Source: Ghidra 12.1.4 headless decompile (`scripts/ExportDecomp.java`), every function exported to `work/decomp/flash_serve/<addr>.c` with `index.tsv` (addr, name, bytes, callees, callers, strings). Classified by 12 parallel Sonnet agents, one slice each, into `analysis/function-map/part-NN.tsv`. Merged: `analysis/function-map.tsv` (all rows), `analysis/function-map-core.tsv` (engine logic only, 401 rows). Ghidra found 3568 functions, more than the 2,112 of the earlier radare2 inventory.

Checks run on the merge: 3568 of 3568 index rows mapped, 0 missing, 0 duplicated, every role on the fixed list.

## Roles

| role | functions |
|---|---|
| gpu-dispatch | 1680 |
| cxx-stl | 674 |
| hip-runtime | 266 |
| crt-runtime | 248 |
| config-env | 216 |
| util | 83 |
| kv-cache-prefix | 73 |
| protocol-net | 61 |
| sampling-speculation | 60 |
| gguf-repack | 58 |
| model-forward | 56 |
| hgn-loader | 38 |
| scheduler-slots | 24 |
| logging-diag | 16 |
| unknown | 15 |

Confidence: high 1546, med 1380, low 642. The gpu-dispatch rows (1680) are host launch stubs that pack arguments and call the HIP launch; the registration function `01968680` lists 1,462 distinct `k_*` kernel names. Most stubs were classified by shared pattern, not read one by one, and are not yet mapped to their kernel names.

## Engine logic

386 functions carry the actual engine logic (the 8 groups below); `function-map-core.tsv` has 401 rows because it also holds the 15 `unknown` rows: model-forward 56, sampling-speculation 60, kv-cache-prefix 73, protocol-net 61, gguf-repack 58, hgn-loader 38, scheduler-slots 24, logging-diag 16. These are the groups for the deep passes that close the host-code unknowns (`unknowns.md`) and feed `request-trace.md`.

## Unexplained

15 functions are role `unknown` (listed below by address); 642 rows have low confidence and are the first to recheck.

- `0182ea00` unknown_0182ea00: 126B, calls 01903630 and 01905000, no strings
- `01853790` unknown_01853790: calls 01898c30 only
- `0187f290` unknown_range_copy: 1.3KB, recursion+memcpy only, no strings
- `0187f7c0` unknown_f7c0: between JSON and prompt-cache clusters, no strings
- `0187f980` unknown_f980: between JSON and prompt-cache clusters, no strings
- `01883590` unknown_3590: no strings, not enough context in index
- `01893120` unknown_1_5KB_bitops: calls FUN_01994340 5x, no strings
- `01893b70` unknown_set_cleanup: erases string rb-tree, calls act_stats
- `01894dd0` unknown_4dd0: no strings, not enough context in index
- `01896988` unknown_6988: no strings, not enough context in index
- `01897940` unknown_7940: no strings, not enough context in index
- `01898720` unknown_8720: no strings, not enough context in index
- `01898840` unknown_8840: no strings, not enough context in index
- `018ca910` unknown_351b_4callers: 4 callers, calls FUN_01940a80 twice, no strings
- `0194395f` misdecoded_catch_pad: bytes after stubs decoded as catch handlers, not real code
