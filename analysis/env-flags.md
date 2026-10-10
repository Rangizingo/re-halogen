# Environment variables read by `flash_serve` 0.17.3

Static analysis only. Closes the "engine-read env flag list is incomplete" gap (`analysis/rebuild-guide.md` section 11, Other gaps). Report: `analysis/closure/env-dline-r5.md`.

## Counts

| Quantity | Value | Command or source |
|---|---|---|
| `getenv` call sites in the binary | 418 | disassembly scan below |
| Distinct variables read through `getenv` | 256 (all `HALOGEN_`; no other prefix) | same scan, distinct string operands |
| Distinct names with a literal in the decompile | 247 | `grep -ho 'getenv *( *"[A-Za-z0-9_]*"' work/decomp/flash_serve/*.c \| sed 's/.*"\(.*\)"/\1/' \| sort -u \| wc -l` |
| Read by `getenv` but missing from the decompile text | 9 | 256 minus 247 (decompile truncated; listed in the table) |
| Standalone `HALOGEN_` strings in the file | 265 | python `re.findall(rb'(?<![A-Za-z0-9_])HALOGEN_[A-Z0-9_]+\x00', binaries/flash_serve)` |
| Strings with no reader | 9 | 265 minus 256 (section at the end) |
| `secure_getenv`, `setenv`, `putenv` imports | 0 | dynsym PLT list (295 JUMP_SLOT entries) has `getenv` only |
| `environ` scans | 1 | `work/decomp/flash_serve/01854310.c:111` (`PTR_environ_019fa550`), the disk-cache fingerprint |

## Method

1. Parse `.rela.plt` and the PLT stubs of `binaries/flash_serve`; `getenv` is the stub at VMA 0x18f4320 (Ghidra 0x19f4320).
2. Scan the executable segment (VMA 0x172b880, file offset 0x172a880, size 0x1c97f0) for every `e8 rel32` whose target is that stub; walk back up to 60 bytes to the nearest `48 8d 3d disp32` (`lea rdi,[rip+disp]`); the target string is the variable name. 418 sites, 0 without a resolved name.
3. For each site decode 45 instructions with capstone: `cmp byte [r],0x30|0x31` gives a flag, a call to `__isoc23_strtol[l]` gives an int, `strtod` a float, `__isoc23_sscanf` a list. The default is the first `mov reg,imm` or `xor` on the NULL branch after `test rax,rax`.
4. Function owner: the nearest preceding entry in `work/decomp/flash_serve/index.tsv` (Ghidra address = VMA + 0x100000).
5. The skip table: `PTR_s_HALOGEN_ACT_STATS_019f60e0` in `01854310.c:146`, 61 pointers resolved through `.rela.dyn` R_X86_64_RELATIVE entries (the file bytes are zero until relocated).

Limits. Type and default come from the first site only and are instruction-pattern heuristics: "see site" means the pattern did not match, "on? (n)" means a `==1` flag whose null branch loaded n. Flags tested with `==1` are on only when the value is exactly `1`; flags tested with `!=0` are on for any value not starting with `0`. The "Selects" column is a one-line gloss: names with a closure writeup cite it in the Cite column; the rest are inferred from the name and the consumer and carry `[n]` where the guess is weak. Sites in one function that re-read the same name (static caches) count separately.

## Cache key column

`FUN_01854310` (disk-cache fingerprint builder, `01854310.c:111-262`) walks `environ`, takes every entry named `HALOGEN_*`, and appends `name=value` for each name NOT in a 61-entry skip table; the pieces are sorted (`FUN_018578b0`) and appended to the fingerprint text. `skip` in the last column means the variable is in that table (changing it does not change the cache key); `key` means a set value changes the fingerprint and so selects a different disk-cache directory. The three path variables `HALOGEN_MTP_HEAD`, `HALOGEN_VISION_TOWER`, `HALOGEN_MATMUL_TUNING_FILE` are also read directly into named fingerprint fields (`01854310.c:93-98`).

## Table (alphabetical)


| # | Variable | Getter (consumer fn, +others) | Sites | Type | Default (unset) | Selects | Cite | Cache key |
|---|---|---|---|---|---|---|---|---|
| 1 | `HALOGEN_ACT_STATS` | FUN_0190c8b0 | 1 | string/other | unset (string) | Activation statistics toggle; path given if nonempty. | 0190c8b0.c, call 0x190c9b2 (decompile has no literal; disasm only) | skip |
| 2 | `HALOGEN_ADMIT_CHUNK` | FUN_0182ba00 | 1 | int | see site | Size of one prefill admit piece while others decode. | 0182ba00.c, call 0x182dd1c; analysis/closure/cache-host.md:55 | skip |
| 3 | `HALOGEN_ADMIT_TICKS` | FUN_0182ba00 | 1 | int | see site | Decode ticks between admit pieces (min 1). | 0182ba00.c, call 0x182dd43; analysis/closure/cache-host.md:60 | key |
| 4 | `HALOGEN_ARENA_UNION` | FUN_01893080 | 1 | int | 1 | 1 unions per-forward arenas to save memory (default 1). [n] | 01893080.c, call 0x1893088 | key |
| 5 | `HALOGEN_ARGMAX_BLOCKS` | FUN_0194bd00 | 1 | int | 64 | Argmax kernel blocks (default 64). | 0194bd00.c, call 0x194bd08 | skip |
| 6 | `HALOGEN_ATTN_FA` | FUN_018bdc00 | 1 | int | 64 | Flash-attention tile/option (default 64). | 018bdc00.c, call 0x18bdc08 | key |
| 7 | `HALOGEN_ATTN_FD` | FUN_0190c570 | 1 | int | 16 | Flash-decode split (default 16). | 0190c570.c, call 0x190c578 | key |
| 8 | `HALOGEN_ATTN_QS` | FUN_018bdd10 | 1 | int | 1 | qs attention variant (default 1). | 018bdd10.c, call 0x18bdd18 | key |
| 9 | `HALOGEN_ATTN_QS_BT` | FUN_01884460 +2 | 3 | int | see site | Block-tile selector for the qs attention kernel. | 01884460.c, call 0x1885969 | key |
| 10 | `HALOGEN_ATTN_QS_BT2_LDS` | FUN_019ef0d0 | 1 | int | 0 | qs_bt2 dynamic LDS in KiB (default 0). | 019ef0d0.c, call 0x19ef0d8; analysis/closure/indexer-attn-host.md:95 | key |
| 11 | `HALOGEN_ATTN_QS_BT2_OPT` | FUN_019ef070 | 1 | int | 0 | qs_bt2 OPT (default 0). | 019ef070.c, call 0x19ef078; analysis/closure/indexer-attn-host-r2.md:31 | key |
| 12 | `HALOGEN_ATTN_QS_BT2_PROBE` | FUN_019ef020 | 1 | int | see site | qs_bt2 probe arg a. | 019ef020.c, call 0x19ef028; analysis/closure/indexer-attn-host.md:93 | key |
| 13 | `HALOGEN_ATTN_QS_BT2_WPE` | FUN_019ef0a0 | 1 | int | 0 | qs_bt2 WPE (default 0). | 019ef0a0.c, call 0x19ef0a8; analysis/closure/indexer-attn-host.md:94 | key |
| 14 | `HALOGEN_ATTN_QS_BT4_OPT` | FUN_019f0920 +1 | 2 | int | 0 | qs_bt4 OPT (default 0). | 019f0920.c, call 0x19f0928 | key |
| 15 | `HALOGEN_ATTN_QS_MIN` | FUN_018bdd40 | 1 | int | 0 | Minimum rows for qs attention (default 0). | 018bdd40.c, call 0x18bdd48 | key |
| 16 | `HALOGEN_ATTN_QS_PROBE` | FUN_019ed5f0 | 1 | int | see site | qs attention probe. | 019ed5f0.c, call 0x19ed5f8 | key |
| 17 | `HALOGEN_ATTN_QS_RSM` | FUN_019ed640 | 1 | int | 0 | qs attention RSM (default 0). | 019ed640.c, call 0x19ed648 | key |
| 18 | `HALOGEN_BF16_OVERRIDE` | FUN_0190c0b0 | 1 | string/other | unset (string) | Test override of a bf16 tensor. | 0190c0b0.c, call 0x190c11e | key |
| 19 | `HALOGEN_BF16_OVERRIDE_DIR` | FUN_0190c0b0 | 1 | string/other | unset (string) | Directory for bf16 override tensors. | 0190c0b0.c, call 0x190c13f | key |
| 20 | `HALOGEN_CACHE_BRANCHES` | FUN_01852ca0 | 2 | int | see site | Branch entries per slot (default 2 in entrypoint math). | 01852ca0.c, call 0x1852ea3; analysis/closure/cache-host.md:36 | key |
| 21 | `HALOGEN_CACHE_CKPT` | FUN_01852ca0 | 1 | flag (!=0) | see site | Prompt-cache checkpoint-entry toggle (0 disables). | 01852ca0.c, call 0x1852e62; analysis/closure/cache-host.md:36 | key |
| 22 | `HALOGEN_CACHE_DIR` | FUN_01853140 | 1 | string/other | unset (string) | Directory for the disk cache tier. | 01853140.c, call 0x185315a; analysis/closure/cache-host.md:65 | skip |
| 23 | `HALOGEN_CACHE_DISK_DEEPEN` | FUN_01852ca0 | 1 | flag (!=0) | on | 0 disables disk-tier deepening of stored entries (default on). | 01852ca0.c, call 0x1852e18; analysis/closure/cache-host.md:36 | key |
| 24 | `HALOGEN_CACHE_DISK_GIB` | FUN_01853140 | 1 | float | 64.0 | Disk cache budget in GiB (default 64, 0 = unbounded). | 01853140.c, call 0x1853195; analysis/closure/cache-host.md:74 | skip |
| 25 | `HALOGEN_CACHE_ENTRIES` | FUN_01852ca0 | 1 | int | see site | Entry cap override; default (BRANCHES<<SNAP3)+FULL+1 per slot. | 01852ca0.c, call 0x1852f60; analysis/closure/cache-host.md:35 | skip |
| 26 | `HALOGEN_CACHE_EVICT` | FUN_01852ca0 | 1 | flag (!=0) | see site | Prompt-cache eviction toggle (0 disables). | 01852ca0.c, call 0x1852e87; analysis/closure/cache-host.md:36 | key |
| 27 | `HALOGEN_CACHE_FILE` | FUN_01884460 +1 | 2 | string/other | unset (string) | Single-file snapshot path for the prompt cache; unset = disk tier off. | 01884460.c, call 0x18884dd; analysis/closure/cache-host.md:36 | skip |
| 28 | `HALOGEN_CACHE_FULL` | FUN_01852ca0 | 1 | flag (!=0) | on | 0 drops the full-prompt cache entry kind (default on). | 01852ca0.c, call 0x1852df3; analysis/closure/cache-host.md:36 | skip |
| 29 | `HALOGEN_CACHE_INPLACE` | FUN_01852ca0 | 1 | flag (!=0) | on | 0 stores prompt-cache entries as copies instead of in place (default 1). | 01852ca0.c, call 0x1852dce; analysis/closure/cache-host.md:23 | skip |
| 30 | `HALOGEN_CACHE_PRUNE_OLD` | FUN_01854e90 | 1 | flag (==1) | see site | 1 prunes old-format disk cache entries at start. | 01854e90.c, call 0x1855719; analysis/closure/cache-host.md:78 | skip |
| 31 | `HALOGEN_CACHE_SNAP3` | FUN_01852ca0 | 1 | flag (!=0) | on | 0 uses 1 snapshot per branch instead of 2 (default on). | 01852ca0.c, call 0x1852e3d; analysis/closure/cache-host.md:36 | skip |
| 32 | `HALOGEN_CK_OVERLAY` | FUN_01884460 +1 | 2 | string/other | unset (string) | Overlay file for the checkpoint: none or 0 = bare, or a path. | 01884460.c, call 0x18849c1; analysis/entrypoint-flow.md:98 | key |
| 33 | `HALOGEN_CK_OVERLAY_SKIP` | FUN_01884460 +1 | 2 | string/other | unset (string) | Skip list/flag for overlay tensors. [n] | 01884460.c, call 0x1884d64 | key |
| 34 | `HALOGEN_COMPOSABLE_CONTEXT` | FUN_018536a0 | 1 | string/other | unset (string) | Composable-context (SEG) cache toggle. | 018536a0.c, call 0x18536c7 | skip |
| 35 | `HALOGEN_COMPOSABLE_CONTEXT_BYTES` | FUN_018536a0 | 1 | float | see site | Byte budget for composable context. | 018536a0.c, call 0x185374c | skip |
| 36 | `HALOGEN_COMPOSABLE_CONTEXT_FLOOR` | FUN_018536a0 | 1 | int | 0 | Minimum unit size for composable context (default 0). | 018536a0.c, call 0x18536f1 | skip |
| 37 | `HALOGEN_COMPOSABLE_CONTEXT_SEAM` | FUN_018536a0 | 1 | int | see site | Seam rows used when joining composable units. | 018536a0.c, call 0x185371f; analysis/closure/cache-host.md:41 | skip |
| 38 | `HALOGEN_DEBUG_SECT` | FUN_01946ba0 | 1 | string/other | unset (string) | Debug section dump toggle (default 0). | 01946ba0.c, call 0x1946bd2 | key |
| 39 | `HALOGEN_DEC_PROBE` | FUN_019468a0 | 1 | int | 0 | Decode probe (default 0). | 019468a0.c, call 0x19468a8; analysis/closure/expert-dispatch-r5.md:58 | key |
| 40 | `HALOGEN_DEQ_CB` | FUN_019467a0 | 1 | int | 1 | Dequant CB variant (default 1). | 019467a0.c, call 0x19467a8; analysis/closure/expert-dispatch-r5.md:134 | key |
| 41 | `HALOGEN_DEQ_KERN` | FUN_019466e0 | 1 | int | 2 | Dequant kernel pick (default 2). | 019466e0.c, call 0x19466e8; analysis/closure/expert-dispatch-r5.md:134 | key |
| 42 | `HALOGEN_DEQ_PROBE` | FUN_01946730 | 1 | int | see site | Dequant probe selector. | 01946730.c, call 0x194673b; analysis/closure/expert-dispatch-r5.md:134 | key |
| 43 | `HALOGEN_DEQ_STREAM` | FUN_01944ff0 | 1 | int | 0 | Dequant stream option (default 0). | 01944ff0.c, call 0x1944ff8 | key |
| 44 | `HALOGEN_DMALLOC_LOG` | FUN_0190c5b0 +1 | 2 | int | 0 | Device-malloc logging (default 0). | 0190c5b0.c, call 0x190c5b8 | skip |
| 45 | `HALOGEN_DN_AB` | FUN_018ab5c0 | 1 | flag (!=0) | on | DeltaNet alpha/beta path toggle (default on). | 018ab5c0.c, call 0x18ae3e8 | key |
| 46 | `HALOGEN_DN_ATT_SCALE` | FUN_019dfc10 | 1 | int | see site | DeltaNet att scale index ((v+8)>>2 into table). | 019dfc10.c, call 0x19dfc18; analysis/closure/dnc-scan3-r3.md:177 | key |
| 47 | `HALOGEN_DN_ATT_UT` | FUN_019dddc0 | 1 | int | 1 | DeltaNet att UT (default 1). | 019dddc0.c, call 0x19dddc8; analysis/closure/dnc-scan3-r3.md:132 | key |
| 48 | `HALOGEN_DN_F32_STORES` | FUN_018b44f0 | 1 | int | 0 | DeltaNet f32 store variant (default 0). | 018b44f0.c, call 0x18b44f8 | key |
| 49 | `HALOGEN_DN_FUSED` | FUN_018b4580 | 1 | int | 1 | Fused DeltaNet toggle (default 1). | 018b4580.c, call 0x18b4588 | key |
| 50 | `HALOGEN_DN_FUSED_GRAM` | FUN_019e4250 +1 | 2 | flag (==1) | off | Fused Gram toggle (default 0). | 019e4250.c, call 0x19e4a92; analysis/closure/dnftaps-r4.md:75 | key |
| 51 | `HALOGEN_DN_FUSED_NORM` | FUN_018ab5c0 | 1 | flag (!=0) | on | Fused DeltaNet norm toggle. | 018ab5c0.c, call 0x18b0836 (decompile has no literal; disasm only) | key |
| 52 | `HALOGEN_DN_FUSED_PAIR` | FUN_019e4250 +1 | 2 | flag (==1) | off | Fused pair toggle (default 0). | 019e4250.c, call 0x19e4abf; analysis/closure/dnftaps-r4.md:75 | key |
| 53 | `HALOGEN_DN_LDS_BAR` | FUN_019d6370 | 1 | flag (!=0) | on | DeltaNet LDS barrier toggle (default on). | 019d6370.c, call 0x19d639b; analysis/closure/dnc-scan3-r3.md:122 | key |
| 54 | `HALOGEN_DN_M10G` | FUN_019cf4a0 | 1 | int | see site | DeltaNet M10 G selector. | 019cf4a0.c, call 0x19cf4a8; analysis/closure/dnc-scan3-r3.md:69 | key |
| 55 | `HALOGEN_DN_M10P` | FUN_019cf460 | 1 | int | see site | DeltaNet M10 P selector. | 019cf460.c, call 0x19cf468; analysis/closure/dnc-scan3-r3.md:68 | key |
| 56 | `HALOGEN_DN_NORM_FOLD` | FUN_018b4550 | 1 | int | 0 | NormFold mode (default 0). | 018b4550.c, call 0x18b4558; analysis/closure/dnc-scan3-r3.md:57 | key |
| 57 | `HALOGEN_DN_NORM_READ` | FUN_018b4520 | 1 | int | 2 | norm_read mode (default 2). | 018b4520.c, call 0x18b4528; analysis/closure/dnc-scan3-r3.md:23 | key |
| 58 | `HALOGEN_DN_NORM_W` | FUN_018b45b0 | 1 | int | 1 | DeltaNet norm wave/width (default 1). | 018b45b0.c, call 0x18b45b8 | key |
| 59 | `HALOGEN_DN_PREP_W` | FUN_019dc7f0 | 1 | int | 1 | DeltaNet prep width (default 1). | 019dc7f0.c, call 0x19dc7f8 | key |
| 60 | `HALOGEN_DN_PROBE` | FUN_019cf410 | 1 | int | see site | DeltaNet scan3 probe P. | 019cf410.c, call 0x19cf418; analysis/closure/dnc-scan3-r3.md:92 | key |
| 61 | `HALOGEN_DN_ROWS` | FUN_018ab5c0 | 1 | flag (!=0) | on | DeltaNet rows-per-block toggle (default on). | 018ab5c0.c, call 0x18b03af | key |
| 62 | `HALOGEN_DN_S12` | FUN_019dcbe0 | 1 | int | 4 | DeltaNet S12 selector (default 4). | 019dcbe0.c, call 0x19dcbe8; analysis/closure/dnc-scan3-r3.md:163 | key |
| 63 | `HALOGEN_DN_S14` | FUN_019c9fe0 +1 | 2 | int | 34 | DeltaNet S14 selector (default 34). | 019c9fe0.c, call 0x19c9fe8; analysis/closure/dnc-scan3-r3.md:70 | key |
| 64 | `HALOGEN_DN_S22` | FUN_019dcc20 | 1 | int | 1 | DeltaNet S22 (default 1). | 019dcc20.c, call 0x19dcc28; analysis/closure/dnc-scan3-r3.md:132 | key |
| 65 | `HALOGEN_DN_S3A` | FUN_019dc670 | 1 | int | 4 | scan3 A selector (default 4). | 019dc670.c, call 0x19dc678; analysis/closure/dnc-scan3-r3.md:132 | key |
| 66 | `HALOGEN_DN_S7` | FUN_019dc6a0 | 1 | int | 0 | scan3 S7 (default 0). | 019dc6a0.c, call 0x19dc6a8 | key |
| 67 | `HALOGEN_DN_SCAN` | FUN_019ca070 +1 | 2 | int | see site | DeltaNet scan family selector (default 0). | 019ca070.c, call 0x19ca078; analysis/closure/dnc-scan3-r3.md:67 | key |
| 68 | `HALOGEN_DN_SCAN_UB` | FUN_019ddd90 | 1 | int | 1 | DeltaNet scan UB (default 1, min(.,2)). | 019ddd90.c, call 0x19ddd98; analysis/closure/dnc-scan3-r3.md:119 | key |
| 69 | `HALOGEN_DN_STEP_SR` | FUN_0193ef80 | 1 | int | 0 | DeltaNet decode step SR (default 0). | 0193ef80.c, call 0x193ef88; analysis/closure/indexer-attn-host-r2.md:75 | key |
| 70 | `HALOGEN_DN_UT5` | FUN_019d7b70 | 1 | flag (!=0) | on | DeltaNet ut5 on unless "0". | 019d7b70.c, call 0x19dc127; analysis/closure/dnc-scan3-r3.md:132 | key |
| 71 | `HALOGEN_DN_UT_BURST` | FUN_019dde20 | 1 | int | 1 | DeltaNet ut burst (default 1). | 019dde20.c, call 0x19dde28; analysis/closure/dnc-scan3-r3.md:132 | key |
| 72 | `HALOGEN_DN_UT_KL` | FUN_019dde50 | 1 | int | 0 | DeltaNet ut KL (default 0). | 019dde50.c, call 0x19dde58; analysis/closure/dnc-scan3-r3.md:132 | key |
| 73 | `HALOGEN_DN_UT_MP` | FUN_019d7b70 | 1 | flag (==1) | off | DeltaNet ut MP flag (default 0). | 019d7b70.c, call 0x19dc24b; analysis/closure/dnc-scan3-r3.md:132 | key |
| 74 | `HALOGEN_DN_UT_RS` | FUN_019dddf0 | 1 | int | 3 | DeltaNet ut RS (default 3). | 019dddf0.c, call 0x19dddf8; analysis/closure/dnc-scan3-r3.md:132 | key |
| 75 | `HALOGEN_DN_XH` | FUN_019dc6d0 | 1 | int | 1 | scan3 XH (default 1). | 019dc6d0.c, call 0x19dc6d8; analysis/closure/dnc-scan3-r3.md:132 | key |
| 76 | `HALOGEN_DRAFTER_DEFAULT` | FUN_01844d10 | 1 | int | 1 | Drafter id used when GEN names none (default 1). | 01844d10.c, call 0x1844d18 | skip |
| 77 | `HALOGEN_DRAFT_VOCAB` | FUN_01891c50 +2 | 3 | string/other | unset (string) | unset or 1 = subset mode 1, 0 = off, cycle = mode 2. | 01891c50.c, call 0x1892c99; analysis/closure/dv-quant-r4.md:19 | key |
| 78 | `HALOGEN_DRAFT_VOCAB_FILE` | FUN_01891c50 | 1 | string/other | unset (string) | Path of a draft-vocab subset file loaded instead of the built-in one. | 01891c50.c, call 0x1891c87; analysis/closure/dv-quant-r3.md:85 | key |
| 79 | `HALOGEN_DRAFT_VOCAB_Q8` | FUN_01891c50 | 1 | flag (==1) | on? (4) | 1 = 8-bit draft-vocab table; default 4-bit NF4. | 01891c50.c, call 0x1891d6f (decompile has no literal; disasm only); analysis/closure/dv-quant-r4.md:63 | key |
| 80 | `HALOGEN_ENGINE_YIELD_MS` | FUN_019036e0 | 1 | int | 250 | Engine yield interval in ms (default 250). | 019036e0.c, call 0x19036e8 | skip |
| 81 | `HALOGEN_EXP_DN_R2` | FUN_01946870 | 1 | int | 0 | Expert down R2 (default 0). | 01946870.c, call 0x1946878; analysis/closure/expert-dispatch-r5.md:58 | key |
| 82 | `HALOGEN_EXP_LPR` | FUN_018eb360 | 1 | int | 16 | Expert loads per row (default 16). | 018eb360.c, call 0x18eb368; analysis/closure/expert-dispatch-r4.md:150 | key |
| 83 | `HALOGEN_EXP_PF` | FUN_018eb390 | 1 | int | 2 | Expert prefetch depth (default 2). | 018eb390.c, call 0x18eb398; analysis/closure/expert-dispatch-r4.md:171 | key |
| 84 | `HALOGEN_FA_LPT` | FUN_019f1b30 +17 | 18 | int | 1 | Flash-attn LPT (default 1, 18 sites). | 019f1b30.c, call 0x19f1b38; analysis/closure/indexer-attn-host-r2.md:25 | key |
| 85 | `HALOGEN_FA_OPT` | FUN_019eb320 | 1 | int | 5 | Flash-attn OPT & 7 (default 5). | 019eb320.c, call 0x19eb328; analysis/closure/indexer-attn-host-r2.md:25 | key |
| 86 | `HALOGEN_FA_RS` | FUN_019eb2f0 | 1 | int | 1 | Flash-attn RS (default 1). | 019eb2f0.c, call 0x19eb2f8; analysis/closure/indexer-attn-host.md:86 | key |
| 87 | `HALOGEN_FA_S21` | FUN_019eb350 | 1 | int | 1 | Flash-attn S21 special path (default 1). | 019eb350.c, call 0x19eb358; analysis/closure/indexer-attn-host-r2.md:26 | key |
| 88 | `HALOGEN_FA_TQ` | FUN_019e97f0 | 1 | int | see site | Flash-attn TQ option. | 019e97f0.c, call 0x19e985c; analysis/closure/fasel-r3.md:81 | key |
| 89 | `HALOGEN_FA_VB` | FUN_019eb2c0 | 1 | int | 1 | Flash-attn VB (default 1). | 019eb2c0.c, call 0x19eb2c8; analysis/closure/indexer-attn-host.md:86 | key |
| 90 | `HALOGEN_FLASH_BF16_GEMM` | FUN_018a7520 | 1 | flag (!=0) | on | bf16 weight GEMM arm toggle (default on). | 018a7520.c, call 0x18a9457; analysis/closure/dequant-dispatch-r3.md:46 | key |
| 91 | `HALOGEN_FLASH_CONV_TB` | FUN_018b42b0 | 1 | int | 1 | DeltaNet conv TB variant (default 1). | 018b42b0.c, call 0x18b42b8 | key |
| 92 | `HALOGEN_FLASH_CONV_XC` | FUN_018b4280 | 1 | int | 1 | DeltaNet conv XC variant (default 1). | 018b4280.c, call 0x18b4288 | key |
| 93 | `HALOGEN_FLASH_DEQ_V8` | FUN_01945430 | 1 | int | 1 | Dequant v8 instance pick (default 1). | 01945430.c, call 0x1945438; analysis/closure/expert-dispatch-r5.md:134 | key |
| 94 | `HALOGEN_FLASH_EMBED_BF16` | FUN_018897f0 | 1 | string/other | unset (string) | Keep embedding in bf16. [n] | 018897f0.c, call 0x1889937 | key |
| 95 | `HALOGEN_FLASH_MOE_BATCH_GU` | FUN_018e90c0 | 1 | int | 0 | Batched gate_up toggle (default 0). | 018e90c0.c, call 0x18e90c8; analysis/closure/expert-dispatch-r5.md:58 | key |
| 96 | `HALOGEN_FLASH_MOE_FUSED_N` | FUN_018e44d0 +1 | 2 | int | 0 | Token threshold for fused route (default 0). | 018e44d0.c, call 0x18e44d8; analysis/closure/expert-dispatch-r4.md:114 | key |
| 97 | `HALOGEN_FLASH_MOE_GEMM` | FUN_018e8fe0 | 1 | int | 8 | MoE GEMM selector (default 8). | 018e8fe0.c, call 0x18e8fe8; analysis/closure/expert-dispatch-r5.md:58 | key |
| 98 | `HALOGEN_FLASH_MOE_GEMM_DN` | FUN_018e9010 | 1 | int | 8 | MoE down GEMM selector (default 8). | 018e9010.c, call 0x18e9018 | key |
| 99 | `HALOGEN_FLASH_MOE_TIME` | FUN_018be390 +3 | 6 | string/other | unset (string) | 1 prints MoE kernel timing. | 018be390.c, call 0x18c9ef0; analysis/request-trace.md:59 | skip |
| 100 | `HALOGEN_FLASH_MOE_V2` | FUN_018d59c0 +1 | 2 | int | 1 | MoE v2 mode (1 or 2, else 0; default 1). | 018d59c0.c, call 0x18d59c8; analysis/closure/expert-dispatch-r4.md:78 | key |
| 101 | `HALOGEN_FLASH_MOE_V2_BN` | FUN_018e4460 | 1 | int | see site | MoE v2 BN tile. | 018e4460.c, call 0x18e4468; analysis/closure/expert-dispatch-r4.md:114 | key |
| 102 | `HALOGEN_FLASH_MOE_V2_CONC` | FUN_018e44a0 | 1 | int | 0 | MoE v2 concurrency (default 0). | 018e44a0.c, call 0x18e44a8; analysis/closure/expert-dispatch-r4.md:114 | key |
| 103 | `HALOGEN_FLASH_MOE_V2_MINN` | FUN_018e4420 | 1 | int | 64 | Min token count for MoE v2 (default 64). | 018e4420.c, call 0x18e4428; analysis/closure/expert-dispatch-r4.md:114 | key |
| 104 | `HALOGEN_FLASH_MOE_V2_SUM` | FUN_018e8b00 | 1 | int | 1 | MoE v2 sum arm (default 1). | 018e8b00.c, call 0x18e8b08; analysis/closure/expert-dispatch-r5.md:58 | key |
| 105 | `HALOGEN_FLASH_NGRAM_RANDOM` | FUN_0189abb0 | 1 | int | 1 | Test knob: random n-gram. [n] | 0189abb0.c, call 0x189abb8 | key |
| 106 | `HALOGEN_FLASH_PIN_TRUNK` | FUN_01889610 | 1 | flag (!=0) | off | HT weights read in place; must not be 0 (error text), default 0 read as flag. | 01889610.c, call 0x1889695; analysis/closure/dequant-host.md:21 | key |
| 107 | `HALOGEN_FLASH_PREFILL_GEMM` | FUN_01884460 +3 | 5 | flag (!=0) | see site | 0 disables the GEMM path for prefill linears (default on). | 01884460.c, call 0x188716d; analysis/closure/dequant-dispatch-r3.md:38 | key |
| 108 | `HALOGEN_FLASH_ROUTER_W32` | FUN_018897f0 | 1 | int | 0 | Router weight width/wave selector (default 0). | 018897f0.c, call 0x1889810 | key |
| 109 | `HALOGEN_FLASH_ROUTE_GEMM` | FUN_018897f0 +1 | 3 | int | 1 | Routing GEMM selector (default 1). | 018897f0.c, call 0x188983c | key |
| 110 | `HALOGEN_FLASH_ROUTE_M` | FUN_018e4550 +1 | 2 | int | 8 | Route M threshold (default 8). | 018e4550.c, call 0x18e4558 | key |
| 111 | `HALOGEN_FRAG_WARN_BLOCKS` | FUN_01906570 | 1 | string/other | unset (string) | Fragmentation warn threshold in blocks (default 512). | 01906570.c, call 0x1906578 | skip |
| 112 | `HALOGEN_FRAG_WARN_STALLS` | FUN_019065a0 | 1 | string/other | unset (string) | Fragmentation warn stall count (default 1000). | 019065a0.c, call 0x19065a8 | skip |
| 113 | `HALOGEN_FUSE2` | FUN_0193ef50 | 1 | int | 1 | Fuse-2 option (default 1). | 0193ef50.c, call 0x193ef58 | key |
| 114 | `HALOGEN_GDN_HIST` | FUN_018ab5c0 | 1 | flag (!=0) | on | GDN histogram stats toggle. [n] | 018ab5c0.c, call 0x18b06e2 | key |
| 115 | `HALOGEN_GDN_STEP_N` | FUN_018b0e10 +1 | 2 | int | 0 | Token count at which GDN decode switches step kernels (default 0). | 018b0e10.c, call 0x18b0e18; analysis/closure/dnc-scan3-r3.md:185 | key |
| 116 | `HALOGEN_GGUF_CACHE` | FUN_019aa860 | 2 | string/other | unset (string) | 1 or a directory: keep the GGUF repack cache there. | 019aa860.c, call 0x19ab11d (decompile has no literal; disasm only); analysis/entrypoint-flow.md:37 | skip |
| 117 | `HALOGEN_GGUF_THREADS` | FUN_019aa860 +2 | 3 | int | see site | Threads for GGUF repack fill (default 8, clamp 1..64). | 019aa860.c, call 0x19aa9de; analysis/hgn-format.md:172 | skip |
| 118 | `HALOGEN_GRAMMAR` | FUN_0182e9a0 +1 | 2 | flag (!=0) | on | 0 disables grammar-constrained decoding. | 0182e9a0.c, call 0x182e9cb | skip |
| 119 | `HALOGEN_HC6_FUSE` | FUN_0190fcc0 | 1 | int | 2 | HC6 fusion level (default 2). | 0190fcc0.c, call 0x190fcc8; analysis/closure/dequant-host.md:88 | key |
| 120 | `HALOGEN_HC6_FUSE_ROWS` | FUN_0190fd40 | 1 | int | see site | HC6 fuse rows. | 0190fd40.c, call 0x190fd48 | key |
| 121 | `HALOGEN_HC6_KS` | FUN_0190fd00 | 1 | int | see site | HC6 K split. | 0190fd00.c, call 0x190fd08 | key |
| 122 | `HALOGEN_HC8_FUSE` | FUN_0190fbc0 | 1 | int | 2 | HC8 fusion level (default 2). | 0190fbc0.c, call 0x190fbc8; analysis/closure/dequant-host.md:88 | key |
| 123 | `HALOGEN_HC_DI_MERGE` | FUN_0190fd80 | 1 | int | 1 | HC DI merge (default 1). | 0190fd80.c, call 0x190fd88 | key |
| 124 | `HALOGEN_HC_FUSE_ROWS` | FUN_0190fc80 | 1 | int | see site | HC fuse rows. | 0190fc80.c, call 0x190fc88 | key |
| 125 | `HALOGEN_HC_INJ_FUSE` | FUN_0190f630 | 1 | int | 0 | HC inject fusion (default 0). | 0190f630.c, call 0x190f638 | key |
| 126 | `HALOGEN_HC_MIXED_F32_ATTN` | FUN_01911580 | 1 | int | 0 | HC mixed f32 attention (default 0). | 01911580.c, call 0x1911588 | key |
| 127 | `HALOGEN_HC_Q8_DEC` | FUN_0189abe0 +2 | 7 | flag (!=0) | on | HC q8 decode kernel arm toggle (default on). | 0189abe0.c, call 0x18a4557; analysis/closure/dequant-host.md:88 | key |
| 128 | `HALOGEN_HC_Q8_FUSE` | FUN_0189abe0 +1 | 6 | flag (!=0) | on | HC q8 fused arm toggle (default on). | 0189abe0.c, call 0x18a472b; analysis/closure/dequant-host.md:88 | key |
| 129 | `HALOGEN_HC_SIG_FUSE` | FUN_0193d970 | 1 | int | 1 | HC sigmoid fusion (default 1). | 0193d970.c, call 0x193d978 | key |
| 130 | `HALOGEN_HC_UP_CHUNK` | FUN_0193d940 | 1 | int | 0 | HC up chunk (default 0). | 0193d940.c, call 0x193d948 | key |
| 131 | `HALOGEN_HC_UP_W` | FUN_0193d910 | 1 | int | 0 | HC up width (default 0). | 0193d910.c, call 0x193d918 | key |
| 132 | `HALOGEN_HOST_RESERVE_GIB` | FUN_01906c60 | 1 | float | 20.0 | GiB of host RAM kept free (default 20). | 01906c60.c, call 0x1906f70 | skip |
| 133 | `HALOGEN_HT_AB_GRP` | FUN_018ab5c0 +1 | 3 | flag (!=0) | on | HT alpha/beta grouped kernel toggle (default on). | 018ab5c0.c, call 0x18b052d | key |
| 134 | `HALOGEN_HT_DEC_FUSE` | FUN_018be390 +1 | 2 | flag (!=0) | on | HT MoE decode fusion (default on; n < 410, topk < 9). | 018be390.c, call 0x18c9f35; analysis/closure/moe-host.md:39 | key |
| 135 | `HALOGEN_HT_DEC_GROUP` | FUN_01940b50 | 1 | int | 2 | HT decode group (default 2). | 01940b50.c, call 0x1940b58; analysis/closure/dequant-host-r2.md:61 | key |
| 136 | `HALOGEN_HT_DEQ_KFAST` | FUN_018f5480 | 1 | flag (!=0) | on | HT dequant K-fast toggle (default on). | 018f5480.c, call 0x18f5b52; analysis/closure/dv-quant-r4.md:37 | key |
| 137 | `HALOGEN_HT_DEQ_PRE` | FUN_018f5480 | 1 | flag (!=0) | on | HT dequant pre-pass toggle (default on). | 018f5480.c, call 0x18f5b92 | key |
| 138 | `HALOGEN_HT_DEQ_W16` | FUN_018f5480 | 1 | flag (!=0) | on | HT dequant 16-wide toggle (default on). | 018f5480.c, call 0x18f5bd0 | key |
| 139 | `HALOGEN_HT_DN_FOLD` | FUN_018be390 | 1 | flag (==1) | off | HT down-proj fold toggle (default 0). | 018be390.c, call 0x18ca53f; analysis/closure/moe-dv-host-r2.md:36 | key |
| 140 | `HALOGEN_HT_GEMV_KL` | FUN_01940ac0 | 1 | int | 0 | HT GEMV K-loop (default 0). | 01940ac0.c, call 0x1940ac8; analysis/closure/dequant-host-r2.md:61 | key |
| 141 | `HALOGEN_HT_GU_SPLIT` | FUN_01940b00 | 1 | int | see site | HT gate_up split. | 01940b00.c, call 0x1940b08; analysis/closure/dequant-host-r2.md:61 | key |
| 142 | `HALOGEN_HT_LIN_BLOCKS` | FUN_01940a80 | 1 | int | see site | HT linear block count. | 01940a80.c, call 0x1940a88 | key |
| 143 | `HALOGEN_HT_LIN_DIRECT` | FUN_018b0e40 +4 | 6 | flag (!=0) | on | HT linear direct (non-chunked) path toggle (default on). | 018b0e40.c, call 0x18b2868; analysis/closure/dequant-host-r2.md:5 | key |
| 144 | `HALOGEN_HT_LIN_GRP` | FUN_018b0e40 | 1 | flag (!=0) | on | HT linear grouped kernel toggle (default on). | 018b0e40.c, call 0x18b223c; analysis/closure/dequant-host-r2.md:41 | key |
| 145 | `HALOGEN_HT_LIN_PIN` | FUN_01928f40 +3 | 17 | flag (==1) | off | HT linear pinned-weights toggle (flag, 17 sites). | 01928f40.c, call 0x192b2f8; analysis/closure/dequant-host-r2.md:60 | key |
| 146 | `HALOGEN_HT_MOE_DNX` | FUN_018ca730 | 1 | int | 1 | HT MoE down-x mode 0..2 (default 1). | 018ca730.c, call 0x18ca738 | key |
| 147 | `HALOGEN_HT_MOE_DV` | FUN_018ca6f0 | 1 | int | 1 | HT MoE dv mode 0..2 (default 1). | 018ca6f0.c, call 0x18ca6f8 | key |
| 148 | `HALOGEN_HT_MOE_GEMM` | FUN_018ca640 | 1 | int | 1 | HT MoE GEMM arm (default 1). | 018ca640.c, call 0x18ca648; analysis/closure/dequant-host-r2.md:61 | key |
| 149 | `HALOGEN_HT_MOE_M128` | FUN_018ca770 | 1 | int | 3 | HT MoE M128 tile selector (default 3). | 018ca770.c, call 0x18ca778; analysis/closure/dequant-dispatch-r3.md:107 | key |
| 150 | `HALOGEN_HT_MOE_M128_DN` | FUN_018ca7b0 | 1 | int | 4096 | M128 threshold for down proj (default 0x1000 = 4096). | 018ca7b0.c, call 0x18ca7b8 | key |
| 151 | `HALOGEN_HT_MOE_M128_GU` | FUN_018ca7e0 | 1 | int | 16384 | M128 threshold for gate_up (default 0x4000 = 16384). | 018ca7e0.c, call 0x18ca7e8 | key |
| 152 | `HALOGEN_HT_MOE_SUM` | FUN_018be390 | 1 | flag (!=0) | on | HT MoE sum arm toggle (default on). | 018be390.c, call 0x18c9e3a | key |
| 153 | `HALOGEN_HT_MOE_V2` | FUN_018ca6b0 | 1 | int | 1 | HT MoE v2 mode 0..3 (default 1). | 018ca6b0.c, call 0x18ca6b8; analysis/closure/expert-dispatch-r5.md:17 | key |
| 154 | `HALOGEN_HT_MOE_WPE` | FUN_018ca670 | 1 | int | 8 | HT MoE waves per expert (default 8). | 018ca670.c, call 0x18ca678 | key |
| 155 | `HALOGEN_HT_TG_BLOCKS` | FUN_0192c150 | 1 | int | see site | HT trunk GEMM blocks (default 80, valid 1..4096). | 0192c150.c, call 0x192c158; analysis/closure/moe-host.md:32 | key |
| 156 | `HALOGEN_HT_TG_DEC` | FUN_0192bc30 | 1 | int | see site | Min M for the HT trunk GEMM on the decode side (clamp 1..8). | 0192bc30.c, call 0x192bc38; analysis/closure/dequant-dispatch-r3.md:53 | key |
| 157 | `HALOGEN_HT_TG_M128` | FUN_0192c190 | 1 | int | 256 | HT trunk GEMM M128 tile (default 0x100 = 256). | 0192c190.c, call 0x192c198; analysis/closure/dequant-host-r2.md:61 | key |
| 158 | `HALOGEN_HT_TRUNK_GEMM` | FUN_019330f0 | 1 | int | 192 | HT trunk GEMM mode (default 0xc0 = 192). | 019330f0.c, call 0x19330f8; analysis/closure/dequant-dispatch-r3.md:53 | key |
| 159 | `HALOGEN_HT_TRUNK_ORIG` | FUN_01884460 +1 | 2 | string/other | unset (string) | Keeps original (unpacked) HT trunk layout. [n] | 01884460.c, call 0x1887249; analysis/closure/dequant-dispatch-r3.md:53 | key |
| 160 | `HALOGEN_HT_TRUNK_RESIDENT` | FUN_01884460 | 1 | string/other | unset (string) | Old name of PREFILL_KEEP_TRUNK; setting it prints a rename error. | 01884460.c, call 0x18871e7 (decompile has no literal; disasm only) | key |
| 161 | `HALOGEN_HT_U24` | FUN_01928f40 +3 | 17 | flag (==1) | off | HT linear u24 variant (flag, 17 sites). | 01928f40.c, call 0x192b4e0; analysis/closure/dequant-host-r2.md:60 | key |
| 162 | `HALOGEN_I4R_DEC_GROUP` | FUN_01940b90 | 1 | int | see site | i4r decode group. | 01940b90.c, call 0x1940b98 | key |
| 163 | `HALOGEN_I4R_DN_FOLD` | FUN_018d2430 | 1 | flag (!=0) | on | i4r down fold toggle (default on). | 018d2430.c, call 0x18d5882 (decompile has no literal; disasm only); analysis/closure/expert-dispatch-r4.md:78 | key |
| 164 | `HALOGEN_I4R_FUSE` | FUN_018d2430 | 1 | flag (==1) | off | i4r fused decode arm (default 0). | 018d2430.c, call 0x18d584e; analysis/closure/expert-dispatch-r4.md:65 | key |
| 165 | `HALOGEN_I4R_HFUSE` | FUN_018e4580 +1 | 3 | flag (!=0) | on | i4r H-fuse toggle (default on). | 018e4580.c, call 0x18e5323 | key |
| 166 | `HALOGEN_I4R_LAST` | FUN_018d2430 +2 | 3 | flag (!=0) | on | i4r last-stage toggle (default on). | 018d2430.c, call 0x18d58fb; analysis/closure/expert-dispatch-r4.md:65 | key |
| 167 | `HALOGEN_I4R_SDFUSE` | FUN_01940f60 | 1 | flag (!=0) | on | i4r shared-down fusion (default on). | 01940f60.c, call 0x1940f8b | key |
| 168 | `HALOGEN_I4R_SORTED` | FUN_01940bd0 | 1 | int | 3 | i4r sorted dispatch mode (default 3). | 01940bd0.c, call 0x1940bd8; analysis/closure/dequant-dispatch-r3.md:107 | key |
| 169 | `HALOGEN_IMAGE_VERSION` | FUN_0182ba00 | 1 | string/other | unset (string) | Image version string, parsed at start and printed in the banner. | 0182ba00.c, call 0x182d2e0; analysis/entrypoint-flow.md:14 | skip |
| 170 | `HALOGEN_INDEXER_BUDGET` | FUN_01853fb0 | 1 | int | 0 | Indexer selection budget (ids per row); 0 = default. | 01853fb0.c, call 0x1853fc8; analysis/closure/fasel-r3.md:60 | key |
| 171 | `HALOGEN_IX_SELECT` | FUN_018b5a30 | 1 | int | 1 | Indexer selection mode (default 1). | 018b5a30.c, call 0x18b5a38; analysis/closure/indexer-attn-host.md:12 | key |
| 172 | `HALOGEN_IX_WMMA` | FUN_018b8500 | 1 | int | 3 | Indexer WMMA mode (default 3). | 018b8500.c, call 0x18b8508; analysis/closure/indexer-attn-host.md:12 | key |
| 173 | `HALOGEN_IX_WMMA_POISON` | FUN_018b8530 | 1 | string/other | unset (string) | Debug: poison the WMMA indexer output. | 018b8530.c, call 0x18b8538; analysis/kernels-indexer.md:32 | key |
| 174 | `HALOGEN_KV_POOL` | FUN_0182ba00 | 1 | flag (!=0) | on | 0 = per-slot KV (26624 B/pos/slot) instead of one shared pool. | 0182ba00.c, call 0x182bca6; analysis/entrypoint-flow.md:88 | skip |
| 175 | `HALOGEN_KV_POOL_FIT` | FUN_01906c60 | 1 | flag (!=0) | on | Pool fit test/override read with the pool sizing (error text names it); 0-flag. [n] | 01906c60.c, call 0x1906c92 | skip |
| 176 | `HALOGEN_KV_POOL_POSITIONS` | FUN_0182ba00 | 2 | int | 0 | Total positions in the shared KV pool; unset derives from ctx. | 0182ba00.c, call 0x182bae4; analysis/entrypoint-flow.md:78 | skip |
| 177 | `HALOGEN_KV_SLOTS` | FUN_0182ba00 | 1 | int | 4 | KV slot count (entrypoint passes --slots); default 4. | 0182ba00.c, call 0x182babd; analysis/entrypoint-flow.md:76 | skip |
| 178 | `HALOGEN_LBF_WAVE` | FUN_0193d8b0 | 1 | int | 1 | bf16 linear wave variant (default 1). | 0193d8b0.c, call 0x193d8b8 | key |
| 179 | `HALOGEN_LIN_M` | FUN_0193cb20 | 1 | int | see site | Linear M threshold. | 0193cb20.c, call 0x193cb28 | key |
| 180 | `HALOGEN_LIN_NARROW` | FUN_0193d8e0 | 1 | int | 0 | Narrow linear variant (default 0). | 0193d8e0.c, call 0x193d8e8 | key |
| 181 | `HALOGEN_LIN_U` | FUN_0193cb70 | 1 | int | 4 | Linear unroll (default 4). | 0193cb70.c, call 0x193cb78 | key |
| 182 | `HALOGEN_LIN_WAVE` | FUN_0193ef20 | 1 | int | 1 | 0 disables wave linear arm (default 1). | 0193ef20.c, call 0x193ef28; analysis/closure/dequant-dispatch-r3.md:23 | key |
| 183 | `HALOGEN_LQ4W_SH` | FUN_019406c0 | 1 | int | 0 | q4 wave shared-memory variant F (default 0). | 019406c0.c, call 0x19406c8; analysis/closure/dequant-dispatch-r3.md:90 | key |
| 184 | `HALOGEN_LQ8_WAVE` | FUN_0190fc50 | 1 | int | 1 | 0 selects legacy q8s32 linear (default 1). | 0190fc50.c, call 0x190fc58; analysis/closure/dequant-dispatch-r3.md:50 | key |
| 185 | `HALOGEN_MATMUL_ALGOS` | FUN_019e7500 | 1 | int | 1 | hipBLASLt algo count requested (default 1). | 019e7500.c, call 0x19e7508; analysis/closure/expert-dispatch-r5b.md:78 | key |
| 186 | `HALOGEN_MATMUL_BUCKET` | FUN_019e7540 | 1 | int | 1 | hipBLASLt M-bucket rounding (default 1). | 019e7540.c, call 0x19e7548; analysis/closure/expert-dispatch-r5b.md:78 | key |
| 187 | `HALOGEN_MATMUL_STATS` | FUN_019e6210 | 1 | int | 0 | hipBLASLt matmul stats print (default 0). | 019e6210.c, call 0x19e6218; analysis/closure/expert-dispatch-r5b.md:78 | skip |
| 188 | `HALOGEN_MATMUL_TUNING_FILE` | FUN_01854310 +1 | 2 | string/other | unset (string) | hipBLASLt tuning file path; also hashed into the cache key. | 01854310.c, call 0x1854601; analysis/closure/cache-host.md:70 | key |
| 189 | `HALOGEN_MOE_CSORT` | FUN_018e8b60 | 1 | int | 1 | Count-sort arm (default 1). | 018e8b60.c, call 0x18e8b68; analysis/closure/expert-dispatch-r5.md:58 | key |
| 190 | `HALOGEN_MOE_DN_ROWS` | FUN_01884460 +1 | 2 | int | see site | Rows per block for MoE down projection. | 01884460.c, call 0x1885baa; analysis/closure/expert-dispatch-r4.md:118 | key |
| 191 | `HALOGEN_MOE_GPU_ROUTE` | FUN_018e8b30 | 1 | int | 1 | GPU routing arm (default 1). | 018e8b30.c, call 0x18e8b38; analysis/closure/expert-dispatch-r4.md:150 | key |
| 192 | `HALOGEN_MOE_H_ORDER` | FUN_018e9040 | 1 | int | 1 | MoE H order option (default 1). | 018e9040.c, call 0x18e9048; analysis/closure/expert-dispatch-r5.md:58 | key |
| 193 | `HALOGEN_MOE_SIZE_HIST` | FUN_018e8cc0 | 1 | int | 0 | Print MoE expert-size histogram (default 0). | 018e8cc0.c, call 0x18e8cc8 | skip |
| 194 | `HALOGEN_MTP` | FUN_0188f710 | 1 | flag (!=0) | on | 0 disables the MTP head (default 1). | 0188f710.c, call 0x188f72b; analysis/closure/mtp-head.md:51 | key |
| 195 | `HALOGEN_MTP_DEPTH` | FUN_0182ba00 | 2 | int | 2 | MTP draft depth (default 2). | 0182ba00.c, call 0x182db0e; analysis/report.md:31 | key |
| 196 | `HALOGEN_MTP_DEPTH_HI` | FUN_0182ba00 | 2 | int | 0 | MTP draft depth for the high-acceptance state (0 = unused). [n] | 0182ba00.c, call 0x182db69 | key |
| 197 | `HALOGEN_MTP_HEAD` | FUN_0182ba00 +2 | 3 | string/other | unset (string) | Path of the MTP head file; also hashed into the disk-cache key. | 0182ba00.c, call 0x182ca6e; analysis/closure/cache-host.md:70 | key |
| 198 | `HALOGEN_MTP_PREFILL` | FUN_018fa150 | 1 | flag (!=0) | on | MTP head prefill toggle (default on). | 018fa150.c, call 0x18fc059 (decompile has no literal; disasm only) | key |
| 199 | `HALOGEN_MTP_WIRE` | FUN_018f34c0 | 1 | string/other | unset (string) | MTP head wire/stream option (default 0x44 read; see mtp-head.md). | 018f34c0.c, call 0x18f4331; analysis/closure/mtp-head.md:62 | skip |
| 200 | `HALOGEN_NGRAM_EARLY` | FUN_01903760 +3 | 4 | flag (!=0) | see site | Early n-gram table startup toggle. | 01903760.c, call 0x190378b | skip |
| 201 | `HALOGEN_NGRAM_EARLY_CHECK` | FUN_018ede30 +3 | 4 | flag (==1) | off | Early n-gram check toggle (default 0). | 018ede30.c, call 0x18f0b04 | skip |
| 202 | `HALOGEN_NGRAM_GATHER_THREADS` | FUN_01903710 | 1 | int | 64 | Threads for n-gram table gather (default 64). | 01903710.c, call 0x1903718; analysis/closure/q5-consumer-r5.md:89 | skip |
| 203 | `HALOGEN_NGRAM_TABLE` | FUN_01884460 | 1 | string/other | unset (string) | Path/selector of the n-gram embedding table. | 01884460.c, call 0x18853b6; analysis/closure/hgn-href.md:64 | key |
| 204 | `HALOGEN_NGRAM_WARM` | FUN_01903eb0 +1 | 2 | flag (!=0) | see site | Warm the n-gram table at start. | 01903eb0.c, call 0x1903edb | skip |
| 205 | `HALOGEN_PLD` | FUN_0182ba00 | 1 | sscanf list | unset (list) | "N,K": prompt-lookup-decode n-gram length and draft count; 0 = off. | 0182ba00.c, call 0x182d401; analysis/closure/pld.md:14 | skip |
| 206 | `HALOGEN_PLD_ADAPT` | FUN_0182ba00 | 1 | sscanf list | unset (list) | "W,floor,R": PLD adaptive window, floor and recovery. | 0182ba00.c, call 0x182d43b; analysis/closure/pld.md:15 | skip |
| 207 | `HALOGEN_PLD_AGREE` | FUN_0182ba00 | 1 | int | see site | PLD agreement count; nonzero also feeds the MTP head window. | 0182ba00.c, call 0x182d4a4; analysis/closure/pld.md:16 | skip |
| 208 | `HALOGEN_PLD_LOOKBACK` | FUN_0182ba00 | 1 | int | see site | PLD lookback window length. | 0182ba00.c, call 0x182d47d; analysis/closure/pld.md:16 | skip |
| 209 | `HALOGEN_PLE_PAR` | FUN_018ede30 | 4 | flag (!=0) | see site | PLE parallel toggle. [n] | 018ede30.c, call 0x18efe06 | key |
| 210 | `HALOGEN_POOL_WAIT_DROP` | FUN_0182ea80 | 1 | flag (!=0) | on | 0 disables dropping a request that waits on the KV pool. [n] | 0182ea80.c, call 0x183ba82 | key |
| 211 | `HALOGEN_PREFILL_CANCEL` | FUN_0182ba00 | 1 | flag (!=0) | off | Flag; 0 turns off cancelling a request during its prefill. [n] | 0182ba00.c, call 0x182dd75; analysis/rebuild-guide.md:99 | key |
| 212 | `HALOGEN_PREFILL_CHUNK` | FUN_01905050 | 1 | int | 0 | Engine prefill chunk size (banner: 32768). | 01905050.c, call 0x1905058; analysis/closure/cache-host.md:47 | key |
| 213 | `HALOGEN_PREFILL_KEEP_TRUNK` | FUN_0182ba00 +3 | 5 | flag (==1) | off | 1 keeps the HT trunk unpacked and resident through prefill (HT_TRUNK_RESIDENT is the renamed old name). | 0182ba00.c, call 0x182bff2 | key |
| 214 | `HALOGEN_PROMPT_CACHE` | FUN_01903680 | 1 | string/other | unset (string) | 0, 1 or 2 prompt-cache mode (exits on other values). | 01903680.c, call 0x1903688; analysis/closure/cache-host.md:33 | skip |
| 215 | `HALOGEN_QK_PREP` | FUN_018b86d0 | 1 | flag (!=0) | on | QK prep kernel toggle (default on). | 018b86d0.c, call 0x18bcaa7 | key |
| 216 | `HALOGEN_QSA_DENSE` | FUN_018b86d0 | 1 | flag (==1) | off | 1 forces dense (non-sparse) qs attention. | 018b86d0.c, call 0x18bc9c0; analysis/closure/indexer-attn-host-r2.md:53 | key |
| 217 | `HALOGEN_QSA_FORCE_SEL` | FUN_01884460 +3 | 4 | flag (==1) | off | 1 forces the sparse selection path in attention. | 01884460.c, call 0x1885906; analysis/closure/indexer-attn-host-r2.md:53 | key |
| 218 | `HALOGEN_QSA_NOSKIP` | FUN_018b86d0 | 1 | flag (==1) | off | 1 disables skipping in qs attention. | 018b86d0.c, call 0x18bcdc5 (decompile has no literal; disasm only) | key |
| 219 | `HALOGEN_QSA_STRIP_MB` | FUN_01884460 | 1 | int | 268435456 | Strip size in bytes for QSA scratch (default 0x10000000 = 268435456). | 01884460.c, call 0x18859ec | key |
| 220 | `HALOGEN_RESERVE_TICK_S` | FUN_01884460 | 1 | int | 30 | Seconds between memory-reserve checks (default 30). | 01884460.c, call 0x1885803 | skip |
| 221 | `HALOGEN_ROPE_W` | FUN_018bda30 +1 | 2 | int | 1 | RoPE kernel wave/width (default 1). | 018bda30.c, call 0x18bda38 | key |
| 222 | `HALOGEN_ROPE_YARN` | FUN_0190d070 | 1 | float | see site | YaRN factor for context past 262144 (float). | 0190d070.c, call 0x190d081; analysis/entrypoint-flow.md:13 | key |
| 223 | `HALOGEN_ROUTE_DEC_M` | FUN_01942d20 | 1 | int | see site | Route decode M threshold. | 01942d20.c, call 0x1942d28 | key |
| 224 | `HALOGEN_ROUTE_HL` | FUN_0190f600 | 1 | int | 1 | Router high/low variant (default 1). | 0190f600.c, call 0x190f608 | key |
| 225 | `HALOGEN_ROUTE_RIDE` | FUN_0193efb0 | 1 | flag (!=0) | on | Route ride-along toggle (default on). | 0193efb0.c, call 0x193efdb | key |
| 226 | `HALOGEN_ROUTE_ROWS` | FUN_018d7510 | 2 | flag (!=0) | on | Router rows toggle (default on). | 018d7510.c, call 0x18e326b | key |
| 227 | `HALOGEN_ROUTE_SG` | FUN_018d7510 +2 | 3 | flag (!=0) | on | Router subgroup toggle (default on). | 018d7510.c, call 0x18e37c3 | key |
| 228 | `HALOGEN_ROUTE_TOPK_W` | FUN_018e8530 | 1 | int | 1 | Router top-k wave variant (default 1). | 018e8530.c, call 0x18e8538 | key |
| 229 | `HALOGEN_ROW_GEMV` | FUN_0193caf0 | 1 | int | 0 | Legacy multi-token q4 row GEMV (default 0). | 0193caf0.c, call 0x193caf8; analysis/closure/dequant-dispatch-r3.md:28 | key |
| 230 | `HALOGEN_ROW_GEMV_M` | FUN_0193cbc0 | 1 | int | see site | Row GEMV M threshold. | 0193cbc0.c, call 0x193cbc8 | key |
| 231 | `HALOGEN_ROW_GEMV_U` | FUN_0193cc10 | 1 | int | 1 | Row GEMV unroll (default 1). | 0193cc10.c, call 0x193cc18 | key |
| 232 | `HALOGEN_S239_EAGER_ERASE` | FUN_0182ea80 | 1 | flag (==1) | off | 1 erases cache entries eagerly. [n] | 0182ea80.c, call 0x183e967 | key |
| 233 | `HALOGEN_S2_NUCLEUS` | FUN_01909da0 | 1 | float | see site | Nucleus term for draft_sample (float). | 01909da0.c, call 0x1909da8; analysis/closure/ngram-dv-host.md:155 | key |
| 234 | `HALOGEN_S2_QOFF` | FUN_01909dd0 | 1 | float | see site | Q offset for draft_sample (float). | 01909dd0.c, call 0x1909dd8; analysis/closure/sampling-host.md:62 | key |
| 235 | `HALOGEN_S2_TSKEW` | FUN_01909e00 | 1 | float | 1.0 | Temperature skew for draft_sample (default 1.0). | 01909e00.c, call 0x1909e08; analysis/closure/ngram-dv-host.md:154 | key |
| 236 | `HALOGEN_S3_FALSIFY` | FUN_0194bd90 | 1 | int | 0 | Mode argument of the S3 falsify kernel (default 0). | 0194bd90.c, call 0x194bd98; analysis/closure/sampler-seed-r3.md:104 | key |
| 237 | `HALOGEN_SAMPLED_DEPTH` | FUN_0182ba00 | 1 | int | see site | Draft depth for sampled speculation; must be >= 1 and below SPEC_N. | 0182ba00.c, call 0x182dc02; analysis/closure/sampling-host.md:64 | key |
| 238 | `HALOGEN_SAMPLED_GATE` | FUN_01853890 | 1 | sscanf list | unset (list) | "t1[,t2]" thresholds in [0,1] gating sampled speculation. | 01853890.c, call 0x18538b2; analysis/closure/sampling-host.md:66 | key |
| 239 | `HALOGEN_SAMPLE_BLOCKS` | FUN_0194bd50 | 1 | int | 64 | Sampler kernel blocks (default 64, clamp 0..256). | 0194bd50.c, call 0x194bd58; analysis/closure/sampling-host-r2.md:14 | skip |
| 240 | `HALOGEN_SCHEMA_ESCAPE` | FUN_0184dcf0 | 1 | string/other | unset (string) | String flag read in the JSON-schema grammar builder (escape handling). [n] | 0184dcf0.c, call 0x184dd00; analysis/entrypoint-flow.md:62 | key |
| 241 | `HALOGEN_SERVE_TRACE` | FUN_0182ea80 +5 | 20 | flag (==1) | off | 1 prints per-event serve trace lines (accept, pending fds, timing). | 0182ea80.c, call 0x1832002; analysis/closure/serve-pending-fd-r4.md:53 | skip |
| 242 | `HALOGEN_SPEC_ADAPT` | FUN_0182ba00 | 1 | sscanf list | unset (list) | Adaptive window triple for head speculation. | 0182ba00.c, call 0x182da78; analysis/closure/pld.md:185 | skip |
| 243 | `HALOGEN_SPEC_N` | FUN_01884460 +1 | 2 | int | 4 | Max speculative tokens per round (default 4). | 01884460.c, call 0x1887c3c; analysis/closure/mtp-head.md:79 | skip |
| 244 | `HALOGEN_STARTUP_PROGRESS` | FUN_01889430 | 1 | flag (!=0) | see site | 0 silences startup progress lines (default on). | 01889430.c, call 0x188957f | skip |
| 245 | `HALOGEN_TAP_FROM_SCAN` | FUN_018951f0 +1 | 4 | flag (!=0) | on | Tap taken from the scan kernel (default 1). | 018951f0.c, call 0x189521b; analysis/closure/dnc-scan3-r4.md:28 | skip |
| 246 | `HALOGEN_TEST_DEFER_REQ` | FUN_0184f2c0 | 1 | string/other | unset (string) | Test hook: defer a request. Default 0. [n] | 0184f2c0.c, call 0x184f2c8 | key |
| 247 | `HALOGEN_UMA_CARVEOUT_GIB` | FUN_019067d0 | 1 | float | see site | Override the BIOS UMA carve-out hint in GiB. | 019067d0.c, call 0x1906a2c; analysis/entrypoint-flow.md:91 | skip |
| 248 | `HALOGEN_VERBOSE` | FUN_019060f0 | 1 | flag (!=0) | see site | Engine verbose logging (entrypoint exports 0). | 019060f0.c, call 0x190611b; analysis/entrypoint-flow.md:10 | skip |
| 249 | `HALOGEN_VISION_ATTN` | FUN_01884460 | 1 | int | 3 | Vision attention variant (default 3). | 01884460.c, call 0x1887570 (decompile has no literal; disasm only) | key |
| 250 | `HALOGEN_VISION_FA_HILO` | FUN_0194bba0 | 1 | int | 1 | Vision flash-attention hi/lo (default 1). | 0194bba0.c, call 0x194bba8 | key |
| 251 | `HALOGEN_VISION_FA_WAVES` | FUN_0194bb60 | 1 | int | see site | Vision flash-attention wave count. | 0194bb60.c, call 0x194bb68 | key |
| 252 | `HALOGEN_VISION_MAX_PIXELS` | FUN_0184c7a0 +1 | 2 | int | see site | Max pixels for a VIMG image; larger is rejected. | 0184c7a0.c, call 0x184c7a8; analysis/entrypoint-flow.md:90 | key |
| 253 | `HALOGEN_VISION_TIMING` | FUN_01947be0 | 1 | flag (==1) | off | 1 prints vision tower timing. | 01947be0.c, call 0x194a31e | skip |
| 254 | `HALOGEN_VISION_TOWER` | FUN_01854310 +2 | 3 | string/other | unset (string) | Vision tower file path; also hashed into the disk-cache key. | 01854310.c, call 0x18545de; analysis/closure/cache-host.md:70 | key |
| 255 | `HALOGEN_WALK_FALSIFY` | FUN_01883020 | 1 | int | 0 | Test mode: perturbs RNG counter and residual-sample inputs (default 0). | 01883020.c, call 0x1883028; analysis/closure/sampling-host.md:50 | key |
| 256 | `HALOGEN_WEIGHTS_LOCK` | FUN_01996d90 +2 | 3 | flag (==1) | see site | 1 asks the engine to mlock weights. | 01996d90.c, call 0x19977a8; analysis/entrypoint-flow.md:103 | key |

## Strings with no reader in the binary

Nine standalone `HALOGEN_` strings exist in `.rodata` and in the cache-key skip table (below) but no `lea` or `getenv` call site references them. They are names the engine is told to ignore when it hashes the environment; whatever sets them is outside `flash_serve` (the entrypoint or a stale name).

| Variable | Evidence |
|---|---|
| `HALOGEN_CACHE_ALIGN` | in skip table slot 3 (0x19f60e0 + 8*i); no code xref |
| `HALOGEN_CACHE_MB` | in skip table slot 10 (0x19f60e0 + 8*i); no code xref |
| `HALOGEN_CACHE_RESERVE_MB` | in skip table slot 12 (0x19f60e0 + 8*i); no code xref |
| `HALOGEN_DRAFTER` | in skip table slot 19 (0x19f60e0 + 8*i); no code xref |
| `HALOGEN_QSA_SEL_STATS` | in skip table slot 47 (0x19f60e0 + 8*i); no code xref |
| `HALOGEN_SERIAL_DUMP` | in skip table slot 50 (0x19f60e0 + 8*i); no code xref |
| `HALOGEN_SERIAL_REF` | in skip table slot 51 (0x19f60e0 + 8*i); no code xref |
| `HALOGEN_SLOT` | in skip table slot 53 (0x19f60e0 + 8*i); no code xref |
| `HALOGEN_SLOT_CTX` | in skip table slot 54 (0x19f60e0 + 8*i); no code xref |

## Not read by `flash_serve` (set elsewhere)

`image/entrypoint.sh` reads and exports `HALOGEN_CTX`, `HALOGEN_MAX_TOK`, `HALOGEN_PORT`, `HALOGEN_API_PORT`, `HALOGEN_BIND`, `HALOGEN_CHECKPOINT`, `HALOGEN_ROPE_YARN` (also read by the engine, table above) and passes them as argv. `HALOGEN_CTX`, `HALOGEN_MAX_TOK` and `HALOGEN_DOWNLOAD` appear in the binary only inside message text (error and EULA strings), never as `getenv` operands. `image/serve_api.py` reads its own set (`HALOGEN_EOS_GUARD`, `HALOGEN_THINKING_BUDGET`, `HALOGEN_SCHEMA_ESCAPE` and others); see `analysis/entrypoint-flow.md`.
