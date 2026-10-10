# Unknowns ledger (Phase 6)

Plain markdown, not the REA ledger. `rea build-reconstruction-obligation-ledger`
(rea-agents 6.3.0) requires `evidence_bundle`, `manifest` and
`reviewed_obligations` objects built from authenticated application Evidence
(checked 2026-10-10: calling it with `{}` returns those three missing
arguments). This workspace's evidence is hand-made disassembly and decompile
output, so those inputs cannot be produced honestly. Each item names its
source writeup and what would close it. Status: OPEN-BOUNDED means the
search is stated and the remaining gap needs host-side decompile, a running
engine, or data not in this workspace.

| id | unknown | source | closes with | status |
|---|---|---|---|---|
| U01 | HNPB header u32 at +0x08 and per-blob u32 pair at +0x50/+0xa0 | npu-engine.md s6 | writer-side decompile (`convert`, 0x1a8608 region) | OPEN-BOUNDED |
| U02 | per-verb argument grammars of the NPU serve protocol | npu-engine.md s6 | annotate `work/npu/0x12b110.pseudo.c` per verb | OPEN-BOUNDED |
| U03 | NPU `main:<graph>` name set | npu-engine.md s6 | analyze the shipped `.elf` graph files | OPEN-BOUNDED |
| U04 | engine behavior for raw `GEN ... 2` (dflash2 id) | dflash2-probe.md s5 | CLOSED 2026-10-10: engine refuses with `error ... asked for drafter 2 (unknown)` (analysis/runtime-run-0.17.3.md) | CLOSED |
| U05 | future flip of the dflash2 format-string literal | dflash2-probe.md s5 | unobservable from pulled set | UNOBSERVABLE |
| U06 | 0.14.x and 0.15.x patch-release deltas | version-deltas.md | pull the missing tags | OPEN-BOUNDED |
| U07 | `HALOGEN_WALK_FALSIFY` effect on acceptance math | version-deltas.md | device side answered (no device change); host feeding untraced | PARTIAL, see U20 |
| U08 | HREF record layout in `.hgn` | hgn-format.md | Ghidra pass on the two call sites (0x1725112, 0x1725afc) | OPEN-BOUNDED |
| U09 | `HtItem` byte layout (stride 72 B known) | kernels-moe.md s9 | byte-level decode of `k_ht_items_group` | OPEN-BOUNDED |
| U10 | `k_ht_tg` meaning; grouped GEMV variants not deep-dived | kernels-moe.md s9 | targeted read | OPEN-BOUNDED |
| U11 | `DnfTaps` tap count per template arg 0..12; per-format bias table | kernels-moe.md s9 | enumerate per instantiation | OPEN-BOUNDED |
| U12 | `scan3` template args, `DncTapArgs`/`DncNormFold` layouts, `ut4/ut5` flags, decode-vs-prefill att variant, `dnc` name | kernels-indexer.md s5 | host dispatch decompile | OPEN-BOUNDED |
| U13 | host half of "resume is bitwise": no prefill redo for cached rows | kernels-attention.md | dispatch-site pass over `main_pseudo.c`/`gf_*.c` | OPEN-BOUNDED |
| U14 | fa2/qs template args 4-6, `FaSel` layout, legacy `k_attn`/`k_attn_fa` dispatch | kernels-attention.md | host dispatch decompile | OPEN-BOUNDED |
| U15 | `SampleRow` field map (offset 0xc uniform, offset 0 temperature are inferred) | kernels-speculation.md | host writer decompile | OPEN-BOUNDED |
| U16 | which `k_smpg_accept_final` mode value is greedy | kernels-speculation.md | trace mode argument at host call site | OPEN-BOUNDED |
| U17 | per-lane n-gram order in `k_ngram_ids` | kernels-speculation.md | host caller decompile | OPEN-BOUNDED |
| U18 | `k_dv_quant*` block layout and argument roles; consumer of the int8 q rows | kernels-speculation.md s4 | host dispatch decompile | OPEN-BOUNDED |
| U19 | `v_sad_u8` codebook decode in `k_ht_*`, `k_i4r_*_gl`; HT bit-width template arg 3/4 meaning; `su`/`sv` signs vs scales | kernels-dequant.md s3 | reverse the decode with a real HT checkpoint (none here) | OPEN-BOUNDED |
| U20 | host feeding of acceptance inputs under falsify mode | kernels-speculation.md s4b | host decompile | OPEN-BOUNDED |
| U21 | dtype-to-kernel dispatch table; `lq*` template args; `hc*_fused` u64 output; power-of-two group scale source | kernels-dequant.md s5 | host dispatch decompile | OPEN-BOUNDED |
| U22 | AMDGCN pseudocode (Ghidra has no shader-ISA module) | ghidra-module-writeup.md | a community AMDGCN processor module, if one appears | OPEN-BOUNDED |
| U23 | seed and counter source of the splitmix64 draw in `k_smpg_pick`/`k_smpg_rpick`; what `s15` in the accept kernel holds | kernels-speculation.md | host caller decompile plus an engine run with a fixed seed | OPEN-BOUNDED |
| U24 | purpose of the float-key radix select in `k_accept_prob_f32`, `k_sample_f32`, `k_resid_sample_f32` | kernels-speculation.md | trace what the digit `v_bfe_u32 ,16,8` indexes | OPEN-BOUNDED |
| U25 | what `k_dv_quant` quantizes to (compare ladder, not round-and-clamp) | kernels-speculation.md s4 | decode the compare constants against a codebook | OPEN-BOUNDED |
| U26 | raw `GEN` and `CSTAT` unanswered on a second connection | runtime-run-0.17.3.md | CLOSED 2026-10-10: only the first connection gets request verbs; second connections get INFO and PING | CLOSED |

Totals: 26 items, 2 closed by the 2026-10-10 engine run (U04, U26). 3 findings were corrected after an independent review
(in-kernel RNG exists, `k_dv_scatter` is a gather, `k_dv_quant` is a different
quantizer); see `kernels-speculation.md`. Closable by decompile: U01-U03,
U08-U18, U20, U21, U23-U25; needing a live run: U19 (U04 and U26 are closed); named limits by
decision: U05, U06, U22; U07 partial. Live evidence already gathered
(2026-10-10, Ai-core, engine 0.17.2): INFO line field 7 (`dflash2`) is 0,
observed (`analysis/evidence/live-0.17.2-startup-and-info.txt`); the 0.17.3 engine run adds `analysis/runtime-run-0.17.3.md`. U26 is closed for CSTAT on 0.17.3 and for GEN on live 0.17.2 only (see that doc for the exact scope).
