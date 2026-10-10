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
| U01 | HNPB header u32 at +0x08 and per-blob u32 pair at +0x50/+0xa0 | npu-engine.md s6 | PARTIAL: header field and per-blob pair semantics worked out in part, `npu-engine.md` s2.2 corrected (analysis/closure/npu-closure.md) | PARTIAL |
| U02 | per-verb argument grammars of the NPU serve protocol | npu-engine.md s6 | PARTIAL: framing and most verbs closed; some numeric field meanings open (analysis/closure/npu-closure.md) | PARTIAL |
| U03 | NPU `main:<graph>` name set | npu-engine.md s6 | graph names live in shipped `.elf` files not in this workspace; needs those files (analysis/closure/npu-closure.md) | OPEN-BOUNDED |
| U04 | engine behavior for raw `GEN ... 2` (dflash2 id) | dflash2-probe.md s5 | CLOSED 2026-10-10: engine refuses with `error ... asked for drafter 2 (unknown)` (analysis/runtime-run-0.17.3.md) | CLOSED |
| U05 | future flip of the dflash2 format-string literal | dflash2-probe.md s5 | unobservable from pulled set | UNOBSERVABLE |
| U06 | 0.14.x and 0.15.x patch-release deltas | version-deltas.md | pull the missing tags | OPEN-BOUNDED |
| U07 | `HALOGEN_WALK_FALSIFY` effect on acceptance math | version-deltas.md | device side answered (no device change); host feeding untraced | PARTIAL |
| U08 | HREF record layout in `.hgn` | hgn-format.md | CLOSED 2026-10-10: HREF is the magic of the `ppl` reference-dump file (`--ref-out` and `--ref`), not an .hgn record; premise was wrong (analysis/closure/hgn-href.md) | CLOSED |
| U09 | `HtItem` byte layout (stride 72 B known) | kernels-moe.md s9 | PARTIAL: HtItem size corrected, field meanings inferred (analysis/closure/moe-host.md) | PARTIAL |
| U10 | `k_ht_tg` meaning; grouped GEMV variants not deep-dived | kernels-moe.md s9 | PARTIAL (analysis/closure/moe-host.md) | PARTIAL |
| U11 | `DnfTaps` tap count per template arg 0..12; per-format bias table | kernels-moe.md s9 | PARTIAL: DnfTaps struct layout solid, tap count per template arg was the wrong question, bias table partial (analysis/closure/moe-host.md) | PARTIAL |
| U12 | `scan3` template args, `DncTapArgs`/`DncNormFold` layouts, `ut4/ut5` flags, decode-vs-prefill att variant, `dnc` name | kernels-indexer.md s5 | PARTIAL: the k_dnc_* family is DeltaNet chunk, not the indexer (analysis/closure/indexer-attn-host.md); grid arithmetic closed | PARTIAL |
| U13 | host half of "resume is bitwise": no prefill redo for cached rows | kernels-attention.md | CLOSED 2026-10-10 for "no prefill redo on cached positions"; warm-vs-cold numerics mechanism visible, size unmeasured (analysis/closure/cache-host.md, cache-key-kvpool.md) | CLOSED |
| U14 | fa2/qs template args 4-6, `FaSel` layout, legacy `k_attn`/`k_attn_fa` dispatch | kernels-attention.md | PARTIAL (analysis/closure/indexer-attn-host.md) | PARTIAL |
| U15 | `SampleRow` field map (offset 0xc uniform, offset 0 temperature are inferred) | kernels-speculation.md | CLOSED for the 32-byte layout and sources; PARTIAL for kernel meaning of +4 and +0xc (analysis/closure/sampling-host.md) | CLOSED |
| U16 | which `k_smpg_accept_final` mode value is greedy | kernels-speculation.md | no host call site found for the mode value; kernel side bounded (analysis/closure/sampling-host.md) | OPEN-BOUNDED |
| U17 | per-lane n-gram order in `k_ngram_ids` | kernels-speculation.md | CLOSED at kernel level (analysis/closure/ngram-dv-host.md) | CLOSED |
| U18 | `k_dv_quant*` block layout and argument roles; consumer of the int8 q rows | kernels-speculation.md s4 | PARTIAL (analysis/closure/ngram-dv-host.md) | PARTIAL |
| U19 | `v_sad_u8` codebook decode in `k_ht_*`, `k_i4r_*_gl`; HT bit-width template arg 3/4 meaning; `su`/`sv` signs vs scales | kernels-dequant.md s3 | PARTIAL: static decode attempted; needs a real HT checkpoint to finish (analysis/closure/dequant-host.md) | PARTIAL |
| U20 | host feeding of acceptance inputs under falsify mode | kernels-speculation.md s4b | CLOSED for WALK_FALSIFY, the S2 flags and SAMPLED_*; HALOGEN_S3_FALSIFY has no consumer found (analysis/closure/sampling-host.md) | CLOSED |
| U21 | dtype-to-kernel dispatch table; `lq*` template args; `hc*_fused` u64 output; power-of-two group scale source | kernels-dequant.md s5 | PARTIAL: dtype table partial; hc6/hc8_fused u64 outputs closed (completion counter); v_exp calls are sigmoid epilogues, not group scales (analysis/closure/dequant-host.md) | PARTIAL |
| U22 | AMDGCN pseudocode (Ghidra has no shader-ISA module) | ghidra-module-writeup.md | a community AMDGCN processor module, if one appears | OPEN-BOUNDED |
| U23 | seed and counter source of the splitmix64 draw in `k_smpg_pick`/`k_smpg_rpick`; what `s15` in the accept kernel holds | kernels-speculation.md | CLOSED for the kernel-side derivation and host row fill; PARTIAL for the wire-seed link (analysis/closure/sampling-host.md) | PARTIAL |
| U24 | purpose of the float-key radix select in `k_accept_prob_f32`, `k_sample_f32`, `k_resid_sample_f32` | kernels-speculation.md | CLOSED at kernel level (analysis/closure/ngram-dv-host.md) | CLOSED |
| U25 | what `k_dv_quant` quantizes to (compare ladder, not round-and-clamp) | kernels-speculation.md s4 | CLOSED at kernel level; codebook inferred as NF4 (analysis/closure/ngram-dv-host.md) | CLOSED |
| U26 | raw `GEN` and `CSTAT` unanswered on a second connection | runtime-run-0.17.3.md | CLOSED 2026-10-10: only the first connection gets request verbs; second connections get INFO and PING | CLOSED |

Totals (counted from the status column of this table, 2026-10-10 after the closure round): CLOSED 9 (U04, U08, U13, U15, U17, U20, U24, U25, U26); OPEN-BOUNDED 4 (U03, U06, U16, U22); PARTIAL 12 (U01, U02, U07, U09, U10, U11, U12, U14, U18, U19, U21, U23); UNOBSERVABLE 1 (U05); 26 rows. Where a row says CLOSED or PARTIAL, its text says which half is closed. Findings corrected along the way: k_dnc_* is DeltaNet chunk, not the indexer; HREF is the ppl reference-dump magic, not an .hgn record; the hc fused u64 output is a completion counter; v_exp in the grouped kernels is a sigmoid epilogue. Reports: analysis/closure/. Live evidence: analysis/runtime-run-0.17.3.md.
