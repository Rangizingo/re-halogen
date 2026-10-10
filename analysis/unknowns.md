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
| U01 | HNPB header u32 at +0x08 and per-blob u32 pair at +0x50/+0xa0 | npu-engine.md s6 | PARTIAL after round 2 (analysis/closure/npu-closure.md, npu-closure-r2.md) | PARTIAL |
| U02 | per-verb argument grammars of the NPU serve protocol | npu-engine.md s6 | PARTIAL after round 2: grammar table in analysis/closure/npu-closure-r2.md; some numeric field meanings open | PARTIAL |
| U03 | NPU `main:<graph>` name set | npu-engine.md s6 | graph names live in shipped `.elf` files not in this workspace; needs those files (analysis/closure/npu-closure.md) | OPEN-BOUNDED |
| U04 | engine behavior for raw `GEN ... 2` (dflash2 id) | dflash2-probe.md s5 | CLOSED 2026-10-10: engine refuses with `error ... asked for drafter 2 (unknown)` (analysis/runtime-run-0.17.3.md) | CLOSED |
| U05 | future flip of the dflash2 format-string literal | dflash2-probe.md s5 | unobservable from pulled set | UNOBSERVABLE |
| U06 | 0.14.x and 0.15.x patch-release deltas | version-deltas.md | pull the missing tags | OPEN-BOUNDED |
| U07 | `HALOGEN_WALK_FALSIFY` effect on acceptance math | version-deltas.md | device side answered (no device change); host feeding untraced | PARTIAL |
| U08 | HREF record layout in `.hgn` | hgn-format.md | CLOSED 2026-10-10: HREF is the magic of the `ppl` reference-dump file (`--ref-out` and `--ref`), not an .hgn record; premise was wrong (analysis/closure/hgn-href.md) | CLOSED |
| U09 | `HtItem` byte layout (stride 72 B known) | kernels-moe.md s9 | PARTIAL: record size is 16 B (second consumer confirms), the host does not fill the records, field meanings are a stride-based inference (analysis/closure/moe-dv-host-r2.md) | PARTIAL |
| U10 | `k_ht_tg` meaning; grouped GEMV variants not deep-dived | kernels-moe.md s9 | PARTIAL: grouped path selection CLOSED (grouped is the default, switch HALOGEN_HT_DEC_GROUP); launch of k_ht_tg and k_ht_tg_sum not found (analysis/closure/moe-dv-host-r2.md) | PARTIAL |
| U11 | `DnfTaps` tap count per template arg 0..12; per-format bias table | kernels-moe.md s9 | PARTIAL after round 2, one new host fact (analysis/closure/moe-dv-host-r2.md) | PARTIAL |
| U12 | `scan3` template args, `DncTapArgs`/`DncNormFold` layouts, `ut4/ut5` flags, decode-vs-prefill att variant, `dnc` name | kernels-indexer.md s5 | PARTIAL after round 2 (analysis/closure/indexer-attn-host-r2.md) | PARTIAL |
| U13 | host half of "resume is bitwise": no prefill redo for cached rows | kernels-attention.md | CLOSED 2026-10-10 for "no prefill redo on cached positions"; warm-vs-cold numerics mechanism visible, size unmeasured (analysis/closure/cache-host.md, cache-key-kvpool.md) | CLOSED |
| U14 | fa2/qs template args 4-6, `FaSel` layout, legacy `k_attn`/`k_attn_fa` dispatch | kernels-attention.md | PARTIAL: fa2 args and legacy dispatch closed, FaSel byte layout proven, slot meanings open (analysis/closure/indexer-attn-host-r2.md) | PARTIAL |
| U15 | `SampleRow` field map (offset 0xc uniform, offset 0 temperature are inferred) | kernels-speculation.md | CLOSED for gating of +4 (top_k) and +8 (top_p); +0xc is min_p, not a uniform (earlier docs corrected); exact thresholds partial (analysis/closure/sampling-host-r2.md) | CLOSED |
| U16 | which `k_smpg_accept_final` mode value is greedy | kernels-speculation.md | CLOSED 2026-10-10: the argument is HALOGEN_S3_FALSIFY (default 0), not a greedy-vs-sampled switch; mode 3 forces acceptance of the fall-through row (analysis/closure/sampling-host-r2.md) | CLOSED |
| U17 | per-lane n-gram order in `k_ngram_ids` | kernels-speculation.md | CLOSED at kernel level (analysis/closure/ngram-dv-host.md) | CLOSED |
| U18 | `k_dv_quant*` block layout and argument roles; consumer of the int8 q rows | kernels-speculation.md s4 | PARTIAL: argument roles and launch geometry closed; output plane offsets and the writers of the done-line dv counters open (analysis/closure/moe-dv-host-r2.md) | PARTIAL |
| U19 | `v_sad_u8` codebook decode in `k_ht_*`, `k_i4r_*_gl`; HT bit-width template arg 3/4 meaning; `su`/`sv` signs vs scales | kernels-dequant.md s3 | PARTIAL: decode recovered statically (16-bit sliding windows, multiply 0x83dcd12d, v_sad_u8 byte sum plus 0x6400, fp16 affine 0xc931); needs a real HT checkpoint to confirm (analysis/closure/dequant-host-r2.md) | PARTIAL |
| U20 | host feeding of acceptance inputs under falsify mode | kernels-speculation.md s4b | CLOSED for WALK_FALSIFY, the S2 flags and SAMPLED_*; HALOGEN_S3_FALSIFY has no consumer found (analysis/closure/sampling-host.md) | CLOSED |
| U21 | dtype-to-kernel dispatch table; `lq*` template args; `hc*_fused` u64 output; power-of-two group scale source | kernels-dequant.md s5 | PARTIAL: decode and small-batch launch CLOSED (kernel identity read from the binary); prefill and per-dtype helper reach partial (analysis/closure/dequant-host-r2.md) | PARTIAL |
| U22 | AMDGCN pseudocode (Ghidra has no shader-ISA module) | ghidra-module-writeup.md | a community AMDGCN processor module, if one appears | OPEN-BOUNDED |
| U23 | seed and counter source of the splitmix64 draw in `k_smpg_pick`/`k_smpg_rpick`; what `s15` in the accept kernel holds | kernels-speculation.md | PARTIAL: wire-seed parse side CLOSED; the copy into the per-request params at +0x10 not located (analysis/closure/sampling-host-r2.md) | PARTIAL |
| U24 | purpose of the float-key radix select in `k_accept_prob_f32`, `k_sample_f32`, `k_resid_sample_f32` | kernels-speculation.md | CLOSED at kernel level (analysis/closure/ngram-dv-host.md) | CLOSED |
| U25 | what `k_dv_quant` quantizes to (compare ladder, not round-and-clamp) | kernels-speculation.md s4 | CLOSED at kernel level; codebook inferred as NF4 (analysis/closure/ngram-dv-host.md) | CLOSED |
| U26 | raw `GEN` and `CSTAT` unanswered on a second connection | runtime-run-0.17.3.md | PARTIAL 2026-10-10: on 0.17.3 a second connection got INFO and PING answers and no CSTAT reply within 10 s (one run); its raw GEN result was not recorded (runtime-run-0.17.3.md:16). Where the engine accepts a second fd while the first is served is not found (no accept call in 0182ea80.c; main keeps a pending-fd vector, main_pseudo.c:1080-1103) | PARTIAL |

Totals (counted from the status column of this table, 2026-10-10 after two closure rounds): CLOSED 9 (U04, U08, U13, U15, U16, U17, U20, U24, U25); OPEN-BOUNDED 3 (U03, U06, U22); PARTIAL 13 (U01, U02, U07, U09, U10, U11, U12, U14, U18, U19, U21, U23, U26); UNOBSERVABLE 1 (U05); 26 rows. Where a row says CLOSED or PARTIAL, its text says which half is closed. Findings corrected along the way: SampleRow +0xc is min_p (not a uniform) and the accept kernel's mode argument is HALOGEN_S3_FALSIFY; k_dnc_* is DeltaNet chunk, not the indexer; HREF is the ppl reference-dump magic, not an .hgn record; the hc fused u64 output is a completion counter; v_exp in the grouped kernels is a sigmoid epilogue. Reports: analysis/closure/. Live evidence: analysis/runtime-run-0.17.3.md.
