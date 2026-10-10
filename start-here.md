# START HERE — halogen RE: state of play (2026-10-10)

Read this first, then `PLAN.md` (the approved 7-phase plan + per-phase
progress log), then `analysis/report.md` (the findings report).

**Overall: ~85% done.** Phases 0, 1, 3, 4, 5 are closed with evidence.
Phase 2 is 3/5 families written. Phase 6 (closure) has not started.

## Done (with artifacts)

| Phase | What | Artifacts |
|---|---|---|
| 0 | Runtime corroboration (live probes dead-ended honestly; protocol grammar corroborated statically) + full version-delta pass over 9 releases | `analysis/runtime-findings.md`, `analysis/version-deltas.md`, `work/versions/*/fatbin-manifest.json`, `analysis/acceptance/deltas_raw.txt` |
| 1 | Fatbin carve + gfx1151 disassembly: 4 code objects, 3.34M lines, 214 slices; Ghidra AMDGCN probed — no shader-ISA module exists, objdump-level is the accepted path | `work/fatbin/`, `analysis/kernels-disasm/`, `analysis/kernels-slices.tar.xz`, `analysis/fatbin-inventory.md`, `analysis/ghidra-module-writeup.md` |
| 2 (3/5) | Kernel-family writeups: attention (k_attn_fa2/qs*), MoE (k_dn_fused16, k_exp*, router), indexer (k_dnc_* sparse attention) | `analysis/kernels-attention.md`, `analysis/kernels-moe.md`, `analysis/kernels-indexer.md` |
| 3 | `.hgn` container: format spec written + Python round-trip implementation + acceptance: 165-file verify matrix (`tmp/verify_matrix.tsv`, 9 PASS / 160 designed FAILs) and 5-test round-trip suite all PASS (`tmp/rt_roundtrip.py`: build→verify, corrupt→checksum-fail, parse-back, truncate→layout-fail, HT negative matrix) | `analysis/hgn-format.md`, `work/hgn/hgnpy.py`, `work/hgn/files/`, `work/hgn/rt/` |
| 4 | NPU engine: getaddrinfo loopback gate proven, XRT call graph, hnpm/hnpb container formats, serve protocol verbs | `analysis/npu-engine.md`, `work/npu/*.pseudo.c` |
| 5 | dflash2: reserved wire-protocol drafter slot, hard-wired 0 in the INFO format string of every engine 0.11.10→0.17.3; shipped second drafter is the n-gram prompt-lookup beside the MTP head | `analysis/dflash2-probe.md` |

## Remaining to 100%

### 1. Phase 2 — two kernel-family writeups (the long pole, ~2-4h)

Slices and per-kernel instruction-class stats are pre-cut:
- **Dequant/quantized-GEMM**: `work/kernels-dequant/` (1141 slices:
  k_lq4w/lq6k/lq8s32/lq8w/lq6w/lbfw/hcup*/hc6_fused/hc8_fused/ht_*/
  i4r_*/q4moe) + `tmp/stats-dequant.csv` → write
  `analysis/kernels-dequant.md`. Must close the acceptance claim "HT
  tensors carry a rotation plane" (k_ht_*; companion-tensor rule
  `.suh`/`.svh` documented in hgn-format.md).
- **Speculation/sampling/acceptance**: `work/kernels-spec/` (43 slices:
  k_smpg_*, k_accept_prob_f32, k_pen_*, k_sample_f32, k_resid_sample_f32,
  k_q_softmax, k_ngram_*, k_dv_quant/k_dv_quant8/k_dv_scatter) +
  `tmp/stats-spec.csv` → write `analysis/kernels-speculation.md`. Must
  answer: greedy trunk-argmax vs probabilistic acceptance in
  k_accept_prob_f32/k_smpg_accept_final (v_exp/v_rcp/RNG presence),
  n-gram pure-lookup claim, what k_dv_* compute (draft-verify quant/
  scatter for the 0.17.2/0.17.3 sampled-speculation path).

**Pitfall (hit twice on 2026-10-10):** subagent writeup runs died from
model context overflow reading the big slices (k_dv_quant is 6068 lines).
Use grep-driven reading: rank kernels from the stats CSVs, grep
instruction classes (v_exp, v_rcp, v_cvt_scalef32, v_permlane, v_mfma,
v_bfe, ds_), then read only targeted 50-150-line windows, max ~15 read
calls. Format template: `analysis/kernels-indexer.md`.

### 2. Phase 6 — closure (~2-3h)

1. Unknowns ledger: REA `record_unknown`/`update_unknown` for every
   bounded-open item collected across the writeups (each writeup has a
   bounded-open section); `build_reconstruction_obligation_ledger` +
   `evaluate_reconstruction_coverage` → `analysis/coverage.md`.
   Note: the REA ledger tools demand an authenticated Evidence bundle
   (ev_/unk_ ids, digests); the CLI session evidence lives in
   `analysis/*.json` (e.g. `layout_flash_serve.json` carries a real
   evidence_id). `export_evidence_bundle` writes the session bundle.
   If the fail-closed schema proves impractical for this workspace's
   hand-made evidence, deliver a plain markdown ledger
   (`analysis/unknowns.md`) + coverage table and say so plainly.
2. Rewrite `analysis/report.md` §6 → "Findings complete; residual
   unknowns: <list>". Update AGENTS.md status line (protected file —
   needs explicit approval per write).
3. Commit each phase pushy-style (detailed bodies), push to
   `https://github.com/Rangizingo/re-halogen` (git auth via gh;
   `gh auth setup-git` — the id_ed25519 key is deploy-key-bound, SSH
   push fails).

## Ground rules (unchanged, from AGENTS.md)

- Never run `flash_serve`/`halogen-tools` beyond `verify`/`inspect` on
  files we created in this workspace.
- Ghidra jobs: `sudo systemd-run --property=MemoryMax=16G`, TMPDIR
  outside dot-paths, PIE base +0x100000.
- Big disassembly listings stay out of git (tarballs; raw listings
  gitignored).
- Every mutation on this host outside ~/re-halogen needs Peter first.
