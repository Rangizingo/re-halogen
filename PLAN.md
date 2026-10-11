# PLAN.md — Finish reverse engineering halogen (autonomous run)

Standing authorization: Peter approved this plan for **autonomous execution**
(2026-10-09): "so I can have you do all of this autonomously without me next
session." Within this plan only, normal work (installs of open-source tools,
file writes under ~/re-halogen, git commits/pushes to Rangizingo/re-halogen,
read-only analysis, running extracted code objects through disassemblers) is
pre-approved. Anything outside this scope — running flash_serve, touching the
live engine (pid 2918), writes outside ~/re-halogen, new repos, spending money,
deleting anything outside the scratch dirs — still requires asking Peter.

Read first: `AGENTS.md` (layout + gotchas), `analysis/report.md` (state of
play). Tooling: REA CLI `~/.hermes/tools/node-26.7.0-linux-x64/bin/rea`,
Ghidra `~/.hermes/tools/ghidra-12.1.2`, r2, pwntools venv
`~/.hermes/venvs/pwn` (`REA_PWNTOOLS_PYTHON`). Ghidra jobs: `sudo systemd-run
--property=MemoryMax=16G` (scripts/run_fals4.sh pattern), TMPDIR outside
dot-paths, PIE base +0x100000 for `rea function`.

Definition of done: every section below has an evidence-backed writeup in
`analysis/`, the unknowns ledger is closed or explicitly bounded, and
report.md's §6 "Not yet done" is empty.

---

## Phase 7: Complete reverse engineering (added 2026-10-10, Peter's goal)

**Goal (Peter, verbatim intent):** 100% reverse engineering, a full, proven
understanding of how halogen-flash-server 0.17.3 works from start to finish:
formats, protocol, which kernel runs when, what each one computes. Someone
could rebuild it from our writeups. Target is the latest release (0.17.3)
only.

**Definition of 100%:**
1. Every function in `flash_serve` (~2,100) and `halogen-npu` has a role label;
   any function we cannot explain is listed by name, not skipped.
2. One request traced end to end with proof at each step: network call in,
   loading, scheduling, prefill, decode, GPU kernels, sampling, streamed reply.
3. The 19 closable unknowns in `analysis/unknowns.md` closed with proof.
4. A rebuild guide (`analysis/rebuild-guide.md`) plus known-good test data
   recorded from a real engine run, so a rebuild can be checked against it.
5. Three named limits stay, by decision: U05 (future releases, unobservable),
   U06 (0.14.x/0.15.x patch releases, dropped by Peter: only history, not the
   latest), U22 (no Ghidra AMDGCN module; GPU code stays at assembly level,
   explained by hand in the five kernel writeups).

**Done already (second pass, branch `phase6-closure`):** all five kernel-family
writeups, `unknowns.md` (22 items), `coverage.md`, report s6, CLAUDE.md.

**Work, in order:**
1. Mac Ghidra: install build tools, build the macOS decompiler for Ghidra
   12.1.4 (`~/tools/ghidra_12.1.4_PUBLIC`, release ships linux/win only).
   Peter picked this Mac for the decompile (the 4090 does not help Ghidra;
   Ai-core is busy serving Hermes). REA (`rea-agents` 6.3.0, registered at
   user scope) drives Ghidra: `GHIDRA_INSTALL_DIR`, PIE base +0x100000.
2. Function map: REA binary overview, then classify all functions by role
   (loading, scheduling, KV cache, sampling, protocol, GPU dispatch, runtime).
   Send Peter the group list for approval before deep passes.
3. Decompile the main path and close the 16 host-code unknowns (U01-U03,
   U08-U18, U20, U21). Grep-driven, targeted windows; context overflow killed
   two earlier subagent runs.
4. End-to-end request trace and rebuild guide.
5. Ai-core run (needs Peter's go and a window): confirm the model file is
   complete, look for an HT checkpoint, run the engine once, record inputs and
   outputs, close U04 and U19.
6. Refresh unknowns, coverage, report; commit.

**Delegation (superseded by Peter's pick, Sonnet agents in parallel, no
DeepSeek spend):** the earlier idea was the DeepSeek lane
(`bits-engine/lab/tools/ds_draft.py`, task file in, reply out, no tools). It
spends Azure credit (stops at $950 or 2026-10-18), and this repo's stop
condition says spending needs Peter's yes. Nothing it returns counts as proof:
every claim is re-checked by grep or a disassembly window before it is written.

**Gates (answered by Peter 2026-10-10):** running the engine allowed; AGENTS.md
edits allowed; full Ai-core access granted now (Peter does not need it for
anything, Hermes may go down); branch push approved; no DeepSeek, use Sonnet
agents in parallel (Haiku only if the subagent-model guard allows it).

**Ai-core facts (read-only survey 2026-10-10):** ssh works; both engine
services inactive; Ghidra 12.1.2 (Linux) and ~/re-halogen present;
`/srv/models/halogen-swift/qwen38-flash-next-mtp.hgn` and the Swift GGUFs
exist; sizes and completeness not yet checked.

**Phase 7 progress (2026-10-10, end of first Phase 7 session):**

Done and pushed on branch `phase6-closure` (PR not yet opened):
- Ghidra 12.1.4 with an arm64 decompiler built on this Mac; `scripts/ExportDecomp.java` exported all 3,568 `flash_serve` functions (0.17.3) and 3,564 (0.17.2) to `work/decomp/` (gitignored).
- `analysis/function-map.md`, `function-map.tsv`, `function-map-core.tsv`: every function has a role (checked against the index, 0 missing, 0 duplicated).
- Client-side docs: `wire-protocol.md`, `entrypoint-flow.md`, `api-layer.md`.
- Live evidence: `runtime-run-0.17.3.md` plus `analysis/evidence/` (U04 and U26 closed; speculation, cache and seed measurements; the 0.17.3 engine ran on Ai-core inside the 0.17.2 image userland).
- `version-compare-0.17.2-0.17.3.md` (host code differs, device code does not).
- Kernel writeups corrected after an independent review (on-GPU splitmix64 RNG exists, `k_dv_scatter` is a gather, HT is a 128-point transform).
- Tests: `tests/` (11 pytest tests, all pass).

Incident: the engine test run deleted the live service's 60 GiB disk prompt cache (details in `runtime-run-0.17.3.md`). Rule from now on: test engines never mount `/srv/halogen-cache`; use their own cache dir and `HALOGEN_CACHE_PRUNE_OLD=0`.

Second Phase 7 session (2026-10-10): `halogen-npu` and `halogen-tools` mapped (1,412 and 3,222 functions), kernel stubs bound 1:1 (1,738), 17 closure reports merged into `analysis/unknowns.md` in two rounds (counted from its table: CLOSED 9, PARTIAL 13, OPEN-BOUNDED 3, UNOBSERVABLE 1; U26 reopened to PARTIAL by the review refuter), `request-trace.md` updated from them, `rebuild-guide.md` written, closure docs refreshed. PR #1 reviewed and merged (p7-3). Then, with Peter's approval to fetch vendor files: U03 closed from the shipped NPU device set and U19 closed by decoding real HT weights (ledger now CLOSED 11, PARTIAL 13, OPEN-BOUNDED 2, UNOBSERVABLE 1; U27 added for gate_up and i4r, gate_up since solved).

Third Phase 7 session (2026-10-10): PR #1 to #5 merged (main a2d60ae). p7-1 to p7-3 proven with harness receipts on the merged head. With Peter's approval to fetch vendor files: U03 closed from the shipped decider-0.8b NPU device set; U19 closed by decoding real HT weights (`scripts/ht_decode.py`, Pearson 0.98 to 0.99 against q4c references, 3 and 4 bit); HT gate_up experts decoded with their own tile walk (0.9922). Corrections made in review and recorded in the docs: U26 reopened to PARTIAL (second-connection GEN never recorded); the `.hgn` test matrix is 169 files, not 165; U19 narrowed and U27 added. Ledger now CLOSED 11, PARTIAL 13, OPEN-BOUNDED 2, UNOBSERVABLE 1, 27 rows. Remaining: p7-5 (i4r decode, U27).

Fourth Phase 7 session (2026-10-10): PR #6 to #8 merged (main 8b691e6). p7-5: v2 i4r experts decoded (gate_up 0.9876, down 0.9885 against q4c), U27 closed, HT qparam 0x1200 shown to be a fixed loader tag. p7-6: 13 round-3 agent reports, each re-checked; refuter reopened U14 and U18. p7-7: Ai-core run 2 with its own cache dir; U26 GEN half observed, serial seed determinism measured. Ledger now CLOSED 16, PARTIAL 8, OPEN-BOUNDED 2, UNOBSERVABLE 1. All three proofs pass on 8b691e6 with receipts.

Fifth Phase 7 session (2026-10-10): PR #10 to #19 merged. p7-8 (static round 4), p7-9 (Ai-core run 3, U26 closed), p7-10 (static round 5: U02, U11, U21 closed), p7-13 (rebuild-guide Other gaps all closed: MTP head, sampler math, cache and SSE, 256 env flags, D-line writers) ticked. Run 4 (U12 kernarg capture and full launch trace: 106 of 1,738 kernel instances on the default greedy path; attempts 1-2 failed on a shim bug, fixed). Kernel rewrite harness merged with k_scale_silu bitwise MATCH. Corrections recorded: U11 reopened by QA then closed in r5; U02 kept PARTIAL in r4 then closed in r5; U21 r4 mode-2 and q4moe gate readings corrected in r5; run 4 shim cause labelled inferred. Ledger: CLOSED 23, PARTIAL 1 (U12), OPEN-BOUNDED 2 (U06, U22), UNOBSERVABLE 1 (U05). Open: p7-11 (U12), p7-12 (U22, every kernel).

**Tracking for the next session** (the project plan_path `plan.md` is this same file on the case-insensitive Mac filesystem; never write a separate plan.md). Change `[ ]` to `[x] verified: <proof>` as each is proven.

- [x] p7-1 verified: `test -f analysis/request-trace.md && test -f analysis/rebuild-guide.md && ! grep -q <em dash> analysis/rebuild-guide.md` printed P7-1-PROOF-PASS (2026-10-10; trace 22 proven, 6 proven in part, 3 inferred; guide 311 lines). Update `analysis/request-trace.md` steps and Gaps from `analysis/closure/` (pld.md, mtp-head.md, cache-key-kvpool.md, kernel-binding.md and the r2 reports), then write `analysis/rebuild-guide.md` (formats, protocol, startup, request path, kernels and dispatch, cache and KV pool, speculation, NPU, proven vs inferred). Proof: both files exist and the guide has no em dash.
- [x] p7-2 verified: `python3 -m pytest -q tests` 11 passed and `grep -c 'Not yet done' analysis/report.md` 0, printed P7-2-PROOF-PASS (2026-10-10). Refresh `analysis/coverage.md`, `analysis/report.md` section 6, `start-here.md`, the AGENTS.md status line and this progress note to the current facts (unknowns counted from the table in `analysis/unknowns.md`; PARTIAL rows stay PARTIAL). Proof: `python3 -m pytest -q tests` passes and report.md section 6 no longer says "Not yet done".
- [x] p7-3 verified: PR #1 merged as 19ce5d7; `gh pr list --state merged --head phase6-closure --json number --jq 'length > 0'` printed true (P7-3-PROOF-PASS); QA review PASS and adversarial refuter report on the PR (comment 6100425403), refuter findings fixed in 763d50a; all three proofs re-run on merged main (2026-10-10). Open the PR for `phase6-closure`, independent review (reviewer on a different model than the author, see `.claude/handoff.md`), then merge. Proof: `gh pr list --state merged --head phase6-closure` is not empty.
- [x] p7-4 verified: PR #3 merged as fe2c87c; `python3 -I scripts/ht_decode.py <data>` printed 0.9924, 0.9931, 0.9839, 0.9848 (control -0.0008); ledger counted CLOSED 11, PARTIAL 12, OPEN-BOUNDED 3, UNOBSERVABLE 1 from its table on main; QA and refuter reports on PR #3 (2026-10-10). Close U03 (NPU graph names) and U19 (HT decode) from vendor files Peter approved fetching on 2026-10-10; update ledger, report, coverage, guide, start-here, AGENTS. Proof: `python3 -I scripts/ht_decode.py <data>` scores above 0.98 on all three tensors and `analysis/unknowns.md` counts CLOSED 11 from its table; PR merged.
- [x] p7-5 verified: PR #7 merged as d88938b; `bash ~/.claude/scripts/proof.sh p7-5` in a clean worktree at d88938b printed PASS (receipt .claude/receipts/p7-5.b5e212ff.json); `python3 -I scripts/ht_decode.py <data>` printed i4r gate_up 0.9876, down_proj 0.9885, control -0.0002; U27 CLOSED; QA (FAIL, fixed in 658b93d) and refuter reports on PR #7 (2026-10-10). U27: (HT gate_up layout done in PR #5) decode one i4r (dtype 23) tensor from the v2 checkpoint by range read. Proof: Pearson above 0.97 against the w4b q4c reference for each, decoder in scripts/ht_decode.py, U27 row updated.
- [x] p7-6 verified: PR #8 merged as 8b691e6; `bash ~/.claude/scripts/proof.sh p7-6` in a clean worktree at 8b691e6 printed PASS (receipt .claude/receipts/p7-6.2bf489ba.json); PARTIAL 8 from the table (was 13); closed by r3 reports U07, U09, U10, U23 (U27 by p7-5); U01, U02, U11, U12, U14, U18, U21 narrowed with reports; QA (FAIL, fixed in 4cbdb02) and refuter (U14, U18 reopened) on PR #8 (2026-10-10). Static closures: for each PARTIAL row in `analysis/unknowns.md` whose missing half needs only decompile or disassembly (U02, U07, U09, U10, U11, U12, U14, U18, U21, U23 candidates), close it with file:line proof in a new `analysis/closure/*-r3.md`, one agent per row, file-disjoint. Proof: PARTIAL count from the table is below 13 and every changed row cites a closure report.
- [x] p7-7 verified: PR #8 merged as 8b691e6; `bash ~/.claude/scripts/proof.sh p7-7` printed PASS (receipt .claude/receipts/p7-7.892830fd.json); run 2 recorded the second-connection GEN (waits until the first closes; U26 stays PARTIAL for CSTAT), serial seed 111 twice identical; Hermes down 19:03:09-19:05:24Z and 19:05:59-19:09:54Z, /health 200 after each (2026-10-10). Runtime closures on Ai-core (Peter approved 2026-10-10 at warp, Hermes may go down during runs): U26 second-connection GEN, serial-sampling seed determinism (seed 111 twice), and any PARTIAL row whose missing half is runtime-only. Own cache dir, HALOGEN_CACHE_PRUNE_OLD=0, never /srv/halogen-cache; restore halogen-swift.service and check /health after. Proof: `analysis/runtime-run-0.17.3-r2.md` exists with the transcripts, and the Hermes health check output is in it.
- [x] p7-8 verified: PR #11 merged as 5eb1035; `bash ~/.claude/scripts/proof.sh p7-8` in a clean worktree at 5eb1035 printed PASS (receipt .claude/receipts/p7-8.e3062474.json); PARTIAL 5 from the table (was 8): U01, U14, U18 closed, U02, U11, U12, U21 narrowed; QA (FAIL, fixed in bf55438, re-QA PASS) and refuter (no row refuted) on PR #11 (2026-10-10). Static round 4: close or bound U01, U02, U11, U12, U14, U18, U21 from decompile and disassembly, one agent per row, file-disjoint `analysis/closure/*-r4.md`, each cite re-checked, QA plus refuter on the PR. Proof: PARTIAL count from the table below 8.
- [x] p7-9 verified: PR #12 merged as c76fce1; `bash ~/.claude/scripts/proof.sh p7-9` in a clean worktree at c76fce1 printed PASS (receipt .claude/receipts/p7-9.f6ba0235.json); U26 CLOSED, ledger CLOSED 20, PARTIAL 4, OPEN-BOUNDED 2, UNOBSERVABLE 1; Hermes down 19:40:54Z to 19:44:24Z, /health 200 after; QA (FAIL, fixed in bef9b14, re-QA PASS) and refuter (all claims hold) on PR #12 (2026-10-10). Ai-core run 3: CSTAT on a waiting second connection, GEN completion vs close separated, third-connection release, plus runtime items from p7-8; own cache dir, prune off, Hermes restored. Proof: `analysis/runtime-run-0.17.3-r3.md` exists with CSTAT and the /health output.
- [x] p7-10 verified: PR #15 merged as 69b519f; in a clean worktree at 69b519f `grep -c '^| U[0-9]* .*| PARTIAL |$' analysis/unknowns.md` printed 1 and the `-lt 4` test printed PASS (no harness receipt: p7-10 has no frozen proof command); U02, U11, U21 CLOSED; QA PASS (2 important fixed) and refuter (all hold) on PR #15 (2026-10-10). Static round 5 (s5 ruling 3, step 1): U02 from the qwen3.5-2b and qwen3-embedding manifests (kind g buckets, range-read, sha256 match npu-models.txt:45,67) plus the STORE/HIT reply loops; U11 matmul-path consumer of repacked q5_0/q5_1/q8_0; U21 plain-path kernels below 64 tokens and HT_MOE_V2 mode 2. Proof: PARTIAL count from the table below 4.
- [x] p7-11 verified: U12 CLOSED (`grep -c '^| U12 .*| CLOSED |$' analysis/unknowns.md` prints 1 on branch p7-12-wave1): run 5 Tap payload (PR #21), n1/n2 (dnc-scan3-r6), scan3 B, C, D, G value-neutral by 83 bitwise dnc_scan3 MATCHes (rewrite/reports/g16b.md; orchestrator re-ran G=0/1 at N 1 and 300, 8 of 8 MATCH, and the v2 listing cites). Merged as 3fafce9 (PR #23, QA PASS, refuter all UPHELD); proof re-run in a clean worktree at 3fafce9: PROOF p7-11 PASS, ledger 24 0 2 1, pytest 88 passed 4 skipped (2026-10-11). Was: Live closures on Ai-core (ruling 3, steps 1-2): U02 STORE/HIT replies from a halogen-npu run on qwen3.5-2b if p7-10 leaves it open (model files to Ai-core, own dirs); U12 NormFold n1/n2 identity and Tap payload by a kernarg capture on a flash_serve engine run. Proof: U02 and U12 rows not PARTIAL.
- [x] p7-11a verified: U12 n1/n2 identity and Tap payload (closure/dnc-scan3-r6.md cites re-run: run 4 matmul D = n1 108, A/B = n2 108, D = n2 0; runtime-run-0.17.3-r5.md: slots (6,11,14)/(-1,-1,14)/(-1,-1,-1) x 36 each, 144 of 144 Tap payloads hash-identical to e<N>.blob slices, e448.rec decoded). Left in p7-11: scan3 B/C/D/G labels from the p7-12 dnc_scan3x rewrite (group g09), then U12 CLOSED.
- [x] p7-12c verified: `grep -c 'AUDIT OK' rewrite/const-audit.md` prints 1738, AUDIT GAP 0, AUDIT ERROR 0, one explained exception (rewrite/const-audit-exceptions.tsv); tests/test_const_audit.py 39 pass; orchestrator check of the helper-call finding in orchestrator-checks.md (2026-10-11). Was: Constant audit (ruling 4c): script `rewrite/harness/const_audit.py` (with tests) extracts every literal from each rewrite's compiled gfx1151 ISA and from the vendor kernel's listing and reports the instance as AUDIT OK only when every vendor literal is accounted for; one AUDIT row per instance in `rewrite/const-audit.md`. Proof: every instance in function-map-kernel-stubs.tsv has AUDIT OK or an explained exception.
- [x] p7-12 verified: the p7-12 proof prints `1738 1738` and exits 0 on branch p7-12-wave2; `grep -c '^| U22 .*| CLOSED |$' analysis/unknowns.md` prints 1; ledger 25 0 1 1; every group re-checked by the orchestrator (rewrite/reports/orchestrator-checks.md). Merge with QA and refuter pending. Was: U22 kernel rewrite (ruling 3, step 3, every kernel): for each kernel instance in `analysis/function-map-kernel-stubs.tsv` (1,738), a HIP rewrite under `rewrite/` built for gfx1151 and an output match against the vendor kernel on Ai-core with the same inputs; debug and dead variants included. Proof: a match table with one row per instance, all MATCH, and U22 closed.
- [x] p7-13 verified: every rebuild-guide s11 "Other gaps" item cites a report whose verdict for it is CLOSED (mtp-head-r5 and r6, sampler-math-r5, cache-sse-r5, env-dline-r5; checked by QA on PR #18, which recomputed mtp-head-r6 from run-r4-kargs.jsonl.gz); cache-sse items are closed from code with no on-disk record observed yet (follow-up in p7-11's next engine run); refuter on PR #18 weakened only the D-line rounds reading, fixed (2026-10-10). Smaller gaps (ruling 3, step 4): rebuild-guide s11 "Other gaps" (MTP head kernels and fc fusion, router top-k, greedy draft step, sampler tie and top_p rules, penalty math, disk cache naming and manifest, cache clock unit, S29 snapshot, S27 SSE framing, engine env flag list, D-line counters). Proof: the "Other gaps" list is empty or each item cites a closure report.

Reviewer gate: the stop-time review gate needs a reviewer on a different model than the Sonnet author; a fix (extra ladder rung and clearer message) was handed to an agent under a 60 minute hook grant (see needs-human). Settings parity warning: `rust-analyzer-lsp` plugin not installed, entry dropped from live `settings.json` only; Peter's call.

---

## Progress log (updated 2026-10-10)

Evidence lives in `analysis/` and `work/`; decompiles in `work/hgn/*.c`
(27 Ghidra JSON passes, 17 cleaned pseudocode files).

**Phase 0 — DONE.** Live probes skipped honestly: engine pid 2918 gone,
127.0.0.1:8730 refuses; 8731 is the bits-live llama stack (not the target).
Protocol grammar corroborated statically (client-side) in
`analysis/runtime-findings.md`. **Version pull rerun 2026-10-10**: all 8
versions (0.11.10/0.12.0/0.13.3/0.13.4/0.15.0/0.16.4/0.17.0/0.17.2) pulled
to `work/versions/`, carved per-version (`carve_fatbin.py`), symbol-set +
string-set diffs run (`scripts/version_deltas.py` → `tmp/deltas_raw.txt`)
→ **`analysis/version-deltas.md` written** (code-object sha256 lineage
across 9 releases; +318 kernels at 0.16.4; k_dv_* at 0.17.2;
HALOGEN_SAMPLED_*/WALK_FALSIFY host strings at 0.17.3; 0.17.2→0.17.3 ships
zero device-code change).

**Phase 1 — DONE at disassembly level.** Clang offload bundle v2 walked
(`scripts/carve_fatbin.py`): 4 code objects in `work/fatbin/` (largest
17.7 MB, 2,982 symbols; `work/fatbin-manifest.json` has offsets/sha256s).
llvm-objdump 21 `--mattr=+gfx1151` disassembled all four →
`analysis/kernels-disasm/*.s` (3.34M lines). 214 per-kernel slices in
`work/kernels/` (k_attn_fa2/qs*, k_dnc_att*/scan3*/prep, k_dn_fused16,
k_lq8w/k_lq4w, k_i4r_rows, k_linear_f32_q4_mt, k_smpg_accept_final,
k_ngram_*). Ghidra AMDGCN: **no module exists** for shader ISA — probed and
documented in `analysis/ghidra-module-writeup.md` (only the F32 management-CPU
module, installed + tested, irrelevant to kernels). Per plan's fallback:
kernel analysis proceeds at objdump level; Ghidra pseudocode for AMDGCN is a
bounded, documented open item. `analysis/fatbin-inventory.md` written
from the manifest (2026-10-10).

**Phase 2 — IN PROGRESS (objdump-level path), 3/5 families written.**
`analysis/kernels-attention.md` (k_attn_fa2/qs*, 264 L),
`analysis/kernels-moe.md` (k_dn_fused16/k_exp*/router, 370 L),
`analysis/kernels-indexer.md` (k_dnc_* sparse-attention, 194 L).
Slices for the remaining families cut 2026-10-10:
`work/kernels-dequant/` (1141: k_lq4w/lq6k/lq8s32/lq8w/lq6w/lbfw/hcup*/
hc6_fused/hc8_fused/ht_*/i4r_*/q4moe) + `work/kernels-spec/` (43:
k_smpg_*/accept_prob/pen_*/sample/ngram_*/dv_quant/dv_scatter), stats in
`tmp/stats-dequant.csv` / `tmp/stats-spec.csv`. Two subagent writeup runs
on these died from model context overflow (transcripts under
`~/.hermes/cache/delegation/live/`); the reads are cheap, the writeup
needs a grep-driven budget — see start-here.md. Acceptance claims:
attention/MoE/indexer claims closed in their writeups; HT rotation-plane
and speculation-cannot-change-output claims need the two missing writeups.

**Phase 3 — DONE.** `.hgn` format fully recovered and
implemented round-trip in `work/hgn/hgnpy.py` (docstring = the spec draft):
104-B header (magic HGN1, version, n_tensors, table_off, data_off, file_bytes,
model_id[64]); 160-B table entries (name[96], dtype, rank≤4, dims u64[4],
absolute payload_off, nbytes, xor32 = XOR of LE u32 words, qparam); payloads
64-B aligned in table order; zero pads at [4,table_off), post-table, inter-
payload, tail; header_sha256 = sha256(file[0:data_off]). 24 dtype codes
(bf16/f32/f16/q4c/fp8r/q8g64/i4l/fp8g/q8s32/q6k/iq4nl/q4a/q5a/ht/q8_0/q5_0/
q5_1/i4r/q6g64…). 165 crafted test containers in `work/hgn/files/`
(valid/bad-xor/truncated/bad-pad/bad-header variants). Repack driver
decompiled: `flash_serve` 0x19af350 (`work/hgn/gs_0x19af350.c`) — GGUF magic
gate, n-gram table requirement (`layers.1.ple.ngram_embedding.weight` /
`per_layer_token_embd.weight`, fatal "Nothing was written" path), 64-B
aligned layout from 0x80, sparse `ftruncate`, `HALOGEN_GGUF_THREADS`
(default 8, clamp 1..64), thread-pool spawn. **`analysis/hgn-format.md`
written; acceptance matrix ran (`tmp/verify_matrix.tsv`, 165 files: 9 PASS
/ 160 FAIL as designed) and the round-trip suite
(`tmp/rt_roundtrip.py` → `work/hgn/rt/`) PASSES all 5: build→verify PASS,
corrupt-payload→checksum FAIL, parse-back field-exact, truncate→layout
FAIL, HT negative matrix (9 qparam forms, all rejected as
"form the engine does not read" + companion-plane rule).**

**Phase 4 — DONE.** `analysis/npu-engine.md` written. getaddrinfo: single call
site 0x35042 in FUN_00133090 (listen), loopback gate proven (sa_data[2]==0x7f /
::1 checks; no connect import) — verdict loopback-only. XRT call graph documented
(device→elf→hw_context→ext::kernel "main:<graph>"→run start/wait→bo
host-only 0x20000000 = XCL_BO_FLAGS_HOST_ONLY, not CACHEABLE). Containers:
devices.hnpm = text key=value manifest (parser FUN_00164090, xor32 FUN_00162fd0);
consts.hnpb = HNPB v1 binary blob pack (FUN_00167960, 0xa8-B table, 4K-aligned
blobs); `<id>.hnpw` = same HNPB v1 reader (no HNPW magic exists). serve/listen
protocol verbs recovered (INFO/GEN/DEC/IMG/EMB/SCO/CLS/PFX/LEDGER/CANCEL/QUIT,
READY banners, ERR/OK replies); worker FUN_001362d0 + dispatcher FUN_0012b110
decompiled (work/npu/*.pseudo.c). Bounded-open: HNPB +0x08 header u32, per-blob
u32 pair @+0x50/+0xa0, per-verb argument grammars.

**Phase 5 — DONE.** `analysis/dflash2-probe.md` written. Verdict: dflash2
is a reserved wire-protocol drafter slot (id 2), hard-wired to 0 in the
INFO format string of every engine 0.11.10→0.17.3 (zero `dflash` strings
in any engine binary); the client 400s it before the engine sees it. The
shipped second drafter is the n-gram prompt-lookup (HALOGEN_PLD,
k_ngram_* kernels) running beside the MTP head. Bounded-open: engine
behavior for a raw `GEN ... 2` on the socket.

**Phase 6 — NOT STARTED.** Ledger, coverage, evidence bundle, report
rewrite. Blocked on Phase 2's last two writeups (see start-here.md).

## Phase 0 — Runtime corroboration (cheapest, do first)

Goal: turn inference into observation on the live engine family.

1. `capture_process_scenario` / plain `curl` against the **local** engine
   front (bits-live llama-server stack is NOT the target; the halogen-family
   engine on this host is pid 2918, engine port 8730 loopback). Read-only
   probes only: `PING`, one tiny `GEN` at temperature 0, `/health` on 8731.
   Do not send SCHEMA/THINK/GUARD variants that mutate state; do not restart,
   signal, or attach to pid 2918. If 8730 refuses PING, skip the live probes
   and note it — do not start a new engine instance.
2. `compare_application_versions`: pull 0.17.2, 0.17.0, 0.16.4, 0.15.0 image
   layers (registry pattern in report.md §intro; token flow: ghcr.io/token +
   manifest + layer blobs). `compare_functions` / `find_changed_behavior` on
   flash_serve across versions; verify 3 changelog claims (0.17.3 sampled
   decode, 0.12.0 indexer rewrite "byte-identical", 0.13.4 end-of-turn guard)
   against binary deltas.
3. Output: `analysis/runtime-findings.md`, `analysis/version-deltas.md`.

Acceptance: protocol grammar claims marked observed vs inferred; ≥3 changelog
claims verified/refuted with evidence IDs.

## Phase 1 — fatbin extraction + AMDGCN toolchain

Goal: disassemble the 23 MB device code with AMD's own LLVM (no Hopper/IDA).

1. Parse clang offload bundle **v2** at file 0x52000 (magic
   `__CLANG_OFFLOAD_BUNDLE__`, version u16 at +22 = 2, offset-table-size u64
   at +32; entries: target triple, encoding, content offset; content region
   precedes the table). Write `scripts/extract_fatbin.py`; verify every
   extracted blob is a valid ELF with `e_machine == 0xe6` (EM_AMDGPU) or LLVM
   bitcode; record counts + sha256s in `analysis/fatbin-inventory.md`.
2. Install AMD LLVM objdump free: `sudo apt-get install -y llvm` (upstream
   LLVM ≥ 19 supports gfx1151) and/or extract `amd-smi`-adjacent LLVM tools
   from the image's `_rocm_sdk_devel` layer (layer 4, 773 MB — it ships
   llvm-objdump with AMDGPU targets). Prefer the ROCm one: exact ISA match.
3. `llvm-objdump -d --mattr=+gfx1151` every code object →
   `analysis/kernels-disasm/*.s`.
4. Ghidra AMDGCN: try community processor modules
   (e.g. `protondfx/ghidra-amdgpu`, GSoC AMDGPU modules). Install into
   `ghidra-12.1.2/Ghidra/Processors/`, run `rea analyze` on one code object.
   If no module handles RDNA3.5 ISA cleanly, fall back: import the
   llvm-objdump listing into Ghidra as binary + apply a hand-written loader
   script, or keep analysis at objdump level with r2 `gcn` plugin as the
   disassembler and do xref/data recovery manually. Record which path worked.

Acceptance: every kernel entry point (the `k_*` symbols in the fatbin symbol
tables) has disassembly; ≥10 core kernels have Ghidra pseudocode.

## Phase 2 — Core kernel analysis

Goal: explain, with pseudocode, what the engine's numbers claim.

Priority kernels (from `analysis/functions.txt` + fatbin symbol tables):
1. Attention: `k_attn_qs*` (block-tiled variants), `k_dnc_att*`, FA variants
   (`HALOGEN_ATTN_FA/FD/QS*` flags select them).
2. MoE: `k_exp_gu_w` (gate/up fused, W8/W16/W32 widths), `k_expert_h_q4`,
   router (`HALOGEN_FLASH_ROUTE_*`).
3. Dequant: `k_deq_q4cp_sel`, HT (`HALOGEN_HT_*`) and I4R (`HALOGEN_I4R_*`)
   families — tie each to the quant families in report.md §5.
4. Speculation: `accept_probs` / `k_accept_prob_f32`, n-gram
   (`k_ngram_ids/hist/segstart/ctx_update`) — confirm trunk-argmax commit
   inside the verify kernel, closing the falsifier argument at kernel level.
5. Indexer (sparse attention): the 0.12.0 rewrite targets.

Method: `open_binary` (Ghidra) per code object, `analyze_function` per
kernel, `annotate_native_function` to name/rename as understanding firms;
`build_call_path` where host-side dispatch is reachable. One writeup per
family: `analysis/kernels-<family>.md` — algorithm, quant formats handled,
SIMD/LDS structure, and which README perf claims it explains.

Acceptance: the four claims "speculation can't change output", "prompt cache
resume is bitwise", "HT tensors carry a rotation plane", "n-gram table is a
pure lookup" each have kernel-level evidence.

## Phase 3 — halogen-tools: .hgn format spec

Goal: written format spec, verified by parsing the shipped checkpoints'
metadata (not the 118 GiB weights — parse headers/tables only).

1. Decompile: `verify`, `inspect`, `convert`/repack paths (entry via
   `rea analyze halogen-tools` overview → main → dispatch).
2. Recover: header layout, tensor-family table, per-tensor
   name/dtype/qparam/dims/nbytes/xor32/sha256 encoding, pads_zero semantics,
   HREF ref-file format (already documented from strings in AGENTS.md),
   GGUF→hgn repack transform (lossless claim).
3. Write `analysis/hgn-format.md`; validate by round-tripping a tiny synthetic
   file through the decompiled logic (python reimplementation vs binary's
   `verify` on a crafted file — running halogen-tools `verify`/`inspect` on a
   file you create in ~/re-halogen is pre-approved; it loads no model).

Acceptance: spec complete enough that a third party writes a reader; 3
round-trip tests pass.

## Phase 4 — halogen-npu

1. Decompile XRT call graph (`xrt::device`, `hw_context`, `bo` map/sync),
   `devices.hnpm` manifest reader, `consts.hnpb`, `hnpw` weight reader.
2. Resolve the `getaddrinfo` question: enumerate call sites, prove loopback
   only (HALOGEN_NPU_* host flags) or report the exception.
3. Write `analysis/npu-engine.md`.

Acceptance: network claim closed; container formats documented.

## Phase 5 — dflash2 drafter

1. Trace probe: serve_api `_probe`/`_parse_info` fields ↔ engine probe
   emitter (strings `drafter_weights`, `dflash2` in the probe line).
2. Find the implementation: candidate = a drafter path in flash_serve with no
   distinctive strings (compare `draft_sample` siblings in the same object
   file region 0x1809xxx-0x180bxxx).
3. Write `analysis/dflash2.md` (or a bounded negative: searched X, not
   present in 0.17.3, evidence IDs).

## Phase 6 — Closure

1. `build_reconstruction_obligation_ledger` → `record_unknown`/
   `update_unknown` for every open question → `verify_unknown_resolution` /
   `verify_reconstruction` where applicable.
2. `evaluate_reconstruction_coverage` → coverage table in
   `analysis/coverage.md`.
3. `export_evidence_bundle` → `analysis/evidence/`.
4. Rewrite `analysis/report.md` §6 → "Findings complete; residual unknowns:
   <list>". Update AGENTS.md status line.
5. Commit each phase (pushy-style: detailed commit bodies). Push at the end.

## Budget & stops

- Phases 0,3,4,5: hours each. Phase 1: ~1 day (module risk). Phase 2: the
  long pole, 2-4 days of compute; if a single Ghidra batch exceeds ~30 min,
  split per code object.
- Stop conditions requiring Peter: any need to run the engine, attach a
  debugger to a live process, install paid software, or write outside
  ~/re-halogen.
- If Ghidra AMDGCN fails entirely: deliver objdump-level analysis + the
  loader-script groundwork, mark Phase 2 pseudocode as the open item, and say
  so plainly in the final report. No silent scope shrink.

## Report-back format (final message to Peter)

Per phase: one line status + artifact paths + the one most interesting
finding. Total: ≤15 lines.

## Autonomy contract rulings

1. (s3, 2026-10-10) SCOPE: Peter chose "Everything, incl. Ai-core runs" for the next session: finish p7-5 (i4r), then close half-solved unknowns by decompile (p7-6) and by Ai-core engine runs (p7-7), with Hermes allowed to go down during runs. Binds the next session's scope; switches in .claude/autonomy.json unchanged.
2. (s4, 2026-10-10) SCOPE: Peter chose "Proceed as scoped": next session runs p7-8 (static round 4 on U01, U02, U11, U12, U14, U18, U21) then p7-9 (Ai-core run 3 for U26 CSTAT). HERMES: "Allowed, post start/end": Ai-core run windows may take Hermes down without asking, with own cache dir, prune off, auto-restore and a /health check, each window posted at start and end. Binds the next session; switches in .claude/autonomy.json unchanged.
3. (s5, 2026-10-10) SCOPE: Peter chose "Go, in that order": static closures of U02, U11, U21, then U12 by an Ai-core capture, then U22 by rewriting kernels in HIP and matching outputs on Ai-core, then the rebuild-guide small gaps. U02 NPU RUN: "Allowed": qwen3.5-2b NPU model files (about 3.6 GB) may be downloaded to Ai-core and halogen-npu run there, only if static work cannot close U02. U22 SCOPE: "Every kernel", all instances including debug and dead variants. U06: Peter said do not fix (stays OPEN-BOUNDED by decision). U05 stays UNOBSERVABLE by definition.
4. (s6, 2026-10-11) U22/U12 CRITERIA (autopilot pre-flight): (a) U12 closes when scan3 B, C, D, G are proven value-neutral (D's extra load and G's full index range exercised), names labelled inferred where the listing does not prove them; (b) dnc_scan3w probes P5 (unwritten LDS) and P8 (unset VGPRs) count as MATCH with the caveat beside their rows; (c) a constant audit (every literal in each rewrite's compiled ISA equals the vendor kernel's, differences explained) is required before U22 closes; (d) compare.py runs one at a time per host, only while halogen-swift /health is 200 and 8 GiB is free.
5. (s6, 2026-10-11) HERMES: Peter in chat: "I don't need Hermes right now so I don't care if you take it down." halogen-swift stays stopped for the rest of the p7-12 run (stopped 01:01:33Z); restore it and check /health 200 when the run ends. Supersedes 4(d) while it is down.
