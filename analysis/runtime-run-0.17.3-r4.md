# Live engine run 4: kernel-argument capture and launch trace, flash_serve 0.17.3 on Ai-core (2026-10-10)

What ran: the run 3 setup (`runtime-run-0.17.3-r3.md`) with an LD_PRELOAD shim (`scripts/capture/kargs_shim.c`, built by `scripts/capture/build_shim.sh`) that logs every kernel launch in order and, for each `k_dnc_scan3*` launch, its kernel-argument bytes and which host globals its NormFold pointers equal. Run script `scripts/engine_run_r4.sh` (own cache dir `work/testcache-r4`, `HALOGEN_CACHE_PRUNE_OLD=0`, `HALOGEN_DN_NORM_FOLD=1`, `HALOGEN_TAP_FROM_SCAN=1`); probe `scripts/engine_probe6.py` (prompts of 300 and 1100 tokens, 4 output tokens each). Evidence in `analysis/evidence/run-r4-*`: `kargs.jsonl.gz` (raw shim log, 27,833 lines; the run window counted 27,821 while the engine was still running), `kargs-table.txt` (per scan3 launch), `kargs-trace.txt` (ordered kernel names per request), `probe6-0.17.3.jsonl`, `probe6-results.txt`, `engine-0.17.3-r4.log`, `window.txt`, `health-after.json`, `binary.sha256` (flash_serve af4f07bb... and the shim). Observed unless marked inferred.

## Service windows (Hermes down)

| attempt | stopped | /health 200 | down | result |
|---|---|---|---|---|
| 1 | 21:28:37Z | 21:30:10Z | 1 min 33 s | engine exited on the first GEN; log lost (container started with `--rm`) |
| 2 | 21:30:53Z | 21:32:33Z | 1 min 40 s | same exit; log kept: `HIP /src/halogen/src/flash_ops.h:4060: invalid argument` (`run-r4-attempt2-engine-tail.txt`) |
| 3 | 21:34:42Z | 21:36:29Z | 1 min 47 s | all 5 requests answered |

Cause of attempts 1 and 2 (most likely cause, inferred; a run with `HALOGEN_DN_NORM_FOLD=1` and no shim was not made): line 4060 is a `hipGetLastError()` check after the `k_deq_q8s32` launch (`work/decomp/flash_serve/018a7520.c:273`, `uVar6 = 0xfdc`). HIP's last error is sticky per thread, so the engine read an error left by the shim's own diagnostic `hipMemPtrGetInfo` on an interior arena pointer (attempt 1's shim log, `run-r4-attempt1-kargs.jsonl`, shows the `"read":4096` fallback that the shim takes when `hipMemPtrGetInfo` fails; the same fallback also covers a pointer with no known size, so this is consistent with, not proof of, the failure). Attempt 3 kept `HALOGEN_DN_NORM_FOLD=1` and changed only the shim and `--rm`, and all its captures carry `shim_err_pre` and `shim_err_post` 0 after clearing. Fix: the shim skips its diagnostics while an error is pending and clears its own errors after them (`shim_err_pre`, `shim_err_post` fields); `scripts/capture/selftest_shim.sh` checks both with a fake sticky-error HIP. `--rm` was dropped from the run script so the log survives.

Cache safety: the engine used `/cache` in its own mount only; `find work/testcache-r4` after the run lists `2c917302cce55089/fingerprint` and nothing else (12K). No disk records were written in this run (the container is removed with `rm -f`, not shut down; inferred: records are written behind or at shutdown).

Health after restore (`run-r4-health-after.json`): `"status":"ok"`, `"version":{"api":"0.17.2","engine":"0.17.2","match":true}`, `"engine":{"responds":true,...}`.

## U12: NormFold pointers (k_dnc_scan3x)

`kargs-table.txt`: 108 `k_dnc_scan3x<2,4,1,0,1,1,0,1,0>` launches captured (grid 48x2x1, kernarg 208 B, rc 0). In every one, n1 equals the value of host global `DAT_019fc638`, n2 equals `DAT_019fc660`, n3 equals `DAT_019fc7b8` at launch time, and w = 128. This confirms the static host sources (`closure/dnc-scan3-r4.md` section 3) at run time. n0 matches no global (a per-layer weight pointer, as the report says). What those arena regions hold beyond their writers is not shown by the capture.

Tap: all Tap slots were -1 in all 108 launches, including the 1100-token extension of a cached 300-token prompt. No checkpoint tap fired, so the Tap payload is not observed. Inferred from `018fa150.c` (tap targets are set from an int[3] argument): the targets are set by a caller this probe does not reach.

## Launch trace (all kernels, in order)

`kargs-trace.txt` splits the trace into the 5 requests by idle gaps. Totals over the run (python over `kargs.jsonl.gz`): 25,333 kernel launches of 106 distinct kernel instances in 73 families, plus 2,008 `hipblasLtMatmul` calls. So the default greedy path with the MTP drafter, at these prompt lengths, uses 106 of the 1,738 kernel instances.

Confirmed live from the trace (request 0, 300-token prefill): routed experts run `k_route_topk_w`, `k_route_hist`, `k_route_scan`, `k_route_scatter`, `k_q4moe_items<64>`, `k_q4moe<1,0,64,0,128,0>`, `k_q4moe<1,2,64,0,128,0>`, `k_slot_moe_sum_rows`, as `closure/expert-dispatch-r5.md` predicts for 64 <= n <= 4966 with plain Q4C-P experts; dense layers run `k_deq_q4cp_v8u<2,0>` or `k_deq_q8s32` then `hipblasLtMatmul`; DeltaNet layers run `k_conv1d_silu_tb<16>`, `k_dn_gates`, `k_dnc_prep_w`, `k_dnc_ut5<1,1,3,1,1,0>`, `k_dnc_scan3x<...>`, `k_conv_state_from_prefill`.

The decode steps of each request (MTP drafter) are in the same trace; their reading for the MTP head order (`closure/mtp-head-r5.md` residuals 1 to 3) is left to the next session.

## Limits

One run, greedy only, 5 prompts. No tap fired; no disk record was written. The shim synchronizes around each scan3 launch, so timings in this run are not engine timings.
