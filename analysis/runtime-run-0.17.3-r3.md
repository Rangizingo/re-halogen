# Live engine run 3: flash_serve 0.17.3 on Ai-core (2026-10-10)

What ran: the run 2 setup (`runtime-run-0.17.3-r2.md`): our `binaries/flash_serve` (sha256 af4f07bb..., `analysis/evidence/run-r3-binary.sha256`) mounted over `/usr/local/bin/flash_serve` in the 0.17.2 image, mode `engine`, the live service's models, its own cache directory (`~/re-halogen/work/testcache-r3` as `/cache`, never `/srv/halogen-cache`) and an env copy with `HALOGEN_CACHE_PRUNE_OLD=0`. Run script `scripts/engine_run_r3.sh` (run 2's script with its own names, refusing an existing out dir); probe `scripts/engine_probe5.py` (on `engine_probe.py`). One attempt, out dir `~/re-halogen/work/run-r3-a1`. Raw evidence in `analysis/evidence/`: `probe5-0.17.3.jsonl` (every line sent and received, 56 lines), `probe5-0.17.3-results.txt`, `engine-0.17.3-r3.log`, `run-r3-window.txt`, `run-r3-health-before.json`, `run-r3-health-after.json`. Everything below is observed in those files unless marked inferred. One run, one prompt (40 tokens), 8-token outputs.

## Service window (Hermes down)

| attempt | halogen-swift stopped | restarted | /health 200 | down |
|---|---|---|---|---|
| 1 | 19:40:54Z | 19:43:23Z | 19:44:24Z | 3 min 30 s |

Cache safety: the test engine used `/cache/d19a28430273f25c` in its own mount (engine log line 57, `0 records in 0 lineages`). No prune line in the engine log (`grep -i -E 'prun|subtree' analysis/evidence/engine-0.17.3-r3.log` prints nothing).

Health after restore (`curl 127.0.0.1:8307/health`, `analysis/evidence/run-r3-health-after.json`), first fields:

```
{"status":"ok","model":"halogen-qwen3.8-flash-next", ... "version":{"api":"0.17.2","engine":"0.17.2","match":true}, ... "engine":{"responds":true,"probe_s":1.002},"busy":false,"slots":4, ... "in_flight":0, ... "queued":0, ...}
```

## U26: CSTAT on a waiting connection, GEN completion vs close, third-connection release

Probe order (`engine_probe5.py` docstring): first INFO; second INFO, then CSTAT and PING sent back to back and read for 15 s; first GEN; first then stays open and idle for 20 s while the second is read; first CSTAT, first closes; second read for 15 s; third INFO, CSTAT (read 10 s), GEN (read 20 s) while the second stays open; second CSTAT, second closes; third read for 20 s; third CSTAT.

Timeline from `probe5-0.17.3.jsonl` (seconds from the first INFO):

| t (s) | connection | event |
|---|---|---|
| 0.00 | first | INFO answered at once |
| 0.00 to 0.67 | second | INFO answered |
| 0.67 | second | CSTAT and PING sent |
| 0.67 to 15.67 | second | no reply (15 s read timeout) |
| 15.67 to 16.25 | first | GEN 900001 answered in full (8 tokens, `D 900001 length 40 8 474.1 ...`) |
| 16.25 to 36.27 | second | no reply while the first is open and idle after its GEN (20 s read timeout) |
| 36.27 | first | CSTAT answered, then the socket is closed by the probe |
| 36.27 | second | the queued CSTAT reply (`C 1 116389652 ...`) then `PONG`, in send order |
| 51.29 | third | INFO answered at once |
| 51.29 to 61.30 | third | CSTAT sent; no reply (10 s) |
| 61.30 to 81.32 | third | GEN 900002 sent; no reply while the second is open (20 s) |
| 81.32 | second | CSTAT answered, then the socket is closed |
| 81.32 to 81.42 | third | the queued CSTAT reply, then GEN 900002 in full (8 tokens, `D 900002 length 40 8 0.2 ...`) |
| 101.34 | third | CSTAT answered at once |

Results (`probe5-0.17.3-results.txt`):

- CSTAT on a waiting connection: not answered while another connection is being served (second: 15 s and then 20 s more; third: 10 s and then 20 s more). It is answered, in order, as soon as the connection is served. Run 1's "no CSTAT reply within 10 s" is the same queueing.
- PING behind a CSTAT on a waiting connection also waits (second: answered after the CSTAT, at release). Run 1 saw a PING answered on a waiting connection when it was the first line sent; here it followed a CSTAT. Inferred: a waiting connection gets inline answers only for lines before its first queued verb; the static path is in `analysis/closure/serve-pending-fd-r4.md`.
- GEN completion is not the release: the first connection's GEN finished at 16.25 s and the second got nothing during the next 20 s while the first stayed open (`RESULT second_release` split `before_close: 0, after_close: 2`).
- Release is on close: the second's queued lines came at 36.27 s, when the first closed. Limit: in both releases of this run the serving connection sent CSTAT less than 1 ms before closing, so close and that CSTAT are not separated here. Run 2 separates them: there the first connection closed right after its GEN with no CSTAT, and the second was released within 0.1 s (`runtime-run-0.17.3-r2.md`). Close is the trigger in both runs.
- Third-connection release observed: the third's CSTAT and GEN waited 30 s while the second was open and were answered from 0.0 s after the second closed (`first_line_after_close_s: 0.0`), all 8 tokens and `D` by 0.102 s after it (t 382.040 to 382.142 in the jsonl).
- INFO is answered at once on every waiting connection (second at 0.67 s, third at 0.00 s).

This matches the accept loop: `main` serves one fd in `FUN_0182ea80` until it closes, `close(fd)`, then takes the oldest fd from its pending vector, or calls `accept` only when the vector is empty (`work/decomp/flash_serve/0182ba00.c:1116-1117`, `analysis/main_pseudo.c:1078-1112`). The other accept site is the queue-while-busy one in `FUN_01844d50` (`closure/serve-pending-fd-r4.md`). Each run rules out one alternative: run 3 shows GEN completion alone does not release (20 s idle), run 2 shows a release with no CSTAT before the close (`engine_probe4.py` closes right after its GEN). The close time is the probe's own `t_close`, not a server log.

CSTAT reply on a served connection (`C 1 116389652 116389652 24 0 1 1 0 0 21.9 ...`, 51 fields after `C`) is the same on every connection before the third's GEN; after it four fields change (counting `C` as 0): 5 goes 0 to 1, 9 goes 0 to 40 (the third's 40 prompt tokens), 11 goes 0.0 to 5.7, 38 goes 0 to 1 (computed by a field-wise compare of the two lines in `probe5-0.17.3-results.txt`; the prompt was cached by the first GEN, and the third's `D` line reads prefill 0.2 ms against 474.1 ms for the first). Field names are in `wire-protocol.md`; this run does not add to them.

## Limits

One run, one prompt, 8-token outputs, one idle window of 20 s. CSTAT before close is not separated from close inside this run (see above; run 2 covers it).
