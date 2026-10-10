# Live engine run 2: flash_serve 0.17.3 on Ai-core (2026-10-10)

What ran: the same setup as `runtime-run-0.17.3.md` (our `binaries/flash_serve`, sha256 af4f07bb..., mounted over `/usr/local/bin/flash_serve` in the 0.17.2 image, mode `engine`, the live service's models), with the fixes from that run's incident: its own cache directory (`~/re-halogen/work/testcache-r2` as `/cache`, never `/srv/halogen-cache`) and an env copy with `HALOGEN_CACHE_PRUNE_OLD=0`. Run script: `scripts/engine_run_r2.sh`; probe: `scripts/engine_probe4.py` (on `engine_probe.py`). Raw evidence in `analysis/evidence/`: `probe4-0.17.3.jsonl` (every line sent and received, 220 lines), `probe4-0.17.3-results.txt`, `engine-0.17.3-r2.log`, `run-r2-window.txt`, `run-r2-binary.sha256`, `run-r2-health-after.json`. Everything below is observed in those files unless marked inferred. One run, one prompt (40 tokens), small outputs.

## Service window (Hermes down)

| attempt | halogen-swift stopped | restarted | /health 200 | down |
|---|---|---|---|---|
| 1 | 19:03:09Z | 19:04:28Z | 19:05:24Z | 2 min 15 s |
| 2 | 19:05:59Z | 19:09:03Z | 19:09:54Z | 3 min 55 s |

Attempt 1 ended early: the probe crashed on its first 30 s read timeout (`OSError: cannot read from timed out object`, a Python socket file object cannot be read again after a timeout). Its only results were the two INFO lines. The probe now reads with `recv` into its own buffer (`engine_probe.py` `Conn.read`), and the run script keeps the transcript when the probe fails. Attempt 2 is the evidence below.

Cache safety: the test engine used `/cache/d19a28430273f25c` in its own mount (engine log line 57). No prune line in the engine log (`grep -i -E 'prun|subtree'` gives nothing). The live service's cache was already empty before the run: its own log at the stop (19:03:09) reads `prompt cache 0 hits / 0 misses, 0 stores`, and `/srv/halogen-cache/a665fe16b2767d6d` holds only `fingerprint` (dated 16:38:55, before the run).

Health after restore (`curl 127.0.0.1:8307/health`, saved as `run-r2-health-after.json`): `"status":"ok"`, `"version":{"api":"0.17.2","engine":"0.17.2","match":true}`, `"engine":{"responds":true,"probe_s":1.001}`, `"busy":false`.

## U26: a second connection's GEN

Timeline from `probe4-0.17.3.jsonl` (seconds from the first INFO):

| t (s) | connection | event |
|---|---|---|
| 0.000 | first | INFO answered at once |
| 0.000 to 0.723 | second | INFO answered |
| 0.723 | second | GEN 900001 sent |
| 30.7, 60.7 | second | no reply (two 30 s read timeouts) |
| 60.8 | second | PING sent; no reply in 10 s |
| 70.8 to 71.3 | first | GEN 900002 answered in full (8 tokens, `D 900002 length ...`) |
| 71.3 | first | socket closed by the probe |
| 71.3 | second | GEN 900003 sent |
| 71.4 to 71.5 | second | `PONG`, then all of GEN 900001 (8 tokens, `D 900001`), then GEN 900003 (`D 900003`) |
| 71.5 | third | GEN 900004 sent while the second stays open; no reply in 60 s |
| 131.6 to 136.1 | second | five more GENs all answered (seed tests below) |

Result: a second connection's INFO is answered, but its GEN and every line after it (a PING too) wait while the first connection is open: 70 s here, then served within 0.1 s of the first closing, in order and with nothing lost. The third connection then waited the same way behind the second. So one connection at a time is served for GEN; the next one is served when the current one closes. This matches the static accept loop: `main` accepts, serves one fd to completion in `FUN_0182ea80`, closes it, then accepts the next (`main_pseudo.c:1070-1135`, `request-trace.md` step on the listener). Not explained: the second connection's INFO (and, in run 1, a PING sent before any GEN) was answered while the first was open; `main` keeps a pending-fd vector (:1080-1103), the likely place (inferred, not traced). The earlier run's "no CSTAT reply within 10 s" fits the same queueing.

## U23: seed determinism, serial and MTP

Temperature 1.2, top_k 0, top_p 1.0, min_p 0, 32 tokens, same 40-token prompt, all on the second connection after it was served:

| request | drafter | seed | first 12 tokens |
|---|---|---|---|
| 900005 | 0 serial | 111 | 785 6722 315 9625 374 785 97016 95854 46784 271 248068 198 |
| 900006 | 0 serial | 111 | identical to 900005 (all 32) |
| 900007 | 0 serial | 222 | 785 6722 315 9625 374 785 393 15 22 220 133461 98920 |
| 900008 | 1 MTP | 111 | 785 6722 315 987 90 426 13609 8681 220 17 68 2205 |
| 900009 | 1 MTP | 111 | identical to 900008 (all 32) |

`RESULT seeds`: `serial_111_equal: true`, `serial_111_vs_222_equal: false`, `mtp_111_equal: true`, `serial_vs_mtp_111_equal: false`. Serial seed determinism was untested in run 1; it holds here (n = 1 repeat). This matches the static result in `closure/sampler-seed-r3.md`: the draw hashes the seed with the token's sequence position, so the same seed on the same prompt gives the same draws. Serial and MTP differ with the same seed, as in run 1 and as the vendor's /health text says ("agree in distribution, not token for token"). The `D` lines show drafter use (positions 7 to 9 are drafter, rounds, commit per `wire-protocol.md`): MTP requests read `1 18 31`, serial ones `0 0 0`.

## Limits

One prompt, one repeat per seed, 32 tokens. The connection rule is observed behavior; the code path that holds a second connection was not traced statically.
