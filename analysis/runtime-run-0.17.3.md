# Live engine run: flash_serve 0.17.3 on Ai-core (2026-10-10)

What ran: our extracted `binaries/flash_serve` (sha256 af4f07bb..., 0.17.3) mounted over the engine binary inside the 0.17.2 image userland, mode `engine` (engine plus watchdog, no API front end), the live service's models and env file, on Ai-core (Strix Halo, gfx1151, 121 GB). The Hermes-serving container `halogen-swift` (0.17.2) was stopped for about 15 minutes and restored; its route returned health 200 about 48 s after restart (session-captures.md; the 15 minutes is an estimate). Raw transcripts: `analysis/evidence/probe-0.17.3.jsonl`, `probe2-0.17.3.jsonl`, `probe3-0.17.3.jsonl`, `engine-0.17.3-run.log`; the live 0.17.2 engine before the stop: `live-0.17.2-startup-and-info.txt`. Probe clients: `scripts/engine_probe.py`, `engine_probe2.py`, `engine_probe3.py`. Everything below is observed unless marked inferred. One run, two prompts, 64-token outputs: small samples.

## Incident: the run deleted the live service's disk prompt cache

The run reused the service's env file and mounted its `/cache`. That env sets `HALOGEN_CACHE_PRUNE_OLD=1`, so on startup the new engine removed "another configuration's subtree a665fe16b2767d6d (60.25 GiB)" (log line 57). The live service's on-disk prompt cache went from 61 GiB to 8 KB, 0 records (hand-copied terminal output in `session-captures.md`, not a machine capture; the log lines in `engine-0.17.3-run.log` 57-58 are saved). It is regenerable derived data (saved conversation prefixes), not weights, and not recoverable: long Hermes conversations pay one full prefill. The restarted service then pruned the 0.17.3 run's own empty subtree (session-captures.md). Cause: my mistake, no separate cache directory and no `HALOGEN_CACHE_PRUNE_OLD=0`. Fix for any future run: give the test container its own `/cache` and set the flag to 0.

## Closed by observation

| item | result | evidence |
|---|---|---|
| U04 raw `GEN ... 2` (dflash2) | engine refuses: `D <req> error 0 0 0 0 asked for drafter 2 (unknown)`; engine log "request refused: asked for drafter 2 (unknown)" | probe-0.17.3.jsonl, run log line 75 |
| dflash2 capability | INFO field 7 is 0 on both engines | INFO lines in all transcripts |
| INFO `version` field (field 17) | follows env `HALOGEN_IMAGE_VERSION`, not the binary. Saved transcripts show only 0.17.2 (the env as shipped); the 0.17.3 reply after setting the env was seen in the session terminal and is copied by hand in `session-captures.md`. Also consistent with `entrypoint-flow.md` line 14 (the banner uses the env var) | probe-0.17.3.jsonl (0.17.2), session-captures.md (0.17.3), manual |
| connection roles (U26) | on 0.17.3, a second connection got `INFO` and `PING` answers and no `CSTAT` reply within 10 s (one run). On the live 0.17.2 engine, five raw `GEN` attempts (3 in the first file, 2 in the retry) got no reply, but those connections were not labeled second in the files: the front end held the first. On 0.17.3 the second-connection `GEN` result was not recorded (the probe stopped at the send). The first connection answered every verb | probe3-0.17.3.jsonl, live-0.17.2-gen-drafters*.txt |

## Behavior measurements

Prompts P (19 tokens) and Q (25 tokens), real tokens from the model tokenizer, 64 generated tokens, greedy unless noted.

- Greedy, MTP drafter (1) cold vs warm cache: identical tokens. `n_cached` 0 then 19.
- Greedy, MTP vs serial (drafter 0): identical on P (warm both) and on Q (serial cold vs MTP warm). Speculation did not change greedy output in these 2 prompts.
- Sampling (temperature 1.2, top_k 0, top_p 1.0, min_p 0), one comparison each: MTP with seed 111 twice gave identical tokens; seed 222 differed; MTP seed 111 vs serial seed 111 differed. Serial seed determinism was not run twice, so it is untested. A speculative run was not the same sample sequence as the serial run in this one case.
- The engine's own startup text for the in-place prompt cache says "a warm answer is NOT bitwise the cold one", and for `HALOGEN_ADMIT_CHUNK=0` "the engine's own chunk keeps the admitted prompt bitwise its alone self". The token-level equality above does not contradict the first statement (it concerns numerics, not tokens, and 64 tokens is short).
- `D` line fields matched `analysis/wire-protocol.md` (30 whitespace-separated parts on done lines, 12 on the error line; `n_cached` at position 10, drafter, rounds, commit at 7-9).
- Startup: GGUF repack 35.6 s, engine ready about 36 s after container start, listening on 8730 with 4 slots over one 524,288-position pool.

## Unexplained

Every request in the engine log reads `generated 2 tokens, 0.0 s` while the transcripts show 64 tokens delivered per request. The log line's meaning (per round, per stream, or something else) is unknown.

## Limits

One engine, one run, small prompts. The seed test shows determinism for MTP only (n=1), not the draw's source (U23 stays open: splitmix64 is in the pick kernels, the seed path is untraced). The probe cannot see logits, so "bitwise" claims are untested beyond token equality.
