# Live engine run: flash_serve 0.17.3 on Ai-core (2026-10-10)

What ran: our extracted `binaries/flash_serve` (sha256 af4f07bb..., 0.17.3) mounted over the engine binary inside the 0.17.2 image userland, mode `engine` (engine plus watchdog, no API front end), the live service's models and env file, on Ai-core (Strix Halo, gfx1151, 121 GB). The Hermes-serving container `halogen-swift` (0.17.2) was stopped for about 15 minutes and restored; its route returned health 200 about 48 s after restart. Raw transcripts: `analysis/evidence/probe-0.17.3.jsonl`, `probe2-0.17.3.jsonl`, `probe3-0.17.3.jsonl`, `engine-0.17.3-run.log`; the live 0.17.2 engine before the stop: `live-0.17.2-startup-and-info.txt`. Probe clients: `scripts/engine_probe.py`, `engine_probe2.py`, `engine_probe3.py`. Everything below is observed unless marked inferred. One run, two prompts, 64-token outputs: small samples.

## Incident: the run deleted the live service's disk prompt cache

The run reused the service's env file and mounted its `/cache`. That env sets `HALOGEN_CACHE_PRUNE_OLD=1`, so on startup the new engine removed "another configuration's subtree a665fe16b2767d6d (60.25 GiB)" (log line 57). The live service's on-disk prompt cache went from 61 GiB to 8 KB, 0 records. It is regenerable derived data (saved conversation prefixes), not weights, and not recoverable: long Hermes conversations pay one full prefill. The restarted service then pruned the 0.17.3 run's own empty subtree. Cause: my mistake, no separate cache directory and no `HALOGEN_CACHE_PRUNE_OLD=0`. Fix for any future run: give the test container its own `/cache` and set the flag to 0.

## Closed by observation

| item | result | evidence |
|---|---|---|
| U04 raw `GEN ... 2` (dflash2) | engine refuses: `D <req> error 0 0 0 0 asked for drafter 2 (unknown)`; engine log "request refused: asked for drafter 2 (unknown)" | probe-0.17.3.jsonl, run log line 75 |
| dflash2 capability | INFO field 7 is 0 on both engines | INFO lines in all transcripts |
| INFO `version` field (field 17) | comes from env `HALOGEN_IMAGE_VERSION`, not the binary: the 0.17.3 binary reported 0.17.2 until the env was set to 0.17.3 | binary hash check, two runs |
| connection roles (U26) | on a second connection `INFO` and `PING` answer, `CSTAT` gets no reply in 10 s; on the live engine four raw `GEN` attempts from a second connection got none. All verbs work on the first connection | probe3-0.17.3.jsonl, live-0.17.2-gen-drafters*.txt |

## Behavior measurements

Prompts P (19 tokens) and Q (25 tokens), real tokens from the model tokenizer, 64 generated tokens, greedy unless noted.

- Greedy, MTP drafter (1) cold vs warm cache: identical tokens. `n_cached` 0 then 19.
- Greedy, MTP vs serial (drafter 0): identical on P (warm both) and on Q (serial cold vs MTP warm). Speculation did not change greedy output in these 2 prompts.
- Sampling (temperature 1.2, top_k 0, top_p 1.0, min_p 0): same seed 111 twice gives identical tokens; seed 222 differs; MTP seed 111 vs serial seed 111 differ. So seeds reproduce on a given drafter, but a speculative run is not the same sample sequence as a serial run, even though both are valid draws.
- The engine's own startup text for the in-place prompt cache says "a warm answer is NOT bitwise the cold one", and for `HALOGEN_ADMIT_CHUNK=0` "the engine's own chunk keeps the admitted prompt bitwise its alone self". The token-level equality above does not contradict the first statement (it concerns numerics, not tokens, and 64 tokens is short).
- `D` line fields matched `analysis/wire-protocol.md` (29 fields on done lines; `n_cached` at position 10, drafter, rounds, commit at 7-9).
- Startup: GGUF repack 35.6 s, engine ready about 36 s after container start, listening on 8730 with 4 slots over one 524,288-position pool.

## Limits

One engine, one run, small prompts. The seed test shows determinism per drafter, not the draw's source (U23 stays open: splitmix64 is in the pick kernels, the seed path is untraced). The probe cannot see logits, so "bitwise" claims are untested beyond token equality.
