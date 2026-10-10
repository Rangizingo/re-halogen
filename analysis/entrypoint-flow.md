# entrypoint.sh: how the container starts and supervises the engine

Source: `image/entrypoint.sh` (1534 lines, comments stripped by the vendor, line 2) and `image/halogen-healthcheck` (35 lines).
Citations are `file:line` against `image/entrypoint.sh` unless marked `hc:` (healthcheck). "inferred" means the script does not say it outright.

## 1. Boot sequence (default mode `all`, line 1370-1429)

Shell setup, before any mode runs:
1. `set -euo pipefail`; `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1` default; `ulimit -c 0` (core dumps off). Lines 3-6.
2. Ports and bind: engine 8730, API 8731, bind 127.0.0.1. Lines 8-10. `HALOGEN_VERBOSE` defaults to 0 and is exported. Line 12.
3. Checkpoint resolution: default `/models/qwen38-flash-next-v2.hgn`; falls back to `qwen38-flash-next-w4b.hgn` only if v2 is absent and w4b present (sets LEGACY_W4B). Lines 14-29.
4. Engine sizing: slots 4, ctx 262144, KV pool positions default `min(2*ctx, 1048576)` and not below ctx, max-tok 32768. Lines 31-41.
5. Long-context rules. ctx above 262144 caps max-tok at 16384 (42-46), routes the prompt cache snapshot to `/var/tmp/halogen-cache.snapshot` when in-place cache is off (47-51), and refuses to start (exit 2) unless `HALOGEN_ROPE_YARN` is set (56-59). YaRN set at ctx 262144 or below only warns (60-62). max-tok is clamped to ctx (53).
6. Banner: prints `halogen-flash-server $HALOGEN_IMAGE_VERSION, mode all`. Lines 1366-1367.

Mode `all`, in order:
1. `_hg_npu_alone` plus `HALOGEN_NPU_MODELS`: NPU-only path (`start_npu_alone`). Line 1374. The `_hg_*` variables look like test hooks (inferred; no HALOGEN_ prefix, used as `${_hg_x:-default}` throughout).
2. `need_ckpt` (line 518-535), which runs in this order:
   - `host_preflight` (215-243): needs `/dev/kfd` (WSL2 `/dev/dxg` gets a specific refusal), warns on missing render node, reads `gfx_target_version` from KFD topology and refuses anything but 110501 (gfx1151, Strix Halo).
   - `maybe_download` (319-410): only if `HALOGEN_DOWNLOAD` is set. Uses `hf download` of the .hgn, overlay, ngram table, vision sidecar and tokenizer, with a free-space check. GGUF files are never downloaded (331-335).
   - `tuning_plan_copy` (435-444): copies the baked `/opt/halogen/flash-tune.plan` to `/tmp/halogen-tune.plan` when `HALOGEN_MATMUL_TUNING_FILE` points at the baked path, because the engine "may rewrite it at exit" (442).
   - Checkpoint must exist (524-528). Then `check_gguf` (568-588) if the first 4 bytes are `GGUF`, else `check_sidecar` (645-685).
   - `legacy_note`, `check_ngram_table` (480-516, exports `HALOGEN_NGRAM_TABLE` when the .hgn carries no table), `check_vision` (597-630, exports resolved `HALOGEN_VISION_TOWER`), `kv_budget_note` (64-110, advisory), `gtt_note` (147-213, can refuse with exit 1 when free GTT is below the need).
3. `need_tokenizer` (687-701): requires `tokenizer.json`; falls back to `<models dir>/tokenizer` if `/tokenizer` is not mounted.
4. `check_defaults` (937-943): runs `serve_api.py --check-defaults`; an invalid request default stops boot.
5. Optional NPU engine (`HALOGEN_NPU_MODELS`): `start_npu_engine 1`, 1287-1335. Runs `halogen-npu listen ...` on port 8740, waits for a `READY` line.
6. Engine launch (1379-1381), backgrounded. Waits via `wait_for_engine` (703-730): polls TCP connect every 2 s, prints a heartbeat every 60 s, no timeout unless `HALOGEN_ENGINE_WAIT_S` > 0. Engine exit before listening is fatal.
7. API launch (1393-1399), backgrounded, only after the engine port accepts a connection.
8. Watchdog launch (1402-1408) unless `HALOGEN_ENGINE_WATCHDOG_S=0`.
9. `wait -n` on engine, API, watchdog and NPU (1412). Any exit tears everything down and the script exits 1 (1429). Details in section 4.

Other modes (1370-1534): `engine` (`start_engine`, engine plus watchdog, no API), `api` (`start_api`, `exec serve_api.py`, `--engine` defaults to `127.0.0.1:$ENG_PORT`, 945-956), `bench` and `sweep` (1431-1485, engine plus API on localhost, then `bench-serving.py` or `halogen-bench.py`), `convert` (1486-1510), `inspect|verify|ppl|niah` (1511-1531). Usage line at 1533.

### GGUF to .hgn repack: where it really happens
- The task brief assumed `halogen-tools` repacks GGUF. It does not. Repack is `flash_serve --repack IN --out OUT --with-table --head $HALOGEN_MTP_HEAD`, line 1501, mode `convert` only. Output is "about 105 GiB for an IQ4_XS build" (1492).
- On a normal start with a GGUF checkpoint there is no separate repack step. The script hands the GGUF straight to `flash_serve --ck` and says it "is repacked into RAM at startup, losslessly, on every start" (570), about 20 s cold and 9 s warm on the reference machine (571). So the repack lives inside the engine binary (the script text states it; the mechanism is inferred).
- `HALOGEN_GGUF_CACHE` (1 or a directory) keeps the repack beside the GGUF or in that directory, about 70 GiB for an 8-bit trunk; a warm restart is about 1 s (571-580). The script only validates the path (578). The engine does the writing (inferred).
- A GGUF trunk needs the engine's own draft head `qwen38-flash-next-mtp.hgn` (magic `HGN1`, 1.4 GiB), checked at 558, exported at 565. A llama.cpp MTP GGUF is rejected (560).
- `halogen-tools` is used only for `inspect|verify` (line 1530). `ppl|niah` go through `python3 /halogen/tools/halogen_tools.py` instead (1528). Both modes get the baked tuning plan copy and the checkpoint's table check first (1521-1525).
- Not found in the script: `hostprep`, `drop_caches`, `hugepage`, `numactl`. No `exec` before the API (the API is a background child, except `start_api` at 949 and the tools modes). `ulimit` appears only at line 6 (core off), 872 and 1275 (memlock read, never set).

## 2. Engine command line and flag routing

Engine (identical in `engine`, `all`, `bench`; lines 904-906, 1379-1381, 1439-1441):

```
env -u <each FE_ONLY flag> /usr/local/bin/flash_serve \
  --ck $HALOGEN_CHECKPOINT --port $ENG_PORT --bind <127.0.0.1 in all/bench; $BIND in engine mode> \
  --slots $ENG_SLOTS --ctx $ENG_CTX --max-tok $ENG_MAX_TOK --kv-pool $ENG_POOL
```

- Only seven values go as argv: `--ck --port --bind --slots --ctx --max-tok --kv-pool`. Everything else the engine learns comes from the environment (inferred: the script wraps it in `env -u` and exports several HALOGEN_ vars it computes).
- Exported by the script for the engine to read (inferred reader = flash_serve): `HALOGEN_CHECKPOINT` (29), `HALOGEN_CACHE_FILE` (49), `HALOGEN_MATMUL_TUNING_FILE` (440), `HALOGEN_NGRAM_TABLE` (503), `HALOGEN_MTP_HEAD` (565), `HALOGEN_VISION_TOWER` (617).
- Mode `all` forces `--bind 127.0.0.1` (1380). Mode `engine` uses `$BIND` (905), which is how the engine can be exposed on its own.

`FE_ONLY_FLAGS` (889-895, 22 names), unset for the engine with `env -u` (896-898):
`ENABLE_THINKING ENGINE_PING_S ENGINE_PROBE_S FREQUENCY_PENALTY KEEPALIVE_TIMEOUT MAX_THINKING_TOKENS MAX_TOKENS_DEFAULT MIN_P MODEL_ID PRESENCE_PENALTY REASONING_EFFORT RECORD_DAYS RECORD_DIR RECORD_GIB RECORD_TEXT REPETITION_PENALTY SSE_KEEPALIVE_S TEMPERATURE TEMPLATE_UNCHECKED THINKING_ANSWER_ROOM TOP_K TOP_P` (all prefixed `HALOGEN_`).

The front end is `python3 /halogen/tools/serve_api.py` with argv (1393-1399):
`--tokenizer $HALOGEN_TOKENIZER --engine 127.0.0.1:$ENG_PORT --host 0.0.0.0 --port $API_PORT --max-tokens-cap ${HALOGEN_MAX_TOKENS_CAP:-65536} --context $ENG_CTX --queue-timeout ${HALOGEN_QUEUE_TIMEOUT:-3600}` plus `--npu 127.0.0.1:8740 --npu-tokenizer <dir>` per NPU model. The API always listens on 0.0.0.0 and ignores `HALOGEN_BIND` (inferred from 1396, 952).

Flags that serve_api.py reads from the environment but this script never mentions (grep of `image/serve_api.py`): `HALOGEN_SCHEMA_ESCAPE`, `HALOGEN_ADMISSION_RESERVE`, `HALOGEN_EOS_GUARD`, `HALOGEN_THINKING_BUDGET_MESSAGE`, `HALOGEN_IMAGE_MODES`. They are not in FE_ONLY_FLAGS, so the engine process inherits them too (harmless if it ignores them; inferred).

## 3. HALOGEN_* flag table

68 distinct names in `entrypoint.sh` (`grep -oE 'HALOGEN_[A-Z0-9_]+' | sort -u | wc -l`), 3 of them retired (marked gone). The healthcheck adds none new (reads PORT, API_PORT, ENGINE_PING_S).
Reader key: S = this script, E = flash_serve (inferred when the script only passes the env through), F = serve_api.py front end, N = NPU path.

| Flag | Default | Effect | Reader |
|---|---|---|---|
| HALOGEN_PORT | 8730 | engine TCP port (8) | S, E argv, hc |
| HALOGEN_API_PORT | 8731 | API port (9) | S, F argv, hc |
| HALOGEN_BIND | 127.0.0.1 | engine bind in `engine` mode only (10, 905) | S, E argv |
| HALOGEN_VERBOSE | 0 | exported (12); consumer not shown | E and/or F, inferred |
| HALOGEN_CHECKPOINT | /models/qwen38-flash-next-v2.hgn (w4b fallback) | trunk file, .hgn or GGUF shard (23-29) | S, E argv |
| HALOGEN_KV_SLOTS | 4 | `--slots` (31) | S, E argv |
| HALOGEN_CTX | 262144 | `--ctx` and `--context` (32, 1398) | S, E, F |
| HALOGEN_KV_POOL_POSITIONS | unset, then min(2*ctx, 1048576), not below ctx | `--kv-pool` (34-39) | S, E argv |
| HALOGEN_MAX_TOK | 32768 (16384 cap above native ctx) | `--max-tok`, prefill arena size (41-53) | S, E argv |
| HALOGEN_ROPE_YARN | unset | static YaRN factor, mandatory above 262144 (4 for 1M, 2 for 524288) (55-62) | S check, E inferred |
| HALOGEN_PROMPT_CACHE | 2 | 0 disables prompt cache; used in memory estimate (47, 69) | S estimate, E |
| HALOGEN_CACHE_INPLACE | 1 | 0 means cache snapshot copies (47, 69) | S estimate, E |
| HALOGEN_CACHE_FILE | unset (auto `/var/tmp/halogen-cache.snapshot` at ctx>262144 with inplace=0) | snapshot file (49) | S sets, E |
| HALOGEN_CACHE_ENTRIES | unset (computed from slots, branches, snap3, full) | cache entry count (69) | S estimate, E |
| HALOGEN_CACHE_BRANCHES | 2 | cache branches per slot (69) | S estimate, E |
| HALOGEN_CACHE_SNAP3 | 1 | cache snapshot variant, 0 halves entries per branch (69) | S estimate, E, F (serve_api also reads it) |
| HALOGEN_CACHE_FULL | 1 | 0 drops the full-prompt entry (69) | S estimate, E |
| HALOGEN_KV_POOL | 1 | 0 means per-slot KV, 26624 B/pos/slot, instead of one pool (67, 96) | S estimate, E |
| HALOGEN_VISION_TOWER | unset | unset: sidecar ignored; 0/no/false/off: off; 1/auto/yes/true: use `qwen38-flash-next-vision.hgn` beside checkpoint; else a path (597-626) | S resolves and exports, E, F |
| HALOGEN_VISION_MAX_PIXELS | 2560x1440 (stated in text only) | caps image size, downscales above (628-629) | E and/or F (serve_api reads it) |
| HALOGEN_UMA_CARVEOUT_GIB | auto from `mem_info_vram_total` | override for the BIOS carve-out hint (197) | S |
| HALOGEN_TOKENIZER | not set in the script; `/tokenizer` assumed by 276 and 689 (default likely from image ENV, inferred) | tokenizer dir with `tokenizer.json` (276, 687-700) | S, F argv |
| HALOGEN_NGRAM_TABLE | unset; script sets to `<ckpt dir>/qwen38-flash-next-ngram.hgn` when the .hgn lacks a table | lookup table path (283, 483-503) | S, E |
| HALOGEN_DOWNLOAD | unset | HF repo to fetch missing files from (320) | S |
| HALOGEN_MATMUL_TUNING_FILE | unset (baked path `/opt/halogen/flash-tune.plan` is the image value, 436) | tuning plan; copied to /tmp (437-440) | S, E |
| HALOGEN_MTP_HEAD | `<ckpt dir>/qwen38-flash-next-mtp.hgn` | draft (MTP) head; GGUF trunks only; must start `HGN1` (544-565) | S, E |
| HALOGEN_GGUF_CACHE | unset | keep the GGUF repack: 1 beside, or a directory (571-580) | S validates, E |
| HALOGEN_CK_OVERLAY | unset (sidecar `<ckpt>.overlay.hgn` beside) | `none`/`0` runs bare checkpoint, or a path (646-661) | S, E |
| HALOGEN_ENGINE_WAIT_S | 0 (wait forever) | boot wait limit (705) | S |
| HALOGEN_ENGINE_PING_S | 30 | PING reply timeout (798, hc:5) | S, hc, F |
| HALOGEN_ENGINE_WATCHDOG_S | 180 | seconds of silence before takedown; 0 turns the watchdog off (804, 1403) | S |
| HALOGEN_ENGINE_WATCHDOG_DEFER_S | 900 | cap on deferred silence under memory compaction; 0 disables the cap (811, 835) | S |
| HALOGEN_WEIGHTS_LOCK | 0 | 1 asks the engine to mlock weights; script only warns on low memlock (871-886) | S warns, E |
| HALOGEN_ENABLE_THINKING | in F | request default (FE_ONLY) | F |
| HALOGEN_ENGINE_PROBE_S | in F | FE-side engine probe interval (FE_ONLY) | F |
| HALOGEN_FREQUENCY_PENALTY | in F | sampling default (FE_ONLY) | F |
| HALOGEN_KEEPALIVE_TIMEOUT | in F | HTTP keepalive (FE_ONLY) | F |
| HALOGEN_MAX_THINKING_TOKENS | in F | thinking budget (FE_ONLY) | F |
| HALOGEN_MAX_TOKENS_DEFAULT | in F | default completion length (FE_ONLY) | F |
| HALOGEN_MIN_P | in F | sampling default (FE_ONLY) | F |
| HALOGEN_MODEL_ID | in F | model name reported (FE_ONLY) | F |
| HALOGEN_PRESENCE_PENALTY | in F | sampling default (FE_ONLY) | F |
| HALOGEN_REASONING_EFFORT | in F | request default (FE_ONLY) | F |
| HALOGEN_RECORD_DAYS | in F | request recorder retention (FE_ONLY; `request_records.py` is the likely reader, inferred) | F |
| HALOGEN_RECORD_DIR | in F | recorder directory (FE_ONLY) | F |
| HALOGEN_RECORD_GIB | in F | recorder size cap (FE_ONLY) | F |
| HALOGEN_RECORD_TEXT | in F | recorder stores text (FE_ONLY) | F |
| HALOGEN_REPETITION_PENALTY | in F | sampling default (FE_ONLY) | F |
| HALOGEN_SSE_KEEPALIVE_S | in F | SSE keepalive interval (FE_ONLY) | F |
| HALOGEN_TEMPERATURE | in F | sampling default (FE_ONLY) | F |
| HALOGEN_TEMPLATE_UNCHECKED | in F | skip chat-template check (FE_ONLY) | F |
| HALOGEN_THINKING_ANSWER_ROOM | in F | tokens reserved for the answer after thinking (FE_ONLY) | F |
| HALOGEN_TOP_K | in F | sampling default (FE_ONLY) | F |
| HALOGEN_TOP_P | in F | sampling default (FE_ONLY) | F |
| HALOGEN_MAX_TOKENS_CAP | 65536 | `--max-tokens-cap` (939, 953, 1397) | F argv |
| HALOGEN_ENGINE | 127.0.0.1:$ENG_PORT | engine address for `api` mode only (951) | S, F argv |
| HALOGEN_QUEUE_TIMEOUT | 3600 | `--queue-timeout` (955, 1344, 1399) | F argv |
| HALOGEN_NPU_MODELS | unset | comma list of NPU model ids or `/models/<dir>` paths; setting it starts the NPU engine (1292, 1378) | S, N |
| HALOGEN_NPU_PORT | 8740 | NPU engine port (1288) | S, N, F |
| HALOGEN_NPU_VERIFY | 1 | 0 skips checking NPU files against the image record (1068, 1135) | S |
| HALOGEN_NPU_WITH_GPU | 0 | 1 starts NPU beside GPU without the fabric-clock hold (1239, 1265) | S |
| HALOGEN_NPU_DOWNLOAD, _REPO, _DIR | gone | retired; script prints a message (1029-1034) | S |
| HALOGEN_IMAGE_VERSION | unknown | printed in the banner (1366); set by the image (inferred) | S, F |
| HALOGEN_API, HALOGEN_API_LOG | set by script | passed only to `bench-serving.py` (1478) | bench tool |
| HALOGEN_FLASH_PIN_TRUNK | 1 (as printed) | "trunk pinned" in `ppl|niah` banner; script only prints it (1525) | E, inferred |

Note on the "in F" rows: defaults for the 18 sampling, thinking and recorder flags are not in the script. They live in `serve_api.py`, which `--check-defaults` (938) validates.

## 4. Supervision

Process tree in `all` mode: entrypoint (PID 1 in the container, inferred) with children flash_serve, serve_api.py, the watchdog function (a backgrounded subshell, 1404) and optionally `halogen-npu`. There is no restart loop. The container's restart policy does restarts (stated at 838, 860).

Healthcheck (`hc:` lines): protocol is a TCP connect to the port, send `PING\n`, expect the line `PONG`, with `read -t $HALOGEN_ENGINE_PING_S` (hc:8-11). `api` mode does HTTP GET `/health` (hc:15-22). `auto` (default) checks the API if its port is open, else the engine (hc:26-30). The healthcheck only reports; it does not restart anything.

Engine watchdog (`engine_watchdog`, 802-868), probe step 15 s (804):
1. Every 15 s send PING (`engine_pong`, 795-800). A PONG resets `last_ok`, `deferred` and the compaction baseline (816-826). If it had deferred, it logs "not a wedge, nothing was taken down" (819).
2. On a missed PONG, read state: `wd_state` returns `D` if any thread of the engine is in uninterruptible sleep, else the main thread state (732-742, from `/proc/<pid>/task/*/stat`). `wd_compact` reads `compact_stall` from `/proc/vmstat` (743-746).
3. Defer (do not count as wedge) if any thread is in state D (830-834), or if `compact_stall` rose since the last probe (830, 841). Deferral is uncapped for D state. For compaction it is capped by `HALOGEN_ENGINE_WATCHDOG_DEFER_S` (900 s): at the cap the container is taken down (835-839), reason given as a wedge under memory pressure (issue #85).
4. When compaction stops with the engine still silent, a fresh grace clock starts and `limit` (180 s) must elapse again (847-851). Silence is then counted from `grace_from` or `last_ok` (852). At `limit`, the watchdog returns 1 (854-864): "wedge, not a stall".
5. Reason for D-state deferral (834): killing an engine inside the kernel is what leaves the driver holding its GPU memory (issue #79). Watchdog return exits the `wait -n` in `all` and `engine` modes (1412, 925).

Why PING can go silent without a wedge (vendor note, 856-857): PING is answered "between decode rounds, between the layers of a prefill, and while the lookup table is being read". That tells us the engine's main loop polls the PING socket at those three points.

Teardown on any component exit (`all`, 1412-1429):
- Prints "a component exited" (1416). Sends TERM to API and NPU (1418). `kill -9` on the watchdog (1420). `stop_engine` (777-794): TERM, wait up to 30 s, then KILL, wait up to 30 s, and report if still alive (return 137, "needs a host reboot"). `wait` API and NPU. `npu_fclk_release`. `wd_gtt_after_exit` (764-776) polls GTT use for up to 6 s and warns if 2.0 GiB or more is still held (issue #79).
- Final `exit 1` always (1429): even a clean engine exit ends the container nonzero in `all`.
- `engine` mode exits with the engine's stop code if nonzero, else the `wait -n` code (933-934).
- TERM/INT trap: `all` kills the engine and NPU only (1383). The API is then reaped by the same teardown (inferred).

Pre-start guards that stop a restart loop from making things worse: `gtt_note` refuses to start when free GTT is below need (185-188) because a start that cannot place its pool "blocks inside the driver" (187). It also warns when 2 GiB or more of GTT is used and no process holds the GPU (173-178).

Defaults on boot waiting: no timeout, 60 s heartbeat with GTT use (725-728). The engine PING watchdog starts only after the port accepts connections (1402 after 1386), so a stuck load is not watched (inferred; D state during load is not handled).

## 5. Bounded open: what the script cannot tell us

The script is a supervisor and a preflight. It proves almost nothing about the engine's internals. Open items:
- Which HALOGEN_ flags `flash_serve` reads. Only the seven argv flags are certain. The env list is inferred from what the script exports and from `env -u`. Needs `strings`, the `fals_env.txt` xref (already in `analysis/`), or a Ghidra pass over `getenv` callers.
- `flash_serve` flags other than the seven seen: `--resident-gib FILE` prints one line `GIB HAS_TABLE [OWN_PREC [GATHER [KEPT]]]` (regex at 458; fields assigned 460-464) and `--repack IN --out OUT --with-table --head H`. Full argv surface unknown.
- Wire protocol on the engine port. Only `PING\n` to `PONG` is shown. Request framing, slot protocol and streaming format are in `serve_api.py` and the binary, not here.
- Where PING is served relative to the scheduler. Text says decode rounds, prefill layer boundaries and table reads (856-857). Exact code paths unknown.
- GGUF repack algorithm, the 1 s warm cache format and whether it is `.hgn`. Script only measures times (571).
- Memory model constants. The script estimates: KV pool 29500 B/position plus 120586240 B/slot (67), 26624 B/pos/slot without pool, 115 MiB per cache entry (69), weights 68 GiB default, work arena `6.0 + 3.1*max_tok/32768` GiB when gather=1 else `8.8*max_tok/32768` (473-475). These are estimates "for this pre-flight check" (99); real sizes come from the engine.
- What `HALOGEN_FLASH_PIN_TRUNK`, `HALOGEN_WEIGHTS_LOCK`, `HALOGEN_VERBOSE`, the CACHE flags and `HALOGEN_MATMUL_TUNING_FILE` do inside the engine. The script only names them.
- Contents of the tuning plan (`flash-tune.plan`). The engine "may rewrite it at exit" (442), which implies an autotuner that persists matmul choices (inferred).
- The mtp head and the "n-gram lookup table" (47.7 GiB, read through the file cache, never held, 92, 101). The script confirms it exists and is read from disk. Its role (speculation by n-gram lookup, a separate MTP draft head) is inferred from names and the draft-head wording at 561; see `analysis/hgn-format.md` for the on-disk side.
- `/usr/local/bin/halogen-tools` is a native binary called for `inspect|verify` only. Its behavior is not visible here.
- NPU engine (`halogen-npu`): protocol, models and plugin are outside the Flash engine. Only listen args and the `READY` line are visible (1299-1322).
- HALOGEN_TOKENIZER default, image ENV and `HALOGEN_IMAGE_VERSION` are set outside this script (no Dockerfile in `image/`).
