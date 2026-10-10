# Engine wire protocol (front end image/serve_api.py to engine, TCP 127.0.0.1:8730)

Source: `image/serve_api.py` (4321 lines), read by grep plus windows. All `file:line` cites are `serve_api.py` unless a file is named. "Inferred" marks anything the front end does not state. Verbs and replies below are only what this client sends and parses; the engine may accept or emit more.

Framing: text lines ending `\n`, fields split on whitespace (`raw.decode().split()`, :836, :868, :1139). Request ids are Python ints. Blobs are base64 on one line.

## 1. Connection and handshake order

Address: `--engine` default `127.0.0.1:8730` (:4204). `entrypoint.sh:8` sets `ENG_PORT=${HALOGEN_PORT:-8730}`; `entrypoint.sh:949-956` passes `--engine ${HALOGEN_ENGINE:-127.0.0.1:$ENG_PORT}`. `--engine none` skips the Flash engine and serves the NPU only (:4248).

Order, from `Engine.connect` (:696-717), run at app startup (`_connect`, :4276):

1. `asyncio.open_connection(host, port)` (:710). One long-lived TCP stream per front end.
2. `INFO\n`, up to 3 attempts (`PROBE_ATTEMPTS = 3`, :440), each waits `PROBE_S` (default 5.0 s, :439) for one line (`_probe`, :855-881). Accepted reply: first field `I` and at least 6 fields (:869). A timeout retries; EOF or an exception stops probing. A bad parse stops probing (:872-875).
3. On probe failure the client serves on fallback info: one slot, no sampling, no vision, no grammar, `probed: False` (`_fallback_info`, :884-888). A later request re-sends `INFO\n` on the same socket (`_reprobe`, :890-909) only if not probed, no reader task, and no active request (:891-893).
4. If INFO says `grammar=1`, send `VOCAB` (section 2) and wait for `V ok` (`_send_vocab`, :773-794). Any other result sets `grammar=False` and records `grammar_error`.
5. `_apply_info` (:719-739): sets slot count from `kv_slots` (:722), builds the semaphore, and if `n_slots > 1` starts the demux reader task (:738-739).

A separate liveness check opens a second short-lived connection, writes `PING\n`, reads one line, expects exactly `PONG`, closes (`probe_live`, :796-828). It is cached 1.0 s (:798). Timeout `HALOGEN_ENGINE_PING_S`, default 30 s (:438). Used by `/health` (:3152).

Reconnect: `_ensure` (:940-945) reconnects when the writer is missing or closing. `_connect_locked` (:700-717) nulls every pending stream queue (puts `None`), cancels the demux task, and reopens. After a timeout or cancel in batch-1 mode the client closes the socket (`close`, :947-954; :1120, :1128, :1258), so the next request reconnects and re-handshakes.

## 2. Client to engine verbs

Verbs this client writes: `PING`, `INFO`, `VOCAB`, `VIMG`, `GEN`, `X`, `CSTAT`. Nothing else is written (grep of every `.write(` in the file: :778, :807, :858, :897, :1010, :1018, :1059, :1071, :1102, :1111, :1164, :1177, :1242). The NPU verbs (DEC IMG EMB SCO CLS PFX LEDGER CANCEL QUIT) do not appear in this client; the NPU path lives in `npu_api`, a module not in `image/` (:4249, :4254).

### PING
`PING\n` on a fresh connection. Reply `PONG` (:807, :813).

### INFO
`INFO\n` (:858, :897). Reply is 3.1.

### VOCAB
`VOCAB <n> <base64(blob)>\n` (:778). Sent only when INFO `grammar=1`.
- `n`: int, vocab size = max token id + 1 (:487, :500).
- blob (`vocab_blob`, :479-501): ASCII `HGV1`, little-endian u32 `n`, then for each id 0..n-1 a record `<BH` (u8 flag, u16 length) followed by bytes. Flag 0 = ordinary token, bytes are the GPT-2 byte-level decoding of the token string (:498-499). Flag 1 = added/special token, bytes are UTF-8 of the token text (:495-497). Missing id: flag 1, length 0 (:492-493). Cached per process (:477, :481).
- A token over 65535 bytes would overflow the u16 (inferred).
Reply: 3.2.

### VIMG (one per image, written before GEN)
`VIMG <req> <first_tok> <H> <W> <base64(rgb)>\n` (:1070-1073).
- `req`: the id the following GEN will use (:1068-1069).
- `first_tok`: index in the prompt token list of the first image pad (`placed.append((len(out), H, W))`, :126-131).
- `H`, `W`: resized pixel size. Raw RGB, `H*W*3` bytes (check at :197). Size is a multiple of `VIS_UNIT = 16*2 = 32` (:40-42), min pixels 256*256 (:43), max `HALOGEN_VISION_MAX_PIXELS` default 2560*1440 (:44, `vis_smart_resize` :147).
- Each image occupies `ntok = (H/32)*(W/32)` pad tokens (:199, :128-130).
- No reply to VIMG is parsed. In batched mode the VIMG writes do not take `wlock` (:1070-1073 come before the locked write at :1009 path; the batched branch is :1098-1100), so lines from concurrent requests could interleave at the line level (observation, not tested).

### GEN
Built at :1074-1097. One line, field order:

```
GEN <req> <max_tokens> <n_eos> [<eos>...] <n_ids> <id>... [<drafter>]
    [SAMPLE <temp> <top_k> <top_p> <min_p> <seed>]
    [LOGPROBS] [TOPLP <n>] [PENALTY <pp> <fp>] [REPEAT <rp>] [BIAS <n> <tok> <bias>...]
    [SNAP <n>] [SNAP2 <n>] [SNAP3 <n>]
    [IMG <k> <first_tok> <H> <W> ...]
    [SCHEMA json | SCHEMA <base64 schema text>] [AFTER <tok>] [ESCAPE <n> <tok>...]
    [THINK <budget> <end_tok> <n_close> <tok>...]
    [GUARD <in_think> <in_call> <think_end> <call_open> <call_end>]
    [SEG <n> <pos>...]
```

Types and sources:
- `req`: int, per-engine counter, first value 1 (`self.req += 1`, :1068).
- `max_tokens`: int. The front end rejects requests that exceed the room left in the context (:3621-3629). The engine may clamp further; see `room_from` in 3.3.
- EOS list: count then ids. Front end sends `{tok.eos_token_id, id("<|im_end|>")}` filtered to ints >= 0 (:2086-2088). Order is set order (inferred).
- Prompt: count then token ids (:1076).
- `<drafter>`: bare integer right after the ids, only if the request names one. `DRAFTERS = {serial:0, mtp:1, dflash2:2}` (:1889). `mtp` needs INFO `mtp`, `dflash2` needs INFO `dflash2` (:1890, :2212-2215). Omitted means the engine default from INFO field `default`.
- `SAMPLE`: only when temperature > 0 (:1880-1887). Formats `%.9g %d %.9g %.9g %d`: temperature, top_k (0 = off), top_p (0 = off), min_p, seed. Seed is `req.seed` or a random 63-bit int (:1884). Greedy is the absence of SAMPLE. Needs INFO `sampling=1` (:2134).
- Penalty string (:1846-1878). `LOGPROBS`, and `TOPLP n` if top_logprobs, are sent even without sampling. The rest are sent only with SAMPLE: `PENALTY <presence> <frequency>` (`%.9g`), only if either is non-zero; `REPEAT <rp>` only if set and not 1.0; `BIAS <n> <id> <value> ...` with ids 0..248319 and at most 20480 entries (:1869, :1873).
- `SNAP`, `SNAP2`, `SNAP3`: int token positions where the engine should snapshot state for the prefix cache; each omitted when 0. From `render_prompt` (:3463-3505): SNAP is the end of the stable prefix (template without generation prompt), SNAP2 is the end of the system turn (first `<|im_start|>user`), SNAP3 is the start of the last user turn or a client-given mark. Constraints: SNAP2 < SNAP, SNAP3 > SNAP2 and < SNAP (:3486-3497, :3503). `HALOGEN_CACHE_SNAP3=0` disables SNAP3 (:47, :3489). Positions shift when image pads expand (:136-138).
- `IMG <k> (<first_tok> <H> <W>) * k`: summary of the VIMG lines just sent (:1083-1085).
- `SCHEMA json` for plain JSON mode, else `SCHEMA <base64(schema text)>` (:1086-1088). Only when INFO `grammar=1` (:2115).
- `AFTER <tok>`: only with SCHEMA, thinking on, chat; value is the think-end token id (:3596, :1089). Inferred: grammar starts after that token.
- `ESCAPE <n> <ids>`: only with SCHEMA and tools; value `[TOOL_CALL_ID]` (:3597, :1090-1091). Inferred: tokens allowed to leave the grammar.
- `THINK <budget> <end_tok> <n_close> <close_ids...>`: thinking budget in tokens, think-end token id, and the token list the engine inserts to force a close (`THINK_CLOSE_IDS`, :3601-3608, :1092-1093). Budget is `max_thinking_tokens` or `max_tokens - answer_room`, whichever is smaller.
- `GUARD a b c d e`: five ints `in_think in_call think_end_id tool_call_id tool_call_end_id` (:2286-2299), `-1` for unknown. Without tools `in_call` is 0 and the last two are -1. Not sent when `HALOGEN_EOS_GUARD=0` (:1340, :2287) or the prompt state is unknown. Inferred purpose: stop end-of-turn tokens inside a think block or open tool call.
- `SEG <n> <pos>...`: token positions of each later `<|im_start|>` boundary (:3472-3478). Sent only when INFO `composable=1` and the list is non-empty (:1095-1096). With images the list is empty (:3587).

Examples built from the code. Greedy, 5-token prompt, two EOS ids, serial drafter, SNAP at 3:

`GEN 7 256 2 151645 151643 5 11 22 33 44 55 0 SNAP 3`

Sampled with logprobs (ids elided, `1` after the id list is the `mtp` drafter):

`GEN 8 512 2 151645 151643 40 ... 1 SAMPLE 0.7 20 0.8 0 123456789 LOGPROBS TOPLP 5 SNAP 30 SNAP2 12 SEG 2 12 30`

The EOS ids 151645 and 151643 are illustrative Qwen values, not read from the repo.

### X (cancel)
`X <req>\n` (:1018, :1059, :1111, :1242). See section 4.

### CSTAT
`CSTAT\n` (:1164, :1177). Reply is the `C` line, 3.5.

## 3. Engine to client lines

### 3.1 INFO reply (`I`)
`I <f1> ... <fN>` (`_parse_info`, :911-938). Accepted when `len(p) >= 6` counting the `I` (:869). Every later field is optional; older engines send fewer. Index is the position in the split line, `p[0]` = `I`.

| p[i] | key | type | meaning (from use) |
|---|---|---|---|
| 1 | mtp | 0/1 | MTP drafter available (:914, :2213) |
| 2 | draft_head | 0/1 | shortlist draft head present (:914, :3205) |
| 3 | ctx | int | real context length; enforced (:915, :2307, :3621) |
| 4 | spec_rows | int | stored (:915); no use seen in front end |
| 5 | default | int | default drafter id (0 serial, 1 mtp, 2 dflash2) (:916, :3198) |
| 6 | drafter_weights | 0/1 | drafter weights loaded (:917, :3207) |
| 7 | dflash2 | 0/1 | dflash2 drafter available (:918) |
| 8 | cache_mb | int | prefix cache cap in MB, 0 = off (:919, :3209) |
| 9 | cache_align | int | cache alignment in tokens (:920, :3213) |
| 10 | kv_slots | int | concurrent request slots, default 1 (:921, :722) |
| 11 | slot_ctx | int | context per slot (:922, :3180) |
| 12 | cache_mode | int | 0 no cache, 1 default, 2+ other (:923, :2360, :3400) |
| 13 | sampling | 0/1 | sampling supported (:924) |
| 14 | rope_factor | float | YaRN factor, 1.0 = native (:925, :3164) |
| 15 | kv_pool | int | KV pool positions (:926, :3181) |
| 16 | vision | 0/1 | vision tower loaded (:927, :3519) |
| 17 | version | str | engine version (:928, :3512) |
| 18 | pld_n | int | PLD n-gram size (:929, :3201) |
| 19 | pld_k | int | PLD chain length (:930, :3202) |
| 20 | checkpoint_format | str | default `hgn` (:931, :3172) |
| 21 | grammar | 0/1 | constrained decoding available (:932, :720) |
| 22 | composable | 0/1 | SEG supported (:933, :1096) |
| 23 | cc_floor | int | composable unit floor, tokens (:934, :3223) |
| 24 | cc_seam | int | seam tokens (:935, :3224) |
| 25 | cc_bytes | int | composable store bytes (:936, :3225) |
| 26 | cc_fingerprint | str | default `-` (:937, :3226) |
| 27 | indexer_budget | int | default 2048 (:938, :3168) |

Defaults when absent: `kv_slots` 1, `cache_mode` 1, `rope_factor` 1.0, `checkpoint_format` hgn, `indexer_budget` 2048 (:921-938). Fallback on probe failure: :884-888. Key names are the front end's, not the engine's.

Quirk: `pld_n` and `pld_k` both require `len(p) >= 20` (:929-930), so an engine that sent only field 18 would read both as 0.

### 3.2 VOCAB reply (`V`)
Reads up to `PROBE_ATTEMPTS + 1 = 4` lines, 60 s each (:780-781). Lines starting `I` are skipped (a late INFO reply). Success is `V ok` (:785). Failure is `V <word> <message...>`; the message is from field index 2 (`split(None, 2)`, :782, :788). Empty reply: "no reply to VOCAB" (:789). On failure the client turns structured output off and logs it (:793).

### 3.3 Generation stream: `T` and `D`
Token: `T <req> <token_id> [<logprob> [<n> (<id> <logprob>) * n]]` (:1040-1044, `t_line_logprob` :503-511). parts[1] req, parts[2] token id, parts[3] logprob of the chosen token (float), parts[4] count of top alternatives, then pairs from parts[5]. Pairs with id < 0 are dropped (:510). Three fields means no logprob.

Done: `D <req> <reason> <n_prompt> <n_gen> <prefill_ms> <decode_ms> [...]` (`_done_dict`, :956-996). Positions in the split line:

| parts[i] | key | type |
|---|---|---|
| 2 | reason | str; `error` is special (:1046) |
| 3 | n_prompt | int |
| 4 | n_gen | int |
| 5 | prefill_ms | float |
| 6 | decode_ms | float |
| 7,8,9 | drafter, rounds, commit | int (len >= 10) |
| 10 | n_cached | int, prompt tokens served from cache |
| 11 | shared | int |
| 12,13 | pld_rounds, pld_accepted | int (len >= 14) |
| 14 | pld_drafted | int (len >= 15) |
| 15 | n_composed | int (len >= 16) |
| 16,17,18,19 | room_from, max_tokens, pool_used, pool_positions | int (len >= 20) |
| 20,21 | think_forced, think_seen | int (len >= 22) |
| 22,23 | kept_think, kept_call | int (len >= 24) |
| 24 | disk_rows | int |
| 25 | disk_ms | float (len >= 26) |
| 26 | head_drafted | int (len >= 27) |
| 27,28,29 | dv_tokens, dv_misses, dv_switches | int (len >= 30) |

Meaning, from how the front end uses them:
- Head rounds = `rounds - pld_rounds`; head accepted = `commit - rounds - shared - pld_accepted` (:2596-2597). Draft count = `head_drafted + pld_drafted`, or `max(0, head_rounds) + pld_drafted` when `head_drafted` is absent (:2598-2600).
- `pld_rounds`, `pld_accepted`, `pld_drafted`: prompt-lookup rounds, tokens accepted, tokens proposed. The log prints `pld_accepted / pld_rounds` (:2414-2416).
- `room_from >= 0` means the engine clamped `max_tokens` from `room_from` to `max_tokens` (:2590-2592). Negative means no clamp (inferred from the `>= 0` test).
- `pool_used`, `pool_positions`: KV pool usage; stored only when `pool_positions` is non-zero (:998-1001).
- `disk_rows`, `disk_ms`: rows restored from the disk cache and the time (:2586-2587).
- `think_forced`, `think_seen`, `kept_think`, `kept_call`: think-budget and end-of-turn guard counters (inferred from names and the GUARD/THINK fields; semantics not shown).
- `dv_*`: not explained anywhere in the file. Inferred: dflash2 verify stats. Unverified.
- `prefix_n` is read from `d` at :2587-2588 but `_done_dict` never sets it, so it is not from the wire here.

Error: `D <req> error ...`. HTTP 400 via `engine_refusal` (:1046-1049, :1151-1152). `engine_refusal` (:643-656) takes `parts[7:]`, skips leading numeric tokens, and joins the rest as the reason. So an error line is expected to carry numeric fields after the reason (observed on 0.17.3: four, `error 0 0 0 0 asked for drafter 2 (unknown)`), as a normal done line does, then optional text (inferred from the `[7:]` slice). No text gives a generic message. The engine's real error strings are not in this file.

### 3.4 Failure signals
- EOF on readline: HTTP 502 "engine closed the connection" (:1134-1138, :1036-1039).
- No line within `FIRST_TOKEN_S = 1800` s before the first token, or `NEXT_TOKEN_S = 300` s after: HTTP 504 (:442-443, :1022, :1116). Batch-1 closes the socket; batched mode does not (:1032 vs :1128).
- Unknown line forms are dropped silently in demux (:840-841). In batch-1 mode, non-matching lines are skipped (:1142-1156).

### 3.5 CSTAT reply (`C`)
`C <entries> <bytes> <last_entry_bytes> <max_entries> <hits> <misses> <stores> <evicted> <prompt_tokens_saved> ...` (`cache_stats`, :1158-1232). Reply needs at least 10 fields including `C` (:1181). Later groups are gated by line length:

| p[i] | keys | gate (len >=) |
|---|---|---|
| 1-9 | entries, bytes, last_entry_bytes, max_entries, hits, misses, stores, evicted, prompt_tokens_saved | 10 |
| 10-13 | store_ms_total, restore_ms_total, last_store_ms, last_restore_ms (floats) | 14 |
| 14 | refused | 15 |
| 15,16 | rows_copied, rows_copied_ms | 17 |
| 17 | superseded | 18 |
| 18-23 | composable_context: chunks, bytes, stored, composed, evicted, composed_tokens | 24 |
| 24-36 | disk: on(0/1), records, lineages, bytes, budget_bytes, hits, misses, persisted, branches, evicted, skipped, restore_ms_total, restored_bytes | 37 |
| 37,38 | tapped, full_hits | 39 |
| 39-42 | pool: waiting_for_room, waiting_s, relocated, cold_resorts | 43 |
| 43-48 | pool: used (p[43]), positions (p[44]), busy_regions, held_regions, room_clamped, moved | 49 |
| 49 | pool.packed | 50 |
| 50 | dropped | 51 |
| 51 | pool.taken_over | 52 |

The `pool` dict is created at len >= 43 and extended at len >= 49 (:1215-1222). Reply read timeout 10 s (:1167, :1179).

## 4. Demux and cancel rules

Two modes, chosen by INFO `kv_slots` (:722, :738, :1098).

Batch-1 (`kv_slots == 1`): no reader task. `generate` reads the socket directly (:1102-1156). Only lines for the active `req` count; `T` and `D` with another id are ignored. `cache_stats` takes the `slots` semaphore so CSTAT does not interleave with a stream (:1176-1180).

Batched (`kv_slots > 1`): one `_demux` task reads every line (:830-853).
- `C ...`: pushed to the single `cstat` queue (:837-839). The sender clears stale `C` lines first (:1161-1162).
- `T` or `D` with a numeric id found in `self.streams`: pushed to that request's queue; unknown ids dropped (:840-848).
- Anything else (`I`, `V`, `PONG`, garbage) is dropped (:840-841).
- On EOF or error, every queue gets `None`, which the stream turns into 502 (:849-853, :1036-1039).
- Writes take `wlock` (:1009, :1017, :1163). VIMG does not (:1070-1073).
- Admission: `slots` semaphore of `kv_slots`; `HALOGEN_ADMISSION_RESERVE` holds back slots for requests with the priority header (:666, :727-737, :741-763).

Cancel (`X <req>`):
- Sent when `ctl["stop"]` is set (client hang up or stop sequence matched). The front end writes `X <req>` once and starts an `ABORT_DRAIN_S = 10` s drain (:445, :1014-1021, :1108-1115).
- During the drain `T` lines are discarded, `D` ends the request, and `D ... error` is swallowed (:1042-1043, :1046-1048, :1149-1150).
- If the stream ends without `D` and no cancel was sent, `finally` writes `X <req>` with no drain (:1055-1061, batched).
- Batch-1 `abort()` (:1234-1258): if a request is active, write `X <req>`, read lines until `D <req>` or 10 s, else close the socket. Called on client disconnect and in final cleanup (:3754, :3802). In batched mode `abort()` returns at once (:1235-1236); the stream already sent `X`.
- `X` has no reply of its own. The engine is expected to end the request with a normal `D` (inferred; the client only waits for it).

## 5. Flags that change what is sent or parsed

Env vars read in `serve_api.py`:

| Variable | Default | Effect | Cite |
|---|---|---|---|
| `HALOGEN_ENGINE_PROBE_S` | 5.0 | per-attempt INFO wait | :439 |
| `HALOGEN_ENGINE_PING_S` | 30.0 | PING wait | :438 |
| `HALOGEN_ADMISSION_RESERVE` | 0 | reserve slots, no wire change | :666 |
| `HALOGEN_CACHE_SNAP3` | on | `0` drops `SNAP3` | :47, :3489 |
| `HALOGEN_EOS_GUARD` | on | `0` drops `GUARD` and the held-EOS logic | :1340, :2287 |
| `HALOGEN_SCHEMA_ESCAPE` | unset | `argmax/off/0/none/plain` turns off `SCHEMA_PREFACE`; wire effect not traced | :32 |
| `HALOGEN_TEMPERATURE` | unset | default temp; above 0 adds `SAMPLE` to every GEN | :1318, :1880 |
| `HALOGEN_TOP_K`, `HALOGEN_TOP_P`, `HALOGEN_MIN_P` | unset | fill `SAMPLE` fields | :1319-1321, :1885-1887 |
| `HALOGEN_PRESENCE_PENALTY`, `HALOGEN_FREQUENCY_PENALTY` | unset | `PENALTY` | :1322-1325, :1854-1857 |
| `HALOGEN_REPETITION_PENALTY` | unset | `REPEAT` | :1326, :1858 |
| `HALOGEN_MAX_TOKENS_DEFAULT` | unset | default `max_tokens` field | :1317 |
| `HALOGEN_ENABLE_THINKING`, `HALOGEN_REASONING_EFFORT` | unset | change prompt and the `THINK`, `AFTER`, `GUARD` fields | :1328-1329, :3596-3611 |
| `HALOGEN_MAX_THINKING_TOKENS`, `HALOGEN_THINKING_ANSWER_ROOM` | unset | `THINK` budget | :1330-1331, :3600-3608 |
| `HALOGEN_THINKING_BUDGET_MESSAGE` | Qwen sentence | tokenized message; likely the source of the close ids (inferred, :2222-2245) | :2222 |
| `HALOGEN_VISION_MAX_PIXELS` | 2560*1440 | image resize cap, so `VIMG` size | :44 |
| `HALOGEN_IMAGE_MODES`, `HALOGEN_IMAGE_VERSION`, `HALOGEN_MODEL_ID` | unset/unknown | status only | :3508, :45, :34 |
| `HALOGEN_KEEPALIVE_TIMEOUT`, `HALOGEN_SSE_KEEPALIVE_S` | 300 / 10 | HTTP side only | :447, :449 |
| `HALOGEN_TEMPLATE_UNCHECKED` | off | skips the template check | :1342 |

`HALOGEN_CTX`, `HALOGEN_ROPE_YARN`, `HALOGEN_VISION_TOWER` appear only in messages (:3627-3628, :3521); they configure the engine and are not read here. Entrypoint: `HALOGEN_PORT` (`entrypoint.sh:8`), `HALOGEN_ENGINE` (`entrypoint.sh:951`), `HALOGEN_MAX_TOKENS_CAP`, `HALOGEN_QUEUE_TIMEOUT` (`entrypoint.sh:954-956`).

Request fields that change the GEN line: `drafter`, `temperature`, `seed`, `logprobs`, `top_logprobs`, `logit_bias`, `response_format`, tools, images. INFO capabilities gate them: sampling (:2134), grammar (:2115), vision (:3519), composable (:1096), `mtp`/`dflash2` (:2213), `cache_mode != 0` for prefix matching (:2360).

## 6. Bounded open: what the front end never reveals

- Full `reason` value set in the `D` line. Only `error` is tested (:1046, :1149).
- Meaning of `spec_rows`, `shared`, `dv_tokens`, `dv_misses`, `dv_switches`, `kept_think`, `kept_call`, `think_forced`, `think_seen`. The client carries them as numbers.
- Whether the engine sends any line besides `I`, `V`, `T`, `D`, `C`, `PONG`. Unknown lines are dropped silently (:840-841).
- Error line shapes for malformed verbs (bad GEN, unknown verb, oversized VOCAB). Only `V <err>` and `D ... error` are parsed.
- What `SNAP`, `SNAP2`, `SNAP3` do inside the engine (cache keys, eviction, legal positions) and what happens when they are inconsistent.
- Sampler order (top_k, top_p, min_p, temperature), RNG algorithm for `seed`, whether `top_k 0` means off, penalty and bias math.
- Grammar engine: JSON schema dialect accepted after `SCHEMA <b64>`, how `AFTER` and `ESCAPE` interact, what `json` selects.
- Speculation: how drafter ids map to code paths, PLD policy (`pld_n`, `pld_k` values), accept rules, MTP head use.
- KV pool layout and the clamp rule behind `room_from`, `max_tokens`, `pool_*`.
- Vision tower: patch merge, normalization, how `first_tok` and `IMG` align with the pad run, engine-side `H`/`W` limits.
- Cancel timing: when after `X` the engine emits `D`, whether partial KV is kept, behavior on a mid-prefill `X`.
- Auth and versioning: none exist in this client. The INFO version string is informational (:3512-3516).
- Multi-connection behavior: the client uses one data socket plus short PING sockets. Whether a second data socket is safe is not shown.
- The engine's read-side line limit (the VOCAB line carries the whole vocab as base64).
