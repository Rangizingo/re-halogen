# API layer helpers (halogen-flash-server 0.17.3)

Files, all under `image/`: `tool_parse.py` (473 lines), `request_records.py` (721), `halogen_tools.py` (251). `bench-serving.py` (199) and `halogen-bench.py` (150) are skimmed only.
All carry the header "source shipped as-is, comments stripped" (line 2). Every claim is file:line from the file named in its section, unless marked "inferred". Nothing was executed; this is a read of the source.

## 1. tool_parse.py

### 1.1 Syntax parsed

One syntax only: the Qwen3-Coder style XML tool call. It is inferred that this matches the Qwen3-Coder chat template, because the default model id is `halogen-qwen3.8-flash-next` (serve_api.py:36). The file never names a model. It does not parse Hermes JSON-in-tag calls, pythonic calls, or OpenAI-native calls.

Grammar, from the regexes at lines 8-15:

```
call   := "<tool_call>" ws "<function=" NAME ">" [ \t]* "\n"? params "</function>"? ws "</tool_call>"
params := ( "<parameter=" KEY ">" [ \t]* "\n"? VALUE "</parameter>" )*
```

- `_BLOCK_RE` (12): `<tool_call>\s*(.*?)\s*</tool_call>`, non-greedy, DOTALL.
- `_FUNC_RE` (13): name is `[^>\n]*`. Only the first `<function=` in a block is used (`search`, 192). A second function in one block is ignored.
- `_PARAM_RE` (14): key is `[^>\n]*`, stripped (174).
- Constants `TC_OPEN`, `TC_CLOSE`, `PARAM_END` at 8-10.

### 1.2 Value boundary rule (`_value_end`, 145-160)

A value ends at the first `</parameter>` that is followed by optional whitespace and then `<parameter=`, `</function>` or `</tool_call>` (`_AFTER_END_RE`, 15; test at 151). A value can therefore contain a literal `</parameter>` (code, XML).
Fallbacks, in order:
1. No clean end, but some `</parameter>` seen before the next `<parameter=`, and (`final` or a later param exists): use the last one seen (154-155).
2. A next `<parameter=` exists: the value ends right before it, no closer needed (156-157).
3. `final` and nothing else: the value runs to end of body (158-159).
4. Otherwise `(None, None)`: the value is still streaming (160).

One trailing `"\n"` is stripped from each value (172-173). The `[ \t]*\n?` after the opening tag is eaten by the regex.
A missing `</function>` is tolerated: the body is cut at `</function>` only if present (199-201).

### 1.3 Type coercion (20-143)

Values are raw strings. `build_arguments` (207-212) coerces each by the tool's declared schema type and returns `json.dumps(d, ensure_ascii=False)`.
- `schema_types` (20-35): accepts `{function:{name,parameters}}` or a bare `{name,parameters}` (25). Non-dict and nameless tools are skipped. Only top-level `properties` are read (29-33).
- `resolve_type` (61-106): walks `type` (string or list), `anyOf`, `oneOf`, depth cap 4 (68). `null` members set a nullable flag. Mixed types: any unknown gives `None`; `string` wins if present; integer+number gives `number`; anything else gives `None` (96-105). Nullable string becomes the sentinel `"string?"` (59, 106).
- `norm_type` (44-57): aliases such as `str`, `bool`, `list`, `dict`, `json`, `none` (37-42), plus prefix matches `int*`, `uint*`, `long*`, `short*`, `unsigned*` to integer and `num*`, `float*`, `double*`, `decimal*` to number (53-56). Unknown returns `None`.
- `coerce` (108-143): `string` returns raw. `"string?"` returns `None` for literal `null` (111-112). Unknown type (`None`): try `json.loads`, but if it decodes to a str keep the raw text (114-119). Typed: literal `null` gives `None` (120); `true`/`false` for boolean (122); int then float for integer/number, non-finite floats stay raw (124-135), an integral float becomes int for `integer` (138); final fallback `json.loads`, else raw string (140-143). A bad value never raises; it degrades to a string.
- A param not in the schema, or a function with no schema, takes the unknown-type path (`fn_types.get(k)` is `None`, 211).

### 1.4 Non-streaming: `split_tool_calls` (214-237)

Returns `(content or None, calls, truncated)`. Each call is `{"id", "type":"function", "function":{"name","arguments"}}`. The id comes from `make_id`, default `"call_" + uuid4().hex[:24]` (17-18).
- Text between blocks is joined and stripped as `content` (222-233).
- A block with no `<function=NAME>`, or an empty name, is skipped and sets `truncated=True` (219-221, 193-196). The `continue` at 221 runs before `gaps`/`pos` update, so the skipped block's raw text stays in the next gap and ends up in `content`.
- An unclosed `<tool_call>` in the tail sets `truncated=True` and stays in `content` (230-231).
- `json_only`: with no calls and no `<tool_call>` anywhere, content is trimmed to start at the first `{`, or `""` if none (234-236). Inferred use: schema-constrained output where the model writes a preface before the JSON (serve_api.py:3659 builds `json_only` from `schema and tools and SCHEMA_PREFACE`).
- No raised exceptions. Malformed input gives fewer calls plus the `truncated` flag.

### 1.5 Streaming: `ToolStream` (239-348)

Stateful. `push(full)` takes the whole decoded text so far and returns `(delta, events)` (285-345).
- Text delta: `_outside` (252-272) removes complete blocks, cuts at an open `<tool_call>`, and holds back a trailing partial prefix of `<tool_call>` (up to 10 chars, 261-267), so the tag is never sent as content. If the new text does not extend what was already sent, it diffs by common prefix (290-293).
- Tool events follow OpenAI `tool_calls` deltas. First sight of block i emits `{index,id,type,function:{name,arguments:""}}` (309-311). Arguments go out as JSON fragments: `{` or `, ` + `"key": value` per finished param (325-328).
- A string param still open streams as it arrives: emit `"key": "` then escaped text (`_esc` = `json.dumps(s)[1:-1]`, 249-250, 330-339). This applies only to a type of exactly `"string"`, not `"string?"` and not unknown (330). Other types wait for completion.
- `_stream_safe` (180-189) holds back a trailing `</parameter>`, or a partial `<`-prefix of `</parameter>` or `<parameter=`, and rstrips, so closer tags do not leak into the value.
- On block close it emits `"}"`, or `"{}"` when there were no params (340-344). A closed call is never re-emitted (312-313).
- Inferred risk: when a string was open and the final value arrives, the code emits `raw[st["v"]:]` (322), which assumes the streamed part is a prefix of the final value. The `rstrip` in `_stream_safe` makes that likely but not proven for values that end in whitespace. Unverified.
- `any_calls()` (347-348): true when at least one call has closed.

### 1.6 `normalize_messages` (350-473)

Not a tool-call parser, but it lives here. Input is OpenAI-style `messages`.
1. A non-dict message raises `ValueError` (353-354). Role `function` becomes `tool`, `developer` becomes `system` (356-359).
2. Assistant `tool_calls[].function.arguments`: a JSON string is parsed to a dict (empty string gives `{}`); invalid JSON or a non-object raises `ValueError` naming the tool (368-380); `None` becomes `{}` (381-382). Inferred reason: the chat template wants dict arguments.
3. Leading system messages merge into one, joined by blank lines (399-405). System content parts must all be `type:"text"`, else `ValueError` (392-394).
4. Later system messages are folded into the previous user turn, or a new user turn, before the next non-tool message (407-431).
5. Consecutive assistant messages with string content merge: content joined by `\n\n`, `reasoning_content` joined, `tool_calls` concatenated (434-451).
6. A run of `tool` messages right after an assistant with `tool_calls` is reordered to the call order, only if every id matches and ids are unique (453-472).

The three `ValueError` sites are the only error handling. Inferred: callers turn them into HTTP 400 (serve_api.py not read).

## 2. request_records.py

### 2.1 Purpose and switches

An opt-in recorder. Off unless `HALOGEN_RECORD_DIR` is set (96-98). If set but start-up fails, the server exits (103-105).
Env knobs (95-109; entrypoint.sh:892-893 passes them through): `HALOGEN_RECORD_DIR`, `HALOGEN_RECORD_GIB` (default 2), `HALOGEN_RECORD_DAYS` (default 90), `HALOGEN_RECORD_TEXT` (default `all`).
Text fields that can be kept: `prompt, reply, reasoning, args, results` (18, 45-59). `none` keeps hashes and metadata only. An unknown name raises `ValueError`.

### 2.2 Lifecycle

- serve_api calls `RC = request_records.from_env(print)` (serve_api.py:2044), stores it on `app.state.records` (2045), and returns `RC.health()` as `request_records` in the health body (3367-3368). `health()` gives `dir, written, dropped, dropped_cap, bytes, text` (162-165).
- `Recorder.begin(...)` makes a `Rec` holding `d` = `t, wire, stream, model, client, sess, src, n_msgs, effort, tool_choice` (139-149). Client family comes from User-Agent regexes: claude-code, codex, opencode, aider, cline, kilo, continue, hermes, anthropic-sdk, openai-sdk, curl, python-http, node-http, else `other` or `none` (24-43).
- `Rec.admitted(cid, wait_ms, sig)` adds `id`, `wait_ms`, `req` (73-76); called at serve_api.py:3654.
- `Rec.finish(...)` runs once (80-82). It adds `finish, n_gen, n_reasoning, eng, reasoning, body, call_ids, tools, json_only` and queues the job (85-91). It swallows exceptions (92-93); called at serve_api.py:3682.
- Wires recorded: `completions`, `chat`, `responses`, `messages` (serve_api.py:3938, 3987, 4062, 4158). `completions` passes `src=None`, so `src[0]` raises inside the try and `_read_src` returns `(None, [])` (453, 482-484). Inferred result: completions records have no `units` or `reply`, and one "could not read a request's messages" note is logged.
- A bounded queue of 2048 (20, 119) feeds one daemon thread `record-writer` (128-129, 175-184). A full queue drops and counts (155-160). A write exception drops and counts (180-182). Drops never block a request.
- NPU path: `submit_npu` (151-153) and `write_npu` (359-379) exist. Grep of `image/*.py` found no caller of `submit_npu`. Inferred: the producer is elsewhere or the path is unused here.

### 2.3 On-disk layout and record format

Root is `HALOGEN_RECORD_DIR`. Dirs are mode 0700, files 0600 (126, 193-194, 443, 497).
```
.salt                                   32 random bytes, created O_EXCL (206-218)
turns/YYYY-MM-DD/HOST-PID-MSEC.jsonl    one JSON record per line, UTC day (64-65, 486-501)
units/XX/<32hex>.json                   content-addressed message bodies (346-357)
```
Rotation: a new file at the day change or when the next line would pass 64 MiB (`ROTATE_BYTES`, 19, 488).

Turn record (`write`, 381-432), compact JSON, `"v":1`:
- `t` (3 decimals), `id`, `wire`, `stream`, `model`
- `sess`: `{src, h}`, h being a 16-hex HMAC of the first non-empty session key (260-266), or `null`
- `client`
- `req`: signature dict from serve_api plus `n_msgs`, `effort`, `tool_choice` (389-393). Only `think` and `effort` are confirmed as keys, via `cmd_stats` (622-623).
- `in`: one entry per tool result the client sent back: `{call_id, is_error}`. `call_id` is kept only if this server issued it (`self.ours`), else `null` (394-396). `is_error` is a bool for Anthropic `tool_result`, `null` for chat and responses (470, 460, 478).
- `out`: `{finish, n_gen, n_reasoning, calls:[{id,name}]}`. Calls are rebuilt by running `tool_parse.split_tool_calls` on the stored `body` with the original ids (384, 398-405).
- `eng` (engine stats; `cache_n` and `prompt_n` confirmed at 610-612), `wait_ms`
- When any text field is enabled and messages were readable: `units` (hash list, one per input message), `reply` (hash), `reasoning` (hash, only if non-empty) (409-429).

A unit is a canonical message `{role, content, tool_calls?, tool_call_id?}` (314-327). Its hash is a 32-hex HMAC-SHA256 of the canonical JSON with the salt, tag `u` (257-258, 347). Identical messages across turns are stored once (348-349, 354-356). Images become `{type:image, h:<16hex hmac>, w, h_px}`; pixel size comes from `data:` URLs through PIL when it can (268-281). Fields not listed in `HALOGEN_RECORD_TEXT` are replaced by `{"omitted": length}` (329-344). The hash is computed before redaction (347 vs 350), so hashes do not change with the redaction setting.

### 2.4 Retention and cap

- Age: `HALOGEN_RECORD_DAYS` (default 90); `<=0` disables (505-506). Pruned at start (203) and at each UTC day change (491-492). Turn files go by mtime, never the open file (570-571); then units no turn references are deleted (544-558, 583).
- Size: cap `HALOGEN_RECORD_GIB` (default 2). A record that would pass it is dropped whole and counted in `dropped_cap` (437-440). The store does not evict to make room; it stops writing. `prune --gib` is the manual evict (663-675).
- Units are written tmp then `os.replace` (441-447). Turn lines are appended and flushed (499-500).
- A wrong-sized `.salt` raises (216-217). `prune --all` removes `turns` and `units` but not the salt (676-680).

### 2.5 CLI (701-718)

`python3 request_records.py stats DIR` prints JSON (597-628): record count, bytes, oldest/newest, counts by wire and client, session count, unattributed count, finish, `think` and `effort` counts, cache hit share `cache_n / (cache_n + prompt_n)`, answered calls, `is_error` rate, unit count.
`prune DIR` takes `--days N`, `--gib G`, `--all [--pinned]`, `--unit HASH`, `--session HASH` (659-699). `--session` rewrites turn files in place and skips files touched in the last 120 s (630-657, though the message at 691 says 2 minutes). `--pinned` names a `pinned/` dir that nothing in this file creates (678).

## 3. halogen_tools.py

### 3.1 What it wraps

Binary: `HALOGEN_TOOLS_BIN`, default `/usr/local/bin/halogen-tools` (26). The script wraps two of the five verbs, `ppl` and `niah` (225-226, 3-8). They are the two that need the tokenizer. `verify`, `inspect` and `convert` are not in this file. `entrypoint.sh:1517-1530` sends `ppl|niah` through this script and execs the binary directly for `inspect|verify`. `convert` has a separate branch just above it (entrypoint.sh, not read in detail).
Entrypoint call: `python3 /halogen/tools/halogen_tools.py MODE FILE --tokenizer "$HALOGEN_TOKENIZER" "$@"` (entrypoint.sh:1528). Before that the entrypoint checks the file exists and sets up the tuning plan and n-gram table (1517-1527). That is outside this file.

### 3.2 Arguments

`parse_known_args` (239): unknown flags pass to the binary unchanged (USAGE 15-16).
`mode {ppl,niah}`, `file`, `--tokenizer` (default `$HALOGEN_TOKENIZER`, else `/tokenizer`), `--corpus`, `--ids`, `--filler-ids`, `--worst N` (0), `--json`, `--depths` (`4096,32768`), `--positions` (`0.05,0.5,0.95`), `--gen` (32), `--out` (226-238).
Rules: ppl needs `--corpus` or `--ids`; niah needs `--corpus` or `--filler-ids` (241-247). For ppl, `--out` is forwarded to the binary (243-244).

### 3.3 ppl flow (70-130)

1. Corpus text is tokenized with `add_special_tokens=False` (65). It must be UTF-8 and give at least 2 ids (61-76). Ids are written as little-endian int32 (`<%di`, 50-56) to a temp `.ids.bin`, deleted afterwards (89-90).
2. Binary call: `halogen-tools ppl FILE --ids PATH --json [--worst N] <passthrough>` (83-85). Stdout is captured; a nonzero exit code is returned as is (91-92).
3. Output is JSON. A single file has a `file` key; a two-file compare has `a` and `b` (101). Fields used: `file, ppl, nll, tokens, chunk, path`, and a `ref` object with `path, topk, kl{mean,median,p90,p99,max}, top1_agreement, dp{mean,rms,abs_p99}, worst[]`. Each worst entry has `pos, kl, ref_p, this_p, ctx_ids, target, ref_top, this_top` (95-124). The paired block is `d["paired"]["paired"]` with `mean_diff, t, ci95, ppl_ratio, better, worse, tied` (125-129).
4. The wrapper adds `ctx_text`, `target_text`, `ref_top_text`, `this_top_text` by decoding with the tokenizer (95-100). `--json` prints the augmented JSON; otherwise a text table.

Errors: `sys.exit(msg)` on a bad tokenizer dir, unreadable or non-UTF-8 corpus, or too-short corpus (45-46, 61-64, 76). `json.loads(r.stdout)` at 93 has no guard, so non-JSON from the binary gives a traceback.

### 3.4 niah flow (135-222)

Builds the haystack in Python, runs the binary on a manifest, scores in Python.
- Three fixed needles `vault7`, `relay`, `ledger`, each `{fact, lead, answer}` (29-42).
- Filler must be at least 256 ids (141). For each needle, depth T and fraction: ids = filler[:p] + fact + filler[p:room] + lead, with `room = T - len(fact) - len(tail)`; skip if `room < 64` (152-155). Filler is tiled when shorter than `room` (157-158). `assert len(ids) == T` (162).
- Output dir is `--out` or `mkdtemp("niah-")` (145). Per case `NAME.ids`; `manifest.tsv` lines `name<TAB>path<TAB>gen`; `cases.json` (164-171). Case name is `KEY_T<T>_p<NN>` (163).
- Binary call: `halogen-tools niah FILE --manifest manifest.tsv --out DIR <passthrough>` (176). The binary writes `NAME.gen` int32 files, read back at 182.
- Score: a hit when the answer appears case-insensitively in the decoded generation (184). `hit_ids` is a literal id-subsequence match, reported beside it (133, 185).
- Report: hits per depth by position and overall (187-221). `--json` emits `{tool, file, depths, positions, gen, by_depth, overall, cases}` (201-204).
- With `--json` the binary's stdout goes to `/dev/null` (177), so its own diagnostics are hidden.

## 4. Connections to serve_api.py (grep of imports and call sites only)

- `tool_parse`: `from tool_parse import TC_CLOSE, TC_OPEN, ToolStream, normalize_messages, split_tool_calls` (serve_api.py:27-28), after a `sys.path.insert` of its own dir (26). Uses: `normalize_messages` at 3964, 4046, 4134; `ToolStream(tools, json_only=...)` at 3660; `split_tool_calls` at 3818, 3852, 3896 and 2689; `TC_OPEN`/`TC_CLOSE` string logic at 2685-2688 and 3559-3563 (inferred: stream holdback and replay checks). `request_records._Store.write` also imports `split_tool_calls` lazily (request_records.py:384).
- `request_records`: `import request_records` (serve_api.py:30). `from_env` at 2044, `begin` through `rec_begin` at 2047-2054, `admitted` at 3654, `finish` at 3682, `health` at 3367-3368.
- `halogen_tools`: not imported by serve_api.py (grep found nothing). It is a standalone CLI reached only through `entrypoint.sh:1528`.
- Bench scripts: not imported. `entrypoint.sh:1475-1480` runs them over HTTP against the local API. Mode `sweep` runs `halogen-bench.py --api URL "$@"`; the other bench mode runs `bench-serving.py ${1:-serial,mtp} ${2:-256} ${3:-low} ${4:-1}`.

## 5. Bench scripts (header skim)

- `bench-serving.py`: positional args `DRAFTERS MAXTOK EFFORT REPS [TEMP] [TOPP]` (24-29). Env `HALOGEN_API`, `HALOGEN_PROMPTS` (default `tools/eval-prompts.json`), `HALOGEN_API_CTR`, `HALOGEN_API_LOG`, `HALOGEN_BENCH_DEPTH`, `HALOGEN_BENCH_PREFIX`, `HALOGEN_BENCH_CPT` (10-22). Compares drafters (`serial`, `mtp`) per prompt case. A depth run needs a user-supplied prefix file; "deliberately no default corpus" (36-38).
- `halogen-bench.py`: "llama-bench-shaped pp/tg sweep" (argparse description, line 42). Flags `--api -p -n -d/--drafters -r --effort --tg-cases --json` (44-58). It POSTs `/v1/chat/completions` with `enable_thinking:false`, `reasoning_effort`, `temperature:0.0` and an optional `drafter` (22-34). `build_prompt` calibrates prompt length from the server's `usage.prompt_tokens` (37-46).
- Names are not what they suggest: `sweep` mode runs `halogen-bench.py`, and `bench-serving.py` is the drafter A/B.

## 6. Bounded open

Not established by this pass:
1. Nothing was run. The grammar is read from code. That the model's chat template emits exactly this form is inferred.
2. The `req` signature keys passed to `Rec.admitted` (serve_api.py:3654) were not read; only `think` and `effort` are confirmed via `cmd_stats`.
3. `eng` contents: only `cache_n` and `prompt_n` are confirmed (request_records.py:610-612).
4. NPU records: `write_npu` has no found producer in `image/*.py`.
5. The `pinned/` store (request_records.py:678) has no writer in these files.
6. `halogen-tools` flags (`--ids`, `--worst`, `--manifest`, `--out`, `--json`, the `ref` and paired compare flags) are known only from how the wrapper calls it and the JSON it expects. The binary was not read here (see `analysis/functions-tools*.txt`).
7. `inspect`, `verify`, `convert` argument forms: only the entrypoint dispatch was seen (entrypoint.sh:1517-1530).
8. The `ToolStream.push` whitespace edge case (1.5) needs a replay test.
9. Untested: on-disk records from a real run, PIL availability in the image, behavior after salt loss.
