# Round 5: disk cache record format, cache clock, end-of-request snapshot, SSE framing (flash_serve 0.17.3, serve_api.py)

## Scope

Four leftovers from `request-trace.md` (Gaps 5, 7, 8; steps S27, S29) and `rebuild-guide.md` section 11 "Other gaps":

1. Disk prompt-cache file naming and the record fields written under `HALOGEN_CACHE_DIR`.
2. The unit of the cache clock (eviction and recency).
3. S29: whether and what the engine stores at the end of a request.
4. S27: SSE framing exactly as `image/serve_api.py` emits it, per route.

Static only, nothing was run. Cite forms: `D:<addr>.c:<line>` = `work/decomp/flash_serve/<addr>.c` (addr is the Ghidra address, file VMA + 0x100000). `VMA 0x17xxxxx` = `objdump -d binaries/flash_serve` address (Ghidra - 0x100000). `serve_api.py:<line>` = `image/serve_api.py`. "Inferred" is marked inline. Bytes of data literals were read from `binaries/flash_serve` at (Ghidra address - 0x100000).

## Verdict: item 1, disk record naming and fields: CLOSED

File names and the manifest layout are read to the byte from both the writer and the reader. Two open points (what buffer each `r<K>` is, and the lineage continue rule) are under Residuals and do not change names or layout.

### 1a. Directory tree

```
HALOGEN_CACHE_DIR/                      mkdir 0700 (D:01854e90.c:126, mode 0x1c0)
  <cfg>/                                16 lowercase hex, FNV-1a 64 of the config text (D:01854e90.c:166 snprintf "%016llx")
    fingerprint                         the config text itself (D:01854e90.c:278 "/fingerprint", :312 fopen "wb", :317 fwrite, :334 rename)
    <lineage>/                          16 lowercase hex, 8 random bytes (D:0186dbc0.c:69), mkdir 0700 (D:0185d3a0.c:131)
      e<N>.rec                          manifest, N = token count of the record
      e<N>.rec.tmp                      only while committing
      e<N>.blob                         saved state bytes
      r<K>                              K = 0,1,2,...: append-grown row files
```

Name builders. All use the join `FUN_0185a9d0(out, root, name)` = root + `/` + name (the `/` is `DAT_0011e49c`, bytes `2f 00`, `D:01858fb0.c:255`):

- Row file: `FUN_0185b560` = join(root, lineage) + `DAT_0013b622` + decimal(K). `DAT_0013b622` is the 2 bytes `/r` (`D:0185b560.c:34`). Decimal digits come from the two-digit table `DAT_001464b0` (`"0001...99"`). Result `<cfg>/<lineage>/r<K>`.
- Manifest: join(root, lineage) + `DAT_00134271` (2 bytes `/e`, `D:0185ed70.c:59`) + signed decimal of `*(uint*)(entry+0x4c)` (`D:0185ed70.c:76`, sign slot `*local_b8 = 0x2d` at :110) + `DAT_0013858b` (4 bytes `.rec`, `D:0185ed70.c:184`). Temporary name = final + `DAT_00119dff` (4 bytes `.tmp`, `D:0185ed70.c:224`).
- Blob: same `/e<N>` stem (`D:0185e6a0.c:60`, :75 reads `+0x4c`) + the literal `".blob"` (`D:0185e6a0.c:182`).
- Reader builds the same names: `/e` at `D:0185ab40.c:165`, number from the manifest's third word, `".blob"` at :278.
- Boot scan treats a file as a manifest when its name starts with `e` and its last 4 bytes are `0x6365722e` (little endian `.rec`) and length is at least 6 (`D:01858fb0.c:233`). Top-level entries named `fingerprint` and dot-names are skipped (`D:01858fb0.c:103-104`).

N is the token count: the manifest repeats it and requires the token array length to equal it (`D:0185ab40.c:79-82`). So one lineage directory holds one `e<N>` per distinct stored prefix length N (inferred: a second store of equal N in one lineage would overwrite).

### 1b. Manifest layout `e<N>.rec` (x86-64 little endian, plain `fwrite`, no padding)

Writer `cache_commit_entry` `0185ed70`, fwrite chain `D:0185ed70.c:234-257`. Reader `FUN_0185ab40`, freads `D:0185ab40.c:57-72`:

| offset | size | field | writer | reader check |
|---|---|---|---|---|
| 0 | u32 | magic `0x43443233` (bytes `33 32 44 43`, ASCII "32DC") | :234 | `local_17c != 0x43443233` rejects, :79 |
| 4 | u32 | version = 1 | :235 `local_84 = 1` | `local_180 != 1` rejects, :80 |
| 8 | u32 | N, token count (`entry+0x4c`) | :236 | must be >= 1, :81 |
| 12 | u32 | length of the u32 token array | :238 (`(+0xa0 - +0x98) >> 2`) | must equal N, :81 |
| 16 | u32 | length of the u32 side array | :239 (`(+0xb8 - +0xb0) >> 2`) | 0, or equal to N, :86-88 |
| 20 | u32 | F, number of `r<K>` files | :240 (`(+0x70 - +0x68) >> 3`) | 1..0x3ff, :91 |
| 24 | u64 | blob length in bytes (`entry+0xd0 - entry+0xc8`) | :241 `local_60` | `blob st_size >= value`, :278-309 |
| 32 | u64[F] | required size of each `r<K>` file | :250 (`entry+0x68`) | each `r<K>` must exist with `st_size >=` value, :135-155 |
| next | u32[N] | token ids | :252 (`entry+0x98`) | |
| next | u32[side] | per-token side ids (role: image/modality ids, same as the RAM entry's second vector; inferred) | :253-254 | |

Commit sequence (`D:0185ed70.c:225-270`): `fopen(tmp,"wb")`, the fwrites above, `fflush`, `fdatasync(fileno)`, `fclose`, `rename(tmp, final)`, then `open(dir,0x90000)` (O_RDONLY|O_DIRECTORY|O_CLOEXEC) and `fsync(dir)`. Failure text `"prompt cache: committing %s failed: %s\n"` then `unlink(tmp)`. The manifest is written LAST: `D:0185d3a0.c:231-233` runs `FUN_0185e360` (rows), then `FUN_0185e6a0` (blob), then `FUN_0185ed70` (manifest), each must return true. A crash leaves orphan `r<K>` bytes or a blob, never a manifest pointing at missing data.

### 1c. `e<N>.blob`

Writer `FUN_0185e6a0`: `open(path, 0x84241, 0x180)` = O_WRONLY|O_CREAT|O_TRUNC|O_DIRECT, mode 0600 (`D:0185e6a0.c:211`). Length written = `(entry+0xd0 - entry+0xc8 + 0xfff) & ~0xfff` (`D:0185e6a0.c:225-226`), i.e. the host byte vector padded to 4 KiB, in pieces of at most 0x4000000 = 64 MiB (:233) copied into the pinned staging buffer `*(ctx+0x228)` (`memcpy` of the vector, `memset` zero for the pad), `pwrite` loop, `fdatasync` (:276). Content: the O(1) snapshot bytes of the entry (copy callback `018fe4f0`/`018fed80`, see `cache-host.md`). That link is inferred from the vector being the only per-record byte vector in the disk entry struct and the 110.8 MiB figure in `cache-host.md`.

### 1d. `r<K>` row files

Writer `cache_write_file` `FUN_0185e360`. For each K where the entry's size `+0x68[K]` exceeds the already persisted size `+0x50[K]` (`D:0185e360.c:34-37`) it opens `r<K>` twice (`D:0185e360.c:39-40`: buffered fd `0x80041` = O_WRONLY|O_CREAT|O_CLOEXEC mode 0600, direct fd `0x84001` = O_WRONLY|O_DIRECT|O_CLOEXEC). For the range `[persisted & ~0xfff, new)` it runs `hipMemcpyAsync(stage, device_buf[K] + off, len, 2)` (kind 2 = DtoH, `D:0185e360.c:66-67`) in pieces up to 64 MiB (:58-62), `pwrite` of the 4 KiB aligned body through the O_DIRECT fd and the sub-4 KiB tail through the buffered fd, then `fdatasync` and `posix_fadvise(..., 4)` (DONTNEED, :97-100). The files only grow: a later record in the same lineage writes just the new range. Restore (`D:01881240.c:190-300`) `pread`s each `r<K>` in 64 MiB pieces and `hipMemcpy`s H2D (kind 1) into the device pointer of slot K; a size-table mismatch is reported with the label `"size table"` (`D:01881240.c:204`).

Branch copy (`cache_branch_copy` `FUN_0185d3a0`): when a record forks from a parent lineage (`entry+0x28` = parent id), every non-empty parent `r<K>` is duplicated with `copy_file_range` into the new lineage's `r<K>` (`D:0185d3a0.c:163-190`, failure text `"prompt cache: branch copy %s -> %s failed: %s\n"`), then the three writers run.

Budget reserve for one store: `blob_len + 0x1000 + sum(new_size[K] - parent_size[K])` (`D:0185d3a0.c:39-67`). When over `HALOGEN_CACHE_DISK_GIB`, `FUN_0185db90` evicts the lineage with the smallest recency number (`D:0185db90.c:86` `*(long*)(lVar7+0x50) < local_68`) and removes its directory tree (`FUN_01858d30` at :202).

### 1e. Lineage ids and which stores reach the disk

- Lineage id = 16 hex digits from `fread(buf,1,8)` on `/dev/urandom`, formatted `"%02x"` x 8 (`D:0186dbc0.c:69`); eight zero bytes if `/dev/urandom` cannot be opened (:21-30, uVar* all 0). Not content-derived.
- Allocation `FUN_0186b7c0` (`D:0186b7c0.c:97-217`): no parent name given, or parent not in the in-RAM lineage map (`disk+0x150` std::map keyed by id string): new random id (:104-109). Parent found and the new prefix length `param_3` is below the parent's recorded length (`*(int*)(param_1+0x40)`, :215): a new random id, with the parent id copied (`M_assign`, :216-217). Otherwise the record continues the parent's lineage (inferred; the tail of the function was not read).
- Which stores are queued: `FUN_0186a470` returns before queueing for a tail checkpoint (`param_7`, `D:0186a470.c:466`), a FULL entry (`param_6`, :469), or when there is no disk tier object (`+0xe0 == 0`, :472). Only plain snapshot leaves (SNAP, SNAP2, SNAP3) and system-prompt anchors reach `FUN_0186b7c0` (`D:0186a470.c:545`). Full-prompt entries stay in RAM.
- Disk tier preconditions stay as in `cache-host.md` (in-place mode, no `HALOGEN_CACHE_FILE`).

### 1f. Config directory and `fingerprint`

Name and hash as in `cache-host.md`. New: the `fingerprint` file holds the same text that is hashed, written to a temp file with `fwrite` and `rename`d onto `<cfg>/fingerprint` (`D:01854e90.c:312-334`), and an existing one is read first (`D:01854e90.c:282`). That a mismatch is detected by text compare is inferred from the read; the compare was not read.

## Verdict: item 2, cache clock unit: CLOSED

There are two clocks. Neither counts requests, tokens or time.

RAM tier (the one `0186b270` victim choice, the `ladder:` trace and the hit touch read): one signed 64-bit counter at `cache+0xa8`, pre-incremented once per event and copied into the entry.

- Store: `D:0186a470.c:461-463`: `lVar6 = *(long *)(param_1 + 0xa8) + 1; *(long *)(param_1 + 0xa8) = lVar6; puVar4[0xb] = lVar6;` (entry+0x58 = new clock). VMA 0x176af5c-0x176af6d: `mov rax,[rbx+0xa8]; inc rax; mov [rbx+0xa8],rax; mov [r15+0x58],rax`.
- Hit (restore touch): `D:0187fb00.c:152-155`: `lVar7 = *(long *)(param_1 + 0xa8) + 1; *(long *)(param_1 + 0xa8) = lVar7; *(long *)(lVar14 + 0x58) = lVar7; *(long *)(lVar14 + 0x90) = lVar7;` (VMA 0x177fe52-0x177fe67). Entry +0x58 is last use; +0x90 gets the same value (role inferred: last hit).
- Reset: zero at construction (VMA 0x1752d45 `mov qword ptr [rdi+0xa8],0x0`); not persisted.
- Consumers: the summary builder copies entry+0x58 into summary+0x08 (VMA 0x176cc74/0x176cc78 `mov rcx,[rbx-0x40]; mov [r14-0x19],rcx`, stride 0x28); eviction tie-break, smaller wins (`D:0186b270.c:66-68`); plain-LRU mode (`D:0186b270.c:77-106`); `0186f7f0` compares `entry+0x58` of candidate and sibling (`D:0186f7f0.c:58`, :97); `0187fb00` uses it to flag an older same-region leaf at entry+0x98 (`D:0187fb00.c:127`, :136, :150).
- Unit: "cache events". One event is one snapshot store or one successful restore; it advances by exactly 1 per event, so it measures LRU order only. Idle time, request length and generated tokens do not move it.

Correction to `cache-key-kvpool.md` 1e/1f ("Unproven: the clock unit and who stamps it (`param_5 = 0xffffffff` ... suggests stamp now)"): the 5th argument of `FUN_0186a470` is the snapshot kind, not the clock. Callers pass 0, 1, 2 for SNAP, SNAP2, SNAP3 (`D:018680b0.c:297`, :304, :314) and `0xffffffff` for non-SNAP stores (`D:018680b0.c:336`, `D:01846b20.c:266`); `FUN_0186f7f0` passes its own `param_5` through (`D:0186f7f0.c:132`). Inside, it is only forwarded to the snapshot copy (`local_1b4`, `D:0186a470.c:331`, :335). The stamp is taken at :461-463.

Disk tier recency (separate): a per-lineage number at lineage-map entry `+0x50` and a counter at `disk+0x1b0`.

- Commit: `D:0185d3a0.c:292-294`: `lVar12 = *(long *)(param_1 + 0x1b0) + 1; *(long *)(param_1 + 0x1b0) = lVar12; *(long *)(lVar10 + 0x50) = lVar12;` (+1 per committed record).
- Boot seed: a lineage's number is the largest `st_mtim.tv_sec` among the files stat'd in its directory (`D:01858fb0.c:476-481`), and the counter starts at the maximum over lineages (`D:01858fb0.c:591-592`). The stat target is the manifest path (inferred from the preceding code). So the disk clock is "epoch seconds as of the last boot, then +1 per record".
- Eviction takes the smallest number (`D:0185db90.c:86`).

## Verdict: item 3, S29 end-of-request snapshot: CLOSED (negative: no entry covering generated tokens is ever stored)

Claim: the engine stores no cache entry at retirement. Every cache store happens during prefill, at or before the end of the prompt.

1. Complete census of store call sites in the binary. Direct `call` sites of `0186a470` (VMA 0x176a470): VMA 0x17473af (in `01846b20`, `D:01846b20.c:266`, composable chunk placement); 0x17685c9, 0x1768623, 0x176867d, 0x176871f (in `018680b0`: `D:018680b0.c:297`, :304, :314, :336); 0x176f9ce (in `0186f7f0`, `D:0186f7f0.c:132`). Direct calls of `0186f7f0` (VMA 0x176f7f0): VMA 0x17685a6, 0x1768605, 0x176865f (all in `018680b0`: `D:018680b0.c:294`, :307, :317). `018680b0` is called only from `01846b20` (`D:01846b20.c:169`, :374, :399), the prefill driver. Method: `objdump -d` text grepped for `call/jmp 0x176a470`, `0x176f7f0`, `0x17680b0` gives only these; `lea ... # 0x176a470` gives none; an 8-byte little endian scan of the ELF for those three addresses finds no function pointer. Over the decompile directory, `FUN_0186a470`/`FUN_0186f7f0` appear only in `01846b20.c`, `018680b0.c`, `0186f7f0.c`. `0182ea80` (decode and retire loop) has no call to either.
2. What prefill stores. `D:018680b0.c:30-40`: `iVar21 = (int)param_2[2]; iVar7 = (int)param_2[4]; if (iVar21 <= iVar7)` (prompt length <= done) takes the "finished" exit with no store and no forward. The three SNAP stores take positions `local_48`, `local_48._4_4_`, `local_40` (`D:018680b0.c:292-318`), inside the prompt (strings "SNAP2 outside 0..n_prompt", `cache-host.md`). The end store is `D:018680b0.c:336`: `FUN_0186a470(param_1,lVar14,iVar21,lVar8,0xffffffff,uVar17,0,bVar4)`; on the branch at :326-335 it uses `iVar21 = (int)param_2[2]` (n_prompt) and `uVar17 = 1` (full flag). The plan's `+0x10` is the prompt length: `D:01860720.c:49` `*(int*)(param_1+2) = param_4`, with `FUN_0187fb00(param_2,param_3,param_4)` at :51. So a store's length is at most n_prompt, never n_prompt + n_gen.
3. What a stored entry holds. Token ids and side vector copied from the prompt (`FUN_0186da80(... local_1a0 + n*4)`, `D:0186a470.c:456`), length `+0x48 = n` (:458), the recurrent and conv state snapshot through the copy callback (`FUN_018fed80` / `FUN_018feef0`, `D:0186a470.c:331-335`; K/V slabs only when in-place is off, `cache-host.md`), flags `+0x88` full = `param_6`, `+0x89` tail = `param_7`, `+0x8a` anchor = `param_8 & ~(param_6|param_7)` (`D:0186a470.c:374-377`), and for FULL entries the saved last-position sampler inputs (`vector<float>` and sub-vectors at entry `+0xa0..+0x148`, copied back into the plan by `D:01860720.c:60-76`, so a full-prompt hit needs no forward call; "logits" is inferred).
4. Retirement. `trace_retire_request` `01845e50` logs `"req %ld retired (%s) after %d tokens"` (`D:01845e50.c:49`) and calls `FUN_01846470(engine, slot)` (`D:01845e50.c:160`; other callers `D:0182ea80.c:1240`, :2233, :3530). `FUN_01846470` (`D:01846470.c`) is the region release: it finds the pool region owned by `slot`; if a cache entry (`cache+0x70` vector, stride 0x160, `*(int*)(entry+0x50) == region base`, :47) or a chunk record (`+0x98 == base`, :52) still uses it, it sets the owner to -2 (HELD, `piVar5[2] = -2`, :43); otherwise it erases the region record (memmove, :53-80). It then sweeps held regions with no remaining entry (:88-110). There is no cache insert.
5. `slot_state_release` `018466e0` is a destructor of the per-request state: `_ZdlPvm` on host vectors, one hash table (`+0x240`), a `0xd60`-byte object (`+0x180`) and a `0x2b0`-byte object (`+0x10`, via `FUN_01870630`, also only frees). It makes no HIP or cache call (`D:018466e0.c:1-104`). This replaces the S28 inference "frees the slot and region": it frees host memory; the pool region is released by `01846470`.
6. Consequence (inferred from 1-4): after a turn the rows of generated tokens remain in the pool region while an entry (the prompt's full entry) keeps it HELD, but no entry maps them. A follow-up request hits the longest entry that is a token prefix of its prompt (`cache-key-kvpool.md` 1b), which is the previous prompt or a SNAP position, and prefills the previous reply plus the new turn again. The banner "a follow-up turn prefills only its new tokens" (`analysis/evidence/engine-0.17.3-run.log:55`) reads consistently if "new" means "past the stored prompt". Not run.

Front-end note: `serve_api.py` keeps its own memory of prompt plus reply for diagnostics only. `RecentPrefixes.add(ids, out)` appends `ids + out` to a 16-slot deque (`serve_api.py:2032-2035`, called at :2396 `recent.add(ids, out)`) and is read to print "closest earlier: N tokens shared ... then its reply differs" (:2391-2395). It never reaches the engine.

Test that turns the static negative into an observation: turn 1 (prompt P1, reply R1), then turn 2 with prompt P1+R1+U2; the turn 2 D line should show `n_cached == len(P1)` (or a SNAP position), not `len(P1)+len(R1)`. Unverified.

## Verdict: item 4, S27 SSE framing: CLOSED for all four routes

All SSE framing is in `serve()` and four generators inside `build_app` (`serve_api.py:2705-3147`, :3589-3769).

### Shared transport

- `serve(stream=True)` returns `StreamingResponse(body(), media_type="text/event-stream")` (`serve_api.py:3769`); no other headers are set. Starlette appends `; charset=utf-8` to a `text/*` media type (library behaviour, inferred, not in this repo). `StreamingResponse` import: :22.
- `body()` picks the generator by `wire` (`serve_api.py:3718-3727`): `sse_responses` for `"responses"`, `sse_messages` for `"messages"`, else `sse` (completions and chat). It wraps it in `keepalive(stream_iter, SSE_KEEPALIVE_S)` (:3728).
- Every frame is a text string ending `\n\n`. No `id:` or `retry:` field exists in the file.
- Keep-alive: `SSE_KEEPALIVE_S = float(os.environ.get("HALOGEN_SSE_KEEPALIVE_S", 10.0))` (`serve_api.py:449`; `<= 0` disables, :2605-2608). `keepalive` (:2604-2631) waits `every` seconds for the next frame and on timeout yields `": keepalive\n\n"` (:2617). `body()` rewrites it (:3729-3733):
  ```
  if ev.startswith(":"):
      n_ka += 1
      yield ('event: ping\ndata: {"type": "ping"}\n\n'
             if wire == "messages" else ev)
      continue
  ```
  So chat, completions and responses carry the SSE comment line `: keepalive` plus a blank line; `/v1/messages` carries an `event: ping` frame. Keep-alives fire only when 10 s pass with no frame (long prefill), and are excluded from the frame count (:3734-3738).
- Errors before the stream exists are JSON, not SSE: validation `400`s in `serve` (:3613-3633), the logprobs refusal (:2152-2157) and `await engine.admit(...)` (:3638, before `StreamingResponse` is created) reach the exception handlers (:2065-2085). Shape: `oai_error` = `{"error": {"message","type","param","code"}}` (:223-226); under `/v1/messages` `anth_error` = `{"type":"error","error":{"type","message"}}` (:239-241), selected by `request.url.path.startswith("/v1/messages")` (:2061).
- Errors after the response started (HTTP 200 already sent): `run()` is created inside `body()` (:3708), so an engine refusal (`D ... error` becomes `HTTPException(400, engine_refusal(parts))`, :1046-1049 and :1151-1152) or any other HTTPException is caught by `guarded()` and turned into a terminal tuple (:2664-2674): `(None, {"reason": "error", "status": e.status_code, "message": str(e.detail), "code": ("engine_busy" if e.status_code == 503 else None)})`. Each generator renders it as its own error frame (below). A non-HTTP exception inside the stream is not caught by `guarded`; the `finally` in `body()` aborts the engine and the connection ends with no terminator (inferred).
- Client hang-up: the `finally` of `body()` (:3740-3762) prints "client disconnected mid-stream ..." when `finished` is false and runs `engine.abort()` under `ABORT_DRAIN_S + 5` s (`ABORT_DRAIN_S = 10.0`, :445). Nothing more is written.
- `json.dumps` default separators (`", "`, `": "`), ASCII-escaped.

### /v1/chat/completions (generator `sse`, `serve_api.py:2705-2796`, chat=True)

Envelope (:2707-2717): `{"id": cid, "created": created, "model": MODEL_ID, "object": "chat.completion.chunk", "choices": [{"index": 0, "finish_reason": fin, **payload}]}`, plus `"usage": null` in every chunk when `stream_options.include_usage` is true (:2715-2716; read by `wants_usage`, :1930-1932). `cid` = `"chatcmpl-"` + 24 hex (:3650 with prefix from :3985).

Order on the wire:
1. Role chunk on the first item the generator yields, even if that item is terminal (:2730-2733):
   `data: {"id":"chatcmpl-...","created":...,"model":...,"object":"chat.completion.chunk","choices":[{"index":0,"finish_reason":null,"delta":{"role":"assistant","content":""}}]}\n\n`
2. Reasoning deltas (:2771-2774): `data: ` + `frame({'delta': {'reasoning_content': rd}})`. Leading whitespace of the first piece is stripped and trailing whitespace is held back one piece (:2761-2770).
3. Content deltas (:2783-2784): `frame({'delta': {'content': cd}})`. With tools active the text is `td` from `tstream.push(pre + buf)` (:2787-2790) and each tool event is `frame({'delta': {'tool_calls': [e]}})` (:2794-2796); with `parallel_tool_calls=false` events with index > 0 are dropped (:2792-2793). Leading `\n` of the first content piece is stripped (:2777-2781).
4. Terminal item `d` (:2734-2751):
   - `fin = {"stop": "stop", "cancel": "stop", "length": "length"}.get(d["reason"], "stop")`, replaced by `"tool_calls"` when `tstream.any_calls()` and `fin != "length"` (:2735-2739).
   - final chunk `frame({'delta': {}}, fin)` plus `last["timings"] = timings(d)` (:2741-2743): `data: {...,"choices":[{"index":0,"finish_reason":"stop","delta":{}}],"timings":{...}}\n\n`.
   - if `include_usage`: one more chunk with `"choices": []`, `"usage": usage(d, n_prompt)`, `"timings"` (:2744-2749).
   - `data: [DONE]\n\n` (:2750); the generator returns.
5. Error (:2726-2728): `data: {"error": {"message","type","param","code"}}\n\n` then `data: [DONE]\n\n`. No finish chunk.

No `event:` line on this route. `logprobs` with `stream=true` is refused with a JSON 400 (:2152-2157).

### /v1/completions (same `sse`, chat=False)

- `object` = `"text_completion"` (:2709-2710); `cid` = `"cmpl-"` + 24 hex (prefix at :3936).
- No role chunk (`role_due = chat`, :2723).
- Deltas: `data: {...,"choices":[{"index":0,"finish_reason":null,"text":"<delta>"}]}\n\n` (:2753); no think split and no tool streaming (the `if not chat:` branch continues before `split.push`, :2752-2757).
- Terminal: `frame({'text': ''}, fin)` + `timings`, optional usage chunk, `data: [DONE]\n\n` (:2741-2750).
- Error: same as chat (:2726-2728).

### /v1/messages (generator `sse_messages`, `serve_api.py:2821-2929`)

Each frame is `ev(kind, **body)` = `"event: %s\ndata: %s\n\n" % (kind, json.dumps(dict({"type": kind}, **body)))` (:2823-2825).
1. `message_start` first, before any engine output (:2827-2832): `message` = `{id ("msg_" + 24 hex), type "message", role "assistant", model, content [], stop_reason null, stop_sequence null, usage {input_tokens: n_prompt, output_tokens: 0, cache_read_input_tokens: 0, cache_creation_input_tokens: 0}}`. `ping` is not emitted here; pings come only from the keep-alive wrapper.
2. Blocks, indexed from `itertools.count(0)` in opening order (:2833, :2852-2855): `content_block_start` (thinking `{"type":"thinking","thinking":""}`; text `{"type":"text","text":""}`; tool `{"type":"tool_use","id","name","input":{}}`), then `content_block_delta` with `thinking_delta` (only if `show_thinking`, :2891-2893), `text_delta` (:2911-2912) or `input_json_delta` `partial_json` (:2927-2929), then `content_block_stop`. Closing a thinking block first emits a `content_block_delta` with `{"type":"signature_delta","signature": think_signature(rs_text)}` (:2846-2849; `"hgn1."` + urlsafe base64 of zlib-compressed text, :243-245), then `content_block_stop` (:2850).
3. Terminal (:2862-2873): close the open block, then `message_delta` with `delta {stop_reason, stop_sequence}` and `usage` (`input_tokens = n_prompt - n_cached`, `output_tokens = n_gen`, `cache_read_input_tokens = n_cached`, `cache_creation_input_tokens = 0`, :2798-2803), then `message_stop`. `stop_reason` (:2805-2812): `"max_tokens"` on length, `"tool_use"` if a call was streamed, `"stop_sequence"` with the matched string if `d["stop_seq"]`, else `"end_turn"`. No `[DONE]` on this route.
4. Error (:2858-2861): `event: error\ndata: {"type":"error","error":{"type": <ANTH_ERROR_TYPES>, "message": ...}}\n\n`, then the generator returns: no `message_delta`, no `message_stop`.

### /v1/responses (generator `sse_responses`, `serve_api.py:2931-3147`)

`ev(kind, **body)` sets `body["type"] = kind` and `body["sequence_number"] = next(seq)` (from 0) and returns `f"event: {kind}\ndata: {json.dumps(body)}\n\n"` (:2936-2939). Ids: `rid` = `"resp_"` + 24 hex (:3648); message `"msg_" + ...` (:2954); reasoning `"rs_" + ...` (:2990); function call `"fc_" + 24 hex` (:3129).
1. `response.created` then `response.in_progress`, both with `response` = `{id, object "response", created_at, status "in_progress", model, output [], parallel_tool_calls, max_output_tokens, usage null}` (:2941-2950).
2. Reasoning item, when the model thinks: `response.output_item.added` (item `type "reasoning"`, `summary []`, `content []`, `encrypted_content null`), with `want_summary` also `response.reasoning_summary_part.added`; deltas `response.reasoning_summary_text.delta` (summary mode) or `response.reasoning_text.delta` (:2997-3020); close with `...text.done`, `response.reasoning_summary_part.done` (summary mode only) and `response.output_item.done` (:3022-3044).
3. Message item: `response.output_item.added` + `response.content_part.added` (part `output_text`, `annotations []`), many `response.output_text.delta` (`item_id, output_index, content_index 0, delta`), closed by `response.output_text.done`, `response.content_part.done`, `response.output_item.done` (:2962-2988, :3104-3118).
4. Function call items: `response.output_item.added` (`type "function_call"`, `status "in_progress"`, `arguments ""`), `response.function_call_arguments.delta`, and at the end `response.function_call_arguments.done` + `response.output_item.done` (:3134-3147, :3055-3067). A still-open text message is closed first (:3125-3127).
5. Terminal (:3050-3078): close open reasoning, message and function-call items in that order; `status = "incomplete"` if `reason == "length"` else `"completed"`; `final = envelope(status, items sorted by output_index, responses_usage(d, n_prompt))`; for incomplete add `incomplete_details = {"reason": "max_output_tokens"}` and emit `response.incomplete`, else `response.completed`; both carry `response=final` and a `timings` key (:3073-3077). No `[DONE]`. No `response.failed` exists in the file.
6. Error (:3047-3049): `ev("error", **error_body(d))` gives `event: error\ndata: {"message":...,"type":"error","param":null,"code":...,"sequence_number":N}\n\n`. Because `ev` assigns `body["type"] = kind` after the merge (:2938), the error class from `error_body` (`"invalid_request_error"`, ...) is overwritten by `"error"`; `message` and `code` survive.

### Not present (grep over `serve_api.py`)

No `id:`/`retry:` fields, no `[DONE]` on `/v1/messages` or `/v1/responses`, no `response.failed`, no `event:` names on chat or completions, no `system_fingerprint`, no logprobs in a stream.

## Corrections to earlier docs

- `request-trace.md` S27 ("I did not read the generator"): closed (item 4). S29: closed, negative (item 3). S28: "`slot_state_release` ... frees the slot and region is inferred": `018466e0` frees host memory only; the pool region is released by `01846470` (`D:01845e50.c:160`).
- `cache-host.md` "Disk cache": "Record file names are built by `cache_name_format` ... exact pattern is not decoded" is decoded (1a); "manifest holds token ids as u32 arrays (inferred)" is confirmed, and the state bytes are in `e<N>.blob`, rows in `r<K>`.
- `cache-key-kvpool.md` 1f/3 (clock unit and stamper unproven; `param_5` suggests "stamp now"): clock is the `cache+0xa8` event counter stamped inside `0186a470` and `0187fb00`; `param_5` is the SNAP kind.
- `rebuild-guide.md` line 279 may drop "disk cache record naming and manifest fields; the cache clock unit; the end-of-request snapshot (S29); SSE framing (S27)". Not edited here.

## Residuals

- `r<K>`: K indexes the engine's list of device buffers (`entry+0x80[K]` is the `hipMemcpyAsync` source, `D:0185e360.c:67`); which buffer each K is (K/V slab per attention layer, state slabs, scratch) was not mapped. F is capped at 0x3ff by the reader. Needed: the builder of the `+0x68`/`+0x80` vectors (the `puVar4+6` / `puVar4+0xc` arguments at `D:0186a470.c:545-546`).
- Lineage rule: only the new-id branches of `FUN_0186b7c0` were read; the continue-same-lineage branch and the equal-N replace case are inferred. No on-disk sample exists (`analysis/evidence/engine-0.17.3-run.log:56`: "0 records in 0 lineages"). Needed: a two-turn run of the real engine with `HALOGEN_CACHE_DIR` set, then `find`, to confirm `e<N>.rec` / `e<N>.blob` / `r<K>` names and the 32-byte header.
- `e<N>.blob` content is inferred to be the O(1) snapshot vector (size and uniqueness argument, 1c); the producer of the `+0xc8` vector was not followed.
- Disk clock boot seed: that the stat in `D:01858fb0.c:468-476` targets the `.rec` path is inferred from the preceding construction.
- SSE: `Content-Type` charset and transfer coding are Starlette/uvicorn behaviour, not read. No stream was captured live. A rebuild can check framing with `curl -N` against `serve_api.py` fed a stub engine that replays `T`/`D` lines (`wire-protocol.md`).
- S29 is a negative proof over direct call sites and data pointers of a stripped binary; the warm-turn `n_cached` check in item 3 is the observation that would confirm it.
