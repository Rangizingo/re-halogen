# NPU closure round 2: U02, U01, U03

Scope: binaries/halogen-npu. Decompiles are work/decomp/halogen-npu/<addr>.c (cited `<addr>.c:line`). Ghidra vaddr minus 0x100000 = file offset; constants read with `xxd -s`. "proof" = read from decompile or bytes. "inferred" = reasoned, not shown. Builds on analysis/closure/npu-closure.md (r1). Note: the worktree copy has no work/ directory (untracked); inputs were read from /Users/peterb/Documents/AI/re-halogen, read-only.

## U02 Serve verb grammars

Verdict: PARTIAL. GEN grammar and GEN reply order closed with proof; the GEN numerics are named from the sampler's use of them. DEC/EMB/SCO/CLS request grammar closed, reply field order closed, meanings of a/c/ints inferred. PFX/INFO/LEDGER/QHN stay as in r1 except the corrections below.

### Compact grammar table

All tokens whitespace-split; token 0 = `VERB` or `VERB@model`; token 1 = tag (default `DAT_001eff65` if absent). Every number is parsed by a full-token check (strtod/strtoll, trailing garbage rejects the line).

| Verb | Request tokens after the tag | Reply |
|---|---|---|
| GEN | `max_new temp top_k top_p min_p penalty seed K stop1..stopK M id1..idM` | `OK tag n_prompt n_gen reason t1 t2 t3 t4 text bucket HASH_A,HASH_B,MISMATCH id id id...` |
| DEC | `n m` then n groups `a c i1..ic`, then n*m ints | `OK tag str int ints.. ints.. t1 t2` |
| EMB | same as DEC | `EMB tag str int u32.. t1 t2` |
| SCO / CLS | same as DEC | `SCO|CLS tag str rowlen u32.. t1 t2` |
| PFX | `n id1..idn` (r1) | `PFX tag a b` (r1) |
| INFO | none | `INFO tag <json-ish>` (r1) |
| ELOAD | `path int` (cache handler, see below) | `OK tag count` |
| EFREE | `int` | not decoded |
| CANCEL | tag of the request to cancel | none (r1) |

### GEN request (proof unless marked)
- Parser `FUN_001b18b0`, called from `0012a4a0.c:112-116` (`cVar1 = FUN_001b18b0(param_3,local_520,&local_160)`). Tokens 2..8 are all required (loop returns 0 if fewer, `001b18b0.c:26-45`). Store map, derived from the stack slot offsets (`(&dStack_90)[uVar8] = dVar12`, `001b18b0.c:30`) and the writes at `:49-62`:
  - token 2 -> `param_3[0]` int (`*param_3 = (int)local_80`). Line is rejected unless `0 < *param_3` (final `return ...(0 < *param_3)`). Used as the decode-loop bound: `while ((int)(...>>2) < *param_3)` (`001ba230.c:2466`), and in the fit check `*param_3 + iVar38 <= bucket*0x10` (`001ba230.c:335`). So token 2 = max new tokens, and prompt+max_new must fit a cache bucket (bucket unit 16 positions). Default preset 0x100 (`0012a4a0.c:112`) is dead because the token is required.
  - token 3 -> float `param_3[1]` = temperature. Sampler `FUN_001b8540.c:66`: `if (*(float *)(param_3 + 4) <= 0.0)` takes the argmax branch; `:614-615` divides logit by it. Greedy when <= 0.
  - token 4 -> int `param_3[5]` = top_k. `001b8540.c:249-258`: `uVar22 = *(uint *)(param_3 + 0x14); ... if ((int)uVar22 < 1) uVar25 = param_2;` (<1 = whole vocab, `param_2` = vocab size).
  - token 5 -> float `param_3[2]` = top_p. `001b8540.c:668`: `if (*(float *)(param_3 + 8) < DAT_001ef014)`; `DAT_001ef014` = 1.0f (bytes `0000 803f` at 0xef014); cumulative-probability cutoff `dVar34 >= top_p`. Pre-set 1.0 in `0012a4a0.c:113` (`local_15c = 0x3f80000000000000`).
  - token 6 -> float `param_3[3]` = min_p. `001b8540.c:675-707`: `if (fVar32 <= 0.0)` skip; else keeps ranks while `prob[i] >= fVar32 * prob[0]`.
  - token 7 -> float `param_3[4]` = additive logit penalty for ids already in history. `001b8540.c:104,237-238`: `if (*(float *)(param_3 + 0x10) != 0.0 && history non-empty)`, `local_60[id] = local_60[id] - *(float *)(param_3 + 0x10)`. Subtractive (presence-style), not multiplicative. History = `param_4` (generated ids; whether prompt ids count is inferred no).
  - token 8 -> u64 `param_3[6..7]` = seed. `001ba230.c:1318-1323`: `uVar107 = *(ulong *)(param_3 + 6); ... *0x5851f42d4c957f2d + lVar53` over 0x138 = 312 words, which is the std::mt19937_64 seeding recurrence. The `^ 0x8000...` branch in the parser (`001b18b0.c:56-62`) is the double-to-u64 conversion for values >= 2^63.
- Token 9 = K (required: `0x120 < size` test, `:63`). K ids (tokens 10..9+K) go to `param_3+8` vector. They are the stop ids: `001ba230.c:2355-2390` compares each sampled id against that vector and on a hit calls `FUN_001b0870(piVar82,&DAT_001ef680)` where `DAT_001ef680` is the string "stop". If K < 1 the list is empty and M starts at token 10.
- Next token = M (parsed, must be `>= DAT_001f4c48`, expected 0.0), then M ids into `param_2`, which is the prompt id vector (`001ba230.c:322` `uVar57 = param_2[1] - *param_2 >> 2` = prompt length `iVar38`). Whole line must be consumed (`param_1[1]-*param_1>>5 == local_a0`).
- Same shape as the Flash engine's GEN line (`image/serve_api.py:1074`: `GEN req max_tokens len(eos) eos... len(ids) ids...`) with the NPU inserting the seven sampler numerics before the eos count. That match raises confidence in the K=stop, M=prompt reading (inference supported by code above).
- Valid ranges: the parser enforces only max_new > 0. Vocab check of ids happens later: "a token id outside the vocabulary" (`001ba230.c:280-315`); "N tokens is longer than the largest prefill bucket" (`:355-360`); "positions past the weight file's RoPE table" (`:344`). Errors return as `ERR tag <msg>`.

### GEN reply (proof of order)
Built in `0012a4a0.c:169-205` as a left-to-right append chain. `local_e0` is the stats struct `param_5` of `FUN_001ba230`.
1. `OK ` (0xef0c9), tag
2. `n_prompt` = `param_5[0]` (`001ba230.c:460` `*param_5 = iVar38`)
3. `n_gen` = `param_5[1]` (`:2467` count of generated ids)
4. finish reason string at `param_5+2`: "stop" (`:2387`) or "cancel" (`:2462`); empty when it ran to max_new (inferred, empty string not seen)
5. t1 = `param_5+10` double x 1000 -> integer (`0012a4a0.c:~178 (long)(ram0x001f4bd0 * local_b8)`, `ram0x001f4bd0` = 1000.0, bytes `0000 0000 0040 8f40` at 0xf4bd0). Set at `001ba230.c:1310` (`dVar114 - dVar113`), prefill phase time. Unit: input is clock seconds (inferred; `FUN_00155ee0` is the timer), so t is milliseconds (inferred).
6. t2 = `param_5+0xc` x 1000, `:2469` whole decode loop time.
7. t3 = `param_5+0xe` x 1000, accumulated `FUN_00157b70` time (`:2135`, the logits read-back path, inferred).
8. t4 = `param_5+0x10` x 1000, accumulated sampler plus adjacent timing (`:2353-2354`, includes `FUN_001b8540`).
9. string at `param_5+0x12` (assigned at `:647`), or "-" when empty (`DAT_001ef3b6` = "-"). Meaning unknown.
10. u32 at `param_5+0x1a` = chosen cache bucket (`local_c84`, `:323,448`: smallest bucket b with prompt+max_new <= b*16).
11. ` %016llx,%016llx,%d` (`0012a4a0.c:241`): `param_5+0x1c` u64 = FNV-1a-style 64-bit hash (`*0x100000001b3`, basis 0x14650fb0739d0383) over a set of buffers (`001ba230.c:648-662`); `param_5+0x1e` u64 = same hash over the output buffer (`:2195-2203`); `param_5+0x20` int = count of memcmp mismatches (`:2189`). A determinism self-check (inferred).
12. generated ids, each preceded by a space.
Failure: `ERR tag <msg>`; malformed line: `ERR tag a malformed GEN line` (`0012a4a0.c:124`).

### DEC / EMB / SCO / CLS request (proof)
Parser `FUN_0011c0b0` (called `0012b110.c:1116`, `0012e9d0.c:616`):
- token 2 = n via strtoll, 1 <= n <= 0x40 (`0011c0b0.c:~70` `uVar19 - 1 < 0x40`); token 3 = m, 1 <= m <= 0x100000 (`lVar20 - 1U < 0x100000`). Fewer than 3 tokens or a bad n/m: parser returns 0 -> malformed.
- n groups follow from token 4: each is `a`, `c` (c <= 0x1000, `bVar38 = local_60 < 0x1001`) then c ints pushed to a per-group list (request +0x38 vector of vectors; `a` pushed to +0x28).
- then exactly n*m ints into a flat vector at request +0x8 (`uVar19 * lVar20 == lVar35`), and the token count must end there.
- Meaning (inferred only): n = number of sequences, a = a per-sequence scalar (label or position base), c ints = that sequence's token ids (c bounded by 4096, the usual max context for these small heads), n*m = an n by m grid of ints (candidate ids or positions). Not shown by code read.

### DEC / EMB / SCO / CLS reply (proof of order, `0012b110.c:1121-1215`, `00125240.c`)
The result struct `param_4` has: string at +0 (model id echo, inferred), ints, a list of vectors at +0x40/+0x48 (stride 0x18), an int list at +0x58..+0x60, a u32 list at +0x70..+0x78, doubles at +0x88 and +0x90.
- DEC (the true branch of the `DEC` compare, `FUN_00125240`): `OK <tag> <str> <int @+0x20, %d> <ints of +0x58 list, %d each> <u32 of every vector in +0x40 list, flattened> <t1> <t2>`. So DEC does answer with `OK `, not `DEC ` (r1 and npu-engine.md say `DEC `; `00125240.c` starts with the 3-byte `DAT_001ef0c9` = "OK ").
- EMB (`0012b110.c:1175`, prefix bytes `454d 4220` = "EMB " at 0xef445): `EMB <tag> <str> <int local_218> <u32 list +0x70..+0x78> <t1> <t2>`. `local_218` meaning unknown (inferred: embedding dimension).
- SCO / CLS (prefix chosen by the compare against 0xef176 "SCO" or 0xef17a "CLS"): `SCO|CLS <tag> <str> <row length = size of first vector in +0x40 list> <u32 of every vector, flattened> <t1> <t2>`. Values print as unsigned decimal u32, possibly float bit patterns (inferred; score takes 2 rows, classify 2..16 per npu-engine.md).
- t1, t2 = `param_4+0x88` and `+0x90` doubles x 1000.0 as integers (same constant as GEN). Probably ms (inferred).
- Malformed request: `ERR <tag> a malformed <VERB> line` (`0012b110.c:1105-1113`).

### Corrections to r1 and a new finding
- Verb literals: bytes at 0xefd82 are `QHN\0QLED\0LOOKUP\0EDUMP\0ELOAD`; 0xefd83 is therefore the string "HN" (r1 read it as "QHN"). The registry handler `FUN_00172600` (registered at `00119560.c:125`, `*(code **)(lVar8 + 0x40) = FUN_00172600`) is a cache command handler accepting `QHN`, `QLED`, `LOOKUP`, `EDUMP` (0xefd92), `ELOAD`, `EFREE` (`00172600.c:104-113,192-215`); else `usage: <verb>`. `0012b110.c:231,256,270` separately compare against "HN", "LEDGER", "INFO" (the front-end-visible verbs); how the front end maps those to QHN/QLED was not traced. This is an open point on QHN/LEDGER naming.
- `ELOAD`: size test `param_2[1]-puVar25 == 0x80` = 4 tokens: `ELOAD tag path int`. Path token goes to `FUN_00170ea0` (HNPB cache-entry loader, uses `FUN_00167960`), int via strtol, reply `OK tag <FUN_00172370 count>`. `EFREE tag int`: 3 tokens (size 0x60), looks the int up in a std::map keyed by int (`00172600.c:192-215`). `LOOKUP` and `QHN`/`QLED` argument lists not decoded.
- The in-memory cache entry object created by ELOAD stores `DAT_001f4bf0` = `{1,1}` at +8 (`00172600.c:119` `puVar11[1] = DAT_001f4bf0`), the same 8 bytes the streaming HNPB writer puts at header +0x04..+0x0b (see U01).

What remains for U02: handler bodies for LOOKUP/QHN/QLED/EDUMP, how PFX/HN/LEDGER/INFO map onto them; GEN reply fields 9 (string) and the IMG argument list; what DEC's a/c/n*m ints mean (need the op that consumes the request in `FUN_0014a6f0`, 1224+ lines not read); `local_218` in the EMB reply.

## U01 HNPB header u32 at +0x08 and per-blob u32 at +0x50

Verdict: PARTIAL. Lifecycle is now clear (what writes them, what the readers do with them); semantics of +0x08 and +0x50 are narrowed but not proven.

### Header +0x08 (proof of values and non-use, inference for meaning)
- Value 3 in `FUN_0016eb50` (bulk writer; callers `00172600.c:179` inside the cache handler and `00171ca0.c:81` = `cache ecopy IN OUT CAP`); value 1 in `FUN_0017a920` (streaming finalizer, `0017a920.c:115-119` `local_e8._4_4_ = (undefined4)DAT_001f4bf0; uStack_e0 = (undefined4)(DAT_001f4bf0 >> 32)`). r1 verified the constants (`01 00 00 00 03 00 00 00` at 0xf4c28, `01 00 00 00 01 00 00 00` at 0xf4bf0).
- Reader `00167960.c:350` copies it to the container (`*(int *)(param_1 + 2) = piVar11[2]`) and checks only magic 0x42504e48, version 1, file size == header +0x28 (`:352-353`), `table_off + n*0xa8 <= size` (`:355`).
- No consumer: callers of the reader are `00145eb0`, `00170ea0`, `001a7be0`, `001d6480` (grep `FUN_00167960`). Their accessors reach blobs only through `FUN_00162ec0` (lookup by name) and then use record `+0x28` offset and `+0x30` size: `0016dc30.c:19-28`, `001b2620.c:55-60`, `00144430.c:428,478`, `00145eb0.c:127,569`, `001a7be0.c:1063-1064`, `001d30b0.c:1529`. None of these read container +0x10 (the header u32) or record +0x20 (the tag). Not every use of those callers was read; this holds for the windows above.
- Meaning, inference: a producer tag. 3 = bulk-written pack (the cache `ecopy`/save path, and the device-set `consts`/`probe` output through `0016eb50`), 1 = streamed pack (`convert`, `001a7be0`/`00196980`). The ELOAD cache object also initialises its own copy to `{1,1}`. The binary neither validates nor branches on it. A shipped file would show it: read bytes 8..11 of one `.hnpw` (convert output, expect 1) and one `consts.hnpb` (expect 1 or 3) to confirm which writer made which; none of those files are in the repo.

### Per-blob record (0xa8 bytes, table at header +0x18)
Table record in the file (positions proof, meaning marked):
- +0x00 name (<= 0x4f chars, NUL padded; strnlen 0x50 in reader `00167960.c:~365`; writer rejects 0 or > 79 chars, `0016eb50.c:1122-1128`)
- +0x50 u32 tag
- +0x54 u32: 1 in the bulk writer (`0016eb50.c:1135` `*(undefined4 *)(pvVar13 + 0x54) = 1`), 0 in the streaming writer (`0017a920.c:88` writes a u64 whose high half is 0). Inferred ndim of a 1-D blob in bulk packs; streamed packs leave it 0.
- +0x58 u64: bulk writer stores `plVar17[8]` (`0016eb50.c:1136`, one u64, inferred element count of the 1-D shape). Streaming writer fills +0x58..+0x68 with up to three u64 from the entry's dims vector (`0017a920.c:89-94`, `auStack_90[uVar2] = ...`). Together: shape/dims array, rank at most 3 (inferred).
- +0x70 u64 data offset (4096-aligned and `off+size <= file`, `00167960.c:427-429`), +0x78 u64 size
- +0x80..+0x98 four u64 copied verbatim from the entry (`0017a920.c:100-104`); never read by the reader (unknown)
- +0xa0 u32 xor32 of blob bytes, verified by `FUN_00162fd0` (r1, proof)

Tag at +0x50: the reader stores it at record +0x20 (r1) and no accessor above compares it. The writer takes it from the in-memory entry's field at +0x20 (`plVar17[4]` bulk; `puVar9[4]` streamed). The one other dtype check in the binary is for safetensors input (`001acdc0`, `0018a5b0` parse `"dtype"` JSON) and `001eb920` ("has an unexpected dtype or shape"), which compare names from the safetensors header, not this tag. Best reading: a dtype/layout code kept for tooling (convert and `probe` printer, `FUN_0016eb50` is mapped as `probe_printer`), not used by the engine at load time. Inferred.

What remains: the entry builders in `00172600`/`00171ca0`/`convert` that set entry field +0x20 (needs the dtype enum, which would also say what 1, 2... mean), and one real `.hnpw`/`consts.hnpb` header dump.

## U03 main:<graph> names

Verdict: STILL OPEN (literal names: none; construction already shown in r1 `00159980.c:~85-100`).

- Literal graph-name strings found in the binary: none of the form `main:<name>`. Only the prefix is built at runtime from `"main"` + `':'` + a runtime string (r1). A scan of the decompile index strings column (`work/decomp/halogen-npu/index.tsv`, awk for `main|graph|gx/|kv/`) returned no graph-name literal rows; `strings -a` shows only these fragments: `gx/LH`, `gx/LL`, `kv/AH`, `meta` (e.g. string-table lines 4536-4701, 8953). They look like per-layer buffer name prefixes (`gx/L<layer>`, `kv/A<n>`), not graph names (inferred).
- Which shipped file holds the rest: `devices/devices.hnpm` (text manifest, `ops`/`plan` records per npu-engine.md section 2.1) names the kernels, and each `devices/<shape>.elf` (u0..u9.elf per `image/npu-models.txt`) holds the entry symbols the XRT kernel lookup resolves with the `main:` prefix. Neither is in the repo, so the name set cannot be enumerated here.
- To close: obtain one model directory's `devices.hnpm` and read the plan/ops names, or decompile `FUN_0013f420` (the name builder r1 flagged) to confirm which record field supplies it.
