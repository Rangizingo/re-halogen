# S10 and S11 closure: prefix-cache key and hit rule, KV pool allocator (flash_serve 0.17.3)

Cites are `D:<addr>.c:<line>` = `work/decomp/flash_serve/<addr>.c`. "proven" = decompile line plus quoted text. "inferred" = reasoned from structure. Extends `analysis/closure/cache-host.md` (cap formula, env knobs, disk tier not repeated).

## Correction to the trace

`key_compare` `01868c70` is NOT the prefix-cache entry key. It is the lookup for the composable CHUNK cache (`param_1[8]` of `01846b20`), called once per pending chunk descriptor (`D:01846b20.c:254`). The prefix entry hit rule is `01861160` (unnamed in `function-map-core.tsv`; I call it `cache_find_longest_prefix`). Callers: `0187fb00` (`D:0187fb00.c:29`, :46) and the admit path of `0182ea80` (`D:0182ea80.c:5679`).

## 1. Prefix cache

### 1a. Entry layout (offsets proven, names inferred)
Entry vector at cache+0x70..+0x78, stride 0x160 (`D:01861160.c:80`, `>>5 * 0x2e8ba2e8ba2e8ba3`). Fields used:
- +0x00 token vector begin (`*plVar10`, compared to the prompt, `D:01861160.c:39`).
- +0x18/+0x20 second vector begin/end, per-token side data (`plVar10[3]`, `[4]`, :42). Inferred: image or modality ids; compared to the request's matching vector `param_4`.
- +0x48 length in tokens (`D:01861160.c:26`). +0x50 region base in the KV pool (`D:0182ea80.c:5683-5685` reads `+0x50` and `+0x48`).
- +0x88 full-entry flag, +0x89 tail-checkpoint flag, +0x8a anchor flag, +0x8b shared-anchor flag (tests and strings at `D:0186cdc0.c:23-36`).

### 1b. Hit rule (`01861160`), proven
Scan every entry. Entry `i` is a candidate when all hold (`D:01861160.c:28-31`):
1. `len != 0 && len <= prompt_len`.
2. `len != prompt_len || full_flag == 1`. Quoted: `(uVar1 != param_3 || ((char)plVar10[0x11] == '\x01'))`. An entry as long as the whole prompt counts only if it is a full entry. A tail checkpoint or anchor never covers the whole prompt.
3. Longer than the best candidate so far (`*(int *)(lVar4 + 0x48 + uVar7 * 0x160) < (int)uVar1`, :31). Longest wins; ties keep the lowest index.
4. Exact int32 token equality over `len` tokens (loop `D:01861160.c:38-41`).
5. Side-vector equality (`D:01861160.c:42-72`): if the entry has no second vector, the request's must be all zero over `len` (or absent); if it has one, the first `len` elements must equal the request's. Absent request vector (`param_4 == 0`) is treated as zeros.
Returns entry index or `0xffffffff`. The match length is the matched entry's `len`, not a computed common prefix: no partial match inside an entry. So `n_cached` is always some entry's length (inferred link to the D line; observed 19 warm in `runtime-run-0.17.3.md` fits a 19-token entry, not re-run).

### 1c. Alignment (`cache_align`), boundary rule proven, INFO field inferred
Snapshot boundaries are rounded down to a multiple of 64 inside a chunk. `D:018680b0.c:98-101`: `rem = (boundary - chunk_start) % chunk; if (0x3f < rem) pos = (rem & 0x7fffffc0) + (boundary - rem)`. `rem < 0x40` means no snapshot at that boundary (:106, :141). `forward` uses the same `0x3f` mask (`D:018fa150.c:50-83`, from S13). The INFO `cache_align` field was not traced to its producer; 64 matches the code, so inferred.

### 1d. What makes an entry usable
Proven:
- The hit rule (1b).
- Restore `0187fb00`: with in-place mode (`+0xd9`, `HALOGEN_CACHE_INPLACE`) and a pooled model (`model+0xcd == 1`) it compares the slot's region base `*(model+0xb0 + 0xd4 + slot*0x148)` with entry `+0x50` (`D:0187fb00.c:55-59`). Equal: rows are already in place. Different: it copies rows with `kv_copy_region` `01901c50` (call follows at :60), or runs cold (S11 strings).
- Admit path of `0182ea80` takes the hit's region as the request's region (`D:0182ea80.c:5679-5686`: `FUN_01861160` then `+0x50` becomes `local_18b0`, `+0x48` becomes `local_17d8`).
- Disk tier is consulted when RAM misses or the disk entry is longer (`D:0187fb00.c:33-47`, insert `018806e0`).
Inferred: a failed restore drops the entry and runs cold.

### 1e. Snapshot store, kinds, branch cap (`0186a470`)
Signature (`D:0186a470.c:3-4`): `(cache, tokens, n, side_vec, clock, full, tail, anchor)`, proven. Kinds from `D:0186cdc0.c:23-36`: full entry (+0x88), tail checkpoint (+0x89), system-prompt anchor (+0x8a; shared +0x8b). An unflagged entry is a leaf; its leaf names ("a branch point", "a conversation's newest leaf", "an older leaf", `D:0186cdc0.c:60-104`) are derived at eviction time, not stored.
Per-conversation cap: `branches << SNAP3` (`D:0186a470.c:156-158`: `*(int*)(param_1+0x88) << (*(byte*)(param_1+0xa1) & 0x1f) <= iVar15`, then counter `+0x118` incremented and the oldest same-conversation leaf replaced, `LAB_0186ab79`). Reading of that block is partial: proven the compare and the counter, inferred the "oldest leaf replaced".
Total cap `+0x8c`: `D:0186a470.c:210-216`: `if (cap <= count) do { FUN_0186b270(...) } while (cap <= count)`.

### 1f. LRU, 24-entry cap, eviction, proven
- Cap in the 0.17.3 banner: 24 (4 slots). Defaults: `*(param_1+0xa0) = 0x101010101` (`D:01852ca0.c:48`) sets bytes +0xa0..+0xa4 to 1: FULL=1, SNAP3=1, DEEPEN=1, CKPT=1, rank-evict=1. Cap = `slots * ((branches<<SNAP3) + FULL + 1)` (`D:01852ca0.c:128-130`). 24 = 4 x 6 gives branches=2 (inferred; the default dword at `_DAT_00113eb0` was not read).
- Victim choice `0186b270` in rank mode (`+0xa4 == 1`, default), `D:0186b270.c:29-76`: lowest rank wins; ties go to the smaller clock (summary `+8`, :66-68); summary byte +0x21 set means skip (:34). Ranks: full entry or tail checkpoint 0 (the test `+0x10`/`+0x11` fails, `uVar11` stays 0, :36-37); leaf with a newer sibling in the same conversation 0 ("older leaf", :48-56); newest leaf 2 (`iVar3 * 2`, :56); branch point 3 (:57-58); anchor `3b+1` = 1, shared anchor 4 (:62-63).
- `HALOGEN_CACHE_EVICT=0` sets `+0xa4 = 0`: pure LRU on the clock, any kind (`D:01852ca0.c:111-118` and its stderr text "at the cap the least recently used entry goes, whatever its kind"; code path `D:0186b270.c:77-106`).
- "forgot a full entry": `"prompt cache: at its %d-entry cap, forgot %s of %d tokens (region %d, the least recently used %s) to store %s\n"` (`D:0186cdc0.c:120`), `%s` = "a full entry" when +0x88 is set (:24). Logged first, then the entry is removed by `0186d0c0(cache, idx, 1)` (`D:0186b270.c:113-114`). Last `%s` is "of its rank" when `+0xa4 != 0`, else "entry" (:116-119).
- Further loss: entries longer than the hit are dropped when their region must grow (2d).
Unproven: the clock unit and who stamps it (`param_5 = 0xffffffff` at `D:01846b20.c:266` suggests "stamp now").

### 1g. Chunk cache key (`01868c70`), proven
Entry stride 0xb8 (`plVar2 + 0x17` longs, `D:01868c70.c:16`), vector at `param_1+0x50..+0x58`. Match (`D:01868c70.c:15-27`): `*entry == param_2` (64-bit id; caller passes `*(piVar26+4)`); `entry[6] == end - start`; `bcmp(entry tokens (+0x18), prompt + start*4, count*4) == 0`; and, when the caller's flag `param_8` is set (`iVar10 == 1`), `entry byte +0xa4 == 1`. Returns 1 on no match; on match it falls into `_Znwm(0x188)` (a restore job object; decompile ends there). The caller advances `param_2[0x65]` (the position) to the chunk end only when the result is 0 (`D:01846b20.c:254-259`), and only for a chunk whose start equals the current position (`piVar26[1] == param_2[0x65]`, :252). It then snapshot-stores if the new position equals SNAP/SNAP2/SNAP3 (:261-266).

## 2. KV pool allocator

### 2a. Structures, proven
- One pool of `P` positions: `engine+0x48` int (`puVar3[9]` in `0184de40`; `(param_1->__data).__spins` in `0182ea80`). 524288 in the 0.17.3 log.
- Region table: sorted vector of 12-byte records `{base, cap, owner}` at `engine+0x50..+0x58` (`D:0184de40.c:34-45`). `owner == -2` is a HELD region (no running request, rows are cache) (`D:0184de40.c:46`, `D:0184e2f0.c:46`); other values are the owning slot. Sorted by base (`D:0184de40.c:197-206` linear search for the insert point).
- Every region end is rounded UP to a multiple of 256 positions: `end = (base + cap + 0xff) & 0xffffff00` (`D:0184de40.c:101-106`; `D:0182ea80.c:5569-5574`, :5653-5659, :5817-5822). 256 is the pool block; 64 is the snapshot align.
- `kv_move_region` `01901ee0`: pooled only (`model+0xcd`, `D:01901ee0.c:16`). Gap must be a multiple of 4 ("the gap is not whole blocks", :24-30). If `|a-b| < n` (overlap) it copies in `|a-b|` chunks, forward when `b < a` (:33-47), backward otherwise (:49-60), each via `01901c50`, then `hipDeviceSynchronize` (:61). Non-overlapping: one `01901c50` call (:73). The "overlapping, copied in chunks" suffix is chosen in `0184de40` (:168-176).
- `01900510 slots_alloc` and `01900aa0 slots_alloc_pooled` are NOT region allocators. They size per-slot device buffers (fixed 0x300000, 0x14000, 0x300 byte buffers per layer). The private-KV variant adds `<<10` and `<<8` sized buffers (`D:01900510.c:55`, :60, :65); the pooled variant does not (`D:01900aa0.c:59-62` zero them). Batch B limited to 1..64 (`D:01900510.c:20`, `D:01900aa0.c:17`). The pooled one sets `model+0xcd = 1` (`D:01900aa0.c:38`) and exits if more than one slot already holds a private KV (:31). So in pooled mode slots own state only; KV comes from the region table. Both decompiles end in `FUN_01892e30` (no-return allocation), so the loop after the first allocation is cut off.

### 2b. Region size for prompt plus max_tokens, proven
`D:0182ea80.c:5649-5662`:
```
need = min(P, max_tokens + prompt_len)
need = round_up_256(need); if (need > P) need = P     // local_1888
```
Quoted: `uVar92 = (int)lVar96 + 0xff; ... uVar92 = uVar57 & 0xffffff00; if ((int)uVar35 < (int)(uVar57 & 0xffffff00)) uVar92 = uVar35;`. The banner's "it reserves prompt + max_tokens" is exact, plus rounding to 256, capped by P.
Admit-time clamp: `D:0182ea80.c:5304-5308`: `max_tokens = min(max_tokens, P - prompt_len)` (`iVar74 = spins - prompt; if (max_tokens <= iVar74) iVar74 = max_tokens`). The condition `prompt < P` at :5217 guards the grammar and refusal block that follows; I did not confirm that a prompt of `>= P` goes to `refuse_request` `018497a0` (inferred).

### 2c. Placement
Proven:
1. If the cache gave a hit, the request first tries the hit's region (`local_18b0`), which must be held (`owner == -2`, `D:0182ea80.c:5690-5693`, :5702-5709).
2. Otherwise first-fit over the sorted table: gap before record `i` is `base_i - prev_end`, `prev_end = round_up_256(prev.base + prev.cap)`; the first gap `>= need` wins (`D:0182ea80.c:5822-5835`; `D:0184de40.c:97-111`; the final gap `P - prev_end >= need` at `D:0184de40.c:112`).
3. If none fits, the wait message (`D:0182ea80.c:5602`): `"kv pool: request %ld waits for %d positions: the pool holds %d of %d in %d busy and %d held regions; %d free, the largest free gap %d (%s)%s; it waits for a retirement\n"`. Free total and largest gap come from the loops at :5565-5600.
Inferred: the new record is inserted at the first-fit base with `cap = need` and `owner = slot` (insert helpers `0184e740` at `D:0182ea80.c:6088`, :6106; `FUN_0187f080` at `D:0184de40.c:208`). Not read to the end.

### 2d. Growth, move, relocation, eviction ladder
Entered when the hit's region is held but `region.cap < need`. Proven that each step exists and what it does; the ORDER is inferred (function is 9500 lines of gotos).
1. Grow in place, `0184e2f0` (called `D:0182ea80.c:6038`, :6913). Allowed when the next record's base (or `P`) is `>= base + need` (`D:0184e2f0.c:63-71`). Entries in that region longer than the hit are forgotten via `0186d0c0(cache, i, 0)` (`D:0184e2f0.c:78-91`), chunks too (:92-135). Log `"... the region at %d this request hit grows in place: forgot %d longer entries%s (%s)\n"` (`D:0184e2f0.c:138`). Cap set to `need` (`piVar4[1] = iVar6`, :168), owner set to the slot (`piVar17[2] = slot`, :171).
2. Move, `0184de40` (`D:0182ea80.c:5957`, :6059, :6915). Preconditions (`D:0184de40.c:46-58`): region held, disk tier object exists (`puVar3[1] != 0`), hit length `> 0`, and for reason "no loss" the flag `*param_1[3]` must be 1. It removes the hit's record (:65-86), then first-fit scans the rest (:97-111). If a gap `>= need` exists (:112): entries longer than the hit are forgotten (:117-141); if the new base equals the old, it is "(grown in place)" (:166); otherwise wait for the disk tier (`"waited %.1f ms for the disk tier to finish reading the region at %d before moving it"`, :152) and `kv_move_region(old, new, hit_len)` (:155); relocation counter `puVar3[0xd]++` (:161). Log `"... moved the %d rows this request resumes from, region %d -> %d%s%s in %.1f ms (%s)\n"` (:186). Only the hit's rows move, not the whole region. If no gap fits, the record is put back (:192-208) and it returns 0.
3. Compact held regions: `D:0182ea80.c:9407` `"moved %d held regions (%lld rows, %.1f ms) up against the next busy region and the region at %d this request resumes from grows in place (no loss)"`; the rows-moved total accumulates in `engine+0x88` (`D:0182ea80.c:9402`).
4. Forget longer entries rather than wait (`D:0182ea80.c:6258`): `"no room ... and forgetting no held region could make any; forgot the %d entries longer than the hit ... rather than wait"`.
5. Forget entries longer than the hit when the LRU victim region is newer and would cost more rows than it frees (`D:0182ea80.c:6645`, text "the least recently used region, at %d, is newer and would cost %d of its %d rows to rebuild"); the `ladder:` trace at :6562 prints hit, rows past it, clock, victim, victim clock, rows, rows to rebuild.
6. Forget the LRU held region in the way with its entries and chunks (`D:0182ea80.c:6999`, counts entries by region at :6965-6990, chunks at :6992-7002).
7. Forget chunks the request would have composed (`D:0182ea80.c:6733`).
8. Else wait for a retirement (2c step 3).

### 2e. `max_tokens` clamp and `room_from`
Proven (engine side), `D:0182ea80.c:6000-6030` (hit region cannot grow, no move allowed):
```
room  = region.cap - prompt_len                    // uVar35
floor = max(max_tokens * 15 / 100, 1024)           // :6003-6005
if ((floor << (budget2 > 0)) <= room) {            // budget2 = uStack_15a4
    saved = max_tokens; max_tokens = room          // local_1558 = old, :6007-6008
    if (budget2 > 0) budget2 = min(budget2, room - floor)
    engine counter (param_1[3]) += 1               // :6014
    log "the turn runs in the %d positions it has left (max_tokens %d -> %d)"   // :6019
    region owner = slot; 01900fc0(...)             // :6020-6027
}
```
The same expression repeats at :6158-6160. So a turn is allowed to run short only when the room left is at least 15 percent of the asked `max_tokens` (minimum 1024; doubled when the second budget is on). The old value `local_1558` is what a `room_from >= 0` would carry (wire-protocol.md); the store of `local_1558` into the reply was not found, so that link is inferred. The meaning of `budget2` (inferred: drafter or MTP lookahead budget) is unconfirmed.
Admit-time clamp is separate: `min(max_tokens, P - prompt_len)` (2b).

## 3. Still unproven
- INFO and CSTAT producers: `cache_align`, `relocated`, `cold_resorts`, `pool_used`. Candidate counters: `puVar3[0xd]` (moves, `D:0184de40.c:161`) for `relocated`; `engine+0x88` (rows moved) and `param_1[3]` (clamps). `cold_resorts` has no matching counter found.
- Default branches dword `_DAT_00113eb0` (2 inferred from 24 = 4 x 6).
- `key_compare` success body (0x188 object) and what the chunk restore copies.
- Clock unit and stamper; side-vector meaning (+0x18); `slots_alloc*` loop tails.
- Ladder step ORDER (2d), and whether each step is reachable on a given path.
- Insert of the new region record (2c, inferred).
- No runtime check: the engine log was not searched for an event matching each message. All findings are static decompile reads.
