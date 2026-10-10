# Env flags and D-line counter writers (round 5)

## Scope

Two items from `analysis/rebuild-guide.md` section 11, "Other gaps":

1. The complete list of environment variables `flash_serve` reads (name, getter, default, selects, cite). Full table: `analysis/env-flags.md`.
2. Writers of the `D` reply-line counters `shared`, `dv_*`, `kept_*`, `think_*` (`analysis/wire-protocol.md`, section 3 D table and section 6).

Static only. Binary `binaries/flash_serve`, decompile `work/decomp/flash_serve/`, Ghidra address = VMA + 0x100000. Disassembly with capstone from the file, nothing executed.

## Verdict: CLOSED (item 1, environment variables)

Counts and commands (also in `analysis/env-flags.md`):

- 418 `getenv` call sites, 256 distinct variables, all `HALOGEN_`. Command: scan the executable segment for `e8 rel32` calls to the `getenv` PLT stub (VMA 0x18f4320), resolve the preceding `lea rdi,[rip+d]` to its string; 418 sites, 0 unresolved.
- The decompile text alone gives 247: `grep -ho 'getenv *( *"[A-Za-z0-9_]*"' work/decomp/flash_serve/*.c | sed 's/.*"\(.*\)"/\1/' | sort -u | wc -l` prints 247. Nine more exist only in disassembly because the decompile is truncated at those sites (`ACT_STATS` `0190c8b0.c`, `DN_FUSED_NORM` `018ab5c0.c`, `DRAFT_VOCAB_Q8` `01891c50.c`, `GGUF_CACHE` `019aa860.c`, `HT_TRUNK_RESIDENT` and `VISION_ATTN` `01884460.c`, `I4R_DN_FOLD` `018d2430.c`, `MTP_PREFILL` `018fa150.c`, `QSA_NOSKIP` `018b86d0.c`). Example, `HALOGEN_DRAFT_VOCAB_Q8` at 0x1891d68: `lea rdi,[str]; call getenv; test rax,rax; je; cmp byte [rax],0x31; sete cl; lea eax,[rcx*4+4]` stored to `[rbp+0x58]`, so default 4 bits, "1" gives 8 bits.
- 265 standalone `HALOGEN_` strings in the file; 256 have a `getenv` reader; 9 have none (`CACHE_ALIGN`, `CACHE_MB`, `CACHE_RESERVE_MB`, `DRAFTER`, `QSA_SEL_STATS`, `SERIAL_DUMP`, `SERIAL_REF`, `SLOT`, `SLOT_CTX`). Those 9 appear only in the skip table below.
- No `secure_getenv`, `setenv`, `putenv` imports. One `environ` scan: `FUN_01854310` (`01854310.c:111`, `PTR_environ_019fa550`).

New finding. `FUN_01854310` is the disk-cache fingerprint builder. It hashes every `HALOGEN_*` variable that is NOT in a 61-pointer skip table at Ghidra 0x19f60e0 (`01854310.c:146`, loop bound `lVar8 != 0x1e0`; pointers resolved through `.rela.dyn`). 52 of the 256 read variables are in that table, plus the 9 unread names, 61 total (52 + 9 = 61 checks out). The other 204 read variables change the cache fingerprint when set. This also explains why strings like `HALOGEN_CTX` and `HALOGEN_MAX_TOK` show up in the binary: message text only.

## Verdict: CLOSED (item 2, D-line writers), with one inferred meaning

### The format and field map

Format string, `work/decomp/flash_serve/0184c9a0.c:99` (`FUN_0184c9a0`, `snprintf` into a 0x1c0 byte buffer, then queued as the reply):

`D %ld %s %d %d %.1f %.1f %d %ld %ld %d %ld %ld %ld %ld %d %d %d %d %d %d %d %d %d %d %.1f %ld %ld %ld %ld`

Ghidra prints the vararg order wrong (floats and ints interleaved), so the map below comes from the disassembly of the call: rcx, r8, r9, then 23 stack pushes (`184cb3e..184cbae`, reversed), xmm0 to xmm2 for the three doubles (`al = 3`). Stack slots resolved to S-relative addresses; `S` here means rsp after the six pushes and `sub rsp,0x298` (not the stats object; the stats object is called R below).

Q = request record (formatter argument 3). R = per-request speculation stats object, `Q+0x10` points to it.

| Field (parts[i]) | Source | Notes |
|---|---|---|
| 1 req | `*(u64*)(Q+0x00)` | `mov rcx,[rsp+0xb8]` = `[Q]` |
| 2 reason | argument 4 of the formatter | string passed by the caller ("stop", "length", "cancel", "error") |
| 3 n_prompt | `Q+0x54` | r9 |
| 4 n_gen | `Q+0x58` | |
| 5 prefill_ms | double at `Q+0x190` | xmm0 |
| 6 decode_ms | `(now - *(i64*)(Q+0x198)) / 1e6`, 0 unless `n_gen >= 2` | xmm1, `184c9f5..184ca29` |
| 7 drafter | `Q+0x0c` | |
| 8 rounds | argument 5 | caller passes `R[8] + R[0]` |
| 9 commit | argument 6 | caller passes `R[1] + rounds + R[2] + R[9] + R[0x10]` |
| 10 n_cached | `Q+0x5c` | |
| 11 shared | argument 7 = `R[2]` = `*(i64*)(R+0x10)` | writers below |
| 12 pld_rounds | argument 8 = `R[8]` | |
| 13 pld_accepted | argument 9 = `R[9]` | |
| 14 pld_drafted | argument 10 = `R[0xc]` | |
| 15 n_composed | `Q+0xd8` | |
| 16 room_from | `Q+0x160` | |
| 17 max_tokens | `Q+0x50` | |
| 18 pool_used | sum of the dword at `+4` of each 12-byte entry in the vector `[P+0x50, P+0x58)`, only if `*(u32*)(P+0x48) != 0` | P = formatter argument 1 (the KV pool) |
| 19 pool_positions | `*(u32*)(P+0x48)` | |
| 20 think_forced | `byte(Q+0x13d) ? byte(Q+0x13e) + 1 : 0` | `184caac..184cabe` |
| 21 think_seen | `Q+0x138` | |
| 22 kept_think | `Q+0x1bc` | |
| 23 kept_call | `Q+0x1c0` | |
| 24 disk_rows | `Q+0xdc` | |
| 25 disk_ms | double at `Q+0xe0` | xmm2 |
| 26 head_drafted | argument 11 = `R[0] - R[0xd] + R[0xe]` | |
| 27 dv_tokens | `*(i64*)(R+0xa0)` | 0 if R is NULL |
| 28 dv_misses | `*(i64*)(R+0xa8)` | |
| 29 dv_switches | `*(i64*)(R+0xb0)` | |

R is passed as argument 13 (`01845e50.c:149` `plVar18 = plVar13[2]`, call at `01845e50.c:157`). In the main loop `0182ea80.c` it is `pauVar8`, so `R[2]` is `*(long*)pauVar8[1]`. That `pauVar8` is the same object as `plVar18` is inferred from identical offsets (the dv writer receives `pauVar8` as its `param_4` and increments `+0xa0..+0xb0`, which the formatter reads from `plVar18`). Callers of the formatter: normal completion `01845e50.c:157`, cancel `01847a30.c:66`, error `01848190.c:450,534`, `0182ea80.c:1161,1226,2226,3487`.

### shared (parts[11])

Writers: `0182ea80.c:8355` and `0182ea80.c:8633`, both `*(long *)pauVar8[1] = *(long *)pauVar8[1] + 1;`. Each sits right after the tick samples one token on its own (`uVar35 = FUN_018514a0(pauVar8,...)`, then the grammar check, then `+1` to `shared`, `+1` to `R+0x28`, then `FUN_01882d20` and the stop check `FUN_01851580`). No other store to `R+0x10` exists in the decompile (`grep -n 'pauVar8\[1\]\b' 0182ea80.c` shows these two plus the `+8` field at `:8587` and `:9004`).

Meaning (inferred from evidence, not stated in code): tokens committed by the tick's own single-token sample, outside a draft/verify round. Live lines from `analysis/evidence/`: `D 900001 length 40 8 474.1 95.5 1 1 7 0 1 ...` has n_gen 8 = commit 7 + shared 1; the PLD line (`analysis/closure/pld.md:162`, `... 64 271.3 1242.7 1 22 63 0 1 4 18 24`) has n_gen 64 = commit 63 + shared 1; a drafter-0 line (`D 900005 length 40 32 ... 0 0 0 40 0 ...`) has commit 0 and shared 0. So with speculation on, `n_gen = commit + shared` in every checked line, and the one shared token is the first token sampled from the prefill output.

### dv_tokens, dv_misses, dv_switches (parts[27..29])

Single writer: `FUN_01882d20` (`01882d20.c:14`, `:17`, `:30`), already found in `analysis/closure/dv-quant-r3.md` section 2 and re-read here against the formatter map:

- `:14` `*(long *)(param_4 + 0xa0) += 1`: dv_tokens, once per token fed while a draft-vocab subset is active (`param_1[2] >= 0`).
- `:17` `*(long *)(param_4 + 0xa8) += bVar3 ^ 1`: dv_misses, where `bVar3 = FUN_018f5320(ctx, subset)` is the subset bitset membership test (token not in the active subset adds 1).
- `:30` `*(long *)(param_4 + 0xb0) += 1`: dv_switches, when `FUN_01871780` changes the active subset id (`param_1[2]`).

Callers (token loops): `01851770.c:106`, `01851dc0.c:284`, `01860e60.c:15`, `0182ea80.c:8357, 8532, 8635, 8940`. The three offsets map to parts[27..29] in that order by the disassembly (r10, rbp, r13 pushed last). Not covered: the scoring formula in `FUN_01871780` (see dv-quant-r3.md Verdict).

### kept_think, kept_call (parts[22..23])

Single writer: `FUN_01849530(Q, token)` (`01849530.c`), the end-of-turn guard run on every committed token (callers `01851580.c:66`, `01851210.c:62`, `0182ea80.c:2186`).

- `01849530.c:40-41`: `Q+0x1bc` (kept_think) is incremented when a stop token arrives while the guard flag `Q+0x1a8` is 1 and the in-think flag `Q+0x1a9` is 1 (and the id is not the think-close id). It keeps up to 16 (`iVar1 < 0x10` returns 0, the stop token is kept inside the think block).
- `01849530.c:60-61`: `Q+0x1c0` (kept_call) is incremented the same way when `Q+0x1aa` (in tool call) is 1, up to 4 (`< 4`).
- Flags and ids come from the `GUARD a b c d e` GEN field (`0182ea80.c:3203-3216`, "malformed GUARD" check, two 0/1 flags and three ids); the store into `Q+0x1a8..0x1b8` was not traced, the layout is read from `01849530.c`.

### think_forced, think_seen (parts[20..21])

- think_seen is `Q+0x138`, incremented once per committed token while a think budget is active and the close token has not been seen: `01851580.c:56-57` and `01851210.c:53-54` (`puVar3 + 0x27`). Budget is `Q+0x114` (THINK field), the close token id is `Q+0x118`.
- `Q+0x13c` is "think closed": set to 1 when the committed token equals `Q+0x118` (`01851580.c:53`, `01851210.c:50`).
- When `think_seen >= budget` with no close, the path writes `*(u16*)(Q+0x13c) = 0x101` (sets `+0x13c` and `+0x13d`, `01851580.c:59`, `01851210.c:56`) and copies the close-token list into the forced-output queue (`FUN_018704e0(Q+0x140, Q+0x120)`). Printed think_forced = 1.
- The guard path (16 kept stop tokens inside a think block) writes the same `0x101` plus `*(u8*)(Q+0x13e) = 1` (`01849530.c:49-50`). Printed think_forced = 2.
- `think_forced = (Q+0x13d != 0) ? Q+0x13e + 1 : 0`: 0 none, 1 forced by budget, 2 forced by the guard.

## Evidence index

- `work/decomp/flash_serve/0184c9a0.c:99` format string; `184cb1c..184cbb5` stack pushes (disassembly, capstone).
- `01845e50.c:149-157` stats object and caller arguments.
- `01849530.c:40-61`, `01851580.c:51-66`, `01851210.c:48-62`, `0182ea80.c:8346-8357`, `:8624-8635`, `01882d20.c:14-30`.
- `01854310.c:111-262` and the skip table `PTR_s_HALOGEN_ACT_STATS_019f60e0`.
- `analysis/env-flags.md` for the 256-row table and the counting method.

## Residuals

- Env table: type and default come from instruction patterns at the first site of each variable. 49 rows say "see site" for the default (sscanf lists, derived values, multi-branch defaults). 101 rows have no prior writeup; their Selects text is name-and-consumer inferred and 15 are marked `[n]` as weak guesses. Nothing was run, so no variable's effect is observed.
- Env table: a name read through a non-literal pointer would not show in the `lea` scan. All 418 sites resolved to literals; code outside the scanned executable segment (none found) was not scanned.
- D line: the stores that set `Q+0x1a8..0x1b8` (GUARD) and `Q+0x114/0x118` (THINK) were not traced (parse locals at `0182ea80.c:3141-3216`); the roles are read from the consumers.
- D line: zero initialisation of the counters (`Q+0x138`, `+0x1bc`, `+0x1c0`, `R+0x10`) was not located; `FUN_01847c40` only moves records. Zero at admission is inferred from the 0 values in the evidence lines.
- `shared` meaning is evidence-based (n_gen = commit + shared in every checked line) but the exact tick path that takes the shared sample is not decoded beyond `FUN_018514a0`.
- `pool_used` summing semantics (12-byte vector entries) is read from the formatter only.
