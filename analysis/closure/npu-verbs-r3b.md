# NPU serve verbs round 3b: LOOKUP, QHN, QLED, EDUMP, HN/LEDGER, GEN hashes, IMG side

Scope: `binaries/halogen-npu`, static only. Decompiles are `work/decomp/halogen-npu/<addr>.c` cited `<addr>.c:line` (Ghidra address = file + 0x100000), read from `/Users/peterb/Documents/AI/re-halogen` because the worktree has no `work/`. "proof" means shown by quoted code or bytes; "inferred" means reasoned, not shown. Continues `analysis/closure/npu-verbs-r3.md` (r3) section 8 and `analysis/closure/npu-closure-r2.md` (r2). Tokens are counted from 0: token 0 = verb, token 1 = tag. Token n sits at byte `n*0x20` of the token vector (`00172600.c:277,280` `puVar25[uVar22 * 4]`, 8-byte elements).

## Verdict

CLOSED for (a): argument lists, ranges, reply fields and the HN/LEDGER versus QHN/QLED mapping. CLOSED for (b): the three GEN reply values. CLOSED for (c) as to what `side` does inside `FUN_001df900`; the set of legal `side` values is data in a device manifest that is not in the repo (section 6). Two corrections to r2/r3 are in section 1. Open leftovers, none in the three asked items, are in section 7.

## 1. Corrections

1. r2 and r3 treat `HN` (0xefd83) and `QHN` (0xefd82) as different strings with an untraced mapping. They are the same bytes: file 0xefd82 is `QHN\0QLED\0LOOKUP\0EDUMP\0ELOAD\0EFREE\0` (python read), so `&DAT_001efd83`, the "HN" compare in `0012b110.c`, is the tail of `QHN`. The verbs are not aliases: `HN` and `LEDGER` are answered inline by the front end `FUN_0012b110` from the per-request result struct, `QHN` and `QLED` by the cache handler `FUN_00172600` from a process-wide singleton (sections 2 and 4).
2. r3 section 7 gave the IMG reply as `IMG <side> ...`. The code builds `IMG <tag> <side> <base64>` (section 6). r3 section 8 item 5 said `side` may map to a larger key; it must equal a key exactly (section 6).

## 2. Dispatch and state (proof unless marked)

The handler `FUN_00172600` is registered under six names. The string-pointer table at vaddr 0x112340 (Ghidra `0x212340`, used by `PTR_DAT_00212340` in `00119560.c`) decoded with python: entries 0..5 are file addresses 0xefd82 `QHN`, 0xefd86 `QLED`, 0xefd8b `LOOKUP`, 0xefd92 `EDUMP`, 0xefd98 `ELOAD`, 0xefdac `EFREE`. The registration loop is `00119560.c:36-75` (inserts each name into the registry map from `FUN_0013e2e0`), and `00119560.c:125` sets `*(code **)(lVar8 + 0x40) = FUN_00172600`. The sibling verbs `STORE` and `HIT` register to `FUN_00177e00` and `FUN_00175f10` (`00119560.c:30-35`: `FUN_0016a880(&local_58,"STORE")`, `"HIT"`).

Handler preamble (`00172600.c:88-101`): `lVar8 = FUN_0016b6d0()` is the singleton (static at Ghidra `0x214020`, `__cxa_guard` init, `0016b6d0.c:25-47`). The tag is token 1, or the literal `?` when the line has 2 tokens or fewer: `if ((ulong)(param_2[1] - lVar18) < 0x21) FUN_0016a880(&local_208,&DAT_001efbdb);` (bytes at 0xefbdb are `?\0`). The reply prefix `OK <tag>` is `FUN_0016d9c0(.., &DAT_001ef0c9, tag)` with `DAT_001ef0c9` = `OK ` and `DAT_001f0c31` = one space (python read: 0xef0c9 `OK \0`, 0xf0c31 ` \0`). Error return: 0 with the message in `param_3` (`LAB_001728ac`); success returns 1 (`LAB_00172d28`, `uVar4 = 1`). Malformed forms fall to `usage: <verb>` (`00172600.c:197` `FUN_0016d9c0(&local_108,"usage: ",*puVar25,puVar25[1])`).

Singleton fields used (offsets from the singleton base):

| Offset | Type | Content | Writer |
|---|---|---|---|
| +0x08 | std::map<int, entry*> (header +0x08, root +0x10, leftmost +0x18, size +0x28) | cache entries keyed by handle id | ELOAD (`00172600.c:136-150`, `FUN_00172370`); EFREE erases (`:207-258`) |
| +0x38 | prefix cache (hash table plus LRU list); capacity from env `HALOGEN_NPU_CACHE_ENTRIES`, default 8 | token-prefix index | `0016b6d0.c:60-63` `getenv("HALOGEN_NPU_CACHE_ENTRIES")`, `uVar5 = 8` |
| +0x90 | vector<u32> | float32 words of the last STORE/HIT hidden-state buffer | `00177e00.c:400`, `00175f10.c:602` `FUN_0016a420(lVar + 0x90,&uStack_1e8)` |
| +0xa8 | int | one integer argument of the last STORE or HIT | `00177e00.c:401` `= iVar5`; `00175f10.c:603` `= iVar11` |
| +0xb0 | std::string | ledger text of the last STORE/HIT | `00177e00.c:402-403` `FUN_00143660(&local_260,&local_1a0); FUN_0016a340(lVar16 + 0xb0,&local_260)`; `00175f10.c:610-615` same plus `" store "` and `" load "` suffix strings |

Why +0x90 is the hidden-state vector (offset reasoning): `&uStack_1e8` is word 3 of the stats struct `local_200` passed as `param_11` to `FUN_001734f0` (`00177e00.c:394`, `00175f10.c:600`). Inside `FUN_001734f0`, word 3 is `param_11[3]`, sized n*H by `FUN_00152800(param_11 + 3,local_260 * uVar14,local_90)` (`001734f0.c:1029`) and read as hidden rows `FUN_00161260(local_178 + param_11[3],...)` (`:1150`). `FUN_0016a420` copies 4-byte elements (`undefined4 *`, `0016a420.c:3-12`). The mapping of `uStack_1e8` to word 3 is by the Ghidra stack layout (inferred). Initial values are zero (`0016b6d0.c:38-54`), so before any STORE/HIT QHN returns `0` plus an empty list and QLED an empty string (inferred from the zero init).

## 3. Grammar table (a)

Counts include the verb and the tag. `bytes` in the code is `count*0x20`.

| Verb | Tokens | Positions and types | Ranges and checks | Reply | Cites |
|---|---|---|---|---|---|
| QHN | any, none read | none | none, extra tokens ignored | `OK <tag> ` + decimal of `+0xa8` + for each u32 of `+0x90`: decimal and one space (see note) | `00172600.c:103,382-440`; `:384` `FUN_0013a710(&local_108,*(undefined4 *)(lVar8 + 0xa8))`; loop `:394` |
| QLED | any, none read | none | none | `OK <tag> <ledger text>` | `00172600.c:106,366-377`; `:374-375` `appendEPKcm(&local_128,*(undefined8 *)(lVar8 + 0xb0),*(undefined8 *)(lVar8 + 0xb8))` |
| LOOKUP | 3 or more | token 2 never read; tokens 3.. are ids, `strtoll(tok,0,10)` stored as 32-bit int, no full-token check | 2 tokens or fewer gives `usage: LOOKUP` (`:191` size `< 0x41`, falls through the EDUMP/ELOAD/EFREE tests); exactly 3 tokens gives an empty list (`:269` size `< 0x61`); no upper bound on count; fewer than 16 ids never match (`0016b1f0.c:20` `if (0xf < param_3)`) | `OK <tag> <entry_id> <matched_tokens>` | `00172600.c:108,191-196,269-352` |
| EDUMP | exactly 4 | token 2: entry id, `strtol(..,0,10)` to int; token 3: output path string | other counts give `usage: EDUMP` (`:164` `!= 0x80` goes to the EFREE test, then usage); unknown id gives `no such entry` (`:185`); file errors from `FUN_0016eb50` | `OK <tag>` | `00172600.c:162-185` |

LOOKUP behaviour (mechanism proof, product reading inferred):
- The ids go to `FUN_0016b4b0(&local_218, singleton+0x38, ids, count)` (`00172600.c:326`). `FUN_0016b1f0` (`0016b1f0.c:20-60`) builds one chained 64-bit hash per 16-token block: seed `0x6a09e667f3bcc909`, each block folds 16 ids with `(*param_2 ^ uVar6) * 0x100000001b3 ^ param_2[1] ...`, then a splitmix-style finaliser `(uVar6 >> 0x1e ^ uVar6) * -0x40a7b892e31b1a47` and two more rounds. The unit is 16 tokens, the same block size as the GEN cache size and ELOAD `cap` (r3 sections 5 and 7).
- `FUN_0016b4b0` tries the last block hash first and walks down (`lVar11 = lVar11 + -1`, label `LAB_0016b577`). A hash hit is accepted only if the stored token vector is not longer than the query and `memcmp(stored, query, stored_len) == 0` (`0016b4b0.c`, label `LAB_0016b63a`). A hit moves the entry to the LRU front (`_List_node_base::_M_transfer`). Result: the longest stored entry whose tokens are a prefix of the query, found at block granularity. It returns a shared_ptr (object pointer, control block) in `local_218`/`local_210`; `local_210` is released at `00172600.c:353-355`.
- Entry id: the code walks the whole map from its leftmost node (`:328-336` `lVar18 = *(long *)(lVar8 + 0x18)`) and keeps the key of any node whose value pointer equals the matched object (`*(long **)(lVar18 + 0x28) == local_218`). `uVar4` starts at `0xffffffff` (`:327`) and goes through the signed `FUN_0013a710` (negative path emits `-`, `0013a710.c:48-55,116`), so no match prints `-1`. A matched entry with no map node also prints `-1`; whether STORE/HIT insert into the prefix cache without a map node was not traced (inferred).
- `matched_tokens` is `(local_218[1] - local_218[0]) >> 2`, 0 for no match (`:338-341`). The empty and the no-match reply is therefore `OK <tag> -1 0`.
- Number order: id first (`FUN_0013a710(&local_188,uVar4)`), length second (`FUN_0013a710(&local_128,lVar18)`), built at `:343-352`.
- Token 2: the loop starts at `uVar22 = 3` and nothing else in the LOOKUP branch reads token 2. By analogy with `PFX tag n ids` it is probably a count this handler does not check (inferred).

QHN note. The string is `OK <tag> ` plus the decimal of `+0xa8` via `FUN_0013b890(&local_168,&local_128,&local_108)` (`:384-391`). The loop appends, for each u32 at `+0x90`, its unsigned decimal followed by one space through `FUN_0016d760(&local_128,&DAT_001f0c31)` (`:395-480`, append at `:480-520`). The decompile shows no separator between the `+0xa8` number and the first list value, so read literally the reply is `OK <tag> <N><v0> <v1> ... ` with a trailing space. I did not check the disassembly: treat the missing separator as inferred and confirm on a live reply. List values are float32 bit patterns in unsigned decimal, same convention as the DEC scores (r3 section 4). `+0xa8` is the int STORE reads as its second numeric argument (`00177e00.c:154` `iVar5 = FUN_0016a240(&local_420)`, the second such call) and HIT as its fourth (`00175f10.c:200`). What those arguments mean belongs to the STORE and HIT grammars, which this round did not close.

EDUMP behaviour: the entry pointer (`node+0x28`) goes to `FUN_0016eb50(entry, &token3, &err)` (`00172600.c:179`). That function builds an HNPB container for the entry and writes it with `fopen(path,"wb")` (`0016eb50.c:1156`): a 0x40-byte header (`:1176` `fwrite(local_278,1,0x40,__s)`), a 0xfc0-byte table (`:1178`), then data. It also formats a manifest text `hnpm 1 entry\nentry P=... capb=... nb=... lin=...` (`0016eb50.c:116-125`). Errors: `cannot write <path>` (`:1159`), `short write to <path>` (`:1210`), `a blob name of 1-79 characters: <name>` (`:1123`). So EDUMP saves a resident cache entry as a file that ELOAD can read back; the round trip is inferred, not shown.

## 4. HN and LEDGER versus QHN and QLED

Front-end verbs in `FUN_0012b110`, reached after the DEC/EMB/SCO/CLS/PFX tests fail (`0012b110.c:215-222`) and before the registry lookup (`:270-300`):
- `HN` (`0012b110.c:231-253`): prefix `HN ` (bytes at 0xef465), tag, a space, then `(long)(param_4[0xc] - param_4[0xb]) >> 2` (`:245`, `_M_insertImEE`), then for each word from `param_4[5]` to `param_4[6]` a space and the unsigned decimal (`:246-252`). Result offset +0x28 is the n*H hidden-state float vector and +0x58 the n-element int list that `FUN_0014a6f0` fills (`0014a6f0.c:932-935`; `0014a6f0.c:960-961` reads the same +0x28 vector for EMB). So `HN <tag> <n> <n*H hidden floats as float32 bit patterns>`. This is proof for the vector identities and the print loop.
- `LEDGER` (`0012b110.c:256-269`): `LEDGER <tag> ` plus `FUN_00143660(local_1c0, param_4 + 0x11)`, the same formatter QLED echoes (prefix literal `LEDGER ` at 0xef469).
- `param_4` is the result struct that `FUN_0014a6f0` receives for DEC/SCO/CLS/EMB (`0012b110.c:1119`). Whether it lives per connection or per process was not traced (inferred: it carries the last head pass).

`QHN` and `QLED` read instead the singleton fields written by `STORE` and `HIT` (section 2). So each pair is the same kind of data (hidden floats, ledger text) from a different producer: `HN`/`LEDGER` for the head-pass verbs, `QHN`/`QLED` for STORE/HIT. The INFO verb is a third front-end verb (`0012b110.c:270`) and was not part of this question.

Ledger text, `FUN_00143660` (`00143660.c:36-46`), format quoted: `wall %.3f device %.3f off %.3f flush %.3f pull %.3f embed %.3f copy %.3f fused %.3f resid %.3f const %.3f norm %.3f head %.3f calls %d syncs_to %lld %.2fMB syncs_from %lld %.2fMB`, then ` fmax %.3f nfused %d minflt %ld majflt %ld nvcsw %ld nivcsw %ld cpus %llx sync_to_ms %.3f sync_from_ms %.3f cpu_user %.2f cpu_sys %.2f`, then ` kernels` and ` <name>:%.3f` pairs (`:48-55`, `snprintf(local_2b8,0x280," %s:%.3f",pdVar3[8],pdVar3[4])`). Units are those of the stats struct doubles; not asserted here.

## 5. GEN HASH_A, HASH_B, MISMATCH (b)

The reply tail is `" %016llx,%016llx,%d"` (`0012a4a0.c:241` `snprintf(local_120,0x40," %016llx,%016llx,%d",local_70._0_8_,local_70._8_8_,(ulong)local_60)`; r2 mapped these to `param_5+0x1c`, `+0x1e`, `+0x20`).

Hash function throughout: 64-bit FNV-1a over bytes, basis `0x14650fb0739d0383`, prime `0x100000001b3`: `uVar107 = (uVar107 ^ *pbVar49) * 0x100000001b3` (`001ba230.c:640-662`, `:2194-2201`).

- HASH_A (`param_5+0x1c`, `001ba230.c:640-664`): FNV-1a over every byte of three lists of byte buffers filled by the batch-1 prefill `FUN_001b43c0` into its out-struct `local_bb0` (call at `001ba230.c:627`; hashed at `:640-662`, walking `local_cf0..local_ce8`, `local_ce8..local_b90`, `local_db0..plVar34`, elements are begin/end/cap triples). The producers in `001b43c0.c` are device read-backs taken per layer after the prefill:
  - list 1 (`param_5[0..2]`): a 0x100000-byte (1 MiB) copy per layer of device buffer `lVar20`, after the error text `"the state read-back at layer "` (`001b43c0.c:586-602`, copy `memcpy(.., *(void **)(lVar20 + 0x38), 0x100000)`, `:622`).
  - list 2 (`param_5[3..5]`): a 0x12000-byte buffer per layer assembled from device buffer `lVar19` in 0x1100-byte strides (`_Znwm(0x12000)` at `001b43c0.c:624`, push at `:897-906`).
  - list 3 (`param_5[6..8]`): per-layer K and V copies after the error text `"the K / V read-back at layer "`, two slices of `sVar23` bytes from buffer `lVar26` (`001b43c0.c:1039-1081`).
  So HASH_A is a fingerprint of the device-resident cache state (recurrent state plus K/V) as read back right after prefill, once per request. With no prefill call (`local_e38 < 1`, `001ba230.c:621-628` clears all lists) the three loops add nothing and the value is the bare basis `14650fb0739d0383` (inferred from the empty ranges).
- HASH_B (`param_5+0x1e`, `001ba230.c:2192-2202`): FNV-1a over the `sVar85` bytes of the logits read-back of the first decode step only. Guard `if (local_d98 == piVar70)`; these are end and begin of the generated-token vector (`:1454-1455` both zero; `:2437-2439` `*local_d98 = ..; local_d98 = local_d98 + 1`; `:2443-2460` `piVar70` set once), so equal means no token appended yet. The bytes are `memcpy(local_d90, *(long *)(lVar73 + 0x38) + *(long *)(local_d48 + 0x80) * 4, sVar85)` after `FUN_00156cb0` syncs the buffer (`:2150-2162`), the logits for the first sampled token. If generation ends before a first decode step the field keeps its initial value (initialiser not located; unverified).
- MISMATCH (`param_5+0x20`, int, `001ba230.c:2166-2190`): a count of unstable logits read-backs, active only when bit 0 of env `HALOGEN_NPU_GEN_DBG` is set. The variable is parsed once into `_DAT_00214148` (`001ba230.c:257-262`, name at file 0xf0c0b, python read `GEN_DBG\0`). With bit 0 (`if ((_DAT_00214148 & 1) != 0)`, `:2166`) the loop sleeps with `nanosleep` on a timespec at file 0xf4b40, which is `{0 s, 2000000 ns}` = 2 ms (python `struct.unpack('<qq')` returned `(0, 2000000)`), re-syncs the logits buffer, copies it again into `local_dd8`, and runs `memcmp(local_dd8,local_d90,sVar85)`; a difference does `param_5[0x20] = param_5[0x20] + 1` (`:2188-2190`). So MISMATCH counts decode steps whose logits changed between two reads 2 ms apart. Without the env variable it is never incremented (initial value not located; expected 0, unverified). Other bits of the same variable gate timing sections (`:2123,2136,2146`), and bit 3 (value 8) disables the 4 KiB rounding of the read-back ranges (`:265` `DAT_00214190 = (byte)_DAT_00214148 >> 3 & 1`). Calling this a DMA coherence check is inferred; the code proves only the double read and the compare.

## 6. IMG `side` inside FUN_001df900 (c)

Proof from `001df900.c`:
- Exact match required. The map at `engine+0x1200` is searched lower_bound style, then the found key is tested with `<=` (`:130` `while (lVar51 = lVar36, param_3 <= *(int *)(lVar51 + 0x20))` and `:138` `if ((param_1 + 0x1200 != lVar46) && (*(int *)(lVar46 + 0x20) <= param_3))`). Found key `>= side` (lower_bound) and `<= side` means `== side`. A miss ends at `:746-747` with `no device set at <side>` (`FUN_0013a710(&local_f0,param_3); FUN_001c9050(&local_f0,"no device set at ")`). There is no rounding up to a larger key. The match supplies the device set for that resolution (`lVar46 = *(long *)(lVar46 + 0x28)`, `:139`).
- Prompt cap: `if (0x200 < (int)(param_2[1] - *param_2 >> 2))` fails with `the prompt is <n> tokens (at most 512)` (`:148-156`). This is the N <= 512 limit of r3 seen from the engine side.
- Latent grid: `iVar33 = side >> 4` (`:382-387`, with the bias for negatives), `lVar66 = g*g`, and a float buffer of `g*g*128` words (`lVar47 = lVar66 * 0x80`, `lVar48 = lVar66 * 0x200` bytes, `:388-410`). So the latent has g = side/16 positions per axis and 128 channels per position (the channel reading is inferred from the 0x80 constant). The sampling loop is guarded by `0xf < param_3` (`:410`).
- Output buffer is `side*side*3` bytes (`lVar47 = (long)param_3 * (long)param_3`, `:612`; times 3 in r3).
- Stage names in the code: `"te ctx alloc"` (`:172`), `"noise"` (`:249`), `"dit euler"` (`:347`), `"decode"` (`:671`). `seed` feeds `FUN_001e1f80(param_4, param_3, buf)` (`:240`), which fills the g-by-g latent noise (r3).

Reply (`0012b110.c:1396-1450`): success is `IMG <tag> <side> <base64>`. The prefix is `FUN_001256c0(local_2e0,&DAT_001ef3f6,tag)` (`IMG ` plus tag), then a space, then `FUN_0013a9b0(&local_2a0,uVar6)` with `uVar6` the 64-bit `side`, then a space, then standard base64 (alphabet literal at `:1430`, `=` padding at `:1447`) of the `side*side*3` RGB8 bytes `local_500`. Failure is `<DAT_001ef39b> <tag> <message>` through `FUN_00125080` (`:1398-1405`).

Data, not code: the legal `side` values. The map keys come from the flux2 device set (`devices.hnpm` of `flux2-klein-4b`, listed at `image/npu-models.txt:75-80`), which is not in the repo; `analysis/evidence/npu-decider-0.8b/devices.hnpm` is the decider set. The insertion site of the map was not found by grepping `+ 0x1200` writes; `0012b110.c:1366-1378` and `001df900.c:122-138` are the only readers found.

## 7. What is left (outside the three asked items)

1. STORE and HIT argument grammars; they define the QHN integer and what the list covers.
2. The QHN separator question: confirm in the disassembly of the append loop (`00172600.c:480-520`) or against a live reply.
3. Initialisers of `param_5+0x1e` (HASH_B) and `+0x20` (MISMATCH), assumed 0.
4. The legal IMG `side` keys (needs the flux2 `devices.hnpm`).

## Reproduce

```
cd /Users/peterb/Documents/AI/re-halogen/work/decomp/halogen-npu
sed -n 88,200p 00172600.c
sed -n 258,360p 00172600.c
sed -n 360,440p 00172600.c
sed -n 1,70p 0016b1f0.c
sed -n 20,70p 0016b4b0.c
sed -n 25,65p 0016b6d0.c
sed -n 28,40p 00119560.c
sed -n 394,406p 00177e00.c
sed -n 598,616p 00175f10.c
sed -n 1026,1030p 001734f0.c
sed -n 36,56p 00143660.c
sed -n 1150,1215p 0016eb50.c
sed -n 215,270p 0012b110.c
sed -n 1396,1450p 0012b110.c
sed -n 255,268p 001ba230.c
sed -n 625,668p 001ba230.c
sed -n 2148,2205p 001ba230.c
sed -n 586,624p 001b43c0.c
sed -n 1036,1082p 001b43c0.c
sed -n 118,160p 001df900.c
sed -n 378,412p 001df900.c
sed -n 736,756p 001df900.c
python3 - <<'E'
import struct
d=open('/Users/peterb/Documents/AI/re-halogen/binaries/halogen-npu','rb').read()
po=struct.unpack_from('<Q',d,0x20)[0]; n=struct.unpack_from('<H',d,0x38)[0]
segs=[struct.unpack_from('<IIQQQQQQ',d,po+i*56) for i in range(n)]
fo=lambda v:[v-s[3]+s[2] for s in segs if s[0]==1 and s[3]<=v<s[3]+s[5]][0]
b=fo(0x112340)
for k in range(6):
    p=struct.unpack_from('<Q',d,b+k*8)[0]; print(k,hex(p),d[p:p+8])
print(d[0xefd82:0xefda0], d[0xef0c9:0xef0cc], d[0xf0c31:0xf0c33], d[0xefbdb:0xefbdd], d[0xef465:0xef469], d[0xf0c0b:0xf0c13], struct.unpack('<qq',d[0xf4b40:0xf4b50]))
E
```
