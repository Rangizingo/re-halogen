# NPU serve verbs round 4: STORE, HIT, and the IMG side key set

Scope: `binaries/halogen-npu`, static only (nothing was run). Decompiles are `work/decomp/halogen-npu/<addr>.c`, cited `<addr>.c:line`; Ghidra address = file + 0x100000. String addresses below are file offsets read with python from the binary (Ghidra = +0x100000). "proof" means shown by quoted code or bytes; "inferred" means reasoned, not shown. Continues `analysis/closure/npu-verbs-r3.md` and `npu-verbs-r3b.md` (r3b). Tokens are counted from 0: token 0 = verb, token 1 = tag. Sources were read from `/Users/peterb/Documents/AI/re-halogen` (the worktree has no `work/`).

## Verdict: PARTIAL

CLOSED for the STORE and HIT token grammars, parse functions, range checks, error strings and reply layouts (sections 1 to 4). CLOSED for where the IMG `side` key comes from: it is the `res=` field of the `elf` records in the model's `devices/devices.hnpm`, so the legal set is manifest data (section 5). OPEN-BOUNDED only for the concrete flux2-klein-4b set, because that `devices.hnpm` (8392 bytes, `image/npu-models.txt:75`) is not in the repo. Semantics of three numbers (cap, the HIT flags, the HIT store/load times) are inferred as marked.

## 1. Shared conventions (proof)

- Both handlers are registered by `00119560.c:30-35` (`FUN_0016a880(&local_58,"STORE")`, `"HIT"`), as `FUN_00177e00` (STORE) and `FUN_00175f10` (HIT). They are called from the front end `FUN_0012b110` as `(**(code **)(lVar47 + 0x58))(lVar47 + 0x40, model, tokens, errstr)` after the model lookup (`0012b110.c:~343-347`). Return 0 means error.
- Error reply on 0: `ERR <tag> <message>`. `DAT_001ef39b` = `ERR ` (file 0xef39b), then `FUN_001256c0(.., "ERR ", tag)`, `append(" ")`, `FUN_00125080(param_1, .., errstr)` (`0012b110.c:342-350`). Success: the handler's own string is the reply (`0012b110.c:351-352`, `EC1EOS4_(param_1,&local_2a0)`).
- Tag: token 1, or `?` (file 0xefbdb) when the line has 2 tokens or fewer (`00177e00.c:132-133`, `00175f10.c:175-176`, test `size < 0x21`).
- Token 2 is a string, the bucket name: if the line has more than 2 tokens (`size >= 0x41`) it is copied and the numeric cursor starts at 3 (`00177e00.c:142-151`, `local_418 = 3`; `00175f10.c:178-181`). With only verb and tag, the name is `""` and the cursor stays at 2.
- Every number is read by `FUN_0016a240` (`0016a240.c:6-16`): `__isoc23_strtoll(token, 0, 10)`, base 10, end pointer NULL, so no trailing-garbage check; a non-numeric token reads as 0; the result is narrowed to a 32-bit int by the callers (`iVar5`, `iVar6`, `uVar4` in `00177e00.c:153-155`). Past the last token it returns 0 and clears the "ok" byte (`0016a240.c:11-13` `*(undefined1 *)(param_1 + 2) = 0; return 0;`). Extra trailing tokens are ignored.
- Bucket name lookup is `FUN_0016eab0` (`0016eab0.c`): `FUN_0013e380` searches the model map at `model+0x280` by name; miss gives `no bucket <name>` (`0016eab0.c:9-12`, string `no bucket ` file 0xef996). Then `FUN_00145b20` (a second `no bucket`, plus `no NPU context free: the <n> loaded buckets are all in use by this request`, `00145b20.c:34,91-94`, file 0xef84a) and `FUN_0016e270` fill an info struct: byte 0 = kind flag (1 for resume), +4 = bucket B (`*(undefined4*)(param_3+4) = *(param_2+100)`), +8 = bucket S (`0016e270.c:~36-37`).
- Both then call the shared engine `FUN_001734f0(model, bucket, info, ids, entries, store_flag, cap, flag8, slots, heads, stats, err)` (`00177e00.c:393`, `00175f10.c:599`); it returns 0 on success, nonzero with the message in `err`.
- All times in replies are `(long)(1000.0 * ms)`, i.e. microseconds (`ram0x001f4bd0` = 1000.0, r3 section 1; `FUN_0013a9b0` prints the integer).

## 2. STORE grammar (proof unless marked)

`STORE <tag> <bucket> <cap> <B> <S> <id_1> ... <id_{B*S}>`

| Token | Field | Parse | Check | Cite |
|---|---|---|---|---|
| 1 | tag | string | none | `00177e00.c:132-139` |
| 2 | bucket | string | must name a loaded bucket else `no bucket <name>` | `00177e00.c:142-151`, `0016eab0.c:9-12` |
| 3 | cap (`uVar4`) | strtoll base 10, int | passed as `param_7` to `FUN_001734f0`; 0 selects the existing cache's default (`iVar11 = param_7; if (param_7 == 0) iVar11 = *piVar48`, `001734f0.c:408-411`); meaning "arena capacity in 16-token blocks" is inferred from `*piVar48 < iVar39 >> 4` giving `the arena's capacity is below the new entry` (`001734f0.c:435-436`, file 0xf2030) | `00177e00.c:153` |
| 4 | B (`iVar5`) | strtoll, int | must equal the bucket's B | `00177e00.c:154,184` |
| 5 | S (`iVar6`) | strtoll, int | must equal the bucket's S | `00177e00.c:155,184` |
| 6.. | B*S token ids | strtoll, 32-bit each, row-major | count is read exactly: the loop runs while `count < B*S` (`lVar16 = (long)iVar5 * (long)iVar6`, `:166`; loop `:171-246`). Fewer tokens than needed gives `a malformed STORE`. If B*S <= 0 no ids are read. | `00177e00.c:166-246` |

Checks, in order of reaching them:
1. Any of the three numbers or an id missing: `a malformed STORE` (`00177e00.c:257`, file 0xefe6c), reply `ERR <tag> a malformed STORE`. The "ok" byte test is `cVar2 = (char)local_410` (`:156`); the id loop exits to `:249-257` when tokens run out.
2. Bucket lookup errors from `FUN_0016eab0` (listed in section 4).
3. Mismatch: `if ((lVar16 != 0) && (((char)local_240 != '\0' || (local_240._4_4_ != iVar5)) || ((int)uStack_238 != iVar6)))` then `a state store runs its own bucket kind at its own B and S` (`00177e00.c:182-187`, file 0xf2198). So STORE needs kind flag 0 (not a resume bucket), `B == bucket B`, `S == bucket S`. Consequence: the id count is fixed per bucket at B*S.
4. In the engine: `a malformed request` if `B*S != ids.size` or `B != slots.size` (`001734f0.c:112-116`, file 0xefa3f); `the store row is not a block boundary` if `S & 0xf != 0` (`:121-124`, file 0xf1fe0); `no embed blob` (`:199-200`, file 0xefdba); `a token id outside the vocabulary` for any id `< 0` or `>= vocab` (`:208-210`, file 0xf1be0); capacity error above; `the arena append at layer <n>` (`001734f0.c:723-725`, file 0xefe18); `the constants are off the 64-byte granule` and `the entry region is off the 64-byte granule` (`:751,768`, files 0xf2118, 0xf20e8).

STORE passes to the engine: `store_flag = 1`, `flag8 = 0`, the slot vector filled with `S - 1` for each of the B rows (`00177e00.c:326`, `iVar6 = iVar6 + -1`, then the unrolled fill `:330-397`; engine check `slot < S` at `001734f0.c:1036-1038` `a slot outside its sequence`, file 0xefa8f), and an empty head-row vector (`&local_260` zeroed at `:305`). So STORE hashes and stores the last position row of each of B sequences (the slot meaning is inferred from the `S-1` fill and the `slot < S` check).

Success reply (`00177e00.c:405,424-578`):

`OK <tag>` then for each stored cache entry ` <id>` (the id from `FUN_00172370(entry)`, the handle that LOOKUP/HIT/EDUMP/EFREE take), then ` <t1> <t2> <t3>`:
- `OK <tag>` = `FUN_0016d9c0(.., &DAT_001ef0c9 "OK ", tag)` (`:405`).
- Each id: `FUN_0013a710` decimal, with a leading space inserted (`:424-428` and the `0x20` store at the `_Znwm(0x1f)` branch).
- Join: `FUN_0016dd50(local_360, reply, len, " ")` (`:581`), then three `FUN_0013b890` concatenations separated by `append(" ")` (`:582-590`). Value order: `local_1a0` (`:580`), `dStack_198` (`:579`), `local_68` (`:578`).
- t1 = total engine wall time: `param_11[0xc] = (long)(dVar55 - dVar54)` (`001734f0.c:1208`), where `dVar54` is taken at entry (`:107`). t2 = `dStack_198` = word `[0xd]`, never written in `FUN_001734f0` (grep for `param_11[0xd]` empty) and zero-initialised (`00177e00.c:263-264`), so always 0 (same pattern as r3 item 3). t3 = `param_11[0x33]`, accumulated at `001734f0.c:732,817` after the arena append and the state/rows read-back loops (strings `the arena append at layer`, `the state or rows read-back at layer`); read as the store phase time (inferred from those strings; HIT's ledger text names it, section 3).
- Number of ids: `plStack_1f8 - local_200` over 16-byte shared_ptr slots. One per row is the natural reading (the engine builds one entry vector the size of the batch, `001734f0.c:872-1028`) but the count rule was not shown (inferred).
- Side effect: the QHN/QLED singleton gets `+0x90` hidden vector, `+0xa8` = B (`iVar5`), `+0xb0` ledger text (`00177e00.c:399-403`, r3b section 2).

Example (shape only): `STORE s1 u3 0 1 128 <128 ids>` gives `OK s1 7 41230 0 38100` (invented numbers; format from the code, not a live reply).

## 3. HIT grammar (proof unless marked)

`HIT <tag> <bucket> <f1> <cap> <f2> <N> <h_1..h_N> <slot_1..slot_N> <M> <row_1..row_M> <id_1 .. id_{N*S}>`

| Token group | Field | Parse | Check | Cite |
|---|---|---|---|---|
| 1 | tag | string | none | `00175f10.c:175-181` |
| 2 | bucket | string | `no bucket`, see section 4 | `:388` |
| 3 | f1 | strtoll (`lVar15`) | only `!= 0` is used: passed as `store_flag` (`lVar15 != 0`, `:599`). When true the engine requires S aligned to 16 and, for a resume bucket, `S >= arena_slot` (`001734f0.c:121,131`) | `:197,599` |
| 4 | cap | strtoll, int (`uVar10`) | passed as `param_7` as in STORE (0 = keep default) | `:198,599` |
| 5 | f2 | strtoll (`lVar16`) | only `!= 0` is used: passed as `flag8` (`:599`); with flag8 set the engine rewrites the arena block count instead of rebuilding (`001734f0.c:406-431`) (meaning inferred) | `:199` |
| 6 | N | strtoll, int (`iVar11`) | the loops run only if `N > 0` (`bVar9 = 0 < iVar11 & ok`, `:205`); N must equal the bucket B (below) | `:200` |
| next N | entry handles h_i | strtoll base 10, int | each looked up in the singleton map `lVar17 + 8` (std::map<int, entry*> of QHN section, r3b section 2); miss gives `no such entry` (`:206-230`, the `lea rsi,[str.no_such_entry]` at file 0x765a7, string at 0xefd9e) but parsing continues | `:206-260` |
| next N | slot_i | strtoll | engine check `0 <= slot < S` else `a slot outside its sequence` (`001734f0.c:1036-1038`) | `:262-361` (second loop, `iVar11 <= iVar26` break) |
| next 1 | M | strtoll (`iVar26`) | count of head rows | `:362` |
| next M | head rows | strtoll, int | each `0 <= row < vocab` else `a head row outside the vocabulary` (`001734f0.c:1080-1082`, file 0xf1cf8) | `:363-445` |
| last N*S | token ids | strtoll, 32-bit | exactly `N * S` ids where S is the bucket's S (`lVar35 = (long)(int)uStack_308`, `if (lVar35 * iVar11 <= lVar39) goto done`, `:473`) | `:466-531` |

Error checks:
1. Any list short of tokens: `a malformed HIT` (`00175f10.c:464,531`, file 0xefe4a); reply `ERR <tag> a malformed HIT`.
2. After the lists: `if ((local_4a8 == 0) && (((char)local_310 == '\0' || (local_310._4_4_ != iVar11)))) FUN_0016d620(.., "a hit runs a resume bucket at its own B")` (`:533-535`, file 0xf2170). So HIT needs a resume-kind bucket (flag 1) and `N == bucket B`.
3. `FUN_0016eab0` runs only if no earlier error: `(local_4a8 != 0) || (local_658 = FUN_0016eab0(...), local_658 == 0)` (`:387-389`). Its `FUN_0016e270` branch for resume buckets (`0016e270.c:71-249`) also demands: a `resume` record (`: no resume record`, file 0xefc7e), no `base0`/`xr` form (`: a resume bucket this binary does not run (base0 or xr)`, file 0xf1e98), an arena slot and `B == 1` (`: the cache's hit runs B = 1 buckets with a per-call arena slot`, `0016e270.c:~214-219`, file 0xf1ed8), and a state size of exactly 0x100000 bytes (0x200000 with `store1`) with `S & 0xf == 0` (`0016e270.c:217-230`; error `<bucket>: the state's bytes (<n>) or S do not match this binary`, files 0xefcd5, 0xf1f40). From the B == 1 rule, N must be 1 for any bucket HIT accepts (inference from combining check 2 with `0016e270.c` B == 1; both quoted).
4. Engine errors as in STORE (`a malformed request` if `ids.size != B*S`, `B != slots.size`, or, for resume, `B != entries.size`, `001734f0.c:112-116,131-134`).

Success reply (`00175f10.c:623-994`): `OK <tag>` (`FUN_0016d9c0 ... "OK "`, `:623`), then in this order, each item followed by a space:
1. For each int in stats vector `[9]` (`puStack_1b8..puStack_1b0`), signed decimal (`FUN_0013a710`, `:628`). This is the per-row argmax over the M head rows: initialised `0xffffffff` (`001734f0.c:1074`) and updated by strict greater-than (`:1196-1202`), so `-1` when M = 0, else an index into the head-row list (same convention as r3 DEC `argmax_i`).
2. For each row vector in `[6]` (24-byte elements, `puStack_1d0..puStack_1c8`), each u32 as unsigned decimal (`:640-780`): the M head-row scores as float32 bit patterns (r3 section 4 convention).
3. For each entry in `[0..1]` (16-byte slots), the handle `FUN_00172370(entry)` as decimal (`:954`). Which entries appear (hit entries versus newly stored ones) was not traced (inferred: new ones, populated when `f1 != 0`).
4. Four times in microseconds, ` <t1> <t2> <t3> <t4>`, built at `:990-994`: `local_1a0` (total wall, `001734f0.c:1208`), `dStack_198` (word `[0xd]`, always 0), `local_68` (word `[0x33]`), `dStack_60` (word `[0x34]`, written at `001734f0.c:758,775`). The success side effect writes the ledger as `<ledger> store <fmt local_68> load <fmt dStack_60>` (`:600-614`, strings `" store "`, `" load "`, format `DAT_001efe5a`). So t3 = store time, t4 = load time (proof of the pairing from that text; each unit is ms in the ledger, microseconds in the reply).
5. HIT also updates the QHN singleton `+0x90`, `+0xa8 = iVar11 = N`, `+0xb0` (`00175f10.c:602-615`).

Order of items 1 to 3 comes from the three loops in source order (`:628`, `:640`, `:954`); the decompile has no separator tokens beyond the single trailing space. The empty list case (M = 0, no store) gives `OK <tag> -1 <t1> 0 <t3> <t4>` for N = 1 (derived, not seen live).

## 4. Error strings seen by these two verbs (file offsets, Ghidra = +0x100000)

| String | File | Raised at |
|---|---|---|
| `a malformed STORE` | 0xefe6c | `00177e00.c:257` |
| `a malformed HIT` | 0xefe4a | `00175f10.c:464,531` |
| `a state store runs its own bucket kind at its own B and S` | 0xf2198 | `00177e00.c:186` |
| `a hit runs a resume bucket at its own B` | 0xf2170 | `00175f10.c:534` |
| `no such entry` | 0xefd9e | `00175f10.c` (call at file 0x765b5) |
| `no bucket <name>` | 0xef996 | `0016eab0.c:9-12`, `00145b20.c:34` |
| `no NPU context free: the <n> loaded buckets are all in use by this request` | 0xef84a | `00145b20.c:91-94` |
| `<bucket>: no pass record` | 0xefcc0 | `0016e270.c` (g path) |
| `<bucket>: no resume record` | 0xefc7e | `0016e270.c:77` |
| `<bucket>: this bucket kind takes no state store` | 0xf1f18 | `0016e270.c` (neither `r` nor `g` with field +0x6c == 3) |
| `<bucket>: a resume bucket this binary does not run (base0 or xr)` | 0xf1e98 | `0016e270.c:249` |
| `<bucket>: the cache's hit runs B = 1 buckets with a per-call arena slot` | 0xf1ed8 | `0016e270.c` |
| `<bucket>: the state's bytes (<n>) or S do not match this binary` | 0xefcd5 + 0xf1f40 | `0016e270.c:~223-229` |
| `a malformed request`, `the store row is not a block boundary`, `no embed blob`, `a token id outside the vocabulary`, `a slot outside its sequence`, `a head row outside the vocabulary`, `the arena's capacity is below the new entry`, `the arena append at layer <n>`, `the state or rows read-back at layer <n>`, `the constants / entry region are off the 64-byte granule` | 0xefa3f, 0xf1fe0, 0xefdba, 0xf1be0, 0xefa8f, 0xf1cf8, 0xf2030, 0xefe18, 0xf2148, 0xf2118/0xf20e8 | `001734f0.c:116,123,200,210,436,725,751,768,1038,1082` |

Kind strings: the bucket `kind` field is compared with `r` (file 0xef0a3, `FUN_00139a60(param_2+4, &DAT_001ef0a3)`) and `g` (`0016e270.c`, `"g"`). The only shipped manifest, `analysis/evidence/npu-decider-0.8b/devices.hnpm`, has `bucket kind=o` records, which match neither, so STORE and HIT on those buckets reach `this bucket kind takes no state store` (inference by reading the two compares; `kind=o` shown in the manifest line 11).

## 5. IMG `side` keys (c)

Lookup (proof): `IMG` front end converts token 2 and tests `0 < iVar55` (`0012b110.c:1363-1364`), then does a lower-bound search in the std::map at `engine+0x1200` (header; root `+0x1208`, leftmost `+0x1210`, size `+0x1220`): `if ((lVar33 + 0x1200 != lVar47) && (*(int *)(lVar47 + 0x20) <= iVar55))` (`0012b110.c:1366-1383`). Miss gives `ERR <tag> no image size <side>x<side>`, built from ` no image size ` (file 0xef3e6), the side number, `DAT_001effbb` (= `x`, file 0xeffbb), the side again (`0012b110.c:1555-1565`). The engine repeats the exact-match test and, on a miss, `no device set at <side>` (`001df900.c:122-139,746-747`, `001dcbf0.c:20-67`, file 0xf0f5f). Keys are 32-bit signed ints (`*(int *)(node + 0x20)`); a key larger than 2^31 is narrowed by the int compare (inference; not tested).

Insertion (proof): the only inserter is `FUN_001d30b0`, which allocates a 0x220-byte device-set object and does `_Rb_tree_insert_and_rebalance` into `param_1 + 0x1200` with `*(long *)(param_1 + 0x1220) + 1` (`001d30b0.c:1369-1405`, second site `:1550-1584`). It is called once per manifest record from the image loader `FUN_001d6480`:

```
FUN_001cfd50(&local_58,*param_2,param_2[1],"/devices.hnpm");  FUN_00163c50 ... FUN_00164090(...)
for (lVar17 = begin; lVar17 != end; lVar17 += 0x58) {
  if (FUN_00139a60(lVar17,&DAT_001ef4ee)) {            // record kind == "elf"  (file 0xef4ee)
    FUN_001c2520(&local_58,&DAT_001efadb);              // key "res"             (file 0xefadb)
    uVar15 = FUN_00162ba0(lVar17,&local_58,0xffffffffffffffff);
    cVar13 = FUN_001d30b0(param_1,uVar15,param_2,local_98,param_5);
```
(`001d6480.c:~466-478`). `FUN_00162ba0` is the keyed-value accessor of a manifest record (`00162ba0.c:6-30`): full-token `strtoll(value, &end, 0)`, default `0xffffffffffffffff` if missing or not fully numeric.

So: the legal sides are exactly the distinct `res=<N>` values on `elf` records of the loaded model's `devices/devices.hnpm`. `res=0` is the shared text-encoder set (the `param_2 == 0` branch, `001d30b0.c:1346-1398`; excluded from the advertised list by `0 < key`, `001d0e40.c:82`). Empty after loading gives `/devices.hnpm names no DiT ELF` (`001d6480.c:~490-496`, `1d7232`). The advertised set is in `/info`: `"image":{"sizes":[...]}` is the positive keys, comma-joined, from `FUN_001d0e40` (`001d0e40.c:77-92,415`; literal `},"image":{"sizes":[` file 0xf0cd7), with `"steps":4,"max_text":512` after it (`001d0e40.c:~437`). Constraints the set must satisfy, from the code: `side` is rounded as `g = side >> 4` with `g*g` latent positions (`001d30b0.c:134-138` `piVar14[1] = iVar12 * iVar12; piVar14[2] = iVar12*iVar12 + 0x200`; `001df900.c:382-410`), the sampling loop needs `side >= 16` (`0xf < param_3`), and the prompt is at most 512 tokens (r3b section 6). Whether non-multiples of 16 are rejected at load was not found.

Bound: the legal values for a given model are data in `<model>/devices/devices.hnpm`. The shipped flux2-klein-4b file is not in the repo (`image/npu-models.txt:75` lists it, 8392 bytes, sha256 `f65663b3...`), and the decider-0.8b manifest has no `elf`/`res` records. A user with the HF repo can read it with `grep '^elf' devices.hnpm`, or ask the running server for `/info` (`image.sizes`). Both need the artifact or a live run, so this part stays OPEN-BOUNDED.

## 6. Residuals

1. IMG concrete set for flux2-klein-4b: needs `devices.hnpm` from `peonist-ai/halogen-npu-flux2-klein-4b` or one `/info` response.
2. Whether STORE returns one id per row or one per block (entry vector size rule, `001734f0.c:872-1028`), and whether HIT lists newly stored or hit entries when `f1 != 0`. Needs one live reply or a longer read of the entry-building loop.
3. Meaning of `cap`, HIT `f1`/`f2` is inferred from the engine's use (`param_6`, `param_7`, `param_8`); names not shown in the binary.
4. No live capture: all reply layouts are from decompile; the missing-separator risk noted in r3b for QHN applies to joins done with `FUN_0016d760`/`append` here too (this round's loops show an explicit trailing `" "` per item, so they are less doubtful).
5. Which shipped bucket kinds are `g` is unknown: the only shipped manifest has `kind=o`, so STORE/HIT would error there; a model with `g`/`r` buckets and `pass`/`resume` records is not in the repo.

## Reproduce

```bash
cd /Users/peterb/Documents/AI/re-halogen/work/decomp/halogen-npu
sed -n 125,260p 00177e00.c ; sed -n 170,540p 00175f10.c
sed -n 20,60p 0016eab0.c ; sed -n 1,120p 0016e270.c
sed -n 1360,1385p 0012b110.c ; sed -n 460,500p 001d6480.c ; sed -n 1360,1405p 001d30b0.c
python3 - <<'E'
d=open('/Users/peterb/Documents/AI/re-halogen/binaries/halogen-npu','rb').read()
for a in (0xefe6c,0xefe4a,0xf2198,0xf2170,0xefd9e,0xef4ee,0xefadb,0xeffbb): print(hex(a), d[a:d.index(b'\0',a)])
E
ssh ai-core 'cd ~/re-halogen && r2 -q -e scr.color=0 -c "pd 6 @ 0x765a0" binaries/halogen-npu'   # lea rsi, str.no_such_entry
```

## 7. Orchestrator addendum: flux2-klein-4b manifest fetched (2026-10-10)

The orchestrator range-read `devices/devices.hnpm` of `peonist-ai/halogen-npu-flux2-klein-4b` at revision `f71d5e353676a7f6964902cbfabea79c7285578e` (bytes 0-8391, HTTP 206, 8392 B). sha256 `f65663b3e9227c2daa2d954725ca59f8e8f2596947f2f53c9b6354d25069b615` matches `image/npu-models.txt:76`. Saved as `analysis/evidence/npu-flux2-klein-4b/devices.hnpm`. Its `elf` records (lines 3-5):

```
elf res=0 file=u0.elf ...
elf res=512 file=u1.elf ...
elf res=256 file=u2.elf ...
```

With section 5 (keys = `res=` of `elf` records, `res=0` is the text encoder set and is excluded by `0 < key`): the legal IMG sides for the only shipped image model are 256 and 512. Any other side replies `ERR <tag> no image size <side>x<side>`.

Revised verdict: PARTIAL, narrowed. The grammar and the IMG side set are closed. Residual 2 (how many ids a STORE reply lists, which entries a HIT reply lists) is a behavioral rule, not naming (PR review finding), and stays open; residuals 3 to 5 are naming or capture limits.
