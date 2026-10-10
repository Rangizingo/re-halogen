# NPU STORE and HIT round 5: reply entry-id count, which entries HIT lists, bucket acceptance from two vendor manifests

Scope: `binaries/halogen-npu`, static only (nothing was run). Decompiles are `work/decomp/halogen-npu/<addr>.c`, cited `<addr>.c:line`; Ghidra address = file + 0x100000. String bytes were read from the binary with python (`d[a:d.index(b'\0',a)]`, file offset = Ghidra address - 0x100000). Continues `analysis/closure/npu-store-hit-r4.md` (r4). Inputs: `analysis/evidence/npu-qwen3.5-2b/devices.hnpm` (129 lines) and `analysis/evidence/npu-qwen3-embedding-0.6b/devices.hnpm` (279 lines). "proof" = quoted code or bytes; "inferred" = reasoned, not shown. Sources were read from `/Users/peterb/Documents/AI/re-halogen`.

## Verdict: CLOSED

Both open questions of r4 (U02) are fixed by code:
- (a) A STORE reply lists exactly B entry ids (one per row of the batch, B = the bucket's B), never one per 16-token block. Section 3.
- (b) A HIT reply lists newly created entries, one per row N (N = B = 1), only when `f1 != 0`; it never lists the entries that were hit. With `f1 == 0` it lists none. Section 4.

Also fixed from the manifests: in these two models no bucket accepts HIT, and only the three qwen3.5-2b `g` buckets accept STORE (sections 1 and 5). Caveat: ids are minted from a global counter when the reply is built (section 3.4), so id VALUES depend on session history; a fresh process starts at 1 by the counter's initializer (inferred that nothing else consumed it first). No live run was done; the timing words are not covered.

## 1. Buckets in the two manifests

`bucket` record fields: `kind` string at struct +0x20, `S` +0x60, `B` +100, `M` +0x68, `xd` +0x6c (default 0) (`00145eb0.c:626-634`; keys `B` file 0xef174, `M` 0xef9c2, `xd` 0xef98e). `FUN_0016e270` copies B and S into the info struct: `*(undefined4 *)(param_3 + 4) = *(undefined4 *)((long)param_2 + 100); *(undefined4 *)(param_3 + 8) = *(undefined4 *)(param_2 + 0xc);` (`0016e270.c:33-34`).

### qwen3.5-2b (`family name=qwen35_hybrid`, `devices.hnpm:2`; `task=generate`, `image/npu-models.txt:64`)

| Bucket | kind | B | S | xd | Pass record | STORE | HIT |
|---|---|---|---|---|---|---|---|
| u0 | g | 1 | 1024 | 3 | `pass xw=8982528 gbase=69632 gst=1048576 fkv=1179648 bucket=u0` (line 10) | accepts | rejects |
| u1 | g | 1 | 4096 | 3 | `pass ... gst=1048576 fkv=4718592 bucket=u1` (line 50) | accepts | rejects |
| u2 | g | 1 | 16384 | 3 | `pass ... gst=1048576 fkv=18874368 bucket=u2` (line 90) | accepts | rejects |

Bucket lines: `devices.hnpm:9,49,89` (`bucket kind=g S=1024 B=1 M=1024 xd=3 name=u0 ...`). No `resume` record exists (record types present by `awk '{print $1}'`: bucket, buf, compat, decode, defaults, family, hnpm, kernel, pass, plan, role, route). `decode` (line 129) is the separate u3.elf decode image, not a bucket.

### qwen3-embedding-0.6b (`family name=qwen3_dense`, `devices.hnpm:2`; `task=embedding`, `image/npu-models.txt:42`)

| Bucket | kind | B | S | xd | Records | STORE | HIT |
|---|---|---|---|---|---|---|---|
| u0 | g | 1 | 128 | 0 | pass (line 13) | rejects | rejects |
| u1 | g | 1 | 512 | 0 | pass (39) | rejects | rejects |
| u2 | g | 1 | 1024 | 0 | pass (65) | rejects | rejects |
| u3 | g | 1 | 2048 | 0 | pass (91) | rejects | rejects |
| u4 | g | 1 | 4096 | 0 | pass (117) | rejects | rejects |
| u5 | g | 8 | 512 | 0 | pass (143) | rejects | rejects |
| u6 | p | 1 | 128 | 0 | `resume bucket=u6 N=128 B=1 P=192 capb=6 kvh=8 blk=32 blkw=2304` (195) | rejects | rejects |
| u7 | p | 1 | 512 | 0 | `resume ... N=512 P=192 capb=6` (223) | rejects | rejects |
| u8 | p | 1 | 128 | 0 | `resume ... N=128 P=256 capb=8` (251) | rejects | rejects |
| u9 | p | 1 | 512 | 0 | `resume ... N=512 P=256 capb=8` (279) | rejects | rejects |

Bucket lines: `devices.hnpm:12,38,64,90,116,142,168,196,224,252`.

### Acceptance rules (proof)

`FUN_0016e270` is the only per-kind gate for both verbs (called from `FUN_0016eab0`, `0016eab0.c:15`):

```
cVar5 = FUN_00139a60(param_2 + 4,&DAT_001ef0a3);        // kind == "r"   (0016e270.c:35; the string at 0xef0a3 is 'r')
if (cVar5 == '\0') {
  cVar5 = FUN_00139a60(param_2 + 4,"g");                // kind == "g"    (:37)
  if ((cVar5 != '\0') && (*(int *)((long)param_2 + 0x6c) == 3)) {   // and xd == 3 (:38)
    ... lVar9 = FUN_0016bf50(param_1 + 0x250,&local_58,param_2);     // "pass" record of this bucket (:40)
    ... "gst" -> param_3+0x30, "fkv" -> +0x38, "xw" -> +0x20, "gbase" -> +0x28   (:46-62)
    goto LAB_0016e5de;                                                  (:64)
  }
  ... pcVar13 = ": this bucket kind takes no state store";                (:66-68)
```
and the shared tail:
```
LAB_0016e5de:
  lVar9 = 0x100000;  if (param_3[1] != '\0') lVar9 = 0x200000;
  if ((*(long *)(param_3 + 0x30) == lVar9) && ((param_3[8] & 0xf) == 0)) return 1;   (0016e270.c:219-224)
```
Key strings: `pass` 0xef9a1, `xw` 0xef9a6, `gbase` 0xef9a9, `gst` 0xefcd1, `fkv` 0xefcb4, `: this bucket kind takes no state store` 0xf1f18.

- `g` path: needs `xd == 3`, a `pass` record naming the bucket, `gst == 0x100000` (the `store1` byte stays 0, so the 0x200000 alternative is not used) and `S % 16 == 0`. It never sets `*param_3`, so the kind flag stays 0 (info is initialized `local_240 = 0x100000000`: byte 0 = 0, +4 = 1; `00177e00.c:175,246`).
- `r` path (resume) sets `*param_3 = 1` (`0016e270.c:80`) and needs a `resume` record, B == 1, no base0/xr, then the same gst/S tail.
- STORE accepts only flag 0, so only `g` with xd 3: `if ((lVar16 != 0) && (((char)local_240 != '\0' || (local_240._4_4_ != iVar5)) || ((int)uStack_238 != iVar6)))` then `"a state store runs its own bucket kind at its own B and S"` (`00177e00.c:183-186`; `lVar16 != 0` means the lookup succeeded, `0016eab0.c:15-16`).
- HIT accepts only flag 1 with `N == B`: `if ((local_4a8 == 0) && (((char)local_310 == '\0' || (local_310._4_4_ != iVar11)))) FUN_0016d620(.., "a hit runs a resume bucket at its own B")` (`00175f10.c:533-534`). `local_4a8` is the length of the error string `local_4b0` (`local_5c8 = &local_4b0`, `00175f10.c:361`), so this text is only set when no earlier message exists.

Results and the cause of each rejection:
- qwen3.5-2b u0, u1, u2: g, xd=3, gst=1048576=0x100000, S in {1024, 4096, 16384} all multiples of 16, B=1. STORE passes the gate and B/S match. HIT passes `FUN_0016e270` but the info flag is 0: `ERR <tag> a hit runs a resume bucket at its own B`.
- embedding u0 to u5 (g, xd=0): the `xd == 3` test fails, so both verbs reply `<bucket>: this bucket kind takes no state store`.
- embedding u6 to u9 (kind `p`): kind is neither `r` nor `g`, same error for both verbs. Their `resume` records are used by a different path: the prefix-cache picker `FUN_0014e0f0` tests kind `p` (`0014e0f0.c:115`, `DAT_001f0f46` = `p` at file 0xf0f46), requires `*(int *)((long)puVar8 + 100) == 1` (B = 1, `:116`) and reads the `P` key of the resume record (`:117-119`, key `P` file 0xefd96); it is reached from `0014a6f0.c:160`. The verb HIT cannot address them even though they carry `resume` records.

Consequence: no shipped manifest in the repo has `kind=r`. HIT has no accepted target in the two models given, so its reply layout stays code-derived (section 4).

## 2. Command grammar recap (from r4, re-checked)

`STORE <tag> <bucket> <cap> <B> <S> <ids: B*S>`; `HIT <tag> <bucket> <f1> <cap> <f2> <N> <h_1..h_N> <slot_1..slot_N> <M> <row_1..row_M> <ids: N*S>`. Token parse and counts per r4 sections 2 and 3 (`00177e00.c:142-246`, `00175f10.c:171-531`); not re-derived here except where stated.

## 3. (a) How many entry ids a STORE reply lists: B, one per batch row

### 3.1 The reply loop reads the engine's output vector

STORE calls the engine with `&local_200` as the stats argument `param_11`:

```
iVar6 = FUN_001734f0(param_1,lVar16,&local_240,&local_400,local_280,1,uVar4,0,&local_2a0,
                     &local_260,&local_200,local_460);          (00177e00.c:393-394)
```
`param_5 = local_280` (zeroed: no input entries), `param_6 = 1` (store_flag), `param_7 = uVar4` (cap), `param_8 = 0`. On success the loop walks `plVar17 = local_200 .. plVar3 = plStack_1f8` in 16-byte steps and prints `FUN_00172370(&local_2a0)` for each (`00177e00.c:406-407,413-424`). The number of ids is the number of 16-byte slots in `param_11[0]..param_11[1]`.

### 3.2 The engine builds one entry per batch row

`FUN_001734f0` writes `param_11[0]`/`[1]` only at the end, copying its local entry vector `local_1f8 .. local_1e8` (`001734f0.c:869-1028`; `uVar22 = (long)local_1e8 - (long)local_1f8` at `:871`, `param_11[1] = *param_11 + uVar22;` at `:1028`). That local vector is filled in one place:

```
if ((param_6 == '\0') || (iVar1 < 1)) {                    (001734f0.c:255)
  local_210 = 0; local_1e8 = 0; local_1f8 = 0;             (:256-258, empty)
} else {
  ...
  do {
    puVar17 = (undefined8 *)_Znwm(0x70);                   (:272, one new entry object)
    ...
    *local_1e8 = (long)puVar44;  FUN_00139d70(local_1e8 + 1);  local_1e8 = local_1e8 + 2;   (:520-522, push)
    FUN_00139cf0(puVar17);
    local_178 = local_178 + 8;
    local_208 = local_208 + uVar40 * 4;
  } while (local_178 != (long)iVar1 << 3);                 (:527)
}
```
- `iVar1 = *(int *)(param_3 + 4)` (`001734f0.c:110`) is the info struct's B, which is the bucket's B (`0016e270.c:33`). `local_178` steps 8 and the loop ends at `iVar1 << 3`, so it runs exactly B times, pushing exactly one entry per pass. The only other exits are error returns (for example `"the arena's capacity is below the new entry"`, `:436`).
- Each entry takes its own row: `puVar41 = (undefined4 *)(local_208 + *param_4)` with `local_208 += uVar40*4` per pass, `uVar40 = (ulong)(int)uVar12` = S (`001734f0.c:~250`), and copies `uVar16 = local_214` ids. On the non-resume path (`*param_3 == 0`) `local_214 = uVar12` = S (`:120`). So each entry covers one full S-token row, not a 16-token block. The `>> 4` quantities in the engine are the arena block counts inside one entry (`:410,433`).
- `B` is both the handler's token-4 value `iVar5` and, by the mismatch check at `00177e00.c:183-186`, equal to the bucket's B.

Count rule: ids listed = B when `store_flag != 0` (always for STORE); 0 when `store_flag == 0` or B < 1.

### 3.3 Arena capacity for a g bucket

For `*param_3 == 0` the entry's arena is built by `FUN_0016cea0(piVar48, param_1 + 0x180, param_7, nLinear)` (`001734f0.c:392`), which stores the cap verbatim: `*param_1 = param_3;` (`0016cea0.c:~37`). There is no "0 selects the default" on this branch (that default exists only on the resume branch, `001734f0.c:410-411`). The check `if (*piVar48 < iVar39 >> 4)` with `iVar39` = the entry's token count (S) then gives `"the arena's capacity is below the new entry"` (`:432-436`). So STORE needs `cap >= S/16`: u0 >= 64, u1 >= 256, u2 >= 1024. This corrects r4 section 2 for STORE: the "0 keeps default" reading applies to HIT's resume branch only.

### 3.4 What an id is

`FUN_00172370(&entryptr)` mints a new id on every call: `iVar2 = *(int *)(lVar3 + 0x30); ... *(int *)(lVar3 + 0x30) = iVar2 + 1;` (`00172370.c:18-20`), inserts a node keyed by `iVar2` into the singleton's `std::map` at `lVar3 + 8` with the entry at node `+0x28` (`:21-110`), pushes the entry onto a list at `lVar3 + 0x38`, and returns `iVar2`. The counter is initialized by `_DAT_00214050 = 1` (`0016b6d0.c:45`; singleton base `0x214020`, so `+0x30` = `0x214050`). Ids are therefore assigned when the reply is built, are sequential integers starting at 1 in a fresh process, and the same counter serves every caller of `FUN_00172370`.

## 4. (b) What a HIT reply lists: new entries, only when f1 != 0

### 4.1 Which vector the loop iterates and where it is filled

The HIT reply loop is `for (puVar28 = local_200; puVar5 != puVar28; puVar28 = puVar28 + 2)` with `puVar5 = puStack_1f8`, calling `uVar10 = FUN_00172370(&local_350);` on each 16-byte slot (`00175f10.c:939,954`). `local_200`/`puStack_1f8` are the first two words of the stats object passed to the engine as `param_11`: `FUN_001734f0(param_1,local_658,&local_310,&local_510,&local_570,lVar15 != 0,uVar10,lVar16 != 0,&local_550,&local_530,&local_200,local_5c8)` (`00175f10.c:599-600`). The hit-entry handles parsed from the command go to a different argument, `&local_570` (`param_5`, a vector of 8-byte raw pointers: `iVar1 != (int)(param_5[1] - *param_5 >> 3)`, `001734f0.c:131`). So the loop never iterates the hit entries; it iterates the engine's output vector, the same `local_1f8..local_1e8` of section 3.2.

### 4.2 Contents: new entries, one per row N (= B = 1)

Same loop as 3.2, on the `*param_3 != 0` (resume) branch:
- `puVar17 = _Znwm(0x70)` allocates a NEW entry each pass (`001734f0.c:272`); the push stores `puVar44 = puVar17 + 2` (`:520`).
- The hit entry is only a source: `FUN_0016a640(puVar44, *(undefined8 *)(*param_5 + local_178))` copies its id vector into the new entry (`:295`); then `piVar48 = (int *)plVar36[9]` takes its arena pointer and `FUN_00139d70(&uStack_68, plVar36 + 10)` shares its refcounted control block (`:404,420`), after `piVar48[1] = iVar39` or a rebuild via `FUN_0016d490` when the block count differs (`:407-421`).
- Count: the loop runs `iVar1` = N times (info +4 = bucket B; HIT requires `N == B`, `00175f10.c:533`; the resume gate requires B == 1, per r4 from `0016e270.c`), so one new entry per HIT with `f1 != 0`.
- With `f1 == 0` the call passes `store_flag = 0`, the guard at `001734f0.c:255` takes the empty branch, `param_11[0..1]` is an empty range, and the loop at `00175f10.c:939` prints nothing: no ids in the reply.
- With `f1 != 0` the engine also needs `(store_at & 0xf) == 0` and `store_at <= S` when `store_at != 0`, else `"the store row is not a block boundary"` (`001734f0.c:128-133`), and each hit entry must match the bucket (id count P, `capb`, linear-layer list), else `"an entry that is not this bucket's (P <P>, capb <capb>)"` (`:175-195`).

The ids in a HIT reply are fresh counter values (3.4), never the `h_i` the caller passed, since every `FUN_00172370` call increments the counter (`00172370.c:18-20`).

Reply shape (r4 section 3, unchanged): `OK <tag> <argmax_i...> <row score words...> <new ids...> <t1> <t2> <t3> <t4>`, in loop order `00175f10.c:628`, `:640`, `:939`, then times `:990-994`.

## 5. Worked examples for qwen3.5-2b

Notation: `<IDS1024>` = 1024 decimal token ids with `0 <= id < vocab`, vocab = `*(lVar13 + 0x30) / (hidden*2)` from the `embed` blob (`001734f0.c:204-210`); the family string gives `v248320` (`devices.hnpm:2`). `grep embed devices.hnpm` finds nothing, so the blob lives in the weights bundle and "no embed blob" (`:199-200`) cannot be ruled out from the manifest.

### STORE (accepted)

```
STORE s1 u0 64 1 1024 <IDS1024>
```
Tokens: tag `s1`; bucket `u0`; cap 64 (= S/16, section 3.3); B 1; S 1024; 1 x 1024 ids. Checks passed: bucket loaded (`0016eab0.c:9`), g/xd=3/pass/gst/S%16 (`0016e270.c:35-64,219-224`), flag 0 and B, S equal to the bucket's (`00177e00.c:183-186`), `1024 & 0xf == 0` (`001734f0.c:121-124`).

Reply: `OK s1 1 <t1> 0 <t3>`. The `1` is the single entry id (B = 1 entry; first id of a fresh process is 1). t1 = total wall time in microseconds, the middle word is always 0, t3 = store phase time (r4 section 2). Same shape for `u1` (cap >= 256, 4096 ids) and `u2` (cap >= 1024, 16384 ids).

Rejected variants: `cap 0` gives `ERR s1 the arena's capacity is below the new entry`; `B 2` gives `ERR s1 a state store runs its own bucket kind at its own B and S`.

### HIT (no accepted line exists)

```
HIT h1 u0 1 64 0 1 1 0 0 <IDS1024>
```
Tokens: f1 1, cap 64, f2 0, N 1, h_1 = 1 (the id from the STORE reply), slot_1 0, M 0 head rows, 1024 ids. It parses (`00175f10.c:171-531`) and the bucket lookup succeeds, then it fails on the flag test: `ERR h1 a hit runs a resume bucket at its own B` (`00175f10.c:533-534`). If h_1 names no live entry, `no such entry` is set first and kept (the `local_4a8 == 0` guard at `:533` skips the later text). Changing `f1` to 0 does not change the outcome.

An `OK` reply with listed ids needs a bucket with `kind=r`, a `resume` record, B = 1 and `gst = 0x100000`; neither manifest has one. There the reply would be `OK h1 <argmax or -1> <scores> <1 new id> <t1> 0 <t3> <t4>` for `f1 != 0`, and the same without the id for `f1 == 0`.

## 6. Residuals

1. No live reply was captured, so id values (counter starts at 1, shared across callers, section 3.4) and the timing words are unverified. Settling command on a machine with the model loaded: send `STORE s1 u0 64 1 1024 <1024 valid ids>` twice. Expected: `OK s1 1 ...` then `OK s2 2 ...`. A reply with two ids in one B=1 STORE would refute the one-entry-per-row rule.
2. No shipped bucket has `kind=r`, so the HIT success path (new-entry id listing) is code-derived only. The `r` literal exists in the binary as a 1-byte tail of the JSON `\r` escape string (file 0xef0a2-0xef0a3); the compare is real (`0016e270.c:35`) but no manifest in the repo uses it, while `p` is used by the prefix picker (`0014e0f0.c:115`). A `kind=r` manifest or a live HIT on a `p` bucket (expect `<bucket>: this bucket kind takes no state store`) would settle it.
3. `embed` blob presence for qwen3.5-2b is not visible in `devices.hnpm`.
4. Meanings of `cap` (blocks), `f1`, `f2` stay inferred from use (r4 residual 3).
5. `HALOGEN_NPU_CACHE_ENTRIES` (`0016b6d0.c:60`) bounds the entry store; eviction and its effect on listed ids were not traced.

## Reproduce

```bash
cd /Users/peterb/Documents/AI/re-halogen/work/decomp/halogen-npu
sed -n 28,70p 0016e270.c ; sed -n 215,230p 0016e270.c
sed -n 250,275p 001734f0.c ; sed -n 515,528p 001734f0.c ; sed -n 866,872p 001734f0.c ; sed -n 1026,1029p 001734f0.c
sed -n 390,425p 00177e00.c ; sed -n 596,601p 00175f10.c ; sed -n 936,956p 00175f10.c
sed -n 14,22p 00172370.c ; sed -n 44,46p 0016b6d0.c
cd ../../.. && grep -nE '^(bucket|pass|resume)' analysis/evidence/npu-qwen3.5-2b/devices.hnpm analysis/evidence/npu-qwen3-embedding-0.6b/devices.hnpm | cut -c1-150
python3 - <<'E'
d=open('binaries/halogen-npu','rb').read()
for a in (0xef0a3,0xf0f46,0xef9a1,0xefcd1,0xef98e,0xf1f18): print(hex(a), d[a:d.index(b'\0',a)])
E
```
