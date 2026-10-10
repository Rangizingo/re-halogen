# U01 round 3: readers of HNPB header +0x08

Scope: `binaries/halogen-npu` (Ghidra C in `work/decomp/halogen-npu/<addr>.c`, Ghidra = file + 0x100000) and `binaries/flash_serve`. Static only, nothing executed. "proof" = read from code or bytes, "inferred" = reasoned.

## 1. flash_serve and halogen-tools do not read HNPB (proof)

- `grep -ac "HNPB\|hnpb"`: flash_serve 0, halogen-tools 0, halogen-npu 5 hits.
- `grep -rl 42504e48 work/decomp/flash_serve`: no files.
- In halogen-npu the magic constant `0x42504e48` appears in exactly three decompiled functions: reader `00167960.c:348`, writers `0016eb50.c:1150` and `0017a920.c:115`.

## 2. The one reader: FUN_00167960 (proof)

`00167960.c:347-353`, `param_1` is the container (long*), `piVar11` the mmap base:

```
*param_1 = (long)piVar11;
if ((*piVar11 == 0x42504e48) && (piVar11[1] == 1)) {
  uVar31 = param_1[1];
  *(int *)(param_1 + 2) = piVar11[2];
  uVar29 = *(ulong *)(piVar11 + 10);
  if (uVar31 == uVar29) {
```

What it does with +0x08 (`piVar11[2]`): stores it unchanged at container byte +0x10 (`param_1 + 2` on a long*). The only gates around it are magic (+0x00), `piVar11[1] == 1` (version, +0x04, else `": not an HNPB v1 file"` at `:1529`), header +0x28 == file size, and `table_off + n*0xa8 <= size` (`:355-356`). No compare, switch or range test on +0x08. Any value, 2 included, is accepted. (The version check is on +0x04, not +0x08. +0x08 is not a version, not a blob count, which is header +0x10, `*(long *)(piVar11 + 4)`, `:354`.)

## 3. Consumers of container +0x10: none found (proof for windows listed, inferred overall)

Callers of the reader (`grep -n FUN_00167960`): `00170ea0.c:79`, `00145eb0.c:108,116`, `001a7be0.c:1057`, `001d6480.c:54`. Container location and what touches its +0x10:

- `00170ea0`: container is `local_98` (+0x10 = `local_88`). `local_88 = 0` at `:71`, no other reference in the file. Callee `FUN_00170cc0` gets the container and calls only `FUN_00162ec0` and `FUN_00162fd0`.
- `FUN_00162ec0` (name lookup) reads container +0x18, +0x38, +0x40 only. `FUN_00162fd0` (xor32 check) reads `*param_1` (base) only (`00162fd0.c:23`).
- `FUN_00162e00` (container destructor): reads `[0]`, `[1]`, `[3]`, `[4]`, `[8]` and frees; `param_1[2]` count in the file is 0.
- `00145eb0`: containers at `param_1+400` and `param_1+0x1f0`, so +0x10 would be `param_1+0x1a0` / `+0x200`. `grep -n "0x1a0\|0x200" 00145eb0.c`: no hits. `001d6480`: container `param_1+0xa0`, +0x10 = `param_1+0xb0`; `grep -n "0xb0" 001d6480.c`: no hits. `001a7be0`: container is the stack slot `local_390`, only `FUN_00162ec0`, `+0x28` offset reads and `FUN_00162e00` use it.
- Limit: I did not grep every function that might take a container pointer from these classes. Field never read in all windows above; "no consumer anywhere" is inferred.

## 4. Writers (proof for values, inferred for roles)

Both write header bytes 4..11 as one 8-byte pair `{version, tag}`:

- Bulk writer `FUN_0016eb50` (fopen "wb"): `uVar11 = DAT_001f4c28; ... *(undefined8 *)((long)local_278 + 4) = uVar11;` (`0016eb50.c:1148-1151`). Bytes at file 0xf4c28: `01 00 00 00 03 00 00 00`, so +0x08 = 3. Callers: `00172600.c:179` (cache save in the engine command loop) and `00171ca0.c:81` (`cache ecopy`).
- Streaming finalizer `FUN_0017a920`: `local_e8._4_4_ = (undefined4)DAT_001f4bf0; uStack_e0 = (undefined4)((ulong)DAT_001f4bf0 >> 0x20);` (`0017a920.c:118-119`). Bytes at file 0xf4bf0: `01 00 00 00 01 00 00 00`, so +0x08 = 1. Caller `001a7be0.c:1168` (`FUN_0017a920(local_390)`).
- ELOAD cache entry object copies the same pair: `puVar11[1] = DAT_001f4bf0;` (`00172600.c:118`), so in memory it is 1.
- No writer in halogen-npu stores 2. Shipped `devices/consts.hnpb` has +0x08 = 2 (`analysis/evidence/npu-decider-0.8b/consts-hnpb-header.txt`: `484e 5042 0100 0000 0200 0000`).

## 5. Answer

- The reader accepts 2 (and any u32). It copies the field into the container and never branches on it. So `halogen-npu` gives it no behavior: not a version check (that is +0x04 == 1), not a blob count (+0x10), not a flags word it tests.
- Writers use it as a per-writer constant: 3 for the bulk path, 1 for the streaming path. A third value, 2, on a shipped file means a third producer, not `halogen-npu` (inferred; the vendor build tool is not in the repo).
- Meaning of the value itself (inferred): producer or flavor tag, ignored at load. The shipped 1/2/3 values are consistent with a small enum of pack origins. The code does not say what 2 names.

## Verdict: PARTIAL

CLOSED: every reader of +0x08 (one, `FUN_00167960`), what it does (copy to container +0x10, no check), that 2 is accepted, which values the two in-binary writers store (3 bulk, 1 streaming) and where.
Remains: what "2" denotes. It needs the producer of the shipped `consts.hnpb` (not in this repo), or a second shipped HNPB (an `.hnpw`, not fetched) showing which value other pack kinds carry. Static work on halogen-npu cannot go further because the field has no consumer. Also unproven: absence of a consumer outside the windows listed in section 3.

## Reproduce

```
cd /Users/peterb/Documents/AI/re-halogen
grep -rl 42504e48 work/decomp/halogen-npu work/decomp/flash_serve
sed -n 347,356p work/decomp/halogen-npu/00167960.c
sed -n 1148,1151p work/decomp/halogen-npu/0016eb50.c
sed -n 115,119p work/decomp/halogen-npu/0017a920.c
grep -n "FUN_00167960" work/decomp/halogen-npu/{00170ea0,00145eb0,001a7be0,001d6480}.c
grep -n "0x1a0\|0x200" work/decomp/halogen-npu/00145eb0.c; grep -n "0xb0" work/decomp/halogen-npu/001d6480.c
python3 -I -c "d=open('binaries/halogen-npu','rb').read();print(d[0xf4bf0:0xf4bf8].hex(),d[0xf4c28:0xf4c30].hex())"
for b in flash_serve halogen-tools halogen-npu; do grep -ac "HNPB\|hnpb" binaries/$b; done
```
