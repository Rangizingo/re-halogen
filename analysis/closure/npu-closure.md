# NPU closure: U01 to U03

Scope: binaries/halogen-npu. Decompile files are work/decomp/halogen-npu/<addr>.c (cited as `<addr>.c:line`). Ghidra vaddr minus 0x100000 = file offset (checked: `xxd -s 0xf4c28` returned the constants below). Labels: "proof" = read directly from decompile or bytes; "inferred" = reasoned, not shown.

## U01 HNPB header u32 at +0x08, per-blob pair at +0x50 and +0xa0

Verdict: PARTIAL

### Correction to analysis/npu-engine.md section 2.2 (proof)
The header offsets in the doc are wrong. The 64-byte header is:
- +0x00 u32 magic 0x42504e48, +0x04 u32 version 1, +0x08 u32 (the U01 field), +0x0c u32 0
- +0x10 u64 n_entries, +0x18 u64 table_off, +0x20 u64 0x1000, +0x28 u64 total file size
- Reader: `00167960.c:350-355` (pseudo `work/npu/0x167960.pseudo.c:376-383`): `*(int *)(param_1 + 2) = piVar12[2]; uVar33 = *(ulong *)(piVar12 + 10); if (uVar37 == uVar33)` (file size must equal header +0x28), then `lVar28 = *(long *)(piVar12 + 4)` (n_entries, +0x10), `local_270 = *(long *)(piVar12 + 6)` (table_off, +0x18), `table_off + n*0xa8 <= size`.
- Writer cross-check, `0016eb50.c:1148-1155`: `local_278[5] = lVar29` (+0x28 size), `*(undefined4 *)local_278 = 0x42504e48`, `local_278[2] = puVar35` (+0x10 count), `local_278[3] = uVar19` (+0x18 table), `local_278[4] = 0x1000` (+0x20).
- Hence the doc's "+0x20 n_entries, +0x30 table_off" is wrong; +0x20 is a constant 0x1000 (data start/alignment, inferred from blob_off 4096 alignment) and +0x28 is the file size.

### Header u32 at +0x08
- The reader only copies it into the container (`00167960.c:350`, container +0x10). I found no comparison against it: in the one container user I read in full (`00170ea0.c`, container is `local_98`, +0x10 is `local_88`) it is initialised to 0 at line 71 and never read again. Not grepped across all 1413 files for every container offset (1d6480, 145eb0 hold the container at +0x190/+0x1f0/+0xa0*8; no consumer found for +0x1a0/+0x200).
- Two writers emit different values (proof, bytes at file 0xf4c28 = `01 00 00 00 03 00 00 00`, and 0xf4bf0 = `01 00 00 00 01 00 00 00`):
  - `FUN_0016eb50` (bulk writer, fopen "wb", callers 00172600 and 00171ca0) writes `{version=1, +0x08=3}` from `DAT_001f4c28` (`0016eb50.c:1150-1151`).
  - `FUN_0017a920` (streaming finalizer, rewrites header at offset 0 after entries are appended; callers 00196980 `convert` and 001a7be0) writes `{version=1, +0x08=1}` from `DAT_001f4bf0` (`0017a920.c:115-119`).
- Answer: a per-writer format/flavor tag, 3 for the bulk-written pack and 1 for the streamed pack. Which file kind each writer produces (consts.hnpb vs .hnpw) is inferred from callers (convert is the streaming one), not proven. Reader does not validate it.

### Per-blob u32 at +0xa0 (proof)
- It is the xor32 of the blob bytes. Writer `0016eb50.c:1139`: `uVar10 = FUN_00162c40(); *(undefined4 *)((long)pvVar13 + 0xa0) = uVar10;` where `FUN_00162c40` is the AVX xor-of-u32-words routine (`00162c40.c:17-45`, same algorithm as `FUN_00162fd0`; Ghidra dropped its args at the call site, so the args (blob ptr, size) are inferred).
- Reader copies it to record+0x24 (`0x167960.pseudo.c:453`, `CONCAT44(*(+0xa0), *(+0x50))`), and the verifier compares it: `00162fd0.c:94: return *(uint *)(param_2 + 0x24) == uVar11;` with `uVar11` the xor32 of `*(param_2+0x28) + base` for `*(param_2+0x30)` bytes. Called by `FUN_0016dc30` (`0016dc30.c`), giving "has the wrong size or checksum".

### Per-blob u32 at +0x50
- Stored at record+0x20 (low half of the pair). Writer: `0016eb50.c:1135-1138`: `*(undefined4 *)(pvVar13 + 0x54) = 1; *(int *)(pvVar13 + 0x50) = (int)lVar37` with `lVar37 = plVar17[4]` (a field of the 0x48-byte in-memory entry), and +0x58 = `plVar17[8]`. Streaming writer `0017a920.c:88`: `uStack_98 = (ulong)*(uint *)(puVar9 + 4)` (same entry field), then up to three u64 values from a vector (+0x58..), then the offset at +0x70 and size at +0x78.
- Not proven: what the entry field means (dtype/layout tag is inferred from it being copied beside a shape-like u64 list at +0x58). The reader never compares it to anything I found. The reader also checks `table_off`, 4096 alignment and `off+size <= size`, nothing on +0x50.
- Entry layout from the streaming writer (proof of positions, inferred meaning): +0x00 name (<=0x4f chars), +0x50 u32 tag, +0x58 u64 x3 (shape/dims, inferred), +0x70 off, +0x78 size, +0x80..+0x98 u64 x3, +0xa0 xor32.

What remains: the producer of the tag (`plVar17[4]`, set by callers 00172600/00171ca0/convert) and any consumer of record+0x20 and container+0x10. Needs a read of those callers' entry builders.

## U02 Serve protocol verb grammars

Verdict: PARTIAL (framing, GEN, DEC/EMB/SCO/CLS shared parser, PFX, INFO, QHN, LEDGER, CANCEL, replies closed; field meanings of the numeric slots are inferred)

### Framing (proof)
- A line is split into whitespace tokens (vector of 0x20-byte strings; `param_3` of `FUN_0012b110`). Token 0 is `VERB` or `VERB@model`: `FUN_00126dd0` does `memchr(__s,0x40,...)` (0x40 = '@', `00126dd0.c:46`) and maps the verb to a head kind: EMB to "embedding", SCO to "score", CLS to "classify", PFX (head "embedding" default, no rename), DEC to "decision", GEN to "generate" (`00126dd0.c:87-103`). Error "no loaded model has a <kind> head" otherwise.
- Token 1 is the request tag echoed in every reply; with 1 token or fewer the default tag is `DAT_001eff65` (`0012b110.c:157-163`; the doc says "0", not read here, inferred).
- Verb literals (bytes at file 0xef172..): EMB, SCO, CLS, GEN at 0xef172/76/7a/7e; IMG 0xef3cc; DEC 0xef433; INFO 0xef3fb; QHN 0xefd83; strings "ERR " 0xef39b, "OK " 0xef0c9, "PFX " 0xef460, "LEDGER " 0xef469, "INFO " 0xef400.

### Verbs
- `INFO <tag>`: reply `INFO <tag> <capability fields>` (body from `FUN_00129cb0`, which emits `"models":[`; JSON-ish). Same reply for a lone `INFO` with an empty model list (`0012b110.c:168-198`, :296-312).
- `GEN[@model] <tag> t2..t8 [K k1..kK] [M m1..mM]`, parser `FUN_001b18b0`, all via strtod with full-token check (`001b18b0.c:26-45`, loop for tokens 2..8):
  - tokens 2..8 (7 required numbers): token 2 int (default 0x100 pre-set at `0012a4a0.c:112` `local_160 = 0x100`), tokens 3,5,6,7 stored as float32 (`vcvtpd2ps`, `001b18b0.c:49-52`), token 4 int, token 8 stored as a 64-bit integer (with sign handling via `^ 0x8000...`, `:56-62`). Meanings (max tokens, temperature, top_p, top_k, seed...) are inferred, not shown here.
  - token 9 optional count K (`0x120` = 9*0x20, `:63`), then K ints from token 10; then a count M and M ints (the second list goes to `param_2`). Line must be consumed exactly. Which list is the prompt vs stop ids: inferred, not proven.
  - Reply: `OK <tag> ` + 9 space-separated fields built with `FUN_0013a710`/`FUN_0013a9b0`, then `snprintf(" %016llx,%016llx,%d")` (two u64 hex values and an int, `0012a4a0.c:241`) then generated token ids. Failure: `ERR <tag> a malformed GEN line` (`:124`). Field names of the 9 reply fields not resolved.
- `DEC`, `EMB`, `SCO`, `CLS` share the parser `FUN_0011c0b0` (called for DEC at `0012e9d0.c:~616`; verbs EMB/SCO/CLS go through the registered handler at `0012b110.c:342-349`, `(**(code **)(lVar47+0x58))(...)` from the registry `FUN_0013e2e0`, which I did not trace to its handler bodies). Grammar from `0011c0b0.c:48-80,270-404`:
  - token 2 = n (strtoll, 1..0x40), token 3 = m (strtoll, 1..0x100000); then n groups each `<a> <c> <c ints>` with c <= 0x1000 (`bVar38 = local_60 < 0x1001`, `:92`), then n*m trailing ints (`uVar19 * lVar20 == lVar35`, `:402`); all tokens must be consumed. What a, c, the ints and the n*m tail mean (token ids, row lengths, position grid) is inferred only.
  - Failures: `ERR <tag> a malformed DEC line` (`0012e9d0.c:627`); the string table shows the same pattern " a malformed " + verb for the others (string at 0xef437).
  - Replies: prefix is the verb (`DEC `, `EMB `, `SCO`, `CLS` per npu-engine.md; body encodings not decoded here).
- `PFX[@model] <tag> n <ids...>` (`0012b110.c:~360-405`): token 2 = n via strtoll (0..0x100000), required token count == n + 3 (`bVar59 = (size>>5) == uVar36 + 3`), tokens 3.. are n int32 ids (strtol). Reply `PFX <tag> <a> <b>` where a = `(long)(ram0x001f4bd0 * double)` of a float returned by `FUN_0014d2b0` and b a count from `FUN_0013a710`; exact value meanings unknown (inferred prefix-cache hit/timing). Malformed: `ERR <tag> a malformed PFX line`.
- `QHN <tag>`: reply `QHN <tag> <count> <u32 ...>`: count = `(param_4[0xc]-param_4[0xb])>>2` then each u32 of the list at `param_4[5..6]` (`0012b110.c:246-266`, verb bytes at 0xefd83 are "QHN"). New finding: the earlier table lists "HN" as a PFX reply; the verb literal is QHN.
- `LEDGER <tag>`: reply `LEDGER <tag> ` + ledger text from `FUN_00143660(param_4+0x11)` (`0012b110.c:~270-285`). No arguments.
- `CANCEL <tag>`: in the per-connection loop, 6-char verb (`0x434e4143`,`0x4c45`, `0012e9d0.c:484-485`) takes the mutex `DAT_00213740` and appends token 1 (`local_200`) to the global cancel vector `DAT_00213710` (`:487-577`). No reply is written (jumps to `LAB_0012f690`; inferred no reply). Argument is the tag of the request to cancel.
- `QUIT`: 4-char compare `0x54495551` in the loop (`0012e9d0.c:357` region); closes the connection, no arguments.
- `IMG[@model] <tag> ...`: strings `a malformed IMG line`, `no image size ` (0xef3d0, 0xef3e6), `IMG ` reply prefix; argument list not decoded (the image verbs live in the flux/IMG handler, not in the paths read).
- Unknown verb: `ERR <tag> unknown verb <tok>` plus ` (an image model answers IMG)` when no model matched (`0012b110.c:184-193`). Registry lookup of other verbs is a std::map<string,handler> (`FUN_0013e2e0`), so verbs beyond the above could be registered at runtime; registrations at 00119560.c not read.
- `READY` is a banner not a verb (npu-engine.md section 5). `ERR <tag> <msg>`: `ERR ` at 0xef39b. `OK <tag> ...`: GEN success prefix (0xef0c9).

What remains: bodies of the registry handlers for EMB/SCO/CLS (0013e2e0 map, registered in 00119560.c), IMG argument list, the semantics of GEN's 7 numerics, the 9 GEN reply fields, PFX reply values.

## U03 main:<graph> name set

Verdict: STILL OPEN

- Proof of the construction: `00159980.c:~85-100`: the string is built as bytes `0x6e69616d` ("main") + `0x3a` (':') followed by the caller's name string (`param_2`), then used for `xrt::ext::kernel`. `FUN_00159980` is reached via `FUN_00159fc0` (callers 00130a20.c:839, 00149720.c:187, 001b3590.c:593, 001d30b0.c:916).
- In `00149720.c` the name arg is `local_118`, built by `FUN_0013f420(&local_118,param_1,param_2,param_3,param_4)` (:53), also used for error text ("<name>: no buffer"). So the name is runtime data produced from a record, not a literal. I did not decompile `FUN_0013f420`.
- Literal search: `strings -a` of the binary shows no `main:<graph>` literals; only the "main" token (0x59b63, 0x59ecc, libc-style, not graph names) and buffer-tag fragments `gx/L...`, `kv/A...`, `meta` (0xefd13-0xefd1d; `gx/L` and `kv/A` look like per-layer buffer names, inferred, not graph names).
- Inputs not in the repo: no devices.hnpm or .elf file exists under /Users/peterb/Documents/AI/re-halogen (find returned none). image/npu-models.txt lists only file names and sizes (u0..u9.elf per model, devices.hnpm sizes 11-24 KB), not graph names.
- Answer: the set is the set of names resolved from each model's devices.hnpm plan/ops records via `FUN_0013f420`, prefixed with `main:`; it cannot be enumerated without the manifest or the ELFs' kernel tables. To close: obtain a devices.hnpm (text, line-oriented) from a model dir and read the plan/ops fields, or decompile `FUN_0013f420` to confirm which field supplies the name.
