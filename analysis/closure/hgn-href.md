# U08: the HREF record

Verdict: CLOSED, with a correction. The premise was wrong. `HREF` is not an
in-.hgn side-file reference. It is the magic of the `halogen-tools ppl`
"reference dump" file (`--ref-out REF.bin` / `--ref REF.bin`), a separate
standalone file. No .hgn record, dtype or tensor uses it. The n-gram table is
found a different way (section 4).

Address note: objdump addresses are file virtual addresses; Ghidra = objdump + 0x100000.
Both HREF sites sit inside one function, `FUN_01825fb0` (`ppl` driver, 59,962 bytes,
`work/decomp/halogen-tools/01825fb0.c`, index.tsv line 7).

## 1. Evidence for the two sites

| Site (objdump) | Ghidra | Role | Evidence |
|---|---|---|---|
| 0x172610b | 0x182610b | reader, `--ref` | `cmpl $0x46455248, 0x650(%rsp)`; decompile 01825fb0.c:1162 `if (((sVar14 == 1) && ((int)local_2c8 == 0x46455248)) && (local_2c8._4_4_ == 2))` |
| 0x1726afa | 0x1826afa | writer, `--ref-out` | `movabsq $0x246455248, %r8`; decompile 01825fb0.c:7608 `local_4c8 = (FILE *)0x246455248;` then 7610 `fwrite(&local_4c8,0x20,1,local_320)` |

The earlier "0x1725112 / 0x1725afc" offsets in hgn-format.md were wrong. Checked with
objdump: 0x1725112 is mid-instruction in a `__throw_length_error` landing pad, and
0x1725afc is an exception cleanup (`operator delete`). Neither holds the tag.
Other binaries: `objdump -d` of flash_serve and halogen-npu has 0 hits for `0x46455248`;
no flash_serve decompile contains it.

Command strings: strings-tools.txt:2047-2048 `[--ref-out REF.bin [--topk 128]]   write the distribution at every position (top-K, tail, argmax)` and `[--ref REF.bin [--per-pos P.bin]]  score against it: KL(ref || this), top-1 agreement, delta p(target)`; :1227 `%s: not a reference dump`.

## 2. File layout (little-endian, x86-64)

Header, 0x20 bytes (writer 01825fb0.c:7606-7610, reader 1161-1166):

| Off | Size | Type | Field | Evidence |
|---|---|---|---|---|
| 0x00 | 4 | bytes | magic `HREF` (0x46455248) | :1162 and :7608 |
| 0x04 | 4 | u32 | version = 2 (reader rejects others: `local_2c8._4_4_ == 2`) | :1162; high half of 0x246455248 |
| 0x08 | 4 | u32 | K (top-K count) | writer :7609 `CONCAT44(local_53c,*(undefined4 *)(pcVar40 + 0x20))`, low half; reader prints it as `K %u`. `--topk` default 128 per usage |
| 0x0c | 4 | u32 | vocab size | high half of the same CONCAT44 (`local_53c`); reader compares to the engine: `%s: vocab %u, the engine's is %d` |
| 0x10 | 8 | u64 | positions N (tokens minus 1) | writer :7607 `(x*0x40000000 + -0x100000000) >> 0x20` = tokens minus 1 (inferred: x is a byte span, x/4 = token count); reader `local_2b8` |
| 0x18 | 8 | u64 | sequence length (the `--chunk`/context length, `param_6[0x98]`) | writer :7606 `SEXT48((int)local_500)`, with `local_500 = *(uint *)(param_6 + 0x98)` (:2184-2185); matches the printed text `reference %s: %llu positions, top-%u, sequences of %llu` (strings-tools.txt:1649). The reader zero-inits and does not range-check it (inferred: informational) |

Reader validity checks (:1167-1174): `1 <= K <= 0x1000` (4096), `1 <= vocab <= 0x1000000`, `N <= 0x100000000`, and
`(K*12 + 0x10) * N <= filesize - 0x20`. Failure prints `the reference dump's header does not fit its file (K %u, vocab %u, %llu positions, %ld bytes)` (:2154).

Body: N fixed-size records, packed back to back from 0x20, record size `12*K + 16` (reader :2097 `(uint)(iVar11 * 0xc) + 0x10`; per-record `fread(__ptr, size, 1, fp)` at :2106, error `%s: short read`). No padding, no per-record tag, no checksum anywhere in the file.

Record (writer 01825fb0.c:7683-7689 and 4362-4420; reader copies at :2111-2145):

| Off | Size | Type | Field | Basis |
|---|---|---|---|---|
| 0x00 | 4 | f32 | `lse - logit[target]`, target NLL at that position | writer :7699 `*(float *)local_690 = (float)((double)pcVar40 - (double)fVar2)` where `pcVar40` is the per-position f64 log-sum-exp and `fVar2` is the target's logit (:7692). Meaning inferred from the arithmetic and the "delta p(target)" usage text |
| 0x04 | 4 | u32 | argmax token id | writer :7700 stores element 0 of the sorted top-K id array (Ghidra typed it `float`; it is the first int32 of `local_d8[pos]`). Inferred: = ids[0] |
| 0x08 | 8 | f64 | log of tail mass (mass outside top-K); constant `DAT_00112568` when tail <= 0 | writer :7692-7701: `pcVar59 = DAT_00112560 - sum(exp(lp))`, `dVar70 = log(pcVar59)` if > 0 else `DAT_00112568`; stored :7701. `DAT_00112560` is presumably 1.0 (inferred; .rodata bytes not read) |
| 0x10 | 4*K | i32[K] | top-K token ids, descending probability | writer memcpy `lVar44*4` from `local_d8[pos]`; reader copies into a 4-byte-element vector (:2125-2133). Order inferred from "top-K ... argmax" |
| 0x10+4K | 8*K | f64[K] | log-probs of those ids: `logit[id] - lse` | writer :7676-7683 `*(double *)(pcVar59 + lVar45*8) = logit - lse`, summed via `exp` for the tail |

The in-memory struct the reader builds is 0x48 bytes per position (vector<int> at +0x18, vector<double> at +0x30), which is a host layout, not the file layout.

Companion file: `--per-pos P.bin` (separate, no tag) is N x 16 B of f32 x 4 `kl, dp, top1, nll` (:7531-7535; message `per-position scores (f32 x 4: kl, dp, top1, nll)`).

## 3. How readers resolve it

The path is the CLI argument (`--ref` / `--ref-out`). No side-file lookup, offset, or checksum: it is opened with `fopen(path,"rb")` (:1156), the 0x20 header is read, the body is read sequentially. Integrity is only the size check above. flash_serve never reads it.

## 4. What the n-gram table actually uses (dtype / HALOGEN_NGRAM_TABLE / ngram_embedding)

- There is no HREF record and no dtype for one. DTYPE_NAMES in `work/hgn/hgnpy.py:33-35` has no reference type. The table is a second, ordinary `HGN1` container (a full .hgn) named by `HALOGEN_NGRAM_TABLE`; default name `qwen38-flash-next-ngram.hgn` beside the checkpoint (image/entrypoint.sh:479, `t="$dir/$NGRAM_TABLE_NAME"` :486, export at :503; analysis/entrypoint-flow.md:93).
- flash_serve gate: `work/decomp/flash_serve/01884460.c:529` `getenv("HALOGEN_NGRAM_TABLE")`; a GGUF is refused (:558 `HALOGEN_NGRAM_TABLE=%s is a GGUF; it must name the .hgn that holds the n-gram table`).
- Lookup is by tensor name, in the main .hgn first and then in the table .hgn: `work/decomp/flash_serve/01894790.c:7-21` (`FUN_01894600` main, then `FUN_019981d0` on the table checkpoint at `param_1+0x298`; miss prints `the n-gram table (%s) is not in %s; point HALOGEN_NGRAM_TABLE at the file that holds it`). The name is `layers.1.ple.ngram_embedding.weight` (01997e40.c:47; also 019af350.c:145 in the repack); the repack's `--with-table` flag is named in the message at 01997e40.c:52.
- "Read in place" therefore means the table .hgn's own payload is mapped and read directly (inferred); the link is by filename and tensor name, not by an offset record.
- The `"ngram_embedding"` string with `0xf` in `01889610.c:22`, `01951440.c:17`, `019513c0.c:12` is a family-name match, not an HREF consumer (no tag compare in those files).
- Which dtype the 47.7 GiB payload carries (fp8r id 6 or fp8g id 10 in hgnpy DTYPE_NAMES) is not shown here. Not established; reading the table .hgn header would settle it.

## 5. Residual items

- Suggested edits (not applied, task limited to this file): correct hgn-format.md section 6 (HREF is the ppl reference dump) and the U08 row in unknowns.md.
- Unverified: byte-level round trip. No REF.bin sample is in the repo; test needed: run `ppl ... --ref-out` on a small ids file and parse with the layout above.
- Inferred items are labelled above (field 0x00 meaning, argmax, tail floor, N derivation, sequence-length use).
