# The `.hgn` checkpoint container — format spec

> **Correction (2026-10-10, see analysis/closure/hgn-href.md).** `HREF` is not an in-table reference to a side file. It is the magic of the `halogen-tools ppl` reference-dump file (`--ref-out`, `--ref`). No .hgn record, dtype or tensor uses it; the n-gram table is located by other means (closure/hgn-href.md section 4).

Reverse-engineered from `halogen-tools` 0.17.3 (`verify`/`inspect` paths,
Ghidra decompiles `work/hgn/gf_0x1847710.c` (verify), `gf_0x18441a0.c`
(inspect --json), `gs_0x19af350.c` (flash_serve's GGUF→.hgn repack driver))
and validated by a Python reimplementation (`work/hgn/hgnpy.py`) that
round-trips files the vendor binary accepts (`work/hgn/roundtrip.py`, 3/3
PASS) plus 169 crafted containers run through the vendor `verify`
(`tmp/verify_matrix.tsv`, summarized below).

A third party can write a reader from this document alone.

## 1. Top-level layout (all integers little-endian)

```
offset 0        header (104 B)
offset 4        pad, [4, table_off)
table_off       tensor table, n_tensors × 160 B (table_off == 104 in practice)
                pad, [table_off + 160·n, data_off)
data_off        payload area: tensor payloads in table order, each 64 B aligned
                (payload_off[i+1] ≥ payload_off[i] + nbytes[i])
                tail: [last payload end, file_bytes)
```

Pad regions are ZERO BY BUILDER CONVENTION, not verify-enforced: crafted
files with 0xDEADBEEF in the header pad, the post-table pad, and the tail
all PASS vendor `verify` (`tmp/padprobe/pad-{hdr2,posttbl,tail3}.hgn`);
`inspect` reports the observed state as the informational `pads_zero`
field. Writers should zero pads (the vendor builder does; consumers must
not read meaning into pad bytes).

### Header (104 B)

| off | size | field | notes |
|---|---|---|---|
| 0 | 4 | magic | `HGN1` |
| 4 | 4 | version | u32; 1 and 2 accepted by 0.17.3's verify (0, 3, 7, 24, 25 rejected: "unsupported version") |
| 8 | 8 | n_tensors | u64 |
| 16 | 8 | table_off | u64; must be 104 ("header: table offset 0 (expected 104)") |
| 24 | 8 | data_off | u64; must be 64 B aligned and ≥ table_off + 160·n_tensors ("data offset not 64 B aligned", "data offset inside the table"); `(data_off − table_off) / 160 ≥ n_tensors` ("the table does not fit in the file") |
| 32 | 8 | file_bytes | u64; must equal the real file size ("file_bytes 104 but the file is 128") |
| 40 | 64 | model_id | UTF-8, NUL-padded, ≤ 63 chars |

`header_sha256 = sha256(file[0:data_off])` — header + tensor table + the zero
pad between them. `inspect` prints it; it is the fingerprint the disk prompt
cache's lineage uses (per report.md §2).

### Tensor table entry (160 B)

| off | size | field | notes |
|---|---|---|---|
| 0 | 96 | name | UTF-8, NUL-terminated, ≤ 95 chars |
| 96 | 4 | dtype | u32 code, §2 |
| 100 | 4 | rank | u32, ≤ 4 ("entry 0 (test) has rank 117835012 (at most 4)") |
| 104 | 32 | dims | u64[4]; only the first `rank` entries meaningful |
| 136 | 8 | payload_off | u64, ABSOLUTE file offset (probe e1/e3: relative offsets fail "points past the end of the file") |
| 144 | 8 | nbytes | u64 payload length; must equal the per-format size implied by dims (§3) |
| 152 | 4 | xor32 | u32 checksum of the payload, §4 |
| 156 | 4 | qparam | u32 quant group size (0 for plain dtypes; format-specific constraints, §3) |

Payloads sit at `data_off`, each start 64 B aligned, in table order.
`payload_off[i+1] ≥ payload_off[i] + nbytes[i]`, gaps zero.

## 2. dtype codes

Recovered from the `inspect` family table + kernel names + verify messages:

| code | name | code | name | code | name |
|---|---|---|---|---|---|
| 0 | bf16 | 8 | i4l | 20 | q8_0 |
| 1 | f32 | 10 | fp8g | 21 | q5_0 |
| 2 | f16 | 11 | q8s32 | 22 | q5_1 |
| 3 | i32 | 12 | q6k | 23 | i4r |
| 4 | i64 | 13 | iq4nl | 24 | q6g64 |
| 5 | q4c | 14 | q4a | 16 | ht |
| 6 | fp8r | 15 | q5a | 7 | q8g64 |

(Codes 9, 17-19, 25+ are unobserved; 25 is rejected as a *version*, not a
dtype. `inspect` groups names into families: attention, FFN (mlp), embedding,
lm head, router, PLE/n-gram, norms/conv/misc.)

## 3. Per-format size rules (measured from verify's "the format implies X B")

For dims `[N, K]` (row-major, K contiguous) — measured by crafting
`nbytes=0` files and reading the implied size:

| dtype | implied bytes | shape constraints (verify messages) |
|---|---|---|
| bf16, f16 | N·K·2 | — |
| f32, i32 | N·K·4 | — |
| i64 | N·K·8 | — |
| q4c | N·K/2 + N·K/1024·(32·4) = N·K/2 + N·K/32 (73,792 for [128,1024]) | rank ≥ 2; interleaved form needs K % 1024 == 0; planar variant (k_deq_q4cp) needs K % 32 == 0 |
| fp8r | N·K + 2·N (fp16 row scales) | rank ≥ 2 |
| q8g64 | N·K + 2·N (fp16 scale per 64) | rank ≥ 2, K % 64 == 0 |
| q8s32 | N·K + 2·N (fp16 scale per 32; 34,816 for [128,256] = 32,768 + 2·128·8? no: 32,768 + 2,048 = 34,816 → 2·N·8 = fp16 scale per 32 groups: N·K/32·2 = 1,024; measured 2,048 → fp16 scale + fp16 zero per 32) | rank ≥ 2, K % 32 == 0 |
| q6k | N·K·6/8 + 2·N (26,880 for [128,256] = 24,576 + 2,304) | K % 32 == 0 (k_lq6k groups 32) |
| iq4nl | N·K/2 + N·K/32·2 (18,432 for [128,256] = 16,384 + 2,048) | K % 32 == 0 |
| q4a | N·K/2 + N·K/32·2 (20,480 for [128,256] = 16,384 + 4,096 → fp16 scale per 32 = 2·N·K/32/2? measured ratio 5/4: N·K/2 + N·K/16·1 = fp16 scale per 32) | K % 32 == 0 |
| q5a | N·K·5/8 + N·K/32·2 (24,576 for [128,256] = 20,480 + 4,096) | K % 32 == 0 |
| ht | engine-read forms only; qparam ∉ {engine set} → "ht qparam N is a form the engine does not read (codebook, tile order or bits)" | HT only on trunk linears + routed experts ("an HT checkpoint carries HT only on the trunk's linears and the routed experts (not the draft head, the router, the embedding, the mixers or PLE)") |
| q8_0 | N·K + 2·N·(K/32) (34,816 for [128,256] = 32,768 + 2,048) | K % 32 == 0 |
| q5_0 | 22,528 for [128,256] = 20,480 (5b) + 2,048 (fp16 scales) | K % 32 == 0 |
| q5_1 | 24,576 for [128,256] = 20,480 + 4,096 | K % 32 == 0 |
| i4r | 865,075,200 for [512,1280,2560] = 512·1280·2560·4.25/8 (4.25 bpw: 4b nibbles + fp16 group-128 scales + per-row term) | qparam must be 128; rank 3 with gate_up [512,1280,2560] / down [512,2560,640] ("i4r needs group 128 and the routed experts' gate_up [512, 1280, 2560] / down [512, 2560, 640]") |
| q6g64 | 26,624 for [128,256] = 24,576 + 2,048 | rank ≥ 2, K % 64 == 0, qparam == 64 |
| i4l | K/2 + K/qparam·2 per row (qparam 32 → 18,432; 64 → 17,408; 128 → 16,896 for [128,256]) | K % qparam == 0 ("i4l needs K % qparam == 0") |
| fp8g | N·K + 4·ceil(N·K/32768) (32,772 for [128,256] = 32,768 + 4) | one f32 global scale per 32,768 elements |

Rank-1 tensors (norms) are checked against the model geometry when the
model_id matches a known checkpoint: "embed_tokens.weight: dims [8,64], the
model's geometry says [248320,2560]". Files with unknown model_id get
"geometry checked on 0 tensors (N with no rule)".

## 4. Checksums

- **xor32**: XOR of the payload read as little-endian u32 words; the tail
  (nbytes % 4 bytes) is zero-padded to a word. Verified three ways: crafted
  files with xor-of-bytes, sum-of-u32le, sum-of-u32be, xor-of-u32be all FAIL
  with "checksum X on disk, table says Y" while xor-of-u32le PASSES
  (`ck3-xor-*.hgn` in the matrix).
- **sha256**: `inspect` prints per-tensor payload sha256 and the header
  sha256; `verify` checks payload checksums ("checksums ... OK").
- Codebook/scale checks for quant formats: "codebook not sorted ascending at
  entry N", "scale not finite at payload byte N", "global scale not finite
  and positive", "have a scale d whose -16·d is not an fp16 (non-finite or
  out of range); refusing rather than rounding it".

## 5. verify / inspect semantics

`verify FILE.hgn [--threads N]`:
- PASS line: `PASS %s: %zu tensors, %.2f GiB, v%llu, model_id %s; header,
  table, layout, payload sizes, checksums, codebooks and scales OK; geometry
  checked on %zu tensors (%zu with no rule)` (exit 0).
- FAIL line: `FAIL %s: %zu problem%s; the first: %s` (exit 1) + one line per
  problem. Single problems print without the count.
- Checks, in order: magic/size → header fields (version, table_off,
  data_off alignment/containment, file_bytes) → per-entry (rank ≤ 4, name,
  payload in bounds, nbytes == implied size, xor32) → quant codebooks/scales
  → geometry vs model_id. (Pad contents are NOT checked; see §1.)

`inspect FILE.hgn [--json] [--no-hash] [--threads N]`:
- JSON: path, model_id, version, n_tensors, file_bytes, families[]
  (class/format/count/params/bytes/bpw), total_bytes, aux_bytes, params,
  header_sha256, header_bytes, pads_zero, tensors[] (name, dtype, qparam,
  dims, nbytes, xor32, sha256).
- Family classes observed: `attention`, `FFN (mlp)`, `norms / conv / misc`
  (+ embedding / lm head / router / PLE per the repack driver's naming).

## 6. Head-file (MTP) and HREF

- The MTP draft head ships as its own `.hgn` (`qwen38-flash-next-mtp.hgn`,
  named by `HALOGEN_MTP_HEAD`). It must carry `mtp.*` tensors ("the MTP head
  file carries no mtp.* tensors"); its header, table, and payload checksums
  are verified against their own file ("the head file's payload does not
  match its own checksum"). A llama.cpp MTP draft GGUF is refused by name.
- `HREF` (4-byte tag, two uses in halogen-tools' .text at 0x1725112 /
  0x1725afc) is the in-table reference encoding for payloads stored in a
  side file (the 47.7 GiB FP8 n-gram lookup table is "read in place" by the
  repack driver, never copied into the .hgn). The exact HREF record layout
  stays an open item (bounded: two call sites, r2 `agi` output in
  `tmp/href_fns.asm`; the Ghidra pass for those functions is queued in
  Phase 6's evidence bundle).

## 7. GGUF → .hgn repack (flash_serve 0x19af350)

- Gate: `gguf: %s is not a GGUF`. Requires the n-gram table tensors
  (`layers.1.ple.ngram_embedding.weight` / `per_layer_token_embd.weight`);
  missing is fatal ("Nothing was written").
- Layout from 0x80, 64 B alignment per tensor, sparse `ftruncate`, parallel
  fill on `HALOGEN_GGUF_THREADS` threads (default 8, clamp 1..64).
- Banner: `repack: %zu tensors, %.2f GiB -> %s on %d threads%s%s` with
  ", with the table" / ", NO head" / ", head from <path>".
- Losslessness: F16 tensors are accepted where the destination is bf16 only
  if every value is bf16-clean ("the F16 values do not all fit bf16 ... this
  read is not lossless").

## 8. Validation record

- 169 crafted containers (`work/hgn/files/`, tarred
  `analysis/hgn-testfiles.tar.xz`) run through vendor `verify` + `inspect`:
  9 PASS (the valid ones: t1, k3, rt-valid, rt-badpad*, rt-f32*, vt1, vt2,
  ck3-xor-*), 160 FAIL with the expected reason classes (table overflow,
  version, alignment, file_bytes, rank, checksum, geometry, size-implied).
  Full matrix: `tmp/verify_matrix.tsv` (also in the evidence bundle).
  rt-badpad.hgn PASSES verify by design: its corruption is a non-zero pad
  byte, which verify does not check (see §1); `inspect` flags it via
  `pads_zero:false`.
- 3/3 round-trips PASS (`work/hgn/roundtrip.py`): hgnpy.build → vendor
  verify PASS → hgnpy.parse → rebuild byte-identical. Files:
  `work/hgn/rt2/rt-{a,b,c}.hgn` (multi-dtype, i32/i64/tail-xor, 64 B-gap
  alignment).
