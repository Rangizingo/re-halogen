# U03 and U01 from shipped NPU files (decider-0.8b)

Date 2026-10-10. Data: the vendor's NPU device set for `decider-0.8b`, fetched with Peter's approval from `peonist-ai/halogen-npu-decider-0.8b` at revision `38adff38bebe32151b2c32b1cb5a18da4ca516f1` (the revision pinned in `image/npu-models.txt`). Only `devices/` was fetched (8 files, 95,450,501 bytes); the 2.27 GB weight file was not. All 8 files pass `sha256sum -c` against the hashes in `image/npu-models.txt` (`analysis/evidence/npu-decider-0.8b/elf-section-groups.txt`, first block). The ELFs stay on Ai-core under `~/re-halogen/work/npu/elf/devices/` (gitignored size); the manifest is committed as `analysis/evidence/npu-decider-0.8b/devices.hnpm`.

## U03 `main:<graph>` name set: CLOSED (static: files plus decompile dataflow; no runtime `main:` string observed)

1. The manifest has 78 `kernel` records (`awk '{print $1}' devices.hnpm | sort | uniq -c`), 30 in bucket u0 and 12 each in u1 to u4; 33 distinct names in all. u0: `k0` to `k21` without the plain `k4` to `k11`, plus `k4_64` to `k11_64` and `k4_128` to `k11_128`. u1 to u4 share the same 12: `k0` to `k3`, `k17` to `k24`.
2. Each `devices/uN.elf` is an XRT full ELF (`.note.xrt.configuration`, `.ctrltext.N`, `.pdi.N`, `.group.N` sections). Its ELF section groups (`readelf -gW`) are signed by `d0`..`d10` and by kernel names. Proven: per bucket, the set of non-`d` group signatures equals the set of manifest kernel names for that bucket, 5 of 5 buckets, `diff` empty (30/30, 12/12, 12/12, 12/12, 12/12). Lists: `analysis/evidence/npu-decider-0.8b/elf-section-groups.txt`.
3. How the engine picks a name. `FUN_0013f420` (`work/decomp/halogen-npu/0013f420.c:19-47`): when the bucket kind is `o` (manifest `bucket kind=o`), it looks the requested name up in the kernel-record map (`param_3 + 0x178`, `:22`); a miss returns the name unchanged (`:23-27`); a hit reads the record id at `+0x110` (`:29`); if it equals the last id used (`param_3 + 0x2c4`, `:30`) and the string at `+0x80` is not empty, that string is returned (`:31-33`), otherwise the string at `+0xa0` (`:35-36`); the last id is updated at `:40`, and an empty pick falls back to the requested name (`:42-45`). The manifest kernel records carry exactly two such fields, `ro=` and `keep=` (for example `kernel bucket=u0 name=k0 ... ro=k17 keep=k18`), and `variant` records map `x0_{T}` style names to `ro`/`keep` pairs. Which offset is `ro` and which is `keep` is inferred (field order in the record). Dataflow to the prefix: `00149720.c:53` calls `FUN_0013f420(&local_118, ...)` and never reassigns `local_118`; `00149720.c:187` passes `&local_118` to `FUN_00159fc0`; `00159fc0.c:53` calls `FUN_00159980(param_2, param_3)`; `00159980.c:95-96` writes `main` (`0x6e69616d`) and `:` (`0x3a`) and `:257` constructs `xrt::ext::kernel` with the string (all `work/decomp/halogen-npu/`).
4. So the set of `main:<graph>` names is `main:` plus the manifest kernel names of the active bucket (30 in u0, 12 in u1 to u4), all present as ELF groups. Plan records name the call order per layer type (`plan bucket=u0 type=linear_attention ops=tin:b19_{T},call:x0_{T},...`; `call:` targets seen: `k12`, `k2`, `k22`, `k3`, `x0_{T}` to `x3_{T}`).

Scope: one model (decider-0.8b, family `qwen35_hybrid`). Other NPU models ship their own `devices/` with their own names; the mechanism is the same (manifest plus ELF groups), the names differ per model.

## U01 HNPB header and per-blob fields: per-blob part CLOSED, header +0x08 still PARTIAL

From `devices/consts.hnpb` (1,645,016 bytes, 15 blobs; `analysis/evidence/npu-decider-0.8b/consts-hnpb-header.txt`):

- Per-blob `+0x50` tag: 4 on 5 blobs whose size is 4 bytes per element, 1 on 10 blobs at 2 bytes per element. The manifest declares the matching buffers as `b9:f32` and `b10:bf16`, `b11:bf16` (13 of 13 slot uses each; blob `u0/w0` is buffer `b9`). So tag 4 is f32 and tag 1 is bf16. Proven for these two values only.
- Per-blob `+0x54`: 1 on all 15 one-dimensional blobs, with `+0x58` the element count. Matches the earlier "ndim of a 1-D blob in bulk packs" reading.
- Per-blob `+0x80..+0x9f`: the SHA-256 of the blob bytes, 15 of 15 (measured). Earlier docs called these "four u64, never read".
- Per-blob `+0xa0`: XOR32 of the blob, 15 of 15 (measured), as stated before.
- Header `+0x08` is 2 in this shipped file. The writers in `halogen-npu` store 3 (bulk, `0016eb50`) or 1 (streamed, `0017a920`) per `closure/npu-closure-r2.md`. So the shipped `consts.hnpb` was written by a tool other than `halogen-npu` (inferred), and the reader ignores the field (`00167960.c:350`). Its meaning stays inferred: a producer tag.

## Not done

- No `.hnpw` header was read (Peter limited the fetch to `devices/`, no weights).
- The NPU programs themselves (AIE code in `.ctrltext`/`.pdi`) were not disassembled.
