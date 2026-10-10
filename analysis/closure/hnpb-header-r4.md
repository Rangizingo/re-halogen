# U01 round 4: any other producer or consumer of HNPB header +0x08

Scope: everything shipped or pulled in this workspace that could write or read an HNPB v1 header: `binaries/{flash_serve,halogen-tools,halogen-npu}`, `image/*` (entrypoint.sh, serve_api.py, halogen_tools.py, npu-models.txt), `work/versions/{0.11.10..0.17.2}/*`, and the decompiles in `work/decomp/`. Static only, nothing executed. Address convention: r2 VMA = file offset here; Ghidra = VMA + 0x100000. "proof" = read from bytes or code; "inference" is labelled.

## Verdict: CLOSED (revised in section 7; OPEN-BOUNDED before the vendor headers were read)

No shipped or pulled artifact contains a producer that stores 2 at HNPB header +0x08. The reader gives the field no behavior. Closing needs one of the files in section 6, none of which is in this workspace.

## 1. Producers outside halogen-npu: none (proof)

Byte scan (`python3 -I`) for the magic `48 4e 50 42`:

- `binaries/flash_serve`: 0 hits. `binaries/halogen-tools`: 0 hits. `binaries/halogen-npu`: 5 hits, at file offsets 0x679f7, 0x70856, 0x7aae8, 0xefc01, 0xf1ddb.
- `grep -aci HNPB` over every file in `image/` and `work/versions/*/*`: only `image/npu-models.txt` matches (a comment line 17 and `devices/consts.hnpb` file records at lines 33, 44, 66). `entrypoint.sh`, `serve_api.py`, `halogen_tools.py` (all versions) have no `HNPB`/`hnpb` string.
- `grep -rn 42504e48 work/decomp/halogen-tools work/decomp/flash_serve work/decomp/flash_serve_0172`: no output.
- The only script call that creates an HNPB-family file is `image/entrypoint.sh:1171`: `"$bin" convert "$p" "$NPU_DIR/$own/devices" "$out" --name "$name" --source "$p" --if-stale`, i.e. `halogen-npu convert` writes the `.hnpw`. `entrypoint.sh:1083` and `:1137` only name `<id>.hnpw` for size/sha checks.
- `work/versions/*` hold `flash_serve` plus scripts, no `halogen-npu`. `grep -ac "halogen-npu" work/versions/<v>/flash_serve` is 0 for 0.15.0, 0.16.4, 0.17.0, 0.17.2. The entrypoints of 0.16.4, 0.17.0, 0.17.2 each have 10 `halogen-npu|hnpw|hnpb` lines (e.g. `work/versions/0.16.4/entrypoint.sh:1133`, `:1153`), so the NPU engine shipped in those images, but its binary was not pulled. 0.15.0 and older entrypoints have no NPU lines. Older halogen-npu builds therefore cannot be checked here. This is a bound, not a negative.

## 2. The three code sites that store or test the HNPB magic in halogen-npu (proof)

The other two magic hits (0xefc01, 0xf1ddb) are error strings (`": not an HNPB v1 file"`, `": too short for an HNPB header"`).

- Reader, VMA 0x679f5: `cmp dword [rbx], 0x42504e48 ; 'HNPB'`, then 0x67a05 `cmp r12d, 1` (version at +4), then 0x67a0f `mov eax, dword [rbx + 8]` and 0x67a16 `mov dword [r14 + 0x10], eax`. Decomp `00167960.c:348-350`. No compare, switch or range test on the +0x08 value. A 2 loads fine.
- Bulk writer, VMA 0x70838 to 0x7085a: `mov rdx, qword [0x000f4c28]` ... `mov dword [rbx], 0x42504e48` ... `mov qword [rbx + 4], rdx`. Decomp `0016eb50.c:1148-1151`. Bytes at file 0xf4c28 = `01 00 00 00 03 00 00 00`, so +0x08 = 3. A disp32 scan of the file below 0xf0000 finds exactly one reference to 0xf4c28 (0x70838).
- Streaming writer, VMA 0x7aae5: `mov dword [rsp], 0x42504e48 ; 'HNPB'`. Decomp `0017a920.c:115,118-119` stores `DAT_001f4bf0` at +4..+11. Bytes at file 0xf4bf0 = `01 00 00 00 01 00 00 00`, so +0x08 = 1.

New in r4:

- The pair at 0xf4bf0 is not HNPB-specific. The disp32 scan finds 8 code references (0x34009, 0x3596d, 0x6d4de, 0x71a65, 0x7293f, 0x73c47, 0x73f08, 0x7ab0b). The decompiled ones are object constructors of the shape `_Znwm(n); p[1] = DAT_001f4bf0; *p = &PTR_FUN_...` (`0016d490.c:17-20`, `00133090.c:249-250`, `00170ea0.c:465-467`, `001734f0.c:275`), which looks like a `std::make_shared` control block (vptr, use_count 1, weak_count 1). Inference: the streaming writer's +0x08 = 1 is the compiler reusing the identical `{1,1}` constant for source `{version = 1, flavor = 1}`. The stored value (1) is proof; the reason for it is inference. The bulk path has its own constant `{1,3}`.
- No instruction stores a literal `{1,2}` pair into an HNPB header. The bytes `01 00 00 00 02 00 00 00` occur 10 times in the file (first at 0xf44a4, 0xf4504, 0xf4964, 0xf55e1, 0xfa384, 0x11376c), inside longer `1,2,3,4` int tables. r2 `axt` finds no reference to 0xf44a4, 0xf4504, 0xf4964 and the writers above use 0xf4bf0 and 0xf4c28 only. Not a producer.

## 3. Writer reachability (proof, call sites re-read)

- Bulk `FUN_0016eb50`: `00171ca0.c:81` (`cVar9 = FUN_0016eb50(local_98,local_e8,&local_128);`) and `00172600.c:179` (`FUN_0016eb50(*(undefined8 *)(lVar18 + 0x28),*param_2 + 0x60,&local_1e8)`).
- Streaming `FUN_0017a920`: `001a7be0.c:1168` (`cVar4 = FUN_0017a920(local_390);`), the `convert` path that `entrypoint.sh:1171` drives (the `convert` to `001a7be0` link is from r3, inferred from the verb table).
- So a user-run convert yields 1, a cache dump yields 3, nothing in the binary yields 2.

## 4. Consumers of the loaded value (container +0x10): none found

- Reader callers (r3): `00170ea0.c:79`, `00145eb0.c:108,116`, `001a7be0.c:1057`, `001d6480.c:54`. Container instances: `param_1+400` and `+0x1f0` (`0013b150.c:156-157`, `00123250.c:209-210`, `0010afa2.c:25-26`), `param_1+0xa0` (`001d2540.c:185`, `001d30b0.c:1523`, `00116b74.c:29`), stack slots `local_98` in `00170ea0` and `local_390` in `001a7be0`.
- New in r4: other users of the lookup helper `FUN_00162ec0` (`0016dc30.c:14`, `001734f0.c:197`, `001b2620.c:50`, `001ba230.c:1313`, `001b43c0.c:289`, `00144430.c:476`, `0014a6f0.c:701`, `001d30b0.c:1523`) would read the field at `param_1+0x1a0`, `+0x200` or `+0xb0`. `grep -n "0x1a0\|0x200)\|param_1 + 0xb0"` on those files finds no container-field read. The hits are other things: `001734f0.c:536` and `0014a6f0.c:395` (0x340/0x348 offsets), `0014a6f0.c:1224` (`param_3 + 0x1a0`, another struct), `001d30b0.c:1368,1461` (numeric 0x200). `FUN_00162ec0` itself reads container +0x38 and +0x40 only (`00162ec0.c:19-21`).
- Result: nothing found that branches on the loaded value. Limit: grep over the container users, not data flow over all functions, so "no consumer anywhere" remains an inference.

## 5. What 2 could mean (inference only)

- The reader ignores it, so 2 is a producer-side tag. Observed values: 1 (convert), 3 (cache dump), 2 on the vendor-published `devices/consts.hnpb` (`analysis/evidence/npu-decider-0.8b/consts-hnpb-header.txt`: `484e 5042 0100 0000 0200 0000`). That fits a small enum of pack origins, but code with only 1 and 3 cannot say which origin 2 names.
- The producer of the shipped `consts.hnpb` is a vendor build step that is not in any pulled image layer (inference): the in-binary `convert` writes weights (`.hnpw`), and the `devices/` set is published on HF repo `peonist-ai/halogen-npu-decider-0.8b` (`image/npu-models.txt`).

## 6. Residuals and what would close it

1. First 64 bytes of any `<id>.hnpw` pinned in `image/npu-models.txt` (for example decider-0.8b at revision `38adff38bebe32151b2c32b1cb5a18da4ca516f1`), by HTTP range read. If it is 2, then 2 is the vendor published-pack tag and 1 is the local convert tag. Needs Peter's go (network fetch outside the repo).
2. `consts.hnpb` headers of the other pinned models (`image/npu-models.txt:44,66`): if they also hold 2, it is the vendor tool tag, not a per-model value.
3. A halogen-npu binary from 0.16.4, 0.17.0 or 0.17.2 (shipped per the entrypoints, not pulled). Only useful if 1 and 2 are inconclusive.
4. The vendor build tool itself.

## Reproduce

```
cd /Users/peterb/Documents/AI/re-halogen
python3 -I -c "import re;[print(b,[hex(m.start()) for m in re.finditer(b'HNPB',open('binaries/'+b,'rb').read())]) for b in ('flash_serve','halogen-tools','halogen-npu')]"
for f in image/* work/versions/*/*; do [ -f "$f" ] && grep -aqi hnpb "$f" && echo "$f"; done
grep -rn 42504e48 work/decomp
python3 -I -c "d=open('binaries/halogen-npu','rb').read();print(d[0xf4bf0:0xf4bf8].hex(),d[0xf4c28:0xf4c30].hex())"
ssh ai-core 'cd ~/re-halogen && r2 -q -e bin.relocs.apply=true -c "pd 14 @ 0x679ee; pd 14 @ 0x70838; pd 4 @ 0x7aae5" binaries/halogen-npu'
grep -n "DAT_001f4bf0\|DAT_001f4c28" work/decomp/halogen-npu/*.c
```

## 7. Orchestrator addendum: residuals 1 and 2 fetched (2026-10-10)

The orchestrator range-read the first 16 bytes of every HNPB file pinned in `image/npu-models.txt` (6 weight files, 3 consts files; `curl -r 0-15` against `huggingface.co/peonist-ai/halogen-npu-<id>/resolve/<revision>/<path>`, HTTP 206 each). Saved as `analysis/evidence/hnpb-headers-vendor.txt`.

| file kind | files | header +0x08 |
|---|---|---|
| `<id>.hnpw` (weights) | decider-0.8b, qwen3-embedding-0.6b, qwen3-reranker-0.6b, qwen3guard-gen-0.6b, qwen3.5-2b, flux2-klein-4b | 1 in all 6 |
| `devices/consts.hnpb` (device constants) | decider-0.8b, qwen3-embedding-0.6b, qwen3.5-2b | 2 in all 3 |

With section 2: the in-binary `convert` path (weights, `FUN_0017a920`) writes 1, the bulk path (`FUN_0016eb50`, called at `00172600.c:179` inside the one handler for QHN, QLED, LOOKUP, EDUMP, ELOAD, EFREE (npu-verbs-r3b.md); EDUMP as the caller is inference) writes 3. The field is a pack-kind tag: 1 = weight pack, 2 = device-constants pack, 3 = bulk dump (proof for the values per file kind; "kind tag" as the name is inference). The reader ignores it (section 2), so it has no runtime effect. The tool that writes kind 2 is the vendor's build step and is not shipped; that does not change the meaning.

Revised verdict: CLOSED.
