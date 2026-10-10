# Live engine run 5: Tap capture with SNAP fields and an on-disk cache record, flash_serve 0.17.3 on Ai-core (2026-10-10)

What ran: the run 4 setup (`runtime-run-0.17.3-r4.md`) with three changes. The probe (`scripts/engine_probe7.py`) sends the SNAP, SNAP2 and SNAP3 wire fields that run 4 never sent, following the recipe in `closure/dnc-scan3-r6.md` ("Probe recipe"). The shim (`scripts/capture/kargs_shim.c`) also dumps the first 64 bytes of each Tap region after the kernel (`post_head`). The run script (`scripts/engine_run_r5.sh`) stops the engine with `podman stop -t 120` (SIGTERM) instead of `rm -f`, and lists the test cache before and after that stop. Own cache dir `work/testcache-r5`, `HALOGEN_CACHE_PRUNE_OLD=0`, `HALOGEN_DN_NORM_FOLD=1`, `HALOGEN_TAP_FROM_SCAN=1`, `HALOGEN_PROMPT_CACHE=2` from the service env. Evidence in `analysis/evidence/run-r5-*`. Observed unless marked inferred.

## Service window (Hermes down)

| stopped | engine stop (SIGTERM) | /health 200 | down |
|---|---|---|---|
| 22:56:29Z | 22:57:39Z, exit 143 at 22:57:40Z | 22:58:32Z | 2 min 3 s |

`run-r5-window.txt`. Health after restore (`run-r5-health-after.json`): `"status":"ok"`, version api/engine 0.17.2, `"match":true`, engine responds. Binaries: `run-r5-binary.sha256` (flash_serve af4f07bb..., same as runs 3 and 4). Every captured launch has `shim_err_pre` 0, `shim_err_post` 0 and rc 0.

## Requests (`run-r5-probe7-results.txt`)

| step | prompt | SNAP, SNAP2, SNAP3 | expected slots (r6 recipe) | D line `cached` field |
|---|---|---|---|---|
| `S1100_snap3` | 1100 fresh ids (salt 7) | 1000, 500, 800 | 6, 11, 14 | 0 |
| `S1100_snaponly` | 1100 fresh ids (salt 8) | 1000, 0, 0 | -1, -1, 14 | 0 |
| `S1200_resume` | first 1000 ids of step 1 + 200 new | none | -1, -1, -1 | 960 |

The engine log ends with `prompt cache 1 hits / 2 misses, 7 stores, 960 prompt tokens reused` (`run-r5-engine-0.17.3-r5.log:72`).

## U12: a Tap fires, and what it writes

Kernarg slots, all 108 `k_dnc_scan3x<2,4,1,0,1,1,0,1,0>` launches (grid 48x2x1, python over `run-r5-kargs.jsonl.gz`): `(6, 11, 14)` in 36 launches, `(-1, -1, 14)` in 36, `(-1, -1, -1)` in 36. That is one launch per DeltaNet layer per request (36 layers), and exactly the slots the r6 recipe predicts (`slot = target/64 - 1`, targets 448, 768, 960). Run 4 had `(-1, -1, -1)` in all 108 (`runtime-run-0.17.3-r4.md`). The pointers at kernarg `+0x90/+0x98/+0xa0` are the same three device addresses in every layer (`0x7ead35600000`, `0x7ead2e000000`, `0x7ead26a00000`): one 3 MiB buffer per tap index, reused layer after layer.

Payload: 144 Tap regions fired (36 x 3 + 36 x 1). Each is 3,145,728 B (`alloc` and `bytes` in the record), and the kernel rewrote 783,872 to 786,432 of its 786,432 u32 words (`words_changed`). 786,432 words = 48 heads x 128 x 128, the f32 recurrent-state shape of `closure/dnc-scan3-r4.md` section 1. The `post_head` bytes read as small f32 values (for example `0d777a39`, about 2.4e-4), consistent with that state (value reading inferred).

The payload is what the prompt cache stores. For each Tap, its full 3 MiB FNV-1a hash (the shim's `fnv`, `kargs_shim.c:103-109`) was recomputed over the slice of the matching on-disk `e<N>.blob` that starts with its `post_head` bytes:

| blob | tap index, request | full-hash match |
|---|---|---|
| lineage `0a85e255.../e448.blob` | k 0, step 1 | 36 of 36 |
| lineage `0a85e255.../e768.blob` | k 1, step 1 | 36 of 36 |
| lineage `0a85e255.../e960.blob` | k 2, step 1 | 36 of 36 |
| lineage `1f007f3a.../e960.blob` | k 2, step 2 | 36 of 36 |

So 144 of 144 Tap payloads appear byte for byte in the cache blobs (command under Reproduce). Layer 0 sits at blob offset 0; the layers follow at strides of 3,227,648 or 3,228,416 B, that is 3 MiB of DeltaNet state plus about 80 KiB more per layer (81,920 B = 10,240 x 4 x 2 matches a bf16 conv window of the 10,240 qkv channels by 4 taps; inferred, not checked byte-wise). A Tap is therefore the per-layer DeltaNet state at the snapshot position, written by `k_dnc_scan3x` during the prefill so that the store needs no second pass, and the store at that position copies it into the entry's O(1) snapshot. Step 3 resumed from the 960-token entry of step 1 (`cached` 960 in its D line).

This closes the U12 "Tap payload" item and r6 residual 1. What remains open in U12 is the scan3 template labels B, C, D, G (p7-12 rewrites).

## On-disk cache record (`closure/cache-sse-r5.md` residual)

`run-r5-cache-files-after-stop.txt` (87 files): one engine fingerprint, lineage `0a85e25566989863` with `e448`, `e768`, `e960` (`.rec` + `.blob`) and `r0`..`r38`, lineage `1f007f3ad6f7d3cc` with `e960` and `r0`..`r38`. Run 4 (force-killed) left only the fingerprint.

When records are written: the `.rec` and `.blob` files carry mtimes 22:57:27.8 to 22:57:27.9 (step 1) and 22:57:32.6 (step 2), before the stop at 22:57:39. The listing and the record heads before and after the graceful stop are identical (`diff` of `cache-files-before-stop.txt` and `-after-stop.txt` on Ai-core printed nothing). So records are written behind each request, not at shutdown; this corrects the run 4 inference ("records are written behind or at shutdown").

`run-r5-e448.rec` (2,136 B), decoded against the static layout in `closure/cache-sse-r5.md` section 1b:

| offset | value | static layout |
|---|---|---|
| 0 | `33 32 44 43` ("32DC") | magic 0x43443233 |
| 4 | 1 | version 1 |
| 8 | 448 | N |
| 12 | 448 | token array length = N |
| 16 | 0 | side array length 0 |
| 20 | 39 | F = number of `r<K>` files (r0..r38 on disk) |
| 24 | u64 116,389,652 | blob length; the `.blob` file is 116,391,936 B = that rounded up to 4 KiB (section 1c) |
| 32 | 39 x u64: 458,752 or 28,672 | required size of each `r<K>` (1,024 or 64 B per token x 448) |
| 344 | 448 x u32 | token ids; equal to the first 448 ids of the step 1 prompt (python check) |

344 + 448 x 4 = 2,136 B, the file size. The `r<K>` files hold 960 tokens (983,040 = 960 x 1,024 and 61,440 = 960 x 64 B): the rows only grow inside a lineage, and e448, e768 and e960 share one set (section 1d). Three records of one request landed in one lineage, which shows the continue-same-lineage branch that `cache-sse-r5.md` residual 2 left inferred; step 2's unrelated prompt got a new lineage. Residual 3 there ("`e<N>.blob` content is inferred to be the O(1) snapshot vector") is now observed for the DeltaNet part: the blob holds the 36 per-layer Tap states (section above). The CSTAT reply at the end (`C 6 698337912 116389652 ...`) repeats the blob length as its third field.

## Not covered

- The ~80 KiB tail of each per-layer blob slice and the roughly 190 KiB after the last layer are not decoded.
- The 288 `memcpy` records (3 MiB each) were not split per request.
- U12 labels B, C, D, G: kernel rewrites (p7-12, `rewrite/`).

## Reproduce

```bash
# on ai-core, after scripts/capture/build_shim.sh and selftest_shim.sh (prints selftest PASS)
bash ~/re-halogen/scripts/engine_run_r5.sh ~/re-halogen/work/run-r5
# slot counts
python3 -I -c "import gzip,json,collections;L=[json.loads(l) for l in gzip.open('analysis/evidence/run-r5-kargs.jsonl.gz','rt') if l.strip()];print(collections.Counter(tuple(x['tap']['slots']) for x in L if x['kind']=='launch'))"
# tap payload vs blob (on ai-core; blobs are container-owned, so podman unshare)
cat > /tmp/tapfull.py <<'E'
import json, sys, numpy as np
M = (1 << 64) - 1
def fnv(b):
    h = 1469598103934665603
    for x in np.frombuffer(b, dtype='<u8').tolist(): h = ((h ^ x) * 1099511628211) & M
    return '%016x' % h
la = [x for x in map(json.loads, filter(str.strip, open('run-r5/kargs.jsonl'))) if x['kind'] == 'launch']
blob, k, req = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d = open(blob, 'rb').read()
rs = [r for x in la[req * 36:req * 36 + 36] for r in x['tap']['regions'] if r['k'] == k]
ok = 0
for r in rs:
    o = d.find(bytes.fromhex(r['post_head']))
    ok += o >= 0 and fnv(d[o:o + (3 << 20)]) == r['post_hash']
print(ok, 'of', len(rs))
E
cd ~/re-halogen/work && podman unshare python3 -I /tmp/tapfull.py testcache-r5/2c917302cce55089/0a85e25566989863/e448.blob 0 0
```
