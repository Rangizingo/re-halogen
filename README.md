# re-halogen

A reverse-engineering workspace for **halogen-flash-server 0.17.3** (Peonist, LLC): an OpenAI-compatible inference server that runs Qwen3.8-Flash-Next on AMD Strix Halo (gfx1151). The vendor ships it as closed binaries. This repo takes them apart and writes down how they work, with evidence for every claim.

There is no app to build here. The output is documentation, plus small tools that prove the documentation right.

## What you get

- **How the server works**, end to end: request in, tokens out (`analysis/request-trace.md`, 31 steps).
- **How to build a compatible server yourself** (`analysis/rebuild-guide.md`).
- **The `.hgn` model file format**, with a parser and builder (`analysis/hgn-format.md`, `scripts/hgnpy.py`).
- **Every GPU kernel rewritten in HIP**: all 1,738 kernel instances in the engine, each one checked to produce bit-for-bit the same output as the vendor's (`rewrite/`).
- **A ledger of unknowns**: 27 open questions, 25 closed, 2 left open on purpose (`analysis/unknowns.md`).

## Where to start reading

1. `start-here.md`: one page on what's done and where everything lives.
2. `analysis/report.md`: the findings.
3. `analysis/rebuild-guide.md`: if you want to build something that talks to the same clients and reads the same files.
4. `PLAN.md`: the phase plan and the log of every work session.

Claims in the docs carry a label: **proven** (the cited line or byte was read), **inferred** (reasoned from structure or names) or **observed** (seen in a live run). Labels are never upgraded without new evidence.

## What's in the repo

| Path | What it is |
| --- | --- |
| `binaries/` | The three vendor programs: `flash_serve` (the GPU engine), `halogen-tools` (verify, inspect and repack model files), `halogen-npu` (the NPU engine). Linux x86-64. |
| `image/` | Vendor files pulled from the container image: entrypoint, Python front end, EULA. |
| `analysis/` | The writeups. `analysis/function-map*.tsv` name every function in the three binaries. |
| `rewrite/` | The HIP rewrite of every GPU kernel, its test specs, result tables and the constant audit. |
| `scripts/` | Extraction and decoding tools. |
| `tests/` | pytest suite for the Python tools. |

Not in git (too big, regenerate them): `work/` (carved code objects, pulled releases, test `.hgn` files), `tmp/`, and the full kernel listings `kernels-disasm-v2/` (`llvm-objdump -d --mcpu=gfx1151` 23.1.3 on the code objects in `work/fatbin/`). Use the v2 listings; the older `kernels-disasm/` (llvm 21) prints about 212,000 instructions as raw `.long` words.

## How to use it

### Run the tests (any machine with Python 3.11+)

```bash
python3 -m pytest -q tests          # 145 pass, 4 skip
```

All 4 skips are HT weight decode tests that need vendor data, see below.

### Check the HT weight decoder against real vendor data

The decoder reads the engine's HT and i4r quantized weights. To check it, fetch about 497 MB of slices from the vendor's published model files (HTTP range reads, no full download):

```bash
bash scripts/fetch_ht_testdata.sh <dir>
python3 -I scripts/ht_decode.py <dir>          # prints decode scores against q4c references
HT_TESTDATA=<dir> python3 -m pytest -q tests  # the 4 skipped tests now run
```

### Read or build `.hgn` model files

`scripts/hgnpy.py` is a Python library (`parse`, `build`, `xor32`). Its docstring is the format spec; the full writeup is `analysis/hgn-format.md`.

To check our understanding against the vendor's own checker, `scripts/verify_matrix.sh` runs `halogen-tools verify` and `inspect` on 169 crafted files in `work/hgn/files/` (expect 9 PASS and 160 designed FAIL). It needs Linux x86-64, and the path at its top points at the analysis host's checkout.

### Compare a kernel rewrite against the vendor kernel (needs the GPU host)

This runs on a gfx1151 machine with ROCm (here, the host `ai-core`). Copy `rewrite/` there, build once, then run a spec:

```bash
rsync -a --exclude __pycache__ rewrite/ ai-core:re-halogen/rewrite/
bash rewrite/harness/build.sh
python3 -I rewrite/harness/compare.py rewrite/specs/<spec>.json --kernel <mangled name> --seeds 1,2,3 --sizes 4096,100003
```

Each case prints `MATCH` (identical output bytes) or `MISMATCH` with the first differing element, and is logged to `rewrite/match-table.md`. Find the spec and kernel name for any instance in that table. `compare.py --help` covers the rest (tolerances, priming, table path).

How the rewrite set is organised:

- `rewrite/kernels/` holds one HIP file per kernel family.
- A bug fix is a new copy (`<family>_c.hip`, then `_d`), never an edit of the original. `rewrite/superseded.tsv` lists which file replaced which; the latest file in each chain is the one that counts.
- `rewrite/reports/` has one report per work group; `rewrite/reports/orchestrator-checks.md` holds the independent re-runs and negative controls.

### Run the constant audit

`rewrite/harness/const_audit.py` checks that every number the vendor kernel uses (floats, and integer bounds such as loop limits and clamps) also appears in our rewrite, or has a written reason that cites both listings. It compiles the rewrites on the GPU host over ssh:

```bash
python3 -I rewrite/harness/const_audit.py --instances all --out rewrite/const-audit.md --jobs 4
```

Current result: 1,738 of 1,738 `AUDIT OK`. Reasons live in `rewrite/const-audit-exceptions.tsv` and `rewrite/const-audit-exceptions.d/`. Its limit, stated plainly: integers that aren't bounds (address offsets, masks, shifts) are accepted by role, so for those the bit-for-bit compare results are the evidence.

## What's still unknown

Two items, both by decision (`analysis/unknowns.md`):

- **U06**: changes in the 0.14.x and 0.15.x patch releases. Those releases were never pulled.
- **U05**: whether a future release flips one dflash2 setting. No shipped release can show that.

## Rules if you work here

- Never run `flash_serve`, and run `halogen-tools` only as `verify` or `inspect` on files made in this workspace.
- A test engine run once deleted the live service's 60 GiB prompt cache. Test engines get their own cache directory and `HALOGEN_CACHE_PRUNE_OLD=0`; they never mount `/srv/halogen-cache`.
- Ghidra rejects paths with a dot-folder in them, and it loads the engine at base `0x100000` (Ghidra address = file address + `0x100000`).

## License

EULA v1.0 section 3 permits unrestricted reverse engineering (`image/EULA.md`). Original material in this repo is CC0 (`LICENSE`). Vendor binaries and image files keep their own licenses. The repo is private for now.
