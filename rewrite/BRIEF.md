# Kernel rewrite agent brief (p7-12, U22)

One agent per group. Goal: a HIP rewrite of every kernel instance in your group's families, each matched against the vendor code object on Ai-core, one MATCH row per instance.

Repo (Mac, main checkout): `/Users/peterb/Documents/AI/re-halogen`. Always use absolute paths. Ai-core: `ssh ai-core`, repo at `~/re-halogen` (has `work/fatbin/*.hsaco`, `rewrite/harness/run_kernel` built, `hipcc`).

## Inputs

- Instances: `analysis/function-map-kernel-stubs.tsv` (column 2 mangled, column 4 family). `grep -P '\t<family>$'`.
- Kernel metadata (object, kernarg layout, LDS, wave size): `rewrite/harness/kernargs.json` (`python3 -I -c` with json; never cat it, it is big).
- ISA listings (Mac, not in git): `kernels-disasm/codeobj_<obj>.s`. Find a kernel with `grep -n '^<mangled>:' kernels-disasm/codeobj_<obj>.s` and its end with the next `^_Z.*:$` line. Read 50-150 line windows only; grep instruction classes first. Never read a whole listing or a kernel over 300 lines in one go. Per-kernel slices may already exist in `work/kernels*/`.
- Existing writeups for your family: `grep -ril <family> analysis/*.md analysis/closure/` (for example `analysis/kernels-*.md`).
- Worked example: `rewrite/kernels/scale.hip`, `rewrite/specs/k_scale_silu.json`, `rewrite/specs/k_scale_silu_wide.json`, harness docs at the top of `rewrite/harness/compare.py` and `rewrite/harness/run_kernel.cpp`.

## Outputs (yours only, file-disjoint)

- `rewrite/kernels/<family>.hip`: one file per family. Templates are fine; each rewrite kernel is `extern "C"` with a name you pass as `rewrite_kernel`. Every step of the math cites listing lines (`codeobj_<obj>.s:<line>`) the way scale.hip does.
- `rewrite/specs/<family>_*.json`: specs. Random inputs with seeds 1,2,3; free sizes at two values (one ragged, not a multiple of the block). Quant weights: random bytes (`rand` with dtype u8/u32) are fine; scales as bf16/f16/f32 in a sane range.
- `rewrite/tables/<group>.md`: your MATCH rows, written by compare.py on Ai-core (`--table ~/re-halogen/rewrite/tables/<group>.md`), then copied back: `scp ai-core:re-halogen/rewrite/tables/<group>.md /Users/peterb/Documents/AI/re-halogen/rewrite/tables/`.
- `rewrite/tables/<group>-negctl.md`: negative controls (below).
- `rewrite/reports/<group>.md`: per instance: status (MATCH / FAIL / BLOCKED), spec, tolerance and why, listing cites for anything non-obvious, blockers. Update it as you go so a follow-up agent can continue.

Do not edit `rewrite/harness/*`, `rewrite/match-table.md`, `analysis/*`, `PLAN.md`, other groups' files, hooks, or anything outside the repo. If the harness lacks a feature you need, write it in your report as BLOCKED with the exact need.

## Running on Ai-core

```bash
rsync -a /Users/peterb/Documents/AI/re-halogen/rewrite/kernels/<family>.hip ai-core:re-halogen/rewrite/kernels/
rsync -a /Users/peterb/Documents/AI/re-halogen/rewrite/specs/<family>_*.json ai-core:re-halogen/rewrite/specs/
ssh ai-core 'mkdir -p ~/re-halogen/rewrite/tables && cd ~/re-halogen && python3 -I rewrite/harness/compare.py rewrite/specs/<spec>.json --kernel <mangled> --seeds 1,2,3 --sizes <a>,<b> --table ~/re-halogen/rewrite/tables/<group>.md'
```

Sync only your own files (never rsync the whole `rewrite/` dir). The GPU is shared with a live service (Hermes): every launch under 2 GiB device memory and well under a few seconds; one compare.py at a time; never touch `halogen-swift`, never run `flash_serve`, never `podman`.

## Rules for a MATCH row

- Default tolerance is bitwise (no `tol`). Use a tolerance only when bitwise is impossible for a reason you can name (for example an atomic f32 accumulation order); state it in the spec `tol`, the report and the .hip comment. Try bitwise first: match the vendor's instruction-level math (v_exp_f32 via `__builtin_amdgcn_exp2f`, rcp vs IEEE divide, fma vs mul+add, reduction order, bf16 RNE without NaN special case).
- A row counts only if compare.py printed MATCH for it. Never write a row by hand.
- Every instance needs at least one MATCH row. Template instances: a templated rewrite plus one spec per instantiation (decode the template args from the mangled name).
- Element-wise kernels with a 16-bit input: add an exhaustive spec (every bit pattern: `iota` dtype u16 start 0 step 1, N = 65536).
- Negative control, one per family at least: copy your .hip to `/tmp/negctl_<family>.hip` on Ai-core, change one constant or operation, run compare.py with `--rewrite-src /tmp/negctl_<family>.hip --table ~/re-halogen/rewrite/tables/<group>-negctl.md`; it must print MISMATCH. Record it in the report.

## Discipline

- No em dashes anywhere. Plain sentences. Cite `file:line` for every claim; label inferred items.
- Context: grep-driven windows, about 15 reads per kernel at most. Big families: write the template once, then cover instances by spec. If you run low on context, stop cleanly with the report listing what is left.
- No new Python in the repo. Data generation goes in spec fills; scratch scripts live in `/tmp` on Ai-core.
- Final message: counts (instances MATCH / total, FAIL, BLOCKED), the table and report paths, and the negative control results.

## Addendum (orchestrator, after g04 and g08)

Random inputs in a wide range hide small constants. On g08's k_dn_recur_step, changing the norm epsilon from 1e-6 to 1e-5 still MATCHed with qkv in [-2, 2] (sum of squares about 170, the epsilon is below one ulp); with qkv in [-0.001, 0.001] the same change gives 8,341 differing outputs. For every constant in your rewrite that can only matter at small or edge magnitudes (epsilons, clamps, guards, scale floors, softplus thresholds, NaN/inf paths), add a spec that exercises it, and make your negative control change that constant and show MISMATCH on that spec. Name these specs `<family>_*_edge.json` or `_small.json`.
