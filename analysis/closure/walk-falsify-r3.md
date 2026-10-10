# U07: HALOGEN_WALK_FALSIFY effect on acceptance math (round 3)

Verdict first: CLOSED. The variable is a host-only integer mode read once in spec_round_run. It changes no kernel argument and no acceptance formula. It perturbs which row indices and which RNG counter feed the host accept test and the residual resample, plus (mode 2) the order of sampler-state updates. Mode values 0..4 are all that the code distinguishes.

Address convention: Ghidra = file + 0x100000. Decomp files are work/decomp/flash_serve/<ghidra addr>.c. Disassembly is r2 on the file address.

## 1. Read and store

- String `HALOGEN_WALK_FALSIFY` at file 0x44f9e (r2 `izz`, .rodata). Only xref: `0x01783021 lea rdi, str.HALOGEN_WALK_FALSIFY` in the getter FUN_01883020 (file 0x1783020).
- Getter 01883020.c:8-12: `getenv("HALOGEN_WALK_FALSIFY")`; if set `__isoc23_strtol(p,0,10)`, else return 0. Disasm: `0x01783028 call getenv`, `0x0178303c call __isoc23_strtol` with `mov edx, 0xa`, `0x01783043 xor eax, eax`.
- Exactly one caller. r2 `/ad call 0x1783020` returns only `0x01752a88 call 0x1783020`, in spec_round_run (FUN_01851dc0; Ghidra 0x1852a88). Decomp 01851dc0.c:120-122: guarded static init `DAT_019fc588 = FUN_01883020(local_e8);` under guard `DAT_019fc590`. Disasm: `0x01752a8d mov dword [0x018fc588], eax` (file 0x18fc588 = Ghidra 0x19fc588).
- Only function text mentioning DAT_019fc588 or FUN_01883020 across the decomp set: 01851dc0.c and 01883020.c (grep -l). Value is cached for the process life (first spec round), so it cannot change at runtime.
- Local copy: 01851dc0.c:124 `iVar14 = DAT_019fc588;`; disasm `0x0175211a mov r13d, dword [0x018fc588]`. r13d holds it for the whole accept loop.

## 2. Every use of the value (complete)

Disasm check: within the loop body from 0x1752102, the only compares of r13d are at 0x17521dc (3), 0x1752296 (2), 0x1752344 (2), 0x17523e8 (4), 0x1752417 (1), and a later `cmp eax, 2` at 0x1752680 is on a different register (eax), see section 5. The decomp has the matching tests at 01851dc0.c:145, 157, 174, 182, 205.

| m | Site | Effect |
|---|------|--------|
| 3 | 01851dc0.c:145-147 `if (iVar14 == 3) iVar6 = 0;` ; disasm `0x017521dc cmp r13d,3 / 0x017521e0 mov eax,r12d / 0x017521e3 mov ecx,0 / 0x017521e8 cmove eax,ecx / 0x017521eb add eax,[rsp+0x24]` | Draft index term of the RNG counter is zeroed. Counter = (0 + base) instead of (i + base), so every draft position draws the same uniform u. u comes from the splitmix64 chain at 01851dc0.c:148-156 (constants 0xd1b54a32d192ed03, seed at `*(lVar16+0x10)`, tag 0x3c6ef372fe94f82a). |
| 4 | 01851dc0.c:174-176 `uVar8 = uVar19 & 0xffffffff; if (iVar14 == 4) uVar8 = 0;` ; disasm `0x017523e8 cmp r13d,4 / 0x017523ec mov esi,r12d / 0x017523ef cmove esi,r10d` (r10d = 0) | Second argument of residual_sample FUN_0190aa50 (call at 01851dc0.c:185, disasm `0x0175243a call 0x180aa50`) becomes 0 instead of the reject index i. |
| 1 | 01851dc0.c:182-184 `if (iVar14 != 1) uVar12 = uVar19 & 0xffffffff;` with uVar12 = 0 set at :173 ; disasm `0x01752417 cmp r13d,1 / 0x0175241b cmovne r10d,r12d` | Last argument of FUN_0190aa50 (pushed as the final stack arg, `0x01752433 push r10`) is 0 instead of i. |
| 2 | 01851dc0.c:157-172 and :205-231 ; disasm `0x01752296 cmp r13d,2`, `0x01752344 cmp r13d,2` | Sampler-state step (FUN_0184ad90, plus FUN_0184a9b0 when the penalty fields at param_3+0x1e0..0x218 are non-neutral) is deferred. Accepted tokens are not stepped one by one inside the loop (:205 `if (iVar14 != 2)` skips the in-loop step); they are stepped after the loop (:219-231 loop over local_e8[0..n)), or, on a reject at i>0, stepped first (:157-172) before residual_sample. Same set of tokens, same calls, different timing. |
| 0, other | no test matches | normal path. Values >= 5 or negative behave as 0 (inferred: no other compare exists in the disasm window or decomp). |

Meaning of the residual_sample parameters (target-logit row for param_2, drafter q row for the last arg) is from sampling-host.md and 0190aa50.c, not re-derived here: label inferred. The code-level fact is only that modes 4 and 1 zero those two arguments.

## 3. What it does NOT change

- The accept ratio and test are the same in every mode: 01851dc0.c:137 `FUN_01909e50(local_f0,1,...,&local_10c,...)` (accept_probs, produces local_10c = target p of the draft token) takes no argument derived from iVar14 (disasm `0x017521a8 call 0x1809e50`, the pushes before it do not reference r13d). Then `dVar21 = local_10c / exp(local_78[i])` (:138-143), capped at DAT_00113bb8 (`if (DAT_00113bb8 <= dVar21) dVar21 = DAT_00113bb8`), compared `dVar21 <= u*2^-24` (:154-156) to choose the reject branch. So the acceptance math is min(1, p/q) vs a uniform in all modes; WALK_FALSIFY only changes the u (mode 3) and what happens after a reject (modes 1, 2, 4).
- No device kernel argument comes from it. The all-accepted path (FUN_01909630 at 01851dc0.c:245, disasm call after LAB_0185254c) receives no iVar14-derived parameter other than the plain draft count (:245 uses `iVar11 + iVar14` where iVar14 was reassigned to local_108 at :240; this is the draft count, not the env value; inferred from the reassignment `iVar14 = (int)local_108;`).
- This agrees with kernels-speculation.md 4b: k_smpg_accept_final / k_accept_prob_f32 are unchanged.

## 4. Comparison with HALOGEN_S3_FALSIFY

Per analysis/closure/sampling-host-r2.md:16, 21: S3 getter FUN_0194bd90 (strtol, default 0) is called at `180a99c: callq 0x184bd90` under guard byte 0x18ff5b0, stored at 0x18ff5a8, and passed as the mode int (second trailing int, s13) of the accept-prob launch; the kernel uses it only in the fall-through block (`s_cmp_lg_u32 s13, 3`, accept = 1.0 if mode == 3). I did not re-derive that here (cited, not re-verified).

Contrast, both inferred as test hooks (the draft_sample message says "deliberately wrong (METHOD 37)" for the S2 flags, 019098d0.c:44-62):
- S3_FALSIFY: device argument; value 3 force-accepts the fall-through row.
- WALK_FALSIFY: host-only; the two share the value 3 as a special mode but are unrelated code paths. WALK_FALSIFY=3 changes the RNG counter, not a kernel mode. There is no link between DAT_019fc588 and the S3 static (separate getters, separate statics, separate consumers).

## 5. Check on the one stray compare

`cmp eax, 2` at 0x1752680 (disasm window line 313) is not r13d. Context was not read; it is outside the accept loop and not a reference to 0x18fc588 (the only load of the static in the window is 0x0175211a). Inferred: unrelated.

## Verdict

CLOSED. What WALK_FALSIFY does to speculative acceptance: nothing to the formula or any kernel argument; mode 3 zeroes the position term of the host RNG counter (01851dc0.c:145), mode 4 zeroes the failing-row index passed to residual_sample (:174), mode 1 zeroes its drafter-row index (:182), mode 2 reorders sampler-state steps (:157, :205). Unproven by runtime only: that these modes produce visibly wrong output (purpose inferred from the "METHOD 37" strings); no run was made, per rules.

## Reproduce

```
ssh ai-core 'cd ~/re-halogen && r2 -q -c "izz~FALSIFY" binaries/flash_serve'
ssh ai-core 'cd ~/re-halogen && r2 -q -e scr.color=0 -c "/ad call 0x1783020; pd 12 @ 0x1783020" binaries/flash_serve'
ssh ai-core 'cd ~/re-halogen && r2 -q -e scr.color=0 -c "pd 14 @ 0x1752a70" binaries/flash_serve'
ssh ai-core 'cd ~/re-halogen && r2 -q -e scr.color=0 -c "pd 400 @ 0x1752102" binaries/flash_serve' | grep -n "0x018fc588\|cmp r13d"
grep -n "DAT_019fc588\|iVar14 == [0-9]\|iVar14 != [0-9]\|FUN_0190aa50" work/decomp/flash_serve/01851dc0.c
grep -ln "FUN_01883020\|019fc588" work/decomp/flash_serve/*.c
```
