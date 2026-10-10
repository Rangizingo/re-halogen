# PLD (prompt-lookup drafting): how it works in halogen-flash-server 0.17.3

Closes gap S20 of `analysis/request-trace.md`. Source: Ghidra decompiles under `work/decomp/flash_serve/` (abbreviated `D/`). All line numbers are lines of the decompile file named. Quotes are copied from the decompile.

Short answer: there are no PLD strings, no PLD kernel and no PLD function. PLD lives inline in the per-request decode loop, the very large function `FUN_0182ea80` (`D/0182ea80.c`, role map name `parse_request_json`, which is a misnomer: the 9,512-line body also holds the engine tick loop). The n-gram match runs on the host CPU over a per-request token history, using a hash table. The `k_ngram_*` kernels are not involved (they serve the n-gram embedding table, as S20 already suspected).

Naming used below. `R` is the request state, `pauVar8` in `0182ea80.c`. Its 16-byte slots are `pauVar8[i]`. The config struct copied from `main` sits at `pauVar8[0x11]`..`[0x13]`.

## 1. Config layout (proven)

`main_pseudo.c:750-752` defaults, then env overrides:

- `uStack_14d8 = 3; uStack_14d4 = 3; uStack_14d0 = 0x20;` so default N=3, K=3, W=32. `local_14cc = 0x3f800000` (floor 1.0f) and `local_14c8 = 0x40` (R=64) at `main_pseudo.c:744-745`.
- `main_pseudo.c:753-755`: `getenv("HALOGEN_PLD")` parsed with `"%d,%d"` into N and K.
- `main_pseudo.c:757-758`: `HALOGEN_PLD_ADAPT` parsed with `"%d,%f,%d"` into W, floor, R.
- `main_pseudo.c:764-772`: `HALOGEN_PLD_LOOKBACK` (high half of `local_14c8`) and `HALOGEN_PLD_AGREE` (low half of `uStack_14c0`).
- `main_pseudo.c:774-783`: PLD is forced off (`uStack_14d8 = 0`, message "prompt lookup off (it runs beside the draft head, and HALOGEN_MTP=0 turned the head off)") when the MTP head is off, and K must be below SPEC_N (`"HALOGEN_PLD K=%d needs HALOGEN_SPEC_N >= %d"`).
- `main_pseudo.c:938-943` copies the values into the engine struct: `uStack_1354 = uStack_14d8; uStack_1350 = uStack_14d4; uStack_134c = uStack_14d0; local_1348 = local_14cc; local_1344 = local_14c8; uStack_133c = CONCAT44(2,(int)uStack_14c0);`

Resulting offsets from the struct base `local_1360`. The base is `pauVar8[0x11]` in the request loop (inferred: the loop reads the same fields at these offsets, see sections 3 to 7):

| offset | field | read in 0182ea80.c as |
|---|---|---|
| +0x0c | N (n-gram size) | `*(int *)(pauVar8[0x11] + 0xc)` (8322, 8636) |
| +0x10 | K (chain cap) | `*(uint *)pauVar8[0x12]` (8665-8666) |
| +0x14 | W (adapt window, rounds) | `*(int *)(pauVar8[0x12] + 4)` (9025) |
| +0x18 | floor (min mean accepted per round) | `*(float *)(pauVar8[0x12] + 8)` (about 9061) |
| +0x1c | R (backoff base, in committed tokens) | `*(int *)(pauVar8[0x12] + 0xc)` (9068) |
| +0x20 | lookback | `pauVar8[0x13]` plain: no reader found (see unproven) |
| +0x24 | agree flag | `*(int *)(pauVar8[0x13] + 4)` (8848, 8926) |

## 2. Where the n-gram match happens: host loop, hash table (proven)

History and index structure `H` sits at `R+0x1b0` (`pauVar8[0x1b] + 8`). From `FUN_01870b60` and `FUN_018708b0`:

- `H+0x00`: int N.
- `H+0x08..0x18`: `std::vector<int>` of every token of the request so far (prompt plus generated).
- `H+0x20`: hash map, key = 64-bit hash of an N-gram, value = int position of the token that FOLLOWED that N-gram.

### 2.1 Hash

FNV-style with prime `0x100000001b3` and seed `0x14650fb0739d0383`, xor-shift by 29 after each token:

```
D/01870b60.c (hash loop)
  local_38 = 0x14650fb0739d0383;
  ...
      local_38 = (local_38 ^ *(uint *)(lVar1 + uVar4 * 4 + uVar2 * 4)) * 0x100000001b3;
      local_38 = local_38 ^ local_38 >> 0x1d;
```

The same hash is computed at lookup time in `D/0182ea80.c:8692-8715` (`uVar91 = 0x14650fb0739d0383;` then `* 0x100000001b3` over `lVar96 + lVar80*4`, where `lVar80 = size - N`). So the key is the hash of the last N tokens of the history.

### 2.2 Insert (history push), `FUN_01870b60(H, token)` (proven)

`D/01870b60.c`: returns if `N < 1` (`if ((int)*param_1 < 1) { return; }`). Appends the token (`*puVar3 = (int)param_2;`). If the vector now holds more than N tokens it hashes the N tokens that precede the new last token, and stores `map[hash] = index of the new token`:

```
D/01870b60.c (tail)
  puVar3 = (undefined4 *)FUN_01871410(param_1 + 8,&local_38);
  *puVar3 = (int)uVar7;
```

`uVar7` is `size - 1` after the append (`uVar7 = uVar7 - 1;`). It overwrites, so the table keeps the MOST RECENT occurrence of each N-gram. The entry for the current suffix is only written when the next token is pushed, so a lookup of the current suffix never finds itself.

### 2.3 Seed from the prompt, `FUN_018708b0` called from `FUN_01860330` (proven)

`D/01860330.c:180-186`: `if (puVar18 == (undefined8 *)0x0 && 0 < *(int *)((long)param_1 + 0x11c)) { *(int *)(param_1 + 0x37) = *(int *)((long)param_1 + 0x11c); ... FUN_018708b0(param_1 + 0x37,param_7,param_8,*(undefined4 *)(param_1 + 0x26)); FUN_01870b60(param_1 + 0x37,param_3);`

At request start (only when N>0 and `puVar18` is null, taken to be the greedy case, see section 3) the history gets its N, then the prompt tokens (`param_7` pointer, `param_8` count) are appended and indexed in one pass (`D/018708b0.c:22-24`, append via `FUN_01871100`; loop at lines 60-99 inserts `map[hash] = position`), then the first sampled token is pushed.

### 2.4 Tokens pushed during decode (proven)

Every committed token goes through `FUN_01870b60`. Call sites in `0182ea80.c`: 7810, 8359, 8543, 8637, 8947; also `D/01860e60.c:19,29` and `D/01851770.c:117`. Guard text at 8636-8637: `if ((0 < *(int *)(pauVar8[0x11] + 0xc)) && (*(long *)pauVar8[0x19] == 0)) { FUN_01870b60(pauVar8[0x1b] + 8, ...`. Site 8947 is the PLD path itself.

## 3. When PLD is tried (gate proven, meaning inferred)

`D/0182ea80.c:8322-8324`:

```
if (((*(int *)(pauVar8[0x11] + 0xc) < 1) || (*(long *)pauVar8[0x19] != 0)) ||
   (*(int *)(pauVar8[0x21] + 8) != 0)) {
LAB_01840408:
```

Control goes to the ordinary MTP-head path (`LAB_01840408`) when N is 0, or `pauVar8[0x19]` is non-null, or the backoff counter `pauVar8[0x21]+8` is nonzero. Otherwise it falls into the PLD block (8663 onward). Reading:

- `pauVar8[0x19] != 0` selects the sampled branch of the head path, which calls `draft_sample` (`FUN_019098d0`) and `accept_probs` (`FUN_01909e50`). The null case is the greedy branch (verify and compare). So PLD runs only in greedy mode. Inferred from structure; the pointer's writer was not traced. It matches the history guards in 2.4.
- The backoff counter is described in section 7.

Round structure: PLD and the MTP head are NOT in one verify. Each tick runs either the PLD block or the head block. If the PLD block yields no chain it exits through `LAB_0183fd73` (`uVar43 = 0`) and then `if (iVar74 < 1) goto LAB_018403fc;` (8848-8849) jumps to the head path in the same tick. The head's next-round draft token is refreshed after a PLD verify (`D/0182ea80.c:8985`: `uVar36 = FUN_018f9c60(lVar80,&local_16b8,uVar91);`, stored to `*pauVar8[0x1a]`), so the head stays primed. "Runs beside the draft head" in the `main` message means this: shared engine, alternating, head drafts for the following round while PLD rounds happen.

## 4. Match rule and chain construction (proven)

Cap on chain length (`D/0182ea80.c:8663-8680`):

```
uVar35 = *(int *)(lVar80 + 0x14) - 1;
uVar57 = ~*(uint *)(lVar80 + 0x2e8) + *(int *)(lVar80 + 0x118);
if ((int)*(uint *)pauVar8[0x12] <= (int)uVar35) { uVar35 = *(uint *)pauVar8[0x12]; }   // min with K
if ((int)uVar57 < (int)uVar35) { uVar35 = uVar57; }                                    // min with KV room
uVar57 = *(int *)(pauVar8[0x22] + 8) - 1;  ... if ((int)uVar57 < 1) goto LAB_0183fd73; // min with tokens left
```

The cap is `min(K, verify width - 1, KV room, tokens left)`. If below 1, no PLD this tick. (Meaning of `+0x14`, `+0x2e8`, `+0x118` engine fields and of `pauVar8[0x22]+8` as "tokens left" is inferred.)

Lookup (`8684-8795`):

1. `uVar35 = *(uint *)(pauVar8[0x1b] + 8)` is N. It needs `size >= N` (`if (uVar43 < uVar91) goto LAB_0183fd73;`), else no match.
2. Hash the last N history tokens (start index `size - N`, 8692-8715), look the hash up in the map (8717-8744, chained lookup comparing `puVar51[1]` with the hash).
3. Hit: `uVar63 = (ulong)*(int *)(puVar51 + 2);` is the stored position (token after the earlier occurrence). `if (uVar43 == uVar63) goto LAB_0183fd73;` (8750) rejects a hit equal to the current size.
4. Chain length `uVar91 = min(cap, size - pos)`; the chain is `history[pos .. pos+len)`, copied to `local_1398` (8753-8795). The draft is the up-to-K tokens that followed the most recent earlier occurrence of the current N-token suffix. No token compare is made against the hash hit (none in 8684-8795), so a collision would yield a wrong chain that verify then rejects (consequence inferred).
5. Grammar: if a grammar object exists (`pauVar8[0x24]`), each chain token goes through grammar vtable calls (about 8797-8828) and the chain is cut at the first token rejected (`lVar80 == -2` shortens `uVar43`). Shape proven, exact semantics unread.

## 5. Verification in the engine (proven)

Chain verify, `D/0182ea80.c:8888-8892`:

```
local_1038._0_4_ = *(undefined4 *)(pauVar8[0x19] + 0xc);   // last committed token
memcpy(local_1038 + 4,&local_1398,sVar64);                  // chain
uVar57 = iVar74 + 1;
FUN_018fa0e0(*(undefined8 *)(pauVar8[0x10] + 8),local_1038,uVar57);   // spec_verify(n = chain+1)
```

`FUN_018fa0e0` is `spec_verify` (index.tsv: "spec_verify(n=%d): needs HALOGEN_MTP=1, 1 <= n <= %d"). One forward pass over `[last_token, chain...]`. The engine leaves the argmax token per row at `engine+0x20 + i*4`.

Accept rule (about 8893-8917): walk `i = 0..`; compare `engine+0x20 + i*4` with `chain[i]`. First mismatch at `i` gives `accepted = i`, `keep = i+1` (8904: `uVar57 = (int)uVar91 + 1;`); mismatch at `i == 0` gives accepted 0, keep 1 (8898-8903). Full match gives `accepted = chain length`, `keep = chain length + 1` (8914: `iVar74 = (int)uVar83;`). Committed tokens are the accepted chain prefix plus one: `uVar36 = *(undefined4 *)(lVar96 + 0x20 + uVar63 * 4);` (8917), the model's own token at the first non-matching (or final) row. Then `FUN_018fc360(lVar96,uVar57)` (`spec_commit(keep)`; index.tsv "spec_commit(%d): no verify pending or keep outside [1, %d]") trims KV to `keep`.

Emission loop (about 8930-8950): per emitted token `FUN_01882d20` (emit), grammar advance via vtable, `FUN_01870b60` (history push, 8947), `FUN_01851580` (stop check, 8948). Greedy PLD output equals plain greedy output by construction, since only tokens the model argmaxed are committed (inferred from the compare rule; byte-equality not tested).

After the loop `FUN_018f9c60` (8985) computes the head's draft for the next round from the verify output, only if engine flags `+0xb7c` and `+0xb7d` say the head is live (8982-8984).

Agree gate (before verify, 8848-8873): when `agree != 0` and the head has no backoff (`*(int *)(pauVar8[0x22] + 4) == 0`), the head's pending draft (`*pauVar8[0x1a]`, computed by `FUN_018f9dc0` if it is -1) is compared with the first chain token:

```
D/0182ea80.c:8869,8871
if ((float)local_1398.tv_sec != (float)iVar32) goto LAB_018403d5;
else if ((float)local_1398.tv_sec != (float)*(int *)pauVar8[0x1a]) { LAB_018403d5: ...
```

LAB_018403d5 leads to `LAB_018403fc` (head path). On disagreement the tick falls to the head path and PLD is skipped that tick. With `HALOGEN_PLD_AGREE=0` the test is skipped (first branch at 8848). So at default agree=1 the head acts as a one-token confirmer for the PLD chain. (Compare is on `float` casts of token ids, decompiler output; exact below 2^24.)

## 6. Counters (proven by formula consistency; the +1 constant inferred)

PLD round bookkeeping at `D/0182ea80.c:8920-8923`:

```
auVar124._0_8_ = (byte *)((long)_DAT_001142f0 + SUB168(pauVar8[4],0));
auVar124._8_8_ = uVar63 + SUB168(pauVar8[4],8);
pauVar8[4] = auVar124;
*(ulong *)pauVar8[6] = *(long *)pauVar8[6] + uVar43;
```

- `pauVar8[4]` low += constant `_DAT_001142f0` (not decoded, presumed 1) per PLD round; high += `uVar63`, tokens accepted from the chain.
- `pauVar8[6]` low += `uVar43`, chain length proposed this round.

Mapping to stats longs (slot k holds longs 2k and 2k+1): long[8] = pld_rounds, long[9] = pld_accepted, long[0xc] = pld_drafted.

D line formatter `FUN_0184c9a0` (callers `01845e50.c:157`, `01847a30.c:66`, `01848190.c:450`, `0182ea80.c:1161,1226,2226`). Call at `01845e50.c:157` passes `lVar5 = plVar18[8] + plVar18[0]` as rounds, `lVar11 = plVar18[1] + lVar5 + plVar18[2] + plVar18[9] + plVar18[0x10]` as commit, then `[2]`, `[8]`, `[9]`, `[0xc]`. Check against `wire-protocol.md:165`, "head accepted = commit - rounds - shared - pld_accepted": rounds includes `[8]`, shared is `[2]`, so the formula closes only if `[9]` is pld_accepted, which makes `[8]` pld_rounds and `[0xc]` pld_drafted by argument order. That the stats object `plVar18` is the same object as `R` (`pauVar8`) is inferred (same slot arithmetic; pointer lineage not shown).

Live data (`analysis/evidence/probe2-0.17.3.jsonl`, first D line): `D 900001 length 19 64 271.3 1242.7 1 22 63 0 1 4 18 24 ...` gives rounds 22, commit 63, shared 1, pld_rounds 4, pld_accepted 18, pld_drafted 24. The INFO line in the same file is `I 1 0 262144 7 1 1 0 110 64 4 262144 2 1 1 524288 1 0.17.2 2 6 gguf ...`: pld_n = 2, pld_k = 6 (fields 18 and 19). 24 drafted / 4 rounds = 6 = K per round, and 18 accepted <= 24. That matches `uVar43 = min(K, ...)` summed per round. Note the INFO line prints version 0.17.2 while the file is named 0.17.3; D-line layout is the same.

## 7. Adaptive backoff (code proven; purpose partly inferred)

`D/0182ea80.c:9025-9072`, at the end of each PLD tick when `W > 0`:

```
iVar84 = *(int *)pauVar8[0x21] + 1;               // rounds in window
iVar74 = iVar74 + *(int *)(pauVar8[0x21] + 4);    // accepted tokens in window (iVar74 = this round's accepted)
if (iVar33 <= iVar84) {                           // window full (W)
  fVar110 = (float)iVar74 / (float)iVar84;        // mean accepted per round
  ... push fVar110 onto a float history vector (pauVar8[0xb]/[0xc])
  *(undefined8 *)pauVar8[0x21] = 0;               // reset window
  if (floor <= fVar110 || R < 1) { backoff streak = 0 }
  else {
    *(int *)(pauVar8[0x21] + 8) = R << min(streak, 6);   // cooldown tokens
    streak += 1;  *(long *)pauVar8[5] += 1;              // backoff count
  }
}
```

While `pauVar8[0x21]+8 > 0` the gate at 8322 sends every tick to the head path. The cooldown drops by one per committed token (8360-8364, 8554-8557, 8638-8642); at 0, `pauVar8[5]+8` is incremented (re-enable count). With defaults W=32, floor=1.0, R=64: if PLD averages under 1.0 accepted tokens per round over 32 rounds, PLD is off for 64, then 128, up to 64 * 2^6 = 4096 tokens, and the streak resets only after a window that meets the floor.

The head has its own separate window (`*(int *)pauVar8[0x11]`, `+4`, `+8`; about 8930 and 8995-9015) from `HALOGEN_SPEC_ADAPT`. When `agree != 0`, PLD rounds also feed that head window (`(pauVar8[0x1a]+4)++`, `(pauVar8[0x1a]+8) += 1 - (accepted==0)` at about 8926-8929). Purpose inferred.

## 8. Owner

- Owner of gate, match, chain build, verify, accept, counters and backoff: `FUN_0182ea80` (`D/0182ea80.c:8322-8324` gate, `8663-8950` PLD block, `9025-9072` backoff). The role-map name `parse_request_json` is a misnomer for this function; it also holds the engine tick loop.
- Helpers: `FUN_01870b60` (history push plus N-gram index), `FUN_018708b0` (bulk prompt index), `FUN_01860330` (request init, calls the seed), `FUN_01860e60` (per-token commit helper that also pushes history), `FUN_01871410` (hash-map find-or-insert), `FUN_018fa0e0` (`spec_verify`), `FUN_018fc360` (`spec_commit`), `FUN_018f9c60` and `FUN_018f9dc0` (head draft).
- Not involved in PLD: `spec_round_run` (`01851dc0`, called from the head path at 8445 when depth > 1), `spec_verify_grammar_check` (`01851770`, called at 8396 from the head greedy-grammar-free path), `draft_sample` (`019098d0`, head sampled mode), the `k_ngram_*` kernels.

## 9. Corrections to earlier docs

- `request-trace.md` S20 said the n-gram loaders `019041a0`-`01904880` and `locate_ngram_table` `01894790` might serve PLD. They do not. PLD uses the in-memory hash map above and no GPU kernel.
- "PLD chain verified in the same round as the MTP head" is not what the code does. Each tick is either a PLD verify (chain up to K) or a head round. They share the engine; the head's one-token draft is an optional agree filter on the PLD chain (default on).

## 10. Still unproven

1. `_DAT_001142f0` value: the per-round increment of pld_rounds is presumed 1 (runtime shows 4 rounds and 24 = 4*6 drafted, consistent, not a read of the constant).
2. That the D-line stats object `plVar18` is the same object as the request state `pauVar8` (construction not traced).
3. `HALOGEN_PLD_LOOKBACK`: stored at config +0x20 but no read found in `0182ea80.c`; `01860330`, `01860e60`, `01851770` not fully read. Could bound lookups or be unused in 0.17.3.
4. `pauVar8[0x19]` null meaning "greedy" is inferred from branch structure.
5. Hash-collision handling: no token compare seen; effect inferred.
6. Grammar chain-cut details read at shape level only.
7. Meaning of engine fields `+0x14`, `+0x118`, `+0x2e8` and `pauVar8[0x22]+8` in the cap is inferred.
8. No runtime test of the match rule itself. Test needed: greedy request with `HALOGEN_PLD=2,6` on a prompt repeating a 2-gram followed by a distinct 6-token tail; expect the first PLD round to show pld_drafted = 6 and pld_accepted = how many of those 6 the model reproduces.
9. Whether the first lookup after the last prompt token can hit the prompt end itself: line 8750 rejects `pos == size`; seeding order not traced for an off-by-one.
