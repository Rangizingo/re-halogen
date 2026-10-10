# NPU serve verbs round 3: U02 numeric fields

Scope: `binaries/halogen-npu`. Decompiles are `work/decomp/halogen-npu/<addr>.c`, cited `<addr>.c:line` (Ghidra address = file address + 0x100000). The worktree has no `work/`, so decompiles and the binary were read from `/Users/peterb/Documents/AI/re-halogen`, read-only. Disassembly was read with r2 on ai-core, read-only. "proof" means shown by quoted code or bytes. "inferred" means reasoned and not shown. Builds on `analysis/closure/npu-closure-r2.md` (r2). Static only.

## Verdict

Verdict: CLOSED for the numeric fields in the r2 grammar table (GEN, DEC, EMB, SCO, CLS, PFX, ELOAD, EFREE), plus the IMG argument list r2 left open. Three corrections to r2 are in section 1. Items outside that table that stay open (HASH_A/HASH_B/MISMATCH buffer semantics, the `"o"` route, LOOKUP/QHN/QLED/EDUMP argument lists) are in section 8 with exact reasons.

## 1. Corrections to r2 (all proof)

1. SCO/CLS and EMB do not use the DEC request grammar. `FUN_0012b110` has three separate request parsers (section 3). r2's "same as DEC" holds only for DEC.
2. All printed time fields are microseconds, not milliseconds. `FUN_00155ee0` returns the `steady_clock::now()` count (nanoseconds) divided by `DAT_001f4bd8`; the bytes at file 0xf4bd8 are the double `1000000.0`, so the value is milliseconds. Every reply multiplies by `ram0x001f4bd0` = double `1000.0` (file 0xf4bd0) before printing as an integer. Milliseconds times 1000 is microseconds. Evidence: `00155ee0.c:6-8` (`return (double)lVar1 / DAT_001f4bd8;`), `00125240.c:94,98`.
3. DEC/EMB/SCO/CLS reply field `t2` is always 0. `uStack_178` is the only source of `param_3+0x90` (`0014a6f0.c:1112` `*(undefined8 *)(param_3 + 0x90) = uStack_178;`) and the only assignment is `0014a6f0.c:403` `uStack_178 = 0;` (grep finds only that and the declaration at line 128). `t1` is the whole `FUN_0014a6f0` wall time: `:161` `dVar55 = (double)FUN_00155ee0();`, `:1064` `local_180 = local_180 - dVar55;`, stored `:1111`.

## 2. DEC, EMB, SCO, CLS: what the verbs compute

All four verbs call the same function `FUN_0014a6f0(engine, req, result, err)` (`0012b110.c:1119` `iVar55 = FUN_0014a6f0(lVar33,local_280,param_4,&local_4c0);`). It is a head-only pass over n padded sequences. The final hidden state at one chosen position per sequence is RMS-normalised, then either dotted against embedding rows (DEC, SCO, CLS) or L2-normalised and truncated (EMB).

Request struct layout (byte offsets, derived from field uses):

| Offset | Content | Written by |
|---|---|---|
| +0x00 int | n, number of sequences | token 2 (`0011c0b0` strtoll at `*param_1+0x40`) |
| +0x04 int | m, tokens per grid row | token 3 (`*param_1+0x60`) |
| +0x08 vec<int> | n*m flat grid of token ids, row-major | trailing ints |
| +0x20 vec<int> | per-sequence slot `s_i`, the position whose hidden state is read | DEC: token `a`; SCO/CLS/EMB: `L_i - 1` |
| +0x38 vec<vec<int>> | per-sequence candidate rows (n entries) | DEC: the `c` lists; SCO/CLS: copy of the engine `rows`; EMB: empty lists |
| +0x50 vec<int> | per-sequence true length `L_i` (empty for DEC) | SCO/CLS/EMB |
| +0x68 int | k, embedding width (0 unless EMB) | EMB third header int |

Shape checks (`0014a6f0.c:211-222`, proof): `0 < n <= bucket+100` (batch capacity), `m <= bucket+0x60` (bucket sequence length), flat size equals `m*n`, slot vec size equals n, candidate-list count equals n; otherwise `a malformed request` (`:215`). The bucket comes from `FUN_0013ea60(engine, n, m)` (`:166`); no match gives `no route for B <n> ... tokens` (`:170-179`).

Per-sequence loop (`0014a6f0.c:280-310`, proof):
- Row i of the grid is copied into a working buffer and padded to the bucket length with the pad id (`:313-330`, copy `local_418[lVar49 + lVar26] = *(int *)(lVar7 + lVar25 * iVar5 * 4 + lVar26 * 4)`, then `= iVar17` fills).
- Slot check `(iVar18 < 0) || (iVar5 = param_2[1], iVar5 <= iVar18)` gives `a slot outside its sequence` (`:305-306`). So DEC requires `0 <= a < m`.
- Length check, only when the +0x50 vec is non-empty: `(iVar5 < 1) || (param_2[1] < iVar5)` gives `a length outside its sequence` (`:300-301`). So SCO/CLS/EMB require `1 <= L_i <= m`.
- Every grid id must be in `[0, vocab)` with vocab at `param_1+0x2d0`, else `a token id outside the vocabulary` (`:386-387`).
- Pad id `iVar17` is the integer attribute `"pad"` of a manifest section, default 0. Disassembly: `0x0004a8d2 lea rsi, [0x000efb16] ; "pad"`, `0x0004a8e1 xor edx, edx`, `0x0004a8e9 call 0x62ba0`. `FUN_00162ba0` (`00162ba0.c:21-31`) runs strtoll on the key's value and returns `param_3` (the zero from `xor edx, edx`) when the key is absent. The section searched is selected by the string built at 0x4a857 (`0x0004a82c lea rsi, [0x000efa38] ; "tokens"`, fall-through path only; the `jg 0x4a840` path was not traced), matched against the 0x58-byte records at engine+0x268. Inferred: the manifest "tokens" section. The pad id is echoed as the `int` in the DEC/SCO/CLS reply (`0014a6f0.c:928` `*(int *)(param_3 + 0x20) = iVar17;`).

Hidden state (`0014a6f0.c:939-950`): for each sequence `FUN_00161060(eps, hidden_row, H, fnorm, ...)` is RMSNorm. `00161060.c:29-37`: `fVar22 = sum_sq; param_1 = fVar22 / (float)H + eps; param_1 = DAT_001ef014 / sqrt(param_1)` (`DAT_001ef014` is float 1.0, file 0xef014 bytes `0000803f`), `:45-46` `out = param_1 * x * w`. Row index is `s_i + i*bucket_len` (`lVar48 = piVar24[lVar46] + lVar33; lVar33 += lVar21`; stride equal to the bucket sequence length is inferred). `H` is `*(int *)(param_1+0xd8)` = manifest `hidden` (`00145eb0.c:535-536`), `fnorm` is the float32 weight blob "fnorm" (`00145eb0.c:561-571`), `eps` is manifest `eps` (`00145eb0.c:520`).

## 3. Request grammars, corrected, with field meanings

Token 0 = `VERB[@model]`, token 1 = tag. Verb literals (python read of the binary): 0xef433 `DEC`, 0xef172 `EMB`, 0xef176 `SCO`, 0xef17a `CLS`, 0xef17e `GEN`, 0xef3cc `IMG`, 0xef3fb `INFO`, 0xefd83 `HN`. The branch is chosen by the compare chain at `0012b110.c:503-505`: DEC is the branch calling `FUN_0011c0b0` (`:1116`), EMB is the `else` at `:903-935`, SCO/CLS is the remaining branch `:506-900`. Numbers are full-token strtoll (`00119ec0.c:14-21`, false on trailing garbage or a missing token). Limits shared by all three: `1 <= n <= 0x40` and `1 <= m <= 0x100000` (`0012b110.c:538` `lVar37 - 1U < 0x100000 ... uVar36 - 1 < 0x40`; `:936-937`).

| Verb | Request after the tag |
|---|---|
| DEC | `n m` then n groups `a c i1..ic`, then n*m grid ints |
| SCO, CLS | `n m L1..Ln` then n*m grid ints |
| EMB | `n m k L1..Ln` then n*m grid ints |

Field meanings:
- `n`: number of sequences. Result lists are sized n (`0014a6f0.c:932-935` `FUN_00152800(param_3+0x28,n*H)`, `FUN_00154d40(param_3+0x40,n)`, `FUN_00152c80(param_3+0x58,n)`).
- `m`: tokens per grid row. Row stride `iVar5 = param_2[1]` (`:317-318`), bounded by `m <= bucket+0x60` (`:219`).
- Grid ints (n*m): token ids, row-major, copied to the model input and vocab-checked (`:315-318, :386-387`).
- DEC `a`: slot index with `0 <= a < m` of the position whose final hidden state is scored (stored in the +0x20 vec, used as the hidden row offset, `:297-299, :942-946`).
- DEC `c`: count of candidate rows for that sequence, `c <= 0x1000` (`bVar38 = local_60 < 0x1001`, r2). The following `c` ints are vocabulary row ids into the tied embedding table (`param_1+0x2c8`, bf16, H values per row). Out of range gives `a head row outside the vocabulary` (`:988-992`).
- SCO/CLS `L_i`: true token count of sequence i, `1 <= L_i <= m` (`0012b110.c:556-557` `bVar59 = 0 < (long)local_588 && (long)local_588 <= lVar37;`). The parser pushes `L_i` to the +0x50 vec and `L_i - 1` to the +0x20 vec (`:575-577, :622-629`), so the scored slot is the last real token.
- SCO/CLS candidate rows are not in the request. Each sequence gets a copy of the engine `rows` list (`0012b110.c:801-820`, size `*(long *)(lVar33 + 200) - *(long *)(lVar33 + 0xc0)`). That list is filled from manifest key `rows` when the model kind string at engine+0xa0 (default `"decision"`) is `score` or `classify` (`00145eb0.c:468-476`). Sizes: score exactly 2 ("no, yes", `00145eb0.c:491-496`), classify 2..16 (`:480-488`, "it takes 2 to 16").
- EMB `k`: embedding width returned. It is the third header int, parsed into `local_218` (`0012b110.c:934-935` `cVar31 = FUN_00119ec0(local_300,param_3,&local_2a0); local_218 = SUB84(local_2a0,0);`). The stack gap from `local_280` (request struct base) to `local_218` is 0x68, which matches struct offset +0x68 read as `param_2[0x1a]` in `0014a6f0.c:283,953-961` (the layout match is inferred; the uses prove the role). Accepted only if `0 < k` (`0012b110.c:944` `if (0 < (long)local_2a0)`; otherwise the malformed-line path) and `k <= H` (`0014a6f0.c:283`, else `a malformed embedding request`, `:1264`).
- EMB `L_i`: as SCO/CLS (`0012b110.c:951-962`, pushes `iVar45` and `iVar45 + -1`; guards `bVar4 = 0 < L`, `bVar59 = L <= m`).
- The token count must end exactly after the grid (`0012b110.c:1301-1305` `if (param_3[1] - *param_3 >> 5 == local_300[0]) goto LAB_0012b85f;`; DEC: `uVar19 * lVar20 == lVar35` in `0011c0b0`, r2).

## 4. Reply fields

DEC (`00125240.c:62-98`): `OK <tag> <bucket> <pad> <argmax_0..argmax_{n-1}> <scores...> <t1> <t2>`.
- `<bucket>`: first string of the chosen route object, copied at `0014a6f0.c:922` (`_M_assign(param_3, puVar27)`, `puVar27` from `FUN_0013ea60`). Inferred as the route (bucket graph) name; the code proves only that it is a string owned by the route record.
- `<pad>`: pad id from section 2 (`00125240.c:74` `_ZNSolsEi(uVar5,*(undefined4 *)(param_3 + 4))`).
- `argmax_i`: index into sequence i's own candidate list (not a vocabulary id) of the highest score. Ties go to the first; `-1` when the list is empty. Proof: initialised `0xffffffff` (`0014a6f0.c:935`, `:1048` `uVar41 = 0xffffffff`), strict greater-than update `*(float *)(lVar21 + (long)iVar39 * 4) < *(float *)(lVar21 + uVar23 * 4)` (`:1050-1055`), stored at `:1057`. Printed with the signed `_ZNSolsEi`.
- `score`: for each candidate id j of sequence i, `dot(hidden_i, E[j])`. `E[j]` is the bf16 row widened by `<< 0x10` (`:1001` `(uint)*(ushort *)(lVar33 + (long)iVar39 * uVar29 * 2 + lVar46 * 2) << 0x10`); the dot is `FUN_00161260(hidden_i, row, H)` (exact accumulator, float returned in a uint, `00161260.c:3`). It is printed as `unsigned long` of the 32-bit pattern (`00125240.c:89` `_M_insertImEERSoT_(local_1c0,uVar1)`, `uVar1` a uint). So each value is the IEEE-754 float32 bit pattern in unsigned decimal. Decode with `struct.unpack('<f', struct.pack('<I', v))`.
- `t1` = whole request time in microseconds; `t2` = literal 0 (section 1).

SCO/CLS (`0012b110.c:1121-1170`, prefix from the compare against 0xef176 `SCO` or 0xef17a `CLS`, `:1123-1131`): `SCO|CLS <tag> <bucket> <rowlen> <scores...> <t1> <t2>`. `<rowlen>` is the size of the first score vector (`:1144-1148`), equal to the engine `rows` count (2 for SCO, 2..16 for CLS). No argmax list. Scores are n groups of `rowlen` float32 bit patterns, in request order (`:1150-1157`).

EMB (`0012b110.c:1175-1215`, prefix bytes `454d 4220` "EMB " at 0xef445): `EMB <tag> <bucket> <k> <values...> <t1> <t2>`. `<k>` is `uVar32 = local_218` (`:1124`, printed `:1183` `_ZNSolsEi(uVar34,uVar32)`), the request's k. EMB does not echo the pad id. Values are `n*k` float32 bit patterns (list at +0x70, unsigned decimal). Each sequence's block is `FUN_00161840(hidden_i, k, out)`: the first k components of the RMS-normed hidden state divided by their L2 norm. Proof: `00161840.c:299-312` sums squares in an exact bit array, `:310-311` `dVar18 = ldexp((double)uVar14,uVar8 - 0x160); dVar18 = SQRT(dVar18);`, then `out[i] = in[i] / dVar18` (`:312-318`); non-finite input writes `DAT_001ef018`, float NaN (bytes `0000c07f` at 0xef018). Call site: `0014a6f0.c:960-961` `FUN_00161840(*(long *)(param_3 + 0x28) + lVar21,(long)param_2[0x1a],*(long *)(param_3 + 0x70) + param_2[0x1a] * lVar33 * 4)`.

## 5. GEN: remaining items

Sources: `001ba230.c` (`param_5` is the stats struct), `0012a4a0.c` (r2).
- `t1`: `param_5+10`, `001ba230.c:1310` `dVar114 - dVar113`, with `dVar113 = FUN_00155ee0()` at function entry (`:255`). Covers validation, bucket choice, setup and the whole prefill. Microseconds.
- `t2`: `param_5+0xc`, `:2469` `dVar114 - dVar113`, with `dVar113` re-sampled at `:1311` (decode loop start). Decode loop wall time, microseconds.
- `t3`: `param_5+0xe`, `:2135` `*(double *)(param_5 + 0xe) = *(double *)(param_5 + 0xe) + local_c60[0];`. `local_c60[0]` is the out-parameter of `FUN_00157b70`, set at `00157b70.c:108-109` (`*param_4 = (double)lVar8 / DAT_001f4bd8 - dVar22;`) after the call at `00157b70.c:102` (`_ZNK3xrt3run4waitERKNSt6chrono8durationIlSt5ratioILl1ELl1000EEEE`). So `t3` is the summed host wait on the XRT NPU run, microseconds after print scaling.
- `t4`: `param_5+0x10`, `:2353-2354` `(dVar117 - dVar116) + (dVar115 - dVar114) + old`. `dVar116` is taken just after the wait returns (`:2132`), `dVar117` just after the sampler `FUN_001b8540` returns (`:2349`), and `dVar115 - dVar114` is the span before `FUN_00157b70` is called (`:2125`). So `t4` is host-side time per step (sampler plus pre-submit work), microseconds.
- Field 9 (string): `param_5+0x12` assigned at `001ba230.c:647`. Disassembly: `0x000bab8c mov rsi, r13`, `0x000baba0 call ..._M_assign...` and `0x000baae9 mov rsi, r13`, `0x000baaf2 call 0xb43c0`; the same register is `lVar54`, the 2nd argument of `FUN_001b43c0` at `:628`. `lVar54` is chosen at `:358-380`: a route whose role string is `'g'` (`**(char **)(lVar44 + 0x20) == 'g'`), `+0x6c == 3`, batch `+100 == 1`, `+0x60 >= prompt_len - 1`, smallest `+0x60` wins; it is 0 for a one-token prompt (`:354`), printed as `-`. So field 9 is the name of the batch-1 prefill route. The selection rule is proof; reading the string as the route "name" is inferred.
- Field 10 (u32): `param_5[0x1a] = local_c84` (`:448`), the smallest listed decode-cache size `b` with `max_new + n_prompt <= b*16` (`:332-338`). Unit is 16-position blocks. Proof.

## 6. PFX

Request: `PFX tag n id1..idn`, `0 <= n <= 0x100000`, total tokens equal `n + 3` (`0012b110.c:405-409` `(0x100000 < uVar36)`, `bVar59 = param_3[1] - *param_3 >> 5 == uVar36 + 3;`); ids are strtol ints from token 3 (`:412-424`). Reply `PFX <tag> <a> <b>` (`:433-443`).
- `a` = `local_360[0]` = `*param_3 = iVar3` (`0014d2b0.c:549`): the token length of the cached prefix actually built. `FUN_0014d2b0` walks routes whose role string is `'p'` (`:186` `puVar1[5] != 1 || *(char *)puVar1[4] != 'p'`), reads each route's integer attribute `"P"` (`:190` `FUN_00162ba0(lVar8, DAT_001efd96, 0)`; bytes at 0xefd96 are `P\0`), and keeps the largest P with `iVar3 < P <= n` (`:195` `if ((iVar7 <= iVar3) || ((int)(param_2[1] - *param_2 >> 2) < iVar7)) goto next`; `:199` `iVar3 = iVar7`). With no qualifying route or n == 0 it returns 0 and both `a` and `b` stay 0 (`:125-126` zero-init, `:202-203` `if (iVar3 == 0) return 0;`).
- `b` = `local_340[0]` x 1000, elapsed build time in microseconds (`:552` `*param_4 = dVar31 - dVar30;`, scaled per section 1).
- The P values are prefill block sizes declared in the manifest for the 'p' routes. The loop is proof; calling them blocks is inferred.

## 7. IMG, ELOAD, EFREE

IMG (verb at 0xef3cc; parser `0012b110.c:1236-1296`):
- Request `IMG tag side seed N id1..idN`. Three strtoll at tokens 2, 3, 4 (`:1260-1263`, stored to `local_540[0]`, `local_540[1]`, `local_530`), then `local_530 - 1U < 0x200` and `tokens == local_530 + 5` (`:1268-1269`), so `1 <= N <= 512`. The N ids start at `*param_3 + 0xa0` (`:1318`).
- `side`: must be `> 0` (`:1364` `if (0 < iVar55)`, `iVar55 = local_540._0_4_`); looked up as lower_bound in an int-keyed map at `engine+0x1200` (`:1365-1376` `iVar55 <= *(int *)(lVar43 + 0x20)`); passed as `param_3` of `FUN_001df900` (`:1394-1396`). The output buffer is `param_3 * param_3 * 3` bytes (`001df900.c:612-615` `lVar47 = (long)param_3 * (long)param_3; ... uVar68 = lVar47 * 3;`). So `side` is the square image edge in pixels, RGB8. The reply is `IMG <side> ...` plus base64 of the buffer (alphabet literal at `0012b110.c:1430`; `:1408-1409` prints `uVar6 = local_540._0_8_`; the 5 bytes at 0xef3f6 are `IMG \0`). The exact reply tag layout beyond `IMG ` was not traced.
- `seed`: passed as `param_4` to `FUN_001df900`, then `FUN_001e1f80(param_4, param_3, buf)` (`001df900.c:240`), which seeds from `param_1 & 0xffffffff` (`001e1f80.c:72` `local_1400[0] = param_1 & 0xffffffff;`) and draws Gaussian noise with Box-Muller (`logf` `:252`, `sqrtf` `:261`, `sincosf` `:276`). Negative seed is malformed (`0012b110.c:1274`). So `seed` is the 32-bit seed for the initial latent noise. The other branch on `*(char *)(param_1 + 0x12f8)` (`001df900.c:236-240`) is not decoded.
- `N`, ids: text-encoder prompt token ids. Inferred from the limit and the strings "an image model answers IMG" (`0012b110.c:183`) and "no text-to-image model is loaded" (`:1296`).

ELOAD `tag path cap`: 4 tokens (`00172600.c:113` size test `== 0x80`). `cap` goes to `FUN_00170ea0(..., cap, ...)` (`00172600.c:135-136`). Entries qualify only if `(entry_len / 16) <= cap` (`00170ea0.c:263-272`: `iVar8 = iVar22 >> 4 ... iVar8 <= param_3`) and the device allocation is `cap * 0x4800` bytes (`0016cea0.c:190` `FUN_001579a0(uVar3, param_2, &local_58, (long)param_3 * 0x4800, param_5)`), also kept in the entry (`0016cea0.c:35` `*param_1 = param_3;`). So `cap` is the entry capacity in 16-position blocks; the arithmetic is proof, the "KV cache" reading is inferred. Reply `OK <tag> <id>`: `id` comes from `FUN_00172370`, a post-increment counter on the singleton from `FUN_0016b6d0` (`00172370.c:19-22` `iVar2 = *(int *)(lVar3 + 0x30); *(int *)(lVar3 + 0x30) = iVar2 + 1;`), and is the key of the new map node (`*(int *)(lVar4 + 0x20) = iVar2`). It is a handle, not a count; r2 called it `count`.

EFREE `tag id`: 3 tokens (`00172600.c:204` rejects other sizes via `!= 0x60`). The int is that handle: strtol at `:203`, then a red-black tree key-range erase (`:207-258`, `_Rb_tree_rebalance_for_erase` at `:243`). An unknown id erasing nothing is inferred from the range-erase shape.

## 8. Not closed, with reasons

None of these is a numeric field of the r2 table.
1. HASH_A, HASH_B, MISMATCH in the GEN reply: the algorithm is proof (FNV-1a, basis 0x14650fb0739d0383 and prime 0x100000001b3; the `movabs` pair at 0xbaba5 and 0xbabb4 confirms the constants), but what the hashed buffers are and what MISMATCH compares against is runtime state inside the roughly 2500-line decode loop (`001ba230.c:2189-2203`). Not traced.
2. The `"o"` route: `cVar14 = FUN_00139a60(puVar27 + 4, "o")` (`0014a6f0.c:226`; "o" at 0xf0fbc) makes every sequence read slot 0's hidden row (`:942-945`). Which shipped route is named `o` is in `devices/*.hnpm`, which is not in the repo.
3. LOOKUP, QHN, QLED, EDUMP argument lists and the HN/LEDGER mapping: carried from r2, untouched.
4. Bucket string source and the manifest section holding `pad` rest on register traces (sections 2, 4); confirming either needs a real `devices.hnpm`.
5. IMG: `side` is bounded by a lower_bound on a map key, so the engine may map it to a larger key or reject it later; not traced into `FUN_001df900`.

## Reproduce

```
cd /Users/peterb/Documents/AI/re-halogen/work/decomp/halogen-npu
sed -n 1,140p 0011c0b0.c
grep -n 'param_2' 0014a6f0.c
sed -n 120,330p 0014a6f0.c
sed -n 920,1075p 0014a6f0.c
grep -n 'iVar17\|uVar29 =' 0014a6f0.c
grep -n 'uStack_178\|local_180 = local_180\|local_130 =' 0014a6f0.c
sed -n 495,640p 0012b110.c
sed -n 800,822p 0012b110.c
sed -n 903,960p 0012b110.c
sed -n 1090,1215p 0012b110.c
sed -n 1236,1296p 0012b110.c
sed -n 1377,1412p 0012b110.c
sed -n 391,450p 0012b110.c
cat 00119ec0.c 00162ba0.c 00155ee0.c
sed -n 296,324p 00161840.c
grep -n 'param_' 00161060.c
sed -n 436,500p 00145eb0.c
sed -n 520,590p 00145eb0.c
sed -n 62,98p 00125240.c
grep -n 'param_5' 001ba230.c
sed -n 322,382p 001ba230.c
sed -n 100,112p 00157b70.c
sed -n 123,203p 0014d2b0.c
sed -n 536,553p 0014d2b0.c
sed -n 112,160p 00172600.c
sed -n 186,260p 00172600.c
sed -n 262,272p 00170ea0.c
grep -n 'param_3' 0016cea0.c
grep -n 'param_3\|param_4' 001df900.c
grep -n 'param_1\|logf\|sqrtf\|sincosf' 001e1f80.c
python3 -c "d=open('/Users/peterb/Documents/AI/re-halogen/binaries/halogen-npu','rb').read(); [print(hex(a), d[a:a+8]) for a in (0xef433,0xef172,0xef176,0xef17a,0xef17e,0xef3cc,0xef3f6,0xef3fb,0xefb16,0xefd83,0xefd96,0xefa38,0xf0fbc)]"
python3 -c "import struct; d=open('/Users/peterb/Documents/AI/re-halogen/binaries/halogen-npu','rb').read(); print([struct.unpack('<d',d[a:a+8]) for a in (0xf4bd0,0xf4bd8)], struct.unpack('<f',d[0xef014:0xef018]), d[0xef018:0xef01c].hex())"
ssh ai-core 'cd ~/re-halogen && r2 -q -e scr.color=0 -e bin.relocs.apply=true -c "s 0x4a815; pd 26" binaries/halogen-npu'
ssh ai-core 'cd ~/re-halogen && r2 -q -e scr.color=0 -e bin.relocs.apply=true -c "s 0x4a89a; pd 24" binaries/halogen-npu'
ssh ai-core 'cd ~/re-halogen && r2 -q -e scr.color=0 -e bin.relocs.apply=true -c "s 0xbab70; pd 22" binaries/halogen-npu'
ssh ai-core 'cd ~/re-halogen && r2 -q -e scr.color=0 -e bin.relocs.apply=true -c "/x 83039d73b00f6514" binaries/halogen-npu'
```

## Addendum (orchestrator, 2026-10-10): the "o" route, from the shipped manifest

The manifest is in the repo: `analysis/evidence/npu-decider-0.8b/devices.hnpm` (sha256 a8ee9327...; copy of `work/npu/elf/devices/devices.hnpm` on Ai-core). Its bucket lines carry a `kind` key: `bucket kind=o S=128 B=1 M=256 xd=0 name=u0` (line 11) and `kind=g` for u1 to u4 (S 256, 1024, 4096 at B=1, and S=256 B=8). So the `"o"` compared at `0014a6f0.c:226` is a bucket kind, and only u0 (the 128-token, batch-1, xd=0 bucket) has it: for that bucket every sequence reads slot 0's hidden row (`:942-945`). That settles item 2's "which shipped route" for this device set. The word `pad` does not occur in the manifest (`grep -c pad` gives 0), so item 4's pad source is not a manifest key in this file; still open.
