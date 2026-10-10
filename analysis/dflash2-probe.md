# Phase 5 — the dflash2 drafter: probe-field trace

Date: 2026-10-10. Question (report.md §6.4): the client API exposes a
drafter named `dflash2` (id 2); flash_serve 0.17.3 has no matching strings.
Where does the capability come from, and can it ever be on?

## 1. The client side (vendor-shipped `image/serve_api.py`)

- `DRAFTERS = {"serial": 0, "mtp": 1, "dflash2": 2}` (serve_api.py:1889)
- `DRAFTER_CAP = {"mtp": "mtp", "dflash2": "dflash2"}` (:1890)
- `drafter_id()` (:2204) maps the request's `drafter` name to the id, and
  **rejects any drafter whose capability bit is false**:
  `400 drafter 'dflash2' unavailable: the loaded checkpoint cannot serve it`
  (:2214). The capability dict comes from `_parse_info` (:918):
  `"dflash2": len(p) >= 8 and p[7] == "1"` — token 7 of the engine's
  `I ...` INFO reply.
- The done-line maps `d.get("drafter")` 1→"mtp", 2→"dflash2" (:2404), so
  the telemetry channel is wired for it end-to-end.

## 2. The engine side (flash_serve 0.17.3, `binaries/flash_serve`)

The INFO emitter is the `snprintf` in `main` (Ghidra
`analysis/main_pseudo.c:852`; file vaddr 0x172d755):

```
"I %d 0 %d %d %d %d 0 %lld %d %d %d %d 1 %g %lld %d %s %d %d %s %d %d %d %d %lld %s %d"
```

Token positions p[2] (draft_head), p[7] (dflash2), p[13] (sampling) are
**literal characters in the format string** — `0`, `0`, `1` — not `%`
slots. The snprintf argument list (main_pseudo.c:853-858) confirms: the
`0`/`0`/`1` never appear among the arguments. In 0.17.3 the engine
answers `dflash2=0` unconditionally, at compile time. There is no code
path, env flag, checkpoint, or runtime state that can raise it.

Engine-side drafter handling that *does* exist:

- The `GEN` line carries an optional integer drafter id (client
  `generate()` appends `" {drafter}"`, serve_api.py:1077).
- Admission trace: `req %ld admitted: slot %d, region base %d cap %d,
  prompt %d, drafter %d, %s` (rodata 0x31253; printf site 0x173d5e0,
  inside `main`'s inlined connection loop).
- The only drafter-rejection message: string pieces `asked for drafter `
  (0x31240, built at 0x1738c18 with an itoa of the requested id) +
  ` (MTP head not staged, HALOGEN_MTP=1)` (0x260c4) — i.e. the engine
  validates drafter ids against the staged MTP head only.
- Startup banner: `MTP depth-%d ... selectable (drafter 1, runs alone)` /
  `selectable (drafter 1, speculates while alone)` (0x3d3ff / 0x2cb2c,
  main_pseudo.c:1045,1054). Drafter 1 = the MTP head. No drafter 2 is
  ever described.

## 3. Version sweep

`strings -a` over every pulled engine (0.11.10, 0.12.0, 0.13.3, 0.13.4,
0.15.0, 0.16.4, 0.17.0, 0.17.2, 0.17.3): zero hits for `dflash` in any
engine binary. The string exists only in the vendor's Python front-end.
`work/CHANGELOG.md` never mentions dflash2.

## 4. Verdict

`dflash2` is a **reserved wire-protocol drafter slot**, not a feature of
any shipped engine:

1. The id (2), the capability token position (INFO p[7]), and the
   done-line mapping are fully wired in the protocol and the client.
2. Every engine from 0.11.10 through 0.17.3 hard-wires the capability to
   0 in the INFO format string; the client therefore 400s any request
   naming `dflash2` before the engine sees it.
3. The engine's GEN parser accepts an integer drafter field and validates
   only against the MTP head (drafter 1). A hand-built `GEN ... 2` sent
   directly to the engine socket bypasses the client gate; what the
   engine does with id 2 on that path is not observable from strings
   alone (bounded-open, §5).
4. The real "second drafter" that ships is the n-gram prompt-lookup
   (`HALOGEN_PLD=N,K`, `k_ngram_*` kernels), which runs *beside* the MTP
   head on greedy mtp requests (engine string: "prompt lookup off (it
   runs beside the draft head, and HALOGEN_MTP=0 turned the head off)";
   serve_api: prompt_lookup `applies_to: greedy requests with the mtp
   drafter`). It is reported through its own INFO fields (p[18], p[19])
   and done-line counters (pld_rounds/pld_accepted/pld_drafted), not
   through the drafter slot.

So: the capability "comes from the probe" exactly as AGENTS.md guessed,
and the probe answer is a compile-time constant no. dflash2 is
forward-compatibility scaffolding — a drafter id reserved for a future
draft head (the naming suggests a flash-attention-2-style second
drafter), shipped dark.

## 5. Bounded-open

- Engine behavior for a raw `GEN ... 2 ...` with the MTP head staged
  (accepted as mtp? rejected at admission? ignored?): requires running
  the engine (out of scope per AGENTS.md ground rules) or a full
  decompile of the GEN parser inside main's 12,829-instruction body.
- Whether a future engine release flips the format-string literal:
  unobservable from the pulled set; the field layout is stable across
  all 9 pulled versions.
