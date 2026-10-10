# Phase 0 — Runtime corroboration

Date: 2026-10-09. Target family: halogen-flash-server 0.17.3.

## Live probes: SKIPPED (documented, not silent)

- Engine port 127.0.0.1:8730: **connection refused** at run time (the
  halogen-family engine pid 2918 referenced in the plan is gone; `ps -p 2918`
  empty). Per PLAN.md Phase 0.1: "If 8730 refuses PING, skip the live probes
  and note it — do not start a new engine instance." No engine was started.
- 127.0.0.1:8731 is up but belongs to the **bits-live llama-server stack**
  (`/v1/models` = qwen3-embedding-0.6b, qwen3-reranker-0.6b, decider-0.8b,
  qwen3guard-gen-0.6b) — explicitly NOT the halogen target. Not probed
  beyond `/health` + `/v1/models` (read-only GETs).
- Consequence: protocol-grammar claims below are corroborated **statically**
  (vendor-shipped `serve_api.py` client code + engine format strings), not
  by live exchange. Marked "inferred (client-side)" vs "observed (live)".

## Protocol grammar (static corroboration)

Engine line protocol, from `image/serve_api.py` (vendor-shipped, comments
stripped) cross-checked against engine strings in `binaries/flash_serve`:

| verb | evidence |
|---|---|
| `INFO` | serve_api `_probe()` writes `INFO\n`, expects `I ...` (serve_api.py:855-886); engine string `queued connection asked INFO while a session was running` (flash_serve @0x41ba5) |
| `GEN` | report.md §1; engine request-admission string `req %ld admitted: slot %d, region base %d cap %d, prompt %d, drafter %d, %s` (@0x31253) |
| `PING`/`PONG` | report.md §1 (settled Phase-1 finding) |
| `VIMG`/`SCHEMA`/`THINK` | engine strings @0x36e83, @0x41c56, @0x29854 |
| `GUARD`/`SEG`/`X` | report.md §1 (settled) |

### Capability probe (`INFO` → `I ...`) — fully decoded

Engine reply format string (flash_serve, file offset **0x1b4d8**):

```
I %d 0 %d %d %d %d 0 %lld %d %d %d %d 1 %g %lld %d %s %d %d %d %d %lld %s %d
```

Field-by-field against `serve_api._parse_info` (serve_api.py:912-940), token
index = `p[i]`:

| p[i] | format token | field | 0.17.3 value |
|---|---|---|---|
| 1 | `%d` | mtp | runtime |
| 2 | **`0` literal** | draft_head | **hard-wired 0** |
| 3 | `%d` | ctx | runtime |
| 4 | `%d` | spec_rows | runtime |
| 5 | `%d` | default | runtime |
| 6 | `%d` | drafter_weights | runtime |
| 7 | **`0` literal** | dflash2 | **hard-wired 0** |
| 8 | `%lld` | cache_mb | runtime |
| 9-12 | `%d` | cache_align, kv_slots, slot_ctx, cache_mode | runtime |
| 13 | **`1` literal** | sampling | **hard-wired 1** |
| 14 | `%g` | rope_factor | runtime |
| 15 | `%lld` | kv_pool | runtime |
| 16 | `%d` | vision | runtime |
| 17 | `%s` | version | runtime |
| 18-19 | `%d` | pld_n, pld_k | runtime |
| 20 | `%s` | checkpoint_format | runtime |
| 21-22 | `%d` | grammar, composable | runtime |
| 23-25 | `%d %d %lld` | cc_floor, cc_seam, cc_bytes | runtime |
| 26 | `%s` | cc_fingerprint | runtime |
| 27 | `%d` | indexer_budget | runtime |

The token count and every `%`-slot line up 1:1 with `_parse_info`'s 28-field
parse — this is the exact emitter. Three capability bits are compile-time
constants in 0.17.3: `draft_head=0`, `dflash2=0`, `sampling=1`.

### Done-line (`D ...`) grammar

Engine format string @0x14e39:

```
D %ld %s %d %d %.1f %.1f %d %ld %ld %d %ld %ld %ld %ld %d %d %d %d %d %d %d %d %.1f %ld %ld %ld %ld
```

`serve_api._done_dict` (serve_api.py:957+) reads parts[2]=reason,
[3]=n_prompt, [4]=n_gen, [5]=prefill_ms, [6]=decode_ms, and (len≥10)
[7]=drafter, [8]=rounds, [9]=commit — consistent with the format string.
The `drafter/rounds/commit` triple is the speculation telemetry channel the
falsifier argument uses (report.md §3).

## Changelog claims verification

Deferred to `version-deltas.md` (in progress: pulling 0.17.2 / 0.17.0 /
0.16.4 / 0.15.0 layers from ghcr.io; anonymous token flow works).
