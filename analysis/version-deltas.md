# Version deltas — halogen-flash-server 0.11.10 → 0.17.3

Date: 2026-10-10. Method: `scripts/pull_versions.sh` (registry layer pulls,
`work/versions/<v>/`), `scripts/carve_fatbin.py` per version (fatbin v2 →
code objects + symbol sets), `scripts/version_deltas.py` (kernel symbol-set
diffs + host string-set diffs). Raw evidence:
`work/versions/*/fatbin-manifest.json`, `tmp/deltas_raw.txt`.
All code objects are gfx1151 (EF flag 0x4a).

## 1. Code-object lineage (sha256 identity across versions)

Each release ships 3-4 AMD code objects in `.hip_fatbin`. Hashing every blob
across all 9 pulled versions shows the device code is a small set of
recompiled units, not a monolith rebuilt every time:

| blob (sha256[:16]) | syms | versions (identical bytes) |
|---|---|---|
| dbaafa57d9b3bd60 | 124 | 0.15.0, 0.16.4, 0.17.0, 0.17.2, 0.17.3 |
| 3e44d5a242b83af3 | 350 | 0.16.4, 0.17.0, 0.17.2, 0.17.3 |
| 5a82ad4cb1c64b6b | 34 | 0.16.4, 0.17.0, 0.17.2, 0.17.3 |
| 18937428b544e8a5 | 2982 | **0.17.2, 0.17.3** (main object) |
| 3aa5102d6e1ae3ce | 2976 | 0.16.4, 0.17.0 (main object) |
| 6d7d75527ec7ff39 | 2480 | 0.15.0 (main object) |
| 2adffc37de61309b | 2004 | 0.12.0, 0.13.3, 0.13.4 (main object) |
| 290800f862c07888 | 1994 | 0.11.10 (main object) |
| f578807ed0564d0a | 340 | 0.11.10, 0.12.0 |
| 0a9bd2a0e8b2f20f | 340 | 0.13.3, 0.13.4 |
| 2e788bf59fb00b80 | 120 | 0.12.0, 0.13.3, 0.13.4 |
| 2d586ee12ae40ded | 344 | 0.15.0 |
| 2e28a5b83562be22 | 118 | 0.11.10 |

Reading:
- **0.17.2 → 0.17.3 changed zero device code** (all four blobs byte-identical;
  the 0.17.3 release is host-side only).
- 0.16.4 → 0.17.0 likewise: fatbin byte-identical (all four blobs).
- The 0.17.2 main object (2982 syms) is the current line; 0.16.4/0.17.0
  share the previous one (2976); 0.15.0 the one before (2480).
- 0.12.0/0.13.3/0.13.4 share one main object (2004) — the 0.13.x line
  shipped no kernel changes at all.
- The 340-sym object recompiled between the 0.12.x and 0.13.x lines
  (two hashes, same symbol count).

## 2. Kernel symbol-set deltas

From `tmp/deltas_raw.txt` (union of FUNC symbols across a version's blobs):

| transition | +added | -removed | total |
|---|---|---|---|
| 0.15.0 → 0.16.4 | +318 | −50 | 1474 → 1742 |
| 0.16.4 → 0.17.0 | 0 | 0 | 1742 |
| 0.17.0 → 0.17.2 | +3 | 0 | 1745 |
| 0.17.2 → 0.17.3 | 0 | 0 | 1745 |

**0.15.0 → 0.16.4** (the big one, +318): new families `k_hc6_fused`/
`k_hc8_fused` (8 template widths each), `k_ht_tg_sum`, `k_ple_out_p`,
`k_dn_fused16` (the MoE down-proj fused kernel, 12+ tap variants),
`k_ht_lin_grp` retarget (−50 = old `k_ht_lin_grp` instantiations replaced
with a new template shape), `k_dnc_scan3x` retarget (DncTapArgs → new
signature). Net: HT (rotation-plane quant) and dn (down-proj fused)
template expansions.

**0.17.0 → 0.17.2** (+3): `k_dv_quant`, `k_dv_quant8`, `k_dv_scatter` —
the draft-verify quant/scatter support kernels for sampled speculation
(see §3).

## 3. Host string deltas, 0.17.2 → 0.17.3 (+13 −5)

Added (excluding the 5 build-date stamps):

```
HALOGEN_SAMPLED_DEPTH
HALOGEN_SAMPLED_GATE
HALOGEN_WALK_FALSIFY
draft_sample: q row %d was not kept
residual_sample: q row %d was not kept
q_keep: row %d outside 0..%d
flash_serve: HALOGEN_SAMPLED_DEPTH=%s needs 1 <= D and HALOGEN_SPEC_N >= D + 1 (is %d)
flash_serve: HALOGEN_SAMPLED_GATE=t1[,t2] needs values in [0, 1]
```

Reading: 0.17.3 is the release that wires **sampled (probabilistic)
speculation** end-to-end on the host side: a depth knob (`SAMPLED_DEPTH`
over the spec window `SPEC_N`), a two-threshold gate (`SAMPLED_GATE=t1,t2`
in [0,1]), and a falsifier walk mode (`WALK_FALSIFY`). The
`q_keep`/`was not kept` messages are the host-side guards for the
`k_dv_quant`/`k_dv_scatter` device path added one release earlier: the
engine keeps a quantized copy of sampled rows ("q rows") for the
acceptance walk, and refuses to run if a row wasn't kept.
Nothing removed except build stamps.

## 4. Family counts across the whole timeline

Per-kernel-name family counts (union of symbol sets; full table in
`tmp/deltas_raw.txt` + this section's probe):

| family | 0.11.10 | 0.12.0-0.13.4 | 0.15.0 | 0.16.4-0.17.0 | 0.17.2-0.17.3 |
|---|---|---|---|---|---|
| lq4w (4-bit linear) | 448 | 448 | 448 | 448 | 448 |
| lq6k / lq8s32 / lbfw | 64/64/16 | same | same | same | same |
| lq6w | 0 | 0 | 32 | 32 | 32 |
| hcup4 | 12 | 12 | 12 | 12 | 12 |
| hcup6/hcup8 | 0 | 0 | 4/4 | 4/4 | 4/4 |
| hcup6w/hcup8w | 0 | 0 | 0 | 4/4 | 4/4 |
| q4moe (namespace q4moe) | 0 | 0 | 20 | 20 | 20 |
| hc6_fused/hc8_fused | 0 | 0 | 0 | 16 | 16 |
| dn_fused16 | 0 | 0 | 0 | 30+ | 30+ |
| smpg_*/accept/ngram | present | present | present | present | present |
| pen_rep_r | 0 | 0 | 0 | 1 | 1 |
| dv_quant/dv_quant8/dv_scatter | 0 | 0 | 0 | 0 | 3 |

The sampling/n-gram speculation core is stable across the entire timeline;
what churns is the quantized-GEMM template space (new widths, fused
variants) and, at the tail, the sampled-speculation trio.

## 5. Changelog cross-check

`work/CHANGELOG.md` claims map as follows: the 0.16.x "HT rotation-plane
checkpoint support" claim is corroborated by the +318 HT/dn template
expansion (§2) and the HT geometry rules in `analysis/hgn-format.md`;
the 0.17.2 "draft-verify quantization" line by the +3 dv kernels;
the 0.17.3 "sampled speculation with falsifier walk" line by §3.
No changelog claim examined in this pass was contradicted by the binaries.

## Bounded-open

- 0.11.10 → 0.15.0 fine-grained per-release deltas (0.14.x, 0.15.x
  patch releases not pulled; the pulled set brackets the big change).
- Whether `HALOGEN_WALK_FALSIFY` changes the acceptance *math* or only the
  *audit* path: kernel-level answer in `analysis/kernels-speculation.md`
  (Phase 2, sibling writeup).
