#!/usr/bin/env python3
"""Characterize sliced AMDGCN kernels: instruction-class counts per kernel.

Usage: kernel_stats.py <kernels-dir> <out-csv>
Counts per kernel: total instrs, MFMA (matrix cores), LDS ops (ds_*),
AGPR traffic (v_accvgpr_read/write), global/vmem loads/stores, crosslane
(v_permlane/v_swizzle/v_lanemask), DPP, v_cvt (dequant conversions),
v_pk_* packed ops, scale/min/max, exp (softmax), and the widest MFMA
variant seen (16x16x16 vs 32x32 etc).
"""
import csv
import os
import re
import sys

CLASSES = [
    ("mfma", re.compile(r"\bv_mfma_")),
    ("wmma", re.compile(r"\bv_wmma_")),
    ("lds", re.compile(r"\bds_")),
    ("agpr", re.compile(r"\bv_accvgpr_")),
    ("gload", re.compile(r"\bglobal_load_")),
    ("gstore", re.compile(r"\bglobal_store_")),
    ("buffer", re.compile(r"\bbuffer_(load|store)_")),
    ("smem", re.compile(r"\bs_load_b|\bs_store_b")),
    ("permlane", re.compile(r"\bv_permlane|\bv_swizzle|\bv_lanemask")),
    ("dpp", re.compile(r"\bdpp_")),
    ("cvt", re.compile(r"\bv_cvt_")),
    ("pk", re.compile(r"\bv_pk_")),
    ("scale", re.compile(r"\bv_scalef|\bv_mullo|\bv_fmak")),
    ("exp", re.compile(r"\bv_exp|\bv_exp2|\bv_log")),
    ("rcp", re.compile(r"\bv_rcp|\bv_rcp")),
    ("fma", re.compile(r"\bv_fma_f32|\bv_fma_f64")),
    ("minmax", re.compile(r"\bv_min_|\bv_max_")),
    ("cmp", re.compile(r"\bv_cmp_")),
    ("bfe_bfm", re.compile(r"\bv_bfe_|\bv_bfm_")),
    ("lit", re.compile(r"\bv_lit_")),
]


def main():
    kdir, out_csv = sys.argv[1], sys.argv[2]
    rows = []
    for fn in sorted(os.listdir(kdir)):
        if not fn.endswith(".s"):
            continue
        counts = {k: 0 for k, _ in CLASSES}
        total = 0
        mfma_shapes = set()
        with open(os.path.join(kdir, fn), encoding="utf-8", errors="replace") as f:
            for line in f:
                m = re.match(r"\t([a-z0-9_.]+)\s", line)
                if not m:
                    continue
                op = m.group(1)
                total += 1
                for k, rx in CLASSES:
                    if rx.match(op):
                        counts[k] += 1
                if op.startswith("v_mfma_"):
                    mm = re.match(r"v_mfma_(f16|f32|i32|g_f8|g_f16)_(\d+)x(\d+)x(\d+)", op)
                    if mm:
                        mfma_shapes.add(f"{mm.group(1)}_{mm.group(2)}x{mm.group(3)}x{mm.group(4)}")
        rows.append({"kernel": fn[:-2], "instrs": total,
                     "mfma_shapes": "|".join(sorted(mfma_shapes)), **counts})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} kernels -> {out_csv}")


if __name__ == "__main__":
    main()
