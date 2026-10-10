#!/usr/bin/env python3
"""Slice llvm-objdump output into per-kernel files.

Usage: slice_kernels.py <disasm.s> <outdir> <name-regex>
Kernel blocks in objdump output start at labels:
  0000000000008500 <SYMNAME>:
A block runs to the next label line.
"""
import os
import re
import sys

def main():
    src, outdir, pat = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    rx = re.compile(pat)
    lab = re.compile(r'^[0-9a-f]{16} <(.+)>:$')
    cur = None
    out = None
    n = 0
    with open(src, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = lab.match(line)
            if m:
                if out:
                    out.close()
                    out = None
                name = m.group(1)
                if rx.match(name):
                    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:120]
                    out = open(os.path.join(outdir, safe + ".s"), "w")
                    cur = name
                    n += 1
            if out:
                out.write(line)
    if out:
        out.close()
    print(f"{src}: {n} kernels sliced to {outdir}")

if __name__ == "__main__":
    main()
