#!/usr/bin/env python3
"""Carve AMDGPU code objects out of flash_serve's .hip_fatbin.

Usage: carve_fatbin.py <elf> <section-offset> <section-size> <outdir>

The .hip_fatbin is a clang offload bundle v2 (magic + num_entries u64 +
table_size u64, entry table at section end). Rather than walk the table
(its entry layout varies by clang version), this carves every ELF image in
the section, sizes each from its own section/program headers, verifies
e_machine == 0xe6 (EM_AMDGPU), extracts FUNC symbols, and pulls the
.amdgpu_target / kernel metadata from .note records.
Prints a JSON manifest.
"""
import hashlib
import json
import os
import re
import struct
import subprocess
import sys


def elf_size(b, off):
    end = 0
    shoff = struct.unpack_from("<Q", b, off + 0x28)[0]
    shentsize = struct.unpack_from("<H", b, off + 0x3A)[0]
    shnum = struct.unpack_from("<H", b, off + 0x3C)[0]
    for i in range(shnum):
        e = off + shoff + i * shentsize
        sh_type = struct.unpack_from("<I", b, e + 4)[0]
        sh_off = struct.unpack_from("<Q", b, e + 0x18)[0]
        sh_size = struct.unpack_from("<Q", b, e + 0x20)[0]
        if sh_type != 8:
            end = max(end, sh_off + sh_size)
    phoff = struct.unpack_from("<Q", b, off + 0x20)[0]
    phentsize = struct.unpack_from("<H", b, off + 0x36)[0]
    phnum = struct.unpack_from("<H", b, off + 0x38)[0]
    for i in range(phnum):
        p = off + phoff + i * phentsize
        p_off = struct.unpack_from("<Q", b, p + 8)[0]
        p_filesz = struct.unpack_from("<Q", b, p + 32)[0]
        end = max(end, p_off + p_filesz)
    end = max(end, shoff + shnum * shentsize)
    return end


def main():
    elf_path, sec_off, sec_sz, outdir = (sys.argv[1], int(sys.argv[2], 0),
                                         int(sys.argv[3], 0), sys.argv[4])
    os.makedirs(outdir, exist_ok=True)
    data = open(elf_path, "rb").read()
    manifest = {"source": elf_path, "section_offset": hex(sec_off),
                "section_size": sec_sz, "blobs": []}
    j = sec_off
    while True:
        j = data.find(b"\x7fELF", j, sec_off + sec_sz)
        if j < 0:
            break
        sz = elf_size(data, j)
        content = data[j:j + sz]
        em = struct.unpack_from("<H", content, 18)[0]
        rec = {"offset": hex(j), "size": sz, "e_machine": hex(em),
               "sha256": hashlib.sha256(content).hexdigest()}
        if em in (0xE0, 0xE3):  # EM_MACHINE (AMD GCN family) / EM_AMDGPU
            name = f"codeobj_{j:x}.hsaco"
            path = os.path.join(outdir, name)
            with open(path, "wb") as f:
                f.write(content)
            rec["file"] = path
            out = subprocess.run(["readelf", "-sW", path], capture_output=True,
                                 text=True).stdout
            syms = [p.split()[-1] for p in out.splitlines()
                    if len(p.split()) >= 8 and p.split()[3] == "FUNC"
                    and p.split()[7] not in ("", "SECTION", "FILE")]
            rec["symbols"] = len(syms)
            rec["symbol_names"] = syms
            note = subprocess.run(["readelf", "-x", ".note", path],
                                  capture_output=True, text=True,
                                  errors="replace").stdout
            m = re.findall(r"gfx[0-9a-z]+", note)
            rec["targets"] = sorted(set(m))
        manifest["blobs"].append(rec)
        j += 4
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
