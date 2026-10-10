#!/usr/bin/env python3
"""Extract per-target code objects from flash_serve's .hip_fatbin.

Usage: extract_fatbin2.py <elf> <magic-file-offset> <outdir>

Verified layout (clang 23, flash_serve 0.17.3):
  clang offload bundle v2: magic(24) + entry-count u64 + table-size u64;
  entry table at (bundle_end - table_size). Each entry:
  triple_len u64, triple (NUL, padded to 8), encoding u32,
  content_offset u64, content_size u64, alignment u32.
  Offsets are relative to the bundle start.
The AMDGCN entry's content is a code object ELF (EM_AMDGPU). Additional
code-object ELFs live in the data region; we carve every ELF in the
section and size it from its own section/program headers.
Classifies blobs, extracts FUNC symbols, prints a JSON manifest.
"""
import hashlib
import json
import struct
import subprocess
import sys
import os

MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
ENCODINGS = {
    0: "SASS", 1: "HSACO", 2: "PTX", 3: "AMDGCN", 4: "BITS",
    5: "SPIRV", 30: "x86_64", 31: "aarch64",
}


def elf_size(b, off):
    """Total file size of the ELF image starting at off (max sh/ph end)."""
    shoff = struct.unpack_from("<Q", b, off + 0x28)[0]
    shentsize = struct.unpack_from("<H", b, off + 0x3A)[0]
    shnum = struct.unpack_from("<H", b, off + 0x3C)[0]
    end = 0
    for i in range(shnum):
        e = off + shoff + i * shentsize
        sh_type = struct.unpack_from("<I", b, e + 4)[0]
        sh_off = struct.unpack_from("<Q", b, e + 0x18)[0]
        sh_size = struct.unpack_from("<Q", b, e + 0x20)[0]
        if sh_type != 8:  # not SHT_NOBITS
            end = max(end, sh_off + sh_size)
    phoff = struct.unpack_from("<Q", b, off + 0x20)[0]
    phentsize = struct.unpack_from("<H", b, off + 0x36)[0]
    phnum = struct.unpack_from("<H", b, off + 0x38)[0]
    for i in range(phnum):
        p = off + phoff + i * phentsize
        p_off = struct.unpack_from("<Q", b, p + 8)[0]
        p_filesz = struct.unpack_from("<Q", b, p + 32)[0]
        end = max(end, p_off + p_filesz)
    return end


def symbols_of(path):
    try:
        out = subprocess.run(["readelf", "-sW", path], capture_output=True,
                             text=True, timeout=300).stdout
    except Exception:
        return []
    names = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[3] == "FUNC" and parts[7] not in ("", "SECTION", "FILE"):
            names.append(parts[-1])
    return names


def main():
    elf_path, magic_off, outdir = sys.argv[1], int(sys.argv[2], 0), sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    data = open(elf_path, "rb").read()
    bundle_start = data.index(MAGIC, magic_off)
    n = struct.unpack_from("<Q", data, bundle_start + 24)[0]
    table_size = struct.unpack_from("<Q", data, bundle_start + 32)[0]
    # section size from ELF section headers -> bundle end
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize = struct.unpack_from("<H", data, 0x3A)[0]
    shnum = struct.unpack_from("<H", data, 0x3C)[0]
    shstrndx = struct.unpack_from("<H", data, 0x3E)[0]
    shstr = struct.unpack_from("<Q", data, shoff + shstrndx * shentsize + 0x18)[0]
    sec_end = None
    for i in range(shnum):
        e = shoff + i * shentsize
        nm = data[shstr + struct.unpack_from("<I", data, e)[0]:].split(b"\x00")[0].decode()
        if nm == ".hip_fatbin":
            sec_off = struct.unpack_from("<Q", data, e + 0x18)[0]
            sec_sz = struct.unpack_from("<Q", data, e + 0x20)[0]
            sec_end = sec_off + sec_sz
    bundle_end = sec_end if sec_end else len(data)
    pos = bundle_end - table_size
    entries = []
    for _ in range(n):
        tl = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        triple = data[pos:pos + tl].rstrip(b"\x00").decode("utf-8", "replace")
        pos += (tl + 7) & ~7
        enc, coff, csize, align = struct.unpack_from("<IQQI", data, pos)
        pos += 24
        entries.append((triple, enc, coff, csize, align))
    print(f"bundle@{hex(bundle_start)} end={hex(bundle_end)} v2_entries={len(entries)} table@+{hex(table_size)}", file=sys.stderr)
    for t, e, c, s, a in entries:
        print(f"  {t} enc={e} off={hex(c)} size={hex(s)}", file=sys.stderr)

    manifest = {"source": elf_path, "bundle_offset": hex(bundle_start),
               "v2_entries": [], "carved_elfs": []}

    def emit(bucket, tag, name, enc, content, parent):
        enc_name = ENCODINGS.get(enc, f"ENC{enc}")
        safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in name)
        fname = f"{tag}_{safe}_{enc_name}.hsaco"
        path = os.path.join(outdir, fname)
        with open(path, "wb") as f:
            f.write(content)
        rec = {"tag": tag, "name": name, "encoding": enc, "encoding_name": enc_name,
               "parent": parent, "size": len(content), "file": path,
               "sha256": hashlib.sha256(content).hexdigest()}
        if content[:4] == b"\x7fELF":
            rec["e_machine"] = hex(struct.unpack_from("<H", content, 18)[0])
            syms = symbols_of(path)
            rec["symbols"] = len(syms)
            rec["symbol_names"] = syms
        manifest[bucket].append(rec)
        return path

    for i, (triple, enc, coff, csize, align) in enumerate(entries):
        content = data[bundle_start + coff:bundle_start + coff + csize]
        emit("v2_entries", f"v2_{i}", triple, enc, content, "bundle")

    # carve every ELF image in the bundle data region
    j = bundle_start
    end = bundle_start + table_size
    while True:
        j = data.find(b"\x7fELF", j, end)
        if j < 0:
            break
        sz = elf_size(data, j)
        content = data[j:j + sz]
        em = struct.unpack_from("<H", content, 18)[0]
        rec = {"offset": hex(j), "size": sz, "e_machine": hex(em),
               "sha256": hashlib.sha256(content).hexdigest()}
        if em == 0xE6:
            path = os.path.join(outdir, f"carved_{j:x}_gfx.hsaco")
            with open(path, "wb") as f:
                f.write(content)
            syms = symbols_of(path)
            rec["file"] = path
            rec["symbols"] = len(syms)
            rec["symbol_names"] = syms
        manifest["carved_elfs"].append(rec)
        j += 4
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
