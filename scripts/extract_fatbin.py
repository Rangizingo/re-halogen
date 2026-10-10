#!/usr/bin/env python3
"""Extract per-target code objects from a clang offload bundle v2 (clang 23).

Usage: extract_fatbin.py <elf> <magic-file-offset> <outdir>

v2 layout: magic(24) + entry-count u64, then entries:
  triple_len u64, encoding u32, content_offset u64, content_size u64,
  alignment u32, triple (NUL, padded to 8). Content blobs precede the table.
The AMDGCN entry's content is a v1 fatbinary container: data blobs, then
entries of (40-byte header + inline string padded to 8):
  ver u16, hdr_size u16, encoding u32, kind u32, content_offset u64,
  content_size u64, string_size u64, string_size2 u64, string...
followed by entry-count u64 + table-size u64 at the container's end.
Classifies blobs (ELF/EM_AMDGPU, LLVM bitcode, ZSTD/GZIP), pulls FUNC
symbols from code objects, prints a JSON manifest to stdout.
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
    5: "SPIRV", 6: "Dxil", 7: "IntelGPU", 8: "IntelGPU-BD",
    9: "IntelGPU-PVC", 10: "IntelGPU-DG2", 30: "x86_64", 31: "aarch64",
}
# v1 fatbinary encoding bitfield
V1_ENC = {1: "HSACO", 2: "CODE_OBJECT", 4: "ASSEMBLY", 8: "IR", 16: "BITCODE",
         32: "DEPENDENT_LIBRARY", 64: "TEXT", 128: "IR_CODE_OBJECT",
         256: "OFFLOAD_IMAGE", 512: "HOST_CODE", 1024: "DEBUG_INFO"}


def classify(b):
    if b[:4] == b"\x7fELF":
        e_machine = struct.unpack_from("<H", b, 18)[0]
        return {"kind": "ELF", "e_machine": hex(e_machine),
                "e_machine_name": "EM_AMDGPU" if e_machine == 0xE6 else "?",
                "e_type": struct.unpack_from("<H", b, 16)[0], "ei_class": b[4]}
    if b[:4] == b"BC\xC0\xDE":
        return {"kind": "LLVM-BITCODE"}
    if b[:3] == b"\x28\xb5\x2f":
        return {"kind": "ZSTD"}
    if b[:2] == b"\x1f\x8b":
        return {"kind": "GZIP"}
    return {"kind": "UNKNOWN", "head": b[:16].hex()}


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


def parse_v2(bundle):
    n = struct.unpack_from("<Q", bundle, 24)[0]
    pos = 40
    entries = []
    for _ in range(n):
        (tl,) = struct.unpack_from("<Q", bundle, pos)
        pos += 8
        triple = bundle[pos:pos + tl].rstrip(b"\x00").decode("utf-8", "replace")
        pos += tl
        pos = 40 + ((pos - 40 + 7) & ~7)
        enc, coff, csize, align = struct.unpack_from("<IQQI", bundle, pos)
        pos += 24
        entries.append((triple, enc, coff, csize, align))
    return entries


def parse_v1_fatbinary(fb):
    """v1 fatbinary container -> (entries, table_base).

    Entries are 56-byte headers (ver u16, hdr_size u16, encoding u32,
    kind u32, content_offset u64, content_size u64, string_size u64,
    string_size2 u64) followed by the inline string padded to 8.
    Entry-count u64 + table-size u64 sit at the container's end.
    """
    n = struct.unpack_from("<Q", fb, len(fb) - 16)[0]
    for guess in range(1, 4096):
        base = len(fb) - 16 - guess
        pos = base
        ok = True
        entries = []
        for _ in range(n):
            if pos + 56 > len(fb) - 16:
                ok = False
                break
            ver, hdrsz, enc, kind, coff, csize, ss, ss2 = struct.unpack_from("<HHIIQQQQ", fb, pos)
            pos += 56
            name = fb[pos:pos + ss].split(b"\x00")[0].decode("utf-8", "replace")
            pos += (ss + 7) & ~7
            entries.append((name, enc, coff, csize))
        if ok and pos == len(fb) - 16:
            return entries, base
    raise SystemExit("v1 fatbinary table parse failed")


def section_size(elf_bytes, name):
    """Size of a named section from ELF section headers."""
    import io
    import struct as st
    shoff = st.unpack_from("<Q", elf_bytes, 0x28)[0]
    shentsize = st.unpack_from("<H", elf_bytes, 0x3A)[0]
    shnum = st.unpack_from("<H", elf_bytes, 0x3C)[0]
    shstrndx = st.unpack_from("<H", elf_bytes, 0x3E)[0]
    shstr_off = st.unpack_from("<Q", elf_bytes, shoff + shstrndx * shentsize + 0x18)[0]
    for i in range(shnum):
        e = shoff + i * shentsize
        nm_off = st.unpack_from("<I", elf_bytes, e)[0]
        nm = elf_bytes[shstr_off + nm_off:elf_bytes.index(b"\x00", shstr_off + nm_off)].decode()
        if nm == name:
            return st.unpack_from("<Q", elf_bytes, e + 0x20)[0]
    return None


def main():
    elf_path, magic_off, outdir = sys.argv[1], int(sys.argv[2], 0), sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    data = open(elf_path, "rb").read()
    bundle_start = data.index(MAGIC, magic_off)
    size = section_size(data, ".hip_fatbin")
    bundle = data[bundle_start:bundle_start + size] if size else data[bundle_start:]
    entries = parse_v2(bundle)
    print(f"bundle@{hex(bundle_start)} v2_entries={len(entries)}", file=sys.stderr)

    manifest = {"source": elf_path, "bundle_offset": hex(bundle_start), "blobs": []}

    def emit(tag, name, enc, content, parent):
        enc_name = ENCODINGS.get(enc, V1_ENC.get(enc, f"ENC{enc}"))
        safe = name.replace("/", "_").replace("(", "_").replace(")", "_").replace(",", "_").replace(":", "_")
        info = classify(content)
        ext = {"ELF": ".hsaco", "LLVM-BITCODE": ".bc", "ZSTD": ".zst", "GZIP": ".gz"}.get(info["kind"], ".bin")
        fname = f"{tag}_{safe}_{enc_name}{ext}"
        path = os.path.join(outdir, fname)
        with open(path, "wb") as f:
            f.write(content)
        rec = {"tag": tag, "name": name, "encoding": enc, "encoding_name": enc_name,
               "parent": parent, "size": len(content), "file": path,
               "sha256": hashlib.sha256(content).hexdigest(), **info}
        if info["kind"] == "ELF":
            syms = symbols_of(path)
            rec["symbols"] = len(syms)
            rec["symbol_names"] = syms
        manifest["blobs"].append(rec)
        return path

    for i, (triple, enc, coff, csize, align) in enumerate(entries):
        content = bundle[coff:coff + csize]
        path = emit(f"v2_{i}", triple, enc, content, "bundle")
        if enc == 3:  # AMDGCN container = v1 fatbinary
            sub, base = parse_v1_fatbinary(content)
            print(f"  AMDGCN container: {len(sub)} v1 entries (table@{hex(base)})", file=sys.stderr)
            for j, (nm, senc, scoff, scsize) in enumerate(sub):
                scontent = content[scoff:scoff + scsize]
                if classify(scontent)["kind"] == "ZSTD":
                    scontent = subprocess.run(["zstd", "-d", "-c"], input=scontent,
                                              capture_output=True, timeout=600).stdout
                emit(f"v2_{i}_v1_{j}", nm, senc, scontent, os.path.basename(path))

    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
