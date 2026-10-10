#!/usr/bin/env python3
"""Extract per-kernel kernarg layouts from code-object notes into kernargs.json.

usage: python3 -I gen_kernargs.py STUBS.tsv OUT.json NOTES_OR_HSACO [...]
  STUBS.tsv  analysis/function-map-kernel-stubs.tsv (instances: col 2 mangled name, col 4 family)
  NOTES      `llvm-readelf --notes` text, or a .hsaco (the script runs llvm-readelf itself)

An instance is a TSV row; it is covered when its mangled name matches a `.name:` in the notes.
Each kernel record: object, family, kernarg_size, max_flat_workgroup_size, group_segment,
private_segment, wavefront_size, args=[[offset, size, value_kind], ...] (explicit and hidden).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HIDDEN_PREFIX = 'hidden_'


def parse_notes(text):
    """mangled name -> record, from the AMDGPU metadata YAML subset printed by llvm-readelf."""
    kernels = {}
    cur = None
    arg = None
    in_args = False

    def close_arg():
        nonlocal arg
        if arg is not None and arg.get('offset') is not None:
            cur['args'].append([arg['offset'], arg.get('size', 0), arg.get('kind', '?')])
        arg = None

    for line in text.splitlines():
        if re.match(r'^  - \.args:', line):
            cur = {'args': []}
            arg = None
            in_args = True
            continue
        if cur is None:
            continue
        if in_args:
            if re.match(r'^      - ', line):
                close_arg()
                arg = {}
                line = '        ' + line[8:]
            if arg is not None:
                m = re.match(r'^        \.(offset|size):\s+(\d+)', line)
                if m:
                    arg[m.group(1)] = int(m.group(2))
                    continue
                m = re.match(r'^        \.value_kind:\s+(\S+)', line)
                if m:
                    arg['kind'] = m.group(1)
                    continue
            if re.match(r'^    \.\w', line):
                close_arg()
                in_args = False
        m = re.match(r'^    \.(kernarg_segment_size|max_flat_workgroup_size|group_segment_fixed_size|'
                     r'private_segment_fixed_size|wavefront_size):\s+(\d+)', line)
        if m:
            cur[m.group(1)] = int(m.group(2))
            continue
        m = re.match(r'^    \.name:\s+(\S+)', line)
        if m:
            kernels[m.group(1)] = cur  # keys after .name (private_segment, wavefront_size) still land in cur
    return kernels


def parse_stubs(text):
    """[(mangled, family)] for each instance row; skips the header."""
    rows = []
    for line in text.splitlines()[1:]:
        f = line.split('\t')
        if len(f) >= 4 and f[1].startswith('_Z'):
            rows.append((f[1], f[3]))
    return rows


def build(rows, notes_by_obj):
    """kernel table plus the names of instances with no note. notes_by_obj: {object: {name: rec}}."""
    where = {}
    for obj, ks in notes_by_obj.items():
        for name, rec in ks.items():
            where.setdefault(name, (obj, rec))
    out, missing = {}, []
    for name, family in rows:
        if name not in where:
            missing.append(name)
            continue
        obj, rec = where[name]
        out[name] = {
            'object': obj, 'family': family,
            'kernarg_size': rec.get('kernarg_segment_size', 0),
            'max_flat_workgroup_size': rec.get('max_flat_workgroup_size', 0),
            'group_segment': rec.get('group_segment_fixed_size', 0),
            'private_segment': rec.get('private_segment_fixed_size', 0),
            'wavefront_size': rec.get('wavefront_size', 0),
            'args': rec['args'],
        }
    return out, missing


def hidden_offsets(args):
    """{'block_count_x': off, 'group_size_x': off, ...} for the harness `hidden` spec key."""
    return {k[len(HIDDEN_PREFIX):]: off for off, _, k in args if k.startswith(HIDDEN_PREFIX)}


def load_notes(path):
    p = Path(path)
    if p.suffix == '.hsaco':
        return subprocess.run(['llvm-readelf', '--notes', str(p)], check=True, capture_output=True, text=True).stdout
    return p.read_text()


def main(argv):
    rows = parse_stubs(Path(argv[1]).read_text())
    notes_by_obj = {}
    for p in argv[3:]:
        notes_by_obj[Path(p).stem.replace('notes_', '')] = parse_notes(load_notes(p))
    table, missing = build(rows, notes_by_obj)
    doc = {
        'instances': len(rows), 'covered': len(table), 'missing': missing,
        'notes_kernels': {o: len(k) for o, k in notes_by_obj.items()},
        'kernels': table,
    }
    Path(argv[2]).write_text(json.dumps(doc, separators=(',', ':')) + '\n')
    print(f'{len(table)} of {len(rows)} instances covered; {len(missing)} missing; '
          f'notes kernels per object: {doc["notes_kernels"]}', file=sys.stderr)


if __name__ == '__main__':
    main(sys.argv)
