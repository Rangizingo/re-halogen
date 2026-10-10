#!/usr/bin/env python3
"""Generate analysis/fatbin-inventory.md from work/fatbin-manifest.json."""
import json, collections, re, os

m = json.load(open('/home/rangizingo/re-halogen/work/fatbin-manifest.json'))
rows = []
for b in m['blobs']:
    names = b.get('symbol_names') or []
    kern = [n for n in names if re.search(r'(^|::|_)k_|^k_', n.split('_Z')[-1]) or n.startswith('_Z')]
    # demangle-lite: extract the function base name
    fam = collections.Counter()
    for n in names:
        mm = re.search(r'1(\d+)k_([a-z0-9_]+?)E', n)
        if mm:
            fam['k_' + mm.group(2).rstrip('_')] += 1
    top = ', '.join(f'{k}×{v}' for k, v in fam.most_common(12))
    rows.append((b, len(names), fam, top))

out = []
out.append('# fatbin inventory — clang offload bundle v2 in flash_serve\n')
out.append(f'Source: `{m["source"]}`, bundle at file offset {m["section_offset"]}, '
           f'{m["section_size"]:,} B. Carved by `scripts/carve_fatbin.py`; '
           f'machine-readable manifest: `work/fatbin-manifest.json`.\n')
out.append('| code object | file offset | size | ELF e_machine | kernel symbols |')
out.append('|---|---|---|---|---|')
for b, nsym, fam, top in rows:
    out.append(f'| `{os.path.basename(b["file"])}` | {b["offset"]} | {b["size"]:,} B | '
               f'{b["e_machine"]} | {nsym} |')
out.append('')
out.append('sha256: ' + '; '.join(f'`{os.path.basename(b["file"])}` {b["sha256"][:16]}…' for b, *_ in rows))
out.append('')
out.append('## Kernel symbol families per code object (top 12 by count)\n')
for b, nsym, fam, top in rows:
    out.append(f'### {os.path.basename(b["file"])} ({nsym} symbols)\n')
    for k, v in sorted(fam.items(), key=lambda kv: (-kv[1], kv[0])):
        out.append(f'- `{k}` ×{v}')
    out.append('')
open('/home/rangizingo/re-halogen/analysis/fatbin-inventory.md', 'w').write('\n'.join(out) + '\n')
print('wrote', sum(len(r[2]) for r in rows), 'family names')
