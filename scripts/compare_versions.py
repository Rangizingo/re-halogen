#!/usr/bin/env python3
"""Compare two Ghidra exports of flash_serve (old dir vs new dir) function by function.

Usage: compare_versions.py OLD_DIR NEW_DIR FUNCTION_MAP_TSV OUT_MD OUT_TSV
Each dir holds <addr>.c plus index.tsv (scripts/ExportDecomp.java output).
Match tiers:
  1. same normalized-body fingerprint            -> unchanged
  2. same unique string set, different body      -> changed
  3. same (size, callees, callers) and no strings, both unique -> changed-shape
  anything left                                  -> added (new only) / removed (old only)
Normalization drops addresses, FUN_/DAT_/LAB_/PTR_ names, local and parameter
names, so a pure relocation does not count as a change.
"""
import collections
import hashlib
import re
import sys
from pathlib import Path

ADDR = re.compile(r'\b(switchD_[0-9a-fA-F]+(_caseD_[0-9a-fA-F]+)?|switchdataD_[0-9a-fA-F]+|(FUN|DAT|LAB|PTR|SUB|thunk_FUN|caseD|s|u)_[0-9a-fA-F]{4,})\b')
HEX = re.compile(r'\b0x[0-9a-fA-F]{5,}\b')
VARS = re.compile(r'\b(local_[0-9a-fA-Z_]+|param_\d+|[a-z]{1,3}Var\d+|puVar\d+|pp?[a-zA-Z]Var\d+|in_[A-Za-z0-9_]+|unaff_[A-Za-z0-9_]+|extraout_[A-Za-z0-9_]+)\b')


def norm(text: str) -> str:
    t = ADDR.sub('A', text)
    t = HEX.sub('H', t)
    t = VARS.sub('V', t)
    return re.sub(r'\s+', ' ', t).strip()


def load(d: Path):
    idx = {}
    for line in d.joinpath('index.tsv').read_text().splitlines()[1:]:
        p = line.split('\t')
        if len(p) < 6:
            p += [''] * (6 - len(p))
        idx[p[0]] = {'addr': p[0], 'name': p[1], 'size': int(p[2] or 0),
                     'callees': p[3], 'callers': p[4], 'strings': p[5]}
    for a, r in idx.items():
        f = d / f'{a}.c'
        r['fp'] = hashlib.sha256(norm(f.read_text()).encode()).hexdigest() if f.exists() else None
    return idx


def main():
    old_d, new_d, fmap, out_md, out_tsv = sys.argv[1:6]
    old, new = load(Path(old_d)), load(Path(new_d))
    role = {}
    for line in Path(fmap).read_text().splitlines()[1:]:
        p = line.split('\t')
        role[p[0]] = (p[1], p[2])

    res = {}  # new addr -> (status, old addr)
    used = set()
    byfp = collections.defaultdict(list)
    for a, r in old.items():
        if r['fp']:
            byfp[r['fp']].append(a)
    for a, r in new.items():
        c = [o for o in byfp.get(r['fp'], []) if o not in used]
        if c:
            # prefer same-name or nearest-size candidate
            c.sort(key=lambda o: (old[o]['name'] != r['name'], abs(old[o]['size'] - r['size'])))
            res[a] = ('unchanged', c[0])
            used.add(c[0])

    def key_strings(r):
        return r['strings'] if r['strings'] else None

    old_by_str = collections.defaultdict(list)
    for a, r in old.items():
        if a not in used and key_strings(r):
            old_by_str[key_strings(r)].append(a)
    new_by_str = collections.Counter(key_strings(r) for a, r in new.items() if a not in res and key_strings(r))
    for a, r in new.items():
        if a in res or not key_strings(r):
            continue
        k = key_strings(r)
        c = old_by_str.get(k, [])
        if len(c) == 1 and new_by_str[k] == 1:
            res[a] = ('changed', c[0])
            used.add(c[0])

    old_by_shape = collections.defaultdict(list)
    for a, r in old.items():
        if a not in used and not r['strings']:
            old_by_shape[(r['size'], r['callees'], r['callers'])].append(a)
    new_shape = collections.Counter((r['size'], r['callees'], r['callers']) for a, r in new.items() if a not in res and not r['strings'])
    for a, r in new.items():
        if a in res or r['strings']:
            continue
        k = (r['size'], r['callees'], r['callers'])
        c = old_by_shape.get(k, [])
        if len(c) == 1 and new_shape[k] == 1:
            res[a] = ('changed-shape', c[0])
            used.add(c[0])

    status = collections.Counter()
    rows = []
    for a, r in new.items():
        st, o = res.get(a, ('added', ''))
        status[st] += 1
        rows.append((a, st, o, role.get(a, ('?', ''))[0], role.get(a, ('?', ''))[1], r['size'], r['name']))
    removed = [(a, old[a]['size'], old[a]['name'], old[a]['strings'][:80]) for a in old if a not in used]
    Path(out_tsv).write_text('addr_new\tstatus\taddr_old\trole\tlabel\tsize\tname\n' + '\n'.join('\t'.join(map(str, r)) for r in rows) + '\n')

    by_role = collections.defaultdict(collections.Counter)
    for a, st, o, rl, lb, sz, nm in rows:
        by_role[rl][st] += 1
    md = ['# Version comparison: flash_serve 0.17.2 vs 0.17.3 (function level)', '',
          f'Old functions {len(old)}, new functions {len(new)}. Method: `scripts/compare_versions.py`, normalized decompiled-body fingerprints (addresses, auto names and local names removed), then unique string-set and shape matches. Matching is heuristic: a function whose body changed AND whose strings changed shows as added plus removed.', '',
          '## Totals (new binary)', '', '| status | functions |', '|---|---|'] + [f'| {k} | {v} |' for k, v in status.most_common()] + [f'| removed (old only) | {len(removed)} |', '',
          '## By role (new binary)', '', '| role | unchanged | changed | changed-shape | added |', '|---|---|---|---|---|']
    for rl, c in sorted(by_role.items(), key=lambda x: -sum(x[1].values())):
        md.append(f"| {rl} | {c['unchanged']} | {c['changed']} | {c['changed-shape']} | {c['added']} |")
    md += ['', '## Changed or added, engine logic only (non-stub roles)', '']
    core = {'model-forward', 'sampling-speculation', 'kv-cache-prefix', 'protocol-net', 'hgn-loader', 'scheduler-slots', 'gguf-repack', 'tokenizer-vocab', 'logging-diag', 'unknown', 'config-env'}
    for a, st, o, rl, lb, sz, nm in rows:
        if st != 'unchanged' and rl in core:
            md.append(f'- `{a}` {st}{" (old " + o + ")" if o else ""} [{rl}] {lb} ({sz} B)')
    md += ['', '## Removed (old only), first 60', '']
    for a, sz, nm, s in removed[:60]:
        md.append(f'- `{a}` {nm} ({sz} B) {s}')
    Path(out_md).write_text('\n'.join(md) + '\n')
    print(dict(status), 'removed', len(removed))


if __name__ == '__main__':
    main()
