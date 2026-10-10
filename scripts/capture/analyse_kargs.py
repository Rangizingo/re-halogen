#!/usr/bin/env python3
"""Turn kargs_shim JSON lines into tables.

usage: python3 -I analyse_kargs.py kargs.jsonl [--hex]
       python3 -I analyse_kargs.py kargs.jsonl --trace [--gap SECONDS]
Prints one row per captured launch, then a NormFold pointer summary and a Tap summary.
--trace prints the ordered kernel list per request (CAPTURE_TRACE=1 runs); requests are split where the
launch timestamps have a gap over SECONDS (default 1.0), and route/lq kernels are tagged.
Kernarg layout (k_dnc_scan3x): Tap at 0x80 {i32 slot[3], i32 pad, ptr p[3]}, NormFold at 0xa8
{ptr n0..n3, i32 w}.
"""
import json
import re
import struct
import sys

TAP_OFF, NORM_OFF = 0x80, 0xA8


def short_name(mangled):
    """k_dnc_scan3x<2,4,1,0,...> from an Itanium-mangled name; True/False shown as 1/0."""
    m = re.search(r'k_dnc_(scan\d\w?)I(.*?)E(?:Ev|v)', mangled)
    if not m:
        return mangled[:60]
    vals = re.findall(r'L[ib](-?\d+)E', m.group(2))
    return f'k_dnc_{m.group(1)}<{",".join(vals)}>'


def decode_kernarg(hexstr):
    """Tap and NormFold fields from the kernarg hex; None when the block is too short."""
    raw = bytes.fromhex(hexstr)
    out = {'size': len(raw)}
    if len(raw) >= NORM_OFF + 40:
        slots = struct.unpack_from('<3i', raw, TAP_OFF)
        ptrs = struct.unpack_from('<3Q', raw, TAP_OFF + 16)
        norm = struct.unpack_from('<4Q', raw, NORM_OFF)
        (w,) = struct.unpack_from('<i', raw, NORM_OFF + 32)
        out.update(slots=slots, tap_p=ptrs, norm=norm, w=w)
    return out


def load(lines):
    return [json.loads(x) for x in lines if x.strip()]


def launch_rows(events):
    rows = []
    for ev in events:
        if ev.get('kind') != 'launch':
            continue
        d = decode_kernarg(ev['kernarg_hex'])
        row = {'idx': ev['idx'], 'kernel': short_name(ev['kernel']), 'grid': 'x'.join(map(str, ev['grid'])),
               'size': d['size'], 'rc': ev.get('rc')}
        if 'slots' in d:
            row['slots'] = ','.join(str(s) for s in d['slots'])
            row['taps'] = sum(1 for p in d['tap_p'] if p)
            row['w'] = d['w']
            eq = []
            for p in ev.get('norm', {}).get('ptrs', []):
                eq.append(f"{p['name']}={'|'.join(p['equal']) or ('0' if p['ptr'] == '0x0' else '?')}")
            row['norm'] = ' '.join(eq)
        rows.append(row)
    return rows


def fmt_table(rows, cols):
    cells = [[str(r.get(c, '')) for c in cols] for r in rows]
    widths = [max(len(c), *(len(x[i]) for x in cells)) if cells else len(c) for i, c in enumerate(cols)]
    line = '  '.join(c.ljust(w) for c, w in zip(cols, widths))
    body = ['  '.join(v.ljust(w) for v, w in zip(x, widths)) for x in cells]
    return '\n'.join([line, '-' * len(line)] + body)


def norm_summary(events):
    """For each of n0..n3: how often it equals each global, and how many distinct values were seen."""
    acc = {}
    for ev in events:
        for p in ev.get('norm', {}).get('ptrs', []):
            a = acc.setdefault(p['name'], {'launches': 0, 'null': 0, 'values': set(), 'equal': {}, 'near': {}, 'hashes': set()})
            a['launches'] += 1
            a['null'] += p['ptr'] == '0x0'
            a['values'].add(p['ptr'])
            if 'hash' in p:
                a['hashes'].add(p['hash'])
            for g in p['equal']:
                a['equal'][g] = a['equal'].get(g, 0) + 1
            for nr in p.get('near', []):
                key = f"{nr['global']}{nr['delta']:+d}"
                a['near'][key] = a['near'].get(key, 0) + 1
    rows = []
    for name in sorted(acc):
        a = acc[name]
        eq = ' '.join(f'{g}:{c}' for g, c in sorted(a['equal'].items())) or 'none'
        near = ' '.join(f'{k}:{c}' for k, c in sorted(a['near'].items(), key=lambda kv: -kv[1])[:3]) or 'none'
        rows.append({'ptr': name, 'launches': a['launches'], 'null': a['null'], 'distinct_values': len(a['values']),
                     'distinct_hashes': len(a['hashes']), 'equals_global': eq, 'near_global_top3': near})
    return rows


def tap_rows(events):
    rows = []
    for ev in events:
        if ev.get('kind') != 'launch':
            continue
        for r in ev.get('tap', {}).get('regions', []):
            states = r.get('vs_state', [])
            same = [s['seq'] for s in states if s.get('words_differ_post') == 0]
            dst = [s['seq'] for s in states if s.get('dst_is_p')]
            rows.append({'idx': ev['idx'], 'k': r['k'], 'slot': r['slot'], 'bytes': r['bytes'],
                         'words_changed': r.get('words_changed', ''), 'pre': r.get('pre_hash', ''),
                         'post': r.get('post_hash', ''), 'p_is_state_dst_of': ','.join(map(str, dst)) or '-',
                         'post_equals_state_src': ','.join(map(str, same)) or '-'})
    return rows


def report(events, with_hex=False):
    out = []
    lrows = launch_rows(events)
    out.append(f'launches captured: {len(lrows)}; memcpy records: '
               f'{sum(1 for e in events if e.get("kind") == "memcpy")}; '
               f'unmatched funcs: {sum(1 for e in events if e.get("kind") == "unmatched")}')
    out.append(fmt_table(lrows, ['idx', 'kernel', 'grid', 'size', 'slots', 'taps', 'w', 'norm', 'rc']))
    out.append('')
    out.append('NormFold pointers vs host globals')
    out.append(fmt_table(norm_summary(events),
                         ['ptr', 'launches', 'null', 'distinct_values', 'distinct_hashes', 'equals_global',
                          'near_global_top3']))
    out.append('')
    t = tap_rows(events)
    out.append('Tap regions (non-null p[k])' if t else 'Tap regions: none (no launch had a non-null p[k])')
    if t:
        out.append(fmt_table(t, ['idx', 'k', 'slot', 'bytes', 'words_changed', 'pre', 'post', 'p_is_state_dst_of',
                                 'post_equals_state_src']))
    if with_hex:
        out.append('')
        for ev in events:
            if ev.get('kind') == 'launch':
                out.append(f'{ev["idx"]} {ev["kernarg_hex"]}')
    return '\n'.join(out)


CATEGORIES = [('route_topk_w', 'k_route_topk_w'), ('route_logits', 'k_route_logits_'), ('lq8', 'k_lq8'),
              ('lq4w', 'k_lq4w'), ('lq6w', 'k_lq6w')]


def kernel_label(mangled):
    """k_name<template ints> from an Itanium name in the halogen anonymous namespace."""
    if not mangled.startswith('_ZN'):
        return mangled[:60]
    pos, name = 3, None
    while True:  # walk the length-prefixed namespace path to the k_* component
        m = re.compile(r'(\d+)').match(mangled, pos)
        if not m:
            return mangled[:60]
        n = int(m.group(1))
        name, pos = mangled[m.end():m.end() + n], m.end() + n
        if name.startswith('k_'):
            break
    rest = mangled[pos:]
    t = re.match(r'I(.*?)E(?:Ev|v)', rest)
    vals = re.findall(r'L[ib](-?\d+)E', t.group(1)) if t else []
    return f'{name}<{",".join(vals)}>' if vals else name


def category(label):
    for cat, prefix in CATEGORIES:
        if label.startswith(prefix):
            return cat
    return None


def trace_items(events):
    """Ordered [(seq, ns, label)] of launches and hipblasLtMatmul calls, names joined from kname lines."""
    names = {e['h']: e['name'] for e in events if e.get('kind') == 'kname'}
    items = []
    for e in events:
        if e.get('kind') == 'trace':
            label = kernel_label(names[e['h']]) if e['h'] in names else 'unknown@' + e['h']
            items.append((e['seq'], e['ns'], label))
        elif e.get('kind') == 'matmul':
            items.append((e['seq'], e['ns'], 'hipblasLtMatmul'))
    return sorted(items)


def split_requests(items, gap_s=1.0):
    """Split the ordered trace where consecutive timestamps are more than gap_s apart."""
    out, cur, last = [], [], None
    for it in items:
        if last is not None and (it[1] - last) / 1e9 > gap_s:
            out.append(cur)
            cur = []
        cur.append(it)
        last = it[1]
    if cur:
        out.append(cur)
    return out


def runs(labels):
    """Collapse consecutive repeats: [(label, count)]."""
    out = []
    for lb in labels:
        if out and out[-1][0] == lb:
            out[-1][1] += 1
        else:
            out.append([lb, 1])
    return [(a, b) for a, b in out]


def trace_report(events, gap_s=1.0):
    out, items = [], trace_items(events)
    reqs = split_requests(items, gap_s)
    out.append(f'trace items: {len(items)} (launches + matmuls); requests (gap > {gap_s}s): {len(reqs)}')
    for i, req in enumerate(reqs):
        labels = [it[2] for it in req]
        counts = {}
        for lb in labels:
            c = category(lb) or ('matmul' if lb == 'hipblasLtMatmul' else None)
            if c:
                counts[c] = counts.get(c, 0) + 1
        span = (req[-1][1] - req[0][1]) / 1e9
        out.append('')
        out.append(f'request {i}: seq {req[0][0]}..{req[-1][0]}, {len(req)} items, {span:.3f}s, '
                   f'{len(set(labels))} distinct; marked: ' + (' '.join(f'{k}={v}' for k, v in sorted(counts.items())) or 'none'))
        for lb, n in runs(labels):
            cat = category(lb)
            tag = f'[{cat}] ' if cat else ('[matmul] ' if lb == 'hipblasLtMatmul' else '')
            out.append(f'  {tag}{lb}' + (f' x{n}' if n > 1 else ''))
    return '\n'.join(out)


def main(argv):
    with open(argv[1]) as f:
        events = load(f)
    if '--trace' in argv:
        gap = float(argv[argv.index('--gap') + 1]) if '--gap' in argv else 1.0
        print(trace_report(events, gap))
        return
    print(report(events, '--hex' in argv))


if __name__ == '__main__':
    main(sys.argv)
