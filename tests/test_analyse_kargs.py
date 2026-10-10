"""Tests for scripts/capture/analyse_kargs.py and gen_kargs_table.py on small synthetic input."""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'capture'))
import analyse_kargs as ak
import gen_kargs_table as gk

NAME = ('_ZN7halogen12_GLOBAL__N_112k_dnc_scan3xILi2ELi4ELb1ELb0ELb0ELi1ELi4ELi0ELb0EEEvPKtPKfS5_S5_S5_S5_S5_PfPKil'
        'PtiS3_S3_S3_S5_NS_10DncTapArgsENS_11DncNormFoldE')


def kernarg(slots, tap_p, norm, w):
    raw = bytearray(208)
    struct.pack_into('<3i', raw, 0x80, *slots)
    struct.pack_into('<3Q', raw, 0x90, *tap_p)
    struct.pack_into('<4Q', raw, 0xA8, *norm)
    struct.pack_into('<i', raw, 0xC8, w)
    return raw.hex()


def ptr(name, v, eq, h):
    return {'name': name, 'ptr': hex(v), 'equal': eq, 'near': [], 'hash': h}


def sample():
    l0 = {'kind': 'launch', 'idx': 0, 'kernel': NAME, 'grid': [48, 2, 1], 'rc': 0,
          'kernarg_hex': kernarg((-1, -1, -1), (0, 0, 0), (0x1000, 0x2000, 0x3000, 0x4000), 128),
          'norm': {'w_i32': 128, 'ptrs': [ptr('n0', 0x1000, [], 'aa'), ptr('n1', 0x2000, ['n1_18fc638'], 'bb'),
                                          ptr('n2', 0x3000, ['n2_18fc660'], 'cc'), ptr('n3', 0x4000, ['n3_18fc7b8'], 'dd')]},
          'tap': {'slots': [-1, -1, -1], 'p': ['0x0'] * 3, 'regions': []}}
    l1 = dict(l0, idx=1, kernarg_hex=kernarg((3, -1, -1), (0x9000, 0, 0), (0x1000, 0x2000, 0x3000, 0x4000), 128),
              tap={'slots': [3, -1, -1], 'p': ['0x9000', '0x0', '0x0'],
                   'regions': [{'k': 0, 'slot': 3, 'p': '0x9000', 'bytes': 3145728, 'pre_hash': 'p1', 'post_hash': 'p2',
                                'words_changed': 1024,
                                'vs_state': [{'seq': 5, 'dst_is_p': 1, 'words_differ_post': 0},
                                             {'seq': 6, 'dst_is_p': 0, 'words_differ_post': 77}]}]})
    return [l0, l1, {'kind': 'memcpy', 'seq': 5}, {'kind': 'unmatched'}, {'kind': 'summary'}]


def test_short_name_extracts_template_values():
    assert ak.short_name(NAME) == 'k_dnc_scan3x<2,4,1,0,0,1,4,0,0>'


def test_decode_kernarg_fields():
    d = ak.decode_kernarg(kernarg((3, -1, -1), (0x9000, 0, 0), (1, 2, 3, 4), 128))
    assert d['slots'] == (3, -1, -1) and d['tap_p'] == (0x9000, 0, 0) and d['norm'] == (1, 2, 3, 4) and d['w'] == 128
    assert ak.decode_kernarg('00' * 92) == {'size': 92}


def test_launch_rows_and_norm_summary():
    ev = sample()
    rows = ak.launch_rows(ev)
    assert [r['taps'] for r in rows] == [0, 1]
    assert rows[0]['norm'] == 'n0=? n1=n1_18fc638 n2=n2_18fc660 n3=n3_18fc7b8'
    s = {r['ptr']: r for r in ak.norm_summary(ev)}
    assert s['n1']['equals_global'] == 'n1_18fc638:2' and s['n0']['equals_global'] == 'none'
    assert s['n3']['launches'] == 2 and s['n3']['distinct_values'] == 1


def test_tap_rows_link_state_copy():
    (r,) = ak.tap_rows(sample())
    assert r['p_is_state_dst_of'] == '5' and r['post_equals_state_src'] == '5' and r['words_changed'] == 1024


def test_report_roundtrips_through_json_and_has_sections():
    lines = [json.dumps(e) for e in sample()]
    text = ak.report(ak.load(lines))
    assert 'launches captured: 2; memcpy records: 1; unmatched funcs: 1' in text
    assert 'NormFold pointers vs host globals' in text and 'Tap regions (non-null p[k])' in text


def test_gen_table_joins_rela_tsv_notes():
    rela = '00000000018f9b00 R_X86_64_RELATIVE  *ABS*+0x00000000018cf4e0\n'
    tsv = f'019cf4e0\t{NAME}\tdemangled\n'
    notes = (f'  - .args:\n      - .offset:         128\n        .size:           40\n        .value_kind:     by_value\n'
             f'      - .offset:         168\n        .size:           40\n        .value_kind:     by_value\n'
             f'    .kernarg_segment_size: 208\n    .name:           {NAME}\n')
    rows = gk.build_rows(gk.parse_rela(rela), gk.parse_stubs(tsv), gk.parse_notes(notes))
    assert [(r[0], r[1], r[3], r[5]) for r in rows] == [(0x18F9B00, 0x18CF4E0, 208, 1)]
    assert '{128,168}, {40,40}' in gk.emit(rows)
    assert gk.emit_names(gk.build_names(gk.parse_rela(rela), gk.parse_stubs(tsv))).count('0x18f9b00UL') == 1


def trace_events():
    def nm(base, t):
        return f'_ZN7halogen12_GLOBAL__N_1{len(base)}{base}I{t}EEvPKt'
    names = {'0x10': nm('k_route_topk_w', 'Li8E'), '0x20': nm('k_lq4w_grp', 'Li2ELi16E'),
             '0x30': '_ZN7halogen12_GLOBAL__N_19k_rmsnormEv'}
    ev = [{'kind': 'kname', 'h': h, 'name': n} for h, n in names.items()]
    # request 0: rms, lq4w x2, matmul, topk; 3.5 s idle; request 1: rms
    seq = [('0x30', 0.0), ('0x20', 0.001), ('0x20', 0.002), ('mm', 0.003), ('0x10', 0.004), ('0x30', 3.5)]
    for i, (h, t) in enumerate(seq):
        ns = int(t * 1e9)
        ev.append({'kind': 'matmul', 'seq': i, 'ns': ns} if h == 'mm' else
                  {'kind': 'trace', 'seq': i, 'ns': ns, 'h': h, 'grid': [1, 1, 1], 'block': [1, 1, 1], 'stream': '0x0'})
    return ev


def test_kernel_label_and_category():
    lb = ak.kernel_label('_ZN7halogen12_GLOBAL__N_110k_lq4w_grpILi2ELi16EEEvPKt')
    assert lb == 'k_lq4w_grp<2,16>' and ak.category(lb) == 'lq4w'
    assert ak.kernel_label('_ZN7halogen12_GLOBAL__N_19k_rmsnormEv') == 'k_rmsnorm' and ak.category('k_rmsnorm') is None
    assert ak.category('k_route_logits_x') == 'route_logits' and ak.category('k_lq8_gemv') == 'lq8'


def test_trace_splits_requests_and_marks_categories():
    items = ak.trace_items(trace_events())
    assert [i[2] for i in items][:4] == ['k_rmsnorm', 'k_lq4w_grp<2,16>', 'k_lq4w_grp<2,16>', 'hipblasLtMatmul']
    assert [len(r) for r in ak.split_requests(items, 1.0)] == [5, 1]
    text = ak.trace_report(trace_events(), 1.0)
    assert 'requests (gap > 1.0s): 2' in text
    assert '[lq4w] k_lq4w_grp<2,16> x2' in text and '[matmul] hipblasLtMatmul' in text and '[route_topk_w]' in text
    assert 'lq4w=2 matmul=1 route_topk_w=1' in text


def test_runs_collapse_only_consecutive():
    assert ak.runs(['a', 'a', 'b', 'a']) == [('a', 2), ('b', 1), ('a', 1)]
