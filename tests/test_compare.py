"""Tests for rewrite/harness/compare.py: expression resolution, tolerance logic, spec building, table rows."""
import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'rewrite' / 'harness'))
import compare as c


def f32s(*v):
    return struct.pack(f'<{len(v)}f', *v)


def bf16s(*bits):
    return struct.pack(f'<{len(bits)}H', *bits)


def test_eval_expr_arith_and_names():
    assert c.eval_expr('N*2+1', {'N': 5}) == 11
    assert c.eval_expr('ceil_div(N,256)', {'N': 257}) == 2
    assert c.eval_expr('0.5+SEED*0.25', {'SEED': 2}) == 1.0
    assert c.eval_expr('-N', {'N': 3}) == -3


@pytest.mark.parametrize('bad', ['__import__("os")', 'N.real', 'open("x")', 'M+1'])
def test_eval_expr_rejects_everything_else(bad):
    with pytest.raises((ValueError, SyntaxError)):
        c.eval_expr(bad, {'N': 1})


def test_resolve_walks_nested_and_keeps_plain_strings():
    t = {'a': ['=N+1', 'zero'], 'b': {'c': '=N*2', 'd': 7}}
    assert c.resolve(t, {'N': 3}) == {'a': [4, 'zero'], 'b': {'c': 6, 'd': 7}}


def test_resolve_float_expression_stays_float():
    assert c.resolve('=0.5+SEED*0.5', {'SEED': 1}) == 1.0


def test_bitwise_match_and_mismatch():
    a = f32s(1.0, 2.0, 3.0)
    assert c.compare_bytes(a, a, 'f32')['match']
    r = c.compare_bytes(a, f32s(1.0, 2.0, 3.0000002), 'f32')
    assert not r['match'] and r['n_diff'] == 1 and r['first_diff'] == 2 and r['max_ulp'] == 1


def test_ulp_tolerance_f32():
    one = struct.unpack('<I', f32s(1.0))[0]
    near = struct.pack('<I', one + 3)
    assert c.compare_bytes(f32s(1.0), near, 'f32', max_ulp=3)['match']
    assert not c.compare_bytes(f32s(1.0), near, 'f32', max_ulp=2)['match']


def test_abs_tolerance():
    assert c.compare_bytes(f32s(1.0), f32s(1.25), 'f32', max_abs=0.25)['match']
    assert not c.compare_bytes(f32s(1.0), f32s(1.25), 'f32', max_abs=0.2)['match']


def test_both_tolerances_must_hold():
    r = c.compare_bytes(f32s(1.0), f32s(1.25), 'f32', max_abs=0.5, max_ulp=1)
    assert not r['match']


def test_bf16_decode_and_ulp():
    r = c.compare_bytes(bf16s(0x3f80), bf16s(0x3f81), 'bf16')
    assert not r['match'] and r['max_ulp'] == 1 and r['max_abs'] == pytest.approx(2 ** -7)


def test_ulp_across_zero_and_sign():
    pz, nz = struct.pack('<f', 0.0), struct.pack('<f', -0.0)
    r = c.compare_bytes(pz, nz, 'f32', max_ulp=0)
    assert r['match'] and r['max_ulp'] == 0
    assert c.compare_bytes(f32s(1e-45), f32s(-1e-45), 'f32')['max_ulp'] == 2


def test_nan_handling():
    nan1 = struct.pack('<I', 0x7fc00000)
    nan2 = struct.pack('<I', 0x7fc00001)
    assert not c.compare_bytes(nan1, nan2, 'f32')['match']          # bitwise sees the payload
    assert c.compare_bytes(nan1, nan1, 'f32')['match']
    assert c.compare_bytes(nan1, nan2, 'f32', max_ulp=0)['match']   # tolerance mode: NaN equals NaN
    r = c.compare_bytes(nan1, f32s(1.0), 'f32', max_abs=1e9, max_ulp=10 ** 9)
    assert not r['match']


def test_int_compare_uses_abs_diff():
    a, b = struct.pack('<2i', 5, 9), struct.pack('<2i', 5, 7)
    r = c.compare_bytes(a, b, 'i32')
    assert not r['match'] and r['max_abs'] == 2 and r['max_ulp'] == 2
    assert c.compare_bytes(a, b, 'i32', max_abs=2)['match']


def test_length_mismatch():
    r = c.compare_bytes(f32s(1.0), f32s(1.0, 2.0), 'f32')
    assert not r['match'] and 'length' in r['note']


def test_explicit_args_drops_hidden():
    args = [[0, 8, 'global_buffer'], [8, 4, 'by_value'], [16, 4, 'hidden_block_count_x']]
    assert c.explicit_args(args) == [(0, 8, 'global_buffer'), (8, 4, 'by_value')]


def test_build_run_spec_injects_layout_and_drops_compare():
    t = {'grid': ['=ceil_div(N,64)', 1, 1], 'compare': {'x': 1}, 'args': []}
    lay = {'kernarg_size': 288, 'args': [[0, 8, 'global_buffer'], [44, 2, 'hidden_group_size_x']]}
    s = c.build_run_spec(t, 'kname', lay, {'N': 100})
    assert s['grid'] == [2, 1, 1] and s['kernel'] == 'kname' and s['kernarg_size'] == 288
    assert s['hidden'] == {'group_size_x': 44} and 'compare' not in s


def test_tol_text():
    assert c.tol_text(None, None) == 'bitwise'
    assert c.tol_text(0.001, 2) == 'abs<=0.001 ulp<=2'


def test_format_row_escapes_pipes():
    row = c.format_row('k', 'fam', 'a|b', 'MATCH', 'bitwise', '2026-10-10')
    assert row == '| k | fam | a/b | MATCH | bitwise | 2026-10-10 |\n'


def test_append_row_creates_header_then_appends(tmp_path):
    p = tmp_path / 'match-table.md'
    c.append_row(p, '| r1 |\n')
    c.append_row(p, '| r2 |\n')
    lines = p.read_text().splitlines()
    assert lines[0].startswith('# Kernel match table')
    assert '| instance | family | test spec | result | tolerance | date |' in lines
    assert lines[-2:] == ['| r1 |', '| r2 |']
    assert sum(1 for x in lines if x.startswith('| instance')) == 1


def test_result_text_words():
    ok = {'match': True, 'n': 4, 'n_diff': 0, 'max_abs': 0.0, 'max_ulp': 0, 'first_diff': None}
    assert c.result_text(ok).startswith('MATCH n=4 diff=0')
    bad = dict(ok, match=False, n_diff=1, first_diff=3)
    assert c.result_text(bad).startswith('MISMATCH') and 'first_diff=3' in c.result_text(bad)


def test_empty_buffers_never_match():
    r = c.compare_bytes(b'', b'', 'bf16')
    assert not r['match'] and r['note'] == 'empty'


def test_untouched_output_detected():
    pre = b'\x9d\xc2' * 8
    assert c.untouched(pre, pre)
    assert not c.untouched(b'\x00\x00' + pre[2:], pre)


def _spec_and_kernargs(tmp_path):
    spec = tmp_path / 's.json'
    spec.write_text(json.dumps({'grid': [1, 1, 1], 'block': [1, 1, 1], 'buffers': {}, 'args': [], 'dump': [],
                                'compare': {'rewrite_src': 'x.hip', 'rewrite_kernel': 'k', 'outputs': []}}))
    ka = tmp_path / 'ka.json'
    ka.write_text(json.dumps({'kernels': {'K': {'object': 'o', 'family': 'f', 'args': []}}}))
    return spec, ka


def test_auto_workdir_is_on_cmp_tmp_and_removed_even_on_error(tmp_path, monkeypatch):
    base = tmp_path / 'cmpbase'
    monkeypatch.setenv('CMP_TMP', str(base))
    seen = []

    def boom(src, out):
        seen.append(Path(out).parent)
        raise RuntimeError('compile failed')
    monkeypatch.setattr(c, 'compile_rewrite', boom)
    spec, ka = _spec_and_kernargs(tmp_path)
    with pytest.raises(RuntimeError):
        c.main([str(spec), '--kernel', 'K', '--kernargs', str(ka), '--no-table'])
    assert seen and seen[0].parent == base   # created under CMP_TMP, not /tmp
    assert not seen[0].exists()              # and removed


def test_given_workdir_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'compile_rewrite', lambda src, out: (_ for _ in ()).throw(RuntimeError('x')))
    spec, ka = _spec_and_kernargs(tmp_path)
    wd = tmp_path / 'keep'
    with pytest.raises(RuntimeError):
        c.main([str(spec), '--kernel', 'K', '--kernargs', str(ka), '--no-table', '--workdir', str(wd)])
    assert wd.is_dir()


def test_mem_available_parses_meminfo(tmp_path):
    p = tmp_path / 'meminfo'
    p.write_text('MemTotal:       126877820 kB\nMemAvailable:    15474356 kB\n')
    assert abs(c.mem_available_gib(str(p)) - 15474356 / 1048576) < 1e-9
    assert c.mem_available_gib(str(tmp_path / 'missing')) is None


def test_wait_for_host_waits_until_memory_is_free(tmp_path, monkeypatch):
    seen = iter([10.0, 12.0, 30.0])
    monkeypatch.setattr(c, 'mem_available_gib', lambda path='/proc/meminfo': next(seen))
    monkeypatch.setattr(c, 'service_healthy', lambda url=None: True)
    slept = []
    monkeypatch.setattr(c.time, 'sleep', slept.append)
    with c.host_slot(tmp_path / 'lock', min_free_gib=20.0):
        pass
    assert len(slept) == 2


def test_wait_for_host_waits_while_service_unhealthy(tmp_path, monkeypatch):
    health = iter([False, False, True])
    monkeypatch.setattr(c, 'mem_available_gib', lambda path='/proc/meminfo': 50.0)
    monkeypatch.setattr(c, 'service_healthy', lambda url=None: next(health))
    slept = []
    monkeypatch.setattr(c.time, 'sleep', slept.append)
    with c.host_slot(tmp_path / 'lock', min_free_gib=8.0):
        pass
    assert len(slept) == 2


def test_service_healthy_false_when_unreachable():
    assert c.service_healthy('http://127.0.0.1:9/health') is False
    assert c.service_healthy('') is True


def test_service_healthy_rejects_non_http_urls():
    with pytest.raises(ValueError):
        c.service_healthy('file:///etc/passwd')


def test_service_down_marker_skips_the_host_gate(tmp_path, monkeypatch):
    here = tmp_path / 'repo' / 'rewrite' / 'harness'
    (tmp_path / 'repo' / 'work' / 'cmp').mkdir(parents=True)
    monkeypatch.setattr(c, 'HERE', here)
    monkeypatch.setattr(c, '_load_gen_kernargs', lambda: None)
    monkeypatch.setenv('CMP_TMP', str(tmp_path / 'base'))

    def no_gate(*a, **k):
        raise AssertionError('host_slot used')
    monkeypatch.setattr(c, 'host_slot', no_gate)
    monkeypatch.setattr(c, 'compile_rewrite', lambda src, out: (_ for _ in ()).throw(RuntimeError('compiled')))
    spec, ka = _spec_and_kernargs(tmp_path)
    args = [str(spec), '--kernel', 'K', '--kernargs', str(ka), '--no-table']
    with pytest.raises(AssertionError):
        c.main(args)
    (tmp_path / 'repo' / 'work' / 'cmp' / '.service-down').touch()
    with pytest.raises(RuntimeError, match='compiled'):
        c.main(args)
