"""Tests for rewrite/harness/const_audit.py on small inline ISA samples."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'rewrite' / 'harness'))
import const_audit as ca

VENDOR = """\
0000000000001000 <_ZN7halogen6k_testEv>:
\ts_load_b32 s30, s[0:1], 0x28                               // 000000001000: F4000780 F8000028
\ts_waitcnt lgkmcnt(0)                                       // 000000001008: BF89FC07
\tv_mov_b32_e32 v3, 0xff800000                               // 00000000100c: 7E0602FF FF800000
\tv_mul_f32_e32 v1, 0.5, v2                                   // 000000001014: 10020480
\tv_fmaak_f32 v4, v1, v2, 0x3e22f983                          // 000000001018: 5A08051 3E22F983
\tv_mul_f32_e32 v5, -4.0, v5                                  // 000000001020: 100A0AF7
\tglobal_load_b32 v1, v[2:3], off offset:0x40                // 000000001024: DC520040 01000002
\ts_cbranch_execz 43                                         // 000000001030: BFA5002B <_ZN7halogen6k_testEv+0x114>
\ts_endpgm                                                   // 000000001034: BFB00000

0000000000002000 <_ZN7halogen7k_otherEv>:
\tv_mov_b32_e32 v3, 0x12345678                               // 000000002000: 7E0602FF 12345678
"""


def lits(text):
    return ca.extract_literals(text.splitlines())


def test_split_kernels_by_header():
    ks = ca.split_kernels(VENDOR)
    assert list(ks) == ['_ZN7halogen6k_testEv', '_ZN7halogen7k_otherEv']
    assert 0x12345678 not in lits('\n'.join(ks['_ZN7halogen6k_testEv']))


def test_hex_literal_and_fmaak_k():
    got = lits(VENDOR)
    assert 0xff800000 in got
    assert 0x3e22f983 in got


def test_inline_float_constants_become_bits():
    got = lits(VENDOR)
    assert 0x3f000000 in got       # 0.5
    assert 0xc0800000 in got       # -4.0
    assert 0x3e22f983 in lits('v_mul_f32 v0, 0.15915494, v1')   # 1/(2*pi) inline


def test_inline_float_equals_hex_spelling():
    assert lits('v_mul_f32 v0, 0.5, v1') == lits('v_mul_f32 v0, 0x3f000000, v1')


def test_ignored_operands():
    got = lits(VENDOR)
    assert 0x28 not in got and 0x40 not in got        # kernarg offset, offset:
    assert 43 not in got and 0x114 not in got        # branch target, label offset
    assert 0 not in got
    assert lits('s_waitcnt vmcnt(0x3) lgkmcnt(0)') == set()
    assert lits('s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT)') == set()
    assert lits('ds_load_b32 v1, v2 offset:0x1234') == set()
    assert lits('s_load_b64 s[2:3], s[0:1], 0x10') == set()
    assert lits('v_mov_b32 v0, 5') == set()            # decimal integer inline


def test_comment_encoding_not_read():
    assert lits('v_add_f32 v0, v1, v2  // 000000001000: 0600ABCD 0xDEADBEEF') == set()


def test_64bit_literal_split_in_halves():
    assert lits('s_mov_b64 s[0:1], 0x3ff0000000000000') == {0x3ff00000, 0}


def test_dual_issue_both_halves():
    got = lits('v_dual_mov_b32 v3, 0xff800000 :: v_dual_mul_f32 v4, 0.5, v0')
    assert got == {0xff800000, 0x3f000000}


def test_f64_inline_is_high_word():
    assert lits('v_mul_f64 v[0:1], 1.0, v[2:3]') == {0x3ff00000}


def test_source_modifiers_stripped():
    assert lits('v_mul_f32 v0, -|0x3f000000|, v1') == {0x3f000000}
    assert lits('v_fma_f32 v0, neg(0x3f800000), v1, v2') == {0x3f800000}


def test_sign_flip_not_missing():
    v = {0x3f000000, 0xc0800000, 0x12345678}
    r = {0xbf000000, 0x40800000, 0x99999999}
    cmp = ca.compare_sets(v, r)
    assert cmp.missing == [0x12345678]
    assert cmp.flipped == [0x3f000000, 0xc0800000]
    assert cmp.extra == [0x99999999]


def test_verdict_ok_with_flip_and_gap():
    ok = ca.compare_sets({0x3f000000}, {0xbf000000})
    assert ca.verdict(ok).startswith('AUDIT OK')
    assert 'sign-flipped' in ca.verdict(ok)
    gap = ca.compare_sets({0x3f000000, 0x358637bd}, {0x3f000000})
    assert ca.verdict(gap) == 'AUDIT GAP 0x358637bd (1e-06)'


def test_exact_match_is_plain_ok():
    assert ca.verdict(ca.compare_sets({1 << 30}, {1 << 30})) == 'AUDIT OK'


def test_row_format_and_resume(tmp_path):
    out = tmp_path / 'ca.md'
    cmp = ca.compare_sets({0x3f000000}, {0x3f000000})
    ca.append_row(out, ca.format_row('inst_a', 'codeobj_53000', cmp))
    ca.append_row(out, ca.format_row('inst_b', 'codeobj_53000', cmp))
    text = out.read_text()
    assert text.count('| instance | object |') == 1
    assert ca.done_instances(out) == {'inst_a', 'inst_b'}
    assert '| inst_a | codeobj_53000 | 1 | 1 |  |  |  | AUDIT OK |' in text


def test_lq4w_flags_from_mangled_name():
    n = '_ZN7halogen7k_lq4wILi2ELi16ELi1ELb0ELi3ELi1EEvv'
    f = ca.extra_flags(n, 'lq4w_c2_b0_e3.json')
    assert f == '-DLQ_N=2 -DLQ_W=16 -DLQ_M=1 -DLQ_B=0 -DLQ_E=3 -DLQ_F=1'
    assert ca.extra_flags('_ZN7halogen5k_foo', 'foo.json') == ''
    p = ca.extra_flags('_ZN7halogen11k_lq4w_pairILi3ELi16ELi2EEvv', 'lq4w_pair_c3.json')
    assert p == '-DLQ_PAIR_N=3 -DLQ_PAIR_F=2'
    g = ca.extra_flags('_ZN7halogen10k_lq4w_grpILi1ELi16ELi2ELi1EEvv', 'lq4w_grp_c1_A.json')
    assert g == '-DLQ_GRP_N=1 -DLQ_GRP_W=16 -DLQ_GRP_MB=2 -DLQ_GRP_MS=16 -DLQ_GRP_F=1'


def test_classify_float_vs_int():
    for v in (0x3f000000, 0xc0800000, 0x358637bd, 0x3e22f983, 0x3ff00000, 0x1fffffe0):
        assert ca.is_float(v), hex(v)
    for v in (0x200, 0x400, 0xffffff00, 0x7fffffff, 0x80000000, 0x7f800000, 0x0fffffff, 0x00010001,
              0x00ffffff, 0x01000000 - 1, 0x3ff):
        assert not ca.is_float(v), hex(v)


def test_int_gaps_do_not_fail_the_audit():
    c = ca.compare_sets({0x3f000000, 0x400, 0x7fffffff}, {0x3f000000, 0x200})
    assert c.missing == []
    assert c.int_missing == [0x400, 0x7fffffff]
    assert ca.verdict(c) == 'AUDIT OK'


def test_float_gap_still_fails_with_int_gaps_present():
    c = ca.compare_sets({0x358637bd, 0x400}, {0x200})
    assert c.missing == [0x358637bd] and c.int_missing == [0x400]
    assert ca.verdict(c) == 'AUDIT GAP 0x358637bd (1e-06)'


def test_float_present_as_inline_fmaak_and_f64_half():
    vendor = lits('v_mul_f32 v0, 0x3f000000, v1\nv_fmaak_f32 v4, v1, v2, 0x3e22f983\ns_mov_b64 s[0:1], 0x3ff0000000000000')
    rew = lits('v_mul_f32 v0, 0.5, v1\nv_fmamk_f32 v4, v1, 0x3e22f983, v2\nv_mul_f64 v[0:1], 1.0, v[2:3]')
    assert ca.compare_sets(vendor, rew).missing == []


def test_row_has_int_column():
    c = ca.compare_sets({0x400}, set())
    row = ca.format_row('i', 'o', c)
    assert row == '| i | o | 1 | 0 |  |  | 0x400 | AUDIT OK |'


def test_spec_mapping_from_match_rows(tmp_path):
    t = tmp_path / 'g99.md'
    t.write_text('| instance | family | test spec | result | tolerance | date |\n| --- | --- | --- | --- | --- | --- |\n'
                 '| inst_a | f | bad.json seed=1 N=7 | MISMATCH n=1 | bitwise | d |\n'
                 '| inst_a | f | good.json seed=1 N=7 | MATCH n=1 diff=0 | bitwise | d |\n'
                 '| inst_a | f | good2.json seed=1 N=7 | MATCH n=1 diff=0 | bitwise | d |\n')
    m = ca.spec_map([t])
    assert m == {'inst_a': 'good.json'}
