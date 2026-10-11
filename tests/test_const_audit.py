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


def test_perm_selector_is_int_role():
    lines = ['v_perm_b32 v1, v2, v3, 0x07060302', 'v_mul_f32 v0, 0x3e22f983, v1']
    assert ca.extract_int_role(lines) == {0x07060302}
    c = ca.compare_sets(lits('\n'.join(lines)), set(), int_role=ca.extract_int_role(lines))
    assert c.missing == [0x3e22f983]
    assert 0x07060302 in c.int_missing


def test_mulhi_magic_is_int_role():
    lines = ['v_mul_hi_i32 v2, 0x66666667, v0', 's_mul_hi_i32 s20, s20, 0x66666667', 's_mul_hi_u32 s1, s2, 0x4ec4ec4f']
    assert ca.extract_int_role(lines) == {0x66666667, 0x4ec4ec4f}


def test_value_also_used_as_float_stays_float():
    lines = ['v_perm_b32 v1, v2, v3, 0x3f000000', 'v_mul_f32 v0, 0x3f000000, v1']
    assert ca.extract_int_role(lines) == set()


def test_udiv_expansion_constants_with_rcp_iflag():
    lines = ['v_rcp_iflag_f32_e32 v1, v1', 'v_mul_f32_e32 v1, 0x4f7ffffe, v1', 'v_cvt_u32_f32_e32 v1, v1']
    assert ca.extract_int_role(lines) == {0x4f7ffffe}
    assert ca.extract_int_role(['v_mul_f32_e32 v1, 0x4f7ffffe, v1']) == set()


def test_permlane_selector_moved_through_register():
    lines = ['s_mov_b32 s52, 0x76543210', 's_mov_b32 s57, 0x3db504f3',
             'v_permlanex16_b32 v11, v11, s52, 0xfedcba98', 'v_mul_f32 v1, s57, v1']
    assert ca.extract_int_role(lines) == {0x76543210, 0xfedcba98}


def test_integer_typed_instructions_are_int_role():
    lines = ['v_and_b32 v1, 0x0ffffffe, v2', 'v_or_b32 v1, 0x64006400, v1', 's_and_b32 s1, s2, 0xff000000',
             'v_and_or_b32 v1, v2, 0x1fff0000, v3']
    assert ca.extract_int_role(lines) == {0x0ffffffe, 0x64006400, 0xff000000, 0x1fff0000}


def test_float_typed_and_movers_stay_float():
    lines = ['v_mul_f32 v1, 0x3e800000, v2', 'v_dual_mov_b32 v3, 0x3da30000 :: v_dual_mul_f32 v4, 0.5, v0',
             'v_cndmask_b32 v1, 0x3f000000, v2, vcc_lo', 'v_cvt_u32_f32 v1, 0x4f000000']
    assert ca.extract_int_role(lines) == set()
    assert ca.extract_int_role(['v_cvt_f32_u32 v1, 0x3f000001']) == {0x3f000001}


def test_mov_literal_read_by_float_instruction_stays_float():
    lines = ['s_mov_b32 s4, 0x3e800000', 'v_mul_f32 v1, s4, v2', 'v_and_b32 v3, s4, v3']
    assert ca.extract_int_role(lines) == set()


def test_dual_mov_selector_and_redefinition():
    lines = ['v_dual_mov_b32 v146, 0x1000504 :: v_dual_mov_b32 v75, v1', 'v_perm_b32 v3, v4, v5, v146',
             's_mov_b32 s3, 0x01010101', 'v_and_b32 v1, s3, v2', 's_mov_b32 s3, 0x3e800000', 'v_mul_f32 v1, s3, v1']
    assert ca.extract_int_role(lines) == {0x01000504, 0x01010101}


def test_selector_chosen_by_cndmask_then_used_by_perm():
    lines = ['v_cndmask_b32 v54, 0x5040100, v146, vcc_lo', 'v_perm_b32 v1, v2, v3, v54']
    assert ca.extract_int_role(lines) == {0x05040100}


def test_selector_through_two_cndmasks():
    lines = ['v_mov_b32 v142, 0x1000504', 'v_cndmask_b32 v11, 0x7060302, v142, vcc_lo', 'v_perm_b32 v1, v2, v3, v11']
    assert ca.extract_int_role(lines) == {0x01000504, 0x07060302}


def test_unread_mov_does_not_make_literal_float():
    lines = ['v_mov_b32 v72, 0x1000504', 'v_mov_b32 v72, 0x1000504', 'v_perm_b32 v1, v2, v3, v72']
    assert ca.extract_int_role(lines) == {0x01000504}


def test_logf_denormal_prescale_and_udiv64_markers():
    log = ['v_cmp_gt_f32_e32 vcc_lo, 0x800000, v0', 'v_cndmask_b32_e64 v1, 1.0, 0x4f800000, vcc_lo', 'v_log_f32_e32 v0, v0']
    assert ca.extract_int_role(log) == {0x4f800000}
    d64 = ['s_fmamk_f32 s2, s12, 0x4f800000, s2', 'v_rcp_f32_e32 v2, s2', 's_mul_f32 s2, s2, 0x5f7ffffc',
           's_mul_f32 s12, s2, 0x2f800000', 's_fmamk_f32 s2, s12, 0xcf800000, s2']
    assert ca.extract_int_role(d64) == {0x4f800000, 0x5f7ffffc, 0x2f800000, 0xcf800000}


def test_selector_shape_tolerates_register_reuse():
    lines = ['v_mov_b32 v1, 0x3020706', 'v_perm_b32 v5, v2, v3, v1', 'v_mul_f32 v6, v1, v1']
    assert ca.extract_int_role(lines) == {0x03020706}
    lines = ['v_mov_b32 v1, 0x3e800000', 'v_perm_b32 v5, v2, v3, v1', 'v_mul_f32 v6, v1, v1']
    assert ca.extract_int_role(lines) == set()


CALLS = (
    '0000000000001000 <_Z6kernelv>:\n'
    '\ts_getpc_b64 s[2:3]    // 000000001000: BE824700\n'
    '\ts_add_u32 s2, s2, 0x100    // 000000001004: 8002FF02 00000100\n'
    '\ts_addc_u32 s3, s3, 0    // 00000000100C: 8203FF03 00000000\n'
    '\ts_swappc_b64 s[30:31], s[2:3]    // 000000001014: BE9E4902\n'
    '\ts_endpgm    // 000000001018: BFB00000\n'
    '\n'
    '0000000000001104 <_ZL6helperv>:\n'
    '\tv_mul_f32 v0, 0x3e800000, v0    // 000000001104: 100000FF 3E800000\n'
    '\n'
    '0000000000002000 <_ZL5otherv>:\n'
    '\tv_mul_f32 v0, 0x40400000, v0    // 000000002000: 100000FF 40400000\n'
)


def test_callee_literals_included_and_negative_offset():
    ks = ca.split_kernels(CALLS)
    addr = {0x1000: '_Z6kernelv', 0x1104: '_ZL6helperv', 0x2000: '_ZL5otherv'}
    assert ca.call_targets(ks['_Z6kernelv']) == {0x1104}
    lines = ca.with_callees('_Z6kernelv', ks.__getitem__, addr)
    got = ca.extract_literals(lines)
    assert 0x3e800000 in got and 0x40400000 not in got
    back = ['s_getpc_b64 s[2:3]  // 000000009000: BE824700', 's_add_u32 s2, s2, 0xffffff00  // 000000009004: X',
            's_addc_u32 s3, s3, lit(0xffffffff)  // 00000000900C: Y']
    assert ca.call_targets(back) == {0x9004 - 0x100}


def test_carry_out_register_is_not_a_read():
    lines = ['s_mov_b32 s5, 0x76543210', 'v_permlanex16_b32 v1, v1, s5, 0xfedcba98',
             'v_div_scale_f32 v86, s5, v68, v67, v68', 'v_mul_f32 v1, s5, v1']
    assert 0x76543210 in ca.extract_int_role(lines)
    lines = ['s_mov_b32 s5, 0x76543210', 'v_div_scale_f32 v86, s5, v68, v67, v68', 'v_mul_f32 v1, s5, v1']
    assert ca.extract_int_role(lines) == set()


def test_rodata_words_parse(monkeypatch):
    text = ('x.co:\tfile format elf64-amdgpu\nContents of section .rodata:\n'
            ' 0940 000080bf 000032bf 000006bf 0000cabe  ......2.........\n'
            ' 0950 0000a33d 00000000                    ...=....\n')

    class R:
        stdout = text

    monkeypatch.setattr(ca.subprocess, 'run', lambda *a, **k: R())
    assert ca.rodata_words(Path('x.co')) == {0xbf800000, 0xbf320000, 0xbf060000, 0xbeca0000, 0x3da30000, 0}


def test_exception_turns_gap_into_ok_with_reason(tmp_path):
    f = tmp_path / 'ex.tsv'
    f.write_text('# c\nk_dv_quant\t0xbd4ccccd\tunrolled fma folded\n')
    ex = ca.load_exceptions(f)
    assert ca.excused_for('_ZN7halogen10k_dv_quantEPKt', ex) == {0xbd4ccccd: 'unrolled fma folded'}
    assert ca.excused_for('_ZN7halogen5k_foo', ex) == {}
    c = ca.compare_sets({0xbd4ccccd, 0x358637bd}, set(), excused={0xbd4ccccd: 'unrolled fma folded'})
    assert c.missing == [0x358637bd]
    assert ca.verdict(c).startswith('AUDIT GAP 0x358637bd')
    c2 = ca.compare_sets({0xbd4ccccd}, set(), excused={0xbd4ccccd: 'unrolled fma folded'})
    assert ca.verdict(c2) == 'AUDIT OK (exception: 0xbd4ccccd (-0.05) unrolled fma folded)'


# ---- integer literals (QA on PR #24: ruling 4c covers every literal, integer and float) ----

def test_int_present_reads_inline_ints_offsets_and_hex():
    got = ca.int_present(['v_add_nc_u32 v1, 0x7e0, v2', 'v_cmp_gt_u32 vcc_lo, 32, v0', 'ds_load_b32 v1, v2 offset:2016',
                          'ds_load_2addr_b32 v[0:1], v2 offset0:4 offset1:8', 'global_load_b32 v1, v[2:3], off offset:-64',
                          's_waitcnt vmcnt(7)', 's_cbranch_scc1 99'])
    assert {0x7e0, 32, 2016, 4, 16, 8, 32, (-64) & 0xffffffff} <= got
    assert 7 not in got and 99 not in got


def test_inline_int_is_a_vendor_literal_except_memory_and_counters():
    got = ca.inline_ints(['v_cmp_gt_u32 vcc_lo, 48, v0', 'v_lshlrev_b32 v1, 2, v1', 'ds_load_b32 v1, v2 offset:40',
                          's_waitcnt vmcnt(3)', 'v_mul_f32 v0, 0.5, v1', 'v_add_nc_u32 v1, -16, v1'])
    assert got == {48, 2, (-16) & 0xffffffff}


def test_range_role_direct_and_through_mov():
    lines = ['v_cmp_gt_u32 vcc_lo, 0x1ff, v0', 's_mov_b32 s4, 0x3c00', 'v_min_u32 v1, s4, v1', 'v_med3_i32 v2, v2, 0x7f, -16',
             'v_and_b32 v3, 0xffff, v3', 's_movk_i32 s7, 0x4460', 's_cselect_b32 s8, 0x60, 0x70']
    assert ca.range_role(lines) == {0x1ff, 0x3c00, 0x7f, (-16) & 0xffffffff}


def test_range_role_through_bound_arithmetic():
    lines = ['s_add_i32 s5, s2, 0xfffff800', 'v_cmp_gt_i32 vcc_lo, s5, v0']
    assert ca.range_role(lines) == {0xfffff800}


def test_unread_literal_has_no_role():
    lines = ['v_mov_b32 v9, 0xf004', 'v_add_nc_u32 v1, 0x60, v2', 'global_store_b32 v[4:5], v1, off']
    assert ca.roled(lines) == {0x60}


def _ints(vendor, rewrite, rng=frozenset(), roled=None, excused=None):
    return ca.compare_sets(vendor, rewrite, int_role=set(vendor), range_vals=set(rng),
                           roled=set(vendor) if roled is None else set(roled), excused=excused)


def test_data_path_int_gap_is_explained_by_role():
    c = _ints({0x7e0, 0xffff0000}, {0x10})
    assert c.int_missing == [0x7e0, 0xffff0000] and c.int_gap == []
    assert ca.verdict(c) == 'AUDIT OK'


def test_bound_gap_fails_unless_equal_adjacent_or_complement():
    c = _ints({0x1ff, 0x400, 0x3c00, 0x10}, {0x200, 0xfffffc00}, rng={0x1ff, 0x400, 0x3c00, 0x10})
    assert c.int_gap == [0x10, 0x3c00]
    assert ca.verdict(c) == 'AUDIT GAP int 0x10 (bound), int 0x3c00 (bound)'


def test_unknown_role_gap_fails():
    c = _ints({0xf004}, set(), roled=set())
    assert c.int_gap == [0xf004]
    assert ca.verdict(c) == 'AUDIT GAP int 0xf004 (no reader)'


def test_bound_not_excused_by_sign_flip_twin():
    c = _ints({0x7f}, {0x807f}, rng={0x7f})
    assert c.int_gap == [0x7f] and c.flipped == []


def test_int_exception_explains_bound():
    c = _ints({0x3c00}, set(), rng={0x3c00}, excused={0x3c00: 'loop end in bytes; rewrite counts rows'})
    assert c.int_gap == []
    assert ca.verdict(c) == 'AUDIT OK (exception: 0x3c00 loop end in bytes; rewrite counts rows)'


def test_row_lists_bound_gaps():
    c = _ints({0x3c00, 0x7e0}, set(), rng={0x3c00})
    assert ca.format_row('i', 'o', c) == '| i | o | 2 | 0 |  |  | 0x7e0 | AUDIT GAP int 0x3c00 (bound) |'


def test_audit_end_to_end_bound_offset_and_complement():
    vendor = ['v_cmp_gt_u32 vcc_lo, 0x1ff, v0', 'v_add_nc_u32 v1, 0x7e0, v2', 'ds_store_b32 v1, v3',
              'v_and_b32 v4, 0xffffff00, v4', 'v_cmp_lt_i32 vcc_lo, 0x3c00, v5', 'v_mul_f32 v6, 0x3e22f983, v6']
    rewrite = ['v_cmp_lt_u32 vcc_lo, 0x200, v0', 'ds_store_b32 v2, v3 offset:64', 'v_and_b32 v4, -0x100, v4',
               'v_mul_f32 v6, 0.15915494, v6']
    c = ca.audit(vendor, rewrite)
    assert c.int_missing == [0x7e0] and c.int_gap == [0x3c00] and c.missing == []
    assert ca.verdict(c) == 'AUDIT GAP int 0x3c00 (bound)'


def test_exceptions_dir_merged_with_main_file(tmp_path):
    main = tmp_path / 'ex.tsv'
    main.write_text('k_a\t0x10\tmain reason\n')
    d = tmp_path / 'const-audit-exceptions.d'
    d.mkdir()
    (d / 'g30.tsv').write_text('# group 30\nk_b\t0x3c00\tloop end in bytes\n')
    assert ca.load_exceptions(main) == [('k_a', 0x10, 'main reason'), ('k_b', 0x3c00, 'loop end in bytes')]
