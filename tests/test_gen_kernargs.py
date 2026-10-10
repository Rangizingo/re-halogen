"""Tests for rewrite/harness/gen_kernargs.py: llvm-readelf notes parsing into kernarg layouts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'rewrite' / 'harness'))
import gen_kernargs as g

NAME = '_ZN7halogen12_GLOBAL__N_112k_scale_siluEPKtPtlf'
NOTES = f"""Displaying notes found in: .note
    AMDGPU Metadata:
        ---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_group_size_x
    .group_segment_fixed_size: 16
    .kernarg_segment_size: 288
    .max_flat_workgroup_size: 1024
    .name:           {NAME}
    .private_segment_fixed_size: 4
    .symbol:         {NAME}.kd
    .wavefront_size: 32
  - .args:
      - .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 256
    .name:           _ZN7halogen2k2Ev
    .private_segment_fixed_size: 0
    .wavefront_size: 32
amdhsa.version:
  - 1
"""
STUBS = f"stub_addr\tmangled_kernel\tdemangled_kernel\tfamily\n018a9540\t{NAME}\tdemangled\tscale\n018a9550\t_ZN7halogen7missingEv\tx\tmiss\n"


def test_parse_notes_args_and_kernel_keys():
    ks = g.parse_notes(NOTES)
    assert set(ks) == {NAME, '_ZN7halogen2k2Ev'}
    k = ks[NAME]
    assert k['args'] == [[0, 8, 'global_buffer'], [8, 4, 'by_value'],
                         [32, 4, 'hidden_block_count_x'], [44, 2, 'hidden_group_size_x']]
    assert k['kernarg_segment_size'] == 288 and k['max_flat_workgroup_size'] == 1024
    assert k['group_segment_fixed_size'] == 16


def test_keys_after_name_are_kept():
    k = g.parse_notes(NOTES)[NAME]
    assert k['private_segment_fixed_size'] == 4 and k['wavefront_size'] == 32


def test_second_kernel_not_polluted_by_first():
    k2 = g.parse_notes(NOTES)['_ZN7halogen2k2Ev']
    assert len(k2['args']) == 1 and k2['kernarg_segment_size'] == 264


def test_parse_stubs_skips_header():
    assert g.parse_stubs(STUBS) == [(NAME, 'scale'), ('_ZN7halogen7missingEv', 'miss')]


def test_build_counts_covered_and_missing():
    table, missing = g.build(g.parse_stubs(STUBS), {'codeobj_a': g.parse_notes(NOTES)})
    assert list(table) == [NAME] and missing == ['_ZN7halogen7missingEv']
    assert table[NAME]['object'] == 'codeobj_a' and table[NAME]['family'] == 'scale'
    assert table[NAME]['kernarg_size'] == 288


def test_hidden_offsets():
    k = g.parse_notes(NOTES)[NAME]
    assert g.hidden_offsets(k['args']) == {'block_count_x': 32, 'group_size_x': 44}
