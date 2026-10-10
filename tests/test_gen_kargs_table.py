"""Tests for scripts/capture/gen_kargs_table.py: relocation, stub and metadata parsing into the shim's tables."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'capture'))
import gen_kargs_table as g

SCAN = '_ZN7halogen12_GLOBAL__N_112k_dnc_scan3xILi2ELi4EEEv'
OTHER = '_ZN7halogen12_GLOBAL__N_14k_glu'

RELA = """00000000018f9dd0 R_X86_64_RELATIVE  *ABS*+0x00000000017b0000
00000000018f9dd8 R_X86_64_RELATIVE  *ABS*+0x00000000017b1000
00000000018f9de0 R_X86_64_RELATIVE  *ABS*+0x00000000017b2000
"""
STUBS = f"addr\tname\n018b0000\t{SCAN}\n018b1000\t{OTHER}\n"
NOTES = f"""  - .args:
      - .offset:         0
        .size:           8
      - .offset:         128
        .size:           40
      - .offset:         168
        .size:           40
    .kernarg_segment_size: 208
    .name:           {SCAN}
"""


def test_parse_rela_maps_slot_to_stub():
    assert g.parse_rela(RELA) == {0x18f9dd0: 0x17b0000, 0x18f9dd8: 0x17b1000, 0x18f9de0: 0x17b2000}


def test_parse_stubs_removes_ghidra_bias_and_skips_header():
    assert g.parse_stubs(STUBS) == {0x17b0000: SCAN, 0x17b1000: OTHER}


def test_parse_notes_reads_args_and_kernarg_size():
    k = g.parse_notes(NOTES)[SCAN]
    assert k['size'] == 208 and k['args'] == [(0, 8), (128, 40), (168, 40)]


def test_build_rows_keeps_only_scan3_with_metadata_and_flags_tap():
    rows = g.build_rows(g.parse_rela(RELA), g.parse_stubs(STUBS), g.parse_notes(NOTES))
    assert rows == [(0x18f9dd0, 0x17b0000, SCAN, 208, [(0, 8), (128, 40), (168, 40)], 1)]
    assert 'KTAB_N 1' in g.emit(rows) and '0x18f9dd0UL' in g.emit(rows)


def test_build_names_covers_every_known_stub_only():
    names = g.build_names(g.parse_rela(RELA), g.parse_stubs(STUBS))
    assert names == [(0x18f9dd0, SCAN), (0x18f9dd8, OTHER)]
    assert 'KNAMES_N 2' in g.emit_names(names)
