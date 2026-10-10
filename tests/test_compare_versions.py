"""Tests for scripts/compare_versions.py matching tiers, using tiny fixture exports."""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / 'scripts' / 'compare_versions.py'
HEADER = 'addr\tname\tbytes\tcallees\tcallers\tstrings\n'


def export(d: Path, funcs):
    d.mkdir()
    rows = []
    for addr, name, size, strings, body in funcs:
        (d / f'{addr}.c').write_text(body)
        rows.append(f'{addr}\t{name}\t{size}\t1\t1\t{strings}')
    (d / 'index.tsv').write_text(HEADER + '\n'.join(rows) + '\n')


def run(tmp_path, old, new):
    export(tmp_path / 'old', old)
    export(tmp_path / 'new', new)
    fmap = tmp_path / 'map.tsv'
    fmap.write_text('addr\trole\tlabel\tconfidence\tevidence\n' + ''.join(
        f'{a}\tutil\tf{a}\thigh\tx\n' for a, *_ in new))
    out_md, out_tsv = tmp_path / 'o.md', tmp_path / 'o.tsv'
    subprocess.run([sys.executable, '-I', str(SCRIPT), str(tmp_path / 'old'), str(tmp_path / 'new'),
                    str(fmap), str(out_md), str(out_tsv)], check=True, capture_output=True)
    rows = [r.split('\t') for r in out_tsv.read_text().splitlines()[1:]]
    return {r[0]: r for r in rows}, out_md.read_text()


def test_relocation_is_unchanged(tmp_path):
    body_old = 'void f(void) {\n  FUN_00001000(local_10, 0x01234567);\n  goto switchD_00001040_caseD_b;\n}\n'
    body_new = 'void f(void) {\n  FUN_00002000(local_28, 0x09abcdef);\n  goto switchD_00002080_caseD_b;\n}\n'
    res, _ = run(tmp_path, [('00001000', 'FUN_00001000', 50, '', body_old)],
                 [('00002000', 'FUN_00002000', 50, '', body_new)])
    assert res['00002000'][1] == 'unchanged'
    assert res['00002000'][2] == '00001000'


def test_real_edit_with_same_strings_is_changed(tmp_path):
    res, _ = run(tmp_path, [('00001000', 'a', 60, 'hello', 'int f(void){ return 1; }')],
                 [('00002000', 'a', 64, 'hello', 'int f(void){ return 2; }')])
    assert res['00002000'][1] == 'changed'


def test_added_and_removed_are_reported(tmp_path):
    res, md = run(tmp_path, [('00001000', 'old_only', 60, 'gone', 'int g(void){ return 7; }')],
                  [('00003000', 'new_only', 80, 'fresh', 'int h(void){ return 9; }')])
    assert res['00003000'][1] == 'added'
    assert '| removed (old only) | 1 |' in md
