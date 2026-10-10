"""Tests for scripts/engine_probe4.py: the seed comparison logic, with gen stubbed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import engine_probe as ep
import engine_probe4 as p4


def test_seeds_reports_equality_per_drafter_and_seed(monkeypatch):
    def fake_gen(c, ids, n, drafter=None, sample=None, wait=120):
        return [drafter * 1000 + sample[4] + i for i in range(n)], f'D x length {drafter}'

    got = {}
    monkeypatch.setattr(ep, 'gen', fake_gen)
    monkeypatch.setattr(ep, 'result', lambda name, **kw: got.update({name: kw}))
    p4.seeds(None)
    r = got['seeds']
    assert r['serial_111_equal'] and r['mtp_111_equal']
    assert not r['serial_111_vs_222_equal'] and not r['serial_vs_mtp_111_equal']
    assert r['lens'] == {k: 32 for k in ('serial_111_a', 'serial_111_b', 'serial_222', 'mtp_111_a', 'mtp_111_b')}
