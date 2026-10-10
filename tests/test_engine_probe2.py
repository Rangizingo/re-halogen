"""Tests for scripts/engine_probe2.py: the varied-prompt probe runs end to end on a fake engine."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_probe as ep
import engine_probe2 as ep2
import fake_engine


def test_probe2_reports_equalities(capsys):
    fake_engine.start(ep)
    ep2.run()
    out = capsys.readouterr().out
    assert 'RESULT P_mtp_vs_serial' in out
    assert '"mtp_warm_eq_serial": true' in out
    assert 'RESULT seeds' in out
    assert '"seed111_twice_equal": true' in out
    assert '"seed111_vs_222_equal": false' in out


def test_cached_reads_n_cached_field():
    assert ep2.cached('D 1 length 19 64 0.1 0.2 1 5 31 19 1') == 19
    assert ep2.cached('D 1 error 0 0 0 0 text') is None
    assert ep2.cached(None) is None
