"""Tests for scripts/engine_probe6.py: the prompt generator and the send plan (no sockets)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import engine_probe6 as p6


def test_make_tokens_length_range_and_determinism():
    a = p6.make_tokens(1100, 1)
    assert len(a) == 1100
    assert a == p6.make_tokens(1100, 1)
    assert all(p6.LO <= t < p6.HI for t in a)
    assert not set(a) & p6.SPECIAL


def test_make_tokens_prefix_stable_and_salt_differs():
    assert p6.make_tokens(300, 1) == p6.make_tokens(1100, 1)[:300]
    assert p6.make_tokens(300, 1) != p6.make_tokens(300, 2)


def test_plan_hits_multiple_64_token_blocks_and_cache_reuse():
    steps = {name: ids for name, ids, _ in p6.plan()}
    assert len(steps['A300_cold']) == 300 and len(steps['A1100_ext']) == 1100
    assert len(steps['A300_cold']) // 64 >= 4 and len(steps['A1100_ext']) // 64 >= 17
    assert steps['A300_warm'] == steps['A300_cold']
    assert steps['A1100_ext'][:300] == steps['A300_cold']
    assert steps['B1100_cold'][:64] != steps['A1100_ext'][:64]
    assert steps['A1100_warm'] == steps['A1100_ext']


def test_plan_max_tokens_small():
    assert all(1 <= mx <= 8 for _, _, mx in p6.plan())
