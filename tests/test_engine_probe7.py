"""Tests for scripts/engine_probe7.py: the SNAP tap plan (no sockets)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import engine_probe7 as p7


def test_expected_slots_follow_the_r6_recipe():
    # analysis/closure/dnc-scan3-r6.md probe recipe: SNAP 1000 SNAP2 500 SNAP3 800 from done = 0 gives 6, 11, 14.
    assert p7.expected_slots((1000, 500, 800), 1100) == [6, 11, 14]
    assert p7.expected_slots((1000, 0, 0), 1100) == [-1, -1, 14]
    assert p7.expected_slots((1100, 0, 0), 1100) == [-1, -1, -1]  # p == n never taps
    assert p7.expected_slots((40, 0, 0), 1100) == [-1, -1, -1]    # under 64 tokens from done


def test_plan_is_cold_unique_and_sends_snaps():
    steps = p7.plan()
    names = [s[0] for s in steps]
    assert names[0] == 'S1100_snap3' and len(set(names)) == len(names)
    first = steps[0]
    assert len(first[1]) == 1100 and first[3] == (1000, 500, 800)
    assert steps[1][1][:64] != first[1][:64]  # second cold prompt shares no first block
    resume = [s for s in steps if s[0] == 'S1200_resume'][0]
    assert resume[1][:1000] == first[1][:1000]
    assert all(1 <= s[2] <= 8 for s in steps)
