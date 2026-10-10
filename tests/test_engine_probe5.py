"""Tests for scripts/engine_probe5.py: the timeline split and the drain loop, with the socket stubbed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import engine_probe5 as p5


def test_split_at_separates_gen_done_from_close():
    lines = [(9.0, 'C x'), (10.5, 'C 1'), (14.0, 'PONG'), (20.0, 'T 1 5')]
    assert p5.split_at(lines, t_gen_done=10.0, t_close=15.0) == {'before_close': 2, 'after_close': 1}


def test_split_at_empty_is_zero_not_missing():
    assert p5.split_at([], 1.0, 2.0) == {'before_close': 0, 'after_close': 0}


def test_first_word_keeps_blank_lines():
    assert p5.first_word([(0, 'PONG'), (0, ''), (0, 'C 1 2')]) == ['PONG', '', 'C']


class FakeConn:
    def __init__(self, replies):
        self.replies = list(replies)

    def read(self, timeout):
        return self.replies.pop(0) if self.replies else ''


def test_drain_skips_timeouts_and_stops_on_eof():
    got = p5.drain(FakeConn(['C 1', None, 'PONG', '']), 5)
    assert [ln for _, ln in got] == ['C 1', 'PONG']
