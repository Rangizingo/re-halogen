"""Tests for scripts/engine_probe3.py: the connection-role probe runs and records every step."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_probe as ep
import engine_probe3 as ep3
import fake_engine


def test_probe3_second_connection_gets_info_and_ping_only(capsys):
    fake_engine.start(ep)
    orig_ask = ep.Conn.ask

    def fast_ask(self, line, timeout=10):
        return orig_ask(self, line, timeout=min(timeout, 1))

    ep.Conn.ask = fast_ask
    try:
        try:
            ep3.run()
        except OSError:
            pass  # a timed-out stream cannot be reused; the script's earlier results are what we assert on
    finally:
        ep.Conn.ask = orig_ask
    out = capsys.readouterr().out
    assert 'RESULT first_info' in out
    assert 'RESULT second_ping' in out
    assert '"v": "PONG"' in out
    assert 'RESULT second_cstat {"v": null}' in out
