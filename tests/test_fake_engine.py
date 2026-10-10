"""Tests for tests/fake_engine.py: the fake engine used by the probe tests behaves like the real grammar."""
import socket
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fake_engine


def connect():
    holder = types.SimpleNamespace(PORT=0)
    fake_engine.start(holder)
    s = socket.create_connection(('127.0.0.1', holder.PORT))
    s.settimeout(2)
    return holder, s, s.makefile('rwb', buffering=0)


def test_owner_gets_every_verb_second_connection_only_info_and_ping():
    holder, owner_sock, owner = connect()
    owner.write(b'CSTAT\n')
    assert owner.readline().startswith(b'C ')
    second = socket.create_connection(('127.0.0.1', holder.PORT))
    second.settimeout(1)
    sf = second.makefile('rwb', buffering=0)
    sf.write(b'PING\n')
    assert sf.readline() == b'PONG\n'
    sf.write(b'INFO\n')
    assert sf.readline().startswith(b'I 1 0 262144')
    owner_sock.close()
    second.close()


def test_gen_streams_tokens_then_done_and_rejects_drafter_two():
    _, sock, f = connect()
    f.write(b'GEN 5 4 2 151645 151643 3 10 11 12\n')
    lines = [f.readline().decode() for _ in range(5)]
    assert [ln.split()[0] for ln in lines] == ['T', 'T', 'T', 'T', 'D']
    f.write(b'GEN 6 4 2 151645 151643 3 10 11 12 2\n')
    assert 'asked for drafter 2 (unknown)' in f.readline().decode()
    sock.close()
