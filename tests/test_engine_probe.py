"""Tests for scripts/engine_probe.py against a fake engine that speaks the wire grammar."""
import socket
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import engine_probe as ep


def serve(sock):
    conn, _ = sock.accept()
    f = conn.makefile('rwb', buffering=0)
    for raw in f:
        p = raw.decode().split()
        if p[0] == 'PING':
            f.write(b'PONG\n')
        elif p[0] == 'INFO':
            f.write(b'I 1 0 262144 7 1 1 0 110 64 4 262144 2 1 1 524288 1 0.17.3 2 6 gguf 1 0 0 0 0 - 2048\n')
        elif p[0] == 'GEN':
            req, n_ids = p[1], int(p[4 + int(p[3])])
            drafter = p[5 + int(p[3]) + n_ids:]
            if drafter == ['2']:
                f.write(f'D {req} error 0 0 0 0 asked for drafter 2 (unknown)\n'.encode())
            else:
                f.write(f'T {req} 11\nT {req} 12\nD {req} length {n_ids} 2 0.1 0.2 0 0 0 0 0 0\n'.encode())


def start():
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    sock.listen(1)
    ep.PORT = sock.getsockname()[1]
    threading.Thread(target=serve, args=(sock,), daemon=True).start()
    return ep.Conn('t')


def test_ask_info_and_ping():
    c = start()
    assert c.ask('INFO').split()[17] == '0.17.3'
    assert c.ask('PING') == 'PONG'


def test_gen_collects_tokens_and_done():
    c = start()
    toks, done = ep.gen(c, [5, 6, 7], 4)
    assert toks == [11, 12]
    assert done.split()[2] == 'length'


def test_gen_with_unknown_drafter_returns_error_done():
    c = start()
    toks, done = ep.gen(c, [5, 6, 7], 4, drafter=2, wait=5)
    assert toks == []
    assert 'error' in done


def _pair():
    """A Conn wired to a local socket pair; the caller writes the engine side."""
    a, b = socket.socketpair()
    c = ep.Conn.__new__(ep.Conn)
    c.name, c.s, c.buf = 't', a, b''
    return c, b


def test_read_still_works_after_a_timeout():
    c, engine = _pair()
    assert c.read(0.05) is None
    engine.sendall(b'PONG\n')
    assert c.read(1) == 'PONG'


def test_read_splits_lines_and_keeps_a_partial_line():
    c, engine = _pair()
    engine.sendall(b'T 1 785\nT 1 67')
    assert c.read(1) == 'T 1 785'
    assert c.read(0.05) is None
    engine.sendall(b'22\nD 1 length\n')
    assert c.read(1) == 'T 1 6722'
    assert c.read(1) == 'D 1 length'


def test_read_returns_empty_on_eof():
    c, engine = _pair()
    engine.close()
    assert c.read(1) == ''
