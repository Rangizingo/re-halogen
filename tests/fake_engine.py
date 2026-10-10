"""A fake flash_serve engine speaking the wire grammar in analysis/wire-protocol.md."""
import socket
import threading

INFO = 'I 1 0 262144 7 1 1 0 110 64 4 262144 2 1 1 524288 1 0.17.3 2 6 gguf 1 0 0 0 0 - 2048\n'
CSTAT = 'C ' + ' '.join(['0'] * 24) + '\n'


def _handle(conn, owner):
    f = conn.makefile('rwb', buffering=0)
    for raw in f:
        p = raw.decode().split()
        if not p:
            continue
        if p[0] == 'PING':
            f.write(b'PONG\n')
        elif p[0] == 'INFO':
            f.write(INFO.encode())
        elif owner and p[0] == 'CSTAT':
            f.write(CSTAT.encode())
        elif owner and p[0] == 'GEN':
            req, n_eos = p[1], int(p[3])
            n_ids = int(p[4 + n_eos])
            rest = p[5 + n_eos + n_ids:]
            seed = int(rest[rest.index('SAMPLE') + 5]) if 'SAMPLE' in rest else 0
            if rest[:1] == ['2']:
                f.write(f'D {req} error 0 0 0 0 asked for drafter 2 (unknown)\n'.encode())
                continue
            for i in range(4):
                f.write(f'T {req} {seed % 100 + i}\n'.encode())
            f.write(f'D {req} length {n_ids} 4 0.1 0.2 0 0 0 0 0 0\n'.encode())


def start(probe_module):
    """Start the fake engine on a free port and point probe_module.PORT at it."""
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    sock.listen(8)
    probe_module.PORT = sock.getsockname()[1]

    def accept_loop():
        first = True
        while True:
            conn, _ = sock.accept()
            threading.Thread(target=_handle, args=(conn, first), daemon=True).start()
            first = False

    threading.Thread(target=accept_loop, daemon=True).start()
