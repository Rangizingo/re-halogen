#!/usr/bin/env python3
"""Sixth probe (p7-10, U12): multi-block prefills so k_dnc_scan3x runs several 64-token blocks per layer.

Run inside the engine's container: podman exec NAME python3 /probe/engine_probe6.py OUT.jsonl
Needs engine_probe.py beside it. Prompts are deterministic pseudo-random token ids (no special ids):
A300 (cold), A300 again (prompt-cache hit), A1100 = A300 + 800 more (prefix extension, 300 is not a multiple
of 64 so the tap/checkpoint boundary logic sees a ragged tail), B1100 (unrelated), A1100 again.
"""
import sys
import time

import engine_probe as ep

GAP_S = 3
SPECIAL = {151643, 151645}
LO, HI = 1000, 100000


def make_tokens(n, salt):
    """n deterministic token ids in [LO, HI), never a special id. Same salt gives a prefix-stable stream."""
    out, x = [], (salt * 2654435761 + 12345) % 2**32
    while len(out) < n:
        x = (x * 1103515245 + 12345) % 2**31
        t = LO + x % (HI - LO)
        if t not in SPECIAL:
            out.append(t)
    return out


def plan():
    """(name, token ids, max_tokens) in send order."""
    a = make_tokens(1100, 1)
    return [('A300_cold', a[:300], 4), ('A300_warm', a[:300], 4), ('A1100_ext', a, 4),
            ('B1100_cold', make_tokens(1100, 2), 4), ('A1100_warm', a, 4)]


def run():
    c = ep.Conn('probe6')
    ep.result('info', v=c.ask('INFO'))
    for name, ids, mx in plan():
        toks, done = ep.gen(c, ids, mx, wait=600)
        ep.result(name, prompt_tokens=len(ids), out_tokens=len(toks), done=done)
        time.sleep(GAP_S)  # idle gap so analyse_kargs.py --trace splits requests by timestamp
    ep.result('cstat_end', v=c.ask('CSTAT'))
    c.s.close()


def main():
    with open(sys.argv[1], 'a') as out:
        ep.OUT = out
        run()


if __name__ == '__main__':
    main()
