#!/usr/bin/env python3
"""Seventh probe (p7-11, U12): make k_dnc_scan3x Taps fire by sending SNAP fields.

Run inside the engine's container: podman exec NAME python3 /probe/engine_probe7.py OUT.jsonl
Recipe from analysis/closure/dnc-scan3-r6.md: with HALOGEN_PROMPT_CACHE=2 a cold prompt with
SNAP 1000 SNAP2 500 SNAP3 800 should set scan3x kernarg slots +0x80/+0x84/+0x88 to 6, 11, 14.
Prompts use salts not used by probe 6, so nothing hits the run 4 cache.
"""
import sys
import time

import engine_probe as ep
from engine_probe6 import make_tokens

GAP_S = 3


def expected_slots(snap, n, done=0):
    """Slots (slot0 <- SNAP2, slot1 <- SNAP3, slot2 <- SNAP) per dnc-scan3-r6.md 2b, for a piece that fits one chunk."""
    out = []
    for p in (snap[1], snap[2], snap[0]):
        d = p - done
        out.append((d & 0x7fffffc0) // 64 - 1 if p and done <= p < done + n and d >= 64 else -1)
    return out


def plan():
    """(name, token ids, max_tokens, (SNAP, SNAP2, SNAP3)) in send order."""
    a = make_tokens(1100, 7)
    return [('S1100_snap3', a, 4, (1000, 500, 800)),
            ('S1100_snaponly', make_tokens(1100, 8), 4, (1000, 0, 0)),
            ('S1200_resume', a[:1000] + make_tokens(200, 9), 4, (0, 0, 0))]


def run():
    c = ep.Conn('probe7')
    ep.result('info', v=c.ask('INFO'))
    for name, ids, mx, snap in plan():
        toks, done = ep.gen(c, ids, mx, wait=600, snap=snap)
        ep.result(name, prompt_tokens=len(ids), snap=list(snap), expect_slots=expected_slots(snap, len(ids)),
                  out_tokens=len(toks), done=done)
        time.sleep(GAP_S)  # idle gap so analyse_kargs.py --trace splits requests by timestamp
    ep.result('cstat_end', v=c.ask('CSTAT'))
    c.s.close()


def main():
    with open(sys.argv[1], 'a') as out:
        ep.OUT = out
        run()


if __name__ == '__main__':
    main()
