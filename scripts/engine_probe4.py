#!/usr/bin/env python3
"""Fourth probe (p7-7): second-connection GEN (U26) and serial-sampling seed determinism (U23).

Run inside the engine's container: podman exec NAME python3 /probe/engine_probe4.py OUT.jsonl
Needs engine_probe.py beside it. Every line sent and received goes to OUT.jsonl.
"""
import sys

import engine_probe as ep

P = [785, 6722, 315, 9625, 374] * 8
HOT = (1.2, 0, 1.0, 0.0)


def connections():
    c1 = ep.Conn('first')
    c2 = ep.Conn('second')
    ep.result('first_info', v=c1.ask('INFO'))
    ep.result('second_info', v=c2.ask('INFO'))
    toks, done = ep.gen(c2, P, 8, wait=60)
    ep.result('second_gen', tokens=toks, done=done)
    ep.result('second_after_gen_ping', v=c2.ask('PING'))
    toks, done = ep.gen(c1, P, 8, wait=60)
    ep.result('first_gen', tokens=toks, done=done)
    c1.s.close()
    toks, done = ep.gen(c2, P, 8, wait=60)
    ep.result('second_gen_after_first_closed', tokens=toks, done=done)
    c3 = ep.Conn('third')
    toks, done = ep.gen(c3, P, 8, wait=60)
    ep.result('third_gen_after_first_closed', tokens=toks, done=done)
    return c3 if done else c2


def seeds(c):
    runs = {}
    for name, drafter, seed in (('serial_111_a', 0, 111), ('serial_111_b', 0, 111), ('serial_222', 0, 222),
                                ('mtp_111_a', 1, 111), ('mtp_111_b', 1, 111)):
        runs[name] = ep.gen(c, P, 32, drafter=drafter, sample=HOT + (seed,))
    t = {k: v[0] for k, v in runs.items()}
    ep.result('seeds', serial_111_equal=t['serial_111_a'] == t['serial_111_b'],
              serial_111_vs_222_equal=t['serial_111_a'] == t['serial_222'],
              mtp_111_equal=t['mtp_111_a'] == t['mtp_111_b'],
              serial_vs_mtp_111_equal=t['serial_111_a'] == t['mtp_111_a'],
              lens={k: len(v) for k, v in t.items()}, done={k: v[1] for k, v in runs.items()})


def main():
    with open(sys.argv[1], 'a') as out:
        ep.OUT = out
        seeds(connections())


if __name__ == '__main__':
    main()
