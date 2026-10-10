#!/usr/bin/env python3
"""Fifth probe (p7-9): CSTAT on a waiting connection, GEN completion vs close, third-connection release (U26).

Run inside the engine's container: podman exec NAME python3 /probe/engine_probe5.py OUT.jsonl
Needs engine_probe.py beside it. Every line sent and received goes to OUT.jsonl.
Order: first INFO; second INFO, CSTAT, PING (waiting); first GEN, then first idles IDLE s open;
first closes; second drains; third INFO, CSTAT, GEN (waiting); second closes; third drains.
"""
import sys
import time

import engine_probe as ep

P = [785, 6722, 315, 9625, 374] * 8
IDLE = 20


def drain(c, secs):
    """Every line c receives within secs, as (time, line); stops early on EOF."""
    got, end = [], time.time() + secs
    while time.time() < end:
        line = c.read(max(0.5, end - time.time()))
        if line is None:
            continue
        if line == '':
            break
        got.append((round(time.time(), 3), line))
    return got


def split_at(lines, t_gen_done, t_close):
    """Count lines that arrived after GEN finished but before close, and after close."""
    return {'before_close': sum(1 for t, _ in lines if t_gen_done <= t < t_close),
            'after_close': sum(1 for t, _ in lines if t >= t_close)}


def first_word(lines):
    return [ln.split()[0] if ln.split() else '' for _, ln in lines]


def run():
    c1 = ep.Conn('first')
    ep.result('first_info', v=c1.ask('INFO'))

    c2 = ep.Conn('second')
    ep.result('second_info', v=c2.ask('INFO'))
    c2.send('CSTAT')
    c2.send('PING')
    early = drain(c2, 15)
    ep.result('second_waiting_cstat_ping', words=first_word(early), lines=[ln for _, ln in early])

    toks, done = ep.gen(c1, P, 8, wait=60)
    t_gen_done = time.time()
    ep.result('first_gen', tokens=len(toks), done=done)
    idle = drain(c2, IDLE)
    ep.result('first_cstat_after_gen', v=c1.ask('CSTAT'))
    t_close = time.time()
    c1.s.close()
    after = drain(c2, 15)
    lines = idle + after
    ep.result('second_release', split=split_at(lines, t_gen_done, t_close), words=first_word(lines),
              idle_s=IDLE, lines=[ln for _, ln in lines])

    c3 = ep.Conn('third')
    ep.result('third_info', v=c3.ask('INFO'))
    c3.send('CSTAT')
    third_early = drain(c3, 10)
    ep.result('third_waiting_cstat', words=first_word(third_early), lines=[ln for _, ln in third_early])
    ep.REQ[0] += 1
    req = ep.REQ[0]
    c3.send(' '.join([f'GEN {req} 8 2 151645 151643 {len(P)}'] + [str(i) for i in P]))
    held = drain(c3, 20)
    ep.result('third_gen_while_second_open', words=first_word(held))
    ep.result('second_cstat_served', v=c2.ask('CSTAT'))
    t2_close = time.time()
    c2.s.close()
    rel = drain(c3, 20)
    ep.result('third_release', words=first_word(rel), first_line_after_close_s=(
        round(rel[0][0] - t2_close, 3) if rel else None), lines=[ln for _, ln in rel])
    ep.result('third_cstat_served', v=c3.ask('CSTAT'))
    c3.s.close()


def main():
    with open(sys.argv[1], 'a') as out:
        ep.OUT = out
        run()


if __name__ == '__main__':
    main()
