#!/usr/bin/env python3
"""Second probe: varied prompts. Run in the engine container: python3 engine_probe2.py OUT.jsonl
Needs engine_probe.py beside it. Tests: cold vs warm cache, serial vs MTP in both orders,
sampling seeds at temperature 1.2. Prompts are real token ids from the model tokenizer."""
import json
import sys

import engine_probe as ep

P = [7734, 264, 2716, 3255, 883, 264, 324, 55965, 51624, 852, 13325, 264, 1876, 303, 264, 16002, 13, 18037, 25]
Q = [7734, 264, 6366, 3255, 883, 264, 324, 55965, 51624, 852, 13325, 264, 1876, 303, 264, 16002, 13, 18037, 25,
     7734, 264, 2716, 3255, 883, 264]
N = 64


def text(toks):
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file('/models/tokenizer/tokenizer.json').decode(toks)[:120]
    except Exception as e:  # noqa: BLE001
        return f'decode failed: {e}'


def cached(done):
    p = (done or '').split()
    return int(p[10]) if len(p) > 10 and p[10].lstrip('-').isdigit() else None


def run():
    c = ep.Conn('owner')
    ep.result('info', info=c.ask('INFO'))
    mc, dmc = ep.gen(c, P, N, drafter=1)           # P: mtp, cold
    mw, dmw = ep.gen(c, P, N, drafter=1)           # P: mtp, warm
    sw, dsw = ep.gen(c, P, N, drafter=0)           # P: serial, warm
    sw2, dsw2 = ep.gen(c, P, N, drafter=0)         # P: serial, warm again
    ep.result('P_mtp_vs_serial', text=text(mc), mtp_cold_eq_warm=mc == mw, mtp_warm_eq_serial=mw == sw,
              serial_eq_serial=sw == sw2, n=len(mc), cached=[cached(dmc), cached(dmw), cached(dsw), cached(dsw2)],
              done=[dmc, dmw, dsw, dsw2])
    sc, dsc = ep.gen(c, Q, N, drafter=0)           # Q: serial, cold
    mq, dmq = ep.gen(c, Q, N, drafter=1)           # Q: mtp, warm
    ep.result('Q_serial_vs_mtp', text=text(sc), serial_cold_eq_mtp_warm=sc == mq, n=len(sc),
              cached=[cached(dsc), cached(dmq)], done=[dsc, dmq])
    a, da = ep.gen(c, P, N, drafter=1, sample=(1.2, 0, 1.0, 0.0, 111))
    b, db = ep.gen(c, P, N, drafter=1, sample=(1.2, 0, 1.0, 0.0, 111))
    d, dd = ep.gen(c, P, N, drafter=1, sample=(1.2, 0, 1.0, 0.0, 222))
    e, de = ep.gen(c, P, N, drafter=0, sample=(1.2, 0, 1.0, 0.0, 111))
    ep.result('seeds', text=text(a), seed111_twice_equal=a == b, seed111_vs_222_equal=a == d,
              mtp111_vs_serial111_equal=a == e, n=[len(a), len(b), len(d), len(e)],
              done=[da, db, dd, de])
    ep.result('cstat_end', cstat=c.ask('CSTAT'))


def main():
    with open(sys.argv[1], 'a') as out:
        ep.OUT = out
        run()


if __name__ == '__main__':
    main()
    json.dumps({})
