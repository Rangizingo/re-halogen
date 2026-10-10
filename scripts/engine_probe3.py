#!/usr/bin/env python3
"""Third probe: connection roles. Opens an owner connection first, then a second one,
and records what each connection gets for INFO, CSTAT, GEN. Needs engine_probe.py beside it."""
import sys

import engine_probe as ep


def run():
    c1 = ep.Conn('first')
    c2 = ep.Conn('second')
    ep.result('first_info', v=c1.ask('INFO'))
    ep.result('second_info', v=c2.ask('INFO'))
    ep.result('second_ping', v=c2.ask('PING'))
    ep.result('second_cstat', v=c2.ask('CSTAT', timeout=10))
    toks, done = ep.gen(c2, [7734, 264, 2716], 4, wait=40)
    ep.result('second_gen', tokens=toks, done=done)
    toks, done = ep.gen(c1, [7734, 264, 2716], 4, wait=40)
    ep.result('first_gen', tokens=toks, done=done)


def main():
    with open(sys.argv[1], 'a') as out:
        ep.OUT = out
        run()


if __name__ == '__main__':
    main()
