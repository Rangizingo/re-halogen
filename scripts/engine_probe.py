#!/usr/bin/env python3
"""Probe a halogen flash_serve engine over its raw wire protocol (127.0.0.1:8730).

Run inside the engine's container: podman exec NAME python3 /tmp/engine_probe.py OUT.jsonl
Grammar from analysis/wire-protocol.md. Every line sent and received goes to OUT.jsonl.
Tests (each prints one RESULT line):
  owner      INFO and PING on a second connection, then CSTAT and GEN on the first (owner) connection
  greedy     same greedy prompt twice (cold then warm cache): token lists equal?
  drafters   greedy with drafter 0 (serial), 1 (mtp), 2 (dflash2): outputs equal? what does 2 return?
  seed       SAMPLE with seed 123 twice and seed 124 once
"""
import json
import socket
import sys
import time

HOST, PORT = '127.0.0.1', 8730
OUT = None
REQ = [900000]


def log(kind, **kw):
    if OUT is None:
        return
    kw['kind'] = kind
    kw['t'] = round(time.time(), 3)
    OUT.write(json.dumps(kw) + '\n')
    OUT.flush()


class Conn:
    def __init__(self, name):
        self.name = name
        self.s = socket.create_connection((HOST, PORT))
        self.f = self.s.makefile('rwb', buffering=0)

    def send(self, line):
        log('send', conn=self.name, line=line[:300])
        self.f.write((line + '\n').encode())

    def read(self, timeout):
        self.s.settimeout(timeout)
        try:
            raw = self.f.readline()
        except TimeoutError:
            log('timeout', conn=self.name, secs=timeout)
            return None
        if not raw:
            log('eof', conn=self.name)
            return ''
        line = raw.decode(errors='replace').rstrip('\n')
        log('recv', conn=self.name, line=line[:300])
        return line

    def ask(self, line, timeout=10):
        self.send(line)
        return self.read(timeout)


def gen(c, ids, max_tokens=24, drafter=None, sample=None, wait=120):
    REQ[0] += 1
    req = REQ[0]
    parts = [f'GEN {req} {max_tokens} 2 151645 151643 {len(ids)}'] + [str(i) for i in ids]
    if drafter is not None:
        parts.append(str(drafter))
    if sample:
        parts.append('SAMPLE {:.9g} {} {:.9g} {:.9g} {}'.format(*sample))
    c.send(' '.join(parts))
    toks, done = [], None
    end = time.time() + wait
    while time.time() < end:
        line = c.read(min(30, max(1, end - time.time())))
        if line is None:
            continue
        if line == '':
            break
        p = line.split()
        if p and p[0] == 'T' and len(p) > 2 and p[1] == str(req):
            toks.append(int(p[2]))
        elif p and p[0] == 'D' and len(p) > 1 and p[1] == str(req):
            done = line
            break
    return toks, done


def result(name, **kw):
    print('RESULT', name, json.dumps(kw), flush=True)
    log('result', name=name, **kw)


def main():
    global OUT
    with open(sys.argv[1], 'a') as OUT:
        run()


def run():
    prompt = [785, 6722, 315, 9625, 374] * 8
    c1 = Conn('owner')
    info = c1.ask('INFO')
    c2 = Conn('second')
    pong = c2.ask('PING')
    cstat = c1.ask('CSTAT')
    toks, done = gen(c1, prompt, 8, wait=60)
    result('owner', info=info, second_ping=pong, owner_cstat=cstat, first_gen_tokens=len(toks), done=done)

    a1, d1 = gen(c1, prompt, 32)
    a2, d2 = gen(c1, prompt, 32)
    result('greedy', cold=a1, warm=a2, equal=a1 == a2, done_cold=d1, done_warm=d2)

    s0, ds0 = gen(c1, prompt, 32, drafter=0)
    s1, ds1 = gen(c1, prompt, 32, drafter=1)
    s2, ds2 = gen(c1, prompt, 32, drafter=2, wait=60)
    result('drafters', serial=s0, mtp=s1, dflash2=s2, serial_eq_mtp=s0 == s1,
           done0=ds0, done1=ds1, done2=ds2)

    r1, dr1 = gen(c1, prompt, 32, sample=(0.7, 20, 0.8, 0, 123))
    r2, dr2 = gen(c1, prompt, 32, sample=(0.7, 20, 0.8, 0, 123))
    r3, _dr3 = gen(c1, prompt, 32, sample=(0.7, 20, 0.8, 0, 124))
    result('seed', seed123_a=r1, seed123_b=r2, seed124=r3, same_seed_equal=r1 == r2,
           diff_seed_equal=r1 == r3, done_a=dr1, done_b=dr2)
    result('cstat_end', cstat=c1.ask('CSTAT'))


if __name__ == '__main__':
    main()
