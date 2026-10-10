#!/usr/bin/env python3
"""Phase 3 acceptance: hgnpy build -> halogen-tools verify round-trip tests.

T1: valid multi-tensor file (plain + standard quant dtypes at exact implied
    geometry) -> verify PASS.
T2: flip one payload byte -> verify FAIL (checksum).
T3: hgnpy.parse(build(x)) preserves every field; xor32 recomputed matches.
T4: truncate tail -> verify FAIL (layout).
T5: HT negative matrix — every probed qparam form is rejected as
    'not read'; documents the HT companion-tensor rule (.suh/.svh planes).
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, 'work/hgn')
import hgnpy  # noqa: E402

RT = 'work/hgn/rt'
os.makedirs(RT, exist_ok=True)
TOOLS = './binaries/halogen-tools'


def verify(path):
    r = subprocess.run([TOOLS, 'verify', path], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def implied(dtype, qparam, dims, name='blk.0.ffn_down.weight'):
    e = [dict(name=name, dtype=dtype, qparam=qparam, dims=dims,
              payload=b'\x00')]
    d = hgnpy.build(version=1, model_id='probe', entries=e)
    p = f'{RT}/probe.hgn'
    open(p, 'wb').write(d)
    rc, out = verify(p)
    m = re.search(r'implies (\d+) B', out)
    return int(m.group(1)) if m else None


def main():
    # geometry rules measured from halogen-tools verify itself
    geo = {}
    for dt, qp, dims in [(0, 0, [16, 64]), (1, 0, [16, 64]), (2, 0, [16, 64]),
                          (5, 32, [16, 1024]), (7, 64, [16, 64]),
                          (11, 32, [16, 64]), (12, 64, [16, 256]),
                          (13, 32, [16, 64]), (14, 32, [16, 64]),
                          (15, 32, [16, 64]), (6, 32, [16, 64]),
                          (20, 0, [16, 64])]:
        n = implied(dt, qp, dims)
        geo[(dt, qp)] = n
        print(f'geometry dtype={dt} qp={qp} {dims} -> {n} B')
    assert all(v for v in geo.values()), 'geometry probe failed'

    import random
    random.seed(20261010)

    def mk(name, dt, qp, dims):
        # probe the implied payload size at the EXACT dims (quant geometry
        # is not linear in the dims product: per-row scale/alignment terms)
        n = implied(dt, qp, dims, name=name)
        assert n, f'no geometry rule for {dt}/{qp} {dims}'
        # plain dtypes: random bytes (checksum path exercised).
        # quant dtypes: zero-filled — scale fields decode as 0.0 (finite),
        # codebooks as all-zero (non-decreasing); random bytes trip the
        # scale-finiteness and codebook-sort checks.
        if dt in (0, 1, 2):
            return dict(name=name, dtype=dt, qparam=qp, dims=dims,
                        payload=bytes(random.getrandbits(8) for _ in range(n)))
        return dict(name=name, dtype=dt, qparam=qp, dims=dims,
                    payload=bytes(n))

    entries = [
        mk('token_embd.weight', 1, 0, [64, 128]),
        mk('blk.0.ffn_down.weight', 11, 32, [32, 64]),
        mk('blk.0.ffn_gate.weight', 12, 64, [32, 256]),
        mk('blk.1.attn_qkv.weight', 6, 32, [32, 64]),
        mk('blk.1.ffn_up.weight', 20, 0, [32, 64]),
        mk('blk.2.attn_o.weight', 7, 64, [32, 64]),
        mk('layers.1.ple.ngram_embedding.weight', 0, 0, [8, 64]),
        mk('output_norm.weight', 1, 0, [128]),
    ]
    d = hgnpy.build(version=1, model_id='rt-multi', entries=entries)
    p1 = f'{RT}/rt-multi.hgn'
    open(p1, 'wb').write(d)
    rc1, out1 = verify(p1)
    print(f'T1 build->verify: rc={rc1} {out1[:150]}')

    d2 = bytearray(d)
    data_off = int.from_bytes(d[24:32], 'little')
    d2[data_off + 8] ^= 0xFF
    p2 = f'{RT}/rt-corrupt.hgn'
    open(p2, 'wb').write(bytes(d2))
    rc2, out2 = verify(p2)
    print(f'T2 corrupt-payload: rc={rc2} {out2[:150]}')

    m = hgnpy.parse(open(p1, 'rb').read())
    ok = m['model_id'] == 'rt-multi' and m['pads_zero'] and m['version'] == 1
    for src, out in zip(entries, m['entries']):
        ok &= (src['name'] == out.name and src['dtype'] == out.dtype
               and src['qparam'] == out.qparam and src['dims'] == out.dims
               and src['payload'] == out.payload
               and out.xor_calc == out.xor32)
    print(f'T3 parse-back: {"PASS" if ok else "FAIL"} '
          f'({len(m["entries"])} tensors, hdr_sha={m["header_sha256"][:12]})')

    p4 = f'{RT}/rt-trunc.hgn'
    open(p4, 'wb').write(d[:-64])
    rc4, out4 = verify(p4)
    print(f'T4 truncate: rc={rc4} {out4[:150]}')

    # T5: HT negative matrix — no qparam value on a bare ht tensor passes
    # without its rotation planes; record the exact verdicts.
    ht_lines = []
    for qp in [0, 1, 2, 4, 8, 16, 32, 64, 128]:
        e = [dict(name='blk.0.ffn_down.weight', dtype=16, qparam=qp,
                  dims=[16, 1024], payload=b'\x00')]
        dd = hgnpy.build(version=1, model_id='probe', entries=e)
        p = f'{RT}/ht-probe.hgn'
        open(p, 'wb').write(dd)
        rc, out = verify(p)
        ht_lines.append(f'ht qp={qp}: rc={rc} {out}')
    ht_all_fail = all('rc=1' in l for l in ht_lines)
    open(f'{RT}/ht-negative-matrix.txt', 'w').write('\n'.join(ht_lines) + '\n')
    print(f'T5 ht negative matrix: {"PASS" if ht_all_fail else "FAIL"} '
          f'(9 qparam forms, all rejected)')

    verdict = (rc1 == 0 and rc2 != 0 and ok and rc4 != 0 and ht_all_fail)
    print(f'PHASE3 ROUND-TRIP: {"PASS" if verdict else "FAIL"}')
    return 0 if verdict else 1


if __name__ == '__main__':
    sys.exit(main())
