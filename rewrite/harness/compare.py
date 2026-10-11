#!/usr/bin/env python3
"""Run a vendor kernel and its HIP rewrite on the same spec, compare outputs, log to match-table.md.

usage (on Ai-core, from the repo root or work/harness):
  python3 -I rewrite/harness/compare.py SPEC.json --kernel MANGLED --seeds 1,2,3 --sizes 4096,100003

SPEC.json is a template: any string value starting with "=" is an arithmetic expression over
N (size), SEED and ceil_div(a, b). The "compare" block names the rewrite:
  {"rewrite_src": "rewrite/kernels/scale.hip", "rewrite_kernel": "k_scale_silu",
   "outputs": [{"buffer": "y", "dtype": "bf16"}], "tol": {"max_abs": 1e-3, "max_ulp": 2}}
No "tol" (or an empty one) means bitwise equality. Exit 0 if every case MATCHes, 1 otherwise.
"""
import argparse
import ast
import datetime
import importlib.util
import json
import contextlib
import fcntl
import operator
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLOAT_FMT = {'f32': ('f', 4), 'f64': ('d', 8), 'f16': ('e', 2)}
INT_FMT = {'i8': ('b', 1), 'u8': ('B', 1), 'i16': ('h', 2), 'u16': ('H', 2),
           'i32': ('i', 4), 'u32': ('I', 4), 'i64': ('q', 8), 'u64': ('Q', 8)}
TABLE_HEADER = ('| instance | family | test spec | result | tolerance | date |\n'
                '| --- | --- | --- | --- | --- | --- |\n')

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow}


def _ceil_div(a, b):
    return -(-a // b)


def eval_expr(text, env):
    """Evaluate +,-,*,/,//,%,** over numbers, env names and ceil_div(). Anything else raises ValueError."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name) and n.id in env:
            return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            return _BIN[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'ceil_div' and len(n.args) == 2:
            return _ceil_div(ev(n.args[0]), ev(n.args[1]))
        raise ValueError(f'unsupported expression: {text!r}')
    return ev(ast.parse(text, mode='eval'))


def resolve(node, env):
    """Walk a JSON value; strings starting with '=' are evaluated."""
    if isinstance(node, str) and node.startswith('='):
        v = eval_expr(node[1:], env)
        return int(v) if isinstance(v, float) and v.is_integer() and '.' not in node else v
    if isinstance(node, list):
        return [resolve(x, env) for x in node]
    if isinstance(node, dict):
        return {k: resolve(v, env) for k, v in node.items()}
    return node


def explicit_args(args):
    """[(offset, size, kind)] for the non-hidden kernel arguments."""
    return [(o, s, k) for o, s, k in args if not k.startswith('hidden_')]


def build_run_spec(template, kernel_name, layout, env):
    """Resolve a template for one kernel layout ({kernarg_size, args}); drops the compare block."""
    spec = resolve({k: v for k, v in template.items() if k != 'compare'}, env)
    spec['kernel'] = kernel_name
    spec['kernarg_size'] = layout['kernarg_size']
    spec['hidden'] = {k[len('hidden_'):]: o for o, _, k in layout['args'] if k.startswith('hidden_')}
    return spec


def _decode(raw, dtype):
    if dtype == 'bf16':
        vals = struct.unpack(f'<{len(raw) // 2}H', raw)
        return [struct.unpack('<f', struct.pack('<I', v << 16))[0] for v in vals]
    if dtype in FLOAT_FMT:
        c, w = FLOAT_FMT[dtype]
        return list(struct.unpack(f'<{len(raw) // w}{c}', raw))
    c, w = INT_FMT[dtype]
    return list(struct.unpack(f'<{len(raw) // w}{c}', raw))


def _bits(raw, dtype):
    w = 2 if dtype in ('bf16', 'f16') else (FLOAT_FMT.get(dtype) or INT_FMT[dtype])[1]
    c = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}[w]
    return list(struct.unpack(f'<{len(raw) // w}{c}', raw)), w * 8


def _ordered(b, nbits):
    """Map float bits to an integer that grows with the float value (+0 and -0 both map to 0)."""
    sign = 1 << (nbits - 1)
    return -(b & (sign - 1)) if b & sign else b


def _is_nan(b, dtype):
    if dtype == 'f32':
        return (b & 0x7fffffff) > 0x7f800000
    if dtype == 'bf16':
        return (b & 0x7fff) > 0x7f80
    if dtype == 'f16':
        return (b & 0x7fff) > 0x7c00
    return (b & 0x7fffffffffffffff) > 0x7ff0000000000000


def compare_bytes(a, b, dtype, max_abs=None, max_ulp=None):
    """Compare two raw buffers. Bitwise unless max_abs or max_ulp is given.

    Returns dict: match, n, n_diff (elements whose bits differ), max_abs, max_ulp, first_diff.
    Bitwise mode compares raw bits, NaN payloads included. With a tolerance, NaN vs NaN counts as
    equal and NaN vs number is an infinite difference. Two empty buffers never match.
    """
    if not a and not b:
        return {'match': False, 'n': 0, 'n_diff': 0, 'max_abs': 0.0, 'max_ulp': 0, 'first_diff': None,
                'note': 'empty'}
    if len(a) != len(b):
        return {'match': False, 'n': 0, 'n_diff': -1, 'max_abs': float('inf'), 'max_ulp': float('inf'),
                'first_diff': None, 'note': f'length {len(a)} vs {len(b)}'}
    ba, nbits = _bits(a, dtype)
    bb, _ = _bits(b, dtype)
    isfloat = dtype in FLOAT_FMT or dtype == 'bf16'
    va = _decode(a, dtype)
    vb = _decode(b, dtype)
    n_diff, worst_abs, worst_ulp, first = 0, 0.0, 0, None
    for i, (x, y) in enumerate(zip(ba, bb)):
        if x == y:
            continue
        if isfloat and (max_abs is not None or max_ulp is not None) and _is_nan(x, dtype) and _is_nan(y, dtype):
            continue
        n_diff += 1
        if first is None:
            first = i
        if isfloat:
            if _is_nan(x, dtype) or _is_nan(y, dtype):
                d, u = float('inf'), 1 << nbits
            else:
                fx, fy = va[i], vb[i]
                d = abs(fx - fy) if fx != fy else 0.0
                u = abs(_ordered(x, nbits) - _ordered(y, nbits))
        else:
            d, u = float(abs(va[i] - vb[i])), abs(va[i] - vb[i])
        worst_abs, worst_ulp = max(worst_abs, d), max(worst_ulp, u)
    if max_abs is None and max_ulp is None:
        ok = n_diff == 0
    else:
        ok = (max_abs is None or worst_abs <= max_abs) and (max_ulp is None or worst_ulp <= max_ulp)
    return {'match': ok, 'n': len(ba), 'n_diff': n_diff, 'max_abs': worst_abs, 'max_ulp': worst_ulp,
            'first_diff': first}


def untouched(out, pre):
    """True when a kernel left its output exactly as it was before the launch."""
    return out == pre


def tol_text(max_abs, max_ulp):
    parts = []
    if max_abs is not None:
        parts.append(f'abs<={max_abs:g}')
    if max_ulp is not None:
        parts.append(f'ulp<={max_ulp:g}')
    return ' '.join(parts) or 'bitwise'


def format_row(instance, family, test, result, tol, date):
    cells = [instance, family, test, result, tol, date]
    return '| ' + ' | '.join(str(c).replace('|', '/') for c in cells) + ' |\n'


def append_row(table_path, row):
    p = Path(table_path)
    if not p.exists() or p.stat().st_size == 0:
        p.write_text('# Kernel match table (U22, PLAN.md p7-12)\n\n' + TABLE_HEADER)
    with p.open('a') as f:
        f.write(row)


def result_text(r):
    word = 'MATCH' if r['match'] else 'MISMATCH'
    return (f"{word} n={r['n']} diff={r['n_diff']} max_abs={r['max_abs']:g} max_ulp={r['max_ulp']:g}"
            + (f" first_diff={r['first_diff']}" if r['first_diff'] is not None else ''))


def _load_gen_kernargs():
    s = importlib.util.spec_from_file_location('gen_kernargs', HERE / 'gen_kernargs.py')
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def compile_rewrite(src, out_co):
    subprocess.run(['hipcc', '--genco', '--no-gpu-bundle-output', '--offload-arch=gfx1151', '-O3', '-o', str(out_co), str(src)],
                   check=True)


def rewrite_layout(gk, co, name):
    """Kernarg layout of `name` in a compiled rewrite object."""
    notes = subprocess.run(['llvm-readelf', '--notes', str(co)], check=True, capture_output=True, text=True).stdout
    ks = gk.parse_notes(notes)
    if name not in ks:
        raise SystemExit(f'rewrite kernel {name} not found in {co}; has {sorted(ks)}')
    rec = ks[name]
    return {'kernarg_size': rec['kernarg_segment_size'], 'args': rec['args']}


def run_kernel(co, spec, outdir):
    # Always the run_kernel built beside this file (build.sh), never a caller-chosen program.
    outdir.mkdir(parents=True, exist_ok=True)
    sp = outdir / 'spec.json'
    sp.write_text(json.dumps(spec, indent=1))
    r = subprocess.run(['./run_kernel', '--co', str(co), '--spec', str(sp), '--out', str(outdir)],
                       cwd=HERE, capture_output=True, text=True, timeout=120, check=False)
    if r.returncode != 0:
        raise RuntimeError(f'run_kernel rc={r.returncode}: {r.stderr.strip()}')


def mem_available_gib(path='/proc/meminfo'):
    """MemAvailable in GiB, or None where /proc/meminfo does not exist."""
    try:
        for line in open(path):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1048576
    except OSError:
        return None
    return None


def service_healthy(url=None):
    """True when the live service answers /health with 200 (or CMP_HEALTH_URL is set to empty)."""
    url = os.environ.get('CMP_HEALTH_URL', 'http://127.0.0.1:8307/health') if url is None else url
    if not url:
        return True
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


@contextlib.contextmanager
def host_slot(lock_path, min_free_gib):
    """One compare run per host at a time, started only while the live service is healthy and
    min_free_gib of host memory is free (no run competes with a service start).

    Ai-core shares its memory with the live halogen-swift service, which refuses to start under
    16 GiB MemAvailable; parallel agent runs pushed it below that (p7-12, 2026-10-11).
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        while True:
            free = mem_available_gib()
            if free is None or (free >= min_free_gib and service_healthy()):
                break
            time.sleep(30)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('spec')
    ap.add_argument('--kernel', required=True, help='vendor mangled name')
    ap.add_argument('--seeds', default='1')
    ap.add_argument('--sizes', default='4096')
    ap.add_argument('--kernargs', default=str(HERE / 'kernargs.json'))
    ap.add_argument('--fatbin-dir', default=str(Path.home() / 're-halogen/work/fatbin'))
    ap.add_argument('--rewrite-src', help='override compare.rewrite_src (relative paths: from the spec file)')
    ap.add_argument('--table', default=str(HERE.parent / 'match-table.md'))
    ap.add_argument('--no-table', action='store_true')
    ap.add_argument('--workdir')
    a = ap.parse_args(argv)

    template = json.loads(Path(a.spec).read_text())
    cmp_cfg = template.get('compare') or {}
    kd = json.loads(Path(a.kernargs).read_text())['kernels']
    if a.kernel not in kd:
        raise SystemExit(f'{a.kernel} not in {a.kernargs}')
    vend = kd[a.kernel]
    gk = _load_gen_kernargs()
    # Own work dir on disk (CMP_TMP, default <repo>/work/cmp), removed after the run: on Ai-core /tmp is RAM-backed,
    # and leftover dirs there starved the live service of memory (p7-12, 2026-10-11).
    auto = a.workdir is None
    if auto:
        base = Path(os.environ.get('CMP_TMP') or HERE.parent.parent / 'work' / 'cmp')
        base.mkdir(parents=True, exist_ok=True)
    work = Path(a.workdir or tempfile.mkdtemp(prefix='cmp_', dir=base)).resolve()
    lock = HERE.parent.parent / 'work' / 'cmp' / '.host.lock'
    try:
        with host_slot(lock, float(os.environ.get('CMP_MIN_FREE_GIB', '8'))):
            return _run(a, template, cmp_cfg, vend, gk, work)
    finally:
        if auto:
            shutil.rmtree(work, ignore_errors=True)


def _run(a, template, cmp_cfg, vend, gk, work):
    work.mkdir(parents=True, exist_ok=True)
    src = Path(a.rewrite_src or cmp_cfg['rewrite_src'])
    if not src.is_absolute() and not src.exists():
        src = Path(a.spec).resolve().parent / src
    rco = work / 'rewrite.co'
    compile_rewrite(src, rco)
    rname = cmp_cfg['rewrite_kernel']
    rlay = rewrite_layout(gk, rco, rname)
    ve = [(o, s) for o, s, _ in explicit_args(vend['args'])]
    re_ = [(o, s) for o, s, _ in explicit_args(rlay['args'])]
    if ve != re_:
        raise SystemExit(f'explicit arg layout differs: vendor {ve} vs rewrite {re_}')
    tol = cmp_cfg.get('tol') or {}
    max_abs, max_ulp = tol.get('max_abs'), tol.get('max_ulp')
    vco = Path(a.fatbin_dir) / (vend['object'] + '.hsaco')
    today = datetime.datetime.now(tz=datetime.timezone.utc).date().isoformat()
    all_ok = True
    for size in [int(s) for s in a.sizes.split(',')]:
        for seed in [int(s) for s in a.seeds.split(',')]:
            env = {'N': size, 'SEED': seed}
            case = f'{Path(a.spec).name} seed={seed} N={size}'
            vdir, rdir = work / f'v_{seed}_{size}', work / f'r_{seed}_{size}'
            run_kernel(vco.resolve(), build_run_spec(template, a.kernel, vend, env), vdir)
            run_kernel(rco.resolve(), build_run_spec(template, rname, rlay, env), rdir)
            res = {'match': True, 'n': 0, 'n_diff': 0, 'max_abs': 0.0, 'max_ulp': 0, 'first_diff': None}
            for out in cmp_cfg['outputs']:
                name = out['buffer']
                vb, rb = (vdir / f'{name}.bin').read_bytes(), (rdir / f'{name}.bin').read_bytes()
                r = compare_bytes(vb, rb, out['dtype'], max_abs, max_ulp)
                # Agreeing on an output neither kernel wrote proves nothing.
                for side, d, b in (('vendor', vdir, vb), ('rewrite', rdir, rb)):
                    if untouched(b, (d / f'{name}.pre.bin').read_bytes()):
                        r['match'] = False
                        print(f'UNTOUCHED {side} output {name} in {d.name}')
                res['match'] &= r['match']
                res['n'] += r['n']
                res['n_diff'] += max(r['n_diff'], 0)
                res['max_abs'] = max(res['max_abs'], r['max_abs'])
                res['max_ulp'] = max(res['max_ulp'], r['max_ulp'])
                if res['first_diff'] is None:
                    res['first_diff'] = r['first_diff']
            all_ok &= res['match']
            text = result_text(res)
            print(f'{text.split()[0]} {a.kernel} {case} {text.split(" ", 1)[1]} tol={tol_text(max_abs, max_ulp)}')
            if not a.no_table:
                append_row(a.table, format_row(a.kernel, vend['family'], case, text,
                                               tol_text(max_abs, max_ulp), today))
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
