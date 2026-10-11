#!/usr/bin/env python3
"""Constant audit (PLAN p7-12c, ruling 4c): every numeric literal a vendor kernel uses must also appear in our rewrite's ISA.

A MATCH row cannot see a constant that differs below output rounding. This tool lists the 32-bit literal operands
of the vendor kernel (llvm-objdump listing in kernels-disasm-v2/) and of the rewrite (compiled for gfx1151 on Ai-core,
disassembled with the same llvm-objdump), and reports vendor values the rewrite lacks.

  python3 -I rewrite/harness/const_audit.py --instances all --out rewrite/const-audit.md --jobs 4

--instances: 'all' (function-map-kernel-stubs.tsv column 2) or a file with one mangled name per line (or a tsv, column 2).
Rerunning skips instances already in --out (AUDIT ERROR rows are retried).

Extracted: hex literals (0x...), inline float constants (0.5, -4.0, 0.15915494) as bit patterns, K of fmaak/madak/fmamk,
SOPK immediates. Ignored: decimal integers (inline ints, branch offsets), s_waitcnt/s_delay_alu operands, every SMEM/VMEM/DS
instruction (offsets), modifiers (offset:, row_shr:, op_sel:), registers, text after '//'. 64-bit values split into halves.
f32 inline constants become f32 bits, f16 ops f16 bits, f64 ops the high word. A vendor value whose sign-flipped twin is
in the rewrite is 'sign-flipped', not missing.
"""
import argparse
import concurrent.futures as cf
import datetime
import json
import mmap
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OBJDUMP = '/opt/homebrew/opt/llvm/bin/llvm-objdump'
HEADER = ('| instance | object | vendor literals | rewrite literals | missing (vendor-only FLOAT) | extra (rewrite-only) '
          '| int-only gaps (info) | verdict |\n'
          '| --- | --- | --- | --- | --- | --- | --- | --- |\n')
MAX_LIST = 40

# Mnemonic prefixes whose operands are offsets, counters or targets, never data constants.
SKIP_PREFIX = ('s_waitcnt', 's_wait', 's_delay_alu', 's_branch', 's_cbranch', 's_clause', 's_nop', 's_sendmsg', 's_endpgm',
               's_barrier', 's_setprio', 's_sleep', 's_trap', 's_code_end', 's_inst_prefetch', 's_load', 's_buffer_load',
               's_store', 's_buffer_store', 's_prefetch', 's_atc', 's_dcache', 's_getpc', 's_setpc', 's_swappc',
               's_call', 'global_', 'flat_', 'scratch_', 'buffer_', 'ds_', 'image_', 'tbuffer_', 'v_swaplane')

_SYM = re.compile(rb'^[0-9a-f]{16} <(.+)>:[ \t]*$', re.MULTILINE)
_HEX = re.compile(r'^0x[0-9a-fA-F]+$')
_FLT = re.compile(r'^-?\d+\.\d*(?:[eE][+-]?\d+)?$')
_MOD_BRACKET = re.compile(r'\b\w+:\[[^\]]*\]')
_MOD_PLAIN = re.compile(r'\b\w+:\S+')
_REG_RANGE = re.compile(r'\b(?:[svat]|ttmp)\[[^\]]*\]')
_WRAP = re.compile(r'\b(?:neg|abs|sext|lit)\(([^()]*)\)')


# ---- listing parsing -------------------------------------------------------

def split_kernels(text: str) -> dict[str, list[str]]:
    """Map symbol name to its listing lines for a small text (tests, rewrite objects)."""
    out: dict[str, list[str]] = {}
    cur: list[str] | None = None
    for line in text.splitlines():
        m = re.match(r'^[0-9a-f]{16} <(.+)>:\s*$', line)
        if m:
            cur = out.setdefault(m.group(1), [])
        elif cur is not None:
            cur.append(line)
    return out


class Listing:
    """Big vendor listing: indexes symbol offsets once, reads one kernel at a time."""

    def __init__(self, path: Path):
        self.path = path
        self._idx: dict[str, tuple[int, int]] | None = None
        self._lock = threading.Lock()

    def index(self) -> dict[str, tuple[int, int]]:
        with self._lock:
            if self._idx is None:
                idx: dict[str, tuple[int, int]] = {}
                with open(self.path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    ms = list(_SYM.finditer(mm))
                    for i, m in enumerate(ms):
                        end = ms[i + 1].start() if i + 1 < len(ms) else len(mm)
                        idx[m.group(1).decode()] = (m.end(), end)
                self._idx = idx
            return self._idx

    def kernel_lines(self, name: str) -> list[str]:
        idx = self.index()
        if name not in idx:
            raise KeyError(f'{name} not in {self.path.name}')
        s, e = idx[name]
        with open(self.path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return mm[s:e].decode().splitlines()


# ---- literal extraction ----------------------------------------------------

def _base_mnemonic(m: str) -> str:
    return re.sub(r'_(e32|e64|dpp|sdwa)$', '', m)


def _inline_float_bits(tok: str, mnem: str) -> int:
    v = float(tok)
    base = _base_mnemonic(mnem)
    if base.endswith('_f64'):
        return struct.unpack('<Q', struct.pack('<d', v))[0] >> 32
    if base.endswith('_f16') and not base.startswith('v_cvt'):
        return struct.unpack('<H', struct.pack('<e', v))[0]
    return struct.unpack('<I', struct.pack('<f', v))[0]


def _operand_tokens(ops: str) -> list[str]:
    ops = _MOD_BRACKET.sub(' ', ops)
    ops = _MOD_PLAIN.sub(' ', ops)
    ops = _REG_RANGE.sub('R', ops)
    for _ in range(3):
        ops = _WRAP.sub(r'\1', ops)
    ops = ops.replace('|', ' ')
    toks = []
    for t in re.split(r'[,\s]+', ops):
        t = t.strip()
        if not _FLT.match(t):
            t = t.lstrip('-')   # '-' on a hex literal is a source modifier; on a float it is the inline constant
        if t:
            toks.append(t)
    return toks


def _instr_literals(part: str) -> set[int]:
    part = part.strip()
    if not part or part.startswith(('.', '<')):
        return set()
    bits = part.split(None, 1)
    mnem = bits[0]
    if mnem.startswith(SKIP_PREFIX) or len(bits) < 2:
        return set()
    out: set[int] = set()
    for t in _operand_tokens(bits[1]):
        if _HEX.match(t):
            v = int(t, 16)
            if v > 0xffffffff:
                out.add((v >> 32) & 0xffffffff)
                out.add(v & 0xffffffff)
            else:
                out.add(v)
        elif _FLT.match(t):
            out.add(_inline_float_bits(t, mnem))
    return out


def extract_literals(lines: list[str]) -> set[int]:
    out: set[int] = set()
    for line in lines:
        code = line.split('//', 1)[0]
        for part in code.split('::'):
            out |= _instr_literals(part)
    return out


# ---- comparison and report -------------------------------------------------

def fmt_val(v: int) -> str:
    if v <= 0xffff:
        return f'0x{v:x}'
    f = struct.unpack('<f', struct.pack('<I', v))[0]
    if not (1e-30 < abs(f) < 1e30):
        return f'0x{v:08x}'
    for p in range(1, 10):
        s = f'{f:.{p}g}'
        if struct.unpack('<f', struct.pack('<f', float(s)))[0] == f:
            return f'0x{v:08x} ({s})'
    return f'0x{v:08x}'


def is_float(v: int) -> bool:
    """FLOAT: read as f32 the exponent field is 1..254, not an integer mask or power of two, not a small integer."""
    if v > 0xffffffff or v < 0x01000000:
        return False
    if not 1 <= (v >> 23) & 0xff <= 254:
        return False
    return (v & (v + 1)) != 0 and (v & (v - 1)) != 0


def _flip(v: int) -> int:
    return v ^ (0x8000 if v <= 0xffff else 0x80000000)


@dataclass
class Cmp:
    vendor: set[int] = field(default_factory=set)
    rewrite: set[int] = field(default_factory=set)
    missing: list[int] = field(default_factory=list)       # FLOAT only
    int_missing: list[int] = field(default_factory=list)   # INT, informational
    extra: list[int] = field(default_factory=list)
    flipped: list[int] = field(default_factory=list)


def compare_sets(vendor: set[int], rewrite: set[int]) -> Cmp:
    c = Cmp(set(vendor), set(rewrite))
    partners: set[int] = set()
    for v in sorted(vendor - rewrite):
        if _flip(v) in rewrite:
            c.flipped.append(v)
            partners.add(_flip(v))
        elif is_float(v):
            c.missing.append(v)
        else:
            c.int_missing.append(v)
    c.extra = sorted(rewrite - vendor - partners)
    return c


def _cap(vals: list[int]) -> str:
    s = ', '.join(fmt_val(v) for v in vals[:MAX_LIST])
    return s + (f', +{len(vals) - MAX_LIST} more' if len(vals) > MAX_LIST else '')


def verdict(c: Cmp) -> str:
    v = 'AUDIT GAP ' + _cap(c.missing) if c.missing else 'AUDIT OK'
    if c.flipped:
        v += ('; ' if c.missing else ' ') + f'(sign-flipped: {_cap(c.flipped)})'
    return v


def format_row(instance: str, obj: str, c: Cmp) -> str:
    return (f'| {instance} | {obj} | {len(c.vendor)} | {len(c.rewrite)} | {_cap(c.missing)} | {_cap(c.extra)} '
            f'| {_cap(c.int_missing)} | {verdict(c)} |')


def error_row(instance: str, obj: str, msg: str) -> str:
    msg = ' '.join(msg.split())[:300].replace('|', '/')
    return f'| {instance} | {obj} | - | - |  |  |  | AUDIT ERROR {msg} |'


def _rows(path: Path) -> list[list[str]]:
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith('| ') and not line.startswith(('| instance', '| ---')):
            rows.append([c.strip() for c in line.strip().strip('|').split('|')])
    return rows


def append_row(path: Path, row: str) -> None:
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, 'a') as f:
        if new:
            f.write(HEADER)
        f.write(row + '\n')


def done_instances(path: Path) -> set[str]:
    return {r[0] for r in _rows(path) if len(r) >= 8 and 'AUDIT ERROR' not in r[7]}


def compact(path: Path) -> None:
    """Keep the last row per instance (an error row is replaced by its retry)."""
    last: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith('| ') and not line.startswith(('| instance', '| ---')):
            last[line.strip().strip('|').split('|')[0].strip()] = line
    Path(path).write_text(HEADER + ''.join(v + '\n' for v in last.values()))


# ---- instance to spec and compile flags -----------------------------------

def spec_map(tables: list[Path], exists=None) -> dict[str, str]:
    """instance -> first spec named by a MATCH row (that `exists(name)` accepts, when given)."""
    m: dict[str, str] = {}
    for t in tables:
        for r in _rows(t):
            if len(r) >= 4 and r[3].startswith('MATCH') and r[0] not in m and (exists is None or exists(r[2].split()[0])):
                m[r[0]] = r[2].split()[0]
    return m


_LQ = re.compile(r'k_lq4wILi(\d+)ELi(\d+)ELi(\d+)ELb(\d)ELi(\d)ELi(\d+)E')
_LQ_PAIR = re.compile(r'k_lq4w_pairILi(\d+)ELi16ELi(\d+)E')
_LQ_GRP = re.compile(r'k_lq4w_grpILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E')


def extra_flags(mangled: str, spec: str) -> str:
    """HIPCC_COMPILE_FLAGS_APPEND the group's run script sets for this instance (rewrite/specs/lq4w_*run_all.sh)."""
    if (m := _LQ_PAIR.search(mangled)):
        return f'-DLQ_PAIR_N={m[1]} -DLQ_PAIR_F={m[2]}'
    if (m := _LQ_GRP.search(mangled)):
        nc, w, mb, f = m.groups()
        return f'-DLQ_GRP_N={nc} -DLQ_GRP_W={w} -DLQ_GRP_MB={mb} -DLQ_GRP_MS={16 if w == "16" else 8} -DLQ_GRP_F={f}'
    if (m := _LQ.search(mangled)):
        nc, w, mm, b, e, f = m.groups()
        return f'-DLQ_N={nc} -DLQ_W={w} -DLQ_M={mm} -DLQ_B={b} -DLQ_E={e} -DLQ_F={f}'
    return ''


# ---- compile on Ai-core ----------------------------------------------------

class Remote:
    """Disk work dir ~/auditwork/<run> on Ai-core; compile, copy the .co back, never touches anything else."""

    def __init__(self, repo: Path, host: str = 'ai-core'):
        self.repo, self.host = repo, host
        self.run = f'run{datetime.datetime.now(tz=datetime.timezone.utc):%Y%m%d%H%M%S}_{uuid.uuid4().hex[:6]}'
        self.dir = f'auditwork/{self.run}'
        self._ready = False
        self._lock = threading.Lock()

    def _ssh(self, cmd: str, timeout: int = 600) -> subprocess.CompletedProcess:
        return subprocess.run(['ssh', '-o', 'BatchMode=yes', self.host, cmd], capture_output=True, text=True,
                              timeout=timeout, check=False)

    def prepare(self) -> None:
        with self._lock:
            if self._ready:
                return
            r = self._ssh(f'mkdir -p ~/{self.dir}/out')
            if r.returncode:
                raise RuntimeError(f'ssh mkdir: {r.stderr.strip()}')
            r = subprocess.run(['rsync', '-a', '--exclude', '__pycache__', f'{self.repo}/rewrite/kernels/',
                                f'{self.host}:{self.dir}/kernels/'], capture_output=True, text=True, check=False)
            if r.returncode:
                raise RuntimeError(f'rsync: {r.stderr.strip()}')
            self._ready = True

    def compile(self, src_name: str, flags: str, tag: str, local_dir: Path) -> Path:
        self.prepare()
        env = f"HIPCC_COMPILE_FLAGS_APPEND='{flags}' " if flags else ''
        cmd = (f"cd ~/{self.dir}/kernels && {env}nice -n 10 hipcc --genco --no-gpu-bundle-output "
               f"--offload-arch=gfx1151 -O3 -o ../out/{tag}.co {src_name}")
        r = self._ssh(cmd)
        if r.returncode:
            raise RuntimeError(f'hipcc {src_name} [{flags}]: {r.stderr.strip()[-300:]}')
        dst = local_dir / f'{tag}.co'
        r = subprocess.run(['scp', '-q', f'{self.host}:{self.dir}/out/{tag}.co', str(dst)], capture_output=True, text=True, check=False)
        if r.returncode:
            raise RuntimeError(f'scp: {r.stderr.strip()}')
        return dst

    def cleanup(self) -> None:
        if self._ready and re.fullmatch(r'auditwork/run\d{14}_[0-9a-f]{6}', self.dir):
            self._ssh(f'rm -rf ~/{self.dir}')


def disassemble(co: Path) -> dict[str, list[str]]:
    r = subprocess.run([OBJDUMP, '-d', '--mcpu=gfx1151', str(co)], capture_output=True, text=True, check=True)
    return split_kernels(r.stdout)


# ---- driver ---------------------------------------------------------------

def load_instances(arg: str, repo: Path) -> list[str]:
    p = repo / 'analysis/function-map-kernel-stubs.tsv' if arg == 'all' else Path(arg)
    names = []
    for line in p.read_text().splitlines():
        cols = line.split('\t')
        n = cols[1] if (arg == 'all' or len(cols) > 1) else cols[0]
        if n and n != 'mangled_kernel' and not n.startswith('#'):
            names.append(n.strip())
    return names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--instances', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--repo', default=str(REPO), help='checkout holding kernels-disasm-v2, rewrite/tables, kernargs')
    ap.add_argument('--disasm-dir', help='default <repo>/kernels-disasm-v2')
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    out = Path(a.out)
    dis = Path(a.disasm_dir or repo / 'kernels-disasm-v2')
    kargs = json.loads((repo / 'rewrite/harness/kernargs.json').read_text())['kernels']
    specs = spec_map(sorted((repo / 'rewrite/tables').glob('*.md')), lambda sp: (repo / 'rewrite/specs' / sp).exists())
    done = done_instances(out)
    todo = [n for n in load_instances(a.instances, repo) if n not in done]
    print(f'{len(todo)} instances to audit', flush=True)

    listings: dict[str, Listing] = {}
    lock = threading.Lock()

    def write(row: str) -> None:
        with lock:
            append_row(out, row)

    jobs: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for n in todo:
        k = kargs.get(n)
        if not k:
            write(error_row(n, '?', 'not in kernargs.json'))
            continue
        obj = k['object']
        spec = specs.get(n)
        if not spec:
            write(error_row(n, obj, 'no MATCH row names a spec'))
            continue
        try:
            sp = json.loads((repo / 'rewrite/specs' / spec).read_text())['compare']
            src = (repo / 'rewrite/specs' / sp['rewrite_src']).resolve()
            if Path(sp['rewrite_src']).is_absolute():   # path from the Ai-core checkout: same file name in rewrite/kernels
                src = (repo / 'rewrite/kernels' / Path(sp['rewrite_src']).name).resolve()
        except (OSError, KeyError, ValueError) as e:
            write(error_row(n, obj, f'spec {spec}: {e}'))
            continue
        jobs.setdefault((str(src), extra_flags(n, spec)), []).append((n, obj, sp['rewrite_kernel']))

    remote = Remote(repo)
    local = Path(tempfile.mkdtemp(prefix='const_audit_'))

    def run_group(key: tuple[str, str], members: list[tuple[str, str, str]]) -> None:
        src, flags = key
        try:
            if Path(src).parent != (repo / 'rewrite/kernels').resolve():
                raise RuntimeError(f'rewrite_src outside rewrite/kernels: {src}')
            co = remote.compile(Path(src).name, flags, uuid.uuid4().hex[:10], local)
            syms = disassemble(co)
            co.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001 - reported per instance
            for n, obj, _ in members:
                write(error_row(n, obj, str(e)))
            return
        for n, obj, rk in members:
            try:
                if rk not in syms:
                    raise KeyError(f'rewrite kernel {rk} not in object (has {sorted(syms)[:4]})')
                with lock:
                    listing = listings.setdefault(obj, Listing(dis / f'{obj}.s'))
                vl = extract_literals(listing.kernel_lines(n))
                write(format_row(n, obj, compare_sets(vl, extract_literals(syms[rk]))))
            except Exception as e:  # noqa: BLE001
                write(error_row(n, obj, str(e)))

    try:
        with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            futs = [ex.submit(run_group, k, m) for k, m in jobs.items()]
            for i, f in enumerate(cf.as_completed(futs), 1):
                f.result()
                if i % 10 == 0:
                    print(f'{i}/{len(futs)} compile groups', flush=True)
    finally:
        remote.cleanup()
        shutil.rmtree(local, ignore_errors=True)
    if out.exists():
        compact(out)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
