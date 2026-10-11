#!/usr/bin/env python3
"""Constant audit (PLAN p7-12c, ruling 4c): every numeric literal a vendor kernel uses must also appear in our rewrite's ISA.

A MATCH row cannot see a constant that differs below output rounding. This tool lists the 32-bit literal operands
of the vendor kernel (llvm-objdump listing in kernels-disasm-v2/) and of the rewrite (compiled for gfx1151 on Ai-core,
disassembled with the same llvm-objdump), and reports vendor values the rewrite lacks.

  python3 -I rewrite/harness/const_audit.py --instances all --out rewrite/const-audit.md --jobs 4

--instances: 'all' (function-map-kernel-stubs.tsv column 2) or a file with one mangled name per line (or a tsv, column 2).
Rerunning skips instances already in --out (AUDIT ERROR rows are retried).

Extracted: hex literals (0x...), inline float constants (0.5, -4.0, 0.15915494) as bit patterns, K of fmaak/madak/fmamk,
SOPK immediates, decimal inline integers. Ignored: branch offsets, s_waitcnt/s_delay_alu operands, every SMEM/VMEM/DS
instruction (offsets), modifiers (offset:, row_shr:, op_sel:), registers, text after '//'. 64-bit values split into halves.
f32 inline constants become f32 bits, f16 ops f16 bits, f64 ops the high word. A vendor value whose sign-flipped twin is
in the rewrite is 'sign-flipped', not missing.
Literals are collected over the kernel plus every device function it calls (pc-relative s_getpc/s_swappc calls), plus the
rewrite object's .rodata words. Integer-role operands (masks, v_perm/v_permlane selectors, mul_hi magic, compiler udiv/logf
expansions) are integers, accounted for by the integer rule (see the block above inline_ints): complement, bound +-1, or
data-path role; a bound or an unread literal with no twin is 'AUDIT GAP int'. rewrite/const-audit-exceptions.tsv
(substring, hex, reason) turns a recorded explanation of a float or integer into 'AUDIT OK (exception: ...)'.
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
          '| int explained by data-path role | verdict |\n'
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
        self.addr: dict[int, str] = {}
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
                        self.addr[int(mm[m.start():m.start() + 16], 16)] = m.group(1).decode()
                self._idx = idx
            return self._idx

    def addr_map(self) -> dict[int, str]:
        self.index()
        return self.addr

    def kernel_lines(self, name: str) -> list[str]:
        idx = self.index()
        if name not in idx:
            raise KeyError(f'{name} not in {self.path.name}')
        s, e = idx[name]
        with open(self.path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return mm[s:e].decode().splitlines()



# ---- device-function calls -------------------------------------------------

_ADDR = re.compile(r'//\s*([0-9A-Fa-f]{12}):')
_LASTOP = re.compile(r'(?:lit\()?(-?(?:0x[0-9a-fA-F]+|\d+))\)?\s*$')


def call_targets(lines: list[str]) -> set[int]:
    """Absolute addresses of pc-relative calls: s_getpc_b64; s_add_u32 sN, sN, lo; s_addc_u32 sM, sM, hi; s_swappc_b64."""
    out: set[int] = set()
    for i, line in enumerate(lines):
        if 's_getpc_b64' not in line.split('//', 1)[0]:
            continue
        m = _ADDR.search(line)
        if not m or i + 2 >= len(lines):
            continue
        lo = _LASTOP.search(lines[i + 1].split('//', 1)[0].rstrip())
        hi = _LASTOP.search(lines[i + 2].split('//', 1)[0].rstrip())
        if not (lo and hi and 's_add_u32' in lines[i + 1] and 's_addc_u32' in lines[i + 2]):
            continue
        off = ((int(hi.group(1), 0) & 0xffffffff) << 32) | (int(lo.group(1), 0) & 0xffffffff)
        if off >= 1 << 63:
            off -= 1 << 64
        out.add(int(m.group(1), 16) + 4 + off)
    return out


def with_callees(name: str, get_lines, addr_to_name: dict[int, str]) -> list[str]:
    """Lines of `name` plus those of every device function it calls, transitively."""
    seen: set[str] = set()
    todo = [name]
    lines: list[str] = []
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.add(n)
        ls = get_lines(n)
        lines += ls
        todo += [addr_to_name[a] for a in call_targets(ls) if a in addr_to_name]
    return lines


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


# Operand roles that carry integers whatever their bit pattern looks like as f32: any instruction whose source type
# (last mnemonic token) is an integer type (masks, shifts, integer add/mul, v_perm and v_permlane selectors, mul_hi
# magic multipliers, v_cvt_f32_u32 sources), except plain data movers, which carry either kind.
_INT_TYPES = {'b32', 'b64', 'b16', 'b8', 'u32', 'i32', 'u16', 'i16', 'u64', 'i64', 'u8', 'i8', 'u24', 'i24'}
_PASSTHROUGH = re.compile(r'_(?:mov|cndmask|cselect|readlane|readfirstlane|writelane|movk|movrel\w*)_|^s_(?:mov|movk|cselect)')
_INT_NAMES = re.compile(r'^v_permlane\w*$')


def is_int_role_mnemonic(mnem: str) -> bool:
    base = _base_mnemonic(mnem)
    if _INT_NAMES.match(base):
        return True
    if _PASSTHROUGH.search(base + '_') or base.endswith('_mov_b32') or base.endswith('_mov_b64'):
        return False
    return base.rsplit('_', 1)[-1] in _INT_TYPES


# Compiler expansions (integer divide, logf denormal prescale). Compiler 32-bit unsigned divide: v_rcp_iflag_f32, * 0x4f7ffffe, v_cvt_u32_f32. 64-bit divide: 2^32, 2^-32, 2^64-2^40, -2^32.
_UDIV32 = {0x4f7ffffe}
_UDIV64 = {0x4f800000, 0x2f800000, 0x5f7ffffc, 0xcf800000}


def _parse(part: str) -> tuple[str, list[str]] | None:
    bits = part.strip().split(None, 1)
    if len(bits) < 2 or bits[0].startswith(('.', '<')):
        return None
    ops = [o.strip() for o in re.split(r',(?![^\[]*\])', _MOD_BRACKET.sub(' ', bits[1]))]
    return _base_mnemonic(bits[0]), ops


def _regs(op: str) -> set[str]:
    out = set()
    for m in re.finditer(r'\b([sv])(\d+)\b|\b([sv])\[(\d+):(\d+)\]', op):
        if m.group(1):
            out.add(m.group(1) + m.group(2))
        else:
            out.update(f'{m.group(3)}{i}' for i in range(int(m.group(4)), int(m.group(5)) + 1))
    return out


_NO_DEST = re.compile(r'^(?:s_cmp|s_bitcmp|buffer_|global_store|flat_store|scratch_store|ds_store|ds_write)')
_REG_WINDOW = 30000


def _sources(m: str, ops: list[str]) -> list[str]:
    """Operands an instruction reads: all but the destination; a second destination (carry out of v_add_co_u32,
    v_div_scale_f32, v_mad_u64_u32) is not a read either."""
    if _NO_DEST.match(m):
        return ops
    srcs = ops[1:]
    if len(ops) > 2 and re.match(r'^v_(?:add|sub|subrev)_co_u32|^v_div_scale_|^v_mad_[iu]64_[iu]32', m):
        srcs = ops[2:]
    return srcs


def _readers(parsed, start: int, dest: set[str]) -> list[str]:
    """Mnemonics of the instructions that read `dest` after instruction `start`, following movs and cndmasks into
    their destinations; each register's scan stops at its next plain write."""
    out: list[str] = []
    seen: set[tuple[int, str]] = set()
    work = [(start, frozenset(dest))]
    while work:
        at, regs = work.pop()
        live = set(regs)
        for j in range(at + 1, min(len(parsed), at + 1 + _REG_WINDOW)):
            nxt = parsed[j]
            if not nxt or not live:
                continue
            (m2, o2), _ = nxt
            srcs = _sources(m2, o2)
            reads = set().union(*(_regs(o) for o in srcs)) if srcs else set()
            hit = live & reads
            if hit:
                if m2.endswith(('_mov_b32', '_cndmask_b32')) or _PASSTHROUGH.search(m2 + '_'):
                    key = (j, ','.join(sorted(hit)))
                    if key not in seen and o2 and not _NO_DEST.match(m2):
                        seen.add(key)
                        work.append((j, frozenset(_regs(o2[0]))))
                else:
                    out.append(m2)
            if not _NO_DEST.match(m2) and o2 and not hit:
                live -= _regs(o2[0])
            if len(o2) > len(srcs) + 1 and not _NO_DEST.match(m2):
                live -= set().union(*(_regs(o) for o in o2[1:len(o2) - len(srcs)]))
    return out


def _consumers(parsed, start: int, dest: set[str]) -> tuple[int, int]:
    """Count integer-role and other readers of `dest` after instruction `start` (see _readers)."""
    rs = _readers(parsed, start, dest)
    ints = sum(1 for m in rs if is_int_role_mnemonic(m))
    return ints, len(rs) - ints


def extract_int_role(lines: list[str]) -> set[int]:
    """Literals that only ever appear as integer-role operands (masks, v_perm/v_permlane selectors, mul_hi magic
    multipliers) or as constants of the compiler's integer-divide expansion; reported under int-only gaps, not as
    float data. A literal moved into a register counts by the instructions that read that register before its next
    write (straight-line scan, through movs and cndmasks): integer readers only means integer."""
    parsed: list[tuple[str, list[str]] | None] = []
    for line in lines:
        code = line.split('//', 1)[0]
        for part in code.split('::'):
            parsed.append(_parse(part))
            parsed[-1] = (parsed[-1], part) if parsed[-1] else None   # keep the text for literal extraction
    int_vals: set[int] = set()
    other: set[int] = set()
    seen_vals: set[int] = set()
    text = '\n'.join(lines)
    udiv: set[int] = set()
    if 'v_rcp_iflag_f32' in text:
        udiv |= _UDIV32
    if 'v_rcp_f32' in text and '0x5f7ffffc' in text:
        udiv |= _UDIV64
    if 'v_log_f32' in text and re.search(r'v_cmp_\w+_f32\w*\s+\w+,\s*0x800000,', text):
        udiv.add(0x4f800000)     # logf: inputs below 2^-126 are scaled by 2^32 first (and 32*ln2 subtracted)
    for i, item in enumerate(parsed):
        if not item:
            continue
        (mnem, ops), part = item
        vals = _instr_literals(part)
        if not vals:
            continue
        seen_vals |= vals
        if mnem.endswith(('_mov_b32', '_cndmask_b32')) or mnem == 's_movk_i32':
            ints, floats = _consumers(parsed, i, _regs(ops[0]))
            # byte-selector shape (every byte 0..7) with an integer reader tolerates register-reuse noise in `floats`
            if ints and (not floats or all(v & 0xf8f8f8f8 == 0 and v > 0xffff for v in vals)):
                int_vals.update(vals)
            elif floats:
                other.update(vals)       # a mov nothing reads (overwritten at once) says nothing about the role
        else:
            (int_vals if is_int_role_mnemonic(mnem) else other).update(vals)
    return (int_vals - other) | (udiv & seen_vals)


def extract_literals(lines: list[str]) -> set[int]:
    out: set[int] = set()
    for line in lines:
        code = line.split('//', 1)[0]
        for part in code.split('::'):
            out |= _instr_literals(part)
    return out


# Integer literals (ruling 4c, every literal). A vendor integer the rewrite does not carry is accounted for in one of
# three ways: the rewrite has its two's complement or bit complement (add vs sub, and vs andn2); it is a bound (read by a
# compare, min, max or med3, directly, through movs, or through up to two add/sub/shift steps) and the rewrite has it
# plus or minus one (x > 127 vs x >= 128); or it is a data-path operand (address offset, mask, shift, stride, selector,
# multiplier) whose wrong value would change the bits of every lane that executes it, so the instance's bitwise MATCH
# rows cover it. A bound or a literal nothing reads gets no role explanation: it fails unless the exceptions file
# records why it differs. Memory-instruction offset fields and counters are encoding, not operands, on the vendor side;
# on the rewrite side they count as places a vendor value can appear.

_NOISE = ('s_waitcnt', 's_wait', 's_delay_alu', 's_branch', 's_cbranch', 's_clause', 's_nop', 's_sendmsg', 's_endpgm',
          's_setprio', 's_sleep', 's_trap', 's_code_end', 's_inst_prefetch', 's_set_inst_prefetch', 's_getpc', 's_setpc',
          's_swappc', 's_call', 's_barrier')
_MEM = ('s_load', 's_buffer_load', 's_store', 's_buffer_store', 's_prefetch', 's_atc', 's_dcache', 'global_', 'flat_',
        'scratch_', 'buffer_', 'ds_', 'image_', 'tbuffer_')
_INT_TOK = re.compile(r'(?<![\w.:])-?\d+(?![\w.])')
_OFFSET = re.compile(r'\boffset(\d?):(-?(?:0x[0-9a-fA-F]+|\d+))')
_RANGE = re.compile(r'cmp|_min_|_max_|_med3_|_minmax|_maxmin')
_BOUND_STEP = re.compile(r'^[vs]_(?:add|sub|subrev|lshl|lshr|ashr)\w*')


def _code_parts(lines: list[str]):
    for line in lines:
        for part in line.split('//', 1)[0].split('::'):
            bits = part.strip().split(None, 1)
            if len(bits) == 2 and not bits[0].startswith(('.', '<')):
                yield bits[0], bits[1]


def inline_ints(lines: list[str]) -> set[int]:
    """Vendor decimal integer operands (inline constants -16..64), outside memory instructions and counters."""
    out: set[int] = set()
    for mnem, ops in _code_parts(lines):
        if mnem.startswith(_NOISE + _MEM):
            continue
        for t in _INT_TOK.findall(_REG_RANGE.sub('R', _MOD_PLAIN.sub(' ', _MOD_BRACKET.sub(' ', ops)))):
            out.add(int(t) & 0xffffffff)
    return out


def int_present(lines: list[str]) -> set[int]:
    """Every integer the rewrite ISA carries: hex and decimal operands, and memory offset fields (offset0/offset1 also
    scaled by the element sizes 4 and 8 and the stride64 forms 256 and 512)."""
    out: set[int] = set()
    for mnem, ops in _code_parts(lines):
        if mnem.startswith(_NOISE):
            continue
        for m in _OFFSET.finditer(ops):
            v = int(m.group(2), 0)
            out |= {(v * k) & 0xffffffff for k in ((1, 4, 8, 256, 512) if m.group(1) else (1,))}
        for t in _operand_tokens(_REG_RANGE.sub('R', ops)):
            if _HEX.match(t) or re.fullmatch(r'\d+', t):
                v = int(t, 0)
                out |= {v >> 32, v & 0xffffffff} if v > 0xffffffff else {v}
        for t in _INT_TOK.findall(_MOD_PLAIN.sub(' ', _MOD_BRACKET.sub(' ', _REG_RANGE.sub('R', ops)))):
            out.add(int(t) & 0xffffffff)
    return out


def _parsed(lines: list[str]):
    out = []
    for line in lines:
        for part in line.split('//', 1)[0].split('::'):
            p = _parse(part)
            out.append((p, part) if p else None)
    return out


def _int_vals(part: str, mnem: str) -> set[int]:
    vals = _instr_literals(part)
    if not mnem.startswith(_NOISE + _MEM):
        vals |= inline_ints([part])
    return vals


def _roles(lines: list[str], only: set[int] | None = None) -> tuple[set[int], set[int]]:
    """(values with any reader or a direct non-mover use, values that reach a bound); `only` limits the trace to
    those values (the audit traces just the vendor integers the rewrite lacks)."""
    parsed = _parsed(lines)
    roled: set[int] = set()
    bound: set[int] = set()

    def reaches_bound(i: int, dest: set[str], depth: int) -> bool:
        for j, m in _reader_sites(parsed, i, dest):
            if _RANGE.search(m + '_'):
                return True
            if depth < 2 and _BOUND_STEP.match(m) and reaches_bound(j, _regs(parsed[j][0][1][0]), depth + 1):
                return True
        return False

    for i, item in enumerate(parsed):
        if not item:
            continue
        (mnem, ops), part = item
        vals = _int_vals(part, mnem)
        if only is not None:
            vals &= only
        if not vals:
            continue
        mover = mnem.endswith(('_mov_b32', '_cndmask_b32', '_mov_b64')) or _PASSTHROUGH.search(mnem + '_')
        if mover:
            if ops and _readers(parsed, i, _regs(ops[0])):
                roled |= vals
                if reaches_bound(i, _regs(ops[0]), 0):
                    bound |= vals
            continue
        roled |= vals
        if _RANGE.search(mnem + '_') or (ops and not _NO_DEST.match(mnem) and _BOUND_STEP.match(mnem)
                                          and reaches_bound(i, _regs(ops[0]), 1)):
            bound |= vals
    return roled, bound


def _reader_sites(parsed, start: int, dest: set[str]) -> list[tuple[int, str]]:
    """(index, mnemonic) of the non-mover readers of `dest` after `start`, through movs (as _readers)."""
    out: list[tuple[int, str]] = []
    work = [(start, frozenset(dest))]
    seen: set[int] = set()
    while work:
        at, regs = work.pop()
        live = set(regs)
        for j in range(at + 1, min(len(parsed), at + 1 + _REG_WINDOW)):
            nxt = parsed[j]
            if not nxt or not live:
                continue
            (m2, o2), _ = nxt
            srcs = _sources(m2, o2)
            reads = set().union(*(_regs(o) for o in srcs)) if srcs else set()
            hit = live & reads
            if hit:
                if m2.endswith(('_mov_b32', '_cndmask_b32')) or _PASSTHROUGH.search(m2 + '_'):
                    if j not in seen and o2 and not _NO_DEST.match(m2):
                        seen.add(j)
                        work.append((j, frozenset(_regs(o2[0]))))
                elif o2:
                    out.append((j, m2))
            if not _NO_DEST.match(m2) and o2 and not hit:
                live -= _regs(o2[0])
    return out


def range_role(lines: list[str]) -> set[int]:
    return _roles(lines)[1]


def roled(lines: list[str]) -> set[int]:
    return _roles(lines)[0]


def audit(vlines: list[str], rlines: list[str], rdata: set[int] = frozenset(), excused: dict[int, str] | None = None) -> 'Cmp':
    """Vendor literals (hex, inline float, inline int) against everything the rewrite carries."""
    vendor = extract_literals(vlines) | inline_ints(vlines)
    rewrite = extract_literals(rlines) | int_present(rlines) | set(rdata)
    int_role = extract_int_role(vlines)
    todo = {v for v in vendor - rewrite if not (is_float(v) and v not in int_role)}
    has_reader, bound = _roles(vlines, todo)
    return compare_sets(vendor, rewrite, int_role, excused, range_vals=bound, roled=has_reader | (vendor - todo))


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
    int_missing: list[int] = field(default_factory=list)   # INT, explained by data-path role
    extra: list[int] = field(default_factory=list)
    flipped: list[int] = field(default_factory=list)
    excused: dict[int, str] = field(default_factory=dict)  # vendor-only value with a recorded explanation
    int_gap: list[int] = field(default_factory=list)       # INT bound or unread literal with no twin
    bound: set[int] = field(default_factory=set)


def compare_sets(vendor: set[int], rewrite: set[int], int_role: set[int] = frozenset(),
                 excused: dict[int, str] | None = None, range_vals: set[int] = frozenset(),
                 roled: set[int] | None = None) -> Cmp:
    """roled None means every vendor value has a reader (callers without role data)."""
    c = Cmp(set(vendor), set(rewrite), bound=set(range_vals))
    partners: set[int] = set()
    excused = excused or {}
    for v in sorted(vendor - rewrite):
        is_f = is_float(v) and v not in int_role
        strict = not is_f and (v in range_vals or (roled is not None and v not in roled))
        if not strict and _flip(v) in rewrite:
            c.flipped.append(v)
            partners.add(_flip(v))
        elif is_f:
            if v in excused:
                c.excused[v] = excused[v]
            else:
                c.missing.append(v)
        elif (~v) & 0xffffffff in rewrite or (-v) & 0xffffffff in rewrite:
            continue
        elif v in range_vals and ((v + 1) & 0xffffffff in rewrite or (v - 1) & 0xffffffff in rewrite):
            continue
        elif strict:
            if v in excused:
                c.excused[v] = excused[v]
            else:
                c.int_gap.append(v)
        else:
            c.int_missing.append(v)
    c.extra = sorted(rewrite - vendor - partners)
    return c


def _cap(vals: list[int]) -> str:
    s = ', '.join(fmt_val(v) for v in vals[:MAX_LIST])
    return s + (f', +{len(vals) - MAX_LIST} more' if len(vals) > MAX_LIST else '')


def verdict(c: Cmp) -> str:
    gaps = [fmt_val(v) for v in c.missing[:MAX_LIST]]
    gaps += [f'int {fmt_val(v)} ({"bound" if v in c.bound else "no reader"})' for v in c.int_gap[:MAX_LIST]]
    if len(c.missing) + len(c.int_gap) > len(gaps):
        gaps.append(f'+{len(c.missing) + len(c.int_gap) - len(gaps)} more')
    bad = bool(c.missing or c.int_gap)
    v = 'AUDIT GAP ' + ', '.join(gaps) if bad else 'AUDIT OK'
    if c.flipped:
        v += ('; ' if bad else ' ') + f'(sign-flipped: {_cap(c.flipped)})'
    if c.excused:
        v += ('; ' if bad else ' ') + '(exception: ' + '; '.join(f'{fmt_val(k)} {r}' for k, r in c.excused.items()) + ')'
    return v


def load_exceptions(path: Path) -> list[tuple[str, int, str]]:
    """TSV: mangled-name substring, hex value, reason. '#' lines ignored. Also reads every *.tsv in the sibling
    directory const-audit-exceptions.d/ (one file per review group)."""
    path = Path(path)
    files = [path] + sorted((path.parent / 'const-audit-exceptions.d').glob('*.tsv'))
    out = []
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if line.strip() and not line.startswith('#'):
                sub, val, reason = (line.split('\t') + ['', ''])[:3]
                out.append((sub, int(val, 16), reason.strip().replace('|', '/')))
    return out


def excused_for(instance: str, exceptions: list[tuple[str, int, str]]) -> dict[int, str]:
    return {v: r for sub, v, r in exceptions if sub in instance}


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


def rodata_words(co: Path) -> set[int]:
    """32-bit little-endian words of .rodata (static __device__ tables compile to data, not immediates)."""
    r = subprocess.run([OBJDUMP, '-s', '-j', '.rodata', str(co)], capture_output=True, text=True, check=False)
    out: set[int] = set()
    for line in r.stdout.splitlines():
        m = re.match(r'^ [0-9a-f]+ ((?:[0-9a-f]{1,8} ?){1,4})', line)
        if m:
            for w in m.group(1).split():
                if len(w) == 8:
                    out.add(int.from_bytes(bytes.fromhex(w), 'little'))
    return out


def disassemble(co: Path) -> tuple[dict[str, list[str]], dict[int, str]]:
    r = subprocess.run([OBJDUMP, '-d', '--mcpu=gfx1151', str(co)], capture_output=True, text=True, check=True)
    addr = {int(m.group(1), 16): m.group(2) for m in re.finditer(r'^([0-9a-f]{16}) <(.+)>:\s*$', r.stdout, re.MULTILINE)}
    return split_kernels(r.stdout), addr


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
    ap.add_argument('--exceptions', help='default <repo>/rewrite/const-audit-exceptions.tsv')
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    out = Path(a.out)
    dis = Path(a.disasm_dir or repo / 'kernels-disasm-v2')
    kargs = json.loads((repo / 'rewrite/harness/kernargs.json').read_text())['kernels']
    specs = spec_map(sorted((repo / 'rewrite/tables').glob('*.md')) + [repo / 'rewrite/match-table.md'], lambda sp: (repo / 'rewrite/specs' / sp).exists())
    exceptions = load_exceptions(Path(a.exceptions or repo / 'rewrite/const-audit-exceptions.tsv'))
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
            if not src.exists():   # path written from the repo root (rewrite/kernels/scale.hip)
                src = (repo / sp['rewrite_src']).resolve()
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
            syms, raddr = disassemble(co)
            rdata = rodata_words(co)
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
                vlines = with_callees(n, listing.kernel_lines, listing.addr_map())
                write(format_row(n, obj, audit(vlines, with_callees(rk, syms.__getitem__, raddr), rdata,
                                               excused_for(n, exceptions))))
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
