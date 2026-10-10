#!/usr/bin/env python3
"""hgnpy: python reimplementation of the .hgn container parser/builder
(reverse-engineered from halogen-tools 0.17.3).

Layout (all little-endian):
  Header (104 B at file offset 0):
    +0  magic 'HGN1'
    +4  version u32
    +8  n_tensors u64
    +16 table_off u64      (== 104 in practice)
    +24 data_off u64       (start of payload area; (data_off-table_off)/160 >= n_tensors)
    +32 file_bytes u64
    +40 model_id[64]       (NUL-padded)
  Tensor table: n_tensors entries of 160 B at table_off:
    +0   name[96]          (inline, NUL-terminated, <=95 chars)
    +96  dtype u32
    +100 rank u32          (<=4)
    +104 dims u64[4]
    +136 payload_off u64   (ABSOLUTE file offset)
    +144 nbytes u64
    +152 xor32 u32         (XOR of LE u32 words of payload, zero-padded tail)
    +156 qparam u32        (quant block size / group param; 0 for plain dtypes)
  Payload area at data_off: tensor payloads, each 64 B aligned, in table order.
  Pads (must be zero): [4,table_off), [table_off+160*n, data_off), inter-payload
  gaps, [last payload end, file_bytes).
  header_sha256 = sha256(file[0:data_off])  (header + table region).
"""
import struct, hashlib

MAGIC = b'HGN1'
HDR_SIZE = 104
ENTRY_SIZE = 160
DTYPE_NAMES = {0:'bf16',1:'f32',2:'f16',3:'i32',4:'i64',5:'q4c',6:'fp8r',7:'q8g64',8:'i4l',
              10:'fp8g',11:'q8s32',12:'q6k',13:'iq4nl',14:'q4a',15:'q5a',16:'ht',
              20:'q8_0',21:'q5_0',22:'q5_1',23:'i4r',24:'q6g64'}

def xor32(data: bytes) -> int:
    n = (len(data) + 3) & ~3
    d = data + b'\x00' * (n - len(data))
    acc = 0
    for i in range(0, n, 4):
        acc ^= struct.unpack_from('<I', d, i)[0]
    return acc & 0xFFFFFFFF

class Entry:
    __slots__ = ('name','dtype','qparam','dims','offset','nbytes','xor32',
                'payload','xor_calc','sha256')
    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

def parse(data: bytes):
    if len(data) < 4 or data[:4] != MAGIC:
        raise ValueError('not an .hgn (the first bytes are not HGN1)')
    if len(data) < HDR_SIZE:
        raise ValueError('shorter than a header')
    (ver,) = struct.unpack_from('<I', data, 4)
    (n_tensors,) = struct.unpack_from('<Q', data, 8)
    (table_off,) = struct.unpack_from('<Q', data, 16)
    (data_off,) = struct.unpack_from('<Q', data, 24)
    (file_bytes,) = struct.unpack_from('<Q', data, 32)
    model_id = data[40:104].split(b'\x00')[0].decode('utf8', 'replace')
    if (data_off - table_off) // 160 < n_tensors:
        raise ValueError('the table does not fit in the file')
    entries = []
    for i in range(n_tensors):
        b = table_off + i * ENTRY_SIZE
        name = data[b:b+96].split(b'\x00')[0].decode('utf8','replace')
        dtype = struct.unpack_from('<I', data, b+96)[0]
        rank  = struct.unpack_from('<I', data, b+100)[0]
        qparam = struct.unpack_from('<I', data, b+156)[0]
        dims = [struct.unpack_from('<Q', data, b+104+8*j)[0] for j in range(rank)]
        (off,) = struct.unpack_from('<Q', data, b+136)
        (nbytes,) = struct.unpack_from('<Q', data, b+144)
        (x,) = struct.unpack_from('<I', data, b+152)
        entries.append(Entry(name=name, dtype=dtype, qparam=qparam, dims=dims,
                             offset=off, nbytes=nbytes, xor32=x))
    for e in entries:
        e.payload = data[e.offset:e.offset+e.nbytes]
        e.xor_calc = xor32(e.payload)
        e.sha256 = hashlib.sha256(e.payload).hexdigest()
    hdr_sha = hashlib.sha256(data[:data_off]).hexdigest()
    pads_zero = True
    ranges = []
    if n_tensors:
        ranges.append((data_off, entries[0].offset))
        for i in range(n_tensors):
            start = entries[i].offset + entries[i].nbytes
            end = entries[i+1].offset if i+1 < n_tensors else file_bytes
            ranges.append((start, end))
    else:
        ranges.append((data_off, file_bytes))
    for a, b in ranges:
        if any(data[a:b]):
            pads_zero = False
    return dict(version=ver, table_off=table_off, data_off=data_off,
                file_bytes=file_bytes, n_tensors=n_tensors, model_id=model_id,
                entries=entries, header_sha256=hdr_sha, pads_zero=pads_zero)

def build(version=1, model_id='', entries=(), payload_align=64,
          tbl_pad=b'', data_pad=b'', file_bytes=None):
    """entries: list of dicts(name, dtype, qparam, dims, payload[, xor32])."""
    mid = model_id.encode()
    assert len(mid) <= 63
    table_off = HDR_SIZE
    data_off = (table_off + len(entries) * ENTRY_SIZE + len(tbl_pad) + 63) & ~63
    laid = []
    off = 0
    for e in entries:
        off = (off + payload_align - 1) // payload_align * payload_align
        laid.append(off)
        off += len(e['payload'])
    total = file_bytes if file_bytes is not None else data_off + off + len(data_pad)
    h = bytearray(total)
    h[0:4] = MAGIC
    struct.pack_into('<I', h, 4, version)
    struct.pack_into('<Q', h, 8, len(entries))
    struct.pack_into('<Q', h, 16, table_off)
    struct.pack_into('<Q', h, 24, data_off)
    struct.pack_into('<Q', h, 32, total)
    h[40:40+len(mid)] = mid
    for i, e in enumerate(entries):
        b = table_off + i * ENTRY_SIZE
        nbname = e['name'].encode()
        assert len(nbname) <= 95
        h[b:b+len(nbname)] = nbname
        struct.pack_into('<I', h, b+96, e['dtype'])
        struct.pack_into('<I', h, b+100, len(e['dims']))
        for j, d in enumerate(e['dims']):
            struct.pack_into('<Q', h, b+104+8*j, d)
        struct.pack_into('<Q', h, b+136, data_off + laid[i])
        struct.pack_into('<Q', h, b+144, len(e['payload']))
        struct.pack_into('<I', h, b+152, e.get('xor32', xor32(e['payload'])))
        struct.pack_into('<I', h, b+156, e.get('qparam', 0))
    for i, e in enumerate(entries):
        p = data_off + laid[i]
        h[p:p+len(e['payload'])] = e['payload']
    if data_pad:
        h[total-len(data_pad):] = data_pad
    return bytes(h)
