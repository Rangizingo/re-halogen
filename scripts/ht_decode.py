"""CPU decoder for halogen HT weights (dtype 16), 3 and 4 bit, checked against q4c references.

usage: python3 -I scripts/ht_decode.py <data_dir>
<data_dir> holds the test slices fetched by scripts/fetch_ht_testdata.sh. Prints Pearson scores of the
decoded HT tensors against the same tensors in the vendor's q4c (w4b) checkpoint.

Format (analysis/closure/ht-decode.md):
- bits = (qparam >> 1) & 0x7f. Tensor [N, K] is cut into 16x16 tiles of 8*bits u32 words, tile id
  (row//128) * (8*K//16) + (col//16) * 8 + (row%128)//16.
- A tile's words in reverse order form one little-endian bit string U of 256*bits bits. Weight j = 128h + 8r + c
  (tile row r, column 8h+c) reads the 16-bit window of U at bit bits*(255-j) mod 256*bits, wrapping.
- code = (bytesum((window * 0x83dcd12d) mod 2^32) - 510) / 147.8
- W = diag(svh) * Hout * Wq * Hin * diag(suh), H = block-diagonal orthonormal 128-point Sylvester Hadamard.
- gate_up experts walk tiles differently (decode_codes_gate_up) and share one input sign vector, mlp.experts.su.
"""
import sys

import numpy as np

MULT = 0x83dcd12d


def hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def tile_codes(raw: np.ndarray, n_tiles: int, bits: int) -> np.ndarray:
    """Codes per stream tile [n_tiles, 256], index j = 128h + 8r + c (tile row r, column 8h+c)."""
    nw, length = 8 * bits, 256 * bits
    words = raw[:n_tiles * nw * 4].view('<u4').reshape(n_tiles, nw)[:, ::-1]
    ub = np.unpackbits(np.ascontiguousarray(words).view(np.uint8), axis=1, bitorder='little')
    ub = np.concatenate([ub, ub[:, :16]], 1).astype(np.uint32)
    pos = (bits * (255 - np.arange(256))) % length
    val = (ub[:, pos[:, None] + np.arange(16)[None, :]] << np.arange(16, dtype=np.uint32)).sum(-1)
    p = (val.astype(np.uint64) * np.uint64(MULT)) & np.uint64(0xffffffff)
    s = (p & 255) + ((p >> 8) & 255) + ((p >> 16) & 255) + (p >> 24)
    return (s.astype(np.float32) - 510.0) / 147.8


def decode_codes(raw: np.ndarray, n: int, k: int, bits: int) -> np.ndarray:
    """Codes Wq [n, k] in the rotated basis (unit variance), dense linears and down_proj experts."""
    c = tile_codes(raw, n * k // 256, bits).reshape(n // 128, k // 16, 8, 2, 16, 8)
    return c.transpose(0, 2, 4, 1, 3, 5).reshape(n, k)


def decode_codes_gate_up(raw: np.ndarray, n: int, k: int, bits: int) -> np.ndarray:
    """gate_up experts [n = gate + up rows, k]: per block of 128 gate rows and the matching 128 up rows,
    per column tile, the 8 gate tile rows then the 8 up tile rows."""
    c = tile_codes(raw, n * k // 256, bits).reshape(n // 256, k // 16, 2, 8, 2, 16, 8)
    return c.transpose(2, 0, 3, 5, 1, 4, 6).reshape(n, k)


def reconstruct(wq: np.ndarray, suh: np.ndarray, svh: np.ndarray) -> np.ndarray:
    h = hadamard(128)
    n, k = wq.shape
    m = (wq.reshape(n, -1, 128) @ h).reshape(n, k)
    m = np.einsum('ij,bjk->bik', h, m.reshape(-1, 128, k)).reshape(n, k)
    return svh[:, None] * m * suh[None, :]


def load_q4c(path: str, n: int, k: int, e: int = 0, n_exp: int = 1, sc_per_row: int | None = None) -> np.ndarray:
    """q4c planar reference: 64 B header (16 f32 codebook), all nibbles (low first), then fp16 scales per 32 along K."""
    cb = np.fromfile(path, np.uint8, 64).view('<f4')
    q = np.fromfile(path, np.uint8, n * k // 2, offset=64 + e * n * k // 2)
    spr = sc_per_row or k // 32
    sc = np.fromfile(path, '<f2', n * spr, offset=64 + n_exp * n * k // 2 + e * n * spr * 2)
    sc = sc.astype(np.float32).reshape(n, spr)[:, :k // 32]
    return cb[np.stack([q & 15, q >> 4], -1).reshape(-1)].reshape(n, k) * np.repeat(sc, 32, 1)


def report(tag: str, w: np.ndarray, ref: np.ndarray) -> None:
    a = w - w.mean(1, keepdims=True)
    r = ref - ref.mean(1, keepdims=True)
    rc = (a * r).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(r, axis=1)
    overall = np.corrcoef(w.ravel(), ref.ravel())[0, 1]
    print(f'{tag} overall {overall:.4f} row min {rc.min():.4f} median {np.median(rc):.4f}')


def main(d: str) -> None:
    f16 = lambda name: np.fromfile(f'{d}/{name}', '<f2').astype(np.float32)
    u8 = lambda name: np.fromfile(f'{d}/{name}', np.uint8)
    w = reconstruct(decode_codes(u8('k3_weight.ht'), 512, 2560, 4), f16('k3_suh.f16'), f16('k3_svh.f16'))
    report('layers.3.self_attn.k_proj 4-bit [512,2560]', w, load_q4c(f'{d}/k3_weight_w4b.q4c', 512, 2560))
    w = reconstruct(decode_codes(u8('sh3_down.ht'), 2560, 640, 4), f16('sh3_down_suh.f16'), f16('sh3_down_svh.f16'))
    report('layers.3.mlp.shared_expert.down_proj 4-bit [2560,640]', w,
           load_q4c(f'{d}/sh3_down_w4b.q4c', 2560, 640, sc_per_row=24))
    raw = u8('e1_down_ht_x0x1.ht')
    su = f16('e1_down_suh_x0x1.f16').reshape(2, 640)
    sv = f16('e1_down_svh_x0x1.f16').reshape(2, 2560)
    per = 2560 * 640 * 3 // 8
    ref = lambda e: load_q4c(f'{d}/e1_down_w4b_full.q4c', 2560, 640, e=e, n_exp=512, sc_per_row=24)
    for e in (0, 1):
        w = reconstruct(decode_codes(raw[e * per:(e + 1) * per], 2560, 640, 3), su[e], sv[e])
        report(f'layers.1.mlp.experts.down_proj expert {e} 3-bit [2560,640]', w, ref(e))
    report('negative control: expert 1 HT vs expert 0 reference', w, ref(0))
    cb = np.fromfile(f'{d}/g0_gu_w4b_cb.bin', '<f4')
    q = u8('g0_gu_w4b_nib_x0.bin')
    sc = f16('g0_gu_w4b_sc_x0.f16').reshape(1280, 80)
    gu_ref = cb[np.stack([q & 15, q >> 4], -1).reshape(-1)].reshape(1280, 2560) * np.repeat(sc, 32, 1)
    w = reconstruct(decode_codes_gate_up(u8('g0_gu_x0.ht'), 1280, 2560, 4), f16('g0_su.f16'), f16('g0_gu_svh_x0.f16'))
    report('layers.0.mlp.experts.gate_up_proj expert 0 4-bit [1280,2560]', w, gu_ref)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
