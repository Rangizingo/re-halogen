import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "ht_decode", Path(__file__).resolve().parent.parent / "scripts" / "ht_decode.py"
)
ht = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ht)


def _code(window: int) -> float:
    p = (window * ht.MULT) & 0xFFFFFFFF
    return (sum((p >> s) & 255 for s in (0, 8, 16, 24)) - 510.0) / 147.8


def _slow_decode_4bit(raw: np.ndarray, n: int, k: int) -> np.ndarray:
    """Per-weight 4-bit rule: lane t = 16*h + r holds word t; window = ({w[t-1], w[t]} >> 4*(7-c)) & 0xffff."""
    words = raw.view("<u4").reshape(-1, 32)
    out = np.empty((n, k), np.float32)
    for row in range(n):
        for col in range(k):
            tile = (row // 128) * (8 * k // 16) + (col // 16) * 8 + (row % 128) // 16
            t = 16 * ((col // 8) % 2) + row % 16
            pair = (int(words[tile, (t - 1) % 32]) << 32) | int(words[tile, t])
            out[row, col] = _code((pair >> (4 * (7 - col % 8))) & 0xFFFF)
    return out


def test_hadamard_is_orthonormal():
    h = ht.hadamard(128)
    assert np.allclose(h @ h.T, np.eye(128))


def test_fast_decoder_matches_per_weight_rule_4bit():
    n, k = 128, 32
    raw = np.random.default_rng(0).integers(0, 256, n * k // 2, dtype=np.uint8)
    assert np.allclose(ht.decode_codes(raw, n, k, 4), _slow_decode_4bit(raw, n, k))


def test_zero_stream_decodes_to_constant():
    n, k = 128, 16
    codes = ht.decode_codes(np.zeros(n * k * 3 // 8, np.uint8), n, k, 3)
    assert np.allclose(codes, -510.0 / 147.8)


def test_reconstruct_inverts_rotation():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((128, 256)).astype(np.float32)
    suh = rng.choice([-1.0, 1.0], 256)
    svh = rng.uniform(0.5, 2.0, 128) * rng.choice([-1.0, 1.0], 128)
    h = ht.hadamard(128)
    rot = (w / svh[:, None]) * suh[None, :]
    rot = np.einsum("ij,bjk->bik", h, (rot.reshape(128, -1, 128) @ h).reshape(-1, 128, 256)).reshape(128, 256)
    assert np.allclose(ht.reconstruct(rot, suh, svh), w, atol=1e-4)


@pytest.mark.skipif(not os.environ.get("HT_TESTDATA"), reason="set HT_TESTDATA to the fetch_ht_testdata.sh dir")
def test_real_tensor_matches_reference():
    d = os.environ["HT_TESTDATA"]
    f16 = lambda name: np.fromfile(f"{d}/{name}", "<f2").astype(np.float32)
    w = ht.reconstruct(
        ht.decode_codes(np.fromfile(f"{d}/k3_weight.ht", np.uint8), 512, 2560, 4), f16("k3_suh.f16"), f16("k3_svh.f16")
    )
    ref = ht.load_q4c(f"{d}/k3_weight_w4b.q4c", 512, 2560)
    assert np.corrcoef(w.ravel(), ref.ravel())[0, 1] > 0.99
