import gzip
import hashlib
import io
import struct
from pathlib import Path

import numpy as np
import pytest

from mct_bench.assets import fetch_asset, preprocess, read_idx_gzip


def asset_spec(payload=b"public model"):
    return {"filename": "model.onnx", "url": "https://example.test/pinned/model.onnx", "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def test_verified_download_atomic_and_cached(tmp_path, monkeypatch):
    spec = asset_spec()
    monkeypatch.setattr("mct_bench.assets.urlopen", lambda *args, **kwargs: io.BytesIO(b"public model"))
    path = fetch_asset(spec, tmp_path)
    assert path.read_bytes() == b"public model"
    assert list(tmp_path.iterdir()) == [path]
    monkeypatch.setattr("mct_bench.assets.urlopen", lambda *args, **kwargs: pytest.fail("cache should avoid network"))
    assert fetch_asset(spec, tmp_path, offline=True) == path


def test_bad_download_never_publishes_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr("mct_bench.assets.urlopen", lambda *args, **kwargs: io.BytesIO(b"evil content"))
    with pytest.raises(ValueError, match="mismatch"):
        fetch_asset(asset_spec(), tmp_path)
    assert not list(tmp_path.iterdir())


def test_existing_corrupt_cache_is_rejected_even_online(tmp_path, monkeypatch):
    (tmp_path / "model.onnx").write_bytes(b"not the source")
    monkeypatch.setattr("mct_bench.assets.urlopen", lambda *args, **kwargs: pytest.fail("corruption must not be silently repaired"))
    with pytest.raises(ValueError, match="mismatch"):
        fetch_asset(asset_spec(), tmp_path)


def test_offline_missing_and_path_traversal(tmp_path):
    with pytest.raises(FileNotFoundError, match="Offline"):
        fetch_asset(asset_spec(), tmp_path, offline=True)
    with pytest.raises(ValueError, match="basename"):
        fetch_asset({**asset_spec(), "filename": "../escape"}, tmp_path)


def write_idx(tmp_path, payload):
    path = tmp_path / "sample.gz"
    path.write_bytes(gzip.compress(payload))
    return path


def test_idx_images_and_preprocessing(tmp_path):
    raw = np.arange(784, dtype=np.uint16).astype(np.uint8).reshape(1, 28, 28)
    path = write_idx(tmp_path, struct.pack(">IIII", 2051, 1, 28, 28) + raw.tobytes())
    decoded = read_idx_gzip(path, "images", expected_count=1)
    np.testing.assert_array_equal(decoded, raw)
    transformed = preprocess(decoded)
    assert transformed.shape == (1, 1, 28, 28)
    assert transformed.dtype == np.float32
    assert transformed.flags.c_contiguous
    assert transformed[0, 0, 0, 0] == pytest.approx(-0.286 / 0.353)
    assert transformed[0, 0, 9, 3] == pytest.approx((1 - 0.286) / 0.353)


@pytest.mark.parametrize("payload,kind,match", [
    (b"", "images", "Truncated"),
    (struct.pack(">IIII", 2049, 1, 28, 28), "images", "magic"),
    (struct.pack(">IIII", 2051, 1, 27, 28), "images", "shape"),
    (struct.pack(">IIII", 2051, 1, 28, 28) + bytes(783), "images", "payload"),
    (struct.pack(">IIII", 2051, 1, 28, 28) + bytes(785), "images", "payload"),
    (struct.pack(">II", 2049, 1) + bytes([10]), "labels", "labels"),
    (struct.pack(">II", 2049, 0), "labels", "count"),
])
def test_malformed_idx_rejected(tmp_path, payload, kind, match):
    with pytest.raises(ValueError, match=match):
        read_idx_gzip(write_idx(tmp_path, payload), kind)


def test_idx_labels_count_and_type(tmp_path):
    path = write_idx(tmp_path, struct.pack(">II", 2049, 2) + bytes([3, 9]))
    labels = read_idx_gzip(path, "labels", expected_count=2)
    np.testing.assert_array_equal(labels, [3, 9])
    assert labels.dtype == np.int64
    with pytest.raises(ValueError, match="expected"):
        read_idx_gzip(path, "labels", expected_count=3)


def test_preprocess_rejects_ambiguous_float_scaling():
    with pytest.raises(ValueError, match="uint8"):
        preprocess(np.zeros((1, 28, 28), dtype=np.float32))
