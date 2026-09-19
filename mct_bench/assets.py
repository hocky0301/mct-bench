"""Pinned public assets, verified downloads, and strict Fashion-MNIST decoding."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Mapping
from urllib.request import Request, urlopen

import numpy as np

MANIFEST_PATH = Path(__file__).with_name("assets_manifest.json")


def read_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(path: str | Path, spec: Mapping) -> None:
    path = Path(path)
    if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
        raise ValueError(f"Asset checksum/size mismatch: {path}; remove it and download again.")


def fetch_asset(spec: Mapping, cache_dir: str | Path, offline: bool = False) -> Path:
    """Reuse verified files; never silently repair a corrupt cache or publish partial bytes."""
    cache_dir = Path(cache_dir)
    name = spec["filename"]
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Asset filename must be a plain basename.")
    destination = cache_dir / name
    if destination.exists():
        verify_asset(destination, spec)
        return destination
    if offline:
        raise FileNotFoundError(f"Offline asset missing: {destination}; run the download command first.")
    if not spec["url"].startswith("https://"):
        raise ValueError("Asset download URLs must use HTTPS.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{name}.", suffix=".part", dir=cache_dir, delete=False) as out:
            temporary = Path(out.name)
            request = Request(spec["url"], headers={"User-Agent": "mct-bench/1.0"})
            with urlopen(request, timeout=60) as response:
                copied = 0
                while chunk := response.read(1024 * 1024):
                    copied += len(chunk)
                    if copied > spec["bytes"]:
                        raise ValueError(f"Asset exceeds pinned size: {name}")
                    out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        verify_asset(temporary, spec)
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_assets(cache_dir: str | Path, offline: bool = False) -> dict[str, Path]:
    return {key: fetch_asset(spec, cache_dir, offline) for key, spec in read_manifest()["assets"].items()}


def read_idx_gzip(path: str | Path, kind: str, expected_count: int | None = None) -> np.ndarray:
    """Validate IDX magic, dimensions, count and exact payload length, then return owned arrays."""
    if kind not in {"images", "labels"}:
        raise ValueError("IDX kind must be images or labels.")
    header_size = 16 if kind == "images" else 8
    with gzip.open(path, "rb") as stream:
        header = stream.read(header_size)
        if len(header) != header_size:
            raise ValueError(f"Truncated IDX header: {path}")
        values = struct.unpack(">IIII" if kind == "images" else ">II", header)
        magic, count = values[:2]
        if magic != (2051 if kind == "images" else 2049):
            raise ValueError(f"Incorrect IDX magic: {path}")
        if count <= 0 or count > 60000:
            raise ValueError(f"Invalid IDX count: {count}")
        if expected_count is not None and count != expected_count:
            raise ValueError(f"IDX count {count} does not match expected {expected_count}")
        if kind == "images" and values[2:] != (28, 28):
            raise ValueError("Fashion-MNIST images must have shape 28 x 28.")
        size = count * (28 * 28 if kind == "images" else 1)
        payload = stream.read(size + 1)
        if len(payload) != size:
            raise ValueError(f"IDX payload length mismatch: {path}")
    array = np.frombuffer(payload, dtype=np.uint8).copy()
    if kind == "images":
        return array.reshape(count, 28, 28)
    if np.any(array > 9):
        raise ValueError("Fashion-MNIST labels must be in [0, 9].")
    return array.astype(np.int64)


def load_data(paths: Mapping[str, str | Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw uint8 images (N,H,W) and int64 labels; calibration uses train only."""
    manifest = read_manifest()
    for key in ("train_images", "train_labels", "test_images", "test_labels"):
        verify_asset(paths[key], manifest["assets"][key])
    dataset = manifest["dataset"]
    return (
        read_idx_gzip(paths["train_images"], "images", dataset["train_count"]),
        read_idx_gzip(paths["train_labels"], "labels", dataset["train_count"]),
        read_idx_gzip(paths["test_images"], "images", dataset["test_count"]),
        read_idx_gzip(paths["test_labels"], "labels", dataset["test_count"]),
    )


def preprocess(images: np.ndarray) -> np.ndarray:
    """Apply the publisher's deterministic normalization to raw Fashion-MNIST pixels."""
    images = np.asarray(images)
    if images.ndim != 3 or images.shape[1:] != (28, 28) or images.dtype != np.uint8:
        raise ValueError("Expected uint8 Fashion-MNIST images of shape [N, 28, 28].")
    result = (images.astype(np.float32) / np.float32(255.0) - np.float32(0.2860)) / np.float32(0.3530)
    return np.ascontiguousarray(result[:, None, :, :])
