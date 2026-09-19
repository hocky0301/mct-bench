"""Opt-in public-asset integration, including settings beyond the two defaults."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.assets
def test_real_public_model_end_to_end(tmp_path):
    cache = os.environ.get("MCT_BENCH_CACHE")
    if not cache:
        pytest.skip("set MCT_BENCH_CACHE to a prepared public asset cache")
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "run"
    command = [sys.executable, "-m", "mct_bench", "run", "--offline", "--cache", str(Path(cache).resolve()),
               "--output", str(output), "--config", str(root / "configs/comparison.json"),
               "--calibration-samples", "32", "--eval-samples", "64", "--warmup", "1", "--repeats", "3"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (output / "comparison.md").read_text().strip()
    assert json.loads((output / "run-status.json").read_text())["status"] == "complete"
    raw = json.loads((output / "results.json").read_text())
    assert len(raw["results"]) == 5
    assert raw["results"][0]["accuracy"] > .85
    for variant in raw["results"]:
        assert variant["onnx_bytes"] == (output / variant["path"]).stat().st_size
        assert len(variant["latency"]["samples_ms"]) == 3
        if variant["name"] != "fp32":
            audit = variant["quantization_audit"]
            assert audit["weight_quantizer_count"] == 7  # six conv + one dense
            assert {q["bits"] for q in audit["weight_quantizers"]} == {variant["weights_bits"]}
            assert variant["export_audit"]["packed_low_bit_storage"] is False
    meta = raw["metadata"]
    assert meta["ort_optimization"] == "ORT_DISABLE_ALL"
    assert meta["provider"] == "CPUExecutionProvider"
    indices = json.loads((output / "sample-indices.json").read_text())
    assert indices["calibration_split"] == "train"
    assert indices["evaluation_split"] == "test"
    assert len(set(indices["calibration_indices"])) == 32
    assert len(set(indices["evaluation_indices"])) == 64
    assert (output / "licenses/Fashion-MNIST.LICENSE").exists()
