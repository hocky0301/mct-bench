import copy
import csv
import json

import pytest

from mct_bench.reporting import build_table, write_reports


@pytest.fixture
def results():
    baseline = {
        "name": "fp32", "weights_bits": 32, "activation_bits": 32,
        "accuracy": 0.91, "onnx_bytes": 1_048_576,
        "latency": {"median_ms": 2.0, "p90_ms": 2.4, "mean_ms": 2.1, "std_ms": 0.2, "samples_ms": [1.9, 2.0, 2.4]},
        "sha256": "123", "path": "models/fp32.onnx", "execution": "FP32 CPU",
    }
    quantized = {
        "name": "w4a8", "weights_bits": 4, "activation_bits": 8,
        "accuracy": 0.89, "onnx_bytes": 1_572_864,
        "latency": {"median_ms": 3.0, "p90_ms": 3.4, "mean_ms": 3.1, "std_ms": 0.3, "samples_ms": [2.8, 3.0, 3.4]},
        "sha256": "456", "path": "models/w4a8.onnx", "execution": "MCT fake-quant ONNX / CPU",
    }
    return [baseline, quantized]


def test_table_uses_actual_files_and_accuracy_percentage_points(results):
    table = build_table(results)
    assert "| Metric | fp32 | w4a8 |" in table
    assert "| Accuracy (%) | 91.0000 | 89.0000 |" in table
    assert "| Accuracy delta (pp) | +0.0000 | -2.0000 |" in table
    assert "| ONNX file size (bytes) | 1048576 | 1572864 |" in table
    assert "| ONNX file size (MiB) | 1.000000 | 1.500000 |" in table
    assert "| Size delta (%) | +0.00 | +50.00 |" in table
    assert "| Median latency delta (%) | +0.00 | +50.00 |" in table
    assert "| Speedup (baseline / variant) | 1.0000 | 0.6667 |" in table
    assert "do not establish packed INT8/INT4 storage or native integer-kernel speed" in table


def test_zero_denominators_are_explicit(results):
    results[0]["onnx_bytes"] = 0
    results[0]["latency"]["median_ms"] = 0.0
    table = build_table(results)
    assert "| Size delta (%) | N/A | N/A |" in table
    assert "| Median latency delta (%) | N/A | N/A |" in table
    assert "| Speedup (baseline / variant) | N/A | 0.0000 |" in table
    results[1]["latency"]["median_ms"] = 0.0
    assert "| Speedup (baseline / variant) | N/A | N/A |" in build_table(results)


def test_reduction_and_accuracy_improvement_signs(results):
    results[1]["onnx_bytes"] = 524_288
    results[1]["latency"]["median_ms"] = 1.0
    results[1]["accuracy"] = 0.92
    table = build_table(results)
    assert "| Accuracy delta (pp) | +0.0000 | +1.0000 |" in table
    assert "| Size delta (%) | +0.00 | -50.00 |" in table
    assert "| Median latency delta (%) | +0.00 | -50.00 |" in table
    assert "| Speedup (baseline / variant) | 1.0000 | 2.0000 |" in table


def test_reports_share_table_and_json_preserves_raw_evidence(tmp_path, results):
    results[0]["accuracy"] = 0.912345678901
    results[1]["extra"] = {"operator_statistics": [1, 2, 3]}
    metadata = {"seed": 42, "hardware": "日本語 CPU", "license_manifest": [{"id": "MIT"}]}
    original = copy.deepcopy(results)
    output = tmp_path / "nested" / "report"
    write_reports(output, results, metadata)
    assert results == original
    raw = json.loads((output / "results.json").read_text())
    assert raw == {"metadata": metadata, "results": original}
    markdown = (output / "comparison.md").read_text()
    assert markdown == build_table(results)
    with (output / "comparison.csv").open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["Metric", "fp32", "w4a8"]
    for row in rows:
        assert "| " + " | ".join(row) + " |" in markdown


def test_variant_names_are_escaped_in_markdown_and_preserved_in_csv(tmp_path, results):
    results[1]["name"] = "w4|a8\nconfiguration"
    write_reports(tmp_path, results, {})
    assert "w4\\|a8<br>configuration" in (tmp_path / "comparison.md").read_text()
    with (tmp_path / "comparison.csv").open(newline="") as stream:
        assert next(csv.reader(stream))[2] == results[1]["name"]


@pytest.mark.parametrize("field,value", [("accuracy", float("nan")), ("accuracy", 1.1), ("accuracy", -0.1), ("onnx_bytes", -1), ("onnx_bytes", 1.5), ("name", "")])
def test_invalid_metrics_are_rejected(results, field, value):
    results[1][field] = value
    with pytest.raises(ValueError):
        build_table(results)


def test_empty_results_duplicate_names_and_nonfinite_latency_are_rejected(results):
    with pytest.raises(ValueError, match="baseline"):
        build_table([])
    results[1]["name"] = "fp32"
    with pytest.raises(ValueError, match="unique"):
        build_table(results)
    results[1]["name"] = "ptq"
    results[1]["latency"]["median_ms"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        build_table(results)


def test_nonfinite_raw_samples_fail_before_creating_reports(tmp_path, results):
    results[1]["latency"]["samples_ms"].append(float("nan"))
    output = tmp_path / "failed"
    with pytest.raises(ValueError):
        write_reports(output, results, {})
    assert not output.exists()
