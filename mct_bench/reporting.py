"""One-table reports with measured deltas and complete machine-readable evidence."""

from __future__ import annotations

import csv
import json
import math
from numbers import Integral, Real
from pathlib import Path


_FOOTNOTE = (
    "Deltas compare each variant with the first (baseline) column. Negative size "
    "and latency deltas mean a reduction; speedup = baseline median / variant median. "
    "Accuracy deltas are percentage points (pp). N/A denotes division by zero.\n\n"
    "MCT fake-quant ONNX represents quantization effects with floating-point operators "
    "and tensors. The measured ONNX file size and CPU latency do not establish packed "
    "INT8/INT4 storage or native integer-kernel speed. Bit widths describe the configured "
    "quantization, not the ONNX tensor storage dtype. "
    "All variants use CPUExecutionProvider with ORT_DISABLE_ALL for export fidelity; "
    "these are unoptimized graph timings."
)


def _number(value: object, label: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a real number")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{label} must be finite and nonnegative")
    return value


def _validate(results: list[dict]) -> None:
    if not isinstance(results, list) or not results:
        raise ValueError("results must contain at least one baseline result")
    names: set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("every result must be a dict")
        name = result.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("variant names must be nonempty and unique")
        names.add(name)
        if _number(result.get("accuracy"), f"{name}.accuracy") > 1:
            raise ValueError("accuracy must be between 0 and 1")
        size = result.get("onnx_bytes")
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 0:
            raise ValueError(f"{name}.onnx_bytes must be a nonnegative integer")
        latency = result.get("latency")
        if not isinstance(latency, dict):
            raise ValueError(f"{name}.latency must be a dict")
        for metric in ("median_ms", "p90_ms", "mean_ms", "std_ms"):
            _number(latency.get(metric), f"{name}.latency.{metric}")


def _relative(current: float, baseline: float) -> str:
    return "N/A" if baseline == 0 else f"{100 * (current / baseline - 1):+.2f}"


def _rows(results: list[dict]) -> list[list[str]]:
    _validate(results)
    baseline = results[0]
    baseline_size = baseline["onnx_bytes"]
    baseline_latency = baseline["latency"]["median_ms"]

    def row(label: str, formatter) -> list[str]:
        return [label, *(formatter(result) for result in results)]

    return [
        row("Weight bits (configured)", lambda r: str(r.get("weights_bits", "N/A"))),
        row("Activation bits (configured)", lambda r: str(r.get("activation_bits", "N/A"))),
        row("Execution", lambda r: str(r.get("execution", "N/A"))),
        row("Threshold error method", lambda r: str(r.get("error_method", "N/A"))),
        row("Bias correction", lambda r: str(r.get("bias_correction", "N/A"))),
        row("Calibration images (train)", lambda r: str(r.get("calibration_samples", "N/A"))),
        row("Evaluation images (test)", lambda r: str(r.get("evaluation_samples", "N/A"))),
        row("Accuracy (%)", lambda r: f"{100 * r['accuracy']:.4f}"),
        row("Accuracy delta (pp)", lambda r: f"{100 * (r['accuracy'] - baseline['accuracy']):+.4f}"),
        row("ONNX file size (bytes)", lambda r: str(r["onnx_bytes"])),
        row("ONNX file size (MiB)", lambda r: f"{r['onnx_bytes'] / (1024 ** 2):.6f}"),
        row("Size delta (%)", lambda r: _relative(r["onnx_bytes"], baseline_size)),
        row("Latency batch size", lambda r: str(r.get("latency_batch_size", "N/A"))),
        row("CPU intra-op threads", lambda r: str(r.get("threads", "N/A"))),
        row("Timed calls per variant", lambda r: str(r.get("timed_calls", "N/A"))),
        row("Latency median (ms/call)", lambda r: f"{r['latency']['median_ms']:.6f}"),
        row("Latency p90 (ms/call)", lambda r: f"{r['latency']['p90_ms']:.6f}"),
        row("Latency mean (ms/call)", lambda r: f"{r['latency']['mean_ms']:.6f}"),
        row("Latency std (ms/call)", lambda r: f"{r['latency']['std_ms']:.6f}"),
        row("Median latency delta (%)", lambda r: _relative(r["latency"]["median_ms"], baseline_latency)),
        row("Speedup (baseline / variant)", lambda r: "N/A" if r["latency"]["median_ms"] == 0 else f"{baseline_latency / r['latency']['median_ms']:.4f}"),
    ]


def _markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def build_table(results: list[dict]) -> str:
    """Render metric rows and variant columns, using the first result as baseline."""
    rows = _rows(results)
    header = ["Metric", *(result["name"] for result in results)]
    table = [header, ["---", *(["---:"] * len(results))], *rows]
    lines = ["| " + " | ".join(_markdown_cell(cell) for cell in row) + " |" for row in table]
    return "\n".join(lines) + "\n\n" + _FOOTNOTE + "\n"


def write_reports(output_dir: Path, results: list[dict], metadata: dict) -> None:
    """Write matching Markdown/CSV tables and raw, unrounded JSON results.

    JSON preserves each result field, latency sample, and metadata field. The
    inputs must be JSON serializable; non-finite JSON numbers are rejected.
    """
    rows = _rows(results)
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict")
    # Validate serialization before creating any output files.
    raw = json.dumps({"metadata": metadata, "results": results}, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.md").write_text(build_table(results), encoding="utf-8")
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Metric", *(result["name"] for result in results)])
        writer.writerows(rows)
    (output_dir / "results.json").write_text(raw, encoding="utf-8")
