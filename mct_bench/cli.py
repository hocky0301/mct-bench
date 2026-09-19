"""CLI orchestration. Downloads, PTQ and evaluation never run during import."""

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import traceback

from .config import Variant, load_variants, variant_dict


def positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare MCT W8A8/W4A8 PTQ using measured standard ONNX artifacts.")
    commands = p.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="download, quantize, export, evaluate and print one comparison table")
    run.add_argument("--output", type=Path, default=None, help="new output directory (existing paths are refused)")
    run.add_argument("--cache", type=Path, default=Path(".cache/mct-bench"))
    run.add_argument("--offline", action="store_true", help="require every hash-verified asset in the cache")
    variants = run.add_mutually_exclusive_group()
    variants.add_argument("--config", type=Path, help="JSON variant configuration")
    variants.add_argument("--bits", type=int, choices=(4, 8), nargs="+", help="weight bit widths, activations fixed at 8")
    run.add_argument("--calibration-samples", type=positive, default=512)
    run.add_argument("--eval-samples", type=positive, default=10000)
    run.add_argument("--calibration-batch-size", type=positive, default=32)
    run.add_argument("--eval-batch-size", type=positive, default=128)
    run.add_argument("--latency-batch-size", type=positive, default=1)
    run.add_argument("--warmup", type=nonnegative, default=20, help="untimed calls per variant")
    run.add_argument("--repeats", type=positive, default=100, help="timed calls per variant")
    run.add_argument("--threads", type=positive, default=1, help="ORT and PyTorch CPU intra-op threads")
    run.add_argument("--seed", type=nonnegative, default=42)
    run.add_argument("--note", default="", help="context to preserve, e.g. other workloads running on the host")
    prepare = commands.add_parser("prepare", help="download and verify pinned public assets")
    prepare.add_argument("--cache", type=Path, default=Path(".cache/mct-bench"))
    prepare.add_argument("--offline", action="store_true")
    commands.add_parser("licenses", help="print licenses and attribution sources")
    return p


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def cpu_model() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if sys.platform == "linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.partition(":")[2].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def execute_run(args) -> Path:
    variants = ([Variant(f"w{b}a8", b) for b in args.bits] if args.bits else load_variants(args.config))
    if len({v.name for v in variants}) != len(variants):
        raise ValueError("--bits must not contain duplicate widths")
    if args.calibration_samples > 60000 or args.eval_samples > 10000:
        raise ValueError("Fashion-MNIST has 60000 training and 10000 test samples")
    if args.latency_batch_size > args.eval_samples:
        raise ValueError("latency-batch-size must be <= eval-samples")
    output = args.output or Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / "run-status.json"
    started = datetime.now(timezone.utc).isoformat()
    write_json(status_path, {"status": "running", "started_utc": started})
    try:
        # The CPU-only contract includes MCT's separately selected quantizer device.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        import numpy as np
        import torch
        from .assets import prepare_assets, load_data, preprocess
        from .model import load_model
        from .measurement import accuracy, benchmark_sessions
        from .quantization import quantize, export_onnx
        from .reporting import write_reports
        from .runtime import OnnxRunner, check_parity, predict

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        random.seed(args.seed)
        np.random.seed(args.seed % (2 ** 32))
        torch.manual_seed(args.seed)
        print("Preparing pinned model and Fashion-MNIST assets…", file=sys.stderr)
        paths = prepare_assets(args.cache, offline=args.offline)
        train_images, train_labels, test_images, test_labels = load_data(paths)
        rng = np.random.default_rng(args.seed)
        calibration_indices = rng.permutation(len(train_images))[:args.calibration_samples]
        evaluation_indices = rng.permutation(len(test_images))[:args.eval_samples]
        calibration = preprocess(train_images[calibration_indices])
        evaluation = preprocess(test_images[evaluation_indices])
        labels = test_labels[evaluation_indices]
        write_json(output / "sample-indices.json", {
            "calibration_split": "train", "calibration_indices": calibration_indices.tolist(),
            "evaluation_split": "test", "evaluation_indices": evaluation_indices.tolist(),
        })

        def representative_data_gen():
            for start in range(0, len(calibration), args.calibration_batch_size):
                yield [torch.from_numpy(calibration[start:start + args.calibration_batch_size].copy())]

        model = load_model(paths["model_onnx"]).cpu().eval()
        parity_inputs = evaluation[:min(32, len(evaluation))]
        with torch.inference_mode():
            reconstructed_logits = model(torch.from_numpy(parity_inputs)).numpy()
        source_runner = OnnxRunner(paths["model_onnx"], args.threads)
        source_parity = check_parity(source_runner(parity_inputs), reconstructed_logits)
        del source_runner
        artifact_dir = output / "models"
        artifact_dir.mkdir()
        baseline_path = artifact_dir / "fp32.onnx"
        torch.onnx.export(model, torch.from_numpy(evaluation[:1]), str(baseline_path),
                          input_names=["images"], output_names=["logits"], opset_version=17,
                          dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                          do_constant_folding=True, dynamo=False)
        runners = {"fp32": OnnxRunner(baseline_path, args.threads)}
        parity = {"source_onnx_to_pytorch": source_parity,
                  "fp32_export": check_parity(reconstructed_logits, runners["fp32"](parity_inputs))}
        results = [{"name": "fp32", "weights_bits": 32, "activation_bits": 32,
                    "execution": "FP32 ONNX / CPU", "path": "models/fp32.onnx"}]
        for variant in variants:
            print(f"MCT PTQ: {variant.name}…", file=sys.stderr)
            # MCT graph transformations may mutate their input; each variant starts from source weights.
            fresh_model = load_model(paths["model_onnx"]).cpu().eval()
            torch.manual_seed(args.seed)
            quant_model, audit = quantize(fresh_model, representative_data_gen,
                                          weights_bits=variant.weights_bits,
                                          activation_bits=variant.activation_bits,
                                          error_method=variant.error_method,
                                          bias_correction=variant.bias_correction)
            path = artifact_dir / f"{variant.name}.onnx"
            # Capture reference before an exporter that may replace wrappers in-place.
            with torch.inference_mode():
                quantized_logits = quant_model(torch.from_numpy(parity_inputs)).detach().cpu().numpy()
            export_metadata = export_onnx(quant_model, representative_data_gen, path)
            runners[variant.name] = OnnxRunner(path, args.threads)
            output_step = audit["output_quantization_step"]
            if output_step is None or output_step <= 0:
                raise RuntimeError("cannot establish final output quantization step for export verification")
            parity[variant.name] = check_parity(quantized_logits, runners[variant.name](parity_inputs),
                                                atol=output_step + 1e-5, rtol=0)
            parity[variant.name]["output_quantization_step"] = output_step
            parity[variant.name]["tolerance_basis"] = "one final-output quantization step plus 1e-5 float roundoff"
            results.append({**variant_dict(variant), "execution": "MCT fake-quant ONNX / CPU",
                            "path": f"models/{variant.name}.onnx", "quantization_audit": audit,
                            "export_audit": export_metadata})
            del quant_model, fresh_model
        print(f"Evaluating {len(labels)} held-out test images per variant…", file=sys.stderr)
        for result in results:
            name = result["name"]
            logits = predict(runners[name], evaluation, args.eval_batch_size)
            result["accuracy"] = accuracy(logits, labels)
            result["correct"] = int(np.sum(logits.argmax(axis=1) == labels))
            result["evaluation_samples"] = int(len(labels))
            result["latency_batch_size"] = args.latency_batch_size
            result["calibration_samples"] = args.calibration_samples if name != "fp32" else 0
            result["timed_calls"] = args.repeats
            result["threads"] = args.threads
            artifact = output / result["path"]
            result["onnx_bytes"] = artifact.stat().st_size
            result["sha256"] = sha256(artifact)
            result["operators"] = runners[name].operator_counts
            result["initializer_storage_types"] = sorted(runners[name].tensor_storage)
        count = min(16, len(evaluation) // args.latency_batch_size)
        latency_inputs = [np.ascontiguousarray(evaluation[i * args.latency_batch_size:
                                                       (i + 1) * args.latency_batch_size])
                          for i in range(count)]
        print(f"Timing CPU inference: batch={args.latency_batch_size}, repeats={args.repeats}…", file=sys.stderr)
        load_before = list(os.getloadavg())
        timings = benchmark_sessions(runners, latency_inputs, args.warmup, args.repeats, args.seed)
        load_after = list(os.getloadavg())
        for result in results:
            result["latency"] = timings[result["name"]]
        package_versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
        metadata = {
            "schema_version": 1, "started_utc": started,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "command": [Path(sys.argv[0]).name, *sys.argv[1:]], "python": sys.version, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "cpu_model": cpu_model(), "load_average_before_timing": load_before,
            "load_average_after_timing": load_after, "user_note": args.note,
            "dependencies": dict(sorted(package_versions.items())), "seed": args.seed,
            "dataset": "Fashion-MNIST", "model": "tsilva/fashionmnist-classifier-cnn",
            "assets": {key: {"filename": path.name, "sha256": sha256(path)} for key, path in paths.items()},
            "calibration_samples": len(calibration), "calibration_split": "train",
            "evaluation_samples": len(evaluation), "evaluation_split": "test",
            "sample_indices_file": "sample-indices.json", "sample_indices_sha256": sha256(output / "sample-indices.json"),
            "preprocessing": "float32 NCHW; (uint8/255 - 0.2860) / 0.3530",
            "calibration_batch_size": args.calibration_batch_size, "evaluation_batch_size": args.eval_batch_size,
            "latency_batch_size": args.latency_batch_size, "latency_input_batches": count,
            "warmup_calls_per_variant": args.warmup, "timed_calls_per_variant": args.repeats,
            "intra_op_threads": args.threads, "inter_op_threads": 1,
            "provider": "CPUExecutionProvider", "ort_optimization": "ORT_DISABLE_ALL",
            "timing": "perf_counter_ns; prepared inputs; randomized variant order per repeat; loading, preprocessing, warmup excluded",
            "export": "ONNX opset 17; standard fake-quant operators; FP32 weight storage (no low-bit packing)",
            "scope": "host CPU graph comparison; not native INT4/INT8 kernel or edge-device benchmark",
            "export_parity": parity,
        }
        root = Path(__file__).resolve().parent.parent
        shutil.copy2(root / "THIRD_PARTY_LICENSES.md", output / "THIRD_PARTY_LICENSES.md")
        shutil.copytree(root / "licenses", output / "licenses")
        shutil.copy2(Path(__file__).with_name("assets_manifest.json"), output / "assets_manifest.json")
        write_reports(output, results, metadata)
        write_json(status_path, {"status": "complete", "started_utc": started,
                                 "completed_utc": metadata["completed_utc"]})
        return output
    except BaseException as exc:
        write_json(status_path, {"status": "failed", "started_utc": started,
                                 "error_type": type(exc).__name__, "error": str(exc)})
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "licenses":
            print((Path(__file__).resolve().parent.parent / "THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8"))
        elif args.command == "prepare":
            from .assets import prepare_assets
            paths = prepare_assets(args.cache, offline=args.offline)
            print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
        else:
            # Reserve stdout for a single machine-capturable Markdown table.
            with contextlib.redirect_stdout(sys.stderr):
                output = execute_run(args)
            print((output / "comparison.md").read_text(encoding="utf-8"))
            print(f"Artifacts: {output}", file=sys.stderr)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"mct-bench: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
