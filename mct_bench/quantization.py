"""MCT PTQ and portable ONNX fake-quant export, pinned to MCT 2.6.0.

W4A8 means four-bit kernel values and eight-bit activations. Exported weights
remain float32. The serialized ONNX byte count is consequently not a packed
four-bit model size; callers must report the actual artifact size.
"""

from __future__ import annotations

import copy
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable


def _validate_settings(weights_bits: int, activation_bits: int, error_method: str) -> str:
    if type(weights_bits) is not int or weights_bits not in (4, 8):
        raise ValueError("weights_bits must be 4 or 8")
    if type(activation_bits) is not int or activation_bits != 8:
        raise ValueError(
            "activation_bits must be 8: the pinned standard ONNX exporter does "
            "not support four-bit activation ranges; use W4A8 for four-bit weights"
        )
    aliases = {"mse": "MSE", "mae": "MAE", "no_clipping": "NOCLIPPING", "noclipping": "NOCLIPPING"}
    try:
        return aliases[error_method]
    except (KeyError, TypeError) as exc:
        raise ValueError("error_method must be mse, mae, or no_clipping") from exc


def make_tpc(weights_bits: int, activation_bits: int = 8):
    """Give every eligible kernel exactly one requested precision candidate.

    Reuse Sony's operator mapping/fusions, changing its kernel bit width. Biases
    and shape/index operations keep the upstream non-quantized policy. This
    describes an experiment, not compliance with a particular edge device.
    """
    _validate_settings(weights_bits, activation_bits, "mse")
    from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR
    from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v1_0.tpc import (
        generate_tpc,
        get_op_quantization_configs,
    )

    base, _, default = get_op_quantization_configs()
    base = base.clone_and_edit(
        attr_to_edit={KERNEL_ATTR: {"weights_n_bits": weights_bits}},
        activation_n_bits=activation_bits,
        supported_input_activation_n_bits=(activation_bits,),
    )
    default = default.clone_and_edit(
        activation_n_bits=activation_bits,
        supported_input_activation_n_bits=(activation_bits,),
    )
    return generate_tpc(
        default_config=default,
        base_config=base,
        mixed_precision_cfg_list=[base],
        name=f"mct_bench_w{weights_bits}a{activation_bits}",
    )


def audit_quantizers(model, weights_bits: int, activation_bits: int = 8) -> dict[str, Any]:
    """Inspect installed quantizers; fail if the output disagrees with its label."""
    from mct_quantizers import PytorchActivationQuantizationHolder, PytorchQuantizationWrapper

    weights = []
    activations = []
    for name, module in model.named_modules():
        if isinstance(module, PytorchQuantizationWrapper):
            for attribute, quantizer in module.weights_quantizers.items():
                actual = int(quantizer.num_bits)
                if actual != weights_bits:
                    raise RuntimeError(f"Kernel {name}.{attribute}: expected {weights_bits} bits, got {actual}")
                weights.append({
                    "module": name,
                    "attribute": str(attribute),
                    "bits": actual,
                    "quantizer": type(quantizer).__name__,
                    "per_channel": bool(getattr(quantizer, "per_channel", False)),
                })
        if isinstance(module, PytorchActivationQuantizationHolder):
            quantizer = module.activation_holder_quantizer
            actual = int(quantizer.num_bits)
            if actual != activation_bits:
                raise RuntimeError(f"Activation {name}: expected {activation_bits} bits, got {actual}")
            activations.append({
                "module": name,
                "bits": actual,
                "quantizer": type(quantizer).__name__,
            })
    if not weights or not activations:
        raise RuntimeError("MCT output must contain both weight and activation quantizers")
    # Follow the actual graph output rather than assuming the last module is it.
    # A one-LSB output difference can result when runtime graph optimization
    # changes float accumulation order at an intermediate rounding boundary.
    output_step = None
    if hasattr(model, "graph") and hasattr(model, "node_to_activation_quantization_holder"):
        outputs = model.graph.get_outputs()
        if len(outputs) == 1:
            holder_name = model.node_to_activation_quantization_holder.get(outputs[0].node.name)
            if holder_name is not None:
                holder = model.get_submodule(holder_name)
                output_step = float(holder.activation_holder_quantizer.scales)
                if not math.isfinite(output_step) or output_step <= 0:
                    raise RuntimeError("Invalid output quantization step")
    return {
        "weights_bits": weights_bits,
        "activation_bits": activation_bits,
        "weight_quantizers": weights,
        "activation_quantizers": activations,
        "weight_quantizer_count": len(weights),
        "activation_quantizer_count": len(activations),
        "output_quantization_step": output_step,
        "bias_policy": "unquantized float32",
        "kernel_quantization": "symmetric per-channel",
        "activation_quantization": "power-of-two per-tensor",
    }


def quantize(
    model,
    representative_data_gen: Callable,
    weights_bits: int,
    activation_bits: int = 8,
    error_method: str = "mse",
    bias_correction: bool = True,
):
    """Apply MCT PTQ to a private CPU copy and return model plus actual-bit audit.

    ``representative_data_gen`` must be a callable returning a fresh iterator of
    ``[float32 NCHW numpy batch]`` each time. MCT can iterate calibration data
    repeatedly. Never provide an exhausted iterator or evaluation data here.
    """
    error_name = _validate_settings(weights_bits, activation_bits, error_method)
    if type(bias_correction) is not bool:
        raise ValueError("bias_correction must be a boolean")
    if not callable(representative_data_gen):
        raise TypeError("representative_data_gen must return a fresh iterator")

    # MCT and mct-quantizers have separate device selection. Both must see CPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import model_compression_toolkit as mct
    from model_compression_toolkit.core.pytorch.pytorch_device_config import set_working_device

    if torch.cuda.is_available():
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' before initializing torch for this CPU benchmark")
    set_working_device("cpu")
    method = getattr(mct.core.QuantizationErrorMethod, error_name)
    settings = mct.core.QuantizationConfig(
        weights_error_method=method,
        activation_error_method=method,
        weights_bias_correction=bias_correction,
    )
    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
        in_module=copy.deepcopy(model).cpu().eval(),
        representative_data_gen=representative_data_gen,
        core_config=mct.core.CoreConfig(quantization_config=settings),
        target_platform_capabilities=make_tpc(weights_bits, activation_bits),
    )
    quantized_model.cpu().eval()
    audit = audit_quantizers(quantized_model, weights_bits, activation_bits)
    audit.update({
        "mct_version": mct.__version__,
        "error_method": error_name.lower(),
        "bias_correction": bias_correction,
    })
    return quantized_model, audit


def export_onnx(
    quantized_model,
    representative_data_gen: Callable,
    path: str | Path,
    *,
    opset: int = 17,
) -> dict[str, Any]:
    """Export dynamic-batch standard ONNX, verify it, and describe real storage.

    The default MCT format uses custom Sony operations, so explicitly select
    FAKELY_QUANT. Conv/Gemm use float weights; activation Q/DQ simulates eight-bit
    precision. This does not promise integer kernels or lower inference time.
    """
    import model_compression_toolkit as mct
    import onnx

    if opset != 17:
        raise ValueError("This harness validates ONNX opset 17 only")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mct.exporter.pytorch_export_model(
        model=quantized_model,
        save_model_path=str(path),
        repr_dataset=representative_data_gen,
        serialization_format=mct.exporter.PytorchExportSerializationFormat.ONNX,
        quantization_format=mct.exporter.QuantizationFormat.FAKELY_QUANT,
        onnx_opset_version=opset,
    )
    exported = onnx.load(str(path))
    onnx.checker.check_model(exported, full_check=True)
    domains = sorted({node.domain for node in exported.graph.node})
    if any(domain not in ("", "ai.onnx") for domain in domains):
        raise RuntimeError(f"Nonstandard ONNX node domains found: {domains}")
    if any(initializer.data_location == onnx.TensorProto.EXTERNAL for initializer in exported.graph.initializer):
        raise RuntimeError("Expected one self-contained ONNX file, found external tensor data")
    node_counts = dict(sorted(Counter(node.op_type for node in exported.graph.node).items()))
    if not node_counts.get("QuantizeLinear") or not node_counts.get("DequantizeLinear"):
        raise RuntimeError("Expected activation QuantizeLinear/DequantizeLinear nodes in fake-quant export")
    storage = Counter()
    for initializer in exported.graph.initializer:
        array = onnx.numpy_helper.to_array(initializer)
        storage[str(array.dtype)] += array.nbytes
    return {
        "path": str(path),
        "opset": opset,
        "format": "ONNX FAKELY_QUANT",
        "execution": "float graph with activation Q/DQ; no integer-kernel guarantee",
        "weight_storage": "float32 (including four-bit quantized kernel values)",
        "packed_low_bit_storage": False,
        "file_bytes": path.stat().st_size,
        "node_counts": node_counts,
        "node_domains": domains,
        "initializer_bytes_by_dtype": dict(storage),
    }
