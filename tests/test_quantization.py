"""Synthetic fixtures test mechanics only; benchmark results use public assets."""

import copy

import numpy as np
import pytest

from mct_bench.quantization import _validate_settings, audit_quantizers, export_onnx, make_tpc, quantize


@pytest.mark.parametrize("bits", [0, 2, 16, True, 4.0])
def test_reject_unsupported_weight_bits(bits):
    with pytest.raises(ValueError, match="weights_bits"):
        _validate_settings(bits, 8, "mse")


def test_four_bit_activations_fail_instead_of_silent_fallback():
    with pytest.raises(ValueError, match="four-bit activation"):
        _validate_settings(4, 4, "mse")


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.integration
def test_uniform_ptq_bits_and_export_parity(bits, tmp_path):
    import onnx
    import onnxruntime as ort
    import torch
    from mct_quantizers import PytorchQuantizationWrapper

    torch.set_num_threads(1)
    torch.manual_seed(21)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 3, 3),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 6 * 6, 2),
    ).eval()
    original = copy.deepcopy(model.state_dict())
    rng = np.random.default_rng(21)
    calibration = rng.random((8, 1, 8, 8), dtype=np.float32)
    heldout = rng.random((3, 1, 8, 8), dtype=np.float32)

    def representative_data_gen():
        for start in (0, 4):
            yield [calibration[start:start + 4]]

    quantized, audit = quantize(model, representative_data_gen, bits)
    assert audit["weight_quantizer_count"] == 2
    assert {entry["bits"] for entry in audit["weight_quantizers"]} == {bits}
    assert {entry["bits"] for entry in audit["activation_quantizers"]} == {8}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    # Check the actual kernel value lattice, in addition to configuration labels.
    for module in quantized.modules():
        if isinstance(module, PytorchQuantizationWrapper):
            for attribute, quantizer in module.weights_quantizers.items():
                weights = module.get_quantized_weights()[attribute].detach().numpy()
                weights = np.moveaxis(weights, quantizer.channel_axis, 0)
                assert all(len(np.unique(channel)) <= 2 ** bits for channel in weights)

    path = tmp_path / f"w{bits}a8.onnx"
    metadata = export_onnx(quantized, representative_data_gen, path)
    assert metadata["packed_low_bit_storage"] is False
    assert metadata["initializer_bytes_by_dtype"]["float32"] > 0
    assert metadata["node_domains"] == [""]
    assert metadata["file_bytes"] == path.stat().st_size
    exported = onnx.load(path)
    assert exported.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch_size"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: heldout})[0]
    with torch.inference_mode():
        expected = quantized(torch.from_numpy(heldout)).numpy()
    output_step = audit["output_quantization_step"]
    assert output_step > 0
    # With graph optimizations enabled, ONNX Runtime may rewrite the exported
    # Quantize/Dequantize pairs and accumulate differently. On macOS arm64 the
    # result stayed within one output bin; on Linux x86-64 (GitHub Actions
    # ubuntu-24.04, measured 2026-09-19) it moved by up to seven bins for this
    # synthetic model. That is exactly why the benchmark compares every setting
    # with ORT_DISABLE_ALL (below). Here the optimized session only has to keep
    # the predictions; the exact-parity check is the plain session.
    assert actual.shape == expected.shape
    assert (actual.argmax(axis=1) == expected.argmax(axis=1)).all()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    plain_session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    plain = plain_session.run(None, {plain_session.get_inputs()[0].name: heldout})[0]
    np.testing.assert_allclose(plain, expected, rtol=1e-5, atol=1e-5)

    with pytest.raises(RuntimeError, match="expected"):
        audit_quantizers(quantized, 4 if bits == 8 else 8)


@pytest.mark.integration
def test_tpc_has_no_mixed_precision_candidates():
    tpc = make_tpc(4)
    for operator in tpc.operator_set:
        if operator.qc_options is not None:
            assert len(operator.qc_options.quantization_configurations) == 1


@pytest.mark.integration
def test_no_quantizers_cannot_pass_audit():
    import torch

    with pytest.raises(RuntimeError, match="both weight and activation"):
        audit_quantizers(torch.nn.Identity(), 8)
