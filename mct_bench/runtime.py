"""One ONNX Runtime CPU execution path for every measured artifact."""

from pathlib import Path

import numpy as np


class OnnxRunner:
    def __init__(self, path: Path, threads: int = 1):
        import onnx
        import onnxruntime as ort

        graph = onnx.load(str(path), load_external_data=False)
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
            raise ValueError("external ONNX tensor files are unsupported: size must include the whole model")
        onnx.checker.check_model(graph, full_check=True)
        custom_domains = {node.domain for node in graph.graph.node} - {"", "ai.onnx"}
        if custom_domains or graph.functions:
            raise ValueError(f"only standard ONNX operators are permitted: {custom_domains}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Optimizer Q/DQ propagation changed the real model's outputs by several
        # quantization bins. Keep the exported computation graph for all variants.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("only CPUExecutionProvider may be active")
        if len(self.session.get_inputs()) != 1 or len(self.session.get_outputs()) != 1:
            raise ValueError("expected a single image input and logits output")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.operator_counts = {op: sum(n.op_type == op for n in graph.graph.node)
                                for op in sorted({n.op_type for n in graph.graph.node})}
        self.tensor_storage = {onnx.TensorProto.DataType.Name(t.data_type)
                               for t in graph.graph.initializer}

    def __call__(self, images: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: images})[0]


def predict(runner, images: np.ndarray, batch_size: int) -> np.ndarray:
    if len(images) == 0 or batch_size < 1:
        raise ValueError("nonempty images and positive batch_size required")
    return np.concatenate([runner(images[i:i + batch_size])
                           for i in range(0, len(images), batch_size)])


def check_parity(reference: np.ndarray, candidate: np.ndarray, *, atol=1e-4, rtol=1e-4) -> dict:
    if reference.shape != candidate.shape or not np.isfinite(candidate).all():
        raise ValueError("export parity failed: invalid output shape or non-finite outputs")
    difference = float(np.max(np.abs(reference - candidate)))
    np.testing.assert_allclose(candidate, reference, atol=atol, rtol=rtol,
                               err_msg="export changed model outputs")
    return {"samples": int(len(reference)), "max_absolute_logit_error": difference,
            "atol": atol, "rtol": rtol,
            "argmax_agreement": float(np.mean(reference.argmax(1) == candidate.argmax(1)))}
