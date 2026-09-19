"""Audited PyTorch view of the publisher's frozen ONNX CNN, without pickle or remote code."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch import nn

from .assets import read_manifest, verify_asset


class FashionCNN(nn.Module):
    """Six convs with publisher-exported batch normalization already folded into weights."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = 1
        for width in (64, 128, 256):
            for _ in range(2):
                layers.extend((nn.Conv2d(previous, width, 3, padding=1), nn.ReLU()))
                previous = width
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Linear(256, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.flatten(self.pool(self.features(images))))


def _validate_graph(source: onnx.ModelProto) -> None:
    graph = source.graph
    expected_ops = ["Conv", "Relu", "Conv", "Relu", "MaxPool"] * 3 + ["GlobalAveragePool", "Flatten", "Gemm"]
    if [node.op_type for node in graph.node] != expected_ops:
        raise ValueError("Source ONNX has an unexpected operation sequence.")
    if len(graph.input) != 1 or len(graph.output) != 1 or len(graph.initializer) != 14:
        raise ValueError("Source ONNX has unexpected inputs, outputs or initializer count.")
    if graph.input[0].name != "images" or graph.output[0].name != "logits":
        raise ValueError("Source ONNX input/output names differ from the pinned architecture.")
    input_shape = graph.input[0].type.tensor_type.shape.dim
    output_shape = graph.output[0].type.tensor_type.shape.dim
    if len(input_shape) != 4 or [d.dim_value for d in input_shape[1:]] != [1, 28, 28]:
        raise ValueError("Source ONNX input shape must be [batch,1,28,28].")
    if len(output_shape) != 2 or output_shape[1].dim_value != 10:
        raise ValueError("Source ONNX output shape must be [batch,10].")
    if graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT or graph.output[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("Source ONNX inputs/outputs must be float32.")
    expected_attributes = {
        "Conv": {"dilations": [1, 1], "group": 1, "kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]},
        "Relu": {},
        "MaxPool": {"ceil_mode": 0, "dilations": [1, 1], "kernel_shape": [2, 2], "pads": [0, 0, 0, 0], "strides": [2, 2]},
        "GlobalAveragePool": {},
        "Flatten": {"axis": 1},
        "Gemm": {"alpha": 1.0, "beta": 1.0, "transB": 1},
    }
    previous = "images"
    used_weights: set[str] = set()
    for node in graph.node:
        if node.domain not in {"", "ai.onnx"} or len(node.output) != 1 or node.input[0] != previous:
            raise ValueError("Source ONNX must use the expected sequential standard-operator graph.")
        expected_input_count = 3 if node.op_type in {"Conv", "Gemm"} else 1
        if len(node.input) != expected_input_count:
            raise ValueError("Source ONNX operation has unexpected inputs.")
        attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if attributes != expected_attributes[node.op_type]:
            raise ValueError(f"Unexpected source ONNX attributes: {node.op_type}")
        used_weights.update(node.input[1:])
        previous = node.output[0]
    if previous != "logits" or used_weights != {tensor.name for tensor in graph.initializer}:
        raise ValueError("Source ONNX initializer or output wiring mismatch.")
    for tensor in graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
            raise ValueError("External tensor data is not accepted for the pinned source model.")
        if tensor.data_type != onnx.TensorProto.FLOAT:
            raise ValueError("Source ONNX tensors must be float32.")
    onnx.checker.check_model(source)


def load_model(path: str | Path) -> FashionCNN:
    """Verify the immutable source bytes, check its graph, copy tensor data to fixed modules."""
    verify_asset(path, read_manifest()["assets"]["model_onnx"])
    source = onnx.load(str(path), load_external_data=False)
    _validate_graph(source)
    tensors = {value.name: numpy_helper.to_array(value).copy() for value in source.graph.initializer}
    model = FashionCNN()
    weighted_nodes = [node for node in source.graph.node if node.op_type in {"Conv", "Gemm"}]
    weighted_layers = [layer for layer in model.features if isinstance(layer, nn.Conv2d)] + [model.classifier]
    with torch.no_grad():
        for node, layer in zip(weighted_nodes, weighted_layers, strict=True):
            for attr, source_name in zip(("weight", "bias"), node.input[1:], strict=True):
                array = tensors[source_name]
                target = getattr(layer, attr)
                if tuple(array.shape) != tuple(target.shape) or not np.isfinite(array).all():
                    raise ValueError(f"Invalid source initializer: {source_name}")
                target.copy_(torch.from_numpy(array))
    model.eval()
    return model
