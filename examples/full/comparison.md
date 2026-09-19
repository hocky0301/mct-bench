| Metric | fp32 | w8a8 | w4a8 |
| --- | ---: | ---: | ---: |
| Weight bits (configured) | 32 | 8 | 4 |
| Activation bits (configured) | 32 | 8 | 8 |
| Execution | FP32 ONNX / CPU | MCT fake-quant ONNX / CPU | MCT fake-quant ONNX / CPU |
| Threshold error method | N/A | mse | mse |
| Bias correction | N/A | True | True |
| Calibration images (train) | 0 | 512 | 512 |
| Evaluation images (test) | 10000 | 10000 | 10000 |
| Accuracy (%) | 94.7600 | 94.7600 | 94.2200 |
| Accuracy delta (pp) | +0.0000 | +0.0000 | -0.5400 |
| ONNX file size (bytes) | 4590949 | 4599177 | 4599177 |
| ONNX file size (MiB) | 4.378270 | 4.386117 | 4.386117 |
| Size delta (%) | +0.00 | +0.18 | +0.18 |
| Latency batch size | 1 | 1 | 1 |
| CPU intra-op threads | 1 | 1 | 1 |
| Timed calls per variant | 100 | 100 | 100 |
| Latency median (ms/call) | 3.992271 | 4.042916 | 4.046145 |
| Latency p90 (ms/call) | 4.178625 | 4.236904 | 4.200058 |
| Latency mean (ms/call) | 4.039760 | 4.071628 | 4.077043 |
| Latency std (ms/call) | 0.172413 | 0.109080 | 0.123031 |
| Median latency delta (%) | +0.00 | +1.27 | +1.35 |
| Speedup (baseline / variant) | 1.0000 | 0.9875 | 0.9867 |

Deltas compare each variant with the first (baseline) column. Negative size and latency deltas mean a reduction; speedup = baseline median / variant median. Accuracy deltas are percentage points (pp). N/A denotes division by zero.

MCT fake-quant ONNX represents quantization effects with floating-point operators and tensors. The measured ONNX file size and CPU latency do not establish packed INT8/INT4 storage or native integer-kernel speed. Bit widths describe the configured quantization, not the ONNX tensor storage dtype. All variants use CPUExecutionProvider with ORT_DISABLE_ALL for export fidelity; these are unoptimized graph timings.
