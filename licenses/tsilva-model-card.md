---
license: mit
library_name: onnx
pipeline_tag: image-classification
tags:
- image-classification
- fashion-mnist
- onnx
- onnxruntime
- pytorch
- dlab
datasets:
- zalando-datasets/fashion_mnist
metrics:
- accuracy
---

# Fashion-MNIST CNN Classifier

This repository contains a validation-selected shallow CNN classifier for Fashion-MNIST, trained with [dlab](https://github.com/tsilva/dlab).

## Architecture

![Fashion-MNIST CNN architecture](assets/architecture.png)

## Results

Seed-confirmed validation performance for this recipe, plus the final one-time test-set evaluation performed after checkpoint selection:

| metric | value |
|---|---:|
| validation loss | 0.2650 ± 0.0098 |
| validation accuracy | 95.14% ± 0.19 pp |
| test loss | 0.2791 |
| test accuracy | 94.76% |
| test correct | 9476 / 10000 |

The ONNX model was exported from the best validation-loss checkpoint from the selected recipe. The test metrics were not used to select the checkpoint and were logged in W&B run [`8h2ywarm`](https://wandb.ai/tsilva/dlab/runs/8h2ywarm).

## Model Details

- Dataset: Fashion-MNIST
- Architecture: shallow CNN
- Channels: `[64, 128, 256]`
- Dropout: `0.2`
- Optimizer: Adam
- Learning rate: `0.001`
- Weight decay: `3e-5`
- Scheduler: cosine
- Label smoothing: `0.02`
- Weight averaging: EMA
- Training augmentation: random affine rotation/translation/scale
- Source W&B run: [`qwfq2ihy`](https://wandb.ai/tsilva/dlab/runs/qwfq2ihy)
- Source checkpoint: `044.ckpt`
- Final test W&B run: [`8h2ywarm`](https://wandb.ai/tsilva/dlab/runs/8h2ywarm)

## Input / Output

Use `model.onnx` for code-independent inference.

- Input name: `images`
- Input shape: `[batch, 1, 28, 28]`
- Input dtype: `float32`
- Output name: `logits`
- Output shape: `[batch, 10]`

Preprocessing:

- Convert image to grayscale.
- Resize to `28 x 28`.
- Scale pixel values to `[0, 1]`.
- Normalize with mean `0.2860` and standard deviation `0.3530`.
- Arrange the tensor as channels-first `[batch, 1, 28, 28]`.

## Usage

Install the runtime dependencies:

```bash
pip install huggingface_hub onnxruntime pillow numpy
```

Run inference with the ONNX model:

```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

LABELS = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

model_path = hf_hub_download(
    repo_id="tsilva/fashion-mnist-classifier-cnn",
    filename="model.onnx",
)

image = Image.open("example.png").convert("L").resize((28, 28))
x = np.asarray(image, dtype=np.float32) / 255.0
x = (x - 0.2860) / 0.3530
x = x[None, None, :, :].astype(np.float32)

session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
logits = session.run(["logits"], {"images": x})[0]
prediction = int(logits.argmax(axis=1)[0])

print(prediction, LABELS[prediction])
```

## Labels

Fashion-MNIST labels:

| id | label |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Files

- `model.onnx`: ONNX export of the validation-selected checkpoint, using opset 17. Prefer this file for portable inference.
- `model.ckpt`: PyTorch Lightning checkpoint for the same model. This is code-dependent and mainly useful for PyTorch-based inspection or continued experimentation.
- `config.yaml`: resolved training config.
- `metrics.csv`: training/validation metric history.
- `modeling.py`: minimal PyTorch model definition and checkpoint loader for users who specifically need the checkpoint path.
- `label_mapping.json`: label id to class name mapping.

## Limitations

Most remaining validation errors are visually plausible confusions between related Fashion-MNIST classes, especially shirt/top/pullover/coat and sneaker/ankle boot cases.
