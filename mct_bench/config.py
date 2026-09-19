"""Small, strict JSON schema: unsupported settings fail before downloading assets."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Variant:
    name: str
    weights_bits: int
    activation_bits: int = 8
    error_method: str = "mse"
    bias_correction: bool = True

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]{0,63}", self.name):
            raise ValueError("variant name must be a safe identifier (1–64 letters/digits/_/-)")
        if self.name.lower() == "fp32":
            raise ValueError("fp32 is reserved for the baseline")
        if type(self.weights_bits) is not int or self.weights_bits not in (4, 8):
            raise ValueError("weights_bits must be 4 or 8")
        if type(self.activation_bits) is not int or self.activation_bits != 8:
            raise ValueError("activation_bits must be 8: this standard ONNX exporter supports W8A8 / W4A8")
        if self.error_method not in ("mse", "no_clipping"):
            raise ValueError("error_method must be mse or no_clipping")
        if type(self.bias_correction) is not bool:
            raise ValueError("bias_correction must be a JSON boolean")


DEFAULT_VARIANTS = (Variant("w8a8", 8), Variant("w4a8", 4))


def load_variants(path: Path | None) -> list[Variant]:
    if path is None:
        return list(DEFAULT_VARIANTS)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"variants"}:
        raise ValueError('configuration must contain exactly one key: "variants"')
    if not isinstance(raw["variants"], list) or not raw["variants"]:
        raise ValueError("variants must be a nonempty list")
    try:
        variants = [Variant(**v) for v in raw["variants"]]
    except TypeError as exc:
        raise ValueError(f"invalid variant fields: {exc}") from exc
    names = [v.name.casefold() for v in variants]
    if len(set(names)) != len(names):
        raise ValueError("variant names must be unique (case insensitive)")
    return variants


def variant_dict(variant: Variant) -> dict:
    return asdict(variant)
