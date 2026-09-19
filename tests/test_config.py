import json

import pytest

from mct_bench.config import Variant, load_variants


@pytest.mark.parametrize("kwargs", [
    {"name": "../bad", "weights_bits": 8},
    {"name": "FP32", "weights_bits": 8},
    {"name": "test", "weights_bits": 3},
    {"name": "test", "weights_bits": True},
    {"name": "test", "weights_bits": 4, "activation_bits": 4},
    {"name": "test", "weights_bits": 4, "bias_correction": "false"},
    {"name": "test", "weights_bits": 4, "error_method": "typo"},
])
def test_bad_variant(kwargs):
    with pytest.raises(ValueError):
        Variant(**kwargs)


@pytest.mark.parametrize("raw", [
    {}, {"variants": []}, {"variants": [], "typo": True},
    {"variants": [{"name": "a", "weights_bits": 4, "unknown": 1}]},
    {"variants": [{"name": "a", "weights_bits": 4}, {"name": "A", "weights_bits": 8}]},
])
def test_bad_config(tmp_path, raw):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_variants(path)


def test_defaults_and_examples():
    assert [v.weights_bits for v in load_variants(None)] == [8, 4]
