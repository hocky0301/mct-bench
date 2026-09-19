import numpy as np
import pytest

from mct_bench.runtime import check_parity, predict


def test_prediction_includes_last_partial_batch():
    inputs = np.arange(35).reshape(7, 5)
    batches = []
    def runner(x):
        batches.append(len(x))
        return x * 2
    np.testing.assert_array_equal(predict(runner, inputs, 3), inputs * 2)
    assert batches == [3, 3, 1]


def test_parity_rejects_silent_export_change():
    reference = np.array([[1., 2.]])
    with pytest.raises(AssertionError):
        check_parity(reference, reference + .01)
    with pytest.raises(ValueError):
        check_parity(reference, np.array([[np.nan, 2.]]))
    assert check_parity(reference, reference)["argmax_agreement"] == 1.
