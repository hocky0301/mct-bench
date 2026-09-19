import numpy as np
import pytest

from mct_bench import measurement
from mct_bench.measurement import accuracy, benchmark_sessions


def test_accuracy_known_scores_and_ties():
    scores = np.array([[0.2, 0.8], [0.4, 0.1], [0.5, 0.5], [-2.0, -1.0]])
    labels = np.array([1, 0, 1, 1])
    assert accuracy(scores, labels) == 0.75
    assert accuracy(scores[:1], labels[:1]) == 1.0


@pytest.mark.parametrize(
    "scores, labels, error",
    [
        (np.empty((0, 2)), np.array([], dtype=int), ValueError),
        (np.empty((2, 0)), np.array([0, 0]), ValueError),
        (np.ones(2), np.array([0, 0]), ValueError),
        (np.ones((2, 2)), np.array([0]), ValueError),
        (np.ones((2, 2)), np.array([[0], [0]]), ValueError),
        (np.array([[np.nan, 1.0]]), np.array([0]), ValueError),
        (np.array([[np.inf, 1.0]]), np.array([0]), ValueError),
        (np.ones((1, 2)), np.array([-1]), ValueError),
        (np.ones((1, 2)), np.array([2]), ValueError),
        (np.ones((1, 2)), np.array([1.0]), TypeError),
        (np.ones((1, 2)), np.array([True]), TypeError),
        (np.array([[1j, 2j]]), np.array([0]), TypeError),
        (np.array([["a", "b"]]), np.array([0]), TypeError),
        ([[1, 2]], np.array([0]), TypeError),
    ],
)
def test_accuracy_rejects_invalid_inputs(scores, labels, error):
    with pytest.raises(error):
        accuracy(scores, labels)


def test_timing_counts_exclude_warmup_and_cycle_identical_inputs(monkeypatch):
    calls = []
    now = 0
    timer_reads = 0

    def clock():
        nonlocal timer_reads
        timer_reads += 1
        return now

    def session(name, duration):
        def run(batch):
            nonlocal now
            calls.append((name, int(batch[0])))
            now += duration
        return run

    monkeypatch.setattr(measurement.time, "perf_counter_ns", clock)
    sessions = {"base": session("base", 2_000_000), "ptq": session("ptq", 500_000)}
    batches = [np.array([10]), np.array([20]), np.array([30])]
    results = benchmark_sessions(sessions, batches, warmup=2, repeats=7, seed=13)

    assert timer_reads == 2 * 2 * 7
    assert len(calls) == 2 * (2 + 7)
    timed_calls = calls[4:]
    for index in range(7):
        pair = timed_calls[2 * index:2 * index + 2]
        assert {name for name, _ in pair} == {"base", "ptq"}
        assert [batch for _, batch in pair] == [10 * (index % 3 + 1)] * 2
    assert results["base"] == {
        "median_ms": 2.0, "p90_ms": 2.0, "mean_ms": 2.0,
        "std_ms": 0.0, "samples_ms": [2.0] * 7,
    }
    assert results["ptq"]["samples_ms"] == [0.5] * 7


def test_variant_order_is_reproducible_and_varies(monkeypatch):
    monkeypatch.setattr(measurement.time, "perf_counter_ns", lambda: 0)

    def record(seed):
        calls = []
        sessions = {name: (lambda batch, name=name: calls.append(name)) for name in ["a", "b", "c"]}
        benchmark_sessions(sessions, [np.ones(1)], warmup=0, repeats=12, seed=seed)
        return [tuple(calls[index:index + 3]) for index in range(0, len(calls), 3)]

    first = record(123)
    assert first == record(123)
    assert first != record(456)
    assert len(set(first)) > 1


def test_inference_failure_is_not_reported_as_a_latency():
    def fail(batch):
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        benchmark_sessions({"broken": fail}, [np.ones(1)], warmup=0, repeats=1, seed=0)


@pytest.mark.parametrize(
    "override, error",
    [
        ({"sessions": {}}, ValueError),
        ({"sessions": {"": lambda batch: batch}}, ValueError),
        ({"sessions": {"base": None}}, TypeError),
        ({"inputs": []}, ValueError),
        ({"inputs": [np.array([])]}, ValueError),
        ({"inputs": [np.array(1)]}, ValueError),
        ({"inputs": [[1]]}, TypeError),
        ({"inputs": [np.array([np.nan])]}, ValueError),
        ({"warmup": -1}, ValueError),
        ({"warmup": True}, TypeError),
        ({"repeats": 0}, ValueError),
        ({"repeats": 1.5}, TypeError),
        ({"seed": -1}, ValueError),
    ],
)
def test_benchmark_rejects_invalid_configuration(override, error):
    kwargs = dict(sessions={"base": lambda batch: batch}, inputs=[np.ones(1)], warmup=0, repeats=1, seed=0)
    kwargs.update(override)
    with pytest.raises(error):
        benchmark_sessions(**kwargs)
