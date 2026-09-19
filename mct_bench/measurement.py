"""Classification accuracy and comparable, model-loading-free latency measurements."""

from __future__ import annotations

import random
import time
from collections.abc import Callable

import numpy as np


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Return top-1 accuracy for an N × C score array and N integer labels.

    Inputs must be finite and nonempty. Scores may be logits or probabilities;
    only their ordering matters. Ties use NumPy's first-maximum convention.
    """
    if not isinstance(logits, np.ndarray) or not isinstance(labels, np.ndarray):
        raise TypeError("logits and labels must be numpy arrays")
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must have nonempty shape (samples, classes)")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape (samples,) matching logits")
    if logits.dtype.kind not in "iuf":
        raise TypeError("logits must contain real numeric scores")
    if labels.dtype.kind not in "iu":
        raise TypeError("labels must contain integer class indices")
    if not np.isfinite(logits).all():
        raise ValueError("logits must be finite")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("labels contain an out-of-range class index")
    return float(np.mean(np.argmax(logits, axis=1) == labels))


def _integer(value: int, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def benchmark_sessions(
    sessions: dict[str, Callable[[np.ndarray], object]],
    inputs: list[np.ndarray],
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, dict[str, float | list[float]]]:
    """Time prepared synchronous inference callables on identical prepared inputs.

    Each variant receives ``warmup`` untimed calls, cycling through ``inputs``.
    Each subsequent repeat cycles through the prepared inputs and shuffles
    variant order independently. Thus each variant has ``repeats`` samples.
    Measurements include the callable's complete synchronous inference call;
    model/session construction, preprocessing and warmup are outside the timer.
    The caller must synchronize asynchronous runtimes inside its callable.
    """
    if not isinstance(sessions, dict) or not sessions:
        raise ValueError("sessions must be a nonempty dict of named callables")
    if any(not isinstance(name, str) or not name for name in sessions):
        raise ValueError("session names must be nonempty strings")
    if any(not callable(session) for session in sessions.values()):
        raise TypeError("every session must be callable")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("inputs must be a nonempty list of numpy arrays")
    for batch in inputs:
        if not isinstance(batch, np.ndarray):
            raise TypeError("every input must be a numpy array")
        if batch.size == 0 or batch.ndim == 0:
            raise ValueError("inference inputs must be nonempty arrays with a batch dimension")
        if batch.dtype.kind not in "biuf" or not np.isfinite(batch).all():
            raise ValueError("inference inputs must contain finite real numbers")
    _integer(warmup, "warmup", 0)
    _integer(repeats, "repeats", 1)
    _integer(seed, "seed", 0)

    names = list(sessions)
    rng = random.Random(seed)
    # Warm every runtime before collecting any measurements. Randomized warmup
    # order also avoids giving a fixed variant the last warmup every time.
    for index in range(warmup):
        order = names.copy()
        rng.shuffle(order)
        batch = inputs[index % len(inputs)]
        for name in order:
            sessions[name](batch)

    samples: dict[str, list[float]] = {name: [] for name in names}
    for index in range(repeats):
        batch = inputs[index % len(inputs)]
        order = names.copy()
        rng.shuffle(order)
        for name in order:
            start = time.perf_counter_ns()
            sessions[name](batch)
            elapsed = time.perf_counter_ns() - start
            samples[name].append(elapsed / 1_000_000.0)

    return {
        name: {
            "median_ms": float(np.median(values)),
            "p90_ms": float(np.percentile(values, 90)),
            "mean_ms": float(np.mean(values)),
            "std_ms": float(np.std(values, ddof=0)),
            "samples_ms": values,
        }
        for name, values in samples.items()
    }
