"""Exact fixed-fit, phase-matched bootstrap calibration of DMM-CUSUM.

Input ``series`` is the chronological Phase I standardized global deviance
S_t^D, before prediction. With fixed estimates and phase-aligned blocks,
resampling this scalar series is exactly equivalent to resampling the complete
network matrices and recomputing their deviances. Innovations are never
resampled. State and prediction history continue across sampled block joins.

Every path has its own uint32 seed and stops only at its first strict crossing.
The numerical guard raises an explicit failure; it is never an observed run
length. Calibration uses common path seeds for all candidate control limits.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
from numba import njit, prange


class BootstrapGuardError(RuntimeError):
    """At least one bootstrap path did not alarm before the numerical guard."""

    def __init__(self, failed_indices, seeds, guard, control_limit):
        self.failed_indices = np.asarray(failed_indices, dtype=np.int64)
        self.failed_seeds = np.asarray(seeds, dtype=np.uint32)[self.failed_indices].copy()
        self.guard = int(guard)
        self.control_limit = float(control_limit)
        super().__init__(
            f"{len(self.failed_indices)} bootstrap path(s) reached guard={guard} "
            f"without an alarm at h={control_limit}; no ARL was calculated"
        )


@njit(cache=True, parallel=True)
def _bootstrap_runs(series, seeds, beta, variance, bmax, k, h,
                    block, templates, start_phase, guard):
    runs = np.full(len(seeds), -1, dtype=np.int64)
    choices = (len(series) - block - start_phase) // templates + 1
    for path in prange(len(seeds)):
        np.random.seed(seeds[path])
        cplus = 0.0
        cminus = 0.0
        prediction_length = 0
        history = np.zeros(max(bmax, 1), dtype=np.float64)
        source_start = 0
        for t in range(guard):
            offset = t % block
            if offset == 0:
                source_start = start_phase + templates * np.random.randint(choices)
            current = series[source_start + offset]
            innovation = current
            for lag in range(prediction_length):
                previous = history[(t - lag - 1) % len(history)]
                innovation -= beta[prediction_length, lag] * previous
            innovation /= math.sqrt(variance[prediction_length])
            cplus = max(0.0, cplus + innovation - k)
            cminus = max(0.0, cminus - innovation - k)
            history[t % len(history)] = current
            state = max(cplus, cminus)
            if state > h:
                runs[path] = t + 1
                break
            if state == 0.0:
                prediction_length = 0
            else:
                prediction_length = min(prediction_length + 1, bmax)
    return runs


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _path_seeds(seeds):
    values = np.asarray(seeds)
    if values.ndim != 1 or values.size == 0 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("seeds must be a nonempty one-dimensional integer array")
    if np.any(values < 0) or np.any(values > np.iinfo(np.uint32).max):
        raise ValueError("seeds must be uint32 values")
    values = np.ascontiguousarray(values, dtype=np.uint32)
    if np.unique(values).size != values.size:
        raise ValueError("path seeds must be unique")
    return values


def _inputs(series, seeds, beta, variance, bmax, k, h, block, templates,
            start_phase, guard):
    series = np.ascontiguousarray(series, dtype=np.float64)
    beta = np.ascontiguousarray(beta, dtype=np.float64)
    variance = np.ascontiguousarray(variance, dtype=np.float64)
    bmax = _integer(bmax, "bmax")
    block = _integer(block, "block", 1)
    templates = _integer(templates, "templates", 1)
    start_phase = _integer(start_phase, "start_phase")
    guard = _integer(guard, "guard", 1)
    if series.ndim != 1 or not len(series) or not np.isfinite(series).all():
        raise ValueError("series must be a finite one-dimensional Phase I sequence")
    if block % templates:
        raise ValueError("block must contain complete template cycles")
    if start_phase >= templates:
        raise ValueError("start_phase must be between zero and templates - 1")
    if len(series) < block + start_phase:
        raise ValueError("Phase I series has no noncircular block of the requested phase")
    if beta.shape != (bmax + 1, bmax) or not np.isfinite(beta).all():
        raise ValueError("beta must be a finite (bmax + 1, bmax) coefficient array")
    if variance.shape != (bmax + 1,) or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("variance must contain bmax + 1 finite positive values")
    if not math.isfinite(k) or k <= 0:
        raise ValueError("k must be finite and positive")
    if not math.isfinite(h) or h < 0:
        raise ValueError("h must be finite and nonnegative")
    return (series, _path_seeds(seeds), beta, variance, bmax, float(k), float(h),
            block, templates, start_phase, guard)


def bootstrap_runs(series, seeds, beta, variance, bmax, k, h, block=32,
                   templates=4, start_phase=0, guard=10_000_000):
    """Return complete first-alarm times, or raise :class:`BootstrapGuardError`.

    Valid source starts are ``start_phase + j * templates`` with the entire
    block inside Phase I. The first Phase I observation has phase zero.
    ``beta[d, :d]`` predicts the current S from its d most recent bootstrap
    observations, nearest lag first. Each path initializes both CUSUMs and the
    capped prediction length to zero. The RNG is NumPy/Numba's legacy MT19937,
    separately initialized from each path seed.
    """
    args = _inputs(series, seeds, beta, variance, bmax, k, h, block, templates,
                   start_phase, guard)
    runs = _bootstrap_runs(*args)
    failed = np.flatnonzero(runs < 0)
    if failed.size:
        raise BootstrapGuardError(failed, args[1], args[-1], args[6])
    return runs


def _unique_seeds(seed, reps):
    """Keep RNG draw order while replacing the rare repeated uint32 draw."""
    rng = np.random.default_rng(seed)
    values = []
    seen = set()
    while len(values) < reps:
        candidate = int(rng.integers(0, 2**32, dtype=np.uint64))
        if candidate not in seen:
            values.append(candidate)
            seen.add(candidate)
    return np.asarray(values, dtype=np.uint32)


def calibrate(series, fit, target, k, reps, seed, block=32, templates=4,
              start_phase=0, *, path_seeds=None, guard=10_000_000,
              tolerance=0.002, max_iterations=40):
    """Calibrate h using fixed-fit bootstrap paths and complete alarm times.

    ``fit`` follows ``run_simulation.SpringFit``. ``tolerance`` is a relative
    target tolerance, so 0.002 means 0.4 observations when target=200. When
    discrete empirical ties prevent that tolerance, return the closest
    evaluated limit and report the error and ``within_tolerance=False``.
    Guard failures propagate; calibration never averages censored substitutes.
    """
    reps = _integer(reps, "reps", 1)
    max_iterations = _integer(max_iterations, "max_iterations", 1)
    if not math.isfinite(target) or target < 1:
        raise ValueError("target ARL must be finite and >= 1")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    seeds = _unique_seeds(seed, reps) if path_seeds is None else _path_seeds(path_seeds)
    if len(seeds) != reps:
        raise ValueError("path_seeds length must equal reps")
    bmax = _integer(fit.bmax, "fit.bmax")
    beta = np.zeros((bmax + 1, bmax), dtype=np.float64)
    if len(fit.beta) != bmax + 1:
        raise ValueError("fit.beta must have one row for each prediction length")
    for order in range(1, bmax + 1):
        if len(fit.beta[order]) != order:
            raise ValueError("fit.beta rows must contain exactly order coefficients")
        beta[order, :order] = fit.beta[order]
    variance = np.asarray(fit.variance, dtype=np.float64)
    cache: dict[float, np.ndarray] = {}
    evaluations: list[dict[str, Any]] = []
    permitted_error = float(tolerance * target)

    def evaluate(h):
        h = float(h)
        if h not in cache:
            runs = bootstrap_runs(series, seeds, beta, variance, bmax, k, h,
                                  block, templates, start_phase, guard)
            cache[h] = runs
            evaluations.append({"h": h, "arl": float(runs.mean()),
                                "max_run": int(runs.max())})
        return float(cache[h].mean())

    low = 0.0
    low_mean = evaluate(low)
    if low_mean > target:
        raise ValueError(
            f"target ARL {target} is below the empirical minimum {low_mean} at h=0"
        )
    high = 1.0
    high_mean = evaluate(high)
    bracket_expansions = 0
    while high_mean < target:
        low, low_mean = high, high_mean
        high *= 1.5
        bracket_expansions += 1
        high_mean = evaluate(high)

    iterations = 0
    best_h = min(cache, key=lambda h: (abs(float(cache[h].mean()) - target), h))
    for _ in range(max_iterations):
        if abs(float(cache[best_h].mean()) - target) <= permitted_error:
            break
        midpoint = (low + high) / 2.0
        if midpoint == low or midpoint == high:
            break
        mean = evaluate(midpoint)
        iterations += 1
        best_h = min(cache, key=lambda h: (abs(float(cache[h].mean()) - target), h))
        if mean < target:
            low, low_mean = midpoint, mean
        else:
            high, high_mean = midpoint, mean

    runs = cache[best_h].copy()
    achieved = float(runs.mean())
    error = abs(achieved - target)
    metadata = {
        "target_arl": float(target), "achieved_arl": achieved,
        "absolute_error": error, "relative_error": error / target,
        "relative_tolerance": float(tolerance),
        "within_tolerance": bool(error <= permitted_error),
        "selected_h": float(best_h), "reps": reps, "seed": int(seed),
        "path_seeds_supplied": path_seeds is not None,
        "path_seeds_unique": True,
        "path_seed_prefix": seeds[:8].tolist(),
        "path_seed_sha256": hashlib.sha256(seeds.astype("<u4").tobytes()).hexdigest(),
        "rng": "per-path uint32 seed; NumPy/Numba MT19937",
        "common_random_numbers": True, "fixed_phase1_fit": True,
        "resampled_statistic": "chronological preprediction S_t^D",
        "block": int(block), "templates": int(templates),
        "start_phase": int(start_phase), "noncircular": True,
        "bmax": bmax, "k": float(k), "guard": int(guard),
        "failed_paths": 0, "censored_paths": 0,
        "bracket_expansions": bracket_expansions,
        "bisection_iterations": iterations,
        "final_bracket": [float(low), float(high)],
        "evaluations": evaluations,
    }
    return float(best_h), runs, metadata
