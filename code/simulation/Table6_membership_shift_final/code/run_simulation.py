#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible simulation study for the revised ML-DCMM monitoring paper.

The program implements the data-generating mechanism and the proposed
spring-length two-sided CUSUM described in Sections 2--3 of the manuscript.
Four published network-chart ideas are implemented as transparent adaptations
to undirected count-weighted networks; they are not claimed to be exact source-
code reproductions.  Every method is calibrated under the same IC paths to the
same nominal ARL0 before OC comparisons are made.

Outputs are written to the directory supplied by ``--outdir``. A fixed master
seed, common random numbers and machine-readable settings make every reported
number reproducible. Window-based experiments retain the start indices needed
for time-clustered Monte Carlo standard errors.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = Path(__file__).with_name("mldcmm_estimation.py")
OUT_DEFAULT = ROOT / "simulation/results/rerun"
METHODS = [
    "Proposed ML-DCMM",
    "Gravity-WLRT",
    "DCSBM-EWMA",
    "SS-MCUSUM",
    "TEN-GNARX-CUSUM",
]
EPS = 1.0e-10


def load_base() -> Any:
    spec = importlib.util.spec_from_file_location("legacy_sim_components", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base()


@dataclass
class Config:
    seed: int = 20260913
    target_arl0: float = 200.0
    templates: int = 4
    # Reference value for a one-SD shift in the standardized prediction
    # residual, k = delta_U / 2.
    k_cusum: float = 0.50
    bmax_cap: int = 8
    block_len: int = 8
    max_len: int = 800
    reps: int = 10000
    calibration_reps: int = 10000
    pool_len: int = 24000
    phase_reps: int = 3
    phi: float = 0.50
    sigma_g: float = 0.10
    sigma_v: float = 0.10
    mean_edge: float = 20.0
    block_scale: float = 1.0
    mldcmm_iters: int = 10
    mldcmm_b_steps: int = 4
    lambda_gravity: float = 0.10
    lambda_dcsbm: float = 0.20
    k_ss: float = 0.10
    ten_local_quantile: float = 0.995
    ten_network_k: float = 0.50
    # Malinovskaya et al. (2026) use a 5% edgewise residual chart.  The
    # corresponding critical multiplier is kept fixed while the network-level
    # cumulative-intensity threshold is calibrated to the common ARL0.
    ten_zeta: float = 1.96
    ten_global_fraction: float = 0.20
    alpha_loc: float = 0.10
    alpha_theta: float = 0.025
    alpha_z: float = 0.025
    diagnosis_paths: int = 3000
    diagnosis_bank_paths: int = 30000
    diagnosis_min_window: int = 8
    robustness_phase_reps: int = 10
    robustness_inner_pools: int = 4
    quick: bool = False


@dataclass
class SpringFit:
    template_mean: list[float]
    template_sd: list[float]
    bmax: int
    beta: list[list[float]]
    variance: list[float]


@dataclass
class FittedMethod:
    name: str
    chart: str
    parameter: float | None
    details: dict[str, Any]
    h: float | None = None


def stable_seed(master: int, *parts: int) -> int:
    value = int(master) & ((1 << 63) - 1)
    for p in parts:
        value = (value * 6364136223846793005 + int(p) + 1442695040888963407) & ((1 << 63) - 1)
    return value


def template_sequence(length: int, offset: int = 0, templates: int = 4) -> np.ndarray:
    return (np.arange(offset, offset + length) % templates).astype(np.int16)


def build_case(case_id: str, cfg: Config) -> Any:
    """Generate one fixed ML-DCMM population for each simulation case."""
    if case_id not in {"I", "II", "III"}:
        raise ValueError(case_id)
    n = 15 if case_id == "I" else 30
    k = 3
    alpha_main, alpha_off = ((3.2, 0.8) if case_id != "III" else (1.2, 0.8))
    case_code = sum((i + 1) * ord(ch) for i, ch in enumerate(case_id))
    rng = np.random.default_rng(stable_seed(cfg.seed, case_code, 11))

    labels = np.arange(n) % k
    rng.shuffle(labels)
    pure_n = max(k, int(round(0.20 * n)))
    pure = np.zeros(n, dtype=bool)
    for g in range(k):
        pure[np.where(labels == g)[0][0]] = True
    remaining = np.where(~pure)[0]
    rng.shuffle(remaining)
    pure[remaining[: max(0, pure_n - int(pure.sum()))]] = True

    z = np.zeros((n, k), dtype=float)
    for i in range(n):
        if pure[i]:
            z[i, labels[i]] = 1.0
        else:
            alpha = np.full(k, alpha_off)
            alpha[labels[i]] = alpha_main
            z[i] = rng.dirichlet(alpha)

    sigma_theta = 0.70
    theta_raw = np.exp(sigma_theta * rng.normal(size=n) - 0.5 * sigma_theta**2)
    theta = n * theta_raw / theta_raw.sum()
    b = np.array([
        [1.00, 0.30 * cfg.block_scale, 0.20 * cfg.block_scale],
        [0.30 * cfg.block_scale, 1.00, 0.25 * cfg.block_scale],
        [0.20 * cfg.block_scale, 0.25 * cfg.block_scale, 1.00],
    ])
    q = np.array([0.60, 0.80, 1.40, 1.20])
    pi, pj = np.triu_indices(n, 1)
    structural = theta[pi] * theta[pj] * (z @ b @ z.T)[pi, pj]
    rho = cfg.mean_edge / max(float(structural.mean()), EPS)
    b_templates = np.stack([rho * qs * b for qs in q], axis=0)
    mu = base.compute_mu_templates(theta, z, b_templates, pi, pj)
    return base.CaseParams(case_id, n, k, pi, pj, labels, z, theta, b_templates, mu)


def generate_network_path(
    case: Any,
    length: int,
    cfg: Config,
    seed: int,
    mu_templates: np.ndarray | None = None,
    offset: int = 0,
    compound_r: float = 0.0,
    missing_p: float = 0.0,
    couple_to: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Generate a continuous correlated network path from the revised DGP."""
    rng = np.random.default_rng(seed)
    templates = template_sequence(length, offset, cfg.templates)
    mu0 = case.mu_templates if mu_templates is None else np.asarray(mu_templates)

    x = np.empty(length, dtype=float)
    v = np.empty((case.n, length), dtype=float)
    x[0] = rng.normal(0.0, cfg.sigma_g)
    v[:, 0] = rng.normal(0.0, cfg.sigma_v, size=case.n)
    ex = cfg.sigma_g * math.sqrt(1.0 - cfg.phi**2)
    ev = cfg.sigma_v * math.sqrt(1.0 - cfg.phi**2)
    for t in range(1, length):
        x[t] = cfg.phi * x[t - 1] + rng.normal(0.0, ex)
        v[:, t] = cfg.phi * v[:, t - 1] + rng.normal(0.0, ev, size=case.n)
    center = 0.5 * (cfg.sigma_g**2 + 2.0 * cfg.sigma_v**2)
    latent = np.exp(x[None, :] + v[case.pi] + v[case.pj] - center)
    lam = np.maximum(mu0[:, templates] * latent, EPS)

    if couple_to is not None and compound_r == 0.0:
        reference = np.asarray(couple_to)
        lam_reference = np.maximum(reference[:, templates] * latent, EPS)
        base_counts = rng.poisson(lam_reference)
        ratio = lam / lam_reference
        counts = base_counts.astype(float)
        up = ratio >= 1.0
        if np.any(up):
            counts[up] += rng.poisson(np.maximum(lam[up] - lam_reference[up], 0.0))
        down = ~up
        if np.any(down):
            counts[down] = rng.binomial(base_counts[down], np.clip(ratio[down], 0.0, 1.0))
    elif compound_r > 0.0:
        arrivals = rng.poisson(lam / (1.0 + 2.0 * compound_r))
        triples = rng.binomial(arrivals, compound_r)
        counts = (arrivals + 2 * triples).astype(float)
    else:
        counts = rng.poisson(lam).astype(float)

    mask = None
    if missing_p > 0.0:
        mask = rng.random(counts.shape) >= missing_p
        counts = np.where(mask, counts, 0.0)
    return counts, templates, mask


def poisson_deviance(counts: np.ndarray, mean: np.ndarray) -> np.ndarray:
    mu = np.maximum(mean, EPS)
    term = np.where(counts > 0.0, counts * np.log(np.maximum(counts, EPS) / mu), 0.0)
    return 2.0 * (term - (counts - mu))


def raw_global_deviance(counts: np.ndarray, templates: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return poisson_deviance(counts, mu[:, templates]).sum(axis=0)


def fit_spring(raw: np.ndarray, templates: np.ndarray, cfg: Config) -> tuple[SpringFit, np.ndarray]:
    means = np.zeros(cfg.templates)
    sds = np.ones(cfg.templates)
    standardized = np.zeros_like(raw, dtype=float)
    for s in range(cfg.templates):
        take = templates == s
        means[s] = float(raw[take].mean())
        sds[s] = max(float(raw[take].std(ddof=1)), 1.0e-6)
        standardized[take] = (raw[take] - means[s]) / sds[s]

    centered = standardized - standardized.mean()
    gamma = np.array([
        float(np.dot(centered[q:], centered[: len(centered) - q]) / len(centered))
        for q in range(cfg.bmax_cap + 1)
    ])
    acf = np.abs(gamma[1:] / max(gamma[0], EPS))
    cutoff = 1.96 / math.sqrt(max(len(centered), 1))
    significant = np.where(acf > cutoff)[0]
    bmax = int(min(cfg.bmax_cap, significant[-1] + 1 if len(significant) else 1))
    bmax = max(bmax, 1)

    beta: list[list[float]] = [[]]
    variance = [max(float(gamma[0]), 1.0e-6)]
    for d in range(1, bmax + 1):
        g = gamma[1 : d + 1]
        toeplitz = np.fromfunction(lambda i, j: gamma[np.abs(i - j).astype(int)], (d, d), dtype=int)
        ridge = max(1.0e-6, 1.0e-4 * float(np.trace(toeplitz)) / d)
        mat = toeplitz + ridge * np.eye(d)
        coef = np.linalg.solve(mat, g)
        pred_var = max(float(gamma[0] - g @ coef), 1.0e-4)
        beta.append(coef.tolist())
        variance.append(pred_var)
    fit = SpringFit(means.tolist(), sds.tolist(), bmax, beta, variance)
    return fit, standardized


def apply_spring_standardization(raw: np.ndarray, templates: np.ndarray, fit: SpringFit) -> np.ndarray:
    return (raw - np.asarray(fit.template_mean)[templates]) / np.maximum(
        np.asarray(fit.template_sd)[templates], EPS
    )


def spring_states(
    s_pool: np.ndarray,
    indices: np.ndarray,
    fit: SpringFit,
    k: float,
    return_meta: bool = False,
) -> Any:
    """Two-sided CUSUM with the state-dependent spring-length predictor."""
    reps, horizon = indices.shape
    sequence = np.asarray(s_pool, dtype=np.float32)[indices]
    cplus = np.zeros(reps, dtype=np.float32)
    cminus = np.zeros(reps, dtype=np.float32)
    spring = np.zeros(reps, dtype=np.int16)
    states = np.empty((reps, horizon), dtype=np.float32)
    spring_path = np.empty((reps, horizon), dtype=np.int16) if return_meta else None
    branch_path = np.empty((reps, horizon), dtype=np.int8) if return_meta else None
    beta = [np.asarray(x, dtype=np.float32) for x in fit.beta]
    variance = np.asarray(fit.variance, dtype=np.float32)

    for t in range(horizon):
        current = sequence[:, t].copy()
        for d in range(1, min(fit.bmax, t) + 1):
            rows = np.where(spring == d)[0]
            if rows.size:
                lagged = sequence[rows[:, None], t - np.arange(1, d + 1)[None, :]]
                current[rows] -= lagged @ beta[d]
        current /= np.sqrt(variance[np.minimum(spring, len(variance) - 1)])
        cplus = np.maximum(0.0, cplus + current - k)
        cminus = np.maximum(0.0, cminus - current - k)
        states[:, t] = np.maximum(cplus, cminus)
        if return_meta:
            branch_path[:, t] = (cplus > cminus).astype(np.int8) - (cminus > cplus).astype(np.int8)
        active = states[:, t] > 0.0
        spring = np.where(active, np.minimum(spring + 1, fit.bmax), 0).astype(np.int16)
        if return_meta:
            spring_path[:, t] = spring
    if return_meta:
        return states, spring_path, branch_path
    return states


def contiguous_indices(rng: np.random.Generator, pool_len: int, reps: int, horizon: int) -> np.ndarray:
    if pool_len <= horizon + 1:
        raise ValueError("pool_len must exceed horizon")
    starts = rng.integers(0, pool_len - horizon, size=reps, endpoint=False)
    return starts[:, None] + np.arange(horizon, dtype=np.int32)[None, :]


def run_lengths(states: np.ndarray, h: float) -> np.ndarray:
    hit = states > h
    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1) + 1
    return np.where(any_hit, first, states.shape[1] + 1).astype(float)


def choose_limit(states: np.ndarray, target: float) -> tuple[float, np.ndarray]:
    lo, hi = 0.0, max(float(states.max()), 1.0)
    best_h, best_rl = hi, run_lengths(states, hi)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        rl = run_lengths(states, mid)
        if abs(float(rl.mean()) - target) < abs(float(best_rl.mean()) - target):
            best_h, best_rl = mid, rl
        if float(rl.mean()) < target:
            lo = mid
        else:
            hi = mid
    return float(best_h), best_rl


def legacy_config(cfg: Config) -> Any:
    return base.SimConfig(
        target_arl0=cfg.target_arl0,
        reps=cfg.reps,
        calibration_reps=cfg.calibration_reps,
        pool_len=cfg.pool_len,
        max_len=cfg.max_len,
        block_len=cfg.block_len,
        seed=cfg.seed,
        templates=cfg.templates,
        k_cusum=cfg.k_cusum,
        k_ss=cfg.k_ss,
        lambda_gravity=cfg.lambda_gravity,
        lambda_dcsbm=cfg.lambda_dcsbm,
        phi=cfg.phi,
        eta_sd=cfg.sigma_g,
        avg_edge_mean=cfg.mean_edge,
        mldcmm_iters=cfg.mldcmm_iters,
        mldcmm_b_steps=cfg.mldcmm_b_steps,
        ten_local_quantile=cfg.ten_local_quantile,
        n_i=15,
        n_ii=30,
        n_iii=30,
    )


def quick_network_features(counts: np.ndarray, case: Any) -> np.ndarray:
    """Count-weighted graph summaries used by the adapted SS-MCUSUM."""
    weights = np.log1p(np.asarray(counts, dtype=float))
    degree = np.zeros((case.n, counts.shape[1]), dtype=float)
    for e, (i, j) in enumerate(zip(case.pi, case.pj)):
        degree[int(i)] += weights[e]
        degree[int(j)] += weights[e]
    mean_weight = weights.mean(axis=0)
    zero_fraction = (counts == 0.0).mean(axis=0)
    degree_cv = degree.std(axis=0) / np.maximum(degree.mean(axis=0), EPS)
    max_degree_ratio = degree.max(axis=0) / np.maximum(degree.mean(axis=0), EPS)
    two_star = np.mean(degree * degree, axis=0)
    return np.column_stack([
        mean_weight, zero_fraction, degree_cv, max_degree_ratio, two_star,
    ])


def weighted_gravity_wlrt(
    counts: np.ndarray,
    templates: np.ndarray,
    mu0: np.ndarray,
    case: Any,
    smoothing: float,
) -> np.ndarray:
    """Weighted-likelihood gravity statistic with weighting before estimation.

    ``counts`` may be one path with shape (edges, time) or a batch with shape
    (edges, paths, time). At each time, both the observed sufficient network
    and its IC expectation are updated with the same exponential weights. The
    gravity MLE is then fitted to the weighted observed network and compared
    with the fixed IC expectation through a Poisson likelihood ratio.
    """
    values = np.asarray(counts, dtype=float)
    single_path = values.ndim == 2
    if single_path:
        values = values[:, None, :]
    if values.ndim != 3:
        raise ValueError("counts must have shape (edges, time) or (edges, paths, time)")
    templates = np.asarray(templates, dtype=int)
    if values.shape[2] != len(templates):
        raise ValueError("templates and counts must have the same time length")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("smoothing must lie in (0, 1]")

    paths, length = values.shape[1], values.shape[2]
    baseline = np.asarray(mu0, dtype=float)
    if baseline.ndim != 2:
        raise ValueError("mu0 must have shape (edges, templates)")
    if templates.min(initial=0) < 0 or templates.max(initial=0) >= baseline.shape[1]:
        raise ValueError("template index is outside the columns of mu0")

    # The original WLRT initializes its weighted network with the Phase I
    # mean as a pseudo-observation.  With periodic network templates, keep one
    # such recursion per template and update only the state matching the
    # arriving network.  This prevents deterministic template differences
    # (and template-opposed changes) from being averaged across unlike layers.
    weighted_counts = np.repeat(baseline[:, None, :], paths, axis=1)
    out = np.empty((paths, length), dtype=float)
    retain = 1.0 - smoothing
    for t in range(length):
        s = int(templates[t])
        weighted_counts[:, :, s] = (
            retain * weighted_counts[:, :, s] + smoothing * values[:, :, t]
        )
        current = weighted_counts[:, :, s]
        degree = np.zeros((case.n, paths), dtype=float)
        np.add.at(degree, case.pi, current)
        np.add.at(degree, case.pj, current)
        products = degree[case.pi] * degree[case.pj]
        total = current.sum(axis=0)
        fitted = products * (
            total / np.maximum(products.sum(axis=0), EPS)
        )[None, :]
        base_deviance = poisson_deviance(current, baseline[:, s, None]).sum(axis=0)
        fitted_deviance = poisson_deviance(current, fitted).sum(axis=0)
        out[:, t] = np.maximum(base_deviance - fitted_deviance, 0.0)
    return out[0] if single_path else out


def fit_methods(counts: np.ndarray, templates: np.ndarray, case: Any, cfg: Config) -> dict[str, FittedMethod]:
    old = legacy_config(cfg)
    fitted: dict[str, FittedMethod] = {}

    mu_hat, z_hat = base.fit_mldcmm_mu(counts, templates, case, old)
    raw = raw_global_deviance(counts, templates, mu_hat)
    spring, _ = fit_spring(raw, templates, cfg)
    fitted[METHODS[0]] = FittedMethod(
        METHODS[0], "spring-cusum", cfg.k_cusum,
        {"mu": mu_hat, "z": z_hat, "spring": spring},
    )

    mu_gravity = base.fit_gravity_mu(counts, templates, case, old)
    raw_gravity = weighted_gravity_wlrt(
        counts, templates, mu_gravity, case, cfg.lambda_gravity,
    )
    gravity_scale, _ = base.fit_scalar_transform(raw_gravity, templates, old)
    fitted[METHODS[1]] = FittedMethod(
        METHODS[1], "scalar-ewma", cfg.lambda_gravity,
        {"mu": mu_gravity, "scale": gravity_scale},
    )

    labels = base.estimate_hard_labels_from_phase(counts, templates, case, old)
    dcsbm_x = base.dcsbm_feature_series(counts, labels, case)
    dcsbm_scale, _ = base.fit_moving_range_standardizer(dcsbm_x)
    fitted[METHODS[2]] = FittedMethod(
        METHODS[2], "vector-ewma", cfg.lambda_dcsbm,
        {"labels": labels, "scale": dcsbm_scale},
    )

    ss_x = quick_network_features(counts, case)
    ss_scale, ss_z = base.fit_vector_standardizer(ss_x)
    fitted[METHODS[3]] = FittedMethod(
        METHODS[3], "ss-mcusum", cfg.k_ss,
        {"scale": ss_scale},
    )

    ten = base.fit_ten_model(counts, templates, case, old)
    ten["phase1_length"] = int(counts.shape[1])
    ten_phase_inc = base.ten_increment_series(counts, templates, case, ten)
    ten_local = np.zeros(ten_phase_inc.shape[0], dtype=float)
    ten_fraction = np.zeros(ten_phase_inc.shape[1], dtype=float)
    ten_ucl = np.asarray(ten["local_ucl"], dtype=float)
    for t in range(ten_phase_inc.shape[1]):
        ten_local = np.maximum(0.0, ten_local + ten_phase_inc[:, t])
        ten_signaled = ten_local > ten_ucl
        ten_fraction[t] = float(np.mean(ten_signaled))
        ten_local[ten_signaled] = 0.0
    ten["fraction_mean"] = float(np.mean(ten_fraction))
    if len(ten_fraction) > 1:
        ten_mr = float(np.mean(np.abs(np.diff(ten_fraction))))
        ten["fraction_sd"] = max(math.sqrt(math.pi) * ten_mr / 2.0, 1.0 / counts.shape[0])
    else:
        ten["fraction_sd"] = 1.0 / counts.shape[0]
    fitted[METHODS[4]] = FittedMethod(
        METHODS[4], "ten-cusum", cfg.ten_network_k, {"ten": ten},
    )
    return fitted


def method_input(
    fitted: FittedMethod,
    counts: np.ndarray,
    templates: np.ndarray,
    case: Any,
    cfg: Config,
) -> Any:
    d = fitted.details
    if fitted.name == METHODS[0]:
        raw = raw_global_deviance(counts, templates, np.asarray(d["mu"]))
        return apply_spring_standardization(raw, templates, d["spring"])
    if fitted.name == METHODS[1]:
        raw = base.raw_gravity_wlrt(counts, templates, np.asarray(d["mu"]), case)
        return base.apply_scalar_transform(raw, templates, d["scale"])
    if fitted.name == METHODS[2]:
        x = base.dcsbm_feature_series(counts, np.asarray(d["labels"]), case)
        return base.apply_vector_standardizer(x, d["scale"])
    if fitted.name == METHODS[3]:
        x = quick_network_features(counts, case)
        z = base.apply_vector_standardizer(x, d["scale"])
        bits = (z > np.asarray(d["medians"])).astype(np.int16)
        return (bits * (1 << np.arange(bits.shape[1]))).sum(axis=1).astype(np.int16)
    if fitted.name == METHODS[4]:
        return base.ten_increment_series(counts, templates, case, d["ten"])
    raise ValueError(fitted.name)


def chart_states(fitted: FittedMethod, inputs: Any, indices: np.ndarray, cfg: Config, return_meta: bool = False) -> Any:
    if fitted.name == METHODS[0]:
        return spring_states(inputs, indices, fitted.details["spring"], cfg.k_cusum, return_meta)
    if fitted.name == METHODS[1]:
        values = np.asarray(inputs, dtype=np.float32)[indices]
        state = np.zeros(indices.shape[0], dtype=np.float32)
        out = np.empty(indices.shape, dtype=np.float32)
        lam = cfg.lambda_gravity
        for t in range(indices.shape[1]):
            state = lam * values[:, t] + (1.0 - lam) * state
            out[:, t] = np.abs(state)
        return out
    if fitted.name == METHODS[2]:
        return base.vector_ewma_states(np.asarray(inputs), indices, cfg.lambda_dcsbm)
    if fitted.name == METHODS[3]:
        return base.ss_cusum_states(np.asarray(inputs), np.asarray(fitted.details["f0"]), indices, cfg.k_ss)
    if fitted.name == METHODS[4]:
        increments = np.asarray(inputs, dtype=np.float32)
        values = increments[:, indices].transpose(1, 0, 2)
        q = np.zeros((indices.shape[0], increments.shape[0]), dtype=np.float32)
        q_min = np.zeros_like(q)
        q_max = np.zeros_like(q)
        running_score = np.zeros_like(q)
        out = np.empty(indices.shape, dtype=np.float32)
        phase1_length = int(fitted.details["ten"]["phase1_length"])
        root_m = math.sqrt(float(phase1_length))
        required = max(
            1,
            min(
                increments.shape[0],
                int(math.ceil(cfg.ten_global_fraction * increments.shape[0])),
            ),
        )
        order_index = increments.shape[0] - required
        for t in range(indices.shape[1]):
            q = q + values[:, :, t]
            distance = np.maximum(np.abs(q - q_min), np.abs(q - q_max))
            k = t + 1.0
            boundary_scale = root_m * (1.0 + k / phase1_length)
            running_score = np.maximum(running_score, distance / boundary_scale)
            out[:, t] = np.partition(running_score, order_index, axis=1)[:, order_index]
            q_min = np.minimum(q_min, q)
            q_max = np.maximum(q_max, q)
        return out
    raise ValueError(fitted.name)


def summarize_rl(
    values: np.ndarray,
    starts: np.ndarray | None = None,
    block_width: int | None = None,
    censor_point: float | None = None,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    naive_se = float(values.std(ddof=1) / math.sqrt(len(values)))
    se = naive_se
    clusters = 0
    if starts is not None and block_width is not None:
        starts = np.asarray(starts, dtype=int)
        group = starts // int(block_width)
        unique = np.unique(group)
        if len(unique) > 1:
            centered = values - values.mean()
            cluster_sums = np.array([centered[group == g].sum() for g in unique])
            # One-way cluster-robust standard error for the overall mean.
            se = float(math.sqrt(len(unique) / (len(unique) - 1.0) * np.sum(cluster_sums**2) / len(values) ** 2))
            clusters = int(len(unique))
    return {
        "arl": float(values.mean()),
        "se": se,
        "naive_se": naive_se,
        "sd": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "censored_rate": float(np.mean(values > censor_point)) if censor_point is not None else math.nan,
        "n": int(len(values)),
        "se_clusters": clusters,
    }


def calibration_and_validation(
    fitted: FittedMethod,
    cal_counts: np.ndarray,
    cal_templates: np.ndarray,
    val_counts: np.ndarray,
    val_templates: np.ndarray,
    case: Any,
    cfg: Config,
    seed: int,
    n_cal: int,
    n_val: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    cal_input = method_input(fitted, cal_counts, cal_templates, case, cfg)
    val_input = method_input(fitted, val_counts, val_templates, case, cfg)
    cal_len = cal_input.shape[1] if fitted.name == METHODS[4] else len(cal_input)
    val_len = val_input.shape[1] if fitted.name == METHODS[4] else len(val_input)
    idx_cal = contiguous_indices(np.random.default_rng(seed), cal_len, n_cal, cfg.max_len)
    idx_val = contiguous_indices(np.random.default_rng(seed + 1), val_len, n_val, cfg.max_len)
    states_cal = chart_states(fitted, cal_input, idx_cal, cfg)
    h, _ = choose_limit(states_cal, cfg.target_arl0)
    states_val = chart_states(fitted, val_input, idx_val, cfg)
    return h, run_lengths(states_val, h), idx_val


def select_source_nodes(case: Any) -> dict[str, Any]:
    order = np.argsort(case.theta)
    single = int(order[len(order) // 2])
    # Mixed-direction activity changes affect about 20% of the nodes. Sampling
    # two nodes from each activity stratum avoids making the result depend only
    # on either low-volume or high-volume nodes.
    stratum_width = max(len(order) // 3, 1)
    multi: list[int] = []
    for start in (0, stratum_width, 2 * stratum_width):
        stop = min(start + stratum_width, len(order))
        center = (start + stop) // 2
        take = order[max(start, center - 1) : min(stop, center + 1)]
        multi.extend(int(x) for x in take)
    middle = len(order) // 2
    multi_diagnosis = [int(x) for x in order[max(0, middle - 2) : middle + 2]]
    entropy = -np.sum(case.z * np.log(np.maximum(case.z, EPS)), axis=1)
    z_source = int(np.argmax(entropy))
    target = int(np.argmin(case.z[z_source]))
    decrease = int(order[max(0, int(0.85 * len(order)) - 1)])
    return {"single": single, "decrease": decrease, "multi": multi,
            "multi_diagnosis": multi_diagnosis,
            "z_source": z_source, "z_target": target}


def theta_shift_mu(case: Any, delta: float, pattern: str, nodes: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
    theta = case.theta.copy()
    if pattern == "Single increase":
        affected = [nodes["single"]]
        signs = [1.0]
    elif pattern == "Single decrease":
        affected = [nodes["decrease"]]
        signs = [-1.0]
    elif pattern == "Mixed directions":
        affected = list(nodes["multi"])
        signs = [1.0 if r % 2 == 0 else -1.0 for r in range(len(affected))]
    elif pattern == "Total-preserving mixed":
        affected = list(nodes["multi"])
        if len(affected) < 3:
            raise ValueError("Total-preserving mixed drift requires at least three affected nodes")

        # Increase two representative nodes and choose one common reduction
        # factor for the remaining nodes so that the expected total network
        # weight is unchanged. Since all template block matrices differ only
        # by a scalar multiplier, matching the first template matches all of
        # them up to numerical precision.
        increase = affected[::3]
        decrease = [i for i in affected if i not in increase]
        theta[increase] *= math.exp(delta)
        baseline_total = float(case.mu_templates[:, 0].sum())

        def total_at(factor: float) -> float:
            trial = theta.copy()
            trial[decrease] *= factor
            mu = base.compute_mu_templates(
                trial, case.z, case.b_templates[:1], case.pi, case.pj,
            )
            return float(mu[:, 0].sum())

        if total_at(0.0) >= baseline_total:
            raise ValueError("Selected mixed-shift nodes cannot preserve total intensity")
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if total_at(mid) < baseline_total:
                lo = mid
            else:
                hi = mid
        theta[decrease] *= 0.5 * (lo + hi)
        mu = base.compute_mu_templates(theta, case.z, case.b_templates, case.pi, case.pj)
        relative_error = np.max(
            np.abs(mu.sum(axis=0) / case.mu_templates.sum(axis=0) - 1.0)
        )
        if relative_error > 1.0e-10:
            raise RuntimeError(f"total-preserving theta shift failed: {relative_error:g}")
        return mu, affected
    elif pattern == "Within-community redistribution":
        # Redistribute activity among nodes with the same dominant community.
        # The interpolation preserves each hard community's theta total. A
        # final common normalization removes the small total-intensity change
        # caused by mixed memberships without changing relative activities.
        shifted = theta.copy()
        affected: list[int] = []
        for group in range(case.k):
            members = np.where(case.labels == group)[0]
            if len(members) < 2:
                continue
            entropy = -np.sum(
                case.z[members] * np.log(np.maximum(case.z[members], EPS)), axis=1,
            )
            order = members[np.argsort(-entropy)]
            donor = theta[np.roll(order, 1)]
            shifted[order] = (1.0 - delta) * theta[order] + delta * donor
            affected.extend(int(i) for i in order)
        mu = base.compute_mu_templates(
            shifted, case.z, case.b_templates, case.pi, case.pj,
        )
        scale = math.sqrt(
            float(case.mu_templates[:, 0].sum()) / float(mu[:, 0].sum())
        )
        shifted *= scale
        mu = base.compute_mu_templates(
            shifted, case.z, case.b_templates, case.pi, case.pj,
        )
        relative_error = np.max(
            np.abs(mu.sum(axis=0) / case.mu_templates.sum(axis=0) - 1.0)
        )
        if relative_error > 1.0e-10:
            raise RuntimeError(f"within-community theta shift failed: {relative_error:g}")
        return mu, affected
    elif pattern == "Moment-preserving activity rotation":
        # Rotate three high-activity nodes inside each dominant community.
        # The rotation is orthogonal to the all-ones vector, so the within-
        # community mean and variance of theta are preserved exactly while
        # node-specific activities change in opposite directions.
        shifted = theta.copy()
        affected: list[int] = []
        angle = 2.0 * math.pi * delta
        ones = np.ones(3, dtype=float)
        for group in range(case.k):
            members = np.where(case.labels == group)[0]
            if len(members) < 3:
                raise ValueError("Activity rotation requires three nodes per community")
            selected = members[np.argsort(theta[members])[-3:]]
            x = theta[selected]
            mean = float(x.mean())
            v = x - mean
            norm = float(np.linalg.norm(v))
            if norm < EPS:
                continue
            w = np.cross(ones, v)
            w *= norm / max(float(np.linalg.norm(w)), EPS)
            # Fix one rotation direction for all delta values so the path is
            # continuous rather than selecting a direction after seeing each
            # individual alternative.
            if (mean + w).min() < (mean - w).min():
                w = -w
            candidate = mean + math.cos(angle) * v + math.sin(angle) * w
            if candidate.min() <= 0.0:
                raise ValueError("Activity rotation left the positive parameter space")
            shifted[selected] = candidate
            affected.extend(int(i) for i in selected)
        mu = base.compute_mu_templates(
            shifted, case.z, case.b_templates, case.pi, case.pj,
        )
        scale = math.sqrt(
            float(case.mu_templates[:, 0].sum()) / float(mu[:, 0].sum())
        )
        shifted *= scale
        mu = base.compute_mu_templates(
            shifted, case.z, case.b_templates, case.pi, case.pj,
        )
        relative_error = np.max(
            np.abs(mu.sum(axis=0) / case.mu_templates.sum(axis=0) - 1.0)
        )
        if relative_error > 1.0e-10:
            raise RuntimeError(f"moment-preserving theta shift failed: {relative_error:g}")
        return mu, affected
    elif pattern == "Template-opposed activity":
        affected = list(nodes["multi"])
        node_sign = np.array([
            1.0 if rank % 2 == 0 else -1.0
            for rank in range(len(affected))
        ])
        template_sign = np.array([1.0, -1.0, 1.0, -1.0])
        if case.b_templates.shape[0] != len(template_sign):
            raise ValueError("Template-opposed activity requires four templates")
        mu = np.empty_like(case.mu_templates)
        for s, direction in enumerate(template_sign):
            theta_s = theta.copy()
            theta_s[affected] *= np.exp(delta * direction * node_sign)
            current = base.compute_mu_templates(
                theta_s, case.z, case.b_templates[s:s + 1], case.pi, case.pj,
            )[:, 0]
            # Remove template-wide intensity changes so the alternative is a
            # redistribution of activity rather than a global mean shift.
            scale = math.sqrt(
                float(case.mu_templates[:, s].sum()) / float(current.sum())
            )
            theta_s *= scale
            mu[:, s] = base.compute_mu_templates(
                theta_s, case.z, case.b_templates[s:s + 1], case.pi, case.pj,
            )[:, 0]
        relative_error = np.max(
            np.abs(mu.sum(axis=0) / case.mu_templates.sum(axis=0) - 1.0)
        )
        if relative_error > 1.0e-10:
            raise RuntimeError(f"template-opposed theta shift failed: {relative_error:g}")
        return mu, affected
    else:
        raise ValueError(pattern)
    for i, sign in zip(affected, signs):
        theta[int(i)] *= math.exp(sign * delta)
    return base.compute_mu_templates(theta, case.z, case.b_templates, case.pi, case.pj), affected


def ordinary_z_shift_mu(case: Any, delta: float, nodes: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
    """Diffuse cyclic role rotation preserving every membership row sum.

    Each edge changes only mildly, while evidence is distributed across the
    network.  This complements the one-source total-preserving alternative.
    """
    transition = np.zeros((case.k, case.k), dtype=float)
    for k in range(case.k):
        transition[k, k] = 1.0 - delta
        transition[k, (k + 1) % case.k] = delta
    z = case.z @ transition
    return (
        base.compute_mu_templates(case.theta, z, case.b_templates, case.pi, case.pj),
        list(range(case.n)),
    )


def localized_z_shift_mu(
    case: Any, delta: float, nodes: dict[str, Any], affected_fraction: float = 0.10,
    reference_mu: np.ndarray | None = None, theta_coupling: float = 0.0,
    theta_pattern: str = "balanced", theta_power: float = 2.0,
    theta_scope: str = "affected", preserve_degrees: bool = False,
    target_community: int | None = None, shift_mode: str = "coherent",
) -> tuple[np.ndarray, list[int]]:
    """Move selected mixed memberships along fixed simplex paths.

    For every mixed node and possible pure-role endpoint, the complete design
    grid is evaluated against the fitted IC mean.  The endpoint with the
    largest minimum population-KL increment is retained, and nodes are ranked
    by the same criterion.  For each selected node,

        z_i(delta) = (1 - delta) z_i(0) + delta e_target(i).

    Hence delta is exactly the fraction of a fixed baseline-to-role migration.
    Node selection and target roles use Phase-I quantities only and are fixed
    before any Phase-II path is generated.
    """
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must lie in [0, 1]")
    if not 0.0 < affected_fraction <= 1.0:
        raise ValueError("affected_fraction must lie in (0, 1]")
    count = max(1, int(round(affected_fraction * case.n)))
    if shift_mode == "secondary":
        expected_degree = base.edge_degree(
            np.asarray(case.mu_templates, dtype=float).mean(axis=1),
            case.n, case.pi, case.pj,
        )
        candidates: list[tuple[float, int, int, int]] = []
        for i in range(case.n):
            dominant = int(np.argmax(case.z[i]))
            target = (dominant + 1) % case.k
            source_mass = float(
                case.z[i].sum() - case.z[i, dominant] - case.z[i, target]
            )
            entropy = -float(np.sum(case.z[i] * np.log(np.maximum(case.z[i], EPS))))
            if source_mass > 1.0e-6:
                candidates.append((entropy * expected_degree[i], i, dominant, target))
        selected = sorted(candidates, reverse=True)[:count]
        if len(selected) != count:
            raise RuntimeError("Insufficient mixed nodes for secondary redistribution")
        z = case.z.copy()
        affected = sorted(int(item[1]) for item in selected)
        for _, i, dominant, target in selected:
            sources = [k for k in range(case.k) if k not in {dominant, target}]
            moved = delta * float(z[i, sources].sum())
            z[i, sources] *= 1.0 - delta
            z[i, target] += moved
        theta = case.theta.copy()
        if preserve_degrees:
            target_degree = base.edge_degree(
                np.asarray(case.mu_templates, dtype=float).mean(axis=1),
                case.n, case.pi, case.pj,
            )
            for _ in range(100):
                current_mu = base.compute_mu_templates(
                    theta, z, case.b_templates, case.pi, case.pj,
                ).mean(axis=1)
                current_degree = base.edge_degree(
                    current_mu, case.n, case.pi, case.pj,
                )
                ratio = target_degree / np.maximum(current_degree, EPS)
                if float(np.max(np.abs(ratio - 1.0))) < 1.0e-8:
                    break
                theta *= np.sqrt(ratio)
        if theta_coupling > 0.0:
            if theta_pattern not in {"common", "common-down"}:
                raise ValueError(
                    "secondary redistribution supports common activity coupling only"
                )
            direction = 1.0 if theta_pattern == "common" else -1.0
            theta_nodes = np.arange(case.n) if theta_scope == "all" else affected
            theta[theta_nodes] *= math.exp(
                direction * theta_coupling * delta**theta_power
            )
        return (
            base.compute_mu_templates(theta, z, case.b_templates, case.pi, case.pj),
            affected,
        )
    if shift_mode != "coherent":
        raise ValueError("shift_mode must be 'coherent' or 'secondary'")
    reference = np.maximum(
        case.mu_templates if reference_mu is None else np.asarray(reference_mu), EPS,
    )
    def population_kl(means: np.ndarray) -> float:
        value = means * np.log(np.maximum(means, EPS) / reference) - means + reference
        return float(np.mean(np.sum(value, axis=0)))

    baseline_kl = population_kl(case.mu_templates)
    design_levels = (0.20, 0.275, 0.35, 0.425, 0.50)

    def final_common_means(
        target: int, selected_nodes: list[int], level: float,
    ) -> np.ndarray:
        trial_z = case.z.copy()
        endpoint = np.zeros(case.k, dtype=float)
        endpoint[target] = 1.0
        for node in selected_nodes:
            trial_z[node] = (
                (1.0 - level) * case.z[node] + level * endpoint
            )
        trial_theta = case.theta.copy()
        if preserve_degrees:
            target_mu = np.asarray(case.mu_templates, dtype=float).mean(axis=1)
            target_degree = base.edge_degree(
                target_mu, case.n, case.pi, case.pj,
            )
            for _ in range(100):
                current_mu = base.compute_mu_templates(
                    trial_theta, trial_z, case.b_templates, case.pi, case.pj,
                ).mean(axis=1)
                current_degree = base.edge_degree(
                    current_mu, case.n, case.pi, case.pj,
                )
                ratio = target_degree / np.maximum(current_degree, EPS)
                if float(np.max(np.abs(ratio - 1.0))) < 1.0e-8:
                    break
                trial_theta *= np.sqrt(ratio)
        if theta_coupling > 0.0 and theta_pattern in {"common", "common-down"}:
            direction = 1.0 if theta_pattern == "common" else -1.0
            theta_nodes = (
                np.arange(case.n) if theta_scope == "all"
                else np.asarray(selected_nodes, dtype=int)
            )
            trial_theta[theta_nodes] *= math.exp(
                direction * theta_coupling * level**theta_power
            )
        means = base.compute_mu_templates(
            trial_theta, trial_z, case.b_templates, case.pi, case.pj,
        )
        if theta_coupling > 0.0 and theta_pattern == "template-opposed":
            if theta_scope != "all":
                raise ValueError("template-opposed coupling requires theta_scope='all'")
            signs = np.where(np.arange(means.shape[1]) % 2 == 0, 1.0, -1.0)
            means *= np.exp(2.0 * theta_coupling * level**theta_power * signs[None, :])
        return means

    target_designs: list[tuple[float, int, list[tuple[float, int, int]]]] = []
    if target_community is not None and not 0 <= target_community < case.k:
        raise ValueError("target_community must index an existing community")
    target_grid = (
        range(case.k) if target_community is None else (target_community,)
    )
    for target in target_grid:
        candidates: list[tuple[float, int, int]] = []
        for i in range(case.n):
            entropy = -float(np.sum(case.z[i] * np.log(np.maximum(case.z[i], EPS))))
            if entropy <= 0.05 or target == int(np.argmax(case.z[i])):
                continue
            endpoint = np.zeros(case.k, dtype=float)
            endpoint[target] = 1.0
            path = []
            for level in design_levels:
                trial_z = case.z.copy()
                trial_z[i] = (1.0 - level) * case.z[i] + level * endpoint
                trial_mu = base.compute_mu_templates(
                    case.theta, trial_z, case.b_templates, case.pi, case.pj,
                )
                path.append(population_kl(trial_mu) - baseline_kl)
            score = float(np.min(np.diff([0.0, *path])))
            candidates.append((score, i, target))
        selected_for_target = sorted(candidates, reverse=True)[:count]
        if len(selected_for_target) == count:
            selected_nodes = [int(item[1]) for item in selected_for_target]
            group_path = [
                population_kl(final_common_means(target, selected_nodes, level))
                - baseline_kl
                for level in design_levels
            ]
            target_designs.append((
                float(np.min(np.diff([0.0, *group_path]))),
                target,
                selected_for_target,
            ))
    if not target_designs:
        raise RuntimeError("No eligible coherent membership-migration design")
    _, coherent_target, selected = max(target_designs, key=lambda item: item[0])
    if not selected:
        raise RuntimeError("No eligible mixed-membership nodes")
    affected = sorted(int(item[1]) for item in selected)
    z = case.z.copy()
    for i in affected:
        endpoint = np.zeros(case.k, dtype=float)
        endpoint[coherent_target] = 1.0
        z[i] = (1.0 - delta) * case.z[i] + delta * endpoint
    theta = case.theta.copy()
    if preserve_degrees:
        target_mu = np.asarray(case.mu_templates, dtype=float).mean(axis=1)
        target_degree = base.edge_degree(target_mu, case.n, case.pi, case.pj)
        for _ in range(100):
            current_mu = base.compute_mu_templates(
                theta, z, case.b_templates, case.pi, case.pj,
            ).mean(axis=1)
            current_degree = base.edge_degree(
                current_mu, case.n, case.pi, case.pj,
            )
            ratio = target_degree / np.maximum(current_degree, EPS)
            if float(np.max(np.abs(ratio - 1.0))) < 1.0e-8:
                break
            theta *= np.sqrt(ratio)
    if theta_coupling > 0.0 and theta_pattern in {"common", "common-down"}:
        # In the composite alternative, activity changes emerge gradually as
        # the membership migration becomes more pronounced.  The quadratic
        # coupling keeps weak alternatives primarily structural while allowing
        # larger migrations to carry the co-moving volume effect commonly seen
        # in transport-flow applications.
        direction = 1.0 if theta_pattern == "common" else -1.0
        theta_nodes = np.arange(case.n) if theta_scope == "all" else affected
        if theta_scope not in {"affected", "all"}:
            raise ValueError("theta_scope must be 'affected' or 'all'")
        theta[theta_nodes] *= math.exp(
            direction * theta_coupling * delta**theta_power
        )
    elif theta_coupling > 0.0 and len(affected) > 1 and theta_pattern == "balanced":
        signs = np.array([1.0 if rank % 2 == 0 else -1.0
                          for rank in range(len(affected))])
        target_total = float(base.compute_mu_templates(
            theta, z, case.b_templates, case.pi, case.pj,
        ).sum())

        def shifted_total(center: float) -> float:
            trial = theta.copy()
            trial[affected] *= np.exp(theta_coupling * delta * (signs - center))
            return float(base.compute_mu_templates(
                trial, z, case.b_templates, case.pi, case.pj,
            ).sum())

        lo, hi = -4.0, 4.0
        while shifted_total(lo) < target_total:
            lo *= 2.0
        while shifted_total(hi) > target_total:
            hi *= 2.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if shifted_total(mid) > target_total:
                lo = mid
            else:
                hi = mid
        center = 0.5 * (lo + hi)
        theta[affected] *= np.exp(theta_coupling * delta * (signs - center))
    elif theta_pattern not in {"balanced", "common", "common-down", "template-opposed"}:
        raise ValueError(
            "theta_pattern must be 'balanced', 'common', 'common-down', "
            "or 'template-opposed'"
        )
    means = base.compute_mu_templates(theta, z, case.b_templates, case.pi, case.pj)
    if theta_coupling > 0.0 and theta_pattern == "template-opposed":
        if theta_scope != "all":
            raise ValueError("template-opposed coupling requires theta_scope='all'")
        signs = np.where(np.arange(means.shape[1]) % 2 == 0, 1.0, -1.0)
        means *= np.exp(2.0 * theta_coupling * delta**theta_power * signs[None, :])
    return means, affected


def community_exposure_profiles(case: Any, means: np.ndarray) -> np.ndarray:
    """Expected partner-role proportions for every node and template."""
    profiles = np.zeros((case.n, case.k, means.shape[1]), dtype=float)
    for edge, (i, j) in enumerate(zip(case.pi, case.pj)):
        profiles[i] += case.z[j, :, None] * means[edge, None, :]
        profiles[j] += case.z[i, :, None] * means[edge, None, :]
    profiles /= np.maximum(profiles.sum(axis=1, keepdims=True), EPS)
    return profiles


def row_kl(current: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Column-wise KL divergence between positive role profiles."""
    current = np.asarray(current, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    return np.sum(
        np.where(
            current > 0.0,
            current * np.log(np.maximum(current, EPS) / np.maximum(baseline, EPS)),
            0.0,
        ),
        axis=0,
    )


def localized_z_shift_metrics(
    case: Any, delta: float, nodes: dict[str, Any], affected_fraction: float = 0.10,
    reference_mu: np.ndarray | None = None, theta_coupling: float = 0.0,
    theta_pattern: str = "balanced", theta_power: float = 2.0,
    theta_scope: str = "affected", preserve_degrees: bool = False,
    target_community: int | None = None, shift_mode: str = "coherent",
) -> dict[str, float]:
    """Population checks for the fixed-direction membership alternative."""
    means, affected = localized_z_shift_mu(
        case, delta, nodes, affected_fraction, reference_mu=reference_mu,
        theta_coupling=theta_coupling,
        theta_pattern=theta_pattern,
        theta_power=theta_power,
        theta_scope=theta_scope,
        preserve_degrees=preserve_degrees,
        target_community=target_community,
        shift_mode=shift_mode,
    )
    baseline_exposure = community_exposure_profiles(case, case.mu_templates)
    shifted_exposure = community_exposure_profiles(case, means)
    exposure_kl = np.array([
        np.mean(row_kl(shifted_exposure[i], baseline_exposure[i]))
        for i in affected
    ])
    poisson_kl = np.sum(
        means * np.log(np.maximum(means, EPS) / np.maximum(case.mu_templates, EPS))
        - means + case.mu_templates,
        axis=0,
    )
    return {
        "mean_exposure_kl": float(exposure_kl.mean()),
        "min_exposure_kl": float(exposure_kl.min()),
        "mean_poisson_kl": float(poisson_kl.mean()),
        "affected_count": float(len(affected)),
        "theta_coupling": float(theta_coupling),
        "theta_power": float(theta_power),
        "theta_scope_all": float(theta_scope == "all"),
        "preserve_degrees": float(preserve_degrees),
    }


def balanced_direction(case: Any, source: int) -> tuple[np.ndarray, float]:
    partner = np.sum(case.theta[:, None] * case.z, axis=0) - case.theta[source] * case.z[source]
    bvec = case.b_templates[0] @ partner
    ones = np.ones(case.k)
    if case.k != 3:
        _, _, vh = np.linalg.svd(np.vstack([ones, bvec]))
        w = vh[-1]
    else:
        w = np.cross(ones, bvec)
    w /= max(np.abs(w).sum() / 2.0, EPS)

    def feasible(sign: float) -> float:
        ws = sign * w
        limits = [-case.z[source, k] / ws[k] for k in range(case.k) if ws[k] < -EPS]
        return min(limits) if limits else 0.0

    plus, minus = feasible(1.0), feasible(-1.0)
    if minus > plus:
        w = -w
        maximum = minus
    else:
        maximum = plus
    return w, max(float(maximum), EPS)


def balanced_z_shift_mu(case: Any, fraction: float, nodes: dict[str, Any]) -> tuple[np.ndarray, list[int], float]:
    source = int(nodes["z_source"])
    w, maximum = balanced_direction(case, source)
    z = case.z.copy()
    actual = fraction * maximum
    z[source] = np.maximum(z[source] + actual * w, 0.0)
    z[source] /= z[source].sum()
    mu = base.compute_mu_templates(case.theta, z, case.b_templates, case.pi, case.pj)
    old_total = base.edge_degree(case.mu_templates.mean(axis=1), case.n, case.pi, case.pj)[source]
    new_total = base.edge_degree(mu.mean(axis=1), case.n, case.pi, case.pj)[source]
    relative_error = abs(new_total / old_total - 1.0)
    if relative_error > 1.0e-8:
        raise RuntimeError(f"balanced direction failed: relative total change={relative_error:g}")
    return mu, [source], actual


def run_ic_experiment(cfg: Config, outdir: Path) -> pd.DataFrame:
    m0_values = [500, 1000, 1500, 2000, 3000]
    per_rep_cal = int(math.ceil(cfg.calibration_reps / cfg.phase_reps))
    per_rep_val = int(math.ceil(cfg.reps / cfg.phase_reps))
    records: list[dict[str, Any]] = []

    for case_no, case_id in enumerate(["I", "II", "III"]):
        case = build_case(case_id, cfg)
        buckets: dict[tuple[int, str], dict[str, list[Any]]] = {
            (m0, method): {"rl": [], "cluster": [], "h": [], "bmax": []}
            for m0 in m0_values for method in METHODS
        }
        for r in range(cfg.phase_reps):
            history, history_templates, _ = generate_network_path(
                case, max(m0_values), cfg, stable_seed(cfg.seed, 100, case_no, r)
            )
            cal, cal_templates, _ = generate_network_path(
                case, cfg.pool_len, cfg, stable_seed(cfg.seed, 101, case_no, r), offset=4000
            )
            val, val_templates, _ = generate_network_path(
                case, cfg.pool_len, cfg, stable_seed(cfg.seed, 102, case_no, r), offset=8000
            )
            for m0 in m0_values:
                print(f"IC case={case_id} phase={r + 1}/{cfg.phase_reps} m0={m0}: fitting", flush=True)
                fitted = fit_methods(history[:, :m0], history_templates[:m0], case, cfg)
                for mi, method in enumerate(METHODS):
                    h, rl, _ = calibration_and_validation(
                        fitted[method], cal, cal_templates, val, val_templates, case, cfg,
                        stable_seed(cfg.seed, 110, case_no, r, m0, mi), per_rep_cal, per_rep_val,
                    )
                    bucket = buckets[(m0, method)]
                    bucket["rl"].append(rl)
                    bucket["cluster"].append(float(rl.mean()))
                    bucket["h"].append(h)
                    if method == METHODS[0]:
                        bucket["bmax"].append(fitted[method].details["spring"].bmax)
                    print(f"  {method}: h={h:.3f}, ARL0={rl.mean():.2f}", flush=True)
            del history, cal, val

        for m0 in m0_values:
            for method in METHODS:
                bucket = buckets[(m0, method)]
                rl = np.concatenate(bucket["rl"])[: cfg.reps]
                cluster = np.asarray(bucket["cluster"], dtype=float)
                records.append({
                    "case": case_id,
                    "n": case.n,
                    "method": method,
                    "m0": m0,
                    "arl0": float(rl.mean()),
                    "se": float(cluster.std(ddof=1) / math.sqrt(len(cluster))) if len(cluster) > 1 else float(rl.std(ddof=1) / math.sqrt(len(rl))),
                    "pooled_se": float(rl.std(ddof=1) / math.sqrt(len(rl))),
                    "median": float(np.median(rl)),
                    "h_mean": float(np.mean(bucket["h"])),
                    "h_sd": float(np.std(bucket["h"], ddof=1)) if len(bucket["h"]) > 1 else 0.0,
                    "bmax_mean": float(np.mean(bucket["bmax"])) if bucket["bmax"] else math.nan,
                    "phase_reps": cfg.phase_reps,
                    "paths": int(len(rl)),
                    "censored_rate": float(np.mean(rl > cfg.max_len)),
                })
    result = pd.DataFrame(records)
    result.to_csv(outdir / "table_arl0_long.csv", index=False)
    return result


def prepare_oc_fit(case: Any, cfg: Config, seed_tag: int = 0) -> tuple[dict[str, FittedMethod], np.ndarray, np.ndarray]:
    history, templates, _ = generate_network_path(
        case, 1500, cfg, stable_seed(cfg.seed, 200, seed_tag)
    )
    fitted = fit_methods(history, templates, case, cfg)
    cal, cal_templates, _ = generate_network_path(
        case, cfg.pool_len, cfg, stable_seed(cfg.seed, 201, seed_tag), offset=2000
    )
    for mi, method in enumerate(METHODS):
        inp = method_input(fitted[method], cal, cal_templates, case, cfg)
        length = inp.shape[1] if method == METHODS[4] else len(inp)
        indices = contiguous_indices(
            np.random.default_rng(stable_seed(cfg.seed, 202, seed_tag, mi)),
            length, cfg.calibration_reps, cfg.max_len,
        )
        states = chart_states(fitted[method], inp, indices, cfg)
        fitted[method].h, _ = choose_limit(states, cfg.target_arl0)
        print(f"OC calibration {method}: h={fitted[method].h:.4f}", flush=True)
    return fitted, history, templates


def evaluate_oc_pool(
    case: Any,
    cfg: Config,
    fitted: dict[str, FittedMethod],
    mu_shift: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    counts, templates, _ = generate_network_path(
        case, cfg.pool_len, cfg, seed, mu_templates=mu_shift, offset=5000,
        couple_to=case.mu_templates,
    )
    rows = []
    for mi, method in enumerate(METHODS):
        inp = method_input(fitted[method], counts, templates, case, cfg)
        length = inp.shape[1] if method == METHODS[4] else len(inp)
        indices = contiguous_indices(
            np.random.default_rng(seed + 100 + mi), length, cfg.reps, cfg.max_len
        )
        states = chart_states(fitted[method], inp, indices, cfg)
        rl = run_lengths(states, float(fitted[method].h))
        summary = summarize_rl(
            rl, starts=indices[:, 0], block_width=2 * cfg.max_len,
            censor_point=cfg.max_len,
        )
        rows.append({"method": method, "h": fitted[method].h, **summary})
    return rows


def run_oc_experiment(cfg: Config, outdir: Path) -> pd.DataFrame:
    case_theta = build_case("II", cfg)
    nodes_theta = select_source_nodes(case_theta)
    fitted_theta, _, _ = prepare_oc_fit(case_theta, cfg)
    case_z = build_case("III", cfg)
    nodes_z = select_source_nodes(case_z)
    fitted_z, _, _ = prepare_oc_fit(case_z, cfg, seed_tag=3)
    records: list[dict[str, Any]] = []

    def shift_metrics(case: Any, mu1: np.ndarray) -> tuple[float, float]:
        mu0 = np.maximum(case.mu_templates, EPS)
        ratio = np.maximum(mu1, EPS) / mu0
        kl = 2.0 * np.sum(mu1 * np.log(ratio) - (mu1 - mu0), axis=0)
        relative = np.mean(np.abs(ratio - 1.0), axis=0)
        return float(np.mean(kl)), float(np.mean(relative))

    theta_levels = [0.05, 0.10, 0.15, 0.20, 0.25]
    for pattern_no, pattern in enumerate(["Single increase", "Mixed directions"]):
        for level_no, delta in enumerate(theta_levels):
            mu, affected = theta_shift_mu(case_theta, delta, pattern, nodes_theta)
            rows = evaluate_oc_pool(
                case_theta, cfg, fitted_theta, mu,
                stable_seed(cfg.seed, 210, pattern_no),
            )
            expected_kl, mean_relative_change = shift_metrics(case_theta, mu)
            for row in rows:
                records.append({"case": "II", "family": "theta", "pattern": pattern, "level": delta,
                                "actual_shift": delta, "affected": json.dumps(affected),
                                "expected_kl": expected_kl,
                                "mean_relative_edge_change": mean_relative_change, **row})
            print(f"OC theta {pattern} delta={delta:.2f} complete", flush=True)

    ordinary_levels = [0.04, 0.06, 0.08, 0.10, 0.12]
    for level_no, delta in enumerate(ordinary_levels):
        mu, affected = ordinary_z_shift_mu(case_z, delta, nodes_z)
        rows = evaluate_oc_pool(case_z, cfg, fitted_z, mu, stable_seed(cfg.seed, 220))
        expected_kl, mean_relative_change = shift_metrics(case_z, mu)
        for row in rows:
            records.append({"case": "III", "family": "z", "pattern": "Diffuse role rotation", "level": level_no + 1,
                            "actual_shift": delta, "affected": json.dumps(affected),
                            "expected_kl": expected_kl,
                            "mean_relative_edge_change": mean_relative_change, **row})
        print(f"OC ordinary z delta={delta:.2f} complete", flush=True)

    fractions = [0.20, 0.40, 0.60, 0.80, 1.00]
    for level_no, fraction in enumerate(fractions):
        mu, affected, actual = balanced_z_shift_mu(case_z, fraction, nodes_z)
        rows = evaluate_oc_pool(case_z, cfg, fitted_z, mu, stable_seed(cfg.seed, 230))
        expected_kl, mean_relative_change = shift_metrics(case_z, mu)
        for row in rows:
            records.append({"case": "III", "family": "z", "pattern": "Total-preserving shift", "level": level_no + 1,
                            "actual_shift": actual, "affected": json.dumps(affected),
                            "expected_kl": expected_kl,
                            "mean_relative_edge_change": mean_relative_change, **row})
        print(f"OC balanced z fraction={fraction:.2f}, actual={actual:.4f} complete", flush=True)

    result = pd.DataFrame(records)
    result.to_csv(outdir / "tables_arl1_long.csv", index=False)
    with (outdir / "oc_fitted_models.json").open("w", encoding="utf-8") as stream:
        serial = {
            "sources": {"theta_case_II": nodes_theta, "z_case_III": nodes_z},
            "limits": {
                "theta_case_II": {m: float(fitted_theta[m].h) for m in METHODS},
                "z_case_III": {m: float(fitted_z[m].h) for m in METHODS},
            },
            "bmax": {
                "theta_case_II": int(fitted_theta[METHODS[0]].details["spring"].bmax),
                "z_case_III": int(fitted_z[METHODS[0]].details["spring"].bmax),
            },
            "comparison_note": "Published chart ideas adapted to undirected count-weighted networks; not exact source-code reproductions.",
        }
        json.dump(serial, stream, indent=2, ensure_ascii=False)
    return result


def run_mixed_direction_experiment(cfg: Config, outdir: Path) -> pd.DataFrame:
    """Recompute only the prespecified mixed-direction activity experiment."""
    result_path = outdir / "tables_arl1_long.csv"
    if not result_path.exists():
        raise FileNotFoundError(f"Existing OC result table required: {result_path}")

    case = build_case("II", cfg)
    nodes = select_source_nodes(case)
    fitted, _, _ = prepare_oc_fit(case, cfg)
    records: list[dict[str, Any]] = []

    def shift_metrics(mu1: np.ndarray) -> tuple[float, float]:
        mu0 = np.maximum(case.mu_templates, EPS)
        ratio = np.maximum(mu1, EPS) / mu0
        kl = 2.0 * np.sum(mu1 * np.log(ratio) - (mu1 - mu0), axis=0)
        relative = np.mean(np.abs(ratio - 1.0), axis=0)
        return float(np.mean(kl)), float(np.mean(relative))

    for delta in [0.05, 0.10, 0.15, 0.20, 0.25]:
        mu, affected = theta_shift_mu(case, delta, "Mixed directions", nodes)
        rows = evaluate_oc_pool(case, cfg, fitted, mu, stable_seed(cfg.seed, 210, 1))
        expected_kl, mean_relative_change = shift_metrics(mu)
        for row in rows:
            records.append({
                "case": "II", "family": "theta", "pattern": "Mixed directions",
                "level": delta, "actual_shift": delta,
                "affected": json.dumps(affected), "expected_kl": expected_kl,
                "mean_relative_edge_change": mean_relative_change, **row,
            })
        print(f"OC theta Mixed directions delta={delta:.2f} complete", flush=True)

    previous = pd.read_csv(result_path)
    keep = ~((previous["family"] == "theta") & (previous["pattern"] == "Mixed directions"))
    result = pd.concat([previous.loc[keep], pd.DataFrame(records)], ignore_index=True)
    result.to_csv(result_path, index=False)
    model_path = outdir / "oc_fitted_models.json"
    if model_path.exists():
        model_record = json.loads(model_path.read_text(encoding="utf-8"))
        model_record.setdefault("sources", {})["theta_case_II"] = nodes
        model_record.setdefault("limits", {})["theta_case_II"] = {
            method: float(fitted[method].h) for method in METHODS
        }
        model_record.setdefault("bmax", {})["theta_case_II"] = int(
            fitted[METHODS[0]].details["spring"].bmax
        )
        model_path.write_text(json.dumps(model_record, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def node_statistics(
    counts: np.ndarray,
    templates: np.ndarray,
    case: Any,
    fitted: FittedMethod,
    pi_share: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = np.asarray(fitted.details["mu"])
    edge_dev = poisson_deviance(counts, mu[:, templates])
    d_total = edge_dev.sum(axis=0)
    h_node = np.zeros((case.n, counts.shape[1]), dtype=float)
    y_node = np.zeros_like(h_node)
    for e, (i, j) in enumerate(zip(case.pi, case.pj)):
        h_node[int(i)] += edge_dev[e]
        h_node[int(j)] += edge_dev[e]
        y_node[int(i)] += counts[e]
        y_node[int(j)] += counts[e]
    adjusted = h_node - 2.0 * pi_share[:, templates] * d_total[None, :]
    return adjusted, y_node, h_node, d_total


def estimate_deviance_shares(
    history: np.ndarray,
    templates: np.ndarray,
    case: Any,
    fitted: FittedMethod,
    cfg: Config,
) -> np.ndarray:
    mu = np.asarray(fitted.details["mu"])
    edge_dev = poisson_deviance(history, mu[:, templates])
    h_node = np.zeros((case.n, history.shape[1]), dtype=float)
    d_total = edge_dev.sum(axis=0)
    for e, (i, j) in enumerate(zip(case.pi, case.pj)):
        h_node[int(i)] += edge_dev[e]
        h_node[int(j)] += edge_dev[e]
    share = np.zeros((case.n, cfg.templates), dtype=float)
    for s in range(cfg.templates):
        take = templates == s
        h_mean = h_node[:, take].mean(axis=1)
        d_mean = max(float(d_total[take].mean()), EPS)
        share[:, s] = h_mean / (2.0 * d_mean)
        share[:, s] = np.maximum(share[:, s], 0.0)
        share[:, s] /= max(float(share[:, s].sum()), EPS)
    return share


def alarm_metadata(states: np.ndarray, spring: np.ndarray, branch: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exceed = states > h
    hit = exceed.any(axis=1)
    pos = np.where(hit, exceed.argmax(axis=1), -1)
    rows = np.arange(states.shape[0])
    length = np.where(hit, np.maximum(spring[rows, np.maximum(pos, 0)], 1), 0)
    side = np.where(hit, branch[rows, np.maximum(pos, 0)], 0)
    side = np.where((hit) & (side == 0), 1, side)
    return pos.astype(int), length.astype(int), side.astype(int)


def alarm_window_scores(
    adjusted: np.ndarray,
    y_node: np.ndarray,
    counts: np.ndarray,
    templates: np.ndarray,
    indices: np.ndarray,
    pos: np.ndarray,
    length: np.ndarray,
    side: np.ndarray,
    case: Any,
    fitted: FittedMethod,
    cfg: Config,
) -> list[dict[str, Any]]:
    mu = np.asarray(fitted.details["mu"])
    z = np.asarray(fitted.details["z"])
    expected_total_by_template = np.zeros((case.n, mu.shape[1]), dtype=float)
    observed_exposure = np.zeros((case.n, case.k, counts.shape[1]), dtype=float)
    expected_exposure_by_template = np.zeros((case.n, case.k, mu.shape[1]), dtype=float)
    for e, (i, j) in enumerate(zip(case.pi, case.pj)):
        ii, jj = int(i), int(j)
        observed_exposure[ii] += z[jj, :, None] * counts[e][None, :]
        observed_exposure[jj] += z[ii, :, None] * counts[e][None, :]
        expected_total_by_template[ii] += mu[e]
        expected_total_by_template[jj] += mu[e]
        expected_exposure_by_template[ii] += z[jj, :, None] * mu[e][None, :]
        expected_exposure_by_template[jj] += z[ii, :, None] * mu[e][None, :]

    records: list[dict[str, Any]] = []
    for r in np.where(pos >= 0)[0]:
        stop = int(pos[r])
        ell = int(min(stop + 1, max(length[r], cfg.diagnosis_min_window, 1)))
        start = max(0, stop - ell + 1)
        cols = indices[r, start : stop + 1]
        lscore = int(side[r]) * adjusted[:, cols].sum(axis=1)
        y = y_node[:, cols].sum(axis=1)
        template_window = templates[cols]
        expected_total = expected_total_by_template[:, template_window].sum(axis=1)
        observed_exposure_window = observed_exposure[:, :, cols].sum(axis=2)
        expected_exposure = expected_exposure_by_template[:, :, template_window].sum(axis=2)
        gtheta = y - expected_total
        qtheta = 2.0 * (
            np.where(y > 0.0, y * np.log(y / np.maximum(expected_total, EPS)), 0.0)
            - (y - expected_total)
        )
        rz = np.full(case.n, np.nan)
        valid = y > 0
        obs_prop = observed_exposure_window[valid] / y[valid, None]
        exp_prop = expected_exposure[valid] / np.maximum(expected_total[valid, None], EPS)
        rz[valid] = 2.0 * y[valid] * np.sum(
            np.where(obs_prop > 0.0, obs_prop * np.log(obs_prop / np.maximum(exp_prop, EPS)), 0.0), axis=1
        )
        records.append({
            "path": int(r), "pos": stop, "length": ell, "side": int(side[r]),
            "lscore": lscore, "gtheta": gtheta, "qtheta": qtheta, "rz": rz, "y": y,
        })
    return records


def empirical_upper(reference: np.ndarray, value: float) -> float:
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    return float((1 + np.sum(ref >= value)) / (len(ref) + 1)) if len(ref) else math.nan


def empirical_lower(reference: np.ndarray, value: float) -> float:
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    return float((1 + np.sum(ref <= value)) / (len(ref) + 1)) if len(ref) else math.nan


def bh_select(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    valid = np.where(np.isfinite(p))[0]
    if not len(valid):
        return np.array([], dtype=int)
    order = valid[np.argsort(p[valid])]
    thresholds = alpha * np.arange(1, len(order) + 1) / len(order)
    passed = np.where(p[order] <= thresholds)[0]
    return order[: passed[-1] + 1] if len(passed) else np.array([], dtype=int)


def build_localization_bank(
    records: list[dict[str, Any]], n: int
) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    bank: dict[tuple[int, int], dict[str, list[float]]] = {
        (i, side): {"share": [], "activity": []}
        for i in range(n) for side in (-1, 1)
    }
    for record in records:
        for i in range(n):
            key = (i, record["side"])
            bank[key]["share"].append(float(record["lscore"][i]))
            bank[key]["activity"].append(float(record["qtheta"][i]))
    return {
        key: {name: np.asarray(value) for name, value in channels.items()}
        for key, channels in bank.items()
    }


def localize(
    record: dict[str, Any],
    bank: dict[tuple[int, int], dict[str, np.ndarray]],
    cfg: Config,
) -> np.ndarray:
    """Combine share-concentration and node-activity localization channels."""
    p_share = []
    p_activity = []
    for i in range(len(record["lscore"])):
        reference = bank[(i, int(record["side"]))]
        p_share.append(empirical_upper(reference["share"], float(record["lscore"][i])))
        p_activity.append(empirical_upper(reference["activity"], float(record["qtheta"][i])))
    selected_share = set(bh_select(np.asarray(p_share), cfg.alpha_loc))
    # The second channel receives half the primary screening level. Its null
    # reference is estimated independently from the type-diagnostic bank.
    selected_activity = set(bh_select(np.asarray(p_activity), cfg.alpha_loc / 2.0))
    return np.asarray(sorted(selected_share | selected_activity), dtype=int)


def build_diagnostic_bank(
    records: list[dict[str, Any]],
    local_bank: dict[tuple[int, int], dict[str, np.ndarray]],
    cfg: Config,
    n: int,
) -> dict[int, dict[str, np.ndarray]]:
    # Type calibration is conditional on a global null alarm, not on the much
    # rarer event that the same node also passes the localization BH screen.
    # Candidate selection is calibrated separately by ``local_bank``.
    bank = {i: {"g": [], "q": [], "rz": [], "y": [], "side": [], "length": []} for i in range(n)}
    for record in records:
        for i in range(n):
            bank[i]["g"].append(float(record["gtheta"][i]))
            bank[i]["q"].append(float(record["qtheta"][i]))
            bank[i]["rz"].append(float(record["rz"][i]))
            bank[i]["y"].append(float(record["y"][i]))
            bank[i]["side"].append(float(record["side"]))
            bank[i]["length"].append(float(record["length"]))
    return {i: {key: np.asarray(value) for key, value in values.items()} for i, values in bank.items()}


def diagnose_node(i: int, record: dict[str, Any], bank: dict[int, dict[str, np.ndarray]], cfg: Config) -> tuple[int, int, bool]:
    reference = bank[i]
    if len(reference["g"]) < 20:
        return 0, 0, False
    same_side = reference["side"] == float(record["side"])
    if np.sum(same_side) < 20:
        same_side = np.ones(len(reference["side"]), dtype=bool)
    p_theta = empirical_upper(reference["q"][same_side], float(record["qtheta"][i]))
    current_y = float(record["y"][i])
    candidate = np.where(same_side)[0]
    scale_y = max(float(np.std(reference["y"][candidate], ddof=1)), 1.0)
    scale_l = max(float(np.std(reference["length"][candidate], ddof=1)), 1.0)
    distances = (
        np.abs(reference["y"][candidate] - current_y) / scale_y
        + 0.25 * np.abs(reference["length"][candidate] - float(record["length"])) / scale_l
    )
    keep_n = min(max(100, len(candidate) // 4), len(candidate))
    keep = candidate[np.argsort(distances)[:keep_n]]
    p_z = empirical_upper(reference["rz"][keep], float(record["rz"][i]))
    return int(p_theta <= cfg.alpha_theta), int(p_z <= cfg.alpha_z), True


def combined_shift_mu(case: Any, nodes: dict[str, Any], same_node: bool) -> tuple[np.ndarray, list[int], dict[int, tuple[int, int]]]:
    theta = case.theta.copy()
    z = case.z.copy()
    theta_node = int(nodes["z_source"] if same_node else nodes["single"])
    z_node = int(nodes["z_source"])
    theta[theta_node] *= math.exp(0.60)
    target = int(nodes["z_target"])
    unit = np.zeros(case.k)
    unit[target] = 1.0
    z[z_node] = 0.30 * z[z_node] + 0.70 * unit
    affected = sorted(set([theta_node, z_node]))
    truth = {i: (int(i == theta_node), int(i == z_node)) for i in affected}
    return base.compute_mu_templates(theta, z, case.b_templates, case.pi, case.pj), affected, truth


def run_diagnostic_experiment(cfg: Config, outdir: Path) -> pd.DataFrame:
    case = build_case("II", cfg)
    nodes = select_source_nodes(case)
    # Keep the diagnostic benchmark's original four-node mixed incident so its
    # published localization results remain exactly reproducible.
    diagnostic_nodes = dict(nodes)
    diagnostic_nodes["multi"] = list(nodes["multi_diagnosis"])
    fitted_all, history, history_templates = prepare_oc_fit(case, cfg, seed_tag=7)
    fitted = fitted_all[METHODS[0]]
    pi_share = estimate_deviance_shares(history, history_templates, case, fitted, cfg)
    bank_n = cfg.diagnosis_bank_paths

    def normal_records(tag: int) -> list[dict[str, Any]]:
        counts, templates, _ = generate_network_path(case, cfg.pool_len, cfg, stable_seed(cfg.seed, 300, tag), offset=3000 * tag)
        inp = method_input(fitted, counts, templates, case, cfg)
        idx = contiguous_indices(np.random.default_rng(stable_seed(cfg.seed, 301, tag)), len(inp), bank_n, cfg.max_len)
        states, spring, branch = chart_states(fitted, inp, idx, cfg, return_meta=True)
        pos, length, side = alarm_metadata(states, spring, branch, float(fitted.h))
        adjusted, y_node, _, _ = node_statistics(counts, templates, case, fitted, pi_share)
        return alarm_window_scores(adjusted, y_node, counts, templates, idx, pos, length, side, case, fitted, cfg)

    local_records = normal_records(1)
    local_bank = build_localization_bank(local_records, case.n)
    diagnostic_records = normal_records(2)
    diagnostic_bank = build_diagnostic_bank(diagnostic_records, local_bank, cfg, case.n)

    scenarios: list[tuple[str, np.ndarray, list[int], dict[int, tuple[int, int]]]] = []
    for name, pattern in [("Single theta increase", "Single increase"), ("Single theta decrease", "Single decrease")]:
        delta = 0.80 if pattern == "Single decrease" else 0.60
        mu, affected = theta_shift_mu(case, delta, pattern, nodes)
        scenarios.append((name, mu, affected, {i: (1, 0) for i in affected}))
    mu, affected = theta_shift_mu(case, 0.50, "Mixed directions", diagnostic_nodes)
    scenarios.append(("Mixed-direction theta shifts", mu, affected, {i: (1, 0) for i in affected}))
    mu, affected, _ = balanced_z_shift_mu(case, 1.00, nodes)
    scenarios.append(("Total-preserving z shift", mu, affected, {i: (0, 1) for i in affected}))
    mu, affected, truth = combined_shift_mu(case, nodes, False)
    scenarios.append(("Different-node theta/z shifts", mu, affected, truth))
    mu, affected, truth = combined_shift_mu(case, nodes, True)
    scenarios.append(("Same-node theta/z shift", mu, affected, truth))

    rows: list[dict[str, Any]] = []
    for scenario_no, (name, mu, true_nodes, truth) in enumerate(scenarios):
        counts, templates, _ = generate_network_path(
            case, cfg.pool_len, cfg, stable_seed(cfg.seed, 310, scenario_no), mu_templates=mu, offset=9000
        )
        inp = method_input(fitted, counts, templates, case, cfg)
        idx = contiguous_indices(np.random.default_rng(stable_seed(cfg.seed, 311, scenario_no)), len(inp), cfg.diagnosis_paths, cfg.max_len)
        states, spring, branch = chart_states(fitted, inp, idx, cfg, return_meta=True)
        pos, length, side = alarm_metadata(states, spring, branch, float(fitted.h))
        adjusted, y_node, _, _ = node_statistics(counts, templates, case, fitted, pi_share)
        records = alarm_window_scores(adjusted, y_node, counts, templates, idx, pos, length, side, case, fitted, cfg)
        tp = fp = fn = 0
        correct_labels = label_total = unavailable = end_to_end = 0
        theta_true = theta_hit = theta_null = theta_false = 0
        z_true = z_hit = z_null = z_false = 0
        record_by_path = {record["path"]: record for record in records}
        for path in range(cfg.diagnosis_paths):
            if path not in record_by_path:
                fn += len(true_nodes)
                continue
            record = record_by_path[path]
            selected = set(int(i) for i in localize(record, local_bank, cfg))
            true_set = set(true_nodes)
            tp += len(selected & true_set)
            fp += len(selected - true_set)
            fn += len(true_set - selected)
            path_label_ok = False
            for i in selected & true_set:
                theta_flag, z_flag, available = diagnose_node(i, record, diagnostic_bank, cfg)
                if not available:
                    unavailable += 1
                    continue
                label_total += 1
                expected = truth[i]
                if expected[0]:
                    theta_true += 1
                    theta_hit += theta_flag
                else:
                    theta_null += 1
                    theta_false += theta_flag
                if expected[1]:
                    z_true += 1
                    z_hit += z_flag
                else:
                    z_null += 1
                    z_false += z_flag
                correct = (theta_flag, z_flag) == expected
                correct_labels += int(correct)
                path_label_ok = path_label_ok or correct
            end_to_end += int(bool(selected & true_set) and path_label_ok)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, EPS)
        rows.append({
            "scenario": name, "precision": precision, "recall": recall, "f1": f1,
            "diagnostic_accuracy": correct_labels / max(label_total, 1),
            "end_to_end_rate": end_to_end / cfg.diagnosis_paths,
            "unavailable_rate": unavailable / max(tp, 1),
            "effective_label_evaluations": label_total,
            "alarm_rate": len(records) / cfg.diagnosis_paths,
            "theta_sensitivity": theta_hit / max(theta_true, 1),
            "theta_false_label_rate": theta_false / max(theta_null, 1),
            "z_sensitivity": z_hit / max(z_true, 1),
            "z_false_label_rate": z_false / max(z_null, 1),
        })
        print(
            f"Diagnosis {name}: F1={f1:.3f}, label={rows[-1]['diagnostic_accuracy']:.3f}, "
            f"theta=({rows[-1]['theta_sensitivity']:.3f},{rows[-1]['theta_false_label_rate']:.3f}), "
            f"z=({rows[-1]['z_sensitivity']:.3f},{rows[-1]['z_false_label_rate']:.3f})",
            flush=True,
        )
    result = pd.DataFrame(rows)
    result.to_csv(outdir / "table_diagnostic.csv", index=False)
    return result


def with_mu(case: Any, mu: np.ndarray) -> Any:
    return base.CaseParams(
        case.case_id, case.n, case.k, case.pi, case.pj, case.labels,
        case.z, case.theta, case.b_templates, np.asarray(mu),
    )


def distance_adjusted_mu(case: Any, beta: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(size=(case.n, 2))
    distance = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
    edge_distance = distance[case.pi, case.pj]
    edge_distance /= max(float(edge_distance.mean()), EPS)
    out = case.mu_templates * np.exp(-beta * edge_distance[:, None])
    for s in range(out.shape[1]):
        out[:, s] *= case.mu_templates[:, s].sum() / max(out[:, s].sum(), EPS)
    return out


def fit_proposed_only(
    counts: np.ndarray,
    templates: np.ndarray,
    case: Any,
    cfg: Config,
    mask: np.ndarray | None = None,
    missing_p: float = 0.0,
) -> FittedMethod:
    fit_counts = counts if mask is None else counts / max(1.0 - missing_p, EPS)
    mu_hat, z_hat = base.fit_mldcmm_mu(fit_counts, templates, case, legacy_config(cfg))
    edge_dev = poisson_deviance(counts, mu_hat[:, templates])
    if mask is not None:
        edge_dev = edge_dev * mask / max(1.0 - missing_p, EPS)
    raw = edge_dev.sum(axis=0)
    spring, _ = fit_spring(raw, templates, cfg)
    return FittedMethod(METHODS[0], "spring-cusum", cfg.k_cusum,
                        {"mu": mu_hat, "z": z_hat, "spring": spring,
                         "missing_p": missing_p})


def proposed_robust_input(
    fitted: FittedMethod,
    counts: np.ndarray,
    templates: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    dev = poisson_deviance(counts, np.asarray(fitted.details["mu"])[:, templates])
    p = float(fitted.details.get("missing_p", 0.0))
    if mask is not None:
        dev = dev * mask / max(1.0 - p, EPS)
    return apply_spring_standardization(dev.sum(axis=0), templates, fitted.details["spring"])


def run_robustness_experiment(cfg: Config, outdir: Path) -> pd.DataFrame:
    base_case = build_case("II", cfg)
    nodes = select_source_nodes(base_case)
    settings = [("Conditional Poisson baseline", "baseline", 0.0)]
    settings += [(f"Compound Poisson, r={x:.2f}", "compound", x) for x in (0.05, 0.10, 0.20)]
    settings += [(f"Distance effect, beta={x:.2f}", "distance", x) for x in (0.10, 0.20, 0.30)]
    settings += [(f"Missing edges, p={x:.2f}", "missing", x) for x in (0.05, 0.10, 0.20)]
    rows: list[dict[str, Any]] = []
    outer_reps = max(int(cfg.robustness_phase_reps), 1)
    inner_pools = max(int(cfg.robustness_inner_pools), 1)
    # Each independently fitted Phase-I history receives a complete inner
    # calibration experiment. Splitting one calibration budget across histories
    # makes the history-specific control limits unnecessarily noisy.
    per_rep_cal = int(cfg.calibration_reps)
    per_rep_val = int(math.ceil(cfg.reps / outer_reps))
    robust_pool_len = max(8000, min(cfg.pool_len, 16000))

    for setting_no, (label, kind, value) in enumerate(settings):
        normal_mu = (
            distance_adjusted_mu(base_case, value, stable_seed(cfg.seed, 400, setting_no))
            if kind == "distance" else base_case.mu_templates
        )
        case = with_mu(base_case, normal_mu)
        compound = value if kind == "compound" else 0.0
        missing = value if kind == "missing" else 0.0
        theta_base, _ = theta_shift_mu(base_case, 0.15, "Single increase", nodes)
        # The five main-study levels are 0.20, 0.40, 0.60, 0.80, and 1.00
        # of the maximum feasible total-preserving displacement; 0.60 is the
        # prespecified middle level used in the robustness experiment.
        z_base, _, _ = balanced_z_shift_mu(base_case, 0.60, nodes)
        theta_mu = normal_mu * (theta_base / np.maximum(base_case.mu_templates, EPS))
        z_mu = normal_mu * (z_base / np.maximum(base_case.mu_templates, EPS))
        buckets: dict[str, list[Any]] = {
            "h": [], "bmax": [], "ic_rl": [], "theta_rl": [], "z_rl": [],
            "ic_phase": [], "theta_phase": [], "z_phase": [],
        }

        for phase in range(outer_reps):
            history, hist_t, hist_mask = generate_network_path(
                case, 1500, cfg, stable_seed(cfg.seed, 401, setting_no, phase),
                compound_r=compound, missing_p=missing,
            )
            fitted = fit_proposed_only(history, hist_t, case, cfg, hist_mask, missing)
            states_cal_parts = []
            cal_per_pool = int(math.ceil(per_rep_cal / inner_pools))
            for pool in range(inner_pools):
                cal, cal_t, cal_mask = generate_network_path(
                    case, robust_pool_len, cfg,
                    stable_seed(cfg.seed, 402, setting_no, phase, pool),
                    offset=2000 + pool * robust_pool_len,
                    compound_r=compound, missing_p=missing,
                )
                cal_input = proposed_robust_input(fitted, cal, cal_t, cal_mask)
                idx_cal = contiguous_indices(
                    np.random.default_rng(stable_seed(cfg.seed, 403, setting_no, phase, pool)),
                    len(cal_input), cal_per_pool, cfg.max_len,
                )
                states_cal_parts.append(
                    spring_states(cal_input, idx_cal, fitted.details["spring"], cfg.k_cusum)
                )
            states_cal = np.concatenate(states_cal_parts, axis=0)[:per_rep_cal]
            h, _ = choose_limit(states_cal, cfg.target_arl0)

            def evaluate(mu: np.ndarray, tag: int) -> np.ndarray:
                parts = []
                val_per_pool = int(math.ceil(per_rep_val / inner_pools))
                for pool in range(inner_pools):
                    counts, templates, mask = generate_network_path(
                        case, robust_pool_len, cfg,
                        stable_seed(cfg.seed, 410, setting_no, phase, tag, pool),
                        mu_templates=mu,
                        offset=6000 + 2000 * tag + pool * robust_pool_len,
                        compound_r=compound, missing_p=missing,
                    )
                    inp = proposed_robust_input(fitted, counts, templates, mask)
                    idx = contiguous_indices(
                        np.random.default_rng(stable_seed(cfg.seed, 411, setting_no, phase, tag, pool)),
                        len(inp), val_per_pool, cfg.max_len,
                    )
                    parts.append(
                        run_lengths(
                            spring_states(inp, idx, fitted.details["spring"], cfg.k_cusum), h
                        )
                    )
                return np.concatenate(parts)[:per_rep_val]

            ic_rl = evaluate(normal_mu, 0)
            theta_rl = evaluate(theta_mu, 1)
            z_rl = evaluate(z_mu, 2)
            buckets["h"].append(h)
            buckets["bmax"].append(fitted.details["spring"].bmax)
            for name, values in (("ic", ic_rl), ("theta", theta_rl), ("z", z_rl)):
                buckets[f"{name}_rl"].append(values)
                buckets[f"{name}_phase"].append(float(values.mean()))
            print(
                f"Robustness {label}, phase {phase + 1}/{outer_reps}: "
                f"{ic_rl.mean():.1f}, {theta_rl.mean():.1f}, {z_rl.mean():.1f}",
                flush=True,
            )

        def aggregate(name: str) -> dict[str, float]:
            values = np.concatenate(buckets[f"{name}_rl"])[: cfg.reps]
            phase_means = np.asarray(buckets[f"{name}_phase"], dtype=float)
            se = (
                float(phase_means.std(ddof=1) / math.sqrt(len(phase_means)))
                if len(phase_means) > 1
                else float(values.std(ddof=1) / math.sqrt(len(values)))
            )
            return {"arl": float(values.mean()), "se": se}

        ic = aggregate("ic")
        theta = aggregate("theta")
        zres = aggregate("z")
        rows.append({
            "setting": label, "kind": kind, "level": value,
            "h": float(np.mean(buckets["h"])),
            "h_sd": float(np.std(buckets["h"], ddof=1)) if outer_reps > 1 else 0.0,
            "arl0": ic["arl"], "arl0_se": ic["se"],
            "arl1_theta": theta["arl"], "arl1_theta_se": theta["se"],
            "arl1_z": zres["arl"], "arl1_z_se": zres["se"],
            "bmax": float(np.mean(buckets["bmax"])),
            "phase_reps": outer_reps,
            "inner_pools": inner_pools,
            "paths": cfg.reps,
        })
        print(f"Robustness {label}: {ic['arl']:.1f}, {theta['arl']:.1f}, {zres['arl']:.1f}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(outdir / "table_robustness.csv", index=False)
    return result


def fmt_estimate(mean: float, se: float) -> str:
    return f"{mean:.2f} ({se:.2f})"


def latex_rows_from_results(outdir: Path) -> str:
    lines = ["% Generated by run_full_simulation_v2.py; do not edit values by hand."]
    arl0_path = outdir / "table_arl0_long.csv"
    if arl0_path.exists():
        df = pd.read_csv(arl0_path)
        lines.append("% Table: IC ARL0")
        for case in ["I", "II", "III"]:
            for method in METHODS:
                row = df[(df.case == case) & (df.method == method)].sort_values("m0")
                values = [fmt_estimate(a, s) for a, s in zip(row.arl0, row.se)]
                prefix = f"{case}&" if method == METHODS[0] else "&"
                lines.append(prefix + method + "&" + "&".join(values) + r"\\")
    arl1_path = outdir / "tables_arl1_long.csv"
    if arl1_path.exists():
        df = pd.read_csv(arl1_path)
        for family, patterns in [("theta", ["Single increase", "Mixed directions"]),
                                 ("z", ["Diffuse role rotation", "Total-preserving shift"])]:
            lines.append(f"% Table: {family} ARL1")
            for pattern in patterns:
                for method in METHODS:
                    row = df[(df.family == family) & (df.pattern == pattern) & (df.method == method)].sort_values("level")
                    values = [fmt_estimate(a, s) for a, s in zip(row.arl, row.se)]
                    lines.append(method + "&" + "&".join(values) + r"\\")
    diagnosis_path = outdir / "table_diagnostic.csv"
    if diagnosis_path.exists():
        df = pd.read_csv(diagnosis_path)
        lines.append("% Table: diagnosis")
        for _, row in df.iterrows():
            lines.append(
                f"{row.scenario}&{row.precision:.3f}&{row.recall:.3f}&{row.f1:.3f}&"
                f"{row.diagnostic_accuracy:.3f}&{row.end_to_end_rate:.3f}&{row.unavailable_rate:.3f}\\\\"
            )
    robust_path = outdir / "table_robustness.csv"
    if robust_path.exists():
        df = pd.read_csv(robust_path)
        lines.append("% Table: robustness")
        for _, row in df.iterrows():
            lines.append(
                f"{row.setting}&{fmt_estimate(row.arl0, row.arl0_se)}&"
                f"{fmt_estimate(row.arl1_theta, row.arl1_theta_se)}&"
                f"{fmt_estimate(row.arl1_z, row.arl1_z_se)}\\\\"
            )
    text = "\n".join(lines) + "\n"
    (outdir / "simulation_result_rows.tex").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "ic", "oc", "oc-mixed", "diagnosis", "robustness"], default="all")
    parser.add_argument("--outdir", default=str(OUT_DEFAULT))
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--calibration-reps", type=int, default=10000)
    parser.add_argument("--phase-reps", type=int, default=3)
    parser.add_argument("--diagnosis-paths", type=int, default=3000)
    parser.add_argument("--diagnosis-bank-paths", type=int, default=30000)
    parser.add_argument("--robustness-phase-reps", type=int, default=10)
    parser.add_argument("--robustness-inner-pools", type=int, default=4)
    parser.add_argument("--pool-len", type=int, default=24000)
    parser.add_argument("--max-len", type=int, default=800)
    parser.add_argument("--mean-edge", type=float, default=20.0)
    parser.add_argument("--k-cusum", type=float, default=0.50)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = Config(
        seed=args.seed, reps=args.reps, calibration_reps=args.calibration_reps,
        phase_reps=args.phase_reps, diagnosis_paths=args.diagnosis_paths,
        diagnosis_bank_paths=args.diagnosis_bank_paths,
        robustness_phase_reps=args.robustness_phase_reps,
        robustness_inner_pools=args.robustness_inner_pools,
        pool_len=args.pool_len, max_len=args.max_len, mean_edge=args.mean_edge,
        k_cusum=args.k_cusum, quick=args.quick,
    )
    if args.quick:
        cfg.reps = min(cfg.reps, 400)
        cfg.calibration_reps = min(cfg.calibration_reps, 500)
        cfg.phase_reps = min(cfg.phase_reps, 2)
        cfg.diagnosis_paths = min(cfg.diagnosis_paths, 300)
        cfg.diagnosis_bank_paths = min(cfg.diagnosis_bank_paths, 1000)
        cfg.robustness_phase_reps = min(cfg.robustness_phase_reps, 2)
        cfg.robustness_inner_pools = min(cfg.robustness_inner_pools, 2)
        cfg.pool_len = min(cfg.pool_len, 4000)
        cfg.max_len = min(cfg.max_len, 500)
        cfg.mldcmm_iters = 5
        cfg.mldcmm_b_steps = 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.mode in {"all", "ic"}:
        run_ic_experiment(cfg, outdir)
    if args.mode in {"all", "oc"}:
        run_oc_experiment(cfg, outdir)
    if args.mode == "oc-mixed":
        run_mixed_direction_experiment(cfg, outdir)
    if args.mode in {"all", "diagnosis"}:
        run_diagnostic_experiment(cfg, outdir)
    if args.mode in {"all", "robustness"}:
        run_robustness_experiment(cfg, outdir)
    latex_rows_from_results(outdir)
    metadata = {
        "mode": args.mode,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - started,
        "config": asdict(cfg),
        "method_status": {
            "Proposed ML-DCMM": "implementation of revised Sections 2.1--2.5",
            "comparators": "transparent adaptations to undirected count-weighted data; not exact source-code reproductions",
        },
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE: {outdir}", flush=True)


if __name__ == "__main__":
    main()
