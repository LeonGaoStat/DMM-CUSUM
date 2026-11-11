#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase I ARL0 simulation for ML-DCMM network monitoring.

This script implements the IC/Phase-I part of the simulation study.  The data
are generated from an undirected ML-DCMM mean structure with hourly seasonality
and short-range temporal dependence.  Each competing chart is fitted from a
finite Phase I sample and its control limit is calibrated to nominal ARL0=200.

The implementation is intentionally self-contained and uses only numpy/pandas
so that the long Monte Carlo run can be launched from the local Codex runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT = ROOT / "simulation" / "results" / "phase1_estimation_rerun"
EPS = 1e-9


@dataclass
class SimConfig:
    target_arl0: float = 200.0
    reps: int = 10000
    calibration_reps: int = 10000
    phase_reps: int = 1
    pool_len: int = 24000
    max_len: int = 1500
    block_len: int = 24
    seed: int = 20260812
    templates: int = 4
    k_cusum: float = 0.50
    k_ss: float = 0.10
    lambda_gravity: float = 0.10
    lambda_dcsbm: float = 0.20
    phi: float = 0.25
    eta_sd: float = 0.05
    avg_edge_mean: float = 2.00
    shrink_templates: float = 4.0
    mldcmm_iters: int = 12
    mldcmm_b_steps: int = 4
    ten_local_quantile: float = 0.995
    calibration_mode: str = "hybrid"
    crossfit_folds: int = 5
    n_i: int = 30
    n_ii: int = 60
    n_iii: int = 60
    case_i_scale: float = 1.8
    case_ii_scale: float = 1.0
    case_iii_scale: float = 1.0
    pure_fraction: float = 0.35
    mix_main: float = 4.0
    mix_off: float = 0.5
    theta_power: float = 0.5
    theta_floor: float = 0.08
    write_details: bool = True
    methods: str = "all"
    common_validation: bool = False
    table_se: str = "pooled"
    validation_mode: str = "true"


@dataclass
class CaseParams:
    case_id: str
    n: int
    k: int
    pi: np.ndarray
    pj: np.ndarray
    labels: np.ndarray
    z: np.ndarray
    theta: np.ndarray
    b_templates: np.ndarray
    mu_templates: np.ndarray


@dataclass
class ScalarFit:
    mean_by_template: list[float]
    sd_by_template: list[float]
    rho: float
    resid_sd: float


@dataclass
class VectorFit:
    mean: list[float]
    sd: list[float]


@dataclass
class MethodFit:
    name: str
    chart: str
    parameter: float | None
    h: float | None = None
    details: dict[str, Any] | None = None


def parse_int_list(x: str) -> list[int]:
    return [int(v.strip()) for v in x.split(",") if v.strip()]


def parse_str_list(x: str) -> list[str]:
    return [v.strip() for v in x.split(",") if v.strip()]


def method_is_enabled(name: str, cfg: SimConfig) -> bool:
    if cfg.methods.strip().lower() == "all":
        return True
    return name in set(parse_str_list(cfg.methods))


def edge_index(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, 1)


def template_sequence(t_len: int, offset: int = 0, templates: int = 24) -> np.ndarray:
    return (np.arange(offset, offset + t_len) % templates).astype(np.int16)


def daily_factor(templates: int = 24) -> np.ndarray:
    hour = np.arange(templates) * (24.0 / max(float(templates), 1.0))
    morning = np.exp(-0.5 * ((hour - 8.0) / 2.1) ** 2)
    evening = np.exp(-0.5 * ((hour - 18.0) / 2.6) ** 2)
    midday = 0.35 * np.exp(-0.5 * ((hour - 13.0) / 3.8) ** 2)
    factor = 0.42 + 0.82 * morning + 0.95 * evening + midday
    return factor / factor.mean()


def build_case(case_id: str, cfg: SimConfig, seed: int) -> CaseParams:
    rng = np.random.default_rng(seed)
    if case_id == "I":
        n = cfg.n_i
        avg_scale = cfg.case_i_scale
        theta_power, theta_floor = 0.45, 0.10
        pure_fraction, mix_main, mix_off = 0.15, 2.0, 1.1
        b0 = np.array([[1.00, 0.30, 0.20], [0.30, 1.00, 0.25], [0.20, 0.25, 1.00]])
    elif case_id == "II":
        n = cfg.n_ii
        avg_scale = cfg.case_ii_scale
        theta_power, theta_floor = 0.60, 0.12
        pure_fraction, mix_main, mix_off = 0.30, 3.2, 0.8
        b0 = np.array([[1.00, 0.30, 0.20], [0.30, 1.00, 0.25], [0.20, 0.25, 1.00]])
    elif case_id == "III":
        n = cfg.n_iii
        avg_scale = cfg.case_iii_scale
        theta_power, theta_floor = 0.60, 0.12
        pure_fraction, mix_main, mix_off = 0.30, 3.2, 0.8
        b0 = np.array([[1.00, 0.10, 0.06], [0.10, 1.00, 0.08], [0.06, 0.08, 1.00]])
    else:
        raise ValueError(f"unknown case_id={case_id}")

    k = 3
    labels = np.repeat(np.arange(k), math.ceil(n / k))[:n]
    rng.shuffle(labels)

    z = np.zeros((n, k), dtype=float)
    pure_count = max(k, int(round(pure_fraction * n)))
    pure_nodes = np.zeros(n, dtype=bool)
    for kk in range(k):
        idx = np.where(labels == kk)[0]
        pure_nodes[idx[0]] = True
    remaining = np.where(~pure_nodes)[0]
    rng.shuffle(remaining)
    pure_nodes[remaining[: max(0, pure_count - pure_nodes.sum())]] = True
    for i in range(n):
        if pure_nodes[i]:
            z[i, labels[i]] = 1.0
        else:
            alpha = np.full(k, mix_off)
            alpha[labels[i]] = mix_main
            z[i] = rng.dirichlet(alpha)

    theta_star = rng.power(theta_power, size=n) + theta_floor
    theta = n * theta_star / theta_star.sum()

    pi, pj = edge_index(n)
    b_templates = np.stack([f * b0 for f in daily_factor(cfg.templates)], axis=0)
    mu = compute_mu_templates(theta, z, b_templates, pi, pj)
    b_templates = b_templates * (cfg.avg_edge_mean * avg_scale / max(mu.mean(), EPS))
    mu = compute_mu_templates(theta, z, b_templates, pi, pj)
    return CaseParams(case_id, n, k, pi, pj, labels, z, theta, b_templates, mu)


def compute_mu_templates(
    theta: np.ndarray,
    z: np.ndarray,
    b_templates: np.ndarray,
    pi: np.ndarray,
    pj: np.ndarray,
) -> np.ndarray:
    out = np.empty((len(pi), b_templates.shape[0]), dtype=float)
    scale = theta[pi] * theta[pj]
    for s in range(b_templates.shape[0]):
        zbz = z @ b_templates[s] @ z.T
        out[:, s] = np.maximum(scale * zbz[pi, pj], EPS)
    return out


def generate_counts(
    mu_templates: np.ndarray,
    templates: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    t_len = len(templates)
    x = np.zeros(t_len, dtype=float)
    x[0] = rng.normal(0.0, cfg.eta_sd)
    eps_sd = cfg.eta_sd * math.sqrt(max(1.0 - cfg.phi**2, EPS))
    for t in range(1, t_len):
        x[t] = cfg.phi * x[t - 1] + rng.normal(0.0, eps_sd)
    eta = np.exp(x - 0.5 * cfg.eta_sd**2)
    mu = mu_templates[:, templates] * eta[None, :]
    return rng.poisson(np.maximum(mu, EPS)).astype(float)


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    mu = np.maximum(mu, EPS)
    y = np.asarray(y, dtype=float)
    term = np.where(y > 0, y * np.log(np.maximum(y, EPS) / mu), 0.0)
    return 2.0 * (term - (y - mu))


def edge_means_by_template(
    a_phase: np.ndarray,
    templates: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    e = a_phase.shape[0]
    global_mean = np.maximum(a_phase.mean(axis=1), EPS)
    totals = np.array([a_phase[:, templates == s].sum() for s in range(cfg.templates)])
    counts = np.array([(templates == s).sum() for s in range(cfg.templates)], dtype=float)
    global_total = max(a_phase.sum() / max(a_phase.shape[1], 1), EPS)
    seasonal = np.where(counts > 0, totals / np.maximum(counts, 1.0) / global_total, 1.0)
    out = np.empty((e, cfg.templates), dtype=float)
    for s in range(cfg.templates):
        idx = templates == s
        n_s = int(idx.sum())
        emp = a_phase[:, idx].mean(axis=1) if n_s > 0 else global_mean * seasonal[s]
        target = global_mean * seasonal[s]
        out[:, s] = (n_s * emp + cfg.shrink_templates * target) / (n_s + cfg.shrink_templates)
    return np.maximum(out, EPS)


def edge_to_matrix(edge_vals: np.ndarray, n: int, pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    mat = np.zeros((n, n), dtype=float)
    mat[pi, pj] = edge_vals
    mat[pj, pi] = edge_vals
    return mat


def edge_degree(edge_vals: np.ndarray, n: int, pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    deg = np.zeros(n, dtype=float)
    np.add.at(deg, pi, edge_vals)
    np.add.at(deg, pj, edge_vals)
    return deg


def estimate_z_spectral(edge_mean: np.ndarray, case: CaseParams) -> np.ndarray:
    n, k, pi, pj = case.n, case.k, case.pi, case.pj
    mat = edge_to_matrix(edge_mean, n, pi, pj)
    deg = np.maximum(mat.sum(axis=1), EPS)
    theta = deg / max(deg.mean(), EPS)
    norm = mat / np.maximum(np.outer(theta, theta), EPS)
    vals, vecs = np.linalg.eigh((norm + norm.T) / 2.0)
    idx = np.argsort(np.abs(vals))[-k:]
    vals = np.maximum(np.abs(vals[idx]), EPS)
    emb = np.abs(vecs[:, idx] * np.sqrt(vals)[None, :]) + 1e-4
    return emb / emb.sum(axis=1, keepdims=True)


def sketched_vertex_search(ratio: np.ndarray, k: int) -> np.ndarray:
    """Farthest-point vertex search used by mixed-SCORE."""
    n = ratio.shape[0]
    if k == 2:
        return np.array([np.argmin(ratio[:, 0]), np.argmax(ratio[:, 0])], dtype=int)
    selected = []
    remaining = set(range(n))
    center = ratio.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(ratio - center, axis=1)))
    selected.append(first)
    remaining.remove(first)
    while len(selected) < k:
        current = np.asarray(selected, dtype=int)
        best_idx, best_dist = None, -np.inf
        for idx in remaining:
            dist = np.linalg.norm(ratio[idx] - ratio[current], axis=1).min()
            if dist > best_dist:
                best_idx, best_dist = idx, dist
        selected.append(int(best_idx))
        remaining.remove(int(best_idx))
    return np.asarray(selected, dtype=int)


def mixed_score_layer_quiet(adj: np.ndarray, k: int) -> np.ndarray:
    """Mixed-SCORE membership estimate for one weighted undirected network."""
    n = adj.shape[0]
    vals_all, vecs_all = np.linalg.eigh((adj + adj.T) / 2.0)
    top = np.argsort(-np.abs(vals_all))[:k]
    vals = vals_all[top]
    vecs = vecs_all[:, top]
    order = np.concatenate([np.where(vals > 0)[0], np.where(vals <= 0)[0]])
    if len(order) < k:
        order = np.arange(k)
    vals = vals[order]
    vecs = vecs[:, order]
    if np.sum(vecs[:, 0] > 0) < n / 2:
        vecs[:, 0] *= -1
    u1 = vecs[:, 0].copy()
    eps = 1e-10
    ratio = np.zeros((n, k - 1), dtype=float)
    clip = max(np.log(max(n, 3)), 1.0)
    for c in range(k - 1):
        ratio[:, c] = np.clip(vecs[:, c + 1] / np.where(np.abs(u1) > eps, u1, eps), -clip, clip)
    pure = sketched_vertex_search(ratio, k)
    vertices = ratio[pure].T
    v_aug = np.vstack([np.ones(k), vertices])
    try:
        v_inv = np.linalg.inv(v_aug)
    except np.linalg.LinAlgError:
        v_inv = np.linalg.pinv(v_aug)
    b1 = np.ones(k, dtype=float)
    for c in range(k):
        v_c = vertices[:, c]
        denom = vals[0] + v_c @ np.diag(vals[1:k]) @ v_c
        b1[c] = math.sqrt(n / max(float(denom), eps))
    z = np.zeros((n, k), dtype=float)
    for i in range(n):
        w = v_inv @ np.concatenate([[1.0], ratio[i]])
        z[i] = np.maximum(w / np.maximum(np.abs(b1), eps), 0.0)
        total = z[i].sum()
        if total <= eps or not np.isfinite(total):
            z[i] = np.ones(k) / k
        else:
            z[i] /= total
    return np.maximum(z, 1e-6) / np.maximum(z.sum(axis=1, keepdims=True), EPS)


def align_z_to_reference(z: np.ndarray, z_ref: np.ndarray) -> np.ndarray:
    k = z.shape[1]
    best, best_err = z, np.inf
    for perm in permutations(range(k)):
        zp = z[:, perm]
        err = float(np.linalg.norm(zp - z_ref) ** 2)
        if err < best_err:
            best, best_err = zp, err
    return best


def estimate_z_mldcmm(a_phase: np.ndarray, templates: np.ndarray, case: CaseParams, cfg: SimConfig) -> np.ndarray:
    """Estimate common ML-DCMM memberships from Phase I template networks.

    The estimator follows the ML-DCMM aggregated mixed-SCORE logic: build
    template-level Phase-I mean networks, estimate memberships on each template
    layer, align the community labels across layers, and average them.  No
    simulated true membership is used.
    """
    mean_edges = edge_means_by_template(a_phase, templates, cfg)
    z_list = []
    for s in range(cfg.templates):
        mat = edge_to_matrix(mean_edges[:, s], case.n, case.pi, case.pj)
        if mat.sum() <= EPS:
            continue
        try:
            z_s = mixed_score_layer_quiet(mat, case.k)
            z_list.append(z_s)
        except Exception:
            continue
    if not z_list:
        aggregate = edge_to_matrix(mean_edges.mean(axis=1), case.n, case.pi, case.pj)
        z_list = [mixed_score_layer_quiet(aggregate, case.k)]
    z_ref = z_list[0]
    z_list = [z_ref] + [align_z_to_reference(z_s, z_ref) for z_s in z_list[1:]]
    z_hat = np.mean(np.stack(z_list, axis=0), axis=0)
    z_hat = np.maximum(z_hat, 1e-6)
    return z_hat / z_hat.sum(axis=1, keepdims=True)


def kmeans_labels(x: np.ndarray, k: int, seed: int = 0, n_iter: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            idx = labels == c
            if idx.any():
                centers[c] = x[idx].mean(axis=0)
            else:
                centers[c] = x[rng.integers(0, n)]
    return labels


def estimate_hard_labels_from_phase(a_phase: np.ndarray, templates: np.ndarray, case: CaseParams, cfg: SimConfig) -> np.ndarray:
    mean_edges = edge_means_by_template(a_phase, templates, cfg)
    mat = edge_to_matrix(mean_edges.mean(axis=1), case.n, case.pi, case.pj)
    deg = np.maximum(mat.sum(axis=1), EPS)
    norm_mat = mat / np.sqrt(np.outer(deg, deg))
    vals, vecs = np.linalg.eigh((norm_mat + norm_mat.T) / 2.0)
    emb = vecs[:, np.argsort(vals)[-case.k:]]
    row_norm = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), EPS)
    return kmeans_labels(emb / row_norm, case.k, seed=cfg.seed + case.n)


def solve_symmetric_b(
    edge_mean: np.ndarray,
    theta: np.ndarray,
    z: np.ndarray,
    case: CaseParams,
) -> np.ndarray:
    k = z.shape[1]
    y = edge_mean / np.maximum(theta[case.pi] * theta[case.pj], EPS)
    cols = []
    pairs = []
    for a in range(k):
        for b in range(a, k):
            if a == b:
                col = z[case.pi, a] * z[case.pj, a]
            else:
                col = z[case.pi, a] * z[case.pj, b] + z[case.pi, b] * z[case.pj, a]
            cols.append(col)
            pairs.append((a, b))
    x = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(x + 1e-8, y, rcond=None)
    coef = np.maximum(coef, 1e-5)
    bmat = np.zeros((k, k), dtype=float)
    for c, (a, b) in zip(coef, pairs):
        bmat[a, b] = c
        bmat[b, a] = c
    return bmat


def symmetric_b_design(theta: np.ndarray, z: np.ndarray, case: CaseParams) -> tuple[np.ndarray, list[tuple[int, int]]]:
    k = z.shape[1]
    scale = theta[case.pi] * theta[case.pj]
    cols = []
    pairs = []
    for a in range(k):
        for b in range(a, k):
            if a == b:
                col = scale * z[case.pi, a] * z[case.pj, a]
            else:
                col = scale * (z[case.pi, a] * z[case.pj, b] + z[case.pi, b] * z[case.pj, a])
            cols.append(col)
            pairs.append((a, b))
    return np.vstack(cols).T, pairs


def poisson_update_symmetric_b(
    edge_mean: np.ndarray,
    theta: np.ndarray,
    z: np.ndarray,
    case: CaseParams,
    b_init: np.ndarray | None,
    steps: int,
) -> np.ndarray:
    """Poisson likelihood multiplicative updates for symmetric B.

    Given theta and z, the ML-DCMM mean is linear in the unique entries of B.
    These updates are the usual Poisson NMF/EM-type updates for nonnegative
    intensity models and are better aligned with the count likelihood than a
    least-squares update.
    """
    x, pairs = symmetric_b_design(theta, z, case)
    if b_init is None:
        coef = np.full(len(pairs), max(float(edge_mean.mean()), EPS), dtype=float)
    else:
        coef = np.array([b_init[a, b] for a, b in pairs], dtype=float)
        coef = np.maximum(coef, 1e-6)
    denom = np.maximum(x.sum(axis=0), EPS)
    y = np.asarray(edge_mean, dtype=float)
    for _ in range(max(steps, 1)):
        mu = np.maximum(x @ coef, EPS)
        numer = x.T @ (y / mu)
        coef = np.maximum(coef * numer / denom, 1e-8)
    bmat = np.zeros((z.shape[1], z.shape[1]), dtype=float)
    for c, (a, b) in zip(coef, pairs):
        bmat[a, b] = c
        bmat[b, a] = c
    return bmat


def fit_mldcmm_mu(a_phase: np.ndarray, templates: np.ndarray, case: CaseParams, cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    mean_edges = edge_means_by_template(a_phase, templates, cfg)
    z_hat = estimate_z_mldcmm(a_phase, templates, case, cfg)
    mu_hat = np.empty_like(mean_edges)
    for s in range(cfg.templates):
        target = mean_edges[:, s]
        deg = edge_degree(target, case.n, case.pi, case.pj)
        theta = np.maximum(deg / max(deg.mean(), EPS), 1e-3)
        bmat = solve_symmetric_b(target, theta, z_hat, case)
        bmat = poisson_update_symmetric_b(target, theta, z_hat, case, bmat, cfg.mldcmm_b_steps)
        for _ in range(cfg.mldcmm_iters):
            qmat = z_hat @ bmat @ z_hat.T
            q_edge = np.maximum(qmat[case.pi, case.pj], EPS)
            denom = np.zeros(case.n, dtype=float)
            np.add.at(denom, case.pi, theta[case.pj] * q_edge)
            np.add.at(denom, case.pj, theta[case.pi] * q_edge)
            theta = np.maximum(deg / np.maximum(denom, EPS), 1e-4)
            theta = theta / max(theta.mean(), EPS)
            bmat = poisson_update_symmetric_b(target, theta, z_hat, case, bmat, cfg.mldcmm_b_steps)
        pred = compute_mu_templates(theta, z_hat, bmat[None, :, :], case.pi, case.pj)[:, 0]
        pred *= max(target.sum(), EPS) / max(pred.sum(), EPS)
        mu_hat[:, s] = np.maximum(pred, EPS)
    return mu_hat, z_hat


def crossfit_proposed_calibration_input(
    a_phase: np.ndarray,
    templates: np.ndarray,
    case: CaseParams,
    cfg: SimConfig,
) -> np.ndarray:
    """Out-of-fold Phase-I residuals for bootstrap control-limit calibration."""
    t_len = a_phase.shape[1]
    folds = min(max(int(cfg.crossfit_folds), 2), max(2, t_len // max(cfg.templates, 1)))
    fold_id = np.arange(t_len) % folds
    pieces = []
    for fold in range(folds):
        hold = fold_id == fold
        train = ~hold
        if train.sum() <= cfg.templates or hold.sum() == 0:
            continue
        mu_train, _ = fit_mldcmm_mu(a_phase[:, train], templates[train], case, cfg)
        raw_train = raw_deviance(a_phase[:, train], templates[train], mu_train)
        scalar_fit, _ = fit_scalar_transform(raw_train, templates[train], cfg)
        raw_hold = raw_deviance(a_phase[:, hold], templates[hold], mu_train)
        pieces.append(apply_scalar_transform(raw_hold, templates[hold], scalar_fit))
    if not pieces:
        mu_full, _ = fit_mldcmm_mu(a_phase, templates, case, cfg)
        raw_full = raw_deviance(a_phase, templates, mu_full)
        _, transformed = fit_scalar_transform(raw_full, templates, cfg)
        return transformed
    return np.concatenate(pieces)


def fit_gravity_mu(a_phase: np.ndarray, templates: np.ndarray, case: CaseParams, cfg: SimConfig) -> np.ndarray:
    mean_edges = edge_means_by_template(a_phase, templates, cfg)
    out = np.empty_like(mean_edges)
    for s in range(cfg.templates):
        deg = edge_degree(mean_edges[:, s], case.n, case.pi, case.pj)
        total = max(mean_edges[:, s].sum(), EPS)
        pred = deg[case.pi] * deg[case.pj] / max(2.0 * total, EPS)
        pred *= total / max(pred.sum(), EPS)
        out[:, s] = np.maximum(pred, EPS)
    return out


def fit_dcsbm_mu(
    a_phase: np.ndarray,
    templates: np.ndarray,
    labels: np.ndarray,
    case: CaseParams,
    cfg: SimConfig,
) -> np.ndarray:
    mean_edges = edge_means_by_template(a_phase, templates, cfg)
    out = np.empty_like(mean_edges)
    k = int(labels.max()) + 1
    for s in range(cfg.templates):
        target = mean_edges[:, s]
        deg = edge_degree(target, case.n, case.pi, case.pj)
        theta = np.ones(case.n, dtype=float)
        for g in range(k):
            idx = labels == g
            theta[idx] = deg[idx] / max(deg[idx].mean(), EPS)
        p = np.zeros((k, k), dtype=float)
        denom = np.zeros((k, k), dtype=float)
        for e, (i, j) in enumerate(zip(case.pi, case.pj)):
            a, b = int(labels[i]), int(labels[j])
            aa, bb = min(a, b), max(a, b)
            p[aa, bb] += target[e]
            denom[aa, bb] += theta[i] * theta[j]
        p = p / np.maximum(denom, EPS)
        p = p + p.T - np.diag(np.diag(p))
        pred = theta[case.pi] * theta[case.pj] * p[labels[case.pi], labels[case.pj]]
        pred *= max(target.sum(), EPS) / max(pred.sum(), EPS)
        out[:, s] = np.maximum(pred, EPS)
    return out


def raw_deviance(a: np.ndarray, templates: np.ndarray, mu_templates: np.ndarray) -> np.ndarray:
    return poisson_deviance(a, mu_templates[:, templates]).sum(axis=0)


def current_gravity_fit(a_t: np.ndarray, case: CaseParams) -> np.ndarray:
    deg = edge_degree(a_t, case.n, case.pi, case.pj)
    total = max(float(a_t.sum()), EPS)
    pred = deg[case.pi] * deg[case.pj] / max(2.0 * total, EPS)
    pred *= total / max(float(pred.sum()), EPS)
    return np.maximum(pred, EPS)


def raw_gravity_wlrt(a: np.ndarray, templates: np.ndarray, mu0: np.ndarray, case: CaseParams) -> np.ndarray:
    base = poisson_deviance(a, mu0[:, templates]).sum(axis=0)
    fit = np.zeros(a.shape[1], dtype=float)
    for t in range(a.shape[1]):
        fit[t] = poisson_deviance(a[:, t], current_gravity_fit(a[:, t], case)).sum()
    return np.maximum(base - fit, 0.0)


def block_pairs(k: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(k) for b in range(a, k)]


def dcsbm_feature_series(a: np.ndarray, labels: np.ndarray, case: CaseParams) -> np.ndarray:
    """Wilson-Stevens-Woodall DCSBM monitoring statistics.

    The DCSBM competitor monitors the estimated block propensities P_rs and the
    within-community standard deviations s_r of degree-correction estimates.
    The returned columns are {P_rs:r<=s} followed by {s_r}.
    """
    k = int(labels.max()) + 1
    pairs = block_pairs(k)
    x = np.zeros((a.shape[1], len(pairs) + k), dtype=float)
    pair_to_idx = {p: idx for idx, p in enumerate(pairs)}
    for t in range(a.shape[1]):
        numerator = np.zeros(len(pairs), dtype=float)
        denominator = np.zeros(len(pairs), dtype=float)
        deg = edge_degree(a[:, t], case.n, case.pi, case.pj)
        theta = np.ones(case.n, dtype=float)
        for g in range(k):
            idx = labels == g
            theta[idx] = deg[idx] / max(deg[idx].mean(), EPS)
            x[t, len(pairs) + g] = theta[idx].std(ddof=1) if idx.sum() > 1 else 0.0
        for e, (i, j) in enumerate(zip(case.pi, case.pj)):
            aa, bb = sorted((int(labels[i]), int(labels[j])))
            idx_pair = pair_to_idx[(aa, bb)]
            numerator[idx_pair] += a[e, t]
            denominator[idx_pair] += theta[i] * theta[j]
        x[t, : len(pairs)] = numerator / np.maximum(denominator, EPS)
    return x


def connected_features_for_one(edge_vals: np.ndarray, case: CaseParams) -> tuple[float, float, float]:
    n = case.n
    adj = [[] for _ in range(n)]
    deg = np.zeros(n, dtype=int)
    active = edge_vals > 0
    for i, j in zip(case.pi[active], case.pj[active]):
        ii, jj = int(i), int(j)
        adj[ii].append(jj)
        adj[jj].append(ii)
        deg[ii] += 1
        deg[jj] += 1

    seen = np.zeros(n, dtype=bool)
    comp_count = 0
    diameters = []
    for start in range(n):
        if seen[start]:
            continue
        comp_count += 1
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for nb in adj[v]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        diam = 0
        for src in comp:
            dist = {src: 0}
            queue = [src]
            qpos = 0
            while qpos < len(queue):
                v = queue[qpos]
                qpos += 1
                for nb in adj[v]:
                    if nb not in dist:
                        dist[nb] = dist[v] + 1
                        queue.append(nb)
            if dist:
                diam = max(diam, max(dist.values()))
        diameters.append(diam)
    return float(deg.mean()), float(comp_count), float(np.mean(diameters) if diameters else 0.0)


def ss_feature_series(a: np.ndarray, case: CaseParams) -> np.ndarray:
    x = np.zeros((a.shape[1], 3), dtype=float)
    for t in range(a.shape[1]):
        x[t] = connected_features_for_one(a[:, t], case)
    return x


def fit_scalar_transform(raw: np.ndarray, templates: np.ndarray, cfg: SimConfig) -> tuple[ScalarFit, np.ndarray]:
    mean = np.zeros(cfg.templates, dtype=float)
    sd = np.ones(cfg.templates, dtype=float)
    z = np.zeros_like(raw, dtype=float)
    global_sd = max(raw.std(ddof=1), EPS)
    global_mean = raw.mean()
    shrink = max(cfg.shrink_templates, 2.0 * cfg.templates)
    for s in range(cfg.templates):
        idx = templates == s
        if idx.sum() > 1:
            n_s = float(idx.sum())
            emp_mean = raw[idx].mean()
            emp_sd = max(raw[idx].std(ddof=1), EPS)
            mean[s] = (n_s * emp_mean + shrink * global_mean) / (n_s + shrink)
            sd[s] = max((n_s * emp_sd + shrink * global_sd) / (n_s + shrink), EPS)
        else:
            mean[s] = global_mean
            sd[s] = global_sd
        z[idx] = (raw[idx] - mean[s]) / sd[s]
    rho = 0.0
    if len(z) > 2 and np.dot(z[:-1], z[:-1]) > EPS:
        rho = float(np.dot(z[:-1], z[1:]) / max(np.dot(z[:-1], z[:-1]), EPS))
    rho = float(np.clip(rho, -0.75, 0.85))
    resid = z.copy()
    resid[1:] = z[1:] - rho * z[:-1]
    resid_sd = max(float(resid[1:].std(ddof=1)), EPS) if len(resid) > 2 else 1.0
    return ScalarFit(mean.tolist(), sd.tolist(), rho, resid_sd), resid / resid_sd


def apply_scalar_transform(raw: np.ndarray, templates: np.ndarray, fit: ScalarFit) -> np.ndarray:
    mean = np.asarray(fit.mean_by_template)
    sd = np.asarray(fit.sd_by_template)
    z = (raw - mean[templates]) / np.maximum(sd[templates], EPS)
    resid = z.copy()
    resid[1:] = z[1:] - fit.rho * z[:-1]
    return resid / max(fit.resid_sd, EPS)


def fit_vector_standardizer(x: np.ndarray) -> tuple[VectorFit, np.ndarray]:
    mean = x.mean(axis=0)
    sd = np.maximum(x.std(axis=0, ddof=1), EPS)
    return VectorFit(mean.tolist(), sd.tolist()), (x - mean) / sd


def fit_moving_range_standardizer(x: np.ndarray) -> tuple[VectorFit, np.ndarray]:
    mean = x.mean(axis=0)
    if x.shape[0] > 1:
        mr = np.abs(np.diff(x, axis=0)).mean(axis=0)
        sd = np.maximum(math.sqrt(math.pi) * mr / 2.0, EPS)
    else:
        sd = np.maximum(x.std(axis=0, ddof=0), EPS)
    return VectorFit(mean.tolist(), sd.tolist()), (x - mean) / sd


def apply_vector_standardizer(x: np.ndarray, fit: VectorFit) -> np.ndarray:
    return (x - np.asarray(fit.mean)) / np.maximum(np.asarray(fit.sd), EPS)


def fit_ten_model(a_phase: np.ndarray, templates: np.ndarray, case: CaseParams, cfg: SimConfig) -> dict[str, Any]:
    x = np.log1p(a_phase)
    tmpl_mean = np.zeros((a_phase.shape[0], cfg.templates), dtype=float)
    global_mean = x.mean(axis=1)
    for s in range(cfg.templates):
        idx = templates == s
        tmpl_mean[:, s] = x[:, idx].mean(axis=1) if idx.sum() > 0 else global_mean
    xc = x - tmpl_mean[:, templates]
    neigh = edge_neighbor_average(xc, case)
    y = xc[:, 1:].ravel()
    design = np.vstack([xc[:, :-1].ravel(), neigh[:, :-1].ravel(), np.ones(y.size)]).T
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    resid_edge = resid.reshape(a_phase.shape[0], -1)
    sq = resid_edge**2
    sq_mean = sq.mean(axis=1)
    sq_sd = np.maximum(sq.std(axis=1, ddof=1), EPS)
    local_ucl = np.quantile((sq - sq_mean[:, None]) / sq_sd[:, None], cfg.ten_local_quantile, axis=1)
    local_ucl = np.maximum(local_ucl, 1.0)
    return {
        "template_mean": tmpl_mean.tolist(),
        "coef": coef.tolist(),
        "sq_mean": sq_mean.tolist(),
        "sq_sd": sq_sd.tolist(),
        "local_ucl": local_ucl.tolist(),
    }


def edge_neighbor_average(x: np.ndarray, case: CaseParams) -> np.ndarray:
    e, t_len = x.shape
    node_sum = np.zeros((case.n, t_len), dtype=float)
    node_count = np.zeros(case.n, dtype=float)
    for idx, (i, j) in enumerate(zip(case.pi, case.pj)):
        node_sum[i] += x[idx]
        node_sum[j] += x[idx]
        node_count[i] += 1
        node_count[j] += 1
    out = np.empty_like(x)
    for idx, (i, j) in enumerate(zip(case.pi, case.pj)):
        total = node_sum[i] + node_sum[j] - 2.0 * x[idx]
        count = max(node_count[i] + node_count[j] - 2.0, 1.0)
        out[idx] = total / count
    return out


def ten_increment_series(a: np.ndarray, templates: np.ndarray, case: CaseParams, fit: dict[str, Any]) -> np.ndarray:
    tmpl_mean = np.asarray(fit["template_mean"])
    coef = np.asarray(fit["coef"])
    sq_mean = np.asarray(fit["sq_mean"])
    sq_sd = np.asarray(fit["sq_sd"])
    x = np.log1p(a)
    xc = x - tmpl_mean[:, templates]
    neigh = edge_neighbor_average(xc, case)
    pred = coef[0] * xc[:, :-1] + coef[1] * neigh[:, :-1] + coef[2]
    resid = xc[:, 1:] - pred
    inc = (resid**2 - sq_mean[:, None]) / np.maximum(sq_sd[:, None], EPS)
    if inc.shape[1] == 0:
        return np.zeros((a.shape[0], 1), dtype=float)
    return inc


def sample_indices(rng: np.random.Generator, pool_len: int, reps: int, max_len: int, block_len: int) -> np.ndarray:
    idx = np.empty((reps, max_len), dtype=np.int32)
    pos = 0
    starts = rng.integers(0, pool_len, size=reps)
    offsets = np.zeros(reps, dtype=np.int32)
    while pos < max_len:
        take = min(block_len, max_len - pos)
        block = (starts[:, None] + offsets[:, None] + np.arange(take)[None, :]) % pool_len
        idx[:, pos : pos + take] = block
        pos += take
        starts = rng.integers(0, pool_len, size=reps)
        offsets[:] = 0
    return idx


def scalar_chart_states(x_pool: np.ndarray, idx: np.ndarray, chart: str, parameter: float) -> np.ndarray:
    reps, max_len = idx.shape
    state = np.zeros(reps, dtype=np.float32)
    out = np.empty((reps, max_len), dtype=np.float32)
    for t in range(max_len):
        x = x_pool[idx[:, t]].astype(np.float32)
        if chart == "cusum":
            state = np.maximum(0.0, state + x - parameter)
        elif chart == "ewma":
            state = np.maximum(0.0, parameter * x + (1.0 - parameter) * state)
        else:
            raise ValueError(chart)
        out[:, t] = state
    return out


def vector_ewma_states(z_pool: np.ndarray, idx: np.ndarray, lam: float) -> np.ndarray:
    reps, max_len = idx.shape
    p = z_pool.shape[1]
    state = np.zeros((reps, p), dtype=np.float32)
    out = np.empty((reps, max_len), dtype=np.float32)
    z_pool32 = z_pool.astype(np.float32)
    for t in range(max_len):
        z = z_pool32[idx[:, t]]
        state = lam * z + (1.0 - lam) * state
        out[:, t] = np.max(np.abs(state), axis=1)
    return out


def ss_cusum_states(cat_pool: np.ndarray, f0: np.ndarray, idx: np.ndarray, k_ss: float) -> np.ndarray:
    reps, max_len = idx.shape
    q = len(f0)
    sobs = np.zeros((reps, q), dtype=np.float32)
    sexp = np.zeros((reps, q), dtype=np.float32)
    out = np.empty((reps, max_len), dtype=np.float32)
    f0 = f0.astype(np.float32)
    for t in range(max_len):
        cats = cat_pool[idx[:, t]]
        g = np.zeros((reps, q), dtype=np.float32)
        g[np.arange(reps), cats] = 1.0
        delta = sobs - sexp + g - f0
        u = ((delta * delta) / np.maximum(sexp + f0, EPS)).sum(axis=1)
        active = u > k_ss
        inc = np.zeros(reps, dtype=np.float32)
        inc[active] = (u[active] - k_ss) / np.maximum(u[active], EPS)
        sobs += g * inc[:, None]
        sexp += f0[None, :] * inc[:, None]
        c = ((sobs - sexp) ** 2 / np.maximum(sexp, EPS)).sum(axis=1)
        reset = ~active
        if reset.any():
            sobs[reset] = 0.0
            sexp[reset] = 0.0
            c[reset] = 0.0
        out[:, t] = c
    return out


def ten_states(inc_pool: np.ndarray, idx: np.ndarray, local_ucl: np.ndarray) -> np.ndarray:
    reps, max_len = idx.shape
    e = inc_pool.shape[0]
    local = np.zeros((reps, e), dtype=np.float32)
    out = np.empty((reps, max_len), dtype=np.float32)
    inc_pool32 = inc_pool.astype(np.float32)
    local_ucl = local_ucl.astype(np.float32)
    for t in range(max_len):
        inc = inc_pool32[:, idx[:, t]].T
        local = np.maximum(0.0, local + inc)
        out[:, t] = np.max(local / local_ucl[None, :], axis=1)
    return out


def run_lengths(states: np.ndarray, h: float) -> np.ndarray:
    exceed = states > h
    hit = exceed.any(axis=1)
    first = exceed.argmax(axis=1) + 1
    return np.where(hit, first, states.shape[1] + 1)


def choose_h(states: np.ndarray, target: float) -> tuple[float, np.ndarray]:
    lo, hi = 0.0, max(float(states.max()), EPS)
    best_h = hi
    best_rl = run_lengths(states, hi)
    best_gap = abs(float(best_rl.mean()) - target)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        rl = run_lengths(states, mid)
        arl = float(rl.mean())
        gap = abs(arl - target)
        if gap < best_gap:
            best_gap = gap
            best_h = mid
            best_rl = rl
        if arl < target:
            lo = mid
        else:
            hi = mid
    return best_h, best_rl


def method_inputs(
    a: np.ndarray,
    templates: np.ndarray,
    case: CaseParams,
    fits: dict[str, MethodFit],
    cfg: SimConfig,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, fit in fits.items():
        d = fit.details or {}
        if name == "Proposed ML-DCMM":
            raw = raw_deviance(a, templates, np.asarray(d["mu_templates"]))
            out[name] = apply_scalar_transform(raw, templates, ScalarFit(**d["scalar_fit"]))
        elif name == "Gravity-WLRT":
            raw = raw_gravity_wlrt(a, templates, np.asarray(d["mu_templates"]), case)
            out[name] = apply_scalar_transform(raw, templates, ScalarFit(**d["scalar_fit"]))
        elif name == "DCSBM-EWMA":
            x = dcsbm_feature_series(a, np.asarray(d["labels"]), case)
            out[name] = apply_vector_standardizer(x, VectorFit(**d["vector_fit"]))
        elif name == "SS-MCUSUM":
            x = ss_feature_series(a, case)
            z = apply_vector_standardizer(x, VectorFit(**d["feature_fit"]))
            med = np.asarray(d["medians"])
            bits = (z > med).astype(np.int16)
            powers = 1 << np.arange(bits.shape[1], dtype=np.int16)
            out[name] = (bits * powers).sum(axis=1).astype(np.int16)
        elif name == "TEN-GNARX-CUSUM":
            out[name] = ten_increment_series(a, templates, case, d["ten_fit"])
        else:
            raise ValueError(name)
    return out


def fit_methods(a_phase: np.ndarray, templates: np.ndarray, case: CaseParams, cfg: SimConfig) -> dict[str, MethodFit]:
    fits: dict[str, MethodFit] = {}

    if method_is_enabled("Proposed ML-DCMM", cfg):
        mu_mldcmm, z_hat = fit_mldcmm_mu(a_phase, templates, case, cfg)
        raw = raw_deviance(a_phase, templates, mu_mldcmm)
        scalar_fit, _ = fit_scalar_transform(raw, templates, cfg)
        details = {"mu_templates": mu_mldcmm.tolist(), "z_hat": z_hat.tolist(), "scalar_fit": asdict(scalar_fit)}
        if cfg.calibration_mode == "crossfit":
            details["crossfit_calibration_input"] = crossfit_proposed_calibration_input(
                a_phase, templates, case, cfg
            ).tolist()
        fits["Proposed ML-DCMM"] = MethodFit(
            "Proposed ML-DCMM",
            "cusum",
            cfg.k_cusum,
            details=details,
        )

    if method_is_enabled("Gravity-WLRT", cfg):
        mu_gravity = fit_gravity_mu(a_phase, templates, case, cfg)
        raw = raw_gravity_wlrt(a_phase, templates, mu_gravity, case)
        scalar_fit, _ = fit_scalar_transform(raw, templates, cfg)
        fits["Gravity-WLRT"] = MethodFit(
            "Gravity-WLRT",
            "ewma",
            cfg.lambda_gravity,
            details={"mu_templates": mu_gravity.tolist(), "scalar_fit": asdict(scalar_fit)},
        )

    hard_labels = estimate_hard_labels_from_phase(a_phase, templates, case, cfg)
    if method_is_enabled("DCSBM-EWMA", cfg):
        mu_dcsbm = fit_dcsbm_mu(a_phase, templates, hard_labels, case, cfg)
        x_d = dcsbm_feature_series(a_phase, hard_labels, case)
        vector_fit, _ = fit_moving_range_standardizer(x_d)
        fits["DCSBM-EWMA"] = MethodFit(
            "DCSBM-EWMA",
            "vector-ewma",
            cfg.lambda_dcsbm,
            details={"mu_templates": mu_dcsbm.tolist(), "labels": hard_labels.tolist(), "vector_fit": asdict(vector_fit)},
        )

    if method_is_enabled("SS-MCUSUM", cfg):
        x_ss = ss_feature_series(a_phase, case)
        feature_fit, z_ss = fit_vector_standardizer(x_ss)
        med = np.median(z_ss, axis=0)
        bits = (z_ss > med).astype(np.int16)
        powers = 1 << np.arange(bits.shape[1], dtype=np.int16)
        cats = (bits * powers).sum(axis=1).astype(np.int16)
        q = 2 ** z_ss.shape[1]
        counts = np.bincount(cats, minlength=q).astype(float) + 0.5
        f0 = counts / counts.sum()
        fits["SS-MCUSUM"] = MethodFit(
            "SS-MCUSUM",
            "ss-mcusum",
            cfg.k_ss,
            details={"feature_fit": asdict(feature_fit), "medians": med.tolist(), "f0": f0.tolist()},
        )

    if method_is_enabled("TEN-GNARX-CUSUM", cfg):
        ten_fit = fit_ten_model(a_phase, templates, case, cfg)
        fits["TEN-GNARX-CUSUM"] = MethodFit(
            "TEN-GNARX-CUSUM",
            "ten-cusum",
            None,
            details={"ten_fit": ten_fit},
        )
    return fits


def calibration_pool_for_method(
    name: str,
    fit: MethodFit,
    phase_inputs: dict[str, Any],
    templates_pool: np.ndarray,
    case: CaseParams,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> Any:
    details = fit.details or {}
    if cfg.calibration_mode == "oracle":
        a_cal = generate_counts(case.mu_templates, templates_pool, cfg, rng)
        return method_inputs(a_cal, templates_pool, case, {name: fit}, cfg)[name]
    if cfg.calibration_mode == "crossfit" and name == "Proposed ML-DCMM":
        if "crossfit_calibration_input" in details:
            return np.asarray(details["crossfit_calibration_input"], dtype=float)
    if cfg.calibration_mode == "empirical" or name in {"SS-MCUSUM", "TEN-GNARX-CUSUM"}:
        if name in phase_inputs:
            return phase_inputs[name]
    elif cfg.calibration_mode in {"parametric", "hybrid"}:
        if name in {"Proposed ML-DCMM", "Gravity-WLRT"}:
            mu = np.asarray(details["mu_templates"])
            a_cal = generate_counts(mu, templates_pool, cfg, rng)
            return method_inputs(a_cal, templates_pool, case, {name: fit}, cfg)[name]
        if name == "DCSBM-EWMA" and name in phase_inputs:
            return phase_inputs[name]
    raise ValueError(name)


def make_states(
    name: str,
    chart: str,
    parameter: float | None,
    inputs: Any,
    fit: MethodFit,
    idx: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    if chart in {"cusum", "ewma"}:
        return scalar_chart_states(np.asarray(inputs), idx, chart, float(parameter))
    if chart == "vector-ewma":
        return vector_ewma_states(np.asarray(inputs), idx, float(parameter))
    if chart == "ss-mcusum":
        f0 = np.asarray((fit.details or {})["f0"], dtype=float)
        return ss_cusum_states(np.asarray(inputs), f0, idx, cfg.k_ss)
    if chart == "ten-cusum":
        local_ucl = np.asarray((fit.details or {})["ten_fit"]["local_ucl"], dtype=float)
        return ten_states(np.asarray(inputs), idx, local_ucl)
    raise ValueError(chart)


def summarize_rl(rl: np.ndarray) -> dict[str, float]:
    n = len(rl)
    return {
        "n": int(n),
        "sum": float(rl.sum()),
        "sumsq": float(np.dot(rl.astype(float), rl.astype(float))),
        "arl": float(rl.mean()),
        "sd": float(rl.std(ddof=1)),
        "se": float(rl.std(ddof=1) / math.sqrt(n)),
        "median": float(np.median(rl)),
        "censored": int((rl == rl.max()).sum()) if rl.max() > 0 else 0,
    }


def run_case_m0(
    case_id: str,
    m0: int,
    cfg: SimConfig,
    outdir: Path,
    phase_rep: int = 0,
    case: CaseParams | None = None,
    a_phase_full: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    combo_seed = cfg.seed + 1009 * (ord(case_id[0]) - ord("A") + 1) + 37 * m0 + 100003 * phase_rep
    rng = np.random.default_rng(combo_seed)
    if case is None:
        case = build_case(case_id, cfg, cfg.seed + 97 * (ord(case_id[0]) - ord("A") + 1))
    templates_phase = template_sequence(m0, offset=0, templates=cfg.templates)
    if a_phase_full is None:
        a_phase = generate_counts(case.mu_templates, templates_phase, cfg, rng)
    else:
        a_phase = a_phase_full[:, :m0]

    t0 = time.time()
    fits = fit_methods(a_phase, templates_phase, case, cfg)
    phase_inputs = method_inputs(a_phase, templates_phase, case, fits, cfg)

    templates_pool = template_sequence(cfg.pool_len, offset=m0 + 101, templates=cfg.templates)
    val_seed_base = combo_seed
    if cfg.common_validation:
        val_seed_base = cfg.seed + 8887 * (ord(case_id[0]) - ord("A") + 1) + 100003 * phase_rep
    val_inputs = None
    if cfg.validation_mode == "true":
        a_val = generate_counts(case.mu_templates, templates_pool, cfg, np.random.default_rng(val_seed_base))
        val_inputs = method_inputs(a_val, templates_pool, case, fits, cfg)

    rows = []
    details_out: dict[str, Any] = {
        "case": case_id,
        "m0": m0,
        "n": case.n,
        "K": case.k,
        "seed": combo_seed,
        "methods": {},
    }

    for method_index, (name, fit) in enumerate(fits.items()):
        print(f"[{case_id}, m0={m0}] calibrating {name}", flush=True)
        cal_pool = calibration_pool_for_method(name, fit, phase_inputs, templates_pool, case, cfg, rng)

        cal_len = cal_pool.shape[1] if name == "TEN-GNARX-CUSUM" else len(cal_pool)
        if cfg.validation_mode == "fitted" and "mu_templates" in (fit.details or {}):
            mu_val = np.asarray((fit.details or {})["mu_templates"], dtype=float)
            a_val = generate_counts(
                mu_val,
                templates_pool,
                cfg,
                np.random.default_rng(val_seed_base + 17003 + method_index),
            )
            val_pool = method_inputs(a_val, templates_pool, case, {name: fit}, cfg)[name]
        elif val_inputs is not None:
            val_pool = val_inputs[name]
        else:
            raise ValueError(f"validation_mode={cfg.validation_mode} is unavailable for {name}")
        val_len = val_pool.shape[1] if name == "TEN-GNARX-CUSUM" else len(val_pool)
        max_len = cfg.max_len
        if cal_len < 5 or val_len < 5:
            raise RuntimeError(f"pool too short for {name}")

        idx_cal = sample_indices(
            np.random.default_rng(combo_seed + 5000 + method_index),
            cal_len,
            cfg.calibration_reps,
            max_len,
            cfg.block_len,
        )
        states_cal = make_states(name, fit.chart, fit.parameter, cal_pool, fit, idx_cal, cfg)
        h, rl_cal = choose_h(states_cal, cfg.target_arl0)

        idx_val = sample_indices(
            np.random.default_rng(val_seed_base + 7000 + method_index),
            val_len,
            cfg.reps,
            max_len,
            cfg.block_len,
        )
        states_val = make_states(name, fit.chart, fit.parameter, val_pool, fit, idx_val, cfg)
        rl_val = run_lengths(states_val, h)
        cal_sum = summarize_rl(rl_cal)
        val_sum = summarize_rl(rl_val)

        fit.h = h
        rows.append(
            {
                "case": case_id,
                "n": case.n,
                "m0": m0,
                "phase_rep": phase_rep,
                "method": name,
                "chart": fit.chart,
                "chart_parameter": fit.parameter,
                "h": h,
                "calibrated_n": cal_sum["n"],
                "calibrated_sum": cal_sum["sum"],
                "calibrated_sumsq": cal_sum["sumsq"],
                "calibrated_arl0": cal_sum["arl"],
                "calibrated_se": cal_sum["se"],
                "actual_n": val_sum["n"],
                "actual_sum": val_sum["sum"],
                "actual_sumsq": val_sum["sumsq"],
                "actual_arl0": val_sum["arl"],
                "actual_sd": val_sum["sd"],
                "actual_rl_se": val_sum["se"],
                "actual_median": val_sum["median"],
                "abs_error": abs(val_sum["arl"] - cfg.target_arl0),
                "reps": cfg.reps,
                "pool_len": cfg.pool_len,
                "max_len": max_len,
                "elapsed_sec_combo_so_far": time.time() - t0,
            }
        )
        details_out["methods"][name] = {
            "chart": fit.chart,
            "parameter": fit.parameter,
            "h": h,
            "calibrated": cal_sum,
            "actual": val_sum,
        }
        del states_cal, states_val

    df = pd.DataFrame(rows)
    if cfg.write_details:
        detail_dir = outdir / "details"
        detail_dir.mkdir(parents=True, exist_ok=True)
        detail_path = detail_dir / f"detail_case_{case_id}_m0_{m0}.json"
        if cfg.phase_reps > 1:
            detail_path = detail_dir / f"detail_case_{case_id}_m0_{m0}_phase_{phase_rep:03d}.json"
        detail_path.write_text(json.dumps(details_out, indent=2, ensure_ascii=False), encoding="utf-8")
    return df, details_out


def write_parameter_notes(outdir: Path) -> None:
    notes = """# Method Parameter Notes for Phase I ARL0 Simulation

This file records the chart-parameter choices used in `run_phase1_arl0_simulation.py`.

| Method | Parameter used in this simulation | Source-paper basis |
|---|---:|---|
| Proposed ML-DCMM CUSUM | `k = 0.50` | Page-type CUSUM with reference value `k = delta/2`; `delta = 1` means a one-standard-deviation sustained upward shift in the decorrelated global deviance input. |
| Gravity-WLRT | `lambda = 0.10` | Li, Qin, Wu and Tsung recommend `lambda` in `[0.05, 0.2]` and use `lambda = 0.1` in their simulation and real-data examples. DOI: https://doi.org/10.1080/16843703.2025.2543705 |
| DCSBM-EWMA | `lambda = 0.20` | Wilson, Stevens and Woodall monitor separate DCSBM parameter estimates, including block propensities `P_rs` and within-community degree-heterogeneity summaries `s_r`, with EWMA smoothing. They cite the usual EWMA range `[0.05, 0.25]` and use `lambda = 0.2` in the DCSBM application. DOI: https://doi.org/10.1002/qre.2520 |
| SS-MCUSUM | `k = 0.10`, `bmax = 20` in the source method | Wang, Xie and Qiu use a multivariate nonparametric CUSUM with pre-specified allowance constant `k`; their numerical examples use `k = 0.1` and recommend `bmax` in `[10, 20]`, taking `bmax = 20`. DOI: https://doi.org/10.1080/00224065.2024.2310843 |
| TEN-GNARX-CUSUM | residual squared Page-CUSUM, no ordinary `k = delta/2` | Malinovskaya et al. use GNARX residuals and a Page-type CUSUM for centered squared deviance residuals with `nu = 0`; local limits are based on residual variance/critical values. DOI: https://doi.org/10.1093/jrsssc/qlag003 |

For Phase I ARL0 comparison, all charts are calibrated to the same nominal `ARL0 = 200`.
The table reports AARL0 values. Parentheses follow the requested reporting mode:
pooled run-length Monte Carlo standard errors for Table-2-style IC simulations,
or standard errors across repeated Phase-I samples for finite-Phase-I stability checks.
"""
    (outdir / "method_parameter_notes.md").write_text(notes, encoding="utf-8")


def format_table(df: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    tmp = aggregate_results(df, cfg)
    se_col = "actual_se" if cfg.table_se == "phase" else "pooled_run_length_se"
    tmp["value"] = tmp.apply(lambda r: f"{r['actual_arl0']:.2f} ({r[se_col]:.2f})", axis=1)
    order = ["Proposed ML-DCMM", "Gravity-WLRT", "DCSBM-EWMA", "SS-MCUSUM", "TEN-GNARX-CUSUM"]
    tmp["method"] = pd.Categorical(tmp["method"], categories=order, ordered=True)
    pivot = tmp.pivot_table(index=["case", "method"], columns="m0", values="value", aggfunc="first", observed=False)
    return pivot.reset_index()


def write_summary(outdir: Path, df: pd.DataFrame, cfg: SimConfig) -> None:
    table = format_table(df, cfg)
    try:
        table_text = table.to_markdown(index=False)
    except ImportError:
        table_text = table.to_string(index=False)
    case_names = ", ".join(str(x) for x in sorted(df["case"].unique()))
    m0_names = ", ".join(str(int(x)) for x in sorted(df["m0"].unique()))
    lines = [
        "# Phase I ARL0 Simulation Summary",
        "",
        "## Run Setting",
        "",
        f"- Nominal ARL0: {cfg.target_arl0:g}",
        (
            f"- Monte Carlo size: {cfg.phase_reps} independent Phase-I samples, "
            f"{cfg.reps} validation run-length paths per Phase-I sample "
            f"({cfg.phase_reps * cfg.reps} run lengths per setting)"
        ),
        f"- Control-limit calibration: {cfg.calibration_reps} IC bootstrap paths for each Phase-I sample",
        f"- Cases: {case_names}",
        f"- Phase I sample sizes: m0 = {m0_names}",
        "- Data generator: undirected ML-DCMM with mixed memberships, degree heterogeneity, Poisson working counts, and an AR(1) common time-intensity factor",
        f"- Temporal dependence: phi = {cfg.phi:g}, eta_sd = {cfg.eta_sd:g}",
        f"- Main simulation templates: {cfg.templates}. These templates represent recurring within-day operating regimes.",
        "",
        "## Chart Parameters",
        "",
        f"- Proposed ML-DCMM CUSUM: k = {cfg.k_cusum:g}",
        f"- Gravity-WLRT: lambda = {cfg.lambda_gravity:g}",
        f"- DCSBM-EWMA: lambda = {cfg.lambda_dcsbm:g}; monitored statistics are DCSBM parameter estimates P_rs and s_r",
        f"- SS-MCUSUM: k = {cfg.k_ss:g}",
        "- TEN-GNARX-CUSUM: residual squared Page-CUSUM; no ordinary k = delta/2 parameter",
        "",
        "## Estimated Actual ARL0",
        "",
        (
            "Values are AARL0 estimates. Parentheses contain standard errors across independent "
            "Phase-I samples."
            if cfg.table_se == "phase"
            else "Values are AARL0 estimates. Pooled run-length Monte Carlo standard errors are in parentheses."
        ),
        "",
        table_text,
        "",
    ]
    (outdir / "phase1_arl0_summary.md").write_text("\n".join(lines), encoding="utf-8")


def aggregate_results(df: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    rows = []
    group_cols = ["case", "n", "m0", "method", "chart", "chart_parameter"]
    for keys, g in df.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, keys))
        phase_means = g["actual_arl0"].to_numpy(dtype=float)
        mean = float(phase_means.mean())
        phase_sd = float(phase_means.std(ddof=1)) if len(phase_means) > 1 else 0.0
        phase_se = phase_sd / math.sqrt(max(len(phase_means), 1))
        rec.update(
            {
                "actual_n": int(g["actual_n"].sum()),
                "actual_arl0": mean,
                "actual_sd": phase_sd,
                "actual_se": phase_se,
                "pooled_run_length_se": float(
                    math.sqrt(
                        max(
                            (
                                float(g["actual_sumsq"].sum())
                                - float(g["actual_sum"].sum()) ** 2 / max(float(g["actual_n"].sum()), 1.0)
                            )
                            / max(float(g["actual_n"].sum()) - 1.0, 1.0),
                            0.0,
                        )
                    )
                    / math.sqrt(max(float(g["actual_n"].sum()), 1.0))
                ),
                "abs_error": abs(mean - cfg.target_arl0),
                "phase_reps": int(g["phase_rep"].nunique()),
                "mean_h": float(g["h"].mean()),
                "sd_h": float(g["h"].std(ddof=1)) if len(g) > 1 else 0.0,
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="I,II,III")
    parser.add_argument("--m0", default="500,1000,1500,2000,3000")
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--calibration-reps", type=int, default=10000)
    parser.add_argument("--pool-len", type=int, default=24000)
    parser.add_argument("--max-len", type=int, default=1500)
    parser.add_argument("--block-len", type=int, default=24)
    parser.add_argument("--target-arl0", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--phase-reps", type=int, default=1)
    parser.add_argument("--avg-edge-mean", type=float, default=2.00)
    parser.add_argument("--shrink-templates", type=float, default=4.0)
    parser.add_argument("--templates", type=int, default=4)
    parser.add_argument("--phi", type=float, default=0.25)
    parser.add_argument("--eta-sd", type=float, default=0.05)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--calibration-mode", choices=["empirical", "parametric", "hybrid", "oracle", "crossfit"], default="hybrid")
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--n-i", type=int, default=30)
    parser.add_argument("--n-ii", type=int, default=60)
    parser.add_argument("--n-iii", type=int, default=60)
    parser.add_argument("--case-i-scale", type=float, default=1.8)
    parser.add_argument("--case-ii-scale", type=float, default=1.0)
    parser.add_argument("--case-iii-scale", type=float, default=1.0)
    parser.add_argument("--pure-fraction", type=float, default=0.35)
    parser.add_argument("--mix-main", type=float, default=4.0)
    parser.add_argument("--mix-off", type=float, default=0.5)
    parser.add_argument("--theta-power", type=float, default=0.5)
    parser.add_argument("--theta-floor", type=float, default=0.08)
    parser.add_argument("--mldcmm-iters", type=int, default=12)
    parser.add_argument("--mldcmm-b-steps", type=int, default=4)
    parser.add_argument("--no-details", action="store_true")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--common-validation", action="store_true")
    parser.add_argument("--table-se", choices=["phase", "pooled"], default="pooled")
    parser.add_argument("--validation-mode", choices=["true", "fitted"], default="true")
    parser.add_argument("--outdir", default=str(OUT_DEFAULT))
    args = parser.parse_args()

    requested_total_reps = args.reps
    phase_reps = max(int(args.phase_reps), 1)
    reps_per_phase = int(math.ceil(requested_total_reps / phase_reps))
    cfg = SimConfig(
        target_arl0=args.target_arl0,
        reps=reps_per_phase,
        calibration_reps=args.calibration_reps,
        phase_reps=phase_reps,
        pool_len=args.pool_len,
        max_len=args.max_len,
        block_len=max(int(args.block_len), 1),
        seed=args.seed,
        avg_edge_mean=args.avg_edge_mean,
        shrink_templates=args.shrink_templates,
        templates=args.templates,
        phi=args.phi,
        eta_sd=args.eta_sd,
        calibration_mode=args.calibration_mode,
        crossfit_folds=args.crossfit_folds,
        n_i=args.n_i,
        n_ii=args.n_ii,
        n_iii=args.n_iii,
        case_i_scale=args.case_i_scale,
        case_ii_scale=args.case_ii_scale,
        case_iii_scale=args.case_iii_scale,
        pure_fraction=args.pure_fraction,
        mix_main=args.mix_main,
        mix_off=args.mix_off,
        theta_power=args.theta_power,
        theta_floor=args.theta_floor,
        mldcmm_iters=args.mldcmm_iters,
        mldcmm_b_steps=args.mldcmm_b_steps,
        write_details=not args.no_details,
        methods=args.methods,
        common_validation=args.common_validation,
        table_se=args.table_se,
        validation_mode=args.validation_mode,
    )
    if args.quick:
        cfg.reps = min(cfg.reps, 300)
        cfg.calibration_reps = min(cfg.calibration_reps, 300)
        cfg.pool_len = min(cfg.pool_len, 1600)
        cfg.max_len = min(cfg.max_len, 500)
        cfg.phase_reps = min(cfg.phase_reps, 2)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_parameter_notes(outdir)
    (outdir / "settings_phase1_arl0.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cases = parse_str_list(args.cases)
    m0s = parse_int_list(args.m0)
    all_rows = []
    start = time.time()
    max_m0 = max(m0s)
    case_params = {
        case_id: build_case(case_id, cfg, cfg.seed + 97 * (ord(case_id[0]) - ord("A") + 1))
        for case_id in cases
    }
    phase_cache: dict[tuple[str, int], np.ndarray] = {}
    for case_id in cases:
        for phase_rep in range(cfg.phase_reps):
            phase_seed = cfg.seed + 1009 * (ord(case_id[0]) - ord("A") + 1) + 100003 * phase_rep
            phase_rng = np.random.default_rng(phase_seed)
            templates_full = template_sequence(max_m0, offset=0, templates=cfg.templates)
            phase_cache[(case_id, phase_rep)] = generate_counts(
                case_params[case_id].mu_templates,
                templates_full,
                cfg,
                phase_rng,
            )
        for m0 in m0s:
            for phase_rep in range(cfg.phase_reps):
                df, _ = run_case_m0(
                    case_id,
                    m0,
                    cfg,
                    outdir,
                    phase_rep=phase_rep,
                    case=case_params[case_id],
                    a_phase_full=phase_cache[(case_id, phase_rep)],
                )
                all_rows.append(df)
                current = pd.concat(all_rows, ignore_index=True)
                current.to_csv(outdir / "phase1_arl0_results_long.csv", index=False)
                aggregate_results(current, cfg).to_csv(outdir / "phase1_arl0_results_aggregated.csv", index=False)
                format_table(current, cfg).to_csv(outdir / "phase1_arl0_table_formatted.csv", index=False)
                write_summary(outdir, current, cfg)
                print(
                    f"finished case={case_id}, m0={m0}, phase_rep={phase_rep + 1}/{cfg.phase_reps}, "
                    f"elapsed={(time.time() - start)/60:.2f} min",
                    flush=True,
                )

    final = pd.concat(all_rows, ignore_index=True)
    final.to_csv(outdir / "phase1_arl0_results_long.csv", index=False)
    aggregate_results(final, cfg).to_csv(outdir / "phase1_arl0_results_aggregated.csv", index=False)
    format_table(final, cfg).to_csv(outdir / "phase1_arl0_table_formatted.csv", index=False)
    write_summary(outdir, final, cfg)
    summary = {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - start,
        "config": asdict(cfg),
        "requested_total_reps_per_case_m0_method": requested_total_reps,
        "actual_total_reps_per_case_m0_method": cfg.reps * cfg.phase_reps,
        "phase_reps_for_aarl_se": cfg.phase_reps,
        "pid": os.getpid(),
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
