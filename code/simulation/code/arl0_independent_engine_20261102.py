"""Exact independent DGP validation for several fixed DMM-CUSUM charts.

Every seed starts a new, stationary lognormal-Poisson network process. The
same counts are supplied to all charts (common random numbers across m0),
but no observations are pooled or reused between independent future paths.
Paths run until all charts signal. The guard is an error, never censoring.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(cache=True, inline='always')
def _draw_network(mu, pi, pj, node, exponentials, common, phi, ex, ev, center, t, s, counts):
    if t > 0:
        common = phi * common + np.random.normal(0., ex)
        for i in range(len(node)):
            node[i] = phi * node[i] + np.random.normal(0., ev)
    common_exp = math.exp(common - center)
    for i in range(len(node)):
        exponentials[i] = math.exp(node[i])
    for e in range(len(pi)):
        rate = mu[s, e] * common_exp * exponentials[pi[e]] * exponentials[pj[e]]
        counts[e] = np.random.poisson(rate)
    return common


@njit(cache=True, inline='always')
def _update_charts(counts, s, log_mu, sum_mu, means, sds, beta, variance, bmaxes,
                   k, hs, lagged, cp, cm, spring, runs, t):
    # sum c(log c - 1) is common to all charts; log(mu) is precomputed.
    entropy = 0.
    for e in range(len(counts)):
        c = counts[e]
        if c > 0:
            entropy += c * (math.log(c) - 1.)
    remaining = 0
    for chart in range(len(runs)):
        if runs[chart] > 0:
            continue
        weighted = 0.
        for e in range(len(counts)):
            weighted += counts[e] * log_mu[chart, s, e]
        raw = 2. * (entropy - weighted + sum_mu[chart, s])
        value = (raw - means[chart, s]) / sds[chart, s]
        p = spring[chart]
        residual = value
        for lag in range(p):
            residual -= beta[chart, p, lag] * lagged[chart, lag]
        u = residual / math.sqrt(variance[chart, p])
        cp[chart] = max(0., cp[chart] + u - k)
        cm[chart] = max(0., cm[chart] - u - k)
        statistic = max(cp[chart], cm[chart])
        if statistic > hs[chart]:
            runs[chart] = t + 1
            continue
        remaining += 1
        for lag in range(bmaxes[chart] - 1, 0, -1):
            lagged[chart, lag] = lagged[chart, lag - 1]
        lagged[chart, 0] = value
        spring[chart] = min(p + 1, bmaxes[chart]) if statistic > 0. else 0
    return remaining


@njit(cache=True)
def _one_run(mu, log_mu, sum_mu, pi, pj, n, seed, phi, sg, sv, means, sds,
             beta, variance, bmaxes, k, hs, start_phase, guard):
    np.random.seed(seed)
    common = np.random.normal(0., sg)
    node = np.random.normal(0., sv, n)
    exponentials = np.empty(n)
    counts = np.empty(len(pi), dtype=np.int64)
    charts = len(hs)
    cp, cm = np.zeros(charts), np.zeros(charts)
    spring = np.zeros(charts, dtype=np.int64)
    lagged = np.zeros((charts, max(1, beta.shape[2])))
    runs = np.full(charts, -1, dtype=np.int64)
    scale = math.sqrt(1. - phi * phi)
    ex, ev = sg * scale, sv * scale
    center = .5 * (sg * sg + 2. * sv * sv)
    for t in range(guard):
        s = (t + start_phase) % mu.shape[0]
        common = _draw_network(mu, pi, pj, node, exponentials, common, phi, ex, ev,
                               center, t, s, counts)
        remaining = _update_charts(counts, s, log_mu, sum_mu, means, sds, beta,
                                    variance, bmaxes, k, hs, lagged, cp, cm, spring, runs, t)
        if remaining == 0:
            break
    return runs


@njit(cache=True, parallel=True)
def _parallel_runs(mu, log_mu, sum_mu, pi, pj, n, seeds, phi, sg, sv, means, sds,
                   beta, variance, bmaxes, k, hs, start_phase, guard):
    result = np.empty((len(seeds), len(hs)), dtype=np.int64)
    for r in prange(len(seeds)):
        result[r] = _one_run(mu, log_mu, sum_mu, pi, pj, n, seeds[r], phi, sg, sv,
                              means, sds, beta, variance, bmaxes, k, hs, start_phase, guard)
    return result


@njit(cache=True)
def _serial_runs(mu, log_mu, sum_mu, pi, pj, n, seeds, phi, sg, sv, means, sds,
                 beta, variance, bmaxes, k, hs, start_phase, guard):
    result = np.empty((len(seeds), len(hs)), dtype=np.int64)
    for r in range(len(seeds)):
        result[r] = _one_run(mu, log_mu, sum_mu, pi, pj, n, seeds[r], phi, sg, sv,
                              means, sds, beta, variance, bmaxes, k, hs, start_phase, guard)
    return result


def _chart_inputs(muhat_all, template_means, template_sds, betas, variances, bmaxes, hs):
    fitted = np.asarray(muhat_all, dtype=np.float64)
    if fitted.ndim != 3:
        raise ValueError('muhat_all must have shape (charts, edges, templates)')
    c, _, s = fitted.shape
    if c == 0 or s == 0:
        raise ValueError('at least one chart and template are required')
    means, sds = np.asarray(template_means, dtype=float), np.asarray(template_sds, dtype=float)
    beta, variance = np.asarray(betas, dtype=float), np.asarray(variances, dtype=float)
    orders, limits = np.asarray(bmaxes, dtype=np.int64), np.asarray(hs, dtype=float)
    if means.shape != (c, s) or sds.shape != (c, s):
        raise ValueError('template means and SDs must have shape (charts, templates)')
    if orders.shape != (c,) or limits.shape != (c,) or np.any(orders < 0):
        raise ValueError('bmaxes and hs must have one valid entry per chart')
    maximum = int(orders.max())
    if beta.ndim != 3 or beta.shape[0] != c or beta.shape[1] <= maximum or beta.shape[2] < maximum:
        raise ValueError('betas must have padded shape (charts, max_bmax+1, max_bmax)')
    if variance.ndim != 2 or variance.shape[0] != c or variance.shape[1] <= maximum:
        raise ValueError('variances must have padded shape (charts, max_bmax+1)')
    for name, a in [('muhat_all', fitted), ('template_means', means), ('template_sds', sds),
                    ('betas', beta), ('variances', variance), ('hs', limits)]:
        if not np.all(np.isfinite(a)):
            raise ValueError(f'{name} must be finite')
    if np.any(fitted <= 0) or np.any(sds <= 0) or np.any(variance <= 0) or np.any(limits <= 0):
        raise ValueError('fitted means, SDs, variances and control limits must be positive')
    log_mu = np.ascontiguousarray(np.log(fitted).transpose(0, 2, 1))
    sum_mu = np.ascontiguousarray(fitted.sum(axis=1))
    return (log_mu, sum_mu, np.ascontiguousarray(means), np.ascontiguousarray(sds),
            np.ascontiguousarray(beta), np.ascontiguousarray(variance),
            np.ascontiguousarray(orders), np.ascontiguousarray(limits))


def _population_inputs(mu, pi, pj, n, phi, sg, sv):
    mu, pi, pj = np.asarray(mu, dtype=float), np.asarray(pi, dtype=np.int64), np.asarray(pj, dtype=np.int64)
    if mu.ndim != 2 or mu.shape[0] != len(pi) or pi.shape != pj.shape or mu.shape[1] == 0:
        raise ValueError('mu must have shape (edges, templates), matching pi and pj')
    if not np.all(np.isfinite(mu)) or np.any(mu <= 0):
        raise ValueError('population means must be finite and positive')
    if np.any(pi < 0) or np.any(pj < 0) or np.any(pi >= n) or np.any(pj >= n) or np.any(pi >= pj):
        raise ValueError('edge indices must satisfy 0 <= pi < pj < n')
    if not np.isfinite(phi + sg + sv) or abs(phi) >= 1 or min(sg, sv) < 0:
        raise ValueError('require |phi| < 1 and nonnegative stationary latent SDs')
    return np.ascontiguousarray(mu.T), np.ascontiguousarray(pi), np.ascontiguousarray(pj)


def validation_runs(mu, muhat_all, pi, pj, n, seeds, phi, sg, sv,
                    template_means, template_sds, betas, variances, bmaxes, k, hs,
                    start_phase=0, guard=10000000, parallel=True):
    """Return uncensored first-alarm lengths, shape ``(len(seeds), charts)``.

    ``mu`` is (edges, templates), ``muhat_all`` is (charts, edges,
    templates). Prediction arrays may be padded to a shared maximum order.
    ``sg`` and ``sv`` are stationary latent SDs, not innovation SDs.
    All fits, normalizations and limits remain fixed. A guard hit raises.
    """
    population, pi, pj = _population_inputs(mu, pi, pj, n, phi, sg, sv)
    chart = _chart_inputs(muhat_all, template_means, template_sds, betas, variances, bmaxes, hs)
    if np.asarray(muhat_all).shape[1:] != np.asarray(mu).shape:
        raise ValueError('population and fitted network dimensions differ')
    seeds = np.asarray(seeds)
    if seeds.ndim != 1 or len(seeds) == 0 or np.any(seeds < 0) or np.any(seeds > 2**32 - 1):
        raise ValueError('seeds must be a nonempty vector of uint32-compatible integers')
    if not np.issubdtype(seeds.dtype, np.integer) or len(np.unique(seeds)) != len(seeds):
        raise ValueError('each independent path must have a distinct integer seed')
    if not np.isfinite(k) or k < 0 or guard < 1 or int(guard) != guard:
        raise ValueError('k must be nonnegative and guard a positive integer')
    log_mu, sum_mu, means, sds, beta, variance, orders, limits = chart
    kernel = _parallel_runs if parallel else _serial_runs
    result = kernel(population, log_mu, sum_mu, pi, pj, int(n), seeds.astype(np.uint32),
                     phi, sg, sv, means, sds, beta, variance, orders, k, limits,
                     int(start_phase), int(guard))
    bad = np.argwhere(result < 1)
    if len(bad):
        r, c = bad[0]
        raise RuntimeError(f'validation guard {guard} reached without alarm: seed={seeds[r]}, chart={c}; no censored ARL returned')
    return result


@njit(cache=True)
def _sample_network(population, pi, pj, n, length, seed, phi, sg, sv, start_phase):
    np.random.seed(seed)
    common = np.random.normal(0., sg)
    node = np.random.normal(0., sv, n)
    exponentials = np.empty(n)
    counts = np.empty(len(pi), dtype=np.int64)
    result = np.empty((length, len(pi)), dtype=np.int64)
    scale = math.sqrt(1. - phi * phi)
    for t in range(length):
        s = (t + start_phase) % population.shape[0]
        common = _draw_network(population, pi, pj, node, exponentials, common, phi,
                               sg * scale, sv * scale, .5 * (sg * sg + 2 * sv * sv), t, s, counts)
        result[t] = counts
    return result


def sample_network_path(mu, pi, pj, n, length, seed, phi=.5, sg=.1, sv=.1, start_phase=0):
    """Generate a replayable exact DGP path as (time, edges); validation aid."""
    population, pi, pj = _population_inputs(mu, pi, pj, n, phi, sg, sv)
    return _sample_network(population, pi, pj, n, int(length), np.uint32(seed),
                            phi, sg, sv, int(start_phase))


@njit(cache=True)
def _replay_counts(counts, templates, log_mu, sum_mu, means, sds, beta, variance, orders, k, limits):
    charts = len(limits)
    cp, cm = np.zeros(charts), np.zeros(charts)
    spring = np.zeros(charts, dtype=np.int64)
    lagged = np.zeros((charts, max(1, beta.shape[2])))
    runs = np.full(charts, -1, dtype=np.int64)
    for t in range(len(counts)):
        remaining = _update_charts(counts[t], templates[t], log_mu, sum_mu, means, sds,
                                    beta, variance, orders, k, limits, lagged, cp, cm, spring, runs, t)
        if remaining == 0:
            break
    return runs


def runs_from_counts(counts, templates, muhat_all, template_means, template_sds,
                     betas, variances, bmaxes, k, hs):
    """Replay supplied counts; -1 denotes no crossing within this supplied path."""
    chart = _chart_inputs(muhat_all, template_means, template_sds, betas, variances, bmaxes, hs)
    counts, templates = np.asarray(counts), np.asarray(templates, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[1] != np.asarray(muhat_all).shape[1]:
        raise ValueError('counts must have shape (time, edges)')
    if len(templates) != len(counts) or np.any(templates < 0) or np.any(templates >= np.asarray(muhat_all).shape[2]):
        raise ValueError('invalid templates')
    if np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError('counts must be nonnegative integers')
    log_mu, sum_mu, means, sds, beta, variance, orders, limits = chart
    return _replay_counts(np.ascontiguousarray(counts, dtype=np.int64), np.ascontiguousarray(templates),
                           log_mu, sum_mu, means, sds, beta, variance, orders, k, limits)
