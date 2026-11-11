"""Independent, uncensored robustness validation for a fixed DMM-CUSUM fit.

Every seed starts a new stationary latent AR process. Counts and masks are
generated afresh until the first strict alarm; paths never share a data pool.
Kind 0 is conditional Poisson, 1 is mean-preserving compound Poisson,
2 is Poisson with caller-supplied distance-adjusted means, and 3 uses
independent missingness with inverse-probability weighted edge deviances.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

import arl0_independent_engine_20261102 as independent


@njit(cache=True, inline="always")
def _draw_network(mu, pi, pj, node, exponentials, common, phi, ex, ev,
                  center, t, s, counts, weights, kind, level):
    if kind == 0 or kind == 2:
        common = independent._draw_network(
            mu, pi, pj, node, exponentials, common, phi, ex, ev,
            center, t, s, counts,
        )
        return common
    if t > 0:
        common = phi * common + np.random.normal(0., ex)
        for i in range(len(node)):
            node[i] = phi * node[i] + np.random.normal(0., ev)
    common_exp = math.exp(common - center)
    for i in range(len(node)):
        exponentials[i] = math.exp(node[i])
    for e in range(len(pi)):
        rate = mu[s, e] * common_exp * exponentials[pi[e]] * exponentials[pj[e]]
        if kind == 1:
            arrivals = np.random.poisson(rate / (1. + 2. * level))
            counts[e] = arrivals + 2 * np.random.binomial(arrivals, level)
        else:
            counts[e] = np.random.poisson(rate)
    if kind == 3:
        inverse = 1. / (1. - level)
        for e in range(len(pi)):
            weights[e] = inverse if np.random.random() >= level else 0.
    return common


@njit(cache=True, inline="always")
def _raw_deviance(counts, weights, log_mu, sum_mu, muhat, s, kind):
    entropy = 0.
    weighted = 0.
    mean_sum = sum_mu[s] if kind != 3 else 0.
    for e in range(len(counts)):
        count = counts[e]
        weight = weights[e] if kind == 3 else 1.
        if count > 0:
            entropy += weight * count * (math.log(count) - 1.)
        weighted += weight * count * log_mu[s, e]
        if kind == 3:
            mean_sum += weight * muhat[s, e]
    return 2. * (entropy - weighted + mean_sum)


@njit(cache=True, inline="always")
def _update(raw, s, means, sds, beta, variance, bmax, k,
            cp, cm, spring, lagged):
    value = (raw - means[s]) / sds[s]
    residual = value
    for lag in range(spring):
        residual -= beta[spring, lag] * lagged[lag]
    u = residual / math.sqrt(variance[spring])
    cp = max(0., cp + u - k)
    cm = max(0., cm - u - k)
    for lag in range(bmax - 1, 0, -1):
        lagged[lag] = lagged[lag - 1]
    if bmax:
        lagged[0] = value
    spring = min(spring + 1, bmax) if max(cp, cm) > 0. else 0
    return cp, cm, spring


@njit(cache=True)
def _one_run(mu, muhat, log_mu, sum_mu, pi, pj, n, seed, kind, level,
             phi, sg, sv, means, sds, beta, variance, bmax, k, h,
             start_phase, guard):
    np.random.seed(seed)
    common = np.random.normal(0., sg)
    node = np.random.normal(0., sv, n)
    exponentials = np.empty(n)
    counts = np.empty(len(pi), dtype=np.int64)
    weights = np.ones(len(pi))
    lagged = np.zeros(max(1, bmax))
    cp, cm, spring = 0., 0., 0
    scale = math.sqrt(1. - phi * phi)
    center = .5 * (sg * sg + 2. * sv * sv)
    for t in range(guard):
        s = (t + start_phase) % mu.shape[0]
        common = _draw_network(mu, pi, pj, node, exponentials, common,
                               phi, sg * scale, sv * scale, center, t, s,
                               counts, weights, kind, level)
        raw = _raw_deviance(counts, weights, log_mu, sum_mu, muhat, s, kind)
        cp, cm, spring = _update(raw, s, means, sds, beta, variance,
                                 bmax, k, cp, cm, spring, lagged)
        if max(cp, cm) > h:
            return t + 1
    return -1


@njit(cache=True, parallel=True)
def _parallel_runs(mu, muhat, log_mu, sum_mu, pi, pj, n, seeds, kind, level,
                   phi, sg, sv, means, sds, beta, variance, bmax, k, h,
                   start_phase, guard):
    runs = np.empty(len(seeds), dtype=np.int64)
    for r in prange(len(seeds)):
        runs[r] = _one_run(mu, muhat, log_mu, sum_mu, pi, pj, n, seeds[r],
                            kind, level, phi, sg, sv, means, sds, beta,
                            variance, bmax, k, h, start_phase, guard)
    return runs


@njit(cache=True)
def _serial_runs(mu, muhat, log_mu, sum_mu, pi, pj, n, seeds, kind, level,
                 phi, sg, sv, means, sds, beta, variance, bmax, k, h,
                 start_phase, guard):
    runs = np.empty(len(seeds), dtype=np.int64)
    for r in range(len(seeds)):
        runs[r] = _one_run(mu, muhat, log_mu, sum_mu, pi, pj, n, seeds[r],
                            kind, level, phi, sg, sv, means, sds, beta,
                            variance, bmax, k, h, start_phase, guard)
    return runs


def _kind(kind, level):
    if kind not in (0, 1, 2, 3) or int(kind) != kind:
        raise ValueError("kind must be 0, 1, 2, or 3")
    if not np.isfinite(level) or (kind in (1, 3) and not 0 <= level < 1):
        raise ValueError("compound and missingness probabilities must lie in [0,1)")
    return int(kind), float(level)


def _chart(muhat, template_mean, template_sd, beta, variance, bmax, h):
    fitted = np.asarray(muhat, dtype=float)
    if fitted.ndim != 2:
        raise ValueError("muhat must have shape (edges, templates)")
    inputs = independent._chart_inputs(
        fitted[None, :, :], np.asarray(template_mean)[None, :],
        np.asarray(template_sd)[None, :], np.asarray(beta)[None, :, :],
        np.asarray(variance)[None, :], np.asarray([bmax]), np.asarray([h]),
    )
    log_mu, sum_mu, means, sds, beta, variance, orders, limits = inputs
    return (np.ascontiguousarray(fitted.T), log_mu[0], sum_mu[0], means[0],
            sds[0], beta[0], variance[0], int(orders[0]), float(limits[0]))


def robust_validation_runs(mu, muhat, pi, pj, n, seeds, kind, level, phi, sg, sv,
                           template_mean, template_sd, beta, variance, bmax, k, h,
                           start_phase=0, guard=10_000_000, parallel=True):
    """Return complete first-alarm lengths, shape ``(reps,)``.

    Both ``mu`` and ``muhat`` have shape (edges, templates). Prediction beta
    is padded to (bmax+1, bmax), variance to (bmax+1,). The numerical guard
    always raises if any path has not signaled; it is never a run length.
    """
    kind, level = _kind(kind, level)
    population, pi, pj = independent._population_inputs(mu, pi, pj, n, phi, sg, sv)
    fitted, log_mu, sum_mu, means, sds, beta, variance, bmax, h = _chart(
        muhat, template_mean, template_sd, beta, variance, bmax, h,
    )
    if population.shape != fitted.shape:
        raise ValueError("population and fitted means must have the same shape")
    seeds = np.asarray(seeds)
    if (seeds.ndim != 1 or not len(seeds) or not np.issubdtype(seeds.dtype, np.integer)
            or np.any(seeds < 0) or np.any(seeds > 2**32 - 1)
            or len(np.unique(seeds)) != len(seeds)):
        raise ValueError("seeds must be distinct uint32-compatible integers")
    if not np.isfinite(k) or k < 0 or guard < 1 or int(guard) != guard:
        raise ValueError("require nonnegative k and positive integer guard")
    if int(start_phase) != start_phase or not 0 <= start_phase < population.shape[0]:
        raise ValueError("start_phase must index a template")
    kernel = _parallel_runs if parallel else _serial_runs
    runs = kernel(population, fitted, log_mu, sum_mu, pi, pj, int(n),
                  seeds.astype(np.uint32), kind, level, phi, sg, sv,
                  means, sds, beta, variance, bmax, k, h,
                  int(start_phase), int(guard))
    bad = np.flatnonzero(runs < 1)
    if len(bad):
        raise RuntimeError(
            f"validation guard {guard} reached without alarm for "
            f"{len(bad)} path(s), first seed={seeds[bad[0]]}; no censored ARL returned"
        )
    return runs


@njit(cache=True)
def _sample(mu, pi, pj, n, length, seed, kind, level, phi, sg, sv, start_phase):
    np.random.seed(seed)
    common = np.random.normal(0., sg)
    node = np.random.normal(0., sv, n)
    exponentials = np.empty(n)
    counts = np.empty(len(pi), dtype=np.int64)
    weights = np.ones(len(pi))
    out = np.empty((length, len(pi)), dtype=np.int64)
    observed = np.ones((length, len(pi)), dtype=np.bool_)
    scale = math.sqrt(1. - phi * phi)
    center = .5 * (sg * sg + 2. * sv * sv)
    for t in range(length):
        s = (t + start_phase) % mu.shape[0]
        common = _draw_network(mu, pi, pj, node, exponentials, common,
                               phi, sg * scale, sv * scale, center, t, s,
                               counts, weights, kind, level)
        out[t] = counts
        if kind == 3:
            observed[t] = weights > 0.
    return out, observed


def sample_robust_path(mu, pi, pj, n, length, seed, kind=0, level=0.,
                       phi=.5, sg=.1, sv=.1, start_phase=0):
    """Return (unmasked_counts, observed), both shaped (time, edges).

    Missing observations remain represented by ``observed=False``. For
    fitting via run_simulation.fit_proposed_only, supply masked_counts.T,
    templates, and observed.T; do not treat missing edges as observed zeros.
    """
    kind, level = _kind(kind, level)
    population, pi, pj = independent._population_inputs(mu, pi, pj, n, phi, sg, sv)
    if int(length) != length or length < 1 or not 0 <= int(seed) <= 2**32 - 1:
        raise ValueError("require positive integer length and uint32 seed")
    if int(start_phase) != start_phase or not 0 <= start_phase < population.shape[0]:
        raise ValueError("start_phase must index a template")
    return _sample(population, pi, pj, int(n), int(length), np.uint32(seed), kind,
                    level, phi, sg, sv, int(start_phase))


@njit(cache=True)
def _replay(counts, observed, templates, muhat, log_mu, sum_mu, means, sds,
             beta, variance, bmax, k, h, kind, level):
    cp, cm, spring = 0., 0., 0
    lagged = np.zeros(max(1, bmax))
    weights = np.ones(counts.shape[1])
    for t in range(len(counts)):
        s = templates[t]
        if kind == 3:
            weights[:] = observed[t] / (1. - level)
        raw = _raw_deviance(counts[t], weights, log_mu, sum_mu, muhat, s, kind)
        cp, cm, spring = _update(raw, s, means, sds, beta, variance,
                                 bmax, k, cp, cm, spring, lagged)
        if max(cp, cm) > h:
            return t + 1
    return -1


def robust_run_from_counts(counts, observed, templates, muhat, template_mean,
                           template_sd, beta, variance, bmax, k, h,
                           kind=0, level=0.):
    """Replay one supplied path; -1 means no crossing in supplied observations."""
    kind, level = _kind(kind, level)
    chart = _chart(muhat, template_mean, template_sd, beta, variance, bmax, h)
    fitted, log_mu, sum_mu, means, sds, beta, variance, bmax, h = chart
    counts = np.asarray(counts)
    observed = np.asarray(observed, dtype=bool)
    templates = np.asarray(templates, dtype=np.int64)
    if counts.ndim != 2 or counts.shape != observed.shape or counts.shape[1] != fitted.shape[1]:
        raise ValueError("counts and observed must have matching (time, edges) shape")
    if np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("counts must be nonnegative integers")
    if len(templates) != len(counts) or np.any(templates < 0) or np.any(templates >= fitted.shape[0]):
        raise ValueError("invalid template indices")
    return int(_replay(np.ascontiguousarray(counts, dtype=np.int64),
                        np.ascontiguousarray(observed), np.ascontiguousarray(templates),
                        fitted, log_mu, sum_mu, means, sds, beta, variance,
                        bmax, k, h, kind, level))
