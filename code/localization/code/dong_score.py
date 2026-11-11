"""ZIP-SBM score baseline inspired by Dong, Chen and Wang (2020).

This is an explicitly adapted baseline, NOT an exact reproduction of their
multilayer directed MZIP-SBM or their published community-pair diagnostic
output.  We use one undirected count layer, estimate hard community labels
from in-control (IC) history, and fit a ZIP distribution to each unordered
community pair and periodic template.  The original score/Fisher quadratic
form is extended to nodes by grouping a node's incident edges by the other
endpoint's fixed community and summing the group-wise quadratic forms.

All likelihood fitting uses each undirected observation only once (i < j).
An edge can contribute to each endpoint's diagnostic score, as required for
node attribution, but is never treated as two independent directed edges.
The score coordinates are log(lambda) and the structural-zero probability q.
Blocks whose ZIP MLE is on q = 0 use the one-parameter Poisson submodel.

Reference: Dong, H., Chen, N., and Wang, K. (2020), Technometrics 62:184-195,
https://doi.org/10.1080/00401706.2019.1625812, score construction (14)-(16).
No OC observations, true labels, anomaly labels, or threshold tuning enter
fit_dong.  Thresholding is deliberately left to the experiment driver.

Public API
----------
fit_dong(history, templates, k, seed=0) -> dict
node_score(fit, windows, templates) -> ndarray, shape (B, n)

Run this file directly for deterministic mathematical and fitting self-tests.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import brentq
from scipy.special import gammaln


_RATE_FLOOR = 1e-8
_Q_BOUNDARY = 1e-8


def _log_zero_probability(q, rate):
    with np.errstate(divide="ignore"):
        return np.logaddexp(np.log(q), np.log1p(-q) - rate)


def _validate_networks(a: np.ndarray, ndim: int, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != ndim or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"{name} must have {ndim} dimensions and square matrices")
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError(f"{name} must contain finite nonnegative counts")
    if not np.allclose(a, np.rint(a), rtol=0, atol=1e-9):
        raise ValueError(f"{name} must contain integer counts")
    if not np.allclose(a, a.swapaxes(-1, -2), rtol=0, atol=1e-9):
        raise ValueError(f"{name} must be undirected/symmetric")
    if np.any(np.abs(np.diagonal(a, axis1=-2, axis2=-1)) > 1e-9):
        raise ValueError(f"{name} must have zero diagonal (no self-loops)")
    return a


def _zip_mle(nobs: float, nzero: float, total: float) -> tuple[float, float]:
    """Global ZIP MLE from sufficient statistics, with q=0 boundary allowed."""
    if nobs <= 0 or total <= 0:
        # An all-zero/empty block cannot identify q and lambda separately.
        # A vanishing-rate Poisson member represents its fitted distribution.
        return 0.0, _RATE_FLOOR
    mean = total / nobs
    z = nzero / nobs
    if z <= np.exp(-mean) + 1e-10:
        return 0.0, max(mean, _RATE_FLOOR)
    positive_mean = total / (nobs - nzero)

    def equation(rate: float) -> float:
        return rate / (-np.expm1(-rate)) - positive_mean

    rate = brentq(equation, 1e-10, max(2.0, positive_mean * 1.1), xtol=1e-12)
    q = float(np.clip(1.0 - mean / rate, 0.0, 1.0 - 1e-10))
    if q < _Q_BOUNDARY:
        return 0.0, max(mean, _RATE_FLOOR)
    return q, max(float(rate), _RATE_FLOOR)


def _fit_blocks(labels, ns, zero_sums, count_sums, k):
    n = labels.size
    ii, jj = np.triu_indices(n, 1)
    s_count = ns.size
    q = np.zeros((s_count, k, k))
    rate = np.full_like(q, _RATE_FLOOR)
    objective = 0.0
    for a in range(k):
        for b in range(a, k):
            mask = ((labels[ii] == a) & (labels[jj] == b))
            if a != b:
                mask |= ((labels[ii] == b) & (labels[jj] == a))
            left, right = ii[mask], jj[mask]
            for s in range(s_count):
                nobs = float(ns[s] * left.size)
                nzero = float(zero_sums[s, left, right].sum())
                total = float(count_sums[s, left, right].sum())
                qs, ls = _zip_mle(nobs, nzero, total)
                q[s, a, b] = q[s, b, a] = qs
                rate[s, a, b] = rate[s, b, a] = ls
                npos = nobs - nzero
                objective += (nzero * _log_zero_probability(qs, ls)
                              + npos * np.log1p(-qs)
                              + total * np.log(ls) - npos * ls)
    # log(y!) is a constant across label assignments and is omitted here.
    return q, rate, float(objective)


def _repair_labels(labels, x, k):
    labels = np.asarray(labels, dtype=int).copy()
    for missing in sorted(set(range(k)) - set(labels.tolist())):
        sizes = np.bincount(labels, minlength=k)
        donor = int(np.argmax(sizes))
        members = np.flatnonzero(labels == donor)
        center = x[members].mean(axis=0)
        moved = members[np.argmax(np.sum((x[members] - center) ** 2, axis=1))]
        labels[moved] = missing
    return labels


def _spectral_start(mean_network, k, rng, normalized):
    a = mean_network.copy()
    if normalized:
        degree = a.sum(axis=1)
        invroot = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
        a = invroot[:, None] * a * invroot[None, :]
    values, vectors = np.linalg.eigh(a)
    x = vectors[:, np.argsort(np.abs(values))[-k:]]
    if normalized:
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, labels = kmeans2(x, k, minit="++", iter=40, seed=rng)
    return _repair_labels(labels, x, k)


def fit_dong(history, templates, k, seed=0, *, n_starts=4, max_iter=20):
    """Fit the single-layer undirected hard ZIP-SBM using IC history only.

    Parameters
    ----------
    history : array, shape (T, n, n)
        Symmetric nonnegative integer matrices with zero diagonal.
    templates : array, shape (T,)
        Observed periodic-template identifiers; every future identifier must
        occur here.  No true community labels or generating parameters enter.
    k : int
        Community count, supplied equally to competing methods.
    seed : int
        Spectral/k-means initialization seed.
    n_starts, max_iter : int
        Fixed IC-only numerical fitting controls.  Both normalized and raw
        spectral starts are fitted and the largest IC objective is retained.

    Returns
    -------
    dict
        Keys include labels (n,), q/lambda (S,k,k), template_values, and
        IC-only optimization metadata.  The classification updates are an
        approximation to likelihood optimization, not the paper's variational
        EM.  This adaptation is recorded in the returned metadata.
    """
    history = _validate_networks(history, 3, "history")
    t_count, n, _ = history.shape
    if t_count < 1 or not isinstance(k, (int, np.integer)) or not (1 <= k <= n):
        raise ValueError("history must be nonempty and 1 <= k <= n")
    templates = np.asarray(templates)
    if templates.shape != (t_count,):
        raise ValueError("templates must have shape (T,)")
    if n_starts < 1 or max_iter < 1:
        raise ValueError("n_starts and max_iter must be positive")
    values, encoded = np.unique(templates, return_inverse=True)
    ns = np.bincount(encoded, minlength=values.size)
    zero_sums = np.stack([(history[encoded == s] == 0).sum(axis=0)
                          for s in range(values.size)])
    count_sums = np.stack([history[encoded == s].sum(axis=0)
                           for s in range(values.size)])
    mean_network = history.mean(axis=0)
    rng = np.random.default_rng(seed)
    best = None
    histories = []
    for start in range(n_starts if k > 1 else 1):
        labels = (_spectral_start(mean_network, k, rng, start % 2 == 0)
                  if k > 1 else np.zeros(n, dtype=int))
        q, rate, objective = _fit_blocks(labels, ns, zero_sums, count_sums, k)
        objectives = [objective]
        for _ in range(max_iter):
            changed = 0
            sizes = np.bincount(labels, minlength=k)
            for i in rng.permutation(n):
                old = labels[i]
                if sizes[old] <= 1:
                    continue
                others = np.arange(n) != i
                other_labels = labels[others]
                node_objectives = np.zeros(k)
                for candidate in range(k):
                    qi = q[:, candidate, other_labels]
                    li = rate[:, candidate, other_labels]
                    nz = zero_sums[:, i, others]
                    nt = ns[:, None]
                    total = count_sums[:, i, others]
                    node_objectives[candidate] = np.sum(
                        nz * _log_zero_probability(qi, li)
                        + (nt - nz) * np.log1p(-qi)
                        + total * np.log(li) - (nt - nz) * li)
                new = int(np.argmax(node_objectives))
                if node_objectives[new] > node_objectives[old] + 1e-9:
                    labels[i] = new
                    sizes[old] -= 1
                    sizes[new] += 1
                    changed += 1
            q, rate, objective = _fit_blocks(labels, ns, zero_sums, count_sums, k)
            objectives.append(objective)
            if changed == 0:
                break
        histories.append(objectives)
        if best is None or objective > best[0]:
            best = objective, labels.copy(), q.copy(), rate.copy(), start
    objective, labels, q, rate, start = best
    return {
        "labels": labels,
        "q": q,
        "lambda": rate,
        "template_values": values,
        "n": n,
        "k": int(k),
        "seed": int(seed),
        "objective_without_log_factorials": objective,
        "selected_start": start,
        "optimization_histories": histories,
        "adaptation": ("single-layer undirected ZIP-SBM; IC-only hard-label "
                       "likelihood fitting; node/partner-community score extension"),
        "source_doi": "10.1080/00401706.2019.1625812",
    }


def _zip_score_info(y, q, rate):
    """ZIP score and exact expected Fisher info in (log(rate), q).

    At a q=0 fitted boundary the q coordinate is held fixed (Poisson
    submodel), so its score and information entries are zero.  Information
    shapes follow broadcast(q, rate), independent of the observed y.
    """
    q, rate = np.broadcast_arrays(np.asarray(q, float), np.asarray(rate, float))
    q = np.clip(q, 0.0, 1.0 - 1e-10)
    rate = np.maximum(rate, _RATE_FLOOR)
    poisson_zero = np.exp(-rate)
    zero_probability = q + (1.0 - q) * poisson_zero
    zero_probability = np.maximum(zero_probability, 1e-300)
    active_q = q >= _Q_BOUNDARY
    y = np.asarray(y, dtype=float)
    score_eta = np.where(y == 0,
                         -(1.0 - q) * rate * poisson_zero / zero_probability,
                         y - rate)
    # Avoid cancellation/underflow in the exact Poisson boundary case.
    score_eta = np.where(active_q, score_eta, y - rate)
    score_q = np.where(y == 0,
                       (1.0 - poisson_zero) / zero_probability,
                       -1.0 / (1.0 - q))
    score_q = np.where(active_q, score_q, 0.0)
    score = np.stack(np.broadcast_arrays(score_eta, score_q), axis=-1)
    information = np.zeros(q.shape + (2, 2))
    information[..., 0, 0] = (
        (1.0 - q) * rate
        - q * (1.0 - q) * rate ** 2 * poisson_zero / zero_probability)
    information[..., 1, 1] = np.where(
        active_q, (1.0 - poisson_zero) / ((1.0 - q) * zero_probability), 0.0)
    information[..., 0, 1] = information[..., 1, 0] = np.where(
        active_q, -rate * poisson_zero / zero_probability, 0.0)
    return score, information


def node_score(fit, windows, templates):
    """Return IC-referenced node scores for windows of shape (B,W,n,n).

    templates can have shape (W,) (same sequence in every replicate) or
    (B,W).  For node i and partner community g, accumulate the ZIP scores
    over time and incident edges, add their expected Fisher matrices, and
    evaluate U' I^+ U.  Summing over g yields the node score.  A numerical
    Moore-Penrose inverse handles a fixed q=0 coordinate and rare nearly
    singular ZIP information; no anomaly labels or OC fitting are used.
    """
    windows = _validate_networks(windows, 4, "windows")
    b_count, width, n, _ = windows.shape
    if n != fit["n"]:
        raise ValueError("node count differs from fitted history")
    templates = np.asarray(templates)
    if templates.shape == (width,):
        templates = np.broadcast_to(templates, (b_count, width))
    if templates.shape != (b_count, width):
        raise ValueError("templates must have shape (W,) or (B,W)")
    encoded = np.full(templates.shape, -1, dtype=int)
    for s, value in enumerate(fit["template_values"]):
        encoded[templates == value] = s
    if np.any(encoded < 0):
        raise ValueError("all scoring templates must occur in the IC history")
    labels = fit["labels"]
    answer = np.zeros((b_count, n))
    for i in range(n):
        for group in range(fit["k"]):
            partners = np.flatnonzero((labels == group) & (np.arange(n) != i))
            if partners.size == 0:
                continue
            # A selected incident edge is present once here, not as both ij/ji.
            y = windows[:, :, i, :][:, :, partners]
            q = fit["q"][encoded, labels[i], group][..., None]
            rate = fit["lambda"][encoded, labels[i], group][..., None]
            score, information = _zip_score_info(y, q, rate)
            u = score.sum(axis=(1, 2))
            fisher = information.sum(axis=(1, 2)) * partners.size
            # Rescale to unit Fisher diagonal before taking an inverse.  This
            # avoids discarding a valid rate coordinate merely because q and
            # log(lambda) have very different numerical information scales.
            diagonal = np.diagonal(fisher, axis1=-2, axis2=-1)
            scale = np.sqrt(np.where(diagonal > 0, diagonal, 1.0))
            scaled_fisher = fisher / (scale[..., :, None] * scale[..., None, :])
            inv = np.linalg.pinv(scaled_fisher, rcond=1e-10, hermitian=True)
            scaled_u = u / scale
            answer[:, i] += np.einsum("bi,bij,bj->b", scaled_u, inv, scaled_u)
    return np.maximum(answer, 0.0)


def _self_test():
    # Analytic scores agree with finite differences of the actual ZIP pmf.
    q, rate = 0.23, 1.8
    y = np.array([0.0, 1.0, 7.0])

    def logpmf(eta, zero_q):
        lam = np.exp(eta)
        return np.where(y == 0,
                        np.log(zero_q + (1.0 - zero_q) * np.exp(-lam)),
                        np.log1p(-zero_q) + y * eta - lam - gammaln(y + 1))

    eps = 1e-6
    eta = np.log(rate)
    score, information = _zip_score_info(y, q, rate)
    finite = np.column_stack([
        (logpmf(eta + eps, q) - logpmf(eta - eps, q)) / (2 * eps),
        (logpmf(eta, q + eps) - logpmf(eta, q - eps)) / (2 * eps)])
    np.testing.assert_allclose(score, finite, rtol=1e-7, atol=1e-8)

    # Sum over the pmf verifies E(score)=0 and E(score score')=Fisher.
    yy = np.arange(100.0)
    ss, ii = _zip_score_info(yy, q, rate)
    probability = ((1.0 - q)
                   * np.exp(yy * eta - rate - gammaln(yy + 1)))
    probability[0] += q
    np.testing.assert_allclose(probability @ ss, 0, atol=1e-12)
    numeric_information = np.einsum("a,ai,aj->ij", probability, ss, ss)
    np.testing.assert_allclose(ii, numeric_information, rtol=1e-10, atol=1e-12)
    extreme_score, extreme_info = _zip_score_info(np.array([0., 1001.]), 0., 1000.)
    np.testing.assert_array_equal(extreme_score[:, 0], [-1000., 1.])
    np.testing.assert_equal(extreme_info[0, 0], 1000.)

    # For q=0, even differing template means give the exact Poisson score.
    rng = np.random.default_rng(984)
    labels = np.array([0, 0, 1, 1])
    rates = np.array([[[1., 2.], [2., 3.]], [[2., 4.], [4., 6.]]])
    manual_fit = {"labels": labels, "q": np.zeros_like(rates), "lambda": rates,
                  "template_values": np.array([0, 1]), "n": 4, "k": 2}
    template = np.array([0, 1, 1, 0])
    networks = np.zeros((7, 4, 4, 4))
    for a in range(4):
        for b in range(a + 1, 4):
            counts = rng.poisson(rates[template, labels[a], labels[b]], size=(7, 4))
            networks[:, :, a, b] = networks[:, :, b, a] = counts
    obtained = node_score(manual_fit, networks, template)
    expected = np.zeros_like(obtained)
    for i in range(4):
        for g in range(2):
            js = np.flatnonzero((labels == g) & (np.arange(4) != i))
            if js.size:
                mu = rates[template, labels[i], g]
                residual = networks[:, :, i, :][:, :, js].sum(axis=(1, 2))
                residual -= mu.sum() * js.size
                expected[:, i] += residual ** 2 / (mu.sum() * js.size)
    np.testing.assert_allclose(obtained, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        obtained, node_score(manual_fit, networks,
                             np.broadcast_to(template, (7, 4))), rtol=0, atol=0)

    # IC-only estimation smoke test on a small identifiable ZIP-SBM.
    labels_true = np.repeat([0, 1], 6)
    t = 80
    history = np.zeros((t, 12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            lam = 3.0 if labels_true[i] == labels_true[j] else 0.25
            counts = rng.poisson(lam, t) * (rng.random(t) > 0.12)
            history[:, i, j] = history[:, j, i] = counts
    fitted = fit_dong(history, np.arange(t) % 2, 2, seed=123)
    assert (np.array_equal(fitted["labels"], labels_true)
            or np.array_equal(1 - fitted["labels"], labels_true))
    assert all(np.min(np.diff(hist)) >= -1e-6
               for hist in fitted["optimization_histories"] if len(hist) > 1)
    assert np.all(np.isfinite(node_score(fitted, history[:8][None], np.arange(8) % 2)))
    print("PASS: finite-difference scores; exact Fisher; Poisson reduction; "
          "template broadcasting; IC fitting and monotone likelihood")


if __name__ == "__main__":
    _self_test()
