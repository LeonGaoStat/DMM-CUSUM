"""ACD historical-reference adaptation for undirected count networks.

Source model: Safdari & De Bacco (2022), Journal of Big Data 9:122,
https://doi.org/10.1186/s40537-022-00669-1, equations (1)-(3).
Author implementation: https://github.com/hds-safdari/Anomaly_Community_Detection
(code/AnomalyDetection.py; undirected=True, assortative=False).

This is NOT the original static ACD run independently on every OC graph.
For a computationally tractable historical-reference comparator, each IC
 template is fitted with M=U W U.T (unconstrained positive U and symmetric
full W); the anomaly mixture prior is fixed before OC evaluation at 0.01.
The alternative Poisson rate is the IC average off-diagonal rate in that
 template, a fixed empirical-Bayes choice when clean IC data contain no
 identifiable anomaly population. These choices are reported in fit metadata.
No true memberships, means, source nodes, or OC observations enter fit.

Conditional on a persistent edge anomaly label over a window, independent
Poisson observations add: Y_ij=sum_t A_ij,t has normal mean sum_t M_ij,s(t)
and anomalous mean sum_t pi_s(t). Bayes' rule gives the exact mixture
posterior for this window-aggregate adaptation. Incident posterior sums
convert its original edge output to a node score, an additional explicit
adaptation; threshold calibration is left to independent IC samples.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.cluster.vq import kmeans2

_EPS = 1e-10


def _validate_networks(a, ndim):
    a = np.asarray(a, dtype=float)
    if a.ndim != ndim or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"Expected {ndim}-dimensional square network array")
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("Counts must be finite and nonnegative")
    if not np.allclose(a, a.swapaxes(-1, -2), atol=1e-9):
        raise ValueError("Undirected ACD requires symmetric networks")
    return a


def _one_template(total, exposure, k, rng, prior, starts=3, maxiter=350):
    n = total.shape[0]
    ii, jj = np.triu_indices(n, 1)
    wi, wj = np.triu_indices(k)
    a = total[ii, jj]
    mean = total / exposure
    alternative_rate = max(float(a.mean() / exposure), _EPS)
    logodds = np.log(prior) - np.log1p(-prior)
    # Alternative rate is fixed from IC only; factorial terms cancel in Q.
    alternative_total = exposure * alternative_rate
    # Spectral initialization uses only the observed template average.
    degree = np.maximum(mean.sum(axis=1), _EPS)
    normalized = mean / np.sqrt(degree[:, None] * degree[None, :])
    eigval, eigvec = np.linalg.eigh(normalized)
    embedding = eigvec[:, np.argsort(eigval)[-k:]]
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), _EPS)
    try:
        _, labels = kmeans2(embedding, k, minit='++', seed=rng)
    except (ValueError, np.linalg.LinAlgError):
        labels = rng.integers(0, k, size=n)

    def unpack(x):
        u = np.exp(x[:n*k].reshape(n, k))
        vals = np.exp(x[n*k:])
        w = np.empty((k, k))
        w[wi, wj] = vals
        w[wj, wi] = vals
        return u, w

    def objective(x, mixture):
        u, w = unpack(x)
        m = np.maximum((u @ w @ u.T)[ii, jj], _EPS)
        expected = exposure * m
        lognormal = a * np.log(expected) - expected
        if mixture:
            logalt = a * np.log(alternative_total) - alternative_total
            q = expit(logodds + logalt - lognormal)
            logjoint = np.logaddexp(np.log1p(-prior) + lognormal,
                                   np.log(prior) + logalt)
            value = -logjoint.sum() / exposure
        else:
            q = np.zeros_like(m)
            value = -lognormal.sum() / exposure
        d = -(1.0-q) * (a / m - exposure) / exposure
        r = np.zeros((n, n))
        r[ii, jj] = d
        gu = ((r+r.T) @ u @ w) * u
        gw = u.T @ r @ u
        gvals = gw[wi, wj] + np.where(wi == wj, 0.0, gw[wj, wi])
        grad = np.concatenate((gu.ravel(), gvals*w[wi,wj]))
        return float(value), grad

    best = None
    bounds = [(-12., 12.)] * (n*k + len(wi))
    for start in range(starts):
        u = np.full((n, k), 0.10)
        u[np.arange(n), labels] = 0.90
        u *= np.sqrt(degree / degree.mean())[:, None]
        if start:
            u *= np.exp(rng.normal(0, .25, size=u.shape))
        # Broad initial full affinity avoids structurally forcing assortativity.
        w = np.full((k, k), alternative_rate * .30)
        np.fill_diagonal(w, alternative_rate * k)
        x0 = np.r_[np.log(np.maximum(u, _EPS)).ravel(), np.log(w[wi,wj])]
        init = minimize(lambda x: objective(x, False), x0, jac=True,
                        method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-5})
        out = minimize(lambda x: objective(x, True), init.x, jac=True,
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-5})
        if np.isfinite(out.fun) and np.all(np.isfinite(out.x)):
            if best is None or out.fun < best.fun:
                best = out
    if best is None:
        raise RuntimeError('ACD optimization failed for every initialization')
    u, w = unpack(best.x)
    # Identifiable presentation only; leaves fitted M unchanged.
    scale = np.maximum(u.mean(axis=0), _EPS)
    u = u / scale
    w = w * scale[:, None] * scale[None, :]
    mu = np.maximum(u @ w @ u.T, _EPS)
    np.fill_diagonal(mu, 0.)
    rmse = float(np.sqrt(np.mean((mu[ii,jj]-mean[ii,jj])**2)))
    return {'mu': mu, 'u': u, 'w': w,
            'alternative_rate': alternative_rate,
            'converged': bool(best.success), 'iterations': int(best.nit),
            'optimizer_message': str(best.message),
            'objective': float(best.fun), 'IC_mean_RMSE': rmse}


def fit_acd(history, templates, k, seed=0, *, anomaly_prior=0.01,
            starts=3, maxiter=350):
    """Fit IC template-specific ACD means; no OC/oracle parameters accepted."""
    history = _validate_networks(history, 3)
    templates = np.asarray(templates)
    if templates.shape != (len(history),):
        raise ValueError('templates must have shape (T,)')
    if not 0 < anomaly_prior < 1:
        raise ValueError('anomaly_prior must lie strictly between 0 and 1')
    if not 1 <= k < history.shape[-1]:
        raise ValueError('k must be between 1 and n-1')
    labels = np.unique(templates)
    rng = np.random.default_rng(seed)
    fitted = []
    for label in labels:
        selected = history[templates == label]
        fitted.append(_one_template(selected.sum(axis=0), len(selected), k,
                                    rng, anomaly_prior, starts, maxiter))
    return {'method': 'ACD historical-reference adaptation',
            'mu': np.stack([x['mu'] for x in fitted]),
            'alternative_rate': np.array([x['alternative_rate'] for x in fitted]),
            'anomaly_prior': float(anomaly_prior), 'template_labels': labels,
            'k': int(k), 'template_fits': fitted, 'seed': int(seed),
            'metadata': {
                'source': 'Safdari and De Bacco (2022), doi:10.1186/s40537-022-00669-1',
                'undirected': True, 'full_affinity': True,
                'prior_rule': 'fixed before OC evaluation; default 0.01',
                'alternative_rate_rule': 'IC mean off-diagonal count per template',
                'adaptation': 'IC-fixed reference; window-aggregate Poisson posterior; incident posterior sum',
                'starts': int(starts), 'maxiter': int(maxiter)}}


def edge_posterior(fit, windows, templates):
    """Return symmetric posterior arrays (B,n,n), zero diagonal."""
    windows = _validate_networks(windows, 4)
    b, length, n, _ = windows.shape
    labels = np.asarray(fit['template_labels'])
    templates = np.asarray(templates)
    if templates.shape == (length,):
        templates = np.broadcast_to(templates, (b, length))
    if templates.shape != (b, length):
        raise ValueError('templates must have shape (W,) or (B,W)')
    if not np.all(np.isin(templates, labels)):
        raise ValueError('A window contains templates absent from IC data')
    # Accumulate template exposures without allocating (B,W,n,n) means.
    exposure = np.stack([(templates == label).sum(axis=1)
                         for label in labels], axis=1)
    normal = np.einsum('bs,sij->bij', exposure, np.asarray(fit['mu']))
    alternative = exposure @ np.asarray(fit['alternative_rate'])
    counts = windows.sum(axis=1)
    normal = np.maximum(normal, _EPS)
    alternative = np.maximum(alternative[:, None, None], _EPS)
    prior = fit['anomaly_prior']
    logodds = np.log(prior)-np.log1p(-prior)
    # Log-factorials cancel exactly between the two Poisson likelihoods.
    ratio = counts * (np.log(alternative)-np.log(normal)) - alternative + normal
    posterior = expit(logodds+ratio)
    posterior[:, np.arange(n), np.arange(n)] = 0.
    if not np.all(np.isfinite(posterior)):
        raise RuntimeError('Nonfinite ACD posterior; refusing silent fallback')
    return posterior


def node_score(fit, windows, templates):
    """Incident sums of ACD edge posterior probabilities; returns (B,n)."""
    return edge_posterior(fit, windows, templates).sum(axis=-1)


def _selftest():
    n = 8
    base = np.full((n,n), 2.)
    np.fill_diagonal(base, 0.)
    fit = {'mu': base[None], 'alternative_rate': np.array([12.]),
           'anomaly_prior': .01, 'template_labels': np.array([0])}
    windows = np.broadcast_to(base, (2, 4, n, n)).copy()
    windows[1,:,0,1] = windows[1,:,1,0] = 12.
    posterior = edge_posterior(fit, windows, np.zeros(4, dtype=int))
    assert posterior[1,0,1] > posterior[0,0,1]
    assert np.allclose(posterior, posterior.swapaxes(-1,-2))
    assert np.all(np.isfinite(posterior))
    assert node_score(fit, windows, np.zeros(4, dtype=int)).shape == (2,n)
    rng = np.random.default_rng(7)
    u = np.r_[np.tile([1.,.1],(4,1)),np.tile([.1,1.],(4,1))]
    mean = u @ np.array([[8.,1.],[1.,8.]]) @ u.T
    a = rng.poisson(mean, size=(100,n,n))
    a = np.triu(a,1); a = a + a.swapaxes(-1,-2)
    estimated = fit_acd(a, np.zeros(100,dtype=int), 2, seed=3, starts=2, maxiter=150)
    assert np.allclose(estimated['mu'], estimated['mu'].swapaxes(-1,-2))
    assert np.all(estimated['mu'][0][np.triu_indices(n,1)] > 0)
    print({'selftest': 'PASS', 'IC_mean_RMSE': estimated['template_fits'][0]['IC_mean_RMSE'],
           'prior': estimated['anomaly_prior']})


if __name__ == '__main__':
    _selftest()
