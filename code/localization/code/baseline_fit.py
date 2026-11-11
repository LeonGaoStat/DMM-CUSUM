"""Historical-only fitting for the exploratory DMM-CUSUM experiment.

No generating memberships or parameters enter this module.  Memberships are
estimated on template-average networks by Mixed-SCORE, aligned, and averaged
with weights proportional to template sample sizes.  Conditional Poisson MLE
then uses each undirected edge once and fixes every diagonal of B to one.

The optional multistart low-rank diagonal completion is a declared estimation
adaptation.  Candidates are selected by historical off-diagonal reconstruction
error, subject to valid Mixed-SCORE simplex recovery.  Signed signal modes are
allowed: no positive-semidefinite B assumption is imposed.  Invalid vertex
scales are rejected rather than replaced by an epsilon.  Returned fitted
networks have diagonal zero; the Poisson objective excludes diagonal entries.
"""

from __future__ import annotations

import json
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize


EPS = 1e-12


def _complete_diagonal(a: np.ndarray, k: int) -> tuple[np.ndarray, dict]:
    """Fit a signed rank-k signal diagonal from observed entries only.

    Zero initialization can trap absolute-eigenvalue truncation at a negative
    mode caused solely by missing diagonal entries.  Degree-based multistarts
    avoid making that one fixed point the estimator.  All candidates retain
    positive and negative signal eigenvalues and are judged without truth.
    """
    degree = a.sum(1)
    diagonal_scale = degree ** 2 / max(degree.sum(), EPS)
    ij = np.triu_indices(len(a), 1)
    observed_norm = max(np.linalg.norm(a[ij]), EPS)
    candidates = []
    best = None
    for scale in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        out = np.array(a, dtype=float, copy=True)
        old = scale * diagonal_scale
        np.fill_diagonal(out, old)
        relative_change = np.inf
        for iteration in range(250):
            values, vectors = np.linalg.eigh(out)
            selected = np.argsort(-np.abs(values))[:k]
            diagonal = np.maximum(
                (vectors[:, selected] ** 2) @ values[selected], 0.0
            )
            diagonal = 0.5 * old + 0.5 * diagonal
            relative_change = np.linalg.norm(diagonal - old) / max(
                np.linalg.norm(diagonal), 1.0
            )
            np.fill_diagonal(out, diagonal)
            old = diagonal
            if relative_change < 1e-8:
                break
        values, vectors = np.linalg.eigh(out)
        selected = np.argsort(-np.abs(values))[:k]
        low_rank = (vectors[:, selected] * values[selected]) @ vectors[:, selected].T
        error = float(np.linalg.norm((low_rank - a)[ij]) / observed_norm)
        record = {
            "initial_degree_scale": scale,
            "iterations": iteration + 1,
            "relative_change": float(relative_change),
            "converged": bool(relative_change < 1e-8),
            "offdiagonal_relative_error": error,
            "eigenvalues": values[selected].tolist(),
            "max_completed_diagonal": float(old.max()),
        }
        try:
            _, membership_diagnostics = _recover_membership(out, k)
            record["valid_simplex"] = True
            record["minimum_vertex_denominator"] = membership_diagnostics[
                "minimum_vertex_denominator"]
            if record["converged"] and (best is None or error < best[0]):
                best = error, out.copy(), len(candidates)
        except ValueError as error_detail:
            record["valid_simplex"] = False
            record["rejection_reason"] = str(error_detail)
        candidates.append(record)
    if best is None:
        raise ValueError(
            "No converged diagonal completion has valid Mixed-SCORE vertex "
            "scales; cannot estimate memberships without an explicit fallback. "
            + json.dumps(candidates)
        )
    return best[1], {
        "method": "signed rank-k multistart diagonal completion",
        "selection": "minimum historical off-diagonal reconstruction error among valid converged candidates",
        "selected_candidate": best[2],
        "offdiagonal_relative_error": best[0],
        "candidates": candidates,
        "positive_semidefinite_constraint": False,
    }


def _vertices(ratio: np.ndarray, k: int) -> np.ndarray:
    """Deterministic successive projections followed by simplex exchanges."""
    n = len(ratio)
    if k == 2:
        return np.array([np.argmin(ratio[:, 0]), np.argmax(ratio[:, 0])])
    points = np.column_stack((np.ones(n), ratio))
    chosen = [int(np.argmax(np.linalg.norm(ratio - ratio.mean(0), axis=1)))]
    while len(chosen) < k:
        q, _ = np.linalg.qr(points[chosen].T, mode="reduced")
        residual = points - (points @ q) @ q.T
        distances = np.linalg.norm(residual, axis=1)
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances)))
    # Each exchange maximizes the simplex volume conditional on other vertices.
    for _ in range(20):
        changed = False
        for c in range(k):
            others = [chosen[j] for j in range(k) if j != c]
            q, _ = np.linalg.qr(points[others].T, mode="reduced")
            residual = points - (points @ q) @ q.T
            distances = np.linalg.norm(residual, axis=1)
            distances[others] = -1
            proposed = int(np.argmax(distances))
            if distances[proposed] > distances[chosen[c]] * (1 + 1e-9):
                chosen[c] = proposed
                changed = True
        if not changed:
            break
    return np.array(chosen)


def _recover_membership(spectral: np.ndarray, k: int) -> tuple[np.ndarray, dict]:
    """Mixed-SCORE geometry; degenerate vertex scales are genuine failures."""
    n = len(spectral)
    values, vectors = np.linalg.eigh(spectral)
    selected = np.argsort(-np.abs(values))[:k]
    # The Perron eigenpair is the reference even with negative signal modes.
    leading = int(np.argmax(values))
    selected = np.array([leading] + [j for j in selected if j != leading])[:k]
    values, vectors = values[selected], vectors[:, selected]
    if vectors[:, 0].sum() < 0:
        vectors[:, 0] *= -1
    if np.min(vectors[:, 0]) <= EPS:
        raise ValueError("Perron reference entries are not all strictly positive")
    u1 = vectors[:, 0]
    ratio = vectors[:, 1:] / u1[:, None]
    clipping = max(float(np.log(max(n, 3))), 1.0)
    ratio = np.clip(ratio, -clipping, clipping)
    anchors = _vertices(ratio, k)
    vertex_rows = ratio[anchors]
    augmented = np.vstack((np.ones(k), vertex_rows.T))
    condition = float(np.linalg.cond(augmented))
    if not np.isfinite(condition) or condition > 1e8:
        raise ValueError("The estimated ratio simplex is singular or ill-conditioned")
    # Columns are vertices: augmented @ w_i = (1, ratio_i)^T.
    weights = np.linalg.lstsq(
        augmented, np.column_stack((np.ones(n), ratio)).T, rcond=None
    )[0].T
    weights = np.maximum(weights, 0.0)
    denominator = values[0] + (vertex_rows ** 2) @ values[1:]
    if np.min(denominator) <= 1e-10 * max(abs(values[0]), 1.0):
        raise ValueError("Nonpositive or numerically degenerate Mixed-SCORE vertex scale")
    # b_1(k) = [lambda_1 + v_k' Lambda_rest v_k]^{-1/2};
    # pi_i is the normalized elementwise product w_i / b_1.
    z = weights * np.sqrt(denominator)[None, :]
    empty = z.sum(1) <= EPS
    z[empty] = 1.0
    z /= z.sum(1, keepdims=True)
    return z, {
        "anchors": anchors.tolist(),
        "eigenvalues": values.tolist(),
        "simplex_condition": condition,
        "nonpositive_vertex_denominators": 0,
        "minimum_vertex_denominator": float(np.min(denominator)),
        "eigenvalues_can_be_negative": True,
    }


def _mixed_score(a: np.ndarray, k: int, complete: bool) -> tuple[np.ndarray, dict]:
    if k == 1:
        return np.ones((len(a), 1)), {"anchors": [0], "diagonal_completion": None}
    spectral, completion = _complete_diagonal(a, k) if complete else (a, None)
    z, diagnostics = _recover_membership(spectral, k)
    diagnostics["diagonal_completion"] = completion
    return z, diagnostics


def _unpack(x: np.ndarray, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.exp(x[:n])
    b = np.eye(k)
    upper = np.triu_indices(k, 1)
    b[upper] = x[n:]
    b[(upper[1], upper[0])] = x[n:]
    return theta, b


def _objective(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean negative working log-likelihood and exact analytic gradient.

    Dividing by the number of edges and omitting N_s multiply the objective
    by positive constants, leaving the conditional maximizer unchanged.
    """
    n, k = z.shape
    i, j = np.triu_indices(n, 1)
    theta, b = _unpack(x, n, k)
    affinity = np.einsum("ij,ij->i", z[i] @ b, z[j])
    theta_product = theta[i] * theta[j]
    mu = np.maximum(theta_product * affinity, EPS)
    difference = mu - y
    value = np.mean(mu - y * np.log(mu))
    gradient = np.zeros_like(x)
    np.add.at(gradient, i, difference)
    np.add.at(gradient, j, difference)
    for c, (a, d) in enumerate(zip(*np.triu_indices(k, 1))):
        direction = z[i, a] * z[j, d] + z[i, d] * z[j, a]
        gradient[n + c] = np.sum((1 - y / mu) * theta_product * direction)
    gradient /= len(y)
    return float(value), gradient


def _fit_template(mean: np.ndarray, z: np.ndarray, maxiter: int) -> tuple:
    n, k = z.shape
    ij = np.triu_indices(n, 1)
    y = mean[ij]
    degree = mean.sum(1)
    if not np.any(y > 0):
        # A strictly positive theta has no attained MLE for an all-zero network.
        # Report this rather than incorrectly claiming numerical convergence.
        raise ValueError("An all-zero template has no interior positive-activity MLE.")
    starts = []
    for off_diagonal in ((0.2, 0.8) if k > 1 else (0.0,)):
        initial_b = np.full((k, k), off_diagonal)
        np.fill_diagonal(initial_b, 1.0)
        typical_affinity = np.mean((z @ initial_b @ z.T)[ij])
        scale = np.sqrt(max(y.mean(), EPS) / max(typical_affinity, EPS))
        theta = scale * np.maximum(degree, EPS) / max(degree.mean(), EPS)
        x0 = np.r_[np.log(np.maximum(theta, 1e-8)),
                   np.full(k * (k - 1) // 2, off_diagonal)]
        result = minimize(
            _objective, x0, args=(y, z), jac=True, method="L-BFGS-B",
            bounds=[(-20.0, 20.0)] * n + [(1e-10, None)] * (len(x0) - n),
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7,
                     "maxls": 40},
        )
        starts.append(result)
    result = min(starts, key=lambda item: item.fun)
    theta, b = _unpack(result.x, n, k)
    mu = np.outer(theta, theta) * (z @ b @ z.T)
    np.fill_diagonal(mu, 0.0)
    gradient = result.jac.copy()
    # Lower-bound coordinates can have positive gradients at a constrained optimum.
    for c in range(n, len(gradient)):
        if result.x[c] <= 1.01e-10:
            gradient[c] = min(gradient[c], 0.0)
    diagnostics = {
        "converged": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "negative_loglik_per_edge": float(result.fun),
        "projected_gradient_max": float(np.max(np.abs(gradient))),
        "starts": [dict(converged=bool(r.success), objective=float(r.fun),
                        iterations=int(r.nit)) for r in starts],
        "off_diagonal_lower_bound": 1e-10,
    }
    return theta, b, mu, diagnostics


def fit_baseline(history: np.ndarray, templates: np.ndarray, k: int, *,
                 maxiter: int = 1200, diagonal_completion: bool = True) -> dict:
    """Return z[n,k], mu[S,n,n], theta[S,n], b[S,k,k] and diagnostics.

    ``templates`` must cover consecutive labels 0,...,S-1, and each template
    must contain historical observations.  Only historical networks are used.
    Fitted B diagonals equal one exactly.  Optimization is nonconvex: the
    convergence reports describe the numerical conditional fits, not a proof
    of finding the global maximum.
    """
    history = np.asarray(history, dtype=float)
    templates = np.asarray(templates)
    if history.ndim != 3 or history.shape[1] != history.shape[2]:
        raise ValueError("history must have shape (T,n,n)")
    t, n, _ = history.shape
    if templates.shape != (t,) or not np.all(templates == templates.astype(int)):
        raise ValueError("templates must contain one integer label per network")
    templates = templates.astype(int)
    labels = np.unique(templates)
    if not np.array_equal(labels, np.arange(len(labels))):
        raise ValueError("templates must cover consecutive labels 0,...,S-1")
    if not 1 <= k < n:
        raise ValueError("require 1 <= k < n")
    if not np.all(np.isfinite(history)) or np.any(history < 0):
        raise ValueError("historical counts must be finite and nonnegative")
    if not np.allclose(history, history.transpose(0, 2, 1)):
        raise ValueError("historical networks must be undirected")
    if not np.allclose(np.diagonal(history, axis1=1, axis2=2), 0):
        raise ValueError("historical networks must have no self-loops")
    if not np.allclose(history, np.round(history)):
        raise ValueError("historical edge weights must be integer counts")
    counts = np.bincount(templates)
    means = np.stack([history[templates == s].mean(0) for s in labels])
    memberships, spectral_diagnostics = zip(*[
        _mixed_score(mean, k, diagonal_completion) for mean in means
    ])
    # The best-observed template defines labels; no truth-based alignment.
    reference_index = int(np.argmax(counts))
    reference = memberships[reference_index]
    aligned, permutations = [], []
    for z in memberships:
        cost = np.sum((reference[:, :, None] - z[:, None, :]) ** 2, axis=0)
        rows, columns = linear_sum_assignment(cost)
        permutation = columns[np.argsort(rows)]
        aligned.append(z[:, permutation])
        permutations.append(permutation.tolist())
    z = np.average(np.stack(aligned), axis=0, weights=counts)
    z /= z.sum(1, keepdims=True)
    fitted = [_fit_template(mean, z, maxiter) for mean in means]
    theta, b, mu, optimization = zip(*fitted)
    return {
        "z": z, "mu": np.stack(mu), "theta": np.stack(theta), "b": np.stack(b),
        "converged": bool(all(item["converged"] for item in optimization)),
        "optimization": list(optimization),
        "diagnostics": {
            "membership_method": "template Mixed-SCORE with deterministic vertex hunting",
            "diagonal_completion": bool(diagonal_completion),
            "diagonal_completion_use": "spectral estimation only; never Poisson fitting",
            "template_sample_sizes": counts.tolist(),
            "spectral": list(spectral_diagnostics),
            "alignment_reference_template": reference_index,
            "column_permutations": permutations,
            "constraint": "theta>0, B symmetric nonnegative, diag(B)=1",
            "objective": "Poisson working log-likelihood over i<j only",
            "true_parameters_used": False,
        },
    }


def _self_test() -> None:
    rng = np.random.default_rng(412057)
    n, k, s_count, t = 30, 3, 3, 180
    z_true = np.tile(np.eye(k), (n // k, 1))
    z_true[15:] = rng.dirichlet(np.full(k, 0.65), n - 15)
    theta_true = rng.uniform(2.5, 5.0, (s_count, n))
    b_true = np.array([[1, .13, .21], [.13, 1, .17], [.21, .17, 1.]])
    templates = np.arange(t) % s_count
    i, j = np.triu_indices(n, 1)
    history = np.zeros((t, n, n))
    true_mu = []
    for s in range(s_count):
        mu = np.outer(theta_true[s], theta_true[s]) * (z_true @ b_true @ z_true.T)
        np.fill_diagonal(mu, 0)
        true_mu.append(mu)
        loc = np.flatnonzero(templates == s)
        y = rng.poisson(mu[i, j], size=(len(loc), len(i)))
        history[loc[:, None], i, j] = y
        history[loc[:, None], j, i] = y
    # A finite-difference check at an interior point is independent of fitting.
    x = np.r_[np.log(theta_true[0]), b_true[np.triu_indices(k, 1)]]
    y = history[templates == 0].mean(0)[i, j]
    _, analytic = _objective(x, y, z_true)
    numerical = np.empty_like(x)
    h = 1e-5
    for c in range(len(x)):
        step = np.zeros_like(x)
        step[c] = h
        numerical[c] = (_objective(x + step, y, z_true)[0]
                        - _objective(x - step, y, z_true)[0]) / (2 * h)
    gradient_error = float(np.max(np.abs(analytic - numerical)))
    assert gradient_error < 1e-6, gradient_error
    result = fit_baseline(history, templates, k)
    assert result["converged"], result["optimization"]
    assert np.allclose(result["mu"], result["mu"].transpose(0, 2, 1))
    assert np.all(result["mu"][:, i, j] > 0)
    assert np.all(np.diagonal(result["mu"], axis1=1, axis2=2) == 0)
    assert np.all(np.diagonal(result["b"], axis1=1, axis2=2) == 1)
    assert np.all(result["b"] >= 0) and np.all(result["theta"] > 0)
    relative_rmse = float(np.sqrt(np.mean((result["mu"][:, i, j]
                                         - np.array(true_mu)[:, i, j]) ** 2))
                          / np.mean(np.array(true_mu)[:, i, j]))
    assert relative_rmse < 0.15, relative_rmse
    regression = _diagonal_regression_test()
    print(json.dumps({
        "passed": True, "gradient_max_absolute_error": gradient_error,
        "fitted_mean_relative_rmse": relative_rmse,
        "optimization": result["optimization"],
        "diagonal_completion": result["diagnostics"]["diagonal_completion"],
        "diagonal_regression": regression,
    }, indent=2))


def _diagonal_regression_test() -> dict:
    """A missing-diagonal trap and a genuinely indefinite admissible B.

    This test recreates the phase-3 IC population and draw exactly, without
    reading any experimental outcome.  Truth is used only to generate test
    counts and assess fitted means after historical-only estimation.
    """
    master, n, k, s_count, t_count = 20261023, 60, 3, 4, 1500
    make_seed = lambda *parts: int(np.random.SeedSequence(
        [master, *parts]).generate_state(1)[0])
    rng = np.random.default_rng(make_seed(10))
    z = np.zeros((n, k))
    for i in range(9):
        z[i, i % k] = 1
    for i in range(9, n):
        alpha = np.full(k, .8)
        alpha[i % k] = 3.2
        z[i] = rng.dirichlet(alpha)
    activity = np.exp(.55 * rng.normal(size=n) - .5 * .55 ** 2)
    phase = rng.uniform(0, 2 * np.pi, n)
    theta = (activity[None, :] * np.sqrt(np.array([.6, .8, 1.4, 1.2]))[:, None]
             * np.exp(.15 * np.sin(phase[None, :]
                                   + np.arange(s_count)[:, None] * np.pi / 2)))
    b = np.array([[1, .30, .20], [.30, 1, .25], [.20, .25, 1.]])
    ij = np.triu_indices(n, 1)
    mu = theta[:, :, None] * theta[:, None, :] * (z @ b @ z.T)[None]
    theta *= np.sqrt(5.0 / mu[:, ij[0], ij[1]].mean())
    templates = np.arange(t_count) % s_count
    results = {}
    for name, affinity, draw_seed in (
        ("phase3_same_IC", b, make_seed(20, 5)),
        ("indefinite_B", np.array([[1., 1.65, .2], [1.65, 1., .25],
                                   [.2, .25, 1.]]), 80931),
    ):
        mu = (theta[:, :, None] * theta[:, None, :]
              * (z @ affinity @ z.T)[None])
        for matrix in mu:
            np.fill_diagonal(matrix, 0)
        rng = np.random.default_rng(draw_seed)
        counts = rng.poisson(mu[templates][:, ij[0], ij[1]])
        history = np.zeros((t_count, n, n))
        history[:, ij[0], ij[1]] = counts
        history[:, ij[1], ij[0]] = counts
        fitted = fit_baseline(history, templates, k)
        relative_error = float(np.linalg.norm(fitted["mu"] - mu) / np.linalg.norm(mu))
        spectral = fitted["diagnostics"]["spectral"]
        assert fitted["converged"]
        assert all(d["nonpositive_vertex_denominators"] == 0 for d in spectral)
        assert all(d["minimum_vertex_denominator"] > 0 for d in spectral)
        assert relative_error < (.025 if name == "phase3_same_IC" else .06)
        if name == "indefinite_B":
            assert np.min(np.linalg.eigvalsh(affinity)) < 0
            assert all(np.min(d["eigenvalues"]) < 0 for d in spectral)
        results[name] = {
            "mean_relative_rmse": relative_error,
            "true_B_eigenvalues": np.linalg.eigvalsh(affinity).tolist(),
            "fitted_signal_eigenvalues": [d["eigenvalues"] for d in spectral],
            "nonpositive_vertex_denominators": 0,
            "converged": True,
        }
    return results


if __name__ == "__main__":
    _self_test()
