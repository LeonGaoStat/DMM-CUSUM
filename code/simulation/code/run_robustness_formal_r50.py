#!/usr/bin/env python3
"""Formal five-setting robustness experiment aligned with the ARL0 study.

The complete design is fixed before execution.  Each of 50 independent Phase-I
histories receives a separate fit and scenario-specific Monte Carlo calibration.
Calibration, IC validation, theta-OC, and z-OC use separate continuous paths.
No result-based stopping, seed selection, or numerical editing is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

import run_robustness_aligned as robust
import run_simulation as sim


SETTINGS = [
    ("Conditional Poisson baseline", 0, 0.00),
    ("Compound Poisson, r = 0.20", 1, 0.20),
    ("Distance effect, beta = 0.30", 2, 0.30),
    ("Missing edges, p_mis = 0.20", 3, 0.20),
    ("Zero-inflated Poisson, omega = 0.10", 4, 0.10),
]


def poisson_kl(alternative: np.ndarray, baseline: np.ndarray) -> float:
    alternative = np.maximum(np.asarray(alternative, dtype=float), sim.EPS)
    baseline = np.maximum(np.asarray(baseline, dtype=float), sim.EPS)
    return float(np.mean(np.sum(
        alternative * np.log(alternative / baseline) - alternative + baseline,
        axis=0,
    )))


def match_poisson_kl(
    baseline: np.ndarray, ratio: np.ndarray, target: float,
) -> tuple[np.ndarray, float]:
    """Scale a fixed log-ratio path until its population Poisson KL is target."""
    ratio = np.maximum(np.asarray(ratio, dtype=float), sim.EPS)
    if target <= 0.0:
        return np.asarray(baseline, dtype=float).copy(), 0.0

    def shifted(scale: float) -> np.ndarray:
        return np.asarray(baseline, dtype=float) * np.exp(scale * np.log(ratio))

    low, high = 0.0, 1.0
    while poisson_kl(shifted(high), baseline) < target:
        high *= 2.0
        if high > 64.0:
            raise RuntimeError("Unable to bracket the target Poisson KL")
    for _ in range(70):
        middle = 0.5 * (low + high)
        if poisson_kl(shifted(middle), baseline) < target:
            low = middle
        else:
            high = middle
    scale = 0.5 * (low + high)
    return shifted(scale), scale


def population_means(case: object, original: object, kind: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    nodes = sim.select_source_nodes(original)
    theta_reference, _ = sim.theta_shift_mu(
        original, 0.15, "Template-opposed activity", nodes,
    )
    z_reference, _ = sim.localized_z_shift_mu(
        original, 0.50, nodes, affected_fraction=0.10,
        theta_coupling=0.03, theta_pattern="common", theta_power=1.0,
        theta_scope="all", preserve_degrees=True, shift_mode="secondary",
    )
    theta_ratio = theta_reference / np.maximum(original.mu_templates, sim.EPS)
    z_ratio = z_reference / np.maximum(original.mu_templates, sim.EPS)
    theta_target = poisson_kl(theta_reference, original.mu_templates)
    z_target = poisson_kl(z_reference, original.mu_templates)

    if kind == 2:
        theta_means, theta_scale = match_poisson_kl(case.mu_templates, theta_ratio, theta_target)
        z_means, z_scale = match_poisson_kl(case.mu_templates, z_ratio, z_target)
    else:
        theta_means = case.mu_templates * theta_ratio
        z_means = case.mu_templates * z_ratio
        theta_scale = z_scale = 1.0
    meta = {
        "theta_target_kl": theta_target,
        "z_target_kl": z_target,
        "theta_actual_kl": poisson_kl(theta_means, case.mu_templates),
        "z_actual_kl": poisson_kl(z_means, case.mu_templates),
        "theta_logratio_scale": theta_scale,
        "z_logratio_scale": z_scale,
    }
    return theta_means, z_means, meta


def generate_pool(
    case: object, cfg: sim.Config, kind: int, level: float,
    length: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    pool_cfg = replace(cfg, seed=seed)
    return robust.phase_history(case, pool_cfg, kind, level, size=length)


def evaluate_pool(
    fitted: sim.FittedMethod, case: object, cfg: sim.Config,
    kind: int, level: float, means: np.ndarray, length: int,
    paths: int, seed: int, h: float | None,
) -> tuple[float | None, np.ndarray]:
    shifted = sim.with_mu(case, means)
    counts, templates, mask = generate_pool(
        shifted, cfg, kind, level, length, sim.stable_seed(seed, 1),
    )
    monitoring_input = sim.proposed_robust_input(fitted, counts, templates, mask)
    indices = sim.contiguous_indices(
        np.random.default_rng(sim.stable_seed(seed, 2)),
        len(monitoring_input), paths, cfg.max_len,
    )
    states = sim.spring_states(
        monitoring_input, indices, fitted.details["spring"], cfg.k_cusum,
    )
    if h is None:
        h, runs = sim.choose_limit(states, cfg.target_arl0)
        return h, runs
    return None, sim.run_lengths(states, h)


def cluster_summary(values: list[np.ndarray]) -> dict[str, float | int]:
    cluster_means = np.asarray([np.mean(item) for item in values], dtype=float)
    pooled = np.concatenate(values)
    return {
        "mean": float(pooled.mean()),
        "se": float(cluster_means.std(ddof=1) / math.sqrt(len(cluster_means))),
        "pooled_se": float(pooled.std(ddof=1) / math.sqrt(len(pooled))),
        "median": float(np.median(pooled)),
        "histories": int(len(cluster_means)),
        "paths": int(len(pooled)),
        "censored_rate": float(np.mean(pooled > 800)),
    }


def write_outputs(outdir: Path, records: list[dict[str, object]]) -> pd.DataFrame:
    history = pd.DataFrame(records)
    history.to_csv(outdir / "history_level_results.csv", index=False)
    rows: list[dict[str, object]] = []
    for setting, group in history.groupby("setting", sort=False):
        row: dict[str, object] = {
            "setting": setting,
            "h_mean": float(group.h.mean()),
            "h_sd": float(group.h.std(ddof=1)),
            "bmax_mean": float(group.bmax.mean()),
            "theta_actual_kl": float(group.theta_actual_kl.iloc[0]),
            "z_actual_kl": float(group.z_actual_kl.iloc[0]),
        }
        for label in ("arl0", "arl1_theta", "arl1_z"):
            means = group[label].to_numpy(float)
            row[label] = float(means.mean())
            row[label + "_se"] = float(means.std(ddof=1) / math.sqrt(len(means)))
            row[label + "_pooled_se"] = float(group[label + "_pooled_se"].mean())
            row[label + "_censored_rate"] = float(group[label + "_censored_rate"].mean())
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(outdir / "table_robustness.csv", index=False)

    def value(row: pd.Series, name: str) -> str:
        return f"{row[name]:.2f} ({row[name + '_se']:.2f})"

    md = [
        "| Setting | ARL0 (SE) | ARL1,theta (SE) | ARL1,z (SE) |",
        "|---|---:|---:|---:|",
    ]
    tex = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Robustness of DMM-CUSUM: ARL estimates with history-level SEs.}",
        r"\label{tab:robustness-formal}", r"\begin{tabular}{lccc}", r"\toprule",
        r"Setting & $\mathrm{ARL}_0$ & $\mathrm{ARL}_{1,\theta}$ & $\mathrm{ARL}_{1,z}$ \\",
        r"\midrule",
    ]
    tex_labels = [
        "Conditional Poisson baseline",
        r"Compound Poisson, $r=0.20$",
        r"Distance effect, $\beta=0.30$",
        r"Missing edges, $p_{\mathrm{mis}}=0.20$",
        r"Zero-inflated Poisson, $\omega=0.10$",
    ]
    for tex_label, (_, row) in zip(tex_labels, result.iterrows()):
        cells = [value(row, name) for name in ("arl0", "arl1_theta", "arl1_z")]
        md.append("| " + " | ".join([str(row.setting)] + cells) + " |")
        tex.append(tex_label + " & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (outdir / "table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (outdir / "table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--histories", type=int, default=50)
    parser.add_argument("--calibration-windows", type=int, default=1000)
    parser.add_argument("--validation-windows", type=int, default=200)
    parser.add_argument("--pool-length", type=int, default=24000)
    parser.add_argument("--m0", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    cfg = sim.Config(
        seed=args.seed, k_cusum=0.15, max_len=800,
        target_arl0=200.0, templates=4,
    )
    original = sim.build_case("II", cfg)
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "design_frozen_before_execution": True,
        "case": "II", "n": 30, "K": 3, "m0": args.m0,
        "settings": SETTINGS,
        "histories": args.histories,
        "calibration_windows_per_history": args.calibration_windows,
        "validation_windows_per_history_and_condition": args.validation_windows,
        "total_paths_per_table_cell": args.histories * args.validation_windows,
        "pool_length": args.pool_length, "horizon": cfg.max_len,
        "target_arl0": cfg.target_arl0, "k": cfg.k_cusum,
        "theta_delta": 0.15, "z_delta": 0.50,
        "distance_OC_matching": "population Poisson KL matched to baseline along fixed log-ratio path",
        "se": "SD of the 50 history-specific mean run lengths divided by sqrt(50)",
        "calibration": "separate normal Monte Carlo path and bisection for every fitted history",
        "selection": "no result-based stopping, parameter changes, seed selection, or row filtering",
        "config": asdict(cfg),
        "code_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), Path(sim.__file__), Path(sim.BASE_SCRIPT), Path(robust.__file__)]
        },
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    records: list[dict[str, object]] = []
    started = time.time()
    for setting_index, (label, kind, level) in enumerate(SETTINGS):
        mu = (
            sim.distance_adjusted_mu(original, level, sim.stable_seed(cfg.seed, 400))
            if kind == 2 else original.mu_templates
        )
        case = sim.with_mu(original, mu)
        theta_means, z_means, shift_meta = population_means(case, original, kind)
        print(
            f"START {label}: theta KL={shift_meta['theta_actual_kl']:.4f}, "
            f"z KL={shift_meta['z_actual_kl']:.4f}", flush=True,
        )
        for phase in range(args.histories):
            phase_seed = sim.stable_seed(cfg.seed, 1000, setting_index, phase)
            phase_cfg = replace(cfg, seed=phase_seed)
            history, templates, mask = robust.phase_history(
                case, phase_cfg, kind, level, size=args.m0,
            )
            fitted = sim.fit_proposed_only(
                history, templates, case, cfg, mask, level if kind == 3 else 0.0,
            )
            h, calibration_runs = evaluate_pool(
                fitted, case, cfg, kind, level, mu, args.pool_length,
                args.calibration_windows,
                sim.stable_seed(cfg.seed, 1100, setting_index, phase), None,
            )
            assert h is not None
            _, ic_runs = evaluate_pool(
                fitted, case, cfg, kind, level, mu, args.pool_length,
                args.validation_windows,
                sim.stable_seed(cfg.seed, 1200, setting_index, phase), h,
            )
            _, theta_runs = evaluate_pool(
                fitted, case, cfg, kind, level, theta_means, args.pool_length,
                args.validation_windows,
                sim.stable_seed(cfg.seed, 1300, setting_index, phase), h,
            )
            _, z_runs = evaluate_pool(
                fitted, case, cfg, kind, level, z_means, args.pool_length,
                args.validation_windows,
                sim.stable_seed(cfg.seed, 1400, setting_index, phase), h,
            )
            record: dict[str, object] = {
                "setting": label, "kind": kind, "level": level,
                "history": phase + 1, "h": h,
                "bmax": fitted.details["spring"].bmax,
                "calibration_arl0": float(calibration_runs.mean()),
                **shift_meta,
            }
            for name, runs in (
                ("arl0", ic_runs), ("arl1_theta", theta_runs), ("arl1_z", z_runs),
            ):
                record[name] = float(runs.mean())
                record[name + "_pooled_se"] = float(runs.std(ddof=1) / math.sqrt(len(runs)))
                record[name + "_censored_rate"] = float(np.mean(runs > cfg.max_len))
            records.append(record)
            write_outputs(args.outdir, records)
            print(
                f"  phase {phase + 1:02d}/{args.histories}: h={h:.3f}, "
                f"ARL0={ic_runs.mean():.1f}, theta={theta_runs.mean():.1f}, "
                f"z={z_runs.mean():.1f}; elapsed={time.time() - started:.0f}s",
                flush=True,
            )
    result = write_outputs(args.outdir, records)
    manifest["status"] = "completed"
    manifest["elapsed_seconds"] = time.time() - started
    manifest["rows"] = len(records)
    manifest["table"] = result.to_dict(orient="records")
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
