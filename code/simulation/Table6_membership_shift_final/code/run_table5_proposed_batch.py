#!/usr/bin/env python3
"""Batched independent-path ARL1 evaluation for DMM-CUSUM only.

All scenarios and chart settings are fixed before reading the OC outcomes.
The control limit comes from the separate equal-ARL simulation benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_simulation as sim


def batch_counts(
    case: object, cfg: sim.Config, means: np.ndarray,
    paths: int, seed: int,
) -> np.ndarray:
    """Independent network paths, preserving shared and node-level AR(1) factors."""
    rng = np.random.default_rng(seed)
    length = cfg.max_len
    common = np.empty((paths, length), dtype=np.float32)
    node = np.empty((paths, case.n, length), dtype=np.float32)
    common[:, 0] = rng.normal(0.0, cfg.sigma_g, size=paths)
    node[:, :, 0] = rng.normal(0.0, cfg.sigma_v, size=(paths, case.n))
    common_noise = rng.normal(
        0.0, cfg.sigma_g * math.sqrt(1.0 - cfg.phi**2),
        size=(paths, length - 1),
    )
    node_noise = rng.normal(
        0.0, cfg.sigma_v * math.sqrt(1.0 - cfg.phi**2),
        size=(paths, case.n, length - 1),
    )
    for t in range(1, length):
        common[:, t] = cfg.phi * common[:, t - 1] + common_noise[:, t - 1]
        node[:, :, t] = cfg.phi * node[:, :, t - 1] + node_noise[:, :, t - 1]
    center = 0.5 * (cfg.sigma_g**2 + 2.0 * cfg.sigma_v**2)
    templates = sim.template_sequence(length, offset=1500, templates=cfg.templates)
    latent = np.exp(
        common[None, :, :] + node[:, case.pi, :].transpose(1, 0, 2)
        + node[:, case.pj, :].transpose(1, 0, 2) - center
    )
    rates = means[:, templates, None].transpose(0, 2, 1) * latent
    return rng.poisson(rates).astype(np.float32)


def proposed_states(
    counts: np.ndarray, fitted: sim.FittedMethod, cfg: sim.Config,
) -> np.ndarray:
    length = counts.shape[2]
    templates = sim.template_sequence(length, offset=1500, templates=cfg.templates)
    mean = np.asarray(fitted.details["mu"])[:, templates]
    raw = sim.poisson_deviance(counts, mean[:, None, :]).sum(axis=0)
    fit = fitted.details["spring"]
    standardized = (
        raw - np.asarray(fit.template_mean)[templates][None, :]
    ) / np.asarray(fit.template_sd)[templates][None, :]
    indices = np.arange(standardized.size, dtype=np.int32).reshape(standardized.shape)
    return sim.spring_states(standardized.ravel(), indices, fit, cfg.k_cusum)


def normal_states(
    case: object, cfg: sim.Config, fitted: sim.FittedMethod,
    repetitions: int, batch_size: int, seed_tag: int,
) -> np.ndarray:
    states = np.empty((repetitions, cfg.max_len), dtype=np.float32)
    for beginning in range(0, repetitions, batch_size):
        size = min(batch_size, repetitions - beginning)
        seed = sim.stable_seed(cfg.seed, seed_tag, beginning)
        counts = batch_counts(case, cfg, case.mu_templates, size, seed)
        states[beginning:beginning + size] = proposed_states(counts, fitted, cfg)
    return states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--calibration-csv")
    parser.add_argument("--calibrate-reps", type=int, default=0)
    parser.add_argument("--validation-reps", type=int, default=0)
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--case", choices=["I", "II", "III"], required=True)
    parser.add_argument("--k-cusum", type=float, default=0.50)
    parser.add_argument("--block-scale", type=float, default=1.0)
    parser.add_argument("--patterns", nargs="*", default=["Template-opposed activity"])
    parser.add_argument("--levels", nargs="*", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.25])
    args = parser.parse_args()
    if args.reps < 2 or args.batch_size < 1:
        raise ValueError("At least two independent paths and a positive batch size are required")
    cfg = sim.Config(
        seed=args.seed, max_len=args.horizon, k_cusum=args.k_cusum,
        block_scale=args.block_scale,
    )
    case = sim.build_case(args.case, cfg)
    sources = sim.select_source_nodes(case)
    history, templates, _ = sim.generate_network_path(
        case, 1500, cfg, sim.stable_seed(cfg.seed, 900),
    )
    fitted = sim.fit_methods(history, templates, case, cfg)[sim.METHODS[0]]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.calibrate_reps:
        calibration_states = normal_states(
            case, cfg, fitted, args.calibrate_reps, args.batch_size, 930,
        )
        h, rl_cal = sim.choose_limit(calibration_states, cfg.target_arl0)
        pd.DataFrame([{
            "method": "DMM-CUSUM", "h": h,
            "calibration_arl0": float(rl_cal.mean()),
            "calibration_censored_rate": float(np.mean(rl_cal > cfg.max_len)),
            "paths": args.calibrate_reps,
        }]).to_csv(outdir / "proposed_calibration.csv", index=False)
        print(f"Calibrated h={h:.5f}, ARL0={rl_cal.mean():.2f}", flush=True)
    else:
        if not args.calibration_csv:
            parser.error("--calibration-csv is required without --calibrate-reps")
        limits = pd.read_csv(args.calibration_csv)
        matching = limits.loc[limits.method.isin([sim.METHODS[0], "DMM-CUSUM"]), "h"]
        if matching.empty:
            raise ValueError("Calibration file does not contain a DMM-CUSUM control limit")
        h = float(matching.iloc[0])
    if args.validation_reps:
        validation_states = normal_states(
            case, cfg, fitted, args.validation_reps, args.batch_size, 931,
        )
        rl = sim.run_lengths(validation_states, h)
        pd.DataFrame([{
            "method": "DMM-CUSUM", "h": h, "validated_arl0": float(rl.mean()),
            "se": float(rl.std(ddof=1) / math.sqrt(args.validation_reps)),
            "censored_rate": float(np.mean(rl > cfg.max_len)),
            "paths": args.validation_reps,
        }]).to_csv(outdir / "proposed_arl0_validation.csv", index=False)
        print(f"Independent ARL0={rl.mean():.2f}", flush=True)
    records = []
    start = time.time()
    for pattern in args.patterns:
        pattern_idx = {
            "Single increase": 0,
            "Single decrease": 1,
            "Mixed directions": 2,
            "Total-preserving mixed": 3,
            "Within-community redistribution": 4,
            "Moment-preserving activity rotation": 5,
            "Template-opposed activity": 6,
        }[pattern]
        for delta in args.levels:
            means, affected = sim.theta_shift_mu(case, delta, pattern, sources)
            runs = np.empty(args.reps, dtype=np.float32)
            for beginning in range(0, args.reps, args.batch_size):
                size = min(args.batch_size, args.reps - beginning)
                seed = sim.stable_seed(cfg.seed, 920, pattern_idx, int(round(delta * 1000)), beginning)
                counts = batch_counts(case, cfg, means, size, seed)
                runs[beginning:beginning + size] = sim.run_lengths(
                    proposed_states(counts, fitted, cfg), h,
                )
            records.append({
                "case": args.case, "pattern": pattern, "delta_theta": delta,
                "method": "DMM-CUSUM",
                "affected_nodes": json.dumps(affected), "h": h,
                "mean_run_length": float(runs.mean()),
                "se": float(runs.std(ddof=1) / math.sqrt(args.reps)),
                "censored_rate": float(np.mean(runs > cfg.max_len)),
                "paths": args.reps, "horizon": cfg.max_len,
                "block_scale": cfg.block_scale,
            })
            pd.DataFrame(records).to_csv(outdir / "proposed_table5.csv", index=False)
            print(f"{pattern} delta={delta:.2f}: ARL={runs.mean():.2f}; "
                  f"elapsed={time.time()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
