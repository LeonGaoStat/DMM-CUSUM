#!/usr/bin/env python3
"""Batched Table 6 evaluation for DMM-CUSUM under membership shifts."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_simulation as sim
from run_table5_proposed_batch import batch_counts, proposed_states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--case", choices=["I", "II", "III"], required=True)
    parser.add_argument("--calibration-csv", required=True)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--affected-fraction", type=float, default=0.10)
    parser.add_argument("--theta-coupling", type=float, default=0.0)
    parser.add_argument("--theta-power", type=float, default=2.0)
    parser.add_argument("--theta-scope", choices=["affected", "all"], default="affected")
    parser.add_argument("--preserve-degrees", action="store_true")
    parser.add_argument(
        "--theta-pattern", choices=["balanced", "common", "common-down", "template-opposed"],
        default="balanced",
    )
    parser.add_argument("--horizon", type=int, default=800)
    parser.add_argument("--k-cusum", type=float, default=0.10)
    parser.add_argument("--target-community", type=int)
    parser.add_argument("--shift-mode", choices=["coherent", "secondary"], default="coherent")
    parser.add_argument(
        "--levels", nargs="*", type=float,
        default=[0.10, 0.15, 0.20, 0.25, 0.30],
    )
    args = parser.parse_args()

    cfg = sim.Config(
        seed=args.seed, max_len=args.horizon, k_cusum=args.k_cusum,
    )
    case = sim.build_case(args.case, cfg)
    nodes = sim.select_source_nodes(case)
    history, templates, _ = sim.generate_network_path(
        case, 1500, cfg, sim.stable_seed(cfg.seed, 900),
    )
    fitted = sim.fit_methods(history, templates, case, cfg)[sim.METHODS[0]]
    limits = pd.read_csv(args.calibration_csv)
    match = limits.loc[limits.method.isin([sim.METHODS[0], "DMM-CUSUM"]), "h"]
    if match.empty:
        raise ValueError("Calibration file does not contain a DMM-CUSUM limit")
    h = float(match.iloc[0])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    started = time.time()
    for delta in args.levels:
        means, affected = sim.localized_z_shift_mu(
            case, delta, nodes, affected_fraction=args.affected_fraction,
            reference_mu=np.asarray(fitted.details["mu"]),
            theta_coupling=args.theta_coupling,
            theta_pattern=args.theta_pattern,
            theta_power=args.theta_power,
            theta_scope=args.theta_scope,
            preserve_degrees=args.preserve_degrees,
            target_community=args.target_community,
            shift_mode=args.shift_mode,
        )
        metrics = sim.localized_z_shift_metrics(
            case, delta, nodes, affected_fraction=args.affected_fraction,
            reference_mu=np.asarray(fitted.details["mu"]),
            theta_coupling=args.theta_coupling,
            theta_pattern=args.theta_pattern,
            theta_power=args.theta_power,
            theta_scope=args.theta_scope,
            preserve_degrees=args.preserve_degrees,
            target_community=args.target_community,
            shift_mode=args.shift_mode,
        )
        runs = np.empty(args.reps, dtype=np.float32)
        for start in range(0, args.reps, args.batch_size):
            size = min(args.batch_size, args.reps - start)
            counts = batch_counts(
                case, cfg, means, size,
                sim.stable_seed(cfg.seed, 960, start),
            )
            runs[start:start + size] = sim.run_lengths(
                proposed_states(counts, fitted, cfg), h,
            )
        records.append({
            "case": args.case,
            "pattern": "Fixed-direction membership migration",
            "delta_z": delta,
            "method": "DMM-CUSUM",
            "affected_nodes": json.dumps(affected),
            "h": h,
            "mean_run_length": float(runs.mean()),
            "se": float(runs.std(ddof=1) / math.sqrt(args.reps)),
            "censored_rate": float(np.mean(runs > cfg.max_len)),
            "paths": args.reps,
            "horizon": cfg.max_len,
            "affected_fraction": args.affected_fraction,
            "theta_coupling": args.theta_coupling,
            "theta_pattern": args.theta_pattern,
            "theta_power": args.theta_power,
            "theta_scope": args.theta_scope,
            "preserve_degrees": args.preserve_degrees,
            **metrics,
        })
        pd.DataFrame(records).to_csv(outdir / "proposed_table6.csv", index=False)
        print(
            f"DMM case={args.case} delta_z={delta:.2f}: "
            f"ARL={runs.mean():.2f}; elapsed={time.time()-started:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
