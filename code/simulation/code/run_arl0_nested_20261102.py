#!/usr/bin/env python3
"""Fixed 100 x 10,000 IC validation, with 10,000 historical bootstrap runs.

The experiment preserves Table 3's data-generating population and estimator.
Only the calibration and validation design are replaced. See the dated PLAN.
Old results and the manuscript are never modified by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

M0_VALUES = [500, 1000, 1500, 2000, 3000]
CASES = ['I', 'II', 'III']
POPULATION_SEED = 20260913
EXPERIMENT_SEED = 20261102


def path_seeds(case_index: int, history_index: int, stage: int, count: int) -> np.ndarray:
    """Disjoint MT19937 initialization seeds across every formal run.

    Stages 0..4 are the five calibrations; stage 5 is common-path validation.
    The odd affine map is a permutation modulo 2**32. Reserved ranges are
    independent of the requested path count so smoke runs cannot alter seeds.
    """
    if not (0 <= case_index < 3 and 0 <= history_index < 100 and 0 <= stage < 6):
        raise ValueError('invalid case/history/stage index')
    if not (0 < count <= 10000):
        raise ValueError('path count must be between 1 and 10000')
    offset = ((case_index * 100 + history_index) * 6 + stage) * 10000
    ids = np.arange(offset, offset + count, dtype=np.uint64)
    return ((ids * np.uint64(1664525) + np.uint64(1013904223 + EXPERIMENT_SEED))
            & np.uint64(0xffffffff)).astype(np.uint32)


def summarize_histories(runs: np.ndarray) -> dict:
    values = np.asarray(runs, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError('expected history-by-path array')
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('unfinished/nonfinite first-passage times cannot be summarized as ARL')
    means = values.mean(axis=1)
    return {
        'arl0': float(means.mean()),
        'se': float(means.std(ddof=1) / math.sqrt(len(means))) if len(means) > 1 else None,
        'phase_histories': len(means),
        'validation_paths_per_history': values.shape[1],
        'paths': values.size,
        'history_mean_sd': float(means.std(ddof=1)) if len(means) > 1 else None,
        'history_mean_min': float(means.min()),
        'history_mean_max': float(means.max()),
        'history_mean_q05': float(np.quantile(means, .05)),
        'history_mean_q50': float(np.quantile(means, .50)),
        'history_mean_q95': float(np.quantile(means, .95)),
        'run_length_max': int(values.max()),
        'unfinished_paths': 0,
    }


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def source_hashes(directory: Path) -> dict:
    names = ['run_arl0_nested_20261102.py', 'arl0_independent_engine_20261102.py',
             'arl0_bootstrap_engine_20261102.py', 'run_simulation.py', 'mldcmm_estimation.py']
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}


def snapshot_sources(outdir: Path) -> Path:
    source = Path(__file__).resolve().parent
    destination = outdir / 'source'
    destination.mkdir(parents=True, exist_ok=True)
    for name, digest in source_hashes(source).items():
        target = destination / name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'source changed since snapshot: {name}')
        else:
            shutil.copy2(source / name, target)
    for name in ['arl0_rerun_20261102_PLAN.md', 'test_run_arl0_nested_20261102.py',
                 'test_arl0_independent_engine_20261102.py', 'test_arl0_bootstrap_engine_20261102.py']:
        if (source / name).exists() and not (destination / name).exists():
            shutil.copy2(source / name, destination / name)
    return destination


def run_history(case_id: str, outer: int, args, cfg, case, directory: Path) -> None:
    import run_simulation as sim
    import arl0_bootstrap_engine_20261102 as boot
    import arl0_independent_engine_20261102 as independent

    case_index = CASES.index(case_id)
    checkpoint = directory / f'history_{outer:03d}.npz'
    report_path = directory / f'history_{outer:03d}.json'
    if checkpoint.exists() and report_path.exists():
        saved = json.loads(report_path.read_text())
        if saved.get('complete'):
            print(f'SKIP case={case_id} history={outer + 1}: complete checkpoint', flush=True)
            return
    started = time.perf_counter()
    history_seed = sim.stable_seed(POPULATION_SEED, EXPERIMENT_SEED, case_index, outer)
    history, templates, _ = sim.generate_network_path(case, max(M0_VALUES), cfg, history_seed)
    fitted = []
    cal_runs = []
    limits = []
    metadata = []
    max_b = cfg.bmax_cap
    betas = np.zeros((len(M0_VALUES), max_b + 1, max_b), dtype=np.float64)
    variances = np.ones((len(M0_VALUES), max_b + 1), dtype=np.float64)
    for j, m0 in enumerate(M0_VALUES):
        fit = sim.fit_proposed_only(history[:, :m0], templates[:m0], case, cfg)
        series = sim.method_input(fit, history[:, :m0], templates[:m0], case, cfg)
        spring = fit.details['spring']
        seeds = path_seeds(case_index, outer, j, args.bootstrap_reps)
        h, runs, information = boot.calibrate(
            series, spring, cfg.target_arl0, cfg.k_cusum, args.bootstrap_reps,
            int(seeds[0]), block=args.block_length, templates=cfg.templates,
            start_phase=m0 % cfg.templates, path_seeds=seeds, guard=args.guard,
            tolerance=args.calibration_tolerance,
        )
        fitted.append(fit)
        cal_runs.append(runs)
        limits.append(h)
        for d in range(1, spring.bmax + 1):
            betas[j, d, :d] = spring.beta[d]
        variances[j, :spring.bmax + 1] = spring.variance
        metadata.append(dict(m0=m0, h=float(h), spring=asdict(spring), calibration=information,
                             calibration_arl=float(np.mean(runs)),
                             calibration_seed_hash=hashlib.sha256(seeds.tobytes()).hexdigest()))
        print(f'CAL case={case_id} history={outer + 1}/{args.histories} m0={m0} '
              f'h={h:.6f} ARLboot={np.mean(runs):.4f} bmax={spring.bmax} '
              f'elapsed={time.perf_counter() - started:.1f}s', flush=True)

    muhat = np.ascontiguousarray(np.stack([fit.details['mu'] for fit in fitted]))
    means = np.asarray([fit.details['spring'].template_mean for fit in fitted], dtype=np.float64)
    sds = np.asarray([fit.details['spring'].template_sd for fit in fitted], dtype=np.float64)
    bmaxes = np.asarray([fit.details['spring'].bmax for fit in fitted], dtype=np.int64)
    seeds = path_seeds(case_index, outer, 5, args.validation_reps)
    runs = independent.validation_runs(
        case.mu_templates, muhat, case.pi, case.pj, case.n, seeds,
        cfg.phi, cfg.sigma_g, cfg.sigma_v, means, sds, betas, variances,
        bmaxes, cfg.k_cusum, np.asarray(limits), start_phase=0, guard=args.guard,
    )
    runs = np.asarray(runs)
    if runs.shape != (args.validation_reps, len(M0_VALUES)) or np.any(runs <= 0):
        raise RuntimeError(f'incomplete validation runs for {case_id}/{outer}: {runs.shape}')
    information = dict(
        case=case_id, history_index=outer, history_seed=int(history_seed),
        population_seed=POPULATION_SEED, complete=True,
        bootstrap_runs_per_fit=args.bootstrap_reps,
        validation_paths=args.validation_reps, block_length=args.block_length,
        history_hash=hashlib.sha256(history.tobytes()).hexdigest(),
        validation_seed_hash=hashlib.sha256(seeds.tobytes()).hexdigest(),
        validation_means=runs.mean(axis=0).tolist(),
        validation_conditional_se=(runs.std(axis=0, ddof=1) / math.sqrt(len(runs))).tolist(),
        elapsed_seconds=time.perf_counter() - started, fits=metadata,
    )
    temporary = checkpoint.with_suffix('.tmp.npz')
    np.savez_compressed(
        temporary, validation_runs=runs, calibration_runs=np.asarray(cal_runs),
        validation_seeds=seeds, h=np.asarray(limits), muhat=muhat,
        zhat=np.stack([fit.details['z'] for fit in fitted]),
        template_means=means, template_sds=sds, betas=betas,
        variances=variances, bmaxes=bmaxes, m0=np.asarray(M0_VALUES),
        true_mu=case.mu_templates,
    )
    temporary.replace(checkpoint)
    write_json(report_path, information)
    print(f'DONE case={case_id} history={outer + 1}/{args.histories} '
          f'ARL0={np.round(runs.mean(axis=0), 3).tolist()} '
          f'elapsed={information["elapsed_seconds"]:.1f}s', flush=True)


def aggregate(outdir: Path, required_histories: int, bootstrap_reps: int, validation_reps: int) -> list:
    rows = []
    old_path = outdir / 'old_table_arl0_long.csv'
    old = pd.read_csv(old_path) if old_path.exists() else None
    for case_id in CASES:
        files = sorted((outdir / f'case_{case_id}').glob('history_*.npz'))
        if not files:
            continue
        collected, hs, cal = [], [], []
        for file in files:
            report = file.with_suffix('.json')
            if not report.exists() or not json.loads(report.read_text()).get('complete'):
                continue
            with np.load(file) as data:
                values = data['validation_runs']
                if values.shape != (validation_reps, len(M0_VALUES)):
                    raise RuntimeError(f'checkpoint path count mismatch: {file}')
                if data['calibration_runs'].shape != (len(M0_VALUES), bootstrap_reps):
                    raise RuntimeError(f'checkpoint calibration count mismatch: {file}')
                collected.append(values.copy())
                hs.append(data['h'].copy())
                cal.append(data['calibration_runs'].mean(axis=1))
        if not collected:
            continue
        runs = np.stack(collected)
        h_values = np.asarray(hs)
        for j, m0 in enumerate(M0_VALUES):
            row = dict(case=case_id, m0=m0, method='DMM-CUSUM',
                       **summarize_histories(runs[:, :, j]),
                       h_mean=float(h_values[:, j].mean()),
                       h_sd=float(h_values[:, j].std(ddof=1)) if len(h_values) > 1 else None,
                       bootstrap_arl_mean=float(np.asarray(cal)[:, j].mean()),
                       bootstrap_runs_per_history=bootstrap_reps,
                       complete=len(collected) == required_histories)
            if old is not None:
                match = old[(old['case'] == case_id) & (old['m0'] == m0)
                            & (old['method'] == 'Proposed ML-DCMM')]
                if len(match) != 1:
                    raise RuntimeError(f'old table row not unique: {case_id}/{m0}')
                prior = match.iloc[0]
                row.update(old_arl0=float(prior.arl0), old_se=float(prior.se),
                           difference=float(row['arl0'] - prior.arl0),
                           relative_difference_pct=float(100 * (row['arl0'] / prior.arl0 - 1)))
            rows.append(row)
    if rows:
        temporary_csv = outdir / f'comparison.{os.getpid()}.tmp.csv'
        pd.DataFrame(rows).to_csv(temporary_csv, index=False)
        temporary_csv.replace(outdir / 'comparison.csv')
        write_json(outdir / 'comparison.json', rows)
        header = '| Case | m0 | Old ARL0 (SE) | New ARL0 (SE) | Histories | Difference |\n|---|---:|---:|---:|---:|---:|\n'
        lines = []
        for row in rows:
            se = f'{row["se"]:.2f}' if row['se'] is not None else 'pending'
            old_text = f'{row["old_arl0"]:.2f} ({row["old_se"]:.2f})' if 'old_arl0' in row else 'NA'
            delta = f'{row["difference"]:+.2f}' if 'difference' in row else 'NA'
            lines.append(f'| {row["case"]} | {row["m0"]} | {old_text} | {row["arl0"]:.2f} ({se}) | {row["phase_histories"]} | {delta} |')
        temporary_md = outdir / f'comparison.{os.getpid()}.tmp.md'
        temporary_md.write_text(header + '\n'.join(lines) + '\n')
        temporary_md.replace(outdir / 'comparison.md')
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=CASES)
    parser.add_argument('--histories', type=int, default=100)
    parser.add_argument('--bootstrap-reps', type=int, default=10000)
    parser.add_argument('--validation-reps', type=int, default=10000)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--block-length', type=int, default=32)
    parser.add_argument('--guard', type=int, default=10000000)
    parser.add_argument('--calibration-tolerance', type=float, default=.002)
    parser.add_argument('--history-start', type=int, default=0)
    parser.add_argument('--history-stop', type=int)
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not (1 <= args.histories <= 100):
        raise ValueError('histories must be in 1..100')
    if not args.smoke and (args.histories, args.bootstrap_reps, args.validation_reps) != (100, 10000, 10000):
        raise ValueError('formal design is fixed at 100 histories x 10000 bootstrap x 10000 validation')
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.snapshot:
        destination = snapshot_sources(args.outdir)
        print(f'Immutable execution source prepared: {destination}', flush=True)
        return
    if args.aggregate_only:
        rows = aggregate(args.outdir, args.histories, args.bootstrap_reps, args.validation_reps)
        print(json.dumps(rows, indent=2))
        return
    import run_simulation as sim
    from numba import set_num_threads
    set_num_threads(args.threads)
    cfg = sim.Config(seed=POPULATION_SEED, k_cusum=.5)
    original_csv = Path(sim.__file__).resolve().parents[1] / 'results/final/table_arl0_long.csv'
    # For execution from the immutable source directory use its saved copy.
    if not (args.outdir / 'old_table_arl0_long.csv').exists():
        if not original_csv.exists():
            original_csv = Path(__file__).resolve().parents[2] / 'simulation/results/final/table_arl0_long.csv'
        shutil.copy2(original_csv, args.outdir / 'old_table_arl0_long.csv')
    manifest = dict(
        experiment='DMM-CUSUM Table 3 nested IC rerun', created_date='2026-11-02',
        smoke=args.smoke, population_config=asdict(cfg),
        cases=CASES, m0_values=M0_VALUES, histories_per_cell=args.histories,
        calibration_runs_per_history=args.bootstrap_reps,
        validation_paths_per_history=args.validation_reps,
        block_length=args.block_length, block_boundary='noncircular, aligned complete cycles',
        bootstrap_source='only the same fitted Phase I history, complete-network-equivalent S^D blocks',
        validation_source='fresh stationary latent-AR lognormal-Poisson paths, no finite-pool windows',
        validation_sharing='common counts across the five m0 charts within each case/history',
        seed_scheme='disjoint uint32 affine permutation by case, outer, stage and path',
        experiment_seed=EXPERIMENT_SEED, safety_guard=args.guard,
        stopping='exact first alarm; numerical safety-guard failures are errors, never run lengths',
        estimator='unchanged Table 3 spectral/multiplicative fit and fit_spring; no optimizer replacement',
        se='sample SD of history-specific validation means divided by sqrt(number of histories)',
        source_hashes=source_hashes(Path(__file__).resolve().parent),
        calibration_tolerance=args.calibration_tolerance,
    )
    path = args.outdir / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError('existing run manifest differs; use a new result directory')
    write_json(path, manifest)
    started = time.perf_counter()
    stop = args.histories if args.history_stop is None else min(args.history_stop, args.histories)
    for case_id in args.cases:
        case = sim.build_case(case_id, cfg)
        directory = args.outdir / f'case_{case_id}'
        directory.mkdir(exist_ok=True)
        for outer in range(args.history_start, stop):
            run_history(case_id, outer, args, cfg, case, directory)
            aggregate(args.outdir, args.histories, args.bootstrap_reps, args.validation_reps)
        print(f'CASE FINISHED {case_id}, elapsed={time.perf_counter() - started:.1f}s', flush=True)
    rows = aggregate(args.outdir, args.histories, args.bootstrap_reps, args.validation_reps)
    done = len(rows) == 15 and all(row['complete'] for row in rows)
    write_json(args.outdir / 'status.json', dict(complete=done, cells=len(rows),
                                               elapsed_seconds=time.perf_counter() - started))
    print(f'RUN FINISHED complete={done} elapsed={time.perf_counter() - started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
