#!/usr/bin/env python3
"""Prespecified robustness rerun: one common population, full first passages.

All four rows are rerun. Nothing is selected by the direction of a result.
The historical OC population is retained; new independent histories and
future paths replace the historical single-fit/finite-window design.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

import arl0_bootstrap_engine_20261102 as boot
import robustness_independent_engine_20261104 as engine
import run_robustness_aligned as previous
import run_robustness_formal_r50 as definitions
import run_simulation as sim

POPULATION_SEED = 20260922
EXPERIMENT_SEED = 20261104
SETTINGS = [
    ('Conditional Poisson baseline', 0, 0.0),
    ('Compound Poisson, r = 0.20', 1, 0.20),
    ('Distance effect, beta = 0.30', 2, 0.30),
    ('Missing edges, p_mis = 0.20', 3, 0.20),
]
SOURCE_NAMES = [
    Path(__file__).name, 'robustness_independent_engine_20261104.py',
    'arl0_bootstrap_engine_20261102.py', 'arl0_independent_engine_20261102.py',
    'run_simulation.py', 'mldcmm_estimation.py',
    'run_robustness_aligned.py', 'run_robustness_formal_r50.py',
]


def dump(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def seeds(setting: int, history: int, stage: int, count: int) -> np.ndarray:
    if not (0 <= setting < 4 and 0 <= history < 100 and 0 <= stage < 5):
        raise ValueError('setting/history/stage out of frozen range')
    if not 1 <= count <= 10000:
        raise ValueError('path count outside 1..10000')
    offset = ((setting * 100 + history) * 5 + stage) * 10000
    ids = np.arange(offset, offset + count, dtype=np.uint64)
    return ((ids * np.uint64(1664525) + np.uint64(1013904223 + EXPERIMENT_SEED))
            & np.uint64(0xffffffff)).astype(np.uint32)


def initialize_worker() -> None:
    from numba import set_num_threads
    set_num_threads(1)


def population(cfg):
    original = sim.build_case('II', cfg)
    distance_seed = sim.stable_seed(POPULATION_SEED, 400)
    rows = []
    for label, kind, level in SETTINGS:
        mu = (sim.distance_adjusted_mu(original, level, distance_seed)
              if kind == 2 else original.mu_templates.copy())
        case = sim.with_mu(original, mu)
        theta, structural, meta = definitions.population_means(case, original, kind)
        rows.append((case, theta, structural, meta))
    return original, rows


def one_history(task):
    setting_index, outer, config, root = task
    cfg = sim.Config(**config)
    outdir = Path(root) / f'setting_{setting_index}'
    outdir.mkdir(exist_ok=True)
    archive = outdir / f'history_{outer:03d}.npz'
    report = archive.with_suffix('.json')
    if archive.exists() and report.exists():
        saved = json.loads(report.read_text())
        if saved.get('complete'):
            return saved
    started = time.perf_counter()
    label, kind, level = SETTINGS[setting_index]
    _, populations = population(cfg)
    case, theta, structural, shift_meta = populations[setting_index]
    history_seed = int(seeds(setting_index, outer, 0, 1)[0])
    counts, observed = engine.sample_robust_path(
        case.mu_templates, case.pi, case.pj, case.n, 1500, history_seed,
        kind, level, cfg.phi, cfg.sigma_g, cfg.sigma_v,
    )
    templates = sim.template_sequence(1500)
    mask = observed.T if kind == 3 else None
    historical_counts = (counts * observed).T.astype(float)
    fitted = sim.fit_proposed_only(
        historical_counts, templates, case, cfg, mask,
        level if kind == 3 else 0.,
    )
    series = sim.proposed_robust_input(fitted, historical_counts, templates, mask)
    spring = fitted.details['spring']
    cal_seeds = seeds(setting_index, outer, 1, cfg.calibration_reps)
    h, calibration, calibration_meta = boot.calibrate(
        series, spring, cfg.target_arl0, cfg.k_cusum,
        cfg.calibration_reps, int(cal_seeds[0]), block=32,
        templates=4, start_phase=0, path_seeds=cal_seeds,
        guard=10000000, tolerance=.002,
    )
    beta, variance = previous.prediction_arrays(spring)
    raw = {
        'history_counts': historical_counts, 'observed': observed,
        'templates': templates, 'muhat': fitted.details['mu'],
        'zhat': fitted.details['z'], 'standardized_phase1': series,
        'calibration_runs': calibration, 'calibration_seeds': cal_seeds,
        'template_mean': spring.template_mean, 'template_sd': spring.template_sd,
        'beta': beta, 'variance': variance, 'h': h, 'bmax': spring.bmax,
    }
    record = {
        'complete': True, 'setting': label, 'setting_index': setting_index,
        'history_index': outer, 'history_seed': history_seed,
        'h': float(h), 'bmax': int(spring.bmax),
        'calibration_arl': float(calibration.mean()),
        'calibration': calibration_meta, 'shift': shift_meta,
    }
    for stage, (name, mu) in enumerate([
        ('arl0', case.mu_templates), ('arl1_theta', theta), ('arl1_z', structural),
    ], start=2):
        validation_seeds = seeds(setting_index, outer, stage, cfg.reps)
        runs = engine.robust_validation_runs(
            mu, fitted.details['mu'], case.pi, case.pj, case.n,
            validation_seeds, kind, level, cfg.phi, cfg.sigma_g, cfg.sigma_v,
            spring.template_mean, spring.template_sd, beta, variance,
            spring.bmax, cfg.k_cusum, h, start_phase=0,
            guard=10000000, parallel=False,
        )
        if runs.shape != (cfg.reps,) or np.any(runs < 1):
            raise RuntimeError('invalid or unfinished run lengths')
        record[name] = float(runs.mean())
        record[name + '_conditional_se'] = float(runs.std(ddof=1) / math.sqrt(len(runs)))
        record[name + '_max'] = int(runs.max())
        raw[name + '_runs'] = runs
        raw[name + '_seeds'] = validation_seeds
    record['elapsed_seconds'] = time.perf_counter() - started
    tmp = archive.with_suffix(f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, **raw)
    tmp.replace(archive)
    dump(report, record)
    return record


def aggregate(outdir: Path, required: int):
    records = []
    for f in sorted(outdir.glob('setting_*/history_*.json')):
        r = json.loads(f.read_text())
        if r.get('complete'):
            records.append(r)
    simple = [{k: v for k, v in r.items() if k not in ('calibration', 'shift')}
              for r in records]
    frame = pd.DataFrame(simple)
    frame.to_csv(outdir / 'history_level_results.csv', index=False)
    rows = []
    for setting_index, (label, _, _) in enumerate(SETTINGS):
        selected = [r for r in records if r['setting_index'] == setting_index]
        if not selected:
            continue
        row = dict(setting=label, setting_index=setting_index,
                   histories=len(selected), complete=len(selected) == required)
        for name in ('arl0', 'arl1_theta', 'arl1_z'):
            values = np.array([r[name] for r in selected])
            row[name] = float(values.mean())
            row[name + '_se'] = (float(values.std(ddof=1) / math.sqrt(len(values)))
                                 if len(values) > 1 else None)
        row['calibration_arl'] = float(np.mean([r['calibration_arl'] for r in selected]))
        row['calibration_outside_tolerance'] = int(sum(
            not r['calibration']['within_tolerance'] for r in selected))
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / 'summary.csv', index=False)
    dump(outdir / 'summary.json', rows)
    if len(rows) == 4 and all(r['complete'] for r in rows):
        table = ['| Setting | ARL0 (SE) | ARL1,theta (SE) | ARL1,z (SE) |',
                 '|---|---:|---:|---:|']
        tex = [r'\begin{table}[htbp]', r'\centering\small',
               r'\caption{Robustness of DMM-CUSUM under a common nested simulation design.}',
               r'\label{tab:robustness-new}', r'\begin{tabular}{lccc}', r'\toprule',
               r'Setting & $\mathrm{ARL}_0$ & $\mathrm{ARL}_{1,\theta}$ & $\mathrm{ARL}_{1,z}$ \\',
               r'\midrule']
        for row in rows:
            cells = [f"{row[n]:.2f} ({row[n+'_se']:.2f})"
                     for n in ('arl0', 'arl1_theta', 'arl1_z')]
            table.append('| ' + ' | '.join([row['setting']] + cells) + ' |')
            tex.append(row['setting'].replace('_', r'\_') + ' & ' + ' & '.join(cells) + r' \\')
        tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
        (outdir / 'table.md').write_text('\n'.join(table) + '\n')
        (outdir / 'table.tex').write_text('\n'.join(tex) + '\n')
        paired = []
        baseline = {r['history_index']: r for r in records if r['setting_index'] == 0}
        for index in range(1, 4):
            selected = [r for r in records if r['setting_index'] == index]
            for name in ('arl0', 'arl1_theta', 'arl1_z'):
                diffs = np.array([r[name] - baseline[r['history_index']][name] for r in selected])
                base_mean = rows[0][name]
                paired.append(dict(setting=SETTINGS[index][0], metric=name,
                                   mean_difference=float(diffs.mean()),
                                   difference_se=float(diffs.std(ddof=1)/math.sqrt(len(diffs))),
                                   percent_change=100. * float(diffs.mean()) / base_mean))
        pd.DataFrame(paired).to_csv(outdir / 'baseline_comparisons.csv', index=False)
    return rows, len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--histories', type=int, default=100)
    parser.add_argument('--bootstrap-reps', type=int, default=10000)
    parser.add_argument('--validation-reps', type=int, default=10000)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.histories <= 100:
        raise ValueError('histories must be 1..100')
    cfg = sim.Config(seed=POPULATION_SEED, k_cusum=.15, block_len=32,
                     reps=args.validation_reps, calibration_reps=args.bootstrap_reps)
    from dataclasses import asdict
    args.outdir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    original, populations = population(cfg)
    manifest = dict(
        experiment='Table 7 robustness rerun aligned to the historical main OC population',
        created_date='2026-11-04', smoke=args.smoke, design_frozen_before_execution=True,
        case='II', n=30, K=3, m0=1500, population_seed=POPULATION_SEED,
        population_mu_sha256=hashlib.sha256(original.mu_templates.tobytes()).hexdigest(),
        experiment_seed=EXPERIMENT_SEED, config=asdict(cfg), settings=SETTINGS,
        histories_per_setting=args.histories, bootstrap_runs_per_history=args.bootstrap_reps,
        validation_runs_per_history_and_mechanism=args.validation_reps,
        total_validation_runs_per_cell=args.histories * args.validation_reps,
        reference_value=.15, block_length=32, maximum_prediction_length=8,
        theta_delta=.15, z_delta=.50,
        theta_construction='six stratified source nodes; template-opposed signs; template total preserved',
        z_construction='10% entropy-by-volume mixed nodes; secondary redistribution; degree compensation; common activity exp(.03*delta_z)',
        distance_construction='fixed normalized Euclidean distances, exp(-.3*d), IC template total preserved',
        distance_shift_matching='prespecified population Poisson KL matched to baseline along each original log-ratio direction; inherited from old robustness protocol, not outcome tuned',
        compound_construction='N~Poisson(lambda/(1+2r)), A=N+2*Binomial(N,r)',
        missing_construction='independent observation probability .8; IPW Phase I counts and Phase II deviance',
        shift_metadata=[x[3] for x in populations],
        calibration='same-history template-matched noncircular block bootstrap; complete first passage; bisection',
        evaluation='fresh independent stationary latent-AR network trajectory per path; first Phase II observation is changed for OC',
        stopping='complete first alarm; guard=10000000 is an error, never a reported run length',
        se='sample SD of history-specific mean run lengths divided by sqrt(histories)',
        common_limit='one calibrated limit per setting/history shared by IC, theta and z evaluations',
        selection='all four prespecified settings and all histories reported; no directional filtering or reseeding',
        workers=args.workers,
        source_hashes={name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in SOURCE_NAMES},
    )
    manifest_path = args.outdir/'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old != manifest:
            raise RuntimeError('existing frozen manifest differs; use a new output directory')
    else:
        dump(manifest_path, manifest)
    snapshot = args.outdir/'source'
    snapshot.mkdir(exist_ok=True)
    for name in SOURCE_NAMES:
        target = snapshot/name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != manifest['source_hashes'][name]:
            raise RuntimeError('existing source snapshot differs')
        if not target.exists():
            shutil.copy2(source/name, target)
    np.savez_compressed(args.outdir/'population.npz', mu=original.mu_templates,
                        z=original.z, theta=original.theta, pi=original.pi, pj=original.pj,
                        **{f'setting_{i}_{name}': value
                           for i, row in enumerate(populations)
                           for name, value in [('ic', row[0].mu_templates), ('theta', row[1]), ('z', row[2])]})
    tasks = [(index, outer, asdict(cfg), str(args.outdir))
             for outer in range(args.histories) for index in range(4)]
    started = time.time()
    print(f'START {len(tasks)} fits; population seed={POPULATION_SEED}; k=.15; workers={args.workers}', flush=True)
    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=context, initializer=initialize_worker,
    ) as executor:
        futures = {executor.submit(one_history, task): task[:2] for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            summaries, done = aggregate(args.outdir, args.histories)
            progress = dict(status='running', completed_fits=done, total_fits=len(tasks),
                            elapsed_seconds=time.time()-started, summaries=summaries)
            dump(args.outdir/'progress.json', progress)
            print(f"DONE {done}/{len(tasks)} setting={row['setting_index']} history={row['history_index']} "
                  f"ARL0={row['arl0']:.3f} theta={row['arl1_theta']:.3f} z={row['arl1_z']:.3f} "
                  f"fit_seconds={row['elapsed_seconds']:.1f} elapsed={time.time()-started:.1f}s", flush=True)
    rows, done = aggregate(args.outdir, args.histories)
    if done != len(tasks) or not all(row['complete'] for row in rows):
        raise RuntimeError('incomplete experiment')
    dump(args.outdir/'progress.json', dict(status='complete', completed_fits=done,
         total_fits=len(tasks), elapsed_seconds=time.time()-started, summaries=rows))
    print('COMPLETE\n'+(args.outdir/'table.md').read_text(), flush=True)


if __name__ == '__main__':
    main()
