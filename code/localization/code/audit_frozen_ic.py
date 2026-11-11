#!/usr/bin/env python3
"""Additional normal-state holdout AFTER an exploratory candidate is frozen."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_search import (NAMES, seed, make_population, draw_history, draw_windows,
                        fit_dmm_aux, fit_dong, fit_acd, make_banks,
                        all_scores, selections, interval, evaluate, candidate)


def main(protocol_path, folder, reps, oc_reps):
    target = folder / 'frozen_ic_audit.csv'
    if target.exists():
        raise RuntimeError('The frozen holdout already exists; do not overwrite it.')
    protocol = json.loads(protocol_path.read_text())
    selected = pd.read_csv(folder / 'selected_validation_metrics.csv').iloc[0]
    n, mean_edge, width, setting = int(selected.n), int(round(selected.mean_edge)), int(selected.window), int(selected.setting_id)
    master = protocol['master_seed']
    confirmation_master = seed(master, 1700, setting)
    lock = {'setting_id': setting, 'n': n, 'mean_edge': mean_edge, 'window': width,
            'composition_effect': float(selected.effect), 'activity_effect': float(selected.activity_effect),
            'ic_reps': reps, 'oc_reps_per_type': oc_reps, 'confirmation_master_seed': confirmation_master,
            'status': 'Frozen before additional holdout draws; retain pass or fail outcome',
            'criterion': protocol['candidate_rule']}
    (folder / 'locked_confirmation_protocol.json').write_text(json.dumps(lock, indent=2))
    population = make_population(mean_edge, master, n=n,
                                 source_volume_quantiles=protocol.get('source_volume_quantiles'))
    history, templates = draw_history(population, protocol['ic_history'], np.random.default_rng(seed(master, 20, mean_edge)))
    saved = np.load(folder / f'fits_n{n}_mean_{mean_edge}.npz')
    dmm = fit_dmm_aux({key: saved[key] for key in ['z', 'mu', 'theta', 'b']}, history, templates)
    # Only observed IC arrays reach either competing estimator.
    fits = [dmm, fit_dong(history, templates, 3, seed=seed(master, 21, mean_edge)),
            fit_acd(history, templates, 3, seed=seed(master, 22, mean_edge))]
    bank = make_banks(fits, history, templates, width, protocol['screening_reference_windows'],
                      master, 100+mean_edge+1000*n, population=population,
                      screening_budget=protocol.get('screening_family_budget',.10))
    rng = np.random.default_rng(seed(master, 1400, setting))
    records = []
    for first in range(0, reps, 100):
        a, t, _, _ = draw_windows(population, min(100, reps-first), width, rng)
        selected_sets = selections(all_scores(fits, a, t), bank)
        for name, matrix in zip(NAMES, selected_sets):
            for j, row in enumerate(matrix):
                found = np.flatnonzero(row)
                records.append({'rep': first+j, 'method': name, 'any_false_selection': bool(len(found)),
                                'false_nodes': len(found), 'selected': ';'.join(map(str, found))})
    frame = pd.DataFrame(records)
    frame.to_csv(folder / 'frozen_ic_audit_replications.csv.gz', index=False)
    rows = []
    for name, g in frame.groupby('method', sort=False):
        hits = int(g.any_false_selection.sum())
        lo, hi = interval(hits, len(g))
        rows.append({'method': name, 'reps': len(g), 'fwer': hits/len(g),
                     'fwer_ci_low': lo, 'fwer_ci_high': hi,
                     'mean_false_nodes': g.false_nodes.mean()})
    summary = pd.DataFrame(rows)
    summary.to_csv(target, index=False)
    if oc_reps:
        type_bank = make_banks(fits, history, templates, width, protocol['diagnostic_reference_windows'],
                               master, 200+mean_edge+1000*n, population=population,
                               screening_budget=protocol.get('screening_family_budget',.10))
        confirmation = evaluate(fits, population, width, float(selected.effect), oc_reps,
                                confirmation_master, 'confirmation', {'local': bank, 'type': type_bank},
                                folder, setting, activity_effect=float(selected.activity_effect), protocol=protocol)
        confirmation.to_csv(folder / 'locked_confirmation_metrics.csv', index=False)
        decision = candidate(confirmation, protocol, float(summary.loc[summary.method == NAMES[0], 'fwer'].iloc[0]))
        (folder / 'locked_confirmation_decision.json').write_text(json.dumps({'passed': bool(decision),
            'scope': 'One locked additional batch at selected exploratory scenario; no general superiority claim'}, indent=2))
        print(confirmation[['kind', 'method', 'precision', 'recall', 'f1', 'exact_source_set_rate', 'end_to_end_type_rate']].to_string(index=False))
    (folder / 'frozen_ic_audit_manifest.json').write_text(json.dumps({
        'status': 'additional holdout after selected candidate was frozen',
        'seed': seed(master, 1400, setting), 'setting_id': setting,
        'replicates': reps, 'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        'scope': 'Unconditional normal fixed windows, not alarm-conditional diagnosis FWER'}, indent=2))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--reps', type=int, default=4000)
    p.add_argument('--oc-reps', type=int, default=2000)
    a = p.parse_args()
    main(a.protocol, a.folder, a.reps, a.oc_reps)
