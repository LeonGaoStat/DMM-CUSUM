#!/usr/bin/env python3
"""Fixed-setting DMM-window confirmation using unchanged original statistics.

This is a new single-method routing wrapper, not a recovered historical file.
No competitor fitting or scoring is permitted. No-argument/help calls do no work.
"""
from pathlib import Path
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent


def install_proposed_routes(core, source_path, audit_path):
    """Keep only the existing DMM branches by an explicit AST deletion list.

    Numerical expressions in the surviving branches are not rewritten. The
    original source files and evaluate/score/bootstrap functions stay unchanged.
    Changes apply only to this private in-process module instance.
    """
    tree = ast.parse(source_path.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    originals = {name: ast.dump(functions[name], include_attributes=False)
                 for name in ('all_scores', 'selections', 'make_banks')}
    scores = copy.deepcopy(functions['all_scores'])
    assert isinstance(scores.body[-1], ast.Return)
    assert isinstance(scores.body[-1].value, ast.Tuple)
    scores.body[-1].value.elts = scores.body[-1].value.elts[:1]

    selection = copy.deepcopy(functions['selections'])
    # Original first binding is (d, dong, acd) = scores.
    selection.body[0] = ast.parse('d = scores[0]').body[0]
    # Original second binding is (dq, cq) = (DMM q, competitor q).
    second = selection.body[1]
    assert isinstance(second, ast.Assign) and isinstance(second.value, ast.Tuple)
    second.targets = [ast.Name(id='dq', ctx=ast.Store())]
    second.value = second.value.elts[0]
    assert isinstance(selection.body[-1].value, ast.List)
    original_dmm_selection = ast.dump(selection.body[-1].value.elts[0], include_attributes=False)
    selection.body[-1].value.elts = selection.body[-1].value.elts[:1]
    assert ast.dump(selection.body[-1].value.elts[0], include_attributes=False) == original_dmm_selection

    banks = copy.deepcopy(functions['make_banks'])
    returned = banks.body[-1].value
    assert isinstance(returned, ast.Dict)
    kept = [(key, value) for key, value in zip(returned.keys, returned.values)
            if isinstance(key, ast.Constant) and key.value not in ('competitor_q', 'dong', 'acd')]
    returned.keys = [key for key, value in kept]
    returned.values = [value for key, value in kept]
    assert [key.value for key in returned.keys] == ['dmm', 'l', 'q', 'dmm_channel_q']

    adapter_tree = ast.fix_missing_locations(ast.Module(body=[scores, selection, banks], type_ignores=[]))
    exec(compile(adapter_tree, str(source_path) + ' [DMM-only routing]', 'exec'), core.__dict__)
    core.NAMES = ('DMM-window',)

    # Copy the original frozen-IC holdout block, from its RNG initialization
    # through summary export. It retains seed(master,1400,setting), chunk100,
    # all per-replication fields, interval computation, and CSV serialization.
    audit_tree = ast.parse(audit_path.read_text())
    audit_main = next(node for node in audit_tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == 'main')
    first = next(i for i, node in enumerate(audit_main.body)
                 if isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id == 'rng' for target in node.targets))
    last = next(i for i, node in enumerate(audit_main.body)
                if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'oc_reps')
    original_ic_block = copy.deepcopy(audit_main.body[first:last])
    ic = ast.parse('def proposed_ic_holdout(fits, population, width, bank, master, setting, reps, folder, target):\n    pass').body[0]
    ic.body = original_ic_block + [ast.Return(value=ast.Name(id='summary', ctx=ast.Load()))]
    ic_tree = ast.fix_missing_locations(ast.Module(body=[ic], type_ignores=[]))
    exec(compile(ic_tree, str(audit_path) + ' [unchanged IC block]', 'exec'), core.__dict__)

    forbidden_calls = {name:0 for name in ('fit_dong', 'fit_acd', 'dong_node_score', 'acd_node_score')}
    def forbidden_for(name):
        def forbidden(*args, **kwargs):
            forbidden_calls[name] += 1
            raise RuntimeError(f'Competitor call forbidden in a proposed-only workflow: {name}')
        return forbidden
    for name in forbidden_calls:
        setattr(core, name, forbidden_for(name))
    return {
        'kind': 'new AST-derived single-method routing adapter; not historical source recovery',
        'original_source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'original_audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        'source_functions_not_edited': ['dmm_score', 'draw_windows', 'draw_history', 'make_population',
            'bh', 'upper_p', 'conditional_composition_p', 'evaluate', 'fit_dmm_aux', 'interval'],
        'deletions_only_in_score_and_bank_routes': [
            'all_scores: remove Dong and ACD tuple entries',
            'selections: remove competitor unpacking/q and two competitor return branches',
            'make_banks: remove competitor_q/dong/acd return fields'],
        'original_route_AST_sha256': {name: hashlib.sha256(value.encode()).hexdigest()
                                     for name, value in originals.items()},
        'adapter_AST_sha256': hashlib.sha256(ast.dump(adapter_tree, include_attributes=False).encode()).hexdigest(),
        'original_IC_block_AST_sha256': hashlib.sha256(ast.dump(
            ast.Module(body=original_ic_block, type_ignores=[]), include_attributes=False).encode()).hexdigest(),
        'competitor_fit_and_score_sentinels_installed': True,
        'competitor_call_counts': forbidden_calls,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Run the full frozen DMM-only confirmation.')
    parser.add_argument('--out', type=Path, default=HERE/'results_proposed',
                        help='New output directory; an existing path is rejected.')
    args = parser.parse_args()
    if not args.execute:
        parser.print_help()
        return
    out = args.out.expanduser().resolve()
    if out.exists():
        parser.error(f'Output already exists: {out}; choose a new --out directory.')
    out.mkdir(parents=True)
    started = time.perf_counter()
    import numpy as np
    import pandas as pd
    import scipy
    sys.path.insert(0, str(HERE/'code'))
    source_path = HERE/'code/run_search.py'
    spec = importlib.util.spec_from_file_location('_private_proposed_localization', source_path)
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    route_evidence = install_proposed_routes(core, source_path, HERE/'code/audit_frozen_ic.py')
    protocol = json.loads((HERE/'inputs/protocol_phase7.json').read_text())
    selected = pd.read_csv(HERE/'inputs/selected_validation_metrics.csv').iloc[0]
    assert (int(selected.setting_id), int(selected.n), int(selected.mean_edge), int(selected.window)) == (3,120,2,8)
    assert float(selected.effect) == .7 and float(selected.activity_effect) == .18
    assert protocol['ic_history'] == 1500
    assert protocol['screening_family_budget'] == .05
    assert protocol['screening_reference_windows'] == 8000
    assert protocol['diagnostic_reference_windows'] == 4000
    assert protocol['conditional_composition_reps'] == 999
    master, setting, width = protocol['master_seed'], 3, 8
    confirmation_master = core.seed(master,1700,setting)
    assert confirmation_master == 3647432203
    stages = []
    def stage(name, function, *pos, **kw):
        first = time.perf_counter()
        print('START '+name, flush=True)
        result = function(*pos, **kw)
        stages.append({'stage':name,'seconds':time.perf_counter()-first})
        print('DONE '+name+' '+str(stages[-1]['seconds']), flush=True)
        (out/'progress.json').write_text(json.dumps(stages,indent=2))
        return result
    manifest = {
        'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
        'actual_argv':sys.argv,'python':sys.version,'python_executable':sys.executable,
        'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
        'platform':platform.platform(),'thread_environment':{key:os.environ.get(key) for key in
            ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
        'protocol':{'setting':3,'n':120,'mean_edge':2,'window':8,'history':1500,
            'screening_reference_windows':8000,'type_reference_windows':4000,
            'IC_reps':4000,'OC_reps_per_type':2000,'composition_effect':.7,'activity_effect':.18,
            'master_seed':master,'confirmation_seed':confirmation_master,
            'IC_seed':core.seed(master,1400,setting),'conditional_composition_reps':999},
        'adapter_evidence':route_evidence,
        'source_sha256':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((HERE/'code').glob('*.py'))},
        'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Fresh DMM fit, DMM-only banks, IC and OC confirmation; no competitor computation.',
        'prior_235_seconds_is_not_this_run_time':True,
    }
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    try:
        population = stage('population',core.make_population,2,master,n=120,
                           source_volume_quantiles=protocol.get('source_volume_quantiles'))
        history, templates = stage('fitting_history',core.draw_history,population,1500,
                                   np.random.default_rng(core.seed(master,20,2)))
        raw_fit = stage('fresh_DMM_fit',core.fit_baseline,history,templates,3)
        fit = stage('DMM_auxiliary_statistics',core.fit_dmm_aux,raw_fit,history,templates)
        np.savez_compressed(out/'fitted_dmm.npz',**{key:fit[key] for key in ('z','mu','theta','b')})
        archived = np.load(HERE/'inputs/archived_fits_n120_mean_2.npz')
        manifest['baseline_exact_matches_archived'] = {
            key:bool(np.array_equal(fit[key],archived[key])) for key in ('z','mu','theta','b')}
        manifest['baseline_all_exact_match'] = all(manifest['baseline_exact_matches_archived'].values())
        bank = stage('screening_bank_8000',core.make_banks,[fit],history,templates,width,8000,
                     master,100+2+1000*120,population=population,screening_budget=.05)
        stage('independent_IC_4000',core.proposed_ic_holdout,[fit],population,width,bank,master,
              setting,4000,out,out/'frozen_ic_audit.csv')
        type_bank = stage('type_bank_4000',core.make_banks,[fit],history,templates,width,4000,
                          master,200+2+1000*120,population=population,screening_budget=.05)
        metrics = stage('OC_3_times_2000',core.evaluate,[fit],population,width,.7,2000,
                        confirmation_master,'confirmation',{'local':bank,'type':type_bank},out,
                        setting,activity_effect=.18,protocol=protocol)
        records = pd.read_csv(out/'confirmation_003_replications.csv.gz')
        strict = {kind:float((group.exact.astype(bool)&group.type_correct.astype(bool)).mean())
                  for kind,group in records.groupby('kind')}
        metrics['exact_source_set_and_type_rate'] = [strict[row.kind] for row in metrics.itertuples()]
        metrics.to_csv(out/'final_dmm_metrics.csv',index=False)
        manifest['full_computation_wall_seconds'] = time.perf_counter()-started
        checks = {}
        for actual, expected in [
            ('final_dmm_metrics.csv','expected_dmm_metrics.csv'),
            ('frozen_ic_audit.csv','expected_dmm_ic_summary.csv'),
            ('confirmation_003_replications.csv.gz','expected_dmm_oc_records.csv.gz'),
            ('frozen_ic_audit_replications.csv.gz','expected_dmm_ic_records.csv.gz')]:
            try:
                pd.testing.assert_frame_equal(pd.read_csv(out/actual),
                    pd.read_csv(HERE/'inputs'/expected),check_exact=True)
                checks[actual] = {'exact_equal_after_CSV_parsing':True}
            except AssertionError as error:
                checks[actual] = {'exact_equal_after_CSV_parsing':False,'details':str(error)}
        manifest['reference_comparisons'] = checks
        manifest['status'] = ('complete' if manifest['baseline_all_exact_match']
            and all(x['exact_equal_after_CSV_parsing'] for x in checks.values())
            and all(value == 0 for value in route_evidence['competitor_call_counts'].values())
            else 'complete_with_differences')
    except BaseException as error:
        manifest.update(status='failed',error=repr(error),wall_seconds_before_failure=time.perf_counter()-started)
        raise
    finally:
        manifest['stages'] = stages
        manifest['competitor_call_counts_confirmed_zero'] = all(
            value == 0 for value in route_evidence['competitor_call_counts'].values())
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
        print(json.dumps({'status':manifest['status'],'out':str(out),
                          'full_computation_wall_seconds':manifest.get('full_computation_wall_seconds')},indent=2),flush=True)


if __name__ == '__main__':
    main()
