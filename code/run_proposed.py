#!/usr/bin/env python3
"""DMM-CUSUM core replay: 8 steps; S2 is optional with --include-robustness."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time


def workflow_selection(plan, source_bytes, include_robustness):
    mode = 'with_robustness' if include_robustness else 'core'
    steps = [step for step in plan['steps']
             if include_robustness or step['name'] != 'robustness_all_full']
    plan_hash = hashlib.sha256(source_bytes).hexdigest()
    selection = {'command_plan_sha256':plan_hash,
                 'include_robustness':include_robustness,
                 'selected_steps':[step['name'] for step in steps]}
    fingerprint = hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest()
    return mode, steps, plan_hash, fingerprint


def scoped_command(command, mode):
    return [value.replace('rerun_proposed/', 'rerun_proposed/'+mode+'/', 1)
            if value.startswith('rerun_proposed/') else value for value in command]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list',action='store_true', help='List the selected 8 core steps, or 9 steps with S2; no computation.')
    parser.add_argument('--execute',action='store_true', help='Run IC2 + Activity3 + saved-limit Membership3. S2 is excluded by default.')
    parser.add_argument('--include-robustness',action='store_true', help='Explicitly add the full S2 robustness workflow as step 9.')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    source=root/'proposed_workflow_commands.json'
    source_bytes=source.read_bytes()
    plan=json.loads(source_bytes)
    mode, steps, plan_hash, fingerprint=workflow_selection(plan,source_bytes,args.include_robustness)
    if args.list or not args.execute:
        print('Mode:',mode,'| steps:',len(steps),'| output: rerun_proposed/'+mode)
        for i,step in enumerate(steps,1):
            print(i,step['name'],step['scope'])
        return
    saved_inputs=json.loads((root/'membership_saved_inputs_manifest.json').read_text())
    for item in saved_inputs['files']:
        path=root/item['package_path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['source_sha256']:
            raise RuntimeError('A saved Membership input differs from its archived SHA: '+item['package_path'])
    out=root/'rerun_proposed'/mode
    status=out/'launcher_progress.json'
    if status.exists() and not args.resume:
        raise RuntimeError('Existing rerun: explicitly use --resume only for its own saved checkpoints.')
    state=json.loads(status.read_text()) if args.resume and status.exists() else {
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'command_plan_sha256':plan_hash,'execution_fingerprint_sha256':fingerprint,
        'selected_mode':mode,'include_robustness':args.include_robustness,
        'selected_steps':[step['name'] for step in steps],'completed_steps':[]}
    if state.get('execution_fingerprint_sha256')!=fingerprint:
        raise RuntimeError('The command plan or robustness selection differs from the saved run.')
    out.mkdir(parents=True,exist_ok=True)
    for target in [out/'results/final/table_arl0_long.csv']:
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            shutil.copy2(root/'simulation/results/final/table_arl0_long.csv',target)
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[key]='1'
    env.update(NUMBA_NUM_THREADS='6',NUMBA_CACHE_DIR=str(out/'numba_cache'),
               PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'simulation/code'))
    logs=out/'logs';logs.mkdir(exist_ok=True)
    for step in steps:
        if any(x['name']==step['name'] and x['exit_code']==0 for x in state['completed_steps']):
            continue
        state.update(status='running',current_step=step['name'])
        status.write_text(json.dumps(state,indent=2)+'\n')
        command=[sys.executable,*scoped_command(step['command'][1:],mode)]
        print('START',step['name'],flush=True)
        start=time.perf_counter()
        with (logs/(step['name']+'.log')).open('a' if args.resume else 'w') as handle:
            result=subprocess.run(command,cwd=root,env=env,stdout=handle,stderr=subprocess.STDOUT)
        state['completed_steps'].append(dict(name=step['name'],exit_code=result.returncode,
                                            wall_seconds=time.perf_counter()-start))
        status.write_text(json.dumps(state,indent=2)+'\n')
        if result.returncode:
            raise SystemExit(result.returncode)
    state.update(status='complete',current_step=None,
                 completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    status.write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    main()
