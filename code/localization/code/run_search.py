#!/usr/bin/env python3
"""Transparent, model-matched EXPLORATORY fixed-window localization benchmark.

Not an online first-alarm simulation, and not a general-superiority claim.
All fit inputs are IC observations; truth is used only by the generator/evaluator.
Every evaluated setting, including failed validation, is retained on disk.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import xlogy

from baseline_fit import fit_baseline
from dong_score import fit_dong, node_score as dong_node_score
from acd_model import fit_acd, node_score as acd_node_score

ROOT = Path(__file__).resolve().parent.parent
NAMES = ('DMM-window', 'Dong-score-node-adapted', 'ACD-historical-node-adapted')
TYPES = ('activity', 'composition', 'joint')


def seed(master, *parts):
    return int(np.random.SeedSequence([master, *[int(p) for p in parts]]).generate_state(1)[0])


def dev(y, mu):
    mu = np.maximum(mu, 1e-12)
    return np.maximum(2 * (xlogy(y, y / mu) - y + mu), 0)


def upper_p(sorted_bank, values):
    """Add-one upper tails, including ties, for node-specific reference columns."""
    b = sorted_bank.shape[0]
    return np.column_stack([
        (b - np.searchsorted(sorted_bank[:, i], values[:, i], side='left') + 1) / (b + 1)
        for i in range(values.shape[1])
    ])


def bh(p, q):
    order = np.argsort(p, axis=1)
    ps = np.take_along_axis(p, order, axis=1)
    passed = ps <= q * np.arange(1, p.shape[1] + 1) / p.shape[1]
    r = np.max(np.where(passed, np.arange(1, p.shape[1] + 1), 0), axis=1)
    result = np.zeros_like(p, dtype=bool)
    for b, count in enumerate(r):
        result[b, order[b, :count]] = True
    return result


def interval(hits, total, z=1.959963984540054):
    if total == 0:
        return (float('nan'), float('nan'))
    p = hits / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total)) / den
    radius = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / den
    return (float(center-radius), float(center+radius))


@dataclass
class Population:
    z: np.ndarray
    theta: np.ndarray
    b: np.ndarray
    mu: np.ndarray
    mixed: np.ndarray
    ei: np.ndarray
    ej: np.ndarray


def make_population(mean_edge, master, n=30, source_volume_quantiles=None):
    rng = np.random.default_rng(seed(master, 10))
    k, s = 3, 4
    z = np.zeros((n, k))
    for i in range(9):
        z[i, i % k] = 1
    for i in range(9, n):
        alpha = np.full(k, .8)
        alpha[i % k] = 3.2
        z[i] = rng.dirichlet(alpha)
    mixed = np.flatnonzero((z.max(axis=1) < .95) & (np.arange(n) >= 9))
    theta_base = np.exp(.55 * rng.normal(size=n) - .5 * .55**2)
    phases = rng.uniform(0, 2*np.pi, n)
    q = np.array([.6, .8, 1.4, 1.2])
    theta = theta_base[None, :] * np.sqrt(q[:, None]) * np.exp(.15*np.sin(phases[None, :] + np.arange(s)[:, None]*np.pi/2))
    b = np.array([[1, .30, .20], [.30, 1, .25], [.20, .25, 1.]])
    mu = theta[:, :, None] * theta[:, None, :] * (z @ b @ z.T)[None]
    ei, ej = np.triu_indices(n, 1)
    scale = np.sqrt(mean_edge / mu[:, ei, ej].mean())
    theta *= scale
    mu *= scale**2
    for a in range(s):
        np.fill_diagonal(mu[a], 0)
    if source_volume_quantiles is not None:
        # A DGP restriction only. This candidate pool is never passed to estimators.
        normal_volume = mu.sum(axis=2).mean(axis=0)
        lo, hi = np.quantile(normal_volume[mixed], source_volume_quantiles)
        mixed = mixed[(normal_volume[mixed] >= lo) & (normal_volume[mixed] <= hi)]
        if len(mixed) == 0:
            raise ValueError('No eligible sources in the predefined normal-volume range')
    return Population(z, theta, b, mu, mixed, ei, ej)


def draw_history(pop, length, rng):
    templates = np.arange(length) % len(pop.mu)
    x = rng.poisson(pop.mu[templates][:, pop.ei, pop.ej])
    a = np.zeros((length, len(pop.z), len(pop.z)), dtype=np.float64)
    a[:, pop.ei, pop.ej] = x
    a[:, pop.ej, pop.ei] = x
    return a, templates


def draw_windows(pop, reps, width, rng, kind=None, effect=0., return_mean=False, activity_effect=None):
    n, s = len(pop.z), len(pop.mu)
    templates = np.arange(width) % s
    edge_mu = np.broadcast_to(pop.mu[templates][:, pop.ei, pop.ej], (reps, width, len(pop.ei))).copy()
    sources = np.full(reps, -1, dtype=int)
    signs = np.zeros(reps, dtype=int)
    if kind is not None:
        sources = rng.choice(pop.mixed, reps)
        signs = np.where(np.arange(reps) % 2 == 0, 1, -1)
        rng.shuffle(signs)
        znew = pop.z[sources].copy()
        if kind in ('composition', 'joint'):
            targets = np.argmin(znew, axis=1)
            znew *= (1-effect)
            znew[np.arange(reps), targets] += effect
        kernel = np.einsum('bk,kl,jl->bj', znew, pop.b, pop.z)
        row = pop.theta[:, sources].T[:, :, None] * pop.theta[None] * kernel[:, None, :]
        row[np.arange(reps)[:, None], np.arange(s)[None], sources[:, None]] = 0
        if kind in ('composition', 'joint'):
            old_total = pop.mu[:, sources, :].sum(axis=2).T
            row *= (old_total / row.sum(axis=2))[:, :, None]
        if kind in ('activity', 'joint'):
            da=effect if activity_effect is None else activity_effect
            row *= np.exp(signs*da)[:, None, None]
        for j, src in enumerate(sources):
            incident = (pop.ei == src) | (pop.ej == src)
            partners = np.where(pop.ei[incident] == src, pop.ej[incident], pop.ei[incident])
            edge_mu[j, :, incident] = row[j, templates][:, partners].T
    values = rng.poisson(edge_mu)
    a = np.zeros((reps, width, n, n), dtype=float)
    a[:, :, pop.ei, pop.ej] = values
    a[:, :, pop.ej, pop.ei] = values
    if return_mean:
        return a, templates, sources, signs, edge_mu
    return a, templates, sources, signs


def draw_bootstrap(history, templates, reps, width, rng):
    starts = np.flatnonzero((templates == 0) & (np.arange(len(templates)) + width <= len(templates)))
    picks = rng.choice(starts, reps)
    ix = picks[:, None] + np.arange(width)
    return history[ix], templates[ix]


def fit_dmm_aux(fit, history, templates):
    n = history.shape[1]
    ei, ej = np.triu_indices(n, 1)
    d = dev(history[:, ei, ej], fit['mu'][templates][:, ei, ej])
    incidence = np.zeros((len(ei), n))
    incidence[np.arange(len(ei)), ei] = 1
    incidence[np.arange(len(ei)), ej] = 1
    h = d @ incidence
    total = d.sum(axis=1)
    share = np.stack([h[templates == s].sum(axis=0) / (2*total[templates == s].sum()) for s in range(len(fit['mu']))])
    fit = dict(fit)
    empirical_mu=np.stack([history[templates==s].mean(0) for s in range(len(fit['mu']))])
    fit.update(ei=ei, ej=ej, incidence=incidence, share=share, empirical_mu=empirical_mu)
    return fit


def dmm_score(fit, windows, templates):
    if templates.ndim == 1:
        templates = np.broadcast_to(templates, windows.shape[:2])
    ei, ej, incidence = fit['ei'], fit['ej'], fit['incidence']
    observed = windows[:, :, ei, ej]
    expected = fit['mu'][templates][:, :, ei, ej]
    d = dev(observed, expected)
    total = d.sum(axis=2)
    incident = d @ incidence
    l = (incident - 2*fit['share'][templates] * total[:, :, None]).sum(axis=1)
    y = observed.sum(axis=1) @ incidence
    y0 = expected.sum(axis=1) @ incidence
    q = dev(y, y0)
    g = y-y0
    o = windows.sum(axis=1) @ fit['z']
    e = fit['mu'][templates].sum(axis=1) @ fit['z']
    prop_o = np.divide(o, y[:, :, None], out=np.zeros_like(o), where=y[:, :, None] > 0)
    prop_e = np.divide(e, y0[:, :, None], out=np.zeros_like(e), where=y0[:, :, None] > 0)
    r = 2*y*np.sum(xlogy(prop_o, prop_o/np.maximum(prop_e, 1e-12)), axis=2)
    r[y <= 0] = np.nan
    return {'l': l, 'q': q, 'g': g, 'r': np.maximum(r, 0), 'y': y}


def all_scores(fits, a, t):
    d = dmm_score(fits[0], a, t)
    return d, dong_node_score(fits[1], a, t), acd_node_score(fits[2], a, t)


def make_banks(fits, history, templates, width, reps, master, tag, population=None, screening_budget=.10):
    collected = []
    rng = np.random.default_rng(seed(master, tag, width))
    for start in range(0, reps, 100):
        if population is None:
            a,t=draw_bootstrap(history,templates,min(100,reps-start),width,rng)
        else:
            a,t,_,_=draw_windows(population,min(100,reps-start),width,rng)
        collected.append(all_scores(fits, a, t))
    d = {key: np.concatenate([row[0][key] for row in collected]) for key in collected[0][0]}
    return {'dmm': d, 'l': np.sort(d['l'], axis=0), 'q': np.sort(d['q'], axis=0),
            'dmm_channel_q': screening_budget/2, 'competitor_q': screening_budget,
            'dong': np.sort(np.concatenate([r[1] for r in collected]), axis=0),
            'acd': np.sort(np.concatenate([r[2] for r in collected]), axis=0)}


def selections(scores, banks):
    d, dong, acd = scores
    dq, cq = banks.get('dmm_channel_q', .05), banks.get('competitor_q', .10)
    return [bh(upper_p(banks['l'], d['l']), dq) | bh(upper_p(banks['q'], d['q']), dq),
            bh(upper_p(banks['dong'], dong), cq), bh(upper_p(banks['acd'], acd), cq)]


def conditional_composition_p(fit,source,total,value,width,rng,reps=999):
    if total<=0 or not np.isfinite(value):return np.nan
    templates=np.arange(width)%len(fit['mu'])
    partners=np.arange(len(fit['z']))!=source
    reference=fit['empirical_mu'][templates,source,:].sum(0)[partners]
    reference=(reference+1e-8)/np.sum(reference+1e-8)
    expected=fit['mu'][templates,source,:].sum(0)[partners]@fit['z'][partners]
    expected/=expected.sum()
    counts=rng.multinomial(int(total),reference,size=reps)
    proportions=(counts@fit['z'][partners])/total
    r=2*total*np.sum(xlogy(proportions,proportions/np.maximum(expected,1e-12)),axis=1)
    return float((np.sum(r>=value)+1)/(reps+1))


def evaluate(fits, pop, width, effect, reps, master, stage, banks, outdir, setting_id,activity_effect=None,protocol=None):
    records = []
    type_banks = banks['type']['dmm']
    for typeno, kind in enumerate(TYPES):
        rng = np.random.default_rng(seed(master, 300 if stage == 'search' else 900, setting_id, typeno))
        for first in range(0, reps, 100):
            count = min(100, reps-first)
            a, t, sources, signs = draw_windows(pop, count, width, rng, kind, effect,activity_effect=activity_effect)
            scores = all_scores(fits, a, t)
            predicted = selections(scores, banks['local'])
            d = scores[0]
            for name, selected in zip(NAMES, predicted):
                for j in range(count):
                    src = int(sources[j])
                    hit = bool(selected[j, src])
                    found = np.flatnonzero(selected[j])
                    row = {'stage':stage, 'setting_id':setting_id, 'kind':kind, 'method':name,
                           'rep':first+j, 'source':src, 'direction':int(signs[j]),
                           'tp':int(hit), 'fp':int(len(found)-hit), 'fn':int(not hit),
                           'exact':int(hit and len(found)==1), 'selected':';'.join(map(str,found)),
                           'type_available':False, 'type_correct':False, 'assigned_type':'not_provided',
                           'volume_flag':np.nan, 'composition_flag':np.nan}
                    if name == 'DMM-window' and hit:
                        # Fixed-window type bank: match observed total for composition.
                        pq = (np.sum(type_banks['q'][:,src] >= d['q'][j,src])+1)/(len(type_banks['q'])+1)
                        normal_y = type_banks['y'][:,src]
                        dist = np.abs(normal_y - d['y'][j,src])
                        # A caliper prevents unsupported extrapolation of IC totals.
                        caliper=3*max(float(np.std(normal_y,ddof=1)),1.)
                        candidates=np.flatnonzero(dist<=caliper)
                        near=candidates[np.argsort(dist[candidates])[:max(200,len(normal_y)//4)]]
                        finite = np.isfinite(type_banks['r'][near,src])
                        near = near[finite]
                        row.update(volume_flag=bool(pq<=.025), assigned_type='composition_unavailable')
                        pz=np.nan
                        if protocol and protocol.get('composition_reference_mode')=='conditional_poisson':
                            conditional_rng=np.random.default_rng(seed(master,700 if stage=='search' else 1100,setting_id,typeno,first+j))
                            pz=conditional_composition_p(fits[0],src,d['y'][j,src],d['r'][j,src],width,conditional_rng,
                                    reps=protocol['conditional_composition_reps'])
                        elif len(near) >= 100 and np.isfinite(d['r'][j,src]):
                            pz=(np.sum(type_banks['r'][near,src]>=d['r'][j,src])+1)/(len(near)+1)
                        if np.isfinite(pz):
                            flag_q, flag_z = pq <= .025, pz <= .025
                            label = {(False,False):'unresolved',(True,False):'activity',(False,True):'composition',(True,True):'joint'}[(flag_q,flag_z)]
                            row.update(type_available=True,type_correct=label==kind,assigned_type=label,
                                       composition_flag=bool(flag_z))
                    records.append(row)
        print(f'{stage} setting={setting_id} {kind} reps={reps} done', flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(outdir / f'{stage}_{setting_id:03d}_replications.csv.gz', index=False)
    result=[]
    for (kind,name), group in frame.groupby(['kind','method'], sort=False):
        tp,fp,fn=group[['tp','fp','fn']].sum()
        precision=float(tp/(tp+fp)) if tp+fp else 0.
        recall=float(tp/(tp+fn))
        f1=float(2*tp/(2*tp+fp+fn))
        lo,hi=interval(int(tp),len(group))
        avail=group['type_available'].sum()
        correct=group['type_correct'].sum()
        result.append({'stage':stage,'setting_id':setting_id,'n':len(pop.z),'mean_edge':float(pop.mu[:,pop.ei,pop.ej].mean()),
                       'window':width,'effect':effect,'kind':kind,'method':name,'reps':len(group),
                       'activity_effect':effect if activity_effect is None else activity_effect,
                       'precision':precision,'recall':recall,'recall_ci_low':lo,'recall_ci_high':hi,
                       'f1':f1,'exact_source_set_rate':float(group['exact'].mean()),
                       'type_coverage':float(avail/len(group)) if name=='DMM-window' else np.nan,
                       'conditional_type_accuracy':float(correct/avail) if avail else np.nan,
                       'end_to_end_type_rate':float(correct/len(group)) if name=='DMM-window' else np.nan})
    return pd.DataFrame(result)


def candidate(result,protocol=None,ic_fwer=None):
    p = result.pivot(index='kind',columns='method',values='f1')
    r = result[result.method==NAMES[0]].recall
    passed=bool((r >= .85).all() and p[NAMES[0]].mean() > max(p[NAMES[1]].mean(),p[NAMES[2]].mean()))
    if protocol and protocol.get('phase', 1)>=3:
        ours=result[result.method==NAMES[0]]
        passed=passed and ic_fwer<=.15 and bool((ours.conditional_type_accuracy>=.85).all()) and bool((ours.type_coverage>=.85).all())
        for field, key in [('precision', 'minimum_precision'),
                           ('exact_source_set_rate', 'minimum_exact_source_set_rate'),
                           ('end_to_end_type_rate', 'minimum_end_to_end_type_rate')]:
            if key in protocol:
                passed=passed and bool((ours[field]>=protocol[key]).all())
        if protocol.get('require_each_type_f1_best'):
            passed=passed and bool((p[NAMES[0]]>p[[NAMES[1],NAMES[2]]].max(axis=1)).all())
    return passed


def validate_ic(fits,pop,width,reps,master,banks,outdir,tag):
    counts=np.zeros(3)
    selected_counts=np.zeros(3)
    rng=np.random.default_rng(seed(master, 500,tag,width))
    for start in range(0,reps,100):
        a,t,_,_=draw_windows(pop,min(100,reps-start),width,rng)
        sets=selections(all_scores(fits,a,t),banks)
        for j,x in enumerate(sets):
            counts[j]+=np.sum(x.any(axis=1))
            selected_counts[j]+=x.sum()
    return [{'setting_id':tag,'window':width,'method':name,'reps':reps,'fwer':float(counts[j]/reps),
             'mean_false_nodes':float(selected_counts[j]/reps)} for j,name in enumerate(NAMES)]


def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,(np.integer,np.floating,np.bool_)):return x.item()
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(y) for y in x]
    return x


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--protocol',type=Path,default=ROOT/'protocol.json')
    parser.add_argument('--output-name',default='exploration')
    args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text())
    outdir=ROOT/'results'/('smoke' if args.smoke else args.output_name)
    outdir.mkdir(parents=True,exist_ok=True)
    protocol_hash=hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    if not args.smoke and (outdir/'all_metrics.csv').exists():
        raise RuntimeError('Results already exist: use an explicit new study directory rather than overwrite trials.')
    (outdir/'run_manifest.json').write_text(json.dumps({'started_utc':datetime.now(timezone.utc).isoformat(),
         'protocol_sha256':protocol_hash,'smoke':args.smoke,'python':sys.version,
         'numpy':np.__version__,'scope':protocol['scope'],
         'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}},indent=2))
    mean_grid=[20] if args.smoke else protocol['mean_edge_grid']
    width_grid=[8] if args.smoke else protocol['window_grid']
    effect_grid=[.3] if args.smoke else protocol['effect_grid']
    reps=12 if args.smoke else protocol['search_reps_per_type']
    bank_reps=240 if args.smoke else protocol['screening_reference_windows']
    search_results=[]
    ic_results=[]
    ident=0
    success=False
    started=time.monotonic()
    n_grid=[protocol['n']] if args.smoke else protocol.get('n_grid',[protocol['n']])
    for n,mean_edge in [(n,m) for n in n_grid for m in mean_grid]:
        pop=make_population(mean_edge,protocol['master_seed'],n=n,
                            source_volume_quantiles=protocol.get('source_volume_quantiles'))
        history,templates=draw_history(pop,protocol['ic_history'],np.random.default_rng(seed(protocol['master_seed'],20,mean_edge)))
        print(f'fitting n={n} mean_edge={mean_edge}',flush=True)
        dmm=fit_dmm_aux(fit_baseline(history,templates,3),history,templates)
        dong=fit_dong(history,templates,3,seed=seed(protocol['master_seed'],21,mean_edge))
        acd=fit_acd(history,templates,3,seed=seed(protocol['master_seed'],22,mean_edge))
        fits=[dmm,dong,acd]
        np.savez_compressed(outdir/f'fits_n{n}_mean_{mean_edge}.npz',z=dmm['z'],mu=dmm['mu'],theta=dmm['theta'],b=dmm['b'],
            generated_mu=pop.mu,generated_z=pop.z,generated_theta=pop.theta,generated_b=pop.b)
        errors={'proposed_relative_mean_rmse':float(np.linalg.norm(dmm['mu']-pop.mu)/np.linalg.norm(pop.mu)),
                'proposed_optimization':{'converged':dmm.get('converged'),
                    'optimization':dmm.get('optimization'),'diagnostics':dmm.get('diagnostics')},
                'dong':{key:dong.get(key) for key in ('adaptation','selected_start','optimization_histories')},
                'acd':{'metadata':acd.get('metadata'), 'templates':[
                    {key:t.get(key) for key in ('converged','iterations','optimizer_message','IC_mean_RMSE')}
                    for t in acd.get('template_fits',[])]}}
        (outdir/f'fit_diagnostics_n{n}_mean_{mean_edge}.json').write_text(json.dumps(jsonable(errors),indent=2))
        for width in width_grid:
            print(f'bootstrap banks mean={mean_edge} width={width}',flush=True)
            ref_pop=pop if protocol.get('reference_mode')=='independent_ic_windows' else None
            banks={'local':make_banks(fits,history,templates,width,bank_reps,protocol['master_seed'],100+mean_edge+1000*n,population=ref_pop,
                                     screening_budget=protocol.get('screening_family_budget',.10)),
                   'type':make_banks(fits,history,templates,width,
                       240 if args.smoke else protocol['diagnostic_reference_windows'],
                       protocol['master_seed'],200+mean_edge+1000*n,population=ref_pop,
                       screening_budget=protocol.get('screening_family_budget',.10))}
            ic_rows=validate_ic(fits,pop,width,30 if args.smoke else protocol['ic_validation_reps'],
                 protocol['master_seed'],banks['local'],outdir,1000*n+mean_edge)
            ic_results.extend(ic_rows)
            pd.DataFrame(ic_results).to_csv(outdir/'ic_validation.csv',index=False)
            for effect,activity_effect in [(z,a) for z in effect_grid for a in protocol.get('activity_effect_grid',[z])]:
                ident+=1
                result=evaluate(fits,pop,width,effect,reps,protocol['master_seed'],'search',banks,outdir,ident,
                                activity_effect=activity_effect,protocol=protocol)
                search_results.append(result)
                pd.concat(search_results).to_csv(outdir/'all_metrics.csv',index=False)
                is_candidate=candidate(result,protocol,ic_rows[0]['fwer'])
                with (outdir/'trial_log.jsonl').open('a') as stream:
                    stream.write(json.dumps({'setting_id':ident,'n':n,'mean_edge':mean_edge,'window':width,'effect':effect,'activity_effect':activity_effect,
                       'candidate':is_candidate,'elapsed_seconds':time.monotonic()-started})+'\n')
                print(result[['kind','method','precision','recall','f1']].to_string(index=False),flush=True)
                if is_candidate and not args.smoke:
                    print('Candidate found; independent validation begins.',flush=True)
                    validation=evaluate(fits,pop,width,effect,protocol['validation_reps_per_type'],
                         protocol['master_seed'],'validation',banks,outdir,ident,activity_effect=activity_effect,protocol=protocol)
                    validation.to_csv(outdir/f'validation_{ident:03d}_metrics.csv',index=False)
                    success=candidate(validation,protocol,ic_rows[0]['fwer'])
                    (outdir/f'validation_{ident:03d}_decision.json').write_text(json.dumps({'passed':success,
                         'setting_id':ident,'note':'All search and validation results retained.'},indent=2))
                    if success:
                        validation.to_csv(outdir/'selected_validation_metrics.csv',index=False)
                        break
            if success:break
        if success:break
    (outdir/'completion.json').write_text(json.dumps({'candidate_validated':success,'settings_evaluated':ident,
            'elapsed_seconds':time.monotonic()-started,'scope':protocol['scope'],
            'all_trials_retained':True},indent=2))
    print(json.dumps({'finished':True,'validated':success,'settings':ident,'seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__':
    main()
