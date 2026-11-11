#!/usr/bin/env python3
"""Five prespecified robustness settings, conditional on one Phase I history.

Calibration resamples template-aligned blocks of Phase I S^D, not control
limits or decorrelated innovations. Independent DGP paths validate each h.
No ranking-based selection of seeds, settings, or reported results is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads

import run_simulation as sim


SETTINGS = [
    ("Conditional Poisson baseline", 0, 0.0),
    ("Compound Poisson, r = 0.10", 1, 0.10),
    ("Distance effect, beta = 0.20", 2, 0.20),
    ("Missing edges, p_mis = 0.10", 3, 0.10),
    ("Zero-inflated Poisson, omega = 0.10", 4, 0.10),
]


def draw_counts(rng, rates, kind, level):
    if kind == 1:
        arrivals = rng.poisson(rates / (1 + 2 * level))
        counts = arrivals + 2 * rng.binomial(arrivals, level)
    elif kind == 4:
        counts = rng.poisson(rates / (1 - level))
        counts *= rng.random(rates.shape) >= level
    else:
        counts = rng.poisson(rates)
    mask = None
    if kind == 3:
        mask = rng.random(rates.shape) >= level
        counts *= mask
    return counts.astype(float), mask


def phase_history(case, cfg, kind, level, size=1500):
    rng = np.random.default_rng(sim.stable_seed(cfg.seed, 900))
    templates = sim.template_sequence(size)
    common = np.empty(size)
    node = np.empty((case.n, size))
    common[0] = rng.normal(0, cfg.sigma_g)
    node[:, 0] = rng.normal(0, cfg.sigma_v, case.n)
    scale = math.sqrt(1 - cfg.phi**2)
    for t in range(1, size):
        common[t] = cfg.phi * common[t-1] + rng.normal(0, cfg.sigma_g * scale)
        node[:, t] = cfg.phi * node[:, t-1] + rng.normal(0, cfg.sigma_v * scale, case.n)
    rates = case.mu_templates[:, templates] * np.exp(
        common + node[case.pi] + node[case.pj]
        - .5 * (cfg.sigma_g**2 + 2 * cfg.sigma_v**2)
    )
    counts, mask = draw_counts(rng, rates, kind, level)
    return counts, templates, mask


def block_indices(rng, size, reps, horizon, block=32, templates=4):
    if block % templates or horizon % block:
        raise ValueError("Block must contain whole template cycles; horizon must contain whole blocks")
    starts = rng.integers(0, (size-block)//templates + 1,
                          size=(reps, horizon//block), dtype=np.int32) * templates
    return (starts[:, :, None] + np.arange(block, dtype=np.int32)).reshape(reps, horizon)


@njit(cache=True, parallel=True)
def bootstrap_maxima(series, starts, beta, variance, bmax, k):
    reps, horizon = starts.shape
    output = np.empty((reps, horizon), np.float32)
    for r in prange(reps):
        cp = 0.0
        cm = 0.0
        spring = 0
        largest = 0.0
        for t in range(horizon):
            u = series[starts[r,t]]
            for lag in range(spring):
                u -= beta[spring, lag] * series[starts[r,t-lag-1]]
            u /= math.sqrt(variance[spring])
            cp = max(0.0, cp + u-k)
            cm = max(0.0, cm-u-k)
            largest = max(largest, cp, cm)
            output[r,t] = largest
            spring = min(spring+1,bmax) if max(cp,cm)>0 else 0
    return output


@njit(cache=True)
def calibration_runs(maxima, h):
    result = np.empty(len(maxima), np.float64)
    for i in range(len(maxima)):
        result[i] = np.searchsorted(maxima[i], h, side="right") + 1
    return result


def calibrate(series, fit, cfg, reps, seed):
    beta, variance = prediction_arrays(fit)
    indices = block_indices(np.random.default_rng(seed), len(series), reps, cfg.max_len)
    maxima = bootstrap_maxima(series, indices, beta, variance, fit.bmax, cfg.k_cusum)
    low, high = 0., float(maxima.max())
    for _ in range(40):
        h = (low+high)/2
        runs = calibration_runs(maxima, h)
        if runs.mean() < cfg.target_arl0:
            low = h
        else:
            high = h
    h = (low+high)/2
    return h, calibration_runs(maxima, h)


def prediction_arrays(fit):
    beta = np.zeros((fit.bmax+1, fit.bmax))
    for i in range(1, fit.bmax+1):
        beta[i,:i] = fit.beta[i]
    return beta, np.asarray(fit.variance)


@njit(cache=True, parallel=True)
def independent_runs(mu, muhat, pi, pj, n, seeds, kind, level, phi,
                     sg, sv, template_mean, template_sd, beta, variance,
                     bmax, k, h, horizon):
    """Each row has an independent seed and stops at its first alarm."""
    runs = np.empty(len(seeds), np.float64)
    scale = math.sqrt(1-phi**2)
    center = .5 * (sg*sg+2*sv*sv)
    for r in prange(len(seeds)):
        np.random.seed(seeds[r])
        common = np.random.normal(0, sg)
        node = np.random.normal(0, sv, n)
        lagged = np.zeros(bmax)
        cp = 0.
        cm = 0.
        spring = 0
        runs[r] = horizon+1
        for t in range(horizon):
            if t > 0:
                common = phi*common+np.random.normal(0, sg*scale)
                for i in range(n):
                    node[i] = phi*node[i]+np.random.normal(0, sv*scale)
            s = t % mu.shape[1]
            raw = 0.
            for e in range(len(pi)):
                rate = mu[e,s] * math.exp(common+node[pi[e]]+node[pj[e]]-center)
                if kind == 1:
                    arrivals = np.random.poisson(rate/(1+2*level))
                    count = arrivals+2*np.random.binomial(arrivals, level)
                elif kind == 4:
                    count = 0 if np.random.random()<level else np.random.poisson(rate/(1-level))
                else:
                    count = np.random.poisson(rate)
                weight = 1.
                if kind == 3:
                    weight = 0. if np.random.random()<level else 1./(1-level)
                mean = muhat[e,s]
                dev = 2*(count*math.log(count/mean)-(count-mean)) if count>0 else 2*mean
                raw += weight*dev
            value = (raw-template_mean[s])/template_sd[s]
            u = value
            for lag in range(spring):
                u -= beta[spring,lag]*lagged[lag]
            u /= math.sqrt(variance[spring])
            cp = max(0.,cp+u-k)
            cm = max(0.,cm-u-k)
            if max(cp,cm)>h:
                runs[r] = t+1
                break
            for j in range(bmax-1,0,-1):
                lagged[j] = lagged[j-1]
            lagged[0] = value
            spring = min(spring+1,bmax) if max(cp,cm)>0 else 0
    return runs


def summary(runs, horizon):
    return {"mean": float(runs.mean()), "se": float(runs.std(ddof=1)/math.sqrt(len(runs))),
            "censored": int(np.sum(runs>horizon)), "paths": len(runs)}


def write_table(outdir, rows):
    headers = ["Setting", "ARL0 (SE)", "ARL1,theta (SE)", "ARL1,z (SE)"]
    md = ["| " + " | ".join(headers) + " |", "|---|---:|---:|---:|"]
    tex = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{Robustness of DMM-CUSUM: estimated run lengths (Monte Carlo SE).}",
           r"\label{tab:robustness-aligned}", r"\begin{tabular}{lccc}", r"\toprule",
           r"Setting & $\mathrm{ARL}_0$ (SE) & $\mathrm{ARL}_{1,\theta}$ (SE) & $\mathrm{ARL}_{1,z}$ (SE) \\",
           r"\midrule"]
    for row in rows:
        values = [f"{row[key]['mean']:.2f} ({row[key]['se']:.2f})" for key in ("ic", "theta", "z")]
        md.append("| " + " | ".join([row['setting']]+values) + " |")
        tex.append(row['setting'].replace('_',r'\_') + " & " + " & ".join(values) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (outdir/'table.md').write_text('\n'.join(md)+'\n')
    (outdir/'table.tex').write_text('\n'.join(tex)+'\n')
    (outdir/'results.json').write_text(json.dumps(rows,indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--reps', type=int, default=10000)
    parser.add_argument('--calibration-reps', type=int, default=10000)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--horizon', type=int, default=4096)
    args = parser.parse_args()
    set_num_threads(args.threads)
    cfg = sim.Config(seed=args.seed,k_cusum=.15,max_len=args.horizon)
    original = sim.build_case('II',cfg)
    nodes = sim.select_source_nodes(original)
    theta_mu, theta_nodes = sim.theta_shift_mu(original,.15,'Template-opposed activity',nodes)
    z_mu, z_nodes = sim.localized_z_shift_mu(original,.50,nodes,
        affected_fraction=.10,theta_coupling=.03,theta_pattern='common',
        theta_power=1.,theta_scope='all',preserve_degrees=True,shift_mode='secondary')
    args.outdir.mkdir(parents=True,exist_ok=True)
    manifest = dict(config=asdict(cfg),case='II',n=30,m0=1500,
        theta_delta=.15,z_delta=.50,theta_nodes=theta_nodes,z_nodes=z_nodes,
        settings=SETTINGS,phase_histories=1,block_length=32,
        calibration='Template-aligned moving-block bootstrap of Phase I S^D; fitted quantities fixed',
        se='SD of independent conditional Phase II run lengths / sqrt(reps)',
        calibration_reps=args.calibration_reps,validation_reps=args.reps,
        z_shift='Table 6 secondary redistribution, degree compensation, all theta multiplied by exp(.03*.50)',
        stopping='First alarm, no reset. Horizon censoring always reported; censored means are restricted means.',
        code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(sim.__file__),Path(sim.BASE_SCRIPT)]})
    (args.outdir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    start=time.time()
    for idx,(label,kind,level) in enumerate(SETTINGS):
        mu = sim.distance_adjusted_mu(original,level,sim.stable_seed(cfg.seed,400)) if kind==2 else original.mu_templates
        case = sim.with_mu(original,mu)
        history,templates,mask = phase_history(case,cfg,kind,level)
        fitted = sim.fit_proposed_only(history,templates,case,cfg,mask,level if kind==3 else 0.)
        series = sim.proposed_robust_input(fitted,history,templates,mask)
        fit=fitted.details['spring']
        h,calruns=calibrate(series,fit,cfg,args.calibration_reps,sim.stable_seed(cfg.seed,500,idx))
        print(f'{label}: calibrated h={h:.5f}, bootstrap ARL0={calruns.mean():.2f}',flush=True)
        beta,variance=prediction_arrays(fit)
        row=dict(setting=label,h=h,bmax=fit.bmax,calibration=summary(calruns,cfg.max_len))
        archive=dict(history=history,templates=templates,mu=mu,muhat=fitted.details['mu'],
            zhat=fitted.details['z'],s_phase=series,calibration_runs=calruns)
        for tag,(name,means) in enumerate([('ic',mu),('theta',mu*theta_mu/original.mu_templates),('z',mu*z_mu/original.mu_templates)]):
            seeds=np.random.default_rng(sim.stable_seed(cfg.seed,600,idx,tag)).choice(
                2**32-1,size=args.reps,replace=False).astype(np.uint32)
            runs=independent_runs(means,np.asarray(fitted.details['mu']),case.pi,case.pj,case.n,
                seeds,kind,level,cfg.phi,cfg.sigma_g,cfg.sigma_v,
                np.asarray(fit.template_mean),np.asarray(fit.template_sd),beta,variance,
                fit.bmax,cfg.k_cusum,h,cfg.max_len)
            row[name]=summary(runs,cfg.max_len)
            archive[name+'_runs']=runs
            archive[name+'_seeds']=seeds
            print(f"  {name}: {runs.mean():.2f} ({row[name]['se']:.2f}), censored={row[name]['censored']}; elapsed={time.time()-start:.0f}s",flush=True)
        np.savez_compressed(args.outdir/f'setting_{idx}.npz',**archive)
        (args.outdir/f'fit_{idx}.json').write_text(json.dumps(asdict(fit),indent=2))
        rows.append(row)
        write_table(args.outdir,rows)


if __name__=='__main__':
    main()
