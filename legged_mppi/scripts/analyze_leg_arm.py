"""Run unchanged Go1-DM MPPI and report leg/arm proposal correlations."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault('MPLCONFIGDIR', '/tmp/leg-arm-matplotlib')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from whole_body_mppi.analysis.leg_arm import LegArmObserver

LEGS = [f'{leg}_{joint}' for leg in ('FR', 'FL', 'RR', 'RL') for joint in ('hip', 'thigh', 'calf')]
ARMS = [f'arm_{i}' for i in range(1, 7)]


def mean(x, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(x, axis=axis)


def run_seed(args, seed, directory):
    from whole_body_mppi.control.tasks.whole_body import WholeBodyController
    from whole_body_mppi.control.tasks.ee_velocity import EEVelocityController
    from whole_body_mppi.control.tasks.definitions import get_task
    from whole_body_mppi.control.tasks.base import PACKAGE_ROOT
    from whole_body_mppi.control.solvers.options import apply_mppi_mode
    from whole_body_mppi.interface.simulator import Simulator
    task = get_task(args.task)
    cls = EEVelocityController if task.get('control_mode') == 'cmd_vel' else WholeBodyController
    agent = cls(args.task, 'go1_dm', 'full')
    try:
        apply_mppi_mode(agent, None if args.mode == 'configured' else args.mode)
        agent.random_generator = np.random.default_rng(seed)
        simulator = Simulator(agent=agent, viewer=False, T=args.steps+1, dt=.01,
                              timeconst=.02, dampingratio=1., ctrl_rate=100,
                              model_path=PACKAGE_ROOT/task['sim_path'], robot='go1_dm',
                              initial_key=task.get('initial_key', 1), model_variant='full')
        with LegArmObserver(agent, args.ess_min, args.ess_fraction) as observer:
            simulator.run()
        observer.save(directory/f'seed_{seed}.npz')
        metadata = dict(seed=seed, task=args.task, mode=args.mode, dt=agent.h,
                        horizon=agent.horizon, n_samples=agent.n_samples,
                        iterations=agent.n_iterations, temperature=agent.temperature,
                        params=agent.params, final_base=simulator.data.qpos[:3].tolist(),
                        simulation_warnings=int(simulator.data.warning.number.sum()))
        (directory/f'seed_{seed}.json').write_text(json.dumps(metadata, indent=2))
    finally:
        agent.shutdown()


def report(args, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups, frames, diagnostics = [], [], []
    for seed in args.seeds:
        with np.load(directory/f'seed_{seed}.npz') as d:
            if args.horizon_index >= d['weighted'].shape[1]:
                raise ValueError('horizon-index exceeds prediction horizon')
            # Last iteration only for repeatability; all iterations remain in NPZ.
            last = np.r_[d['update'][1:] != d['update'][:-1], True]
            w, u = d['weighted'][last, args.horizon_index], d['uniform'][last, args.horizon_index]
            steps, low = d['update'][last], d['low_ess'][last]
            for i in range(len(w)):
                frames.append(dict(seed=seed, update=int(steps[i]), ess=float(d['ess'][last][i]),
                                   low=bool(low[i]), w=w[i].tolist(), u=u[i].tolist()))
            for idx in range(len(d['ess'])):
                diagnostics.append([seed, int(d['update'][idx]), int(d['iteration'][idx]),
                                    int(d['count'][idx]), float(d['ess'][idx]), bool(d['low_ess'][idx])])
            for segment in range(args.segments):
                mask = (steps >= segment*args.steps/args.segments) & (steps < (segment+1)*args.steps/args.segments)
                valid = mask & ~low
                coverage = valid.sum()/max(mask.sum(), 1)
                groups.append(dict(seed=seed, segment=segment, coverage=coverage,
                                   count=int(valid.sum()), w=mean(w[valid]), u=mean(u[valid])))
    with (directory/'ess.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'update', 'iteration', 'samples', 'ess', 'low_ess'])
        writer.writerows(diagnostics)
    gw, gu = np.array([g['w'] for g in groups]), np.array([g['u'] for g in groups])
    mw, mu = mean(gw), mean(gu)
    sign = np.divide(np.sum((np.sign(gw) == np.sign(mw)) & np.isfinite(gw), axis=0),
                     np.sum(np.isfinite(gw), axis=0), out=np.full((12, 6), np.nan),
                     where=np.sum(np.isfinite(gw), axis=0) > 0)
    # Resample entire independent seed runs, never adjacent updates/horizons.
    sw = np.array([mean([g['w'] for g in groups if g['seed'] == seed]) for seed in args.seeds])
    su = np.array([mean([g['u'] for g in groups if g['seed'] == seed]) for seed in args.seeds])
    rng = np.random.default_rng(9173)  # Analysis-only RNG, independent of MPPI.
    draws = rng.integers(0, len(args.seeds), size=(2000, len(args.seeds)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ci_w = np.nanpercentile(mean(sw[draws], axis=1), [2.5, 97.5], axis=0)
        ci_delta = np.nanpercentile(mean((sw-su)[draws], axis=1), [2.5, 97.5], axis=0)
    if len(args.seeds) < 2:
        ci_w[:] = np.nan
        ci_delta[:] = np.nan
    # Descriptive screening only, not a multiple-comparison significance test.
    coverage_ok = all(g['coverage'] >= .8 and g['count'] >= 20 for g in groups)
    warning_count = sum(json.loads((directory/f'seed_{seed}.json').read_text()).get('simulation_warnings', 0)
                        for seed in args.seeds if (directory/f'seed_{seed}.json').exists())
    enough = len(args.seeds) >= 5 and args.segments >= 3 and coverage_ok and warning_count == 0
    candidate = (np.abs(mw) >= .15) & (np.abs(mw-mu) >= .1) & (sign >= .8) & enough
    with (directory/'repeatability.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['leg', 'arm', 'weighted_mean', 'uniform_mean', 'difference',
                         'group_sign_agreement', 'group_weighted_std', 'weighted_ci_low', 'weighted_ci_high', 'difference_ci_low', 'difference_ci_high', 'candidate'])
        for i in range(12):
            for j in range(6):
                values = gw[:, i, j]
                writer.writerow([LEGS[i], ARMS[j], mw[i,j], mu[i,j], mw[i,j]-mu[i,j],
                                 sign[i,j], float(np.std(values[np.isfinite(values)])) if np.isfinite(values).any() else np.nan,
                                 ci_w[0,i,j], ci_w[1,i,j], ci_delta[0,i,j], ci_delta[1,i,j], bool(candidate[i,j])])
    with (directory/'segments.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'segment', 'valid_updates', 'coverage', 'leg', 'arm', 'weighted', 'uniform'])
        for g in groups:
            for i in range(12):
                for j in range(6):
                    writer.writerow([g['seed'], g['segment'], g['count'], g['coverage'], LEGS[i], ARMS[j], g['w'][i,j], g['u'][i,j]])
    fig, axes = plt.subplots(1, 4, figsize=(16, 6), constrained_layout=True)
    cmap = plt.get_cmap('coolwarm').copy()
    cmap.set_bad('#aaaaaa')
    for ax, matrix, title in zip(axes, [mw, mu, mw-mu, sign], ['Weighted', 'Uniform', 'Weighted - Uniform', 'Sign agreement']):
        limit = 2 if title == 'Weighted - Uniform' else 1
        im = ax.imshow(matrix, vmin=0 if title == 'Sign agreement' else -limit, vmax=limit, cmap=cmap)
        ax.set_xticks(range(6), ARMS, rotation=60)
        ax.set_yticks(range(12), LEGS)
        ax.set_title(title)
        for i in range(12):
            for j in range(6):
                value = matrix[i, j]
                ax.text(j, i, f'{value:.3f}' if np.isfinite(value) else 'N/A',
                        ha='center', va='center', fontsize=6)
        fig.colorbar(im, ax=ax, shrink=.65)
    quality = 'coverage sufficient for exploratory screen' if enough else 'INSUFFICIENT reliable coverage'
    fig.suptitle(f'{args.task} | h={args.horizon_index} | low ESS excluded | {quality}')
    fig.savefig(directory/'summary.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for seed in args.seeds:
        rows = np.array([r[1:5] for r in diagnostics if r[0] == seed])
        ax.plot(rows[:, 0], rows[:, 3], label=f'seed {seed}', alpha=.6)
    threshold = max(args.ess_min, args.ess_fraction * diagnostics[-1][3])
    ax.axhline(threshold, color='red', linestyle='--', label='ESS threshold (last batch)')
    ax.set(xlabel='MPPI update', ylabel='Weighted ESS')
    ax.legend()
    fig.savefig(directory/'ess.png', dpi=150)
    plt.close(fig)
    # Standalone per-update heatmaps, no server or extra dependencies.
    payload = json.dumps(frames).replace('NaN', 'null')
    html = '''<!doctype html><meta charset="utf-8"><title>Leg-arm correlation</title>
<style>body{font:16px sans-serif;margin:30px} canvas{max-width:100%} input{width:80%}</style>
<h2>Leg-arm correlations: final iteration, horizon HINDEX</h2>
<p>Blue: -1, white: 0, red: +1; gray: undefined. Difference uses range [-2,2].</p>
<input id="slider" type="range" min="0" value="0"><p id="status"></p><canvas id="plot" width="1200" height="600"></canvas>
<script>const frames=PAYLOAD,legs=LEGS;const s=document.getElementById('slider'),c=document.getElementById('plot').getContext('2d');s.max=frames.length-1;
function draw(){const f=frames[+s.value];document.getElementById('status').textContent=`seed ${f.seed}, update ${f.update}, ESS ${f.ess.toFixed(1)} ${f.low?'— LOW ESS: unstable estimate':''}`;
c.clearRect(0,0,1200,600);c.font='13px sans-serif';['Weighted','Uniform','Weighted − Uniform'].forEach((title,p)=>{c.fillStyle='black';c.fillText(title,125+p*400,25);for(let j=0;j<6;j++)c.fillText('arm '+(j+1),100+p*400+j*46,50);for(let i=0;i<12;i++){c.fillStyle='black';c.fillText(legs[i],p*400,78+i*43);for(let j=0;j<6;j++){let v=p===0?f.w[i][j]:p===1?f.u[i][j]:(f.w[i][j]===null||f.u[i][j]===null?null:f.w[i][j]-f.u[i][j]);let a=Math.min(1,Math.abs(v)/(p===2?2:1)),q=Math.round(255*(1-a));c.fillStyle=v===null?'#aaa':v>=0?`rgb(255,${q},${q})`:`rgb(${q},${q},255)`;c.fillRect(100+p*400+j*46,60+i*43,45,42);c.fillStyle='black';c.fillText(v===null?'N/A':v.toFixed(2),103+p*400+j*46,85+i*43);}}});}s.oninput=draw;draw();</script>'''
    html = html.replace('HINDEX', str(args.horizon_index)).replace('PAYLOAD', payload).replace('LEGS', json.dumps(LEGS))
    (directory/'updates.html').write_text(html)
    low_fraction = float(np.mean([r[-1] for r in diagnostics]))
    final_low_fraction = float(np.mean([frame['low'] for frame in frames]))
    verdict = ('Insufficient reliable coverage; do not infer repeatability.' if not enough else
               'Some pairs merit a controlled cross-covariance sampling experiment.' if candidate.any() else
               'No pairs pass the descriptive repeatability screen; cross-covariance sampling is not justified by this run.')
    text = f'''# Leg-arm correlation analysis

Task: {args.task}; mode: {args.mode}; seeds: {args.seeds}; steps/seed: {args.steps}; segments/seed: {args.segments}; horizon index: {args.horizon_index}.

Simulation warnings: {warning_count}. Low ESS fraction (all iterations): {low_fraction:.1%}. ESS flag: ESS < max({args.ess_min}, {args.ess_fraction} × batch samples).

Final-iteration low ESS fraction: {final_low_fraction:.1%}. Reliable coverage across seed/segment groups: {min(g['coverage'] for g in groups):.1%} to {max(g['coverage'] for g in groups):.1%}.

Candidate pairs: {int(candidate.sum())}. {verdict}

Maximum absolute group-mean weighted correlation: {float(np.nanmax(np.abs(mw))) if np.isfinite(mw).any() else 'undefined'}. Maximum absolute paired weighted-minus-uniform mean: {float(np.nanmax(np.abs(mw-mu))) if np.isfinite(mw-mu).any() else 'undefined'}.

Screen: at least 5 seeds, 3 segments, >=80% reliable updates and >=20 reliable updates in every seed/segment; |mean weighted r| >=0.15, |mean weighted r - mean uniform r| >=0.10, and same sign in >=80% of seed/segment means. These are exploratory thresholds, not significance tests. CSV confidence intervals use 2000 whole-seed bootstrap resamples (pointwise 95%, no multiplicity correction; few seeds limit precision). Seeds and segments are equally weighted. Low-ESS updates are excluded from repeatability, retained in raw data and viewer. Fixed/zero-variance axes are NaN.

This measures post-clipping action proposals conditioned on cost, not causal coupling. No horizon pooling. All horizons/iterations and weighted cross-covariance are stored in seed NPZ files; viewer and summary use only the final iteration at the specified horizon. Low ESS flags uncertainty in correlation estimation; it does not by itself imply poor control performance. Constant or low-ESS samples cannot support reliable estimates. Check simulation warnings and final state in seed JSON files. Separate phase/command-conditioned analyses and longer independent runs are needed before generalizing; cancellation across gait phases can conceal coupling. Overlapping horizons/adjacent updates are not independent observations. No p-values are claimed across the 72 pairs. Improvement in task performance requires a separate controlled intervention; this analysis does not modify sampling.
'''
    (directory/'report.md').write_text(text)
    print(text, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['ee_reach', 'ee_walk', 'ee_reach_cmd_vel', 'ee_walk_cmd_vel'], default='ee_walk')
    p.add_argument('--mode', choices=['mppi', 'configured', 'dial-mpc'], default='configured')
    p.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33, 44, 55])
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--segments', type=int, default=3)
    p.add_argument('--horizon-index', type=int, default=0)
    p.add_argument('--ess-min', type=float, default=20.)
    p.add_argument('--ess-fraction', type=float, default=.2)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    if args.steps < 1 or not 1 <= args.segments <= args.steps or args.horizon_index < 0 or len(set(args.seeds)) != len(args.seeds):
        p.error('Invalid steps, segments, horizon index, or duplicate seeds')
    if not args.report_only and args.output.exists() and any(args.output.iterdir()):
        p.error('Use a new output directory to avoid overwriting an experiment')
    if args.report_only:
        original = json.loads((args.output/'experiment.json').read_text())
        for key in ('task', 'mode', 'seeds', 'steps', 'ess_min', 'ess_fraction'):
            setattr(args, key, original[key])
        if args.segments > args.steps:
            p.error('segments exceeds saved experiment steps')
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        (args.output/'experiment.json').write_text(json.dumps(vars(args), default=str, indent=2))
        for seed in args.seeds:
            print(f'Running seed {seed}', flush=True)
            run_seed(args, seed, args.output)
    report(args, args.output)


if __name__ == '__main__':
    main()
