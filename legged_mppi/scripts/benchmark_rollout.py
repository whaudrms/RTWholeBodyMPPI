"""Deterministic rollout/cost/solver microbenchmark; no rendering or config edits."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whole_body_mppi.control.tasks.ee_velocity import EEVelocityController
from whole_body_mppi.control.solvers.options import apply_mppi_mode


def measure(function, repetitions):
    for _ in range(3):
        function()
    times = []
    for _ in range(repetitions):
        start = perf_counter()
        function()
        times.append(perf_counter()-start)
    return dict(median_ms=float(np.median(times)*1000), p95_ms=float(np.percentile(times, 95)*1000))


def main(output, samples=128, repeats=20, reference=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    agent = EEVelocityController()
    try:
        agent.set_params(40, .1, samples)
        apply_mppi_mode(agent, 'mpopi')
        data = mujoco.MjData(agent.model)
        mujoco.mj_resetDataKeyframe(agent.model, data, agent.model.key('home').id)
        obs = np.r_[data.qpos, data.qvel]
        initial = np.tile(np.r_[0., obs], (samples, 1))
        rng = np.random.default_rng(123)
        actions = np.clip(agent.sampling_init + rng.normal(0, .03, (samples, 40, 18)), agent.act_min, agent.act_max)
        states = np.empty((samples, 40, len(initial[0])))
        agent.joints_ref = agent.gait_scheduler.gait[:, agent.gait_scheduler.indices[:40]]
        def rollout():
            agent.rollout_func(states, actions, initial, num_workers=agent.num_workers, nstep=40)
        rollout()
        costs = agent.task_cost(states, actions)
        result = dict(mujoco=mujoco.__version__, samples=samples, horizon=40, workers=agent.num_workers,
                      iterations=agent.n_iterations, repeats=repeats,
                      physics=measure(rollout, repeats), cost=measure(lambda: agent.task_cost(states, actions), repeats))
        # Profile complete solves using exactly the same observation/seed/warm start.
        actions_out, solve_times, physics_times, cost_times = [], [], [], []
        roll, taskcost = agent.rollout_func, agent.task_cost
        clock = dict(physics=0., cost=0.)
        def timed_roll(*args, **kwargs):
            start = perf_counter()
            value = roll(*args, **kwargs)
            clock['physics'] += perf_counter()-start
            return value
        def timed_cost(*args, **kwargs):
            start = perf_counter()
            value = taskcost(*args, **kwargs)
            clock['cost'] += perf_counter()-start
            return value
        agent.rollout_func, agent.task_cost = timed_roll, timed_cost
        for _ in range(repeats):
            clock.update(physics=0., cost=0.)
            start = perf_counter()
            actions_out.append(agent.update(obs).copy())
            solve_times.append(perf_counter()-start)
            physics_times.append(clock['physics'])
            cost_times.append(clock['cost'])
        for label, times in (('solve', solve_times), ('solve_physics', physics_times), ('solve_cost', cost_times)):
            result[label] = dict(median_ms=float(np.median(times)*1000), p95_ms=float(np.percentile(times, 95)*1000))
        arrays = dict(states=states.copy(), costs=costs, selected_actions=np.array(actions_out))
        np.savez_compressed(output/'reference.npz', **arrays)
        if reference:
            old = np.load(Path(reference)/'reference.npz')
            result['max_absolute_difference'] = {name: float(np.max(np.abs(array-old[name]))) for name, array in arrays.items()}
            prior = json.loads((Path(reference)/'benchmark.json').read_text())
            result['speedup'] = {key: prior[key]['median_ms']/result[key]['median_ms'] for key in ('physics', 'cost', 'solve')}
        (output/'benchmark.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
    finally:
        agent.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--reference')
    args = parser.parse_args()
    main(args.output, args.samples, args.repeats, args.reference)
