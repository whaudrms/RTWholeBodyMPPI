"""Measure synchronous locomani solve latency and scenario progress without a GUI."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from mani_mppi.control.controllers.mppi_locomani import MPPI
from mani_mppi.interface.simulator import Simulator
from mani_mppi.utils.tasks import get_task


def latency_summary(records, deadline_ms):
    if not records:
        return {"count": 0}
    values = np.array([row["update_ms"] for row in records])
    return {
        "count": len(records),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
        "deadline_misses": int(np.count_nonzero(values > deadline_ms)),
        "deadline_miss_fraction": float(np.mean(values > deadline_ms)),
        "phase_mean_ms": {
            key: float(np.mean([row["phase_ms"][key] for row in records]))
            for key in records[0]["phase_ms"]
        },
        "iteration_counts": {
            str(count): sum(row["iterations"] == count for row in records)
            for count in sorted({row["iterations"] for row in records})
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=10,
                        help="Initial updates excluded only from steady-state statistics.")
    parser.add_argument("--performance-profile", choices=("rt20",), default=None)
    parser.add_argument("--rollout-mode", choices=("hermite", "original_spline"), default="hermite")
    parser.add_argument("--anneal-iterations", type=int, default=None)
    parser.add_argument("--deadline-ms", type=float, default=20.0,
                        help="Reporting threshold; does not change the controller budget.")
    parser.add_argument("--output", type=Path, default=Path("locomani_latency.json"))
    args = parser.parse_args()
    if args.steps < 1 or not 0 <= args.warmup < args.steps:
        parser.error("Require steps > warmup >= 0")
    if not np.isfinite(args.deadline_ms) or args.deadline_ms <= 0:
        parser.error("deadline-ms must be finite and positive")

    records = []
    failure = None
    # Keep stdout machine-readable; retain planner diagnostics on stderr.
    with contextlib.redirect_stdout(sys.stderr):
        started = time.perf_counter()
        agent = MPPI(rollout_mode=args.rollout_mode,
                     anneal_iterations=args.anneal_iterations,
                     performance_profile=args.performance_profile)
        try:
            task = get_task("locomani")
            dt = float(agent.model.opt.timestep)
            sim = Simulator(agent=agent, viewer=False, T=args.steps, dt=dt,
                            ctrl_rate=1.0 / dt, plot_enabled=False,
                            model_path=str(PACKAGE_ROOT / "mani_mppi" / task["sim_path"]))
            initialization_ms = (time.perf_counter() - started) * 1000
            for step in range(args.steps):
                observation = np.concatenate((sim.data.qpos, sim.data.qvel))
                try:
                    started = time.perf_counter()
                    control = agent.update(observation)
                    update_ms = (time.perf_counter() - started) * 1000
                    if not np.isfinite(control).all():
                        raise RuntimeError("Non-finite control")
                    sim.step(control)
                    if not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all():
                        raise RuntimeError("Non-finite simulator state")
                    observation = np.concatenate((sim.data.qpos, sim.data.qvel))
                    # Update site poses to the post-step state for error reporting.
                    mujoco.mj_kinematics(sim.model, sim.data)
                    goal = int(agent.goal_index)
                    error = float(np.linalg.norm(
                        sim.data.site_xpos[sim.ee_site_id] - agent.ee_goal_pos[goal]))
                    rotation = np.empty(9)
                    mujoco.mju_quat2Mat(rotation, sim.data.qpos[3:7])
                    tilt = float(np.degrees(np.arccos(np.clip(rotation[8], -1, 1))))
                    records.append({
                        "step": step, "sim_time_s": float(sim.data.time),
                        "update_ms": update_ms,
                        "phase_ms": {key: value * 1000 for key, value in agent.last_timing.items()},
                        "iterations": len(getattr(agent, "iteration_diagnostics", [])),
                        "budget_limited": bool(getattr(agent, "budget_limited", False)),
                        "selection_source": agent.selection_source,
                        "base_height_m": float(sim.data.qpos[2]),
                        "base_tilt_deg": tilt, "goal_index": goal,
                        "ee_position_error_m": error,
                    })
                    if agent.goal_reached(observation):
                        agent.next_goal()
                    if (step + 1) % 100 == 0:
                        print(f"step={step + 1}/{args.steps}, goal={agent.goal_index}, "
                              f"last_update={update_ms:.2f} ms", file=sys.stderr)
                except Exception as error:
                    failure = {"step": step, "type": type(error).__name__, "message": str(error)}
                    break
            settings = {
                "rollout_mode": agent.rollout_mode,
                "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                "performance_profile": args.performance_profile,
                "n_samples": agent.n_samples, "workers": agent.num_workers,
                "chunk_size": agent.rollout_chunk_size,
                "horizon": agent.horizon, "dt_s": dt,
                "max_iterations": agent.anneal_iterations,
                "planning_budget_ms": agent.planning_budget_ms,
                "validate_mean": agent.validate_mean,
                "collision_broadphase": bool(agent.params.get("collision_broadphase", False)),
                "seed": agent.params.get("seed"), "mujoco_version": mujoco.__version__,
            }
            scenario = {
                "requested_steps": args.steps, "completed_steps": len(records),
                "goal_index": int(agent.goal_index), "goals": len(agent.ee_goal_pos),
                "task_success": bool(agent.task_success), "failure": failure,
                "min_base_height_m": min((row["base_height_m"] for row in records), default=None),
                "max_base_tilt_deg": max((row["base_tilt_deg"] for row in records), default=None),
                "final_ee_position_error_m": records[-1]["ee_position_error_m"] if records else None,
            }
        finally:
            agent.close()
    report = {
        "settings": settings, "initialization_ms": initialization_ms,
        "deadline_ms": args.deadline_ms, "warmup_updates": args.warmup,
        "all_updates": latency_summary(records, args.deadline_ms),
        "steady_state": latency_summary(records[args.warmup:], args.deadline_ms),
        "scenario": scenario, "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
