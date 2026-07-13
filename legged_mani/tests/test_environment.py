"""Headless checks for the B2-Z1 model and vectorized rollout interface."""

import mujoco
import numpy as np

from legged_mani.interface import B2Z1Env


def run_check() -> None:
    env = B2Z1Env()
    sit = env.model.key_ctrl[env.model.key("sit").id].copy()
    initial = env.reset()
    observation = initial
    for _ in range(100):
        observation = env.step(sit)

    controls = np.broadcast_to(sit, (4, 20, 16)).copy()
    states = env.rollout(controls, initial_observation=initial)
    rng = np.random.default_rng(0)
    sigma = np.array([0.03, 0.06, 0.06] * 4 + [0.03] * 4)
    sampled = env.rollout(
        sit + rng.normal(size=(30, 40, 16)) * sigma,
        initial_observation=initial,
    )
    floor_id = env.model.geom("floor").id
    clearance = [
        mujoco.mj_geomDistance(
            env.model, env.data, floor_id, env.model.geom(f"{foot}_foot").id,
            0.2, np.zeros(6),
        )
        for foot in ("FR", "FL", "RR", "RL")
    ]
    assert observation.shape == (45,)
    assert states.shape == (4, 20, 46)
    assert sampled.shape == (30, 40, 46)
    assert np.isfinite(states).all() and np.isfinite(sampled).all()
    assert observation[2] > 0.08 and min(clearance) >= 0.0
    print("environment: PASS (nq=23, nv=22, nu=16, observation=45)")


if __name__ == "__main__":
    run_check()
