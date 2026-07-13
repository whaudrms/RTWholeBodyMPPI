"""One-update and short closed-loop MPPI integration checks."""

import numpy as np

from legged_mani.control import B2Z1SitHoldMPPI
from legged_mani.interface import B2Z1Env, MPPISimulator


def run_check(steps: int = 5) -> None:
    env = B2Z1Env()
    initial = env.reset("sit")
    with B2Z1SitHoldMPPI() as controller:
        action = controller.update(initial)
        assert action.shape == (16,)
        assert np.array_equal(controller.reference, initial)
        assert np.isfinite(action).all()
        assert np.all(action >= env.action_low) and np.all(action <= env.action_high)
        assert controller.state_rollouts.shape == (30, 40, 46)
        assert np.isfinite(controller.last_costs).all()

        controller.reset_planner()
        result = MPPISimulator(controller, env).run(steps=steps)
        assert result.observations.shape == (steps + 1, 45)
        assert result.controls.shape == (steps, 16)
        assert np.isfinite(result.observations).all()
        assert np.isfinite(result.controls).all()
        assert result.observations[-1, 2] > 0.08
        print(
            "mppi: PASS "
            f"(action={action.shape}, rollout={controller.state_rollouts.shape}, "
            f"final_z={result.observations[-1, 2]:.4f})"
        )


if __name__ == "__main__":
    run_check()
