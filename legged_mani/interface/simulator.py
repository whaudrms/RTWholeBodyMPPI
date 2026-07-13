"""Closed-loop MuJoCo simulator for the standalone MPPI controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from legged_mani.interface.environment import B2Z1Env


@dataclass
class SimulationResult:
    observations: np.ndarray
    controls: np.ndarray
    costs: np.ndarray


class MPPISimulator:
    def __init__(self, controller, env: B2Z1Env | None = None):
        self.controller = controller
        self.env = env or B2Z1Env(dt=controller.model.opt.timestep)
        if self.env.model.nu != self.controller.act_dim:
            raise ValueError("Environment and controller action dimensions differ")

    def run(self, steps: int = 20, keyframe: str = "sit") -> SimulationResult:
        observation = self.env.reset(keyframe)
        observations = np.empty((steps + 1, self.env.observation_dim))
        controls = np.empty((steps, self.env.model.nu))
        costs = np.empty(steps)
        observations[0] = observation
        for index in range(steps):
            action = self.controller.update(observation)
            observation = self.env.step(action)
            if (
                hasattr(self.controller, "next_goal")
                and hasattr(self.controller, "goal_thresh")
            ):
                goal_error = np.linalg.norm(
                    self.controller.body_ref[:3] - observation[:3]
                )
                if goal_error < self.controller.goal_thresh[
                    self.controller.goal_index
                ]:
                    self.controller.next_goal()
            controls[index] = action
            costs[index] = self.controller.eval_best_trajectory()
            observations[index + 1] = observation
        return SimulationResult(observations, controls, costs)
