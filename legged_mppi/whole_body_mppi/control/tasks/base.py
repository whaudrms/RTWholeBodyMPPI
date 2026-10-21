"""Go1 task integration: model configuration and shared plan evaluation."""
from pathlib import Path

import mujoco
import numpy as np
import yaml

from whole_body_mppi.control.solvers.base import MPPISolver

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONTROL_ROOT = Path(__file__).resolve().parents[1]
GAIT_DIR = CONTROL_ROOT / 'gait_scheduler/gaits'


class BaseTaskController(MPPISolver):
    def __init__(self, model_path, config_path, *, model=None):
        with open(config_path, encoding='utf-8') as stream:
            self.params = yaml.safe_load(stream)
        model = model if model is not None else mujoco.MjModel.from_xml_path(str(model_path))
        model.opt.timestep = self.params['dt']
        model.opt.enableflags = 1
        model.opt.o_solref = np.array(self.params['o_solref'])
        super().__init__(
            model, self.params,
            initial_action=[-0.3, 1.34, -2.83, 0.3, 1.34, -2.83] * 2,
            action_min=[-0.863, -0.686, -2.818] * 4,
            action_max=[0.863, 4.501, -0.888] * 4)
        self.cost_func = self.calculate_total_cost

    def task_cost(self, states, actions):
        """Interpret rollout state layout and evaluate the task objective."""
        raise NotImplementedError

    def solve_task(self, obs):
        actions = self.optimize_actions(
            obs, lambda actions: self.task_cost(self.state_rollouts, actions))
        self.gait_scheduler.roll()
        return self.commit_plan(actions)

    def eval_best_trajectory(self):
        if self.obs is None:
            return None
        states = np.zeros((1, self.horizon,
                           mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)))
        actions = self.selected_trajectory[None]
        initial_state = np.array([np.concatenate([[0], self.obs])])
        self.rollout_func(states, actions, initial_state,
                          num_workers=self.num_workers, nstep=self.horizon)
        return self.task_cost(states, actions)[0]
