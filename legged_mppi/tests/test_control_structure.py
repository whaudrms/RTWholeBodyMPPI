"""Exercise task-independent solver use and the refactored integration paths."""
import unittest

import mujoco
import numpy as np

from whole_body_mppi.control.solvers.base import MPPISolver
from whole_body_mppi.control.solvers.options import apply_mppi_mode
from whole_body_mppi.control.tasks.base import CONTROL_ROOT, PACKAGE_ROOT
from whole_body_mppi.control.tasks.definitions import TASKS
from whole_body_mppi.control.tasks.locomotion import LocomotionController
from whole_body_mppi.control.tasks.locomanipulation import LocomanipulationController


class ControlStructureTests(unittest.TestCase):
    def test_solver_without_go1_task_or_cost(self):
        model = mujoco.MjModel.from_xml_string('''
            <mujoco><worldbody><body><joint name="slide" type="slide"/>
            <geom type="sphere" size="0.1" mass="1"/></body></worldbody>
            <actuator><motor joint="slide"/></actuator></mujoco>''')
        params = dict(dt=0.01, horizon=5, n_samples=6, n_iterations=2,
                      noise_sigma=[0.1], n_workers=2, sample_type='normal',
                      n_knots=3, seed=42, **{'lambda': 0.1})
        for mode in ('mppi', 'dial-mpc', 'cma-mppi', 'mpopi', 'mbd'):
            with self.subTest(mode=mode):
                solver = MPPISolver(model, params, initial_action=[0.],
                                    action_min=[-1.], action_max=[1.])
                try:
                    apply_mppi_mode(solver, mode)
                    obs = np.zeros(model.nq + model.nv)
                    def cost(actions):
                        self.assertTrue(np.isfinite(solver.state_rollouts).all())
                        return np.sum((actions - 0.2)**2, axis=(1, 2))
                    plan = solver.optimize_actions(obs, cost)
                    self.assertEqual(plan.shape, (5, 1))
                    self.assertTrue(np.isfinite(plan).all())
                    action = solver.commit_plan(plan)
                    np.testing.assert_array_equal(action, plan[0])
                    solver.set_params(6, 0.2, 8)
                    self.assertEqual(solver.state_rollouts.shape[:2], (8, 6))
                    self.assertIsNone(solver.cma_covariance)
                    self.assertIsNone(solver.mbd_forward_trace)
                finally:
                    solver.shutdown()

    def test_registry_resources_resolve(self):
        for task, params in TASKS.items():
            with self.subTest(task=task):
                self.assertTrue((CONTROL_ROOT / params['config_path']).is_file())
                self.assertTrue((PACKAGE_ROOT / params['model_path']).is_file())
                self.assertTrue((PACKAGE_ROOT / params['sim_path']).is_file())

    def test_legacy_imports(self):
        from whole_body_mppi.control.controllers.mppi_locomotion import MPPI
        from whole_body_mppi.control.controllers.mppi_locomanipulation import MPPI_box_push
        from whole_body_mppi.utils.tasks import TASKS as legacy_tasks
        self.assertIs(MPPI, LocomotionController)
        self.assertIs(MPPI_box_push, LocomanipulationController)
        self.assertIs(legacy_tasks, TASKS)

    def test_selected_plan_cost_for_both_state_layouts(self):
        for cls, task in ((LocomotionController, 'stand'),
                          (LocomanipulationController, 'push_box')):
            with self.subTest(task=task):
                agent = cls(task)
                try:
                    self.assertIsNone(agent.eval_best_trajectory())
                    data = mujoco.MjData(agent.model)
                    mujoco.mj_resetDataKeyframe(agent.model, data, 1)
                    agent.update(np.r_[data.qpos, data.qvel])
                    self.assertTrue(np.isfinite(agent.eval_best_trajectory()))
                finally:
                    agent.shutdown()


if __name__ == '__main__':
    unittest.main()
