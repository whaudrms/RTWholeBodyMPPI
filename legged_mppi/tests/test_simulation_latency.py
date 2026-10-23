"""Measured latency, stale-plan indexing, warm start and real-time pacing."""
import itertools
import unittest
from unittest.mock import patch
import numpy as np

import test_simulation_results
from whole_body_mppi.interface.simulator import Simulator


class SimulationLatencyTests(unittest.TestCase):
    def make_simulator(self):
        helper = test_simulation_results.SimulationResultTests()
        sim = helper.make_simulator()
        self.addCleanup(sim.agent.shutdown)
        sim.T = 9
        # Rebuild arrays for the longer experiment.
        return Simulator(agent=sim.agent, viewer=False, T=9,
                         model_path=sim.model_path, sync=True)

    def test_delayed_plan_runs_old_commands_and_skips_elapsed_prefix(self):
        sim = self.make_simulator()
        origin = sim.data.qpos[7:].copy()
        plan = np.tile(origin, (5, 1))
        plan[:, 0] += np.arange(5)*.01
        observations = []
        warm_starts = []
        def update(obs):
            observations.append(obs.copy())
            warm_starts.append(sim.agent.trajectory.copy())
            sim.agent.selected_trajectory = plan.copy()
            sim.agent.commit_plan(plan.copy())
            return plan[0].copy()
        ticks = itertools.count()
        # Each measured solve/evaluation takes 12.5ms => arrival after 30ms.
        with patch.object(sim.agent, 'update', side_effect=update), \
             patch.object(sim.agent, 'eval_best_trajectory', return_value=123.), \
             patch('whole_body_mppi.interface.simulator.wall_time.perf_counter', side_effect=lambda: next(ticks)*.0125):
            sim.run()
        np.testing.assert_allclose(sim.ctrl[:, :3], np.tile(origin[:, None], (1, 3)))
        np.testing.assert_allclose(sim.ctrl[:, 3], plan[3])
        np.testing.assert_allclose(sim.ctrl[:, 4], plan[4])
        np.testing.assert_allclose(sim.ctrl[:, 5], plan[4])
        self.assertTrue(sim.plan_exhausted[5])
        self.assertEqual(len(observations), 3)
        self.assertFalse(np.allclose(observations[0], observations[1]))
        np.testing.assert_allclose(sim.plan_application_delay_seconds[[3, 6]], [.03, .03])
        np.testing.assert_allclose(warm_starts[1], plan[[3, 4, 4, 4, 4]])
        self.assertTrue(np.isnan(sim.cost[0, 0]))
        self.assertEqual(sim.cost[0, 3], 123.)

    def test_sync_waits_for_simulation_deadlines(self):
        sim = self.make_simulator()
        sim.agent.selected_trajectory = np.tile(sim.data.qpos[7:], (sim.agent.horizon, 1))
        with patch.object(sim.agent, 'update', return_value=sim.data.ctrl.copy()), \
             patch.object(sim.agent, 'eval_best_trajectory', return_value=0.), \
             patch('whole_body_mppi.interface.simulator.wall_time.perf_counter', return_value=0.), \
             patch('whole_body_mppi.interface.simulator.wall_time.sleep') as sleep:
            sim.run()
        self.assertEqual(sleep.call_count, 8)
        self.assertAlmostEqual(sleep.call_args_list[0].args[0], .01)
        self.assertAlmostEqual(sleep.call_args_list[-1].args[0], .08)

    def test_plan_command_holds_tail(self):
        command, exhausted = Simulator.plan_command(np.array([[1.], [2.]]), .05, .01)
        np.testing.assert_equal(command, [2.])
        self.assertTrue(exhausted)

    def test_velocity_commands_advance_by_physics_time_in_both_modes(self):
        from whole_body_mppi.control.tasks.ee_velocity import EEVelocityController
        from whole_body_mppi.control.tasks.base import PACKAGE_ROOT
        for sync in (False, True):
            with self.subTest(sync=sync):
                agent = EEVelocityController('ee_walk_cmd_vel')
                self.addCleanup(agent.shutdown)
                agent.set_params(5, .1, 6)
                agent.phase_ends = np.array([.01, .02, .03])
                sim = Simulator(agent=agent, viewer=False, T=9, robot='go1_dm', initial_key='home',
                                model_path=PACKAGE_ROOT/agent.task_data['sim_path'], sync=sync)
                def update(obs):
                    return agent.commit_plan(np.tile(agent.sampling_init, (agent.horizon, 1)))
                ticks = itertools.count()
                with patch.object(agent, 'update', side_effect=update), \
                     patch.object(agent, 'eval_best_trajectory', return_value=0.), \
                     patch('whole_body_mppi.interface.simulator.wall_time.perf_counter', side_effect=lambda: next(ticks)*.0125), \
                     patch('whole_body_mppi.interface.simulator.wall_time.sleep'):
                    sim.run()
                np.testing.assert_array_equal(sim.command_phase_indices, [0, 1, 2, 2, 2, 2, 2, 2, 2])
                np.testing.assert_allclose(sim.body_velocity_commands[0], [0, .2, 0, 0, 0, 0, 0, 0, 0])
                np.testing.assert_allclose(sim.ee_velocity_commands[0], [0, .2, 0, 0, 0, 0, 0, 0, 0])
                self.assertIsNone(sim.ee_errors)
                self.assertIsNone(sim.ee_targets)
