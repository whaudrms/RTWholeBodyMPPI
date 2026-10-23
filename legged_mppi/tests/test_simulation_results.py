"""Report alignment, early-stop trimming and playback without re-solving."""
import csv
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from whole_body_mppi.control.tasks.locomotion import LocomotionController
from whole_body_mppi.control.tasks.ee_velocity import EEVelocityController
from whole_body_mppi.control.tasks.base import PACKAGE_ROOT
from whole_body_mppi.interface.simulator import Simulator


class SimulationResultTests(unittest.TestCase):
    def make_simulator(self, ee=False):
        agent = EEVelocityController() if ee else LocomotionController('stand')
        self.addCleanup(agent.shutdown)
        agent.set_params(5, .1, 6)
        return Simulator(agent=agent, robot='go1_dm' if ee else 'go1', initial_key='home' if ee else 1,
                         viewer=False, T=3, model_path=PACKAGE_ROOT/agent.task_data['sim_path'])

    def test_report_contains_aligned_states_costs_timing_and_ee_targets(self):
        sim = self.make_simulator(ee=True)
        sim.run()
        with tempfile.TemporaryDirectory() as root:
            folder = sim.save_results('mpopi', root)
            self.assertEqual(folder.parent, Path(root)/'mpopi'/'ee_reach_cmd_vel')
            for name in ('summary.json', 'config.json', 'trajectory.npz', 'metrics.csv', 'metrics.png', 'ee_tracking.png'):
                self.assertTrue((folder/name).is_file())
            data = np.load(folder/'trajectory.npz')
            self.assertEqual(data['qpos'].shape, (25, 3))
            np.testing.assert_allclose(data['time'], [0, .01, .02])
            np.testing.assert_allclose(data['qpos'][:, -1], sim.data.qpos)
            self.assertEqual(np.isfinite(data['solve_seconds']).sum(), 2)
            self.assertTrue(np.all(np.isfinite(data['cost'])))
            self.assertNotIn('ee_targets', data.files)
            self.assertNotIn('ee_errors', data.files)
            self.assertIn('ee_velocity_commands', data.files)
            self.assertIn('command_phase_indices', data.files)
            with (folder/'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertAlmostEqual(float(rows[0]['predicted_horizon_cost']), sim.cost[0, 0])
            summary = json.loads((folder/'summary.json').read_text())
            self.assertAlmostEqual(summary['solve_total_seconds'], np.nansum(sim.solve_seconds))
            self.assertEqual(summary['ee_velocity_frame'], 'world')
            self.assertNotIn('ee_position_error_m', summary)
            self.assertNotIn('body_position_error_m', summary)
            self.assertTrue((folder/'metrics.png').stat().st_size > 1000)

    def test_early_viewer_close_does_not_export_unrecorded_tail(self):
        sim = self.make_simulator()
        sim.T = 3
        sim.viewer = types.SimpleNamespace(is_alive=False, close=lambda: None)
        sim.run()
        self.assertEqual(sim.recorded_steps, 2)
        with tempfile.TemporaryDirectory() as root:
            folder = sim.save_results('mppi', root)
            self.assertEqual(np.load(folder/'trajectory.npz')['time'].shape, (2,))
            self.assertFalse((folder/'ee_tracking.png').exists())

    def test_unavailable_display_raises_before_native_viewer_creation(self):
        sim = self.make_simulator()
        sim.recorded_steps = 1
        with patch('glfw.init', return_value=False), patch.object(sim, 'initialize_rendering') as render:
            with self.assertRaisesRegex(RuntimeError, 'Cannot open playback display'):
                sim.playback()
            render.assert_not_called()

    def test_playback_uses_states_without_solving_and_restores_final_state(self):
        sim = self.make_simulator(ee=True)
        sim.run()
        original = sim.get_state().copy()
        original_time = sim.data.time
        viewer = types.SimpleNamespace(is_alive=True, render=lambda: None, close=lambda: None,
                                       add_marker=lambda **kwargs: None)
        module = types.SimpleNamespace(MujocoViewer=lambda *args, **kwargs: viewer)
        with patch.dict('sys.modules', {'mujoco_viewer': module}), \
                patch('glfw.init', return_value=True), \
                patch.object(sim, 'initialize_rendering'), \
                patch.object(sim.agent, 'update', side_effect=AssertionError('Playback must not solve')), \
                patch('whole_body_mppi.interface.simulator.mujoco.mj_step', side_effect=AssertionError('Playback must not step')), \
                patch('whole_body_mppi.interface.simulator.wall_time.perf_counter', side_effect=[0., .03]):
            sim.playback()
        np.testing.assert_array_equal(sim.get_state(), original)
        self.assertEqual(sim.data.time, original_time)
        self.assertIsNone(sim.viewer)
