import threading
import time
import unittest

from whole_body_mppi.control.multi_rate_planner import MultiRatePlanner


class FakeController:
    def __init__(self):
        self.sampling_init = [99]
        self.selected_trajectory = None
        self.advance_steps = []

    def update(self, state, advance_steps=1):
        self.advance_steps.append(advance_steps)
        self.selected_trajectory = [
            [state[0] + index] for index in range(10)
        ]


class BlockingController(FakeController):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.states = []

    def update(self, state, advance_steps=1):
        self.states.append(state[0])
        if len(self.states) == 1:
            self.started.set()
            if not self.release.wait(timeout=1.0):
                raise RuntimeError("test did not release planner")
        super().update(state, advance_steps)


def wait_for_origin(planner, origin_tick):
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        result = planner.latest_result()
        if result is not None and result.origin_tick == origin_tick:
            return result
        time.sleep(0.001)
    raise AssertionError("planner did not publish a result in time")


class MultiRatePlannerTest(unittest.TestCase):
    def test_plan_is_indexed_from_state_snapshot_tick(self):
        controller = FakeController()
        planner = MultiRatePlanner(controller, 100, 25)
        try:
            self.assertEqual(planner.command(0), [99])
            self.assertTrue(planner.request([10], 0))
            wait_for_origin(planner, 0)
            self.assertEqual(planner.command(3), [13])
        finally:
            planner.shutdown()

    def test_warm_start_advance_matches_elapsed_control_ticks(self):
        controller = FakeController()
        planner = MultiRatePlanner(controller, 100, 25)
        try:
            planner.request([0], 0)
            wait_for_origin(planner, 0)
            self.assertFalse(planner.request([1], 1))
            self.assertTrue(planner.request([4], 4))
            wait_for_origin(planner, 4)
            self.assertEqual(controller.advance_steps, [0, 4])
        finally:
            planner.shutdown()

    def test_rates_must_have_an_integer_stride(self):
        with self.assertRaises(ValueError):
            MultiRatePlanner(FakeController(), 100, 30)

    def test_busy_planner_keeps_only_the_newest_pending_state(self):
        controller = BlockingController()
        planner = MultiRatePlanner(controller, 100, 25)
        try:
            planner.request([0], 0)
            self.assertTrue(controller.started.wait(timeout=1.0))
            planner.request([4], 4)
            planner.request([8], 8)
            controller.release.set()
            wait_for_origin(planner, 8)
            self.assertEqual(controller.states, [0, 8])
            self.assertEqual(controller.advance_steps, [0, 8])
        finally:
            controller.release.set()
            planner.shutdown()


if __name__ == "__main__":
    unittest.main()
