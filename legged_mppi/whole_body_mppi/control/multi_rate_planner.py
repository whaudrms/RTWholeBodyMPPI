"""Non-blocking bridge between a fast control loop and a slower MPPI planner."""

from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class PlanResult:
    trajectory: object
    origin_tick: int
    update_seconds: float


class MultiRatePlanner:
    """Run MPPI updates on one worker without blocking command publication.

    Planner requests are single-slot: if MPPI misses its period, an older
    queued state is replaced by the newest state instead of accumulating lag.
    Commands are indexed from the tick at which the planning state was taken,
    which compensates for the time spent computing the plan.
    """

    def __init__(
        self,
        controller,
        control_rate_hz=100,
        planner_rate_hz=25,
        transition_callback=None,
    ):
        if control_rate_hz <= 0 or planner_rate_hz <= 0:
            raise ValueError("control and planner rates must be positive")
        ratio = float(control_rate_hz) / float(planner_rate_hz)
        self.plan_stride = int(round(ratio))
        if self.plan_stride < 1 or abs(ratio - self.plan_stride) > 1e-9:
            raise ValueError("control_rate_hz must be an integer multiple of planner_rate_hz")

        self.controller = controller
        self.transition_callback = transition_callback
        self._condition = threading.Condition()
        self._pending = None
        self._latest = None
        self._error = None
        self._last_planned_tick = None
        self._stopping = False
        self._worker = threading.Thread(
            target=self._run,
            name="mppi-planner",
            daemon=True,
        )
        self._worker.start()

    def request(self, state, control_tick):
        """Queue the newest state when this tick is on a planner boundary."""
        if control_tick % self.plan_stride:
            return False
        with self._condition:
            self._raise_if_failed()
            self._pending = (state.copy(), int(control_tick))
            self._condition.notify()
        return True

    def command(self, control_tick):
        """Return the latency-aligned action from the newest completed plan."""
        with self._condition:
            self._raise_if_failed()
            result = self._latest

        if result is None:
            return self.controller.sampling_init.copy()

        index = max(0, int(control_tick) - result.origin_tick)
        index = min(index, len(result.trajectory) - 1)
        return result.trajectory[index].copy()

    def latest_result(self):
        with self._condition:
            self._raise_if_failed()
            return self._latest

    def shutdown(self):
        with self._condition:
            self._stopping = True
            self._condition.notify()
        self._worker.join()

    def _raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError("MPPI planner worker failed") from self._error

    def _run(self):
        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._stopping:
                        self._condition.wait()
                    if self._stopping:
                        return
                    state, origin_tick = self._pending
                    self._pending = None

                if self._last_planned_tick is None:
                    advance_steps = 0
                else:
                    advance_steps = max(
                        0, origin_tick - self._last_planned_tick
                    )
                self._last_planned_tick = origin_tick

                if self.transition_callback is not None:
                    self.transition_callback(
                        self.controller,
                        state,
                        max(1, advance_steps),
                    )

                start = time.monotonic()
                self.controller.update(
                    state,
                    advance_steps=advance_steps,
                )
                update_seconds = time.monotonic() - start
                result = PlanResult(
                    trajectory=self.controller.selected_trajectory.copy(),
                    origin_tick=origin_tick,
                    update_seconds=update_seconds,
                )
                with self._condition:
                    self._latest = result
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
