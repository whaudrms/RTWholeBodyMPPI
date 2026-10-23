"""Threaded MuJoCo batch rollouts, with private worker data."""
import threading
from concurrent.futures import ThreadPoolExecutor

import mujoco
from mujoco import rollout


class MujocoRollout:
    def __init__(self, model, num_workers):
        self.model = model
        self.num_workers = num_workers
        self.thread_local = threading.local()
        self.executor = ThreadPoolExecutor(
            max_workers=num_workers, initializer=self.thread_initializer)

    def thread_initializer(self):
        """Initialize thread-local storage for MuJoCo data."""
        self.thread_local.data = mujoco.MjData(self.model)

    def shutdown(self):
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=True)

    def call_rollout(self, initial_state, ctrl, state):
        """
        Perform a rollout of the model given the initial state and control actions.

        Args:
            initial_state (np.ndarray): Initial state of the model.
            ctrl (np.ndarray): Control actions to apply during the rollout.
            state (np.ndarray): State array to store the results of the rollout.
        """
        rollout.rollout(self.model, self.thread_local.data, skip_checks=True,
                        nroll=state.shape[0], nstep=state.shape[1],
                        initial_state=initial_state, control=ctrl, state=state)

    def threaded_rollout(self, state, ctrl, initial_state, num_workers=32, nstep=5):
        """
        Perform rollouts in parallel using a thread pool.

        Args:
            state (np.ndarray): Array to store the results of the rollouts.
            ctrl (np.ndarray): Control actions for the rollouts.
            initial_state (np.ndarray): Initial states for the rollouts.
            num_workers (int): Number of parallel threads to use.
            nstep (int): Number of steps in each rollout.
        """
        num_workers = min(num_workers, self.num_workers, len(initial_state))
        if num_workers == 0:
            return
        # The selected-plan diagnostic contains one rollout. Execute locally
        # instead of submitting a single task and waiting on the worker pool.
        if num_workers == 1:
            if not hasattr(self.thread_local, 'data'):
                self.thread_initializer()
            self.call_rollout(initial_state, ctrl, state)
            return

        # Small, balanced chunks let the pool pick up more work when contact-
        # heavy candidates take longer. Bound submission overhead to two chunks
        # per worker and avoid creating extra chunks with fewer than four rows.
        chunks = min(2*num_workers, max(num_workers, len(initial_state)//4))
        quotient, remainder = divmod(len(initial_state), chunks)
        futures = []
        start = 0
        for worker in range(chunks):
            stop = start + quotient + (worker < remainder)
            futures.append(self.executor.submit(self.call_rollout, initial_state[start:stop],
                                                ctrl[start:stop], state[start:stop]))
            start = stop
        for future in futures:
            future.result()
