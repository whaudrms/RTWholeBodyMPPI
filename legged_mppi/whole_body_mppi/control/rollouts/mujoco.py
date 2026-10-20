"""Threaded MuJoCo batch rollouts, with private worker data."""
import concurrent.futures
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
        num_workers = min(num_workers, len(initial_state))
        if num_workers == 0:
            return
        n = len(initial_state) // num_workers

        # Divide tasks into chunks for each worker
        chunks = [(initial_state[i * n:(i + 1) * n], ctrl[i * n:(i + 1) * n], state[i * n:(i + 1) * n])
                for i in range(num_workers - 1)]

        # Add remaining chunk
        chunks.append((initial_state[(num_workers - 1) * n:], ctrl[(num_workers - 1) * n:], state[(num_workers - 1) * n:]))

        # Submit tasks to thread pool
        futures = [self.executor.submit(self.call_rollout, *chunk) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Ensure all threads complete execution

