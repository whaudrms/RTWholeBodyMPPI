"""Model-derived MPPI sampling and parallel MuJoCo rollout machinery.

Adapted from ``legged_mppi/whole_body_mppi/control/controllers/base_controller.py``.
Robot dimensions, actuator limits, and initial controls are intentionally read
from the model instead of being hard-coded for Go1.
"""

from __future__ import annotations

import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import yaml
from mujoco import rollout


class BaseMPPI:
    """Shared MPPI controller adapted from the original Go1 base controller.

    The original sampling and threaded MuJoCo rollout role is retained. Robot
    dimensions, limits, and the nominal control are derived from the supplied
    B2-Z1 model so task controllers only need to define their cost and update.
    """

    def __init__(self, model_path: str | Path, config_path: str | Path):
        with Path(config_path).open("r", encoding="utf-8") as stream:
            self.params = yaml.safe_load(stream)

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = float(self.params["dt"])
        rollout_cone = self.params.get("rollout_cone")
        if rollout_cone is not None:
            cone_types = {
                "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
                "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
            }
            try:
                self.model.opt.cone = cone_types[rollout_cone]
            except KeyError as error:
                raise ValueError(
                    f"Unsupported rollout_cone: {rollout_cone}"
                ) from error
        if "o_solref" in self.params:
            # The original rollout model enables MuJoCo's global contact
            # override before applying o_solref.
            self.model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_OVERRIDE
            self.model.opt.o_solref = np.asarray(self.params["o_solref"], dtype=float)
        self.temperature = float(self.params["lambda"])
        self.horizon = int(self.params["horizon"])
        self.n_samples = int(self.params["n_samples"])
        self.num_workers = max(1, min(int(self.params["n_workers"]), self.n_samples))
        self.sample_type = self.params.get("sample_type", "normal")
        self.n_knots = int(self.params.get("n_knots", 4))
        self.random_generator = np.random.default_rng(self.params.get("seed", 42))

        self.act_dim = self.model.nu
        self.act_min = self.model.actuator_ctrlrange[:, 0].copy()
        self.act_max = self.model.actuator_ctrlrange[:, 1].copy()
        keyframe = self.params.get("keyframe", "sit")
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key_id < 0:
            raise ValueError(f"Unknown MPPI initialization keyframe: {keyframe}")
        sampling_init = self.params.get("sampling_init")
        if sampling_init is None:
            self.sampling_init = self.model.key_ctrl[key_id].copy()
        else:
            self.sampling_init = np.asarray(sampling_init, dtype=float)
            if self.sampling_init.shape != (self.act_dim,):
                raise ValueError(
                    f"sampling_init must have {self.act_dim} entries, "
                    f"got {self.sampling_init.shape}"
                )
            if not np.isfinite(self.sampling_init).all():
                raise ValueError("sampling_init must be finite")

        self.base_noise_sigma = self._build_noise_sigma(self.params["noise_sigma"])
        self.noise_sigma = self.base_noise_sigma.copy()
        if self.noise_sigma.shape != (self.act_dim,):
            raise ValueError(
                f"noise_sigma must have {self.act_dim} entries, got {self.noise_sigma.shape}"
            )

        full_state_dim = mujoco.mj_stateSize(
            self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value
        )
        self.state_rollouts = np.empty(
            (self.n_samples, self.horizon, full_state_dim), dtype=float
        )
        self.capture_rollout_sensors = False
        self.sensor_rollouts = None
        self.thread_local = threading.local()
        self.executor = ThreadPoolExecutor(
            max_workers=self.num_workers, initializer=self._thread_initializer
        )
        self._closed = False
        # Keep the same extension points as the original base_controller.
        self.rollout_func = self.rollout_actions
        self.cost_func = None
        self.reset_planner()

    def _build_noise_sigma(self, config) -> np.ndarray:
        """Build one exploration stddev for every model actuator."""
        if not isinstance(config, dict):
            sigma = np.asarray(config, dtype=float)
            if sigma.shape != (self.act_dim,):
                raise ValueError(
                    f"noise_sigma must have {self.act_dim} entries, got {sigma.shape}"
                )
            return sigma

        required = ("hip", "thigh", "calf", "arm")
        missing = [name for name in required if name not in config]
        if missing:
            raise ValueError(f"noise_sigma is missing actuator groups: {missing}")

        sigma = []
        for actuator_id in range(self.act_dim):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
            )
            if name.endswith("_hip"):
                group = "hip"
            elif name.endswith("_thigh"):
                group = "thigh"
            elif name.endswith("_calf"):
                group = "calf"
            elif name.startswith("joint"):
                group = "arm"
            else:
                raise ValueError(f"Unknown actuator group for '{name}'")
            sigma.append(float(config[group]))

        sigma = np.asarray(sigma, dtype=float)
        if not np.isfinite(sigma).all() or np.any(sigma < 0.0):
            raise ValueError("noise_sigma values must be finite and non-negative")
        return sigma

    def set_noise_for_gait(self, gait_name: str) -> None:
        """Apply the configured scalar exploration scale for a gait."""
        scales = self.params.get("gait_noise_scale", {})
        scale = float(scales.get(gait_name, 1.0))
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(f"Invalid gait noise scale for '{gait_name}': {scale}")
        self.noise_sigma = self.base_noise_sigma * scale

    def reset_planner(self) -> None:
        self.trajectory = np.repeat(
            self.sampling_init[None, :], self.horizon, axis=0
        )
        self.selected_trajectory = self.trajectory.copy()

    def generate_noise(self, size: tuple[int, ...]) -> np.ndarray:
        return self.random_generator.normal(size=size) * self.noise_sigma

    def perturb_action(self) -> np.ndarray:
        if self.sample_type == "normal":
            noise = self.generate_noise((self.n_samples, self.horizon, self.act_dim))
            actions = self.trajectory[None, :, :] + noise
        elif self.sample_type == "cubic":
            from scipy.interpolate import CubicSpline

            indices = np.unique(np.rint(
                np.linspace(0, self.horizon - 1, self.n_knots)
            ).astype(int))
            noise = self.generate_noise((self.n_samples, len(indices), self.act_dim))
            # Preserve the full-rate nominal trajectory.  Interpolating the
            # absolute actions from only a few knots smooths away short gait
            # swing phases; only the exploration perturbation should be
            # represented by the cubic spline.
            smooth_noise = CubicSpline(indices, noise, axis=1)(
                np.arange(self.horizon)
            )
            actions = self.trajectory[None, :, :] + smooth_noise
        else:
            raise ValueError(f"Unsupported sample_type: {self.sample_type}")
        return np.clip(actions, self.act_min, self.act_max)

    def _thread_initializer(self) -> None:
        self.thread_local.data = mujoco.MjData(self.model)

    def enable_rollout_sensors(self) -> None:
        """Capture model sensor outputs alongside each rollout state."""
        self.capture_rollout_sensors = True
        self.sensor_rollouts = np.empty(
            (self.n_samples, self.horizon, self.model.nsensordata), dtype=float
        )

    def _call_rollout(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        states: np.ndarray,
        sensors: np.ndarray | None = None,
    ) -> None:
        rollout.rollout(
            self.model, self.thread_local.data, skip_checks=True,
            nroll=states.shape[0], nstep=states.shape[1],
            initial_state=initial_state, control=controls, state=states,
            sensordata=sensors,
        )

    def rollout_actions(self, observation: np.ndarray, controls: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=float)
        controls = np.asarray(controls, dtype=float)
        expected = self.model.nq + self.model.nv
        if observation.shape != (expected,):
            raise ValueError(f"Expected observation shape {(expected,)}, got {observation.shape}")

        if controls.ndim != 3 or controls.shape[1:] != (self.horizon, self.act_dim):
            raise ValueError(
                f"Expected controls shape (N, {self.horizon}, {self.act_dim}), "
                f"got {controls.shape}"
            )

        n_rollouts = controls.shape[0]
        if n_rollouts < 1:
            raise ValueError("controls must contain at least one rollout")

        full_state_dim = mujoco.mj_stateSize(
            self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value
        )
        if n_rollouts == self.n_samples:
            state_rollouts = self.state_rollouts
        else:
            state_rollouts = np.empty(
                (n_rollouts, self.horizon, full_state_dim), dtype=float
            )

        sensor_rollouts = None
        if self.capture_rollout_sensors:
            expected_sensor_shape = (
                n_rollouts,
                self.horizon,
                self.model.nsensordata,
            )
            if (
                n_rollouts == self.n_samples
                and self.sensor_rollouts is not None
                and self.sensor_rollouts.shape == expected_sensor_shape
            ):
                sensor_rollouts = self.sensor_rollouts
            else:
                sensor_rollouts = np.empty(expected_sensor_shape, dtype=float)
                if n_rollouts == self.n_samples:
                    self.sensor_rollouts = sensor_rollouts

        initial = np.repeat(
            np.concatenate(([0.0], observation))[None, :], n_rollouts, axis=0
        )
        n_workers = min(self.num_workers, n_rollouts)
        boundaries = np.linspace(
            0, n_rollouts, n_workers + 1, dtype=int
        )
        chunks = [
            slice(start, stop)
            for start, stop in zip(boundaries[:-1], boundaries[1:])
            if stop > start
        ]
        futures = [
            self.executor.submit(
                self._call_rollout,
                initial[index],
                controls[index],
                state_rollouts[index],
                None if sensor_rollouts is None else sensor_rollouts[index],
            )
            for index in chunks
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
        return state_rollouts[:, :, 1:]

    def close(self) -> None:
        if not self._closed:
            self.executor.shutdown(wait=True)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self):
        if hasattr(self, "executor"):
            self.close()
