"""Small MuJoCo environment matching the MPPI rollout state contract."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import mujoco
import numpy as np
from mujoco import rollout


MODEL_PATH: Final = Path(__file__).resolve().parents[1] / "models" / "scene.xml"

LEG_JOINT_NAMES: Final = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)
ARM_JOINT_NAMES: Final = (
    "Rotation", "Pitch", "Elbow", "Wrist_Pitch"
)
ACTUATOR_NAMES: Final = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    *ARM_JOINT_NAMES,
)


class Go2SOArmEnv:
    """Go2 with a four-axis SO-ARM100 arm and position-target controls."""

    def __init__(self, model_path: str | Path = MODEL_PATH, dt: float = 0.01):
        self.model_path = Path(model_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.model.opt.timestep = dt
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        self.data = mujoco.MjData(self.model)
        self.joint_names = LEG_JOINT_NAMES + ARM_JOINT_NAMES
        self.actuator_names = ACTUATOR_NAMES
        self._validate_contract()
        self.qpos_indices = self._joint_addresses("qpos")
        self.qvel_indices = self._joint_addresses("qvel")
        self.reset()

    @property
    def observation_dim(self) -> int:
        return self.model.nq + self.model.nv

    @property
    def fullphysics_dim(self) -> int:
        return mujoco.mj_stateSize(
            self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value
        )

    @property
    def action_low(self) -> np.ndarray:
        return self.model.actuator_ctrlrange[:, 0].copy()

    @property
    def action_high(self) -> np.ndarray:
        return self.model.actuator_ctrlrange[:, 1].copy()

    def _joint_addresses(self, kind: str) -> np.ndarray:
        addresses = self.model.jnt_qposadr if kind == "qpos" else self.model.jnt_dofadr
        return np.array([
            addresses[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.joint_names
        ])

    def _validate_contract(self) -> None:
        if (self.model.nq, self.model.nv, self.model.nu) != (23, 22, 16):
            raise ValueError(
                "Unexpected Go2 + SO-ARM100 dimensions: "
                f"nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}"
            )
        actual = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        )
        if actual != ACTUATOR_NAMES:
            raise ValueError(f"Actuator order mismatch: {actual} != {ACTUATOR_NAMES}")

    def reset(self, keyframe: str = "sit") -> np.ndarray:
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key_id < 0:
            raise ValueError(f"Unknown keyframe: {keyframe}")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def observation(self) -> np.ndarray:
        return np.concatenate((self.data.qpos, self.data.qvel)).copy()

    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.qpos_indices].copy()

    def joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self.qvel_indices].copy()

    def step(self, action: np.ndarray, frame_skip: int = 1) -> np.ndarray:
        action = np.asarray(action, dtype=float)
        if action.shape != (self.model.nu,):
            raise ValueError(f"Expected action shape {(self.model.nu,)}, got {action.shape}")
        self.data.ctrl[:] = np.clip(action, self.action_low, self.action_high)
        for _ in range(frame_skip):
            mujoco.mj_step(self.model, self.data)
        return self.observation()

    def rollout(
        self, actions: np.ndarray, initial_observation: np.ndarray | None = None
    ) -> np.ndarray:
        """Return FULLPHYSICS states ``[time, qpos, qvel]`` for action batches."""
        actions = np.asarray(actions, dtype=float)
        squeeze = actions.ndim == 2
        if squeeze:
            actions = actions[None, ...]
        if actions.ndim != 3 or actions.shape[2] != self.model.nu:
            raise ValueError(
                f"Expected actions shaped (rollouts, horizon, {self.model.nu}), "
                f"got {actions.shape}"
            )
        actions = np.clip(actions, self.action_low, self.action_high)
        obs = self.observation() if initial_observation is None else np.asarray(
            initial_observation, dtype=float
        )
        if obs.shape != (self.observation_dim,):
            raise ValueError(
                f"Expected observation shape {(self.observation_dim,)}, got {obs.shape}"
            )
        initial_state = np.repeat(
            np.concatenate(([0.0], obs))[None, :], actions.shape[0], axis=0
        )
        states = np.empty(
            (actions.shape[0], actions.shape[1], self.fullphysics_dim), dtype=float
        )
        rollout.rollout(
            self.model, mujoco.MjData(self.model), skip_checks=False,
            initial_state=initial_state, control=actions, state=states,
        )
        return states[0] if squeeze else states


# Preserve the original public name for callers shared with legged_mani.
Go2ARXEnv = Go2SOArmEnv
Go2OpenManipulatorEnv = Go2SOArmEnv
B2Z1Env = Go2SOArmEnv
