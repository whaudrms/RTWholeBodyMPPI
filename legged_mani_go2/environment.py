"""Compatibility import for the local Go2 + SO-ARM100 environment."""

from .mani_mppi.interface.environment import (  # noqa: F401
    ACTUATOR_NAMES,
    ARM_JOINT_NAMES,
    LEG_JOINT_NAMES,
    MODEL_PATH,
    B2Z1Env,
    Go2ARXEnv,
    Go2OpenManipulatorEnv,
    Go2SOArmEnv,
)

__all__ = [
    "ACTUATOR_NAMES", "ARM_JOINT_NAMES", "LEG_JOINT_NAMES", "MODEL_PATH",
    "B2Z1Env", "Go2ARXEnv", "Go2OpenManipulatorEnv", "Go2SOArmEnv"
]
