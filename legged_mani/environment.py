"""Compatibility import; implementation lives in :mod:`legged_mani.interface`."""

from legged_mani.interface.environment import (  # noqa: F401
    ACTUATOR_NAMES,
    ARM_JOINT_NAMES,
    LEG_JOINT_NAMES,
    MODEL_PATH,
    B2Z1Env,
)

__all__ = [
    "ACTUATOR_NAMES", "ARM_JOINT_NAMES", "LEG_JOINT_NAMES", "MODEL_PATH", "B2Z1Env"
]
