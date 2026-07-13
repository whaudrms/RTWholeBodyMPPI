"""Compatibility import; implementation lives in :mod:`legged_mani.whole_body_mppi.interface`."""

from legged_mani.mani_mppi.interface.environment import (  # noqa: F401
    ACTUATOR_NAMES,
    ARM_JOINT_NAMES,
    LEG_JOINT_NAMES,
    MODEL_PATH,
    B2Z1Env,
)

__all__ = [
    "ACTUATOR_NAMES", "ARM_JOINT_NAMES", "LEG_JOINT_NAMES", "MODEL_PATH", "B2Z1Env"
]
