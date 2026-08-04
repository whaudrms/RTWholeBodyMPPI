"""Compatibility helpers for mujoco-python-viewer on newer MuJoCo releases."""

import mujoco


def install_mjv_move_camera_compat():
    """Accept the legacy MuJoCo viewer's extra ``MjvScene`` argument.

    mujoco-python-viewer 0.1.4 calls ``mjv_moveCamera`` with the pre-3.2
    signature ``(model, action, dx, dy, scene, camera)``.  Current MuJoCo uses
    ``(model, action, dx, dy, camera)``.  The wrapper preserves both forms so
    the third-party viewer can handle mouse drag and scroll events.
    """
    current = mujoco.mjv_moveCamera
    if getattr(current, "_rtwholebodymppi_compat", False):
        return

    version = tuple(int(part) for part in mujoco.__version__.split(".")[:3])
    if version < (3, 2, 0):
        return

    def move_camera(model, action, reldx, reldy, *scene_and_camera):
        if len(scene_and_camera) == 1:
            camera = scene_and_camera[0]
        elif len(scene_and_camera) == 2:
            _, camera = scene_and_camera
        else:
            raise TypeError(
                "mjv_moveCamera compatibility wrapper expects a camera, "
                "optionally preceded by a scene"
            )
        return current(model, action, reldx, reldy, camera)

    move_camera._rtwholebodymppi_compat = True
    mujoco.mjv_moveCamera = move_camera
