"""View or smoke-test the Go1 + DM assembly without ROS or MPPI."""
import argparse
from pathlib import Path
import time

import mujoco
import numpy as np

MODEL = Path(__file__).resolve().parents[1] / 'whole_body_mppi/models/go1_dm/scene.xml'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true', help='Run the posture hold without opening a window')
    parser.add_argument('--seconds', type=float, default=5.0, help='Headless simulation duration')
    parser.add_argument('--render', type=Path, help='Save a PNG of the initial assembly (set MUJOCO_GL=egl on headless Linux)')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be finite and positive')
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    print(f'nq={model.nq}, nv={model.nv}, nu={model.nu}, total mass={model.body_mass.sum():.3f} kg')
    if args.render:
        from PIL import Image
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, 0.45]
        camera.distance, camera.azimuth, camera.elevation = 1.9, 125, -20
        with mujoco.Renderer(model, height=720, width=960) as renderer:
            renderer.update_scene(data, camera=camera)
            args.render.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(renderer.render()).save(args.render)
        print(f'Saved {args.render}')
    if args.headless:
        for _ in range(int(np.ceil(args.seconds / model.opt.timestep))):
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or np.any(data.warning.number):
            raise RuntimeError('Simulation produced invalid state or MuJoCo warnings')
        print(f't={data.time:.3f} s, base height={data.qpos[2]:.3f} m, warnings=0')
    else:
        from mujoco import viewer as mjviewer
        with mjviewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0, 0, 0.4]
            viewer.cam.distance = 1.9
            while viewer.is_running():
                start = time.monotonic()
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.monotonic() - start)))


if __name__ == '__main__':
    main()
