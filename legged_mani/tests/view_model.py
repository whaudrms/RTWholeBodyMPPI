import mujoco.viewer

from legged_mani.mani_mppi.interface.environment import B2Z1Env


def main() -> None:
    env = B2Z1Env()
    env.reset("sit")
    mujoco.viewer.launch(env.model, env.data)


if __name__ == "__main__":
    main()