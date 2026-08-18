import mujoco.viewer

from mani_mppi.interface.environment import Go2SOArmEnv


def main() -> None:
    env = Go2SOArmEnv()
    env.reset("sit")
    mujoco.viewer.launch(env.model, env.data)


if __name__ == "__main__":
    main()
