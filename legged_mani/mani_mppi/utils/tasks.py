"""
Task definitions for robot navigation and behavior scenarios.

Each task is represented as a dictionary containing key parameters:
- `goal_pos`: List of target positions in the format [x, y, z].
- `default_orientation`: Default orientation of the robot as a quaternion [w, x, y, z].
- `cmd_vel`: Commanded velocities in the format [linear x, linear y] in body frame.
- `goal_thresh`: Thresholds for achieving goals.
- `desired_gait`: Gait type for each phase of the task.
- `waiting_times`: Time in milliseconds to wait at each phase.
- `model_path`: Path to the robot's model file.
- `config_path`: Path to the robot's configuration file.
- `sim_path`: Path to the simulation file.
"""

# MPPI rollouts must use the same contact environment as the simulator.
DEFAULT_MODEL_PATH = 'models/b2_z1_base.xml'
DEFAULT_CONFIG_PATH = 'configs/mppi_locomotion.yml'
DEFAULT_SIM_PATH = 'models/scene.xml'
DEFAULT_ORIENTATION = [[1, 0, 0, 0]]


TASKS = {
    "walk_straight": {
        "goal_pos": [[0, 0, 0.5], 
                     [1, 0, 0.5], 
                     [2, 0, 0.5]],
        "default_orientation": DEFAULT_ORIENTATION,
        "cmd_vel": [[0.0, 0.0], 
                    [0.2, 0.0], 
                    [0.0, 0.0]],
        "goal_thresh": [0.2, 
                        0.2, 
                        0.2],
        "desired_gait": ['in_place', 
                         'walk_fast', 
                         'in_place'],
        "waiting_times": [0, 
                          0, 
                          0],
        "model_path": DEFAULT_MODEL_PATH,
        "config_path": DEFAULT_CONFIG_PATH,
        "sim_path": DEFAULT_SIM_PATH
    },
    "stand": {
        "goal_pos": [[0, 0, 0.5]],
        "default_orientation": DEFAULT_ORIENTATION,
        "cmd_vel": [[0.0, 0.0]],
        "goal_thresh": [0.2],
        "desired_gait": ['in_place'],
        "waiting_times": [0],
        "model_path": DEFAULT_MODEL_PATH,
        "config_path": DEFAULT_CONFIG_PATH,
        "sim_path": DEFAULT_SIM_PATH
    },
    "locomani": {
        "goal_pos": [[0, 0, 0.5]],
        "cmd_vel": [[0.0, 0.0]],
        "desired_gait": ["in_place"],
        "goal_thresh": [0.1],
        "waiting_times": [0],

        "ee_site": "gripper_center",
        "ee_goal_pos": [[0.7, 0.0, 0.6]],
        "ee_goal_quat": [[1, 0, 0, 0]],

        "model_path": DEFAULT_MODEL_PATH,
        "config_path": "configs/mppi_locomani.yml",
        "sim_path": DEFAULT_SIM_PATH,
    },
}

def get_task(task_name):
    """
    Retrieve task configuration by name.

    Args:
        task_name (str): Name of the task. Must be one of the keys in TASKS.

    Returns:
        dict: Task configuration dictionary.

    Raises:
        ValueError: If the task_name is not found in TASKS.
    """
    if task_name not in TASKS:
        raise ValueError(f"Task '{task_name}' not found. Available tasks: {list(TASKS.keys())}")
    return TASKS[task_name]
