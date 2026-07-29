"""
Task definitions for robot navigation and behavior scenarios.

Each task is represented as a dictionary containing key parameters:
- `goal_pos`: List of target positions in the format [x, y, z].
- `default_orientation`: Default orientation of the robot as a quaternion [w, x, y, z].
- `cmd_vel`: Commanded velocities in the format [linear x, linear y] in body frame.
- `goal_thresh`: Thresholds for achieving goals.
- `desired_gait`: Gait type for each phase of the task.
- `waiting_times`: Number of simulator steps to wait at each EE waypoint.
- `box_pos_ref`: World-frame box position target for manipulation tasks.
- `model_path`: Path to the robot's model file.
- `config_path`: Path to the robot's configuration file.
- `sim_path`: Path to the simulation file.
"""

# MPPI rollouts must use the same contact environment as the simulator.
DEFAULT_MODEL_PATH = 'models/b2_z1_base.xml'
DEFAULT_CONFIG_PATH = 'configs/mppi_locomotion.yml'
DEFAULT_SIM_PATH = 'models/scene.xml'
DEFAULT_ORIENTATION = [[1, 0, 0, 0]]
STAND_BASE_HEIGHT = 0.543542
BIG_BOX_SCALE = 1.5
BIG_BOX_TOP_HEIGHT = 0.35 * BIG_BOX_SCALE


TASKS = {
    "walk_straight": {
        "goal_pos": [[0, 0, STAND_BASE_HEIGHT],
                     [1, 0, STAND_BASE_HEIGHT],
                     [2, 0, STAND_BASE_HEIGHT]],
        "default_orientation": DEFAULT_ORIENTATION,
        "cmd_vel": [[0.0, 0.0], 
                    [0.2, 0.0], 
                    [0.0, 0.0]],
        "goal_thresh": [0.2, 
                        0.2, 
                        0.2],
        "desired_gait": ['in_place', 
                         'walk_fast', 
                         'walk_fast'],
        "waiting_times": [0, 
                          0, 
                          0],
        "model_path": DEFAULT_MODEL_PATH,
        "config_path": DEFAULT_CONFIG_PATH,
        "sim_path": DEFAULT_SIM_PATH
    },
    "big_box": {
        # Scale the Go1 big-box footprint and route by 1.5 for B2 while
        # preserving the original climb approach and orientation profile. 
        "goal_pos": [[0, 0, STAND_BASE_HEIGHT],
                     [0.4 * BIG_BOX_SCALE, 0, STAND_BASE_HEIGHT],
                     [0.7 * BIG_BOX_SCALE, 0,
                      STAND_BASE_HEIGHT + BIG_BOX_TOP_HEIGHT
                      + 0.08 * BIG_BOX_SCALE],
                     [1.0 * BIG_BOX_SCALE, 0,
                      STAND_BASE_HEIGHT + BIG_BOX_TOP_HEIGHT
                      + 0.03 * BIG_BOX_SCALE],
                     [1.0 * BIG_BOX_SCALE, 0,
                      STAND_BASE_HEIGHT + BIG_BOX_TOP_HEIGHT
                      + 0.03 * BIG_BOX_SCALE]],
        "default_orientation": DEFAULT_ORIENTATION,
        "cmd_vel": [[0.0, 0.0],
                    [0.5, 0.0],
                    [0.5, 0.0],
                    [0.5, 0.0],
                    [0.0, 0.0]],
        "goal_thresh": [0.2] * 5,
        "desired_gait": ['in_place',
                         'walk',
                         'trot',
                         'trot',
                         'in_place'],
        "waiting_times": [50, 0, 0, 0, 200],
        "model_path": 'models/b2_z1_base_big_box.xml',
        "config_path": "configs/mppi_big_box.yml",
        "sim_path": 'models/scene_big_box.xml'
    },
    "stand": {
        "goal_pos": [[0, 0, STAND_BASE_HEIGHT]],
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
        "ee_site": "gripper_center",
        # World-frame EE waypoints. The controller tracks one waypoint at a
        # time and switches after the current waypoint is reached.
        "ee_goal_pos": [
            [0.85, 0.00, 0.80],
            [0.85, 0.00, 0.30],
            [0.85, 0.00, 0.2],
            [1.0, 0.00, 0.48],
            [1.5, 0.5, 0.80],
            [0.0, 0.5, 0.40],
        ],
        # One [w, x, y, z] quaternion for each EE waypoint.
        "ee_goal_quat": [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        "ee_pos_thresh": 0.03,
        "ee_ori_thresh": 1.0, # no orientation tracking
        "waiting_times": [20, 20, 20, 20, 20,20],

        "model_path": DEFAULT_MODEL_PATH,
        "config_path": "configs/mppi_locomani.yml",
        "sim_path": DEFAULT_SIM_PATH,
    },
    "ee_tracking": {
        "ee_site": "gripper_center",
        # Independent copy of the EE waypoint sequence for the original
        # fixed-body-reference tracker.
        "ee_goal_pos": [
            [0.85, 0.00, 0.80],
            [0.85, 0.00, 0.40],
            [1.00, 0.00, 0.5],
            [1.30, 1.00, 0.80],
        ],
        "ee_goal_quat": [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        "ee_pos_thresh": 0.03,
        "ee_ori_thresh": 1.0,
        "waiting_times": [20, 20, 20, 20],

        "model_path": DEFAULT_MODEL_PATH,
        "config_path": "configs/mppi_ee_tracking.yml",
        "sim_path": DEFAULT_SIM_PATH,
    },
    "push_box": {
        # The controller derives its moving EE contact target and CEM base pose
        # from the observed box pose; no scripted locomotion phases are stored.
        "ee_site": "gripper_center",
        "box_joint": "box_joint",
        "box_geom": "box_geom",
        "box_pos_ref": [3.0, 2.0, 0.19],

        "model_path": 'models/b2_z1_base_push_box.xml',
        "config_path": 'configs/mppi_push_box.yml',
        "sim_path": 'models/scene_push_box.xml'
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
