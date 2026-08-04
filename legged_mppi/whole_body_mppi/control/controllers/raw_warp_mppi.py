"""Raw Warp implementation of a fully GPU-resident MPPI planner update."""

import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import warp as wp
import mujoco.mjx.third_party.mujoco_warp as mjw
from scipy.interpolate import CubicSpline


@wp.kernel
def _reset_qpos(
    observation: wp.array(dtype=wp.float32),
    qpos: wp.array2d(dtype=wp.float32),
):
    world, index = wp.tid()
    qpos[world, index] = observation[index]


@wp.kernel
def _reset_qvel(
    observation: wp.array(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    qacc_warmstart: wp.array2d(dtype=wp.float32),
    qfrc_applied: wp.array2d(dtype=wp.float32),
    nq: int,
):
    world, index = wp.tid()
    qvel[world, index] = observation[nq + index]
    qacc_warmstart[world, index] = 0.0
    qfrc_applied[world, index] = 0.0


@wp.kernel
def _reset_world(
    time_array: wp.array(dtype=wp.float32),
    costs: wp.array(dtype=wp.float32),
):
    world = wp.tid()
    time_array[world] = 0.0
    costs[world] = 0.0


@wp.kernel
def _sample_normal(
    seed: wp.array(dtype=wp.int32),
    trajectory: wp.array2d(dtype=wp.float32),
    noise_sigma: wp.array(dtype=wp.float32),
    act_min: wp.array(dtype=wp.float32),
    act_max: wp.array(dtype=wp.float32),
    actions: wp.array3d(dtype=wp.float32),
    horizon: int,
    act_dim: int,
):
    world, step, actuator = wp.tid()
    offset = (world * horizon + step) * act_dim + actuator
    random_state = wp.rand_init(seed[0], offset)
    value = trajectory[step, actuator] + wp.randn(random_state) * noise_sigma[actuator]
    actions[world, step, actuator] = wp.clamp(
        value, act_min[actuator], act_max[actuator]
    )


@wp.kernel
def _sample_knots(
    seed: wp.array(dtype=wp.int32),
    trajectory: wp.array2d(dtype=wp.float32),
    noise_sigma: wp.array(dtype=wp.float32),
    knot_indices: wp.array(dtype=wp.int32),
    knots: wp.array3d(dtype=wp.float32),
    knot_count: int,
    act_dim: int,
):
    world, knot, actuator = wp.tid()
    offset = (world * knot_count + knot) * act_dim + actuator
    random_state = wp.rand_init(seed[0], offset)
    knots[world, knot, actuator] = (
        trajectory[knot_indices[knot], actuator]
        + wp.randn(random_state) * noise_sigma[actuator]
    )


@wp.kernel
def _interpolate_actions(
    knots: wp.array3d(dtype=wp.float32),
    spline_basis: wp.array2d(dtype=wp.float32),
    act_min: wp.array(dtype=wp.float32),
    act_max: wp.array(dtype=wp.float32),
    actions: wp.array3d(dtype=wp.float32),
    knot_count: int,
):
    world, step, actuator = wp.tid()
    value = float(0.0)
    for knot in range(knot_count):
        value += spline_basis[step, knot] * knots[world, knot, actuator]
    actions[world, step, actuator] = wp.clamp(
        value, act_min[actuator], act_max[actuator]
    )


@wp.kernel
def _set_control(
    actions: wp.array3d(dtype=wp.float32),
    control: wp.array2d(dtype=wp.float32),
    step: int,
):
    world, actuator = wp.tid()
    control[world, actuator] = actions[world, step, actuator]


@wp.kernel
def _accumulate_cost(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    actions: wp.array3d(dtype=wp.float32),
    joints_ref: wp.array2d(dtype=wp.float32),
    body_ref: wp.array(dtype=wp.float32),
    box_ref: wp.array(dtype=wp.float32),
    q_robot_diag: wp.array(dtype=wp.float32),
    q_box_diag: wp.array(dtype=wp.float32),
    r_diag: wp.array(dtype=wp.float32),
    joint_qpos_indices: wp.array(dtype=wp.int32),
    joint_qvel_indices: wp.array(dtype=wp.int32),
    costs: wp.array(dtype=wp.float32),
    step: int,
    push_box: int,
    robot_qpos_adr: int,
    robot_qvel_adr: int,
):
    world = wp.tid()
    total = float(0.0)

    # Position L1 term. Position is excluded from the quadratic state term.
    for index in range(3):
        difference = qpos[world, robot_qpos_adr + index] - body_ref[index]
        total += wp.abs(difference * q_robot_diag[index])

    # Quaternion distance. Locomotion applies it to all four quaternion
    # weights; push-box applies it only to the scalar quaternion slot.
    dot = float(0.0)
    for index in range(4):
        dot += qpos[world, robot_qpos_adr + 3 + index] * body_ref[3 + index]
    quaternion_distance = 1.0 - wp.abs(dot)
    if push_box == 1:
        total += quaternion_distance * quaternion_distance * q_robot_diag[3]
    else:
        for index in range(4):
            total += (
                quaternion_distance
                * quaternion_distance
                * q_robot_diag[3 + index]
            )

    # Joint position, velocity, and the controller's PD/torque penalty.
    for actuator in range(12):
        joint_position = qpos[world, joint_qpos_indices[actuator]]
        joint_velocity = qvel[world, joint_qvel_indices[actuator]]
        position_error = joint_position - joints_ref[actuator, step]
        velocity_error = joint_velocity - joints_ref[12 + actuator, step]
        torque_error = (
            50.0 * (actions[world, step, actuator] - joint_position)
            - 3.0 * joint_velocity
        )
        total += position_error * position_error * q_robot_diag[7 + actuator]
        total += velocity_error * velocity_error * q_robot_diag[25 + actuator]
        total += torque_error * torque_error * r_diag[actuator]

    # Inverse quaternion rotation of the world-frame base linear velocity.
    qw = qpos[world, robot_qpos_adr + 3]
    qx = qpos[world, robot_qpos_adr + 4]
    qy = qpos[world, robot_qpos_adr + 5]
    qz = qpos[world, robot_qpos_adr + 6]
    vx = qvel[world, robot_qvel_adr]
    vy = qvel[world, robot_qvel_adr + 1]
    vz = qvel[world, robot_qvel_adr + 2]
    local_x = (
        (1.0 - 2.0 * (qy * qy + qz * qz)) * vx
        + 2.0 * (qx * qy + qw * qz) * vy
        + 2.0 * (qx * qz - qw * qy) * vz
    )
    local_y = (
        2.0 * (qx * qy - qw * qz) * vx
        + (1.0 - 2.0 * (qx * qx + qz * qz)) * vy
        + 2.0 * (qy * qz + qw * qx) * vz
    )
    local_z = (
        2.0 * (qx * qz + qw * qy) * vx
        + 2.0 * (qy * qz - qw * qx) * vy
        + (1.0 - 2.0 * (qx * qx + qy * qy)) * vz
    )
    local_error_x = local_x - body_ref[7]
    local_error_y = local_y - body_ref[8]
    local_error_z = local_z - body_ref[9]
    total += local_error_x * local_error_x * q_robot_diag[19]
    total += local_error_y * local_error_y * q_robot_diag[20]
    total += local_error_z * local_error_z * q_robot_diag[21]
    for index in range(3):
        angular_error = qvel[world, robot_qvel_adr + 3 + index] - body_ref[10 + index]
        total += angular_error * angular_error * q_robot_diag[22 + index]

    if push_box == 1:
        for index in range(3):
            box_error = qpos[world, index] - box_ref[index]
            total += wp.abs(box_error * q_box_diag[index])

    costs[world] = costs[world] + total


@wp.kernel
def _compute_weights(
    costs: wp.array(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    diagnostics: wp.array(dtype=wp.float32),
    sample_count: int,
    temperature: float,
):
    minimum = costs[0]
    maximum = costs[0]
    for sample in range(1, sample_count):
        minimum = wp.min(minimum, costs[sample])
        maximum = wp.max(maximum, costs[sample])
    cost_range = wp.max(maximum - minimum, 1.1920929e-7)
    weight_sum = float(0.0)
    for sample in range(sample_count):
        weight = wp.exp(-(costs[sample] - minimum) / (temperature * cost_range))
        weights[sample] = weight
        weight_sum += weight
    diagnostics[0] = minimum
    diagnostics[1] = weight_sum


@wp.kernel
def _reduce_plan(
    actions: wp.array3d(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    diagnostics: wp.array(dtype=wp.float32),
    act_min: wp.array(dtype=wp.float32),
    act_max: wp.array(dtype=wp.float32),
    plan: wp.array2d(dtype=wp.float32),
    sample_count: int,
):
    step, actuator = wp.tid()
    weighted_action = float(0.0)
    for sample in range(sample_count):
        weighted_action += weights[sample] * actions[sample, step, actuator]
    value = weighted_action / (diagnostics[1] + 1.0e-10)
    plan[step, actuator] = wp.clamp(
        value, act_min[actuator], act_max[actuator]
    )


class RawWarpMPPI:
    """One fixed-shape MPPI update captured as a single CUDA graph."""

    def __init__(self, controller, cost_mode):
        if cost_mode not in ("locomotion", "push_box"):
            raise ValueError(f"Unsupported Warp cost mode: {cost_mode}")
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError("The warp backend requires a CUDA-capable GPU")

        self.device = "cuda:0"
        self.cost_mode = cost_mode
        self.n_samples = controller.n_samples
        self.horizon = controller.horizon
        self.act_dim = controller.act_dim
        self.nq = controller.model.nq
        self.nv = controller.model.nv
        self.temperature = float(controller.temperature)
        self.sample_type = controller.sample_type
        self.last_update_ms = None

        if self.sample_type not in ("normal", "cubic"):
            raise ValueError(
                f"Warp MPPI does not support sample_type={self.sample_type!r}"
            )

        self.model = mjw.put_model(controller.model)
        # These capacities are per world; make_data multiplies contact/CCD
        # storage by nworld internally. Passing sample-scaled values here
        # squares the allocation and exhausts VRAM for large MPPI batches.
        self.data = mjw.make_data(
            controller.model,
            nworld=self.n_samples,
            nconmax=32,
            nccdmax=32,
            njmax=192,
        )

        def array(value, dtype=wp.float32):
            return wp.array(np.asarray(value), dtype=dtype, device=self.device)

        self.observation = array(
            np.concatenate((controller.model.qpos0, np.zeros(self.nv))),
        )
        self.trajectory = array(controller.trajectory.astype(np.float32))
        self.noise_sigma = array(controller.noise_sigma.astype(np.float32))
        self.joints_ref = array(controller.joints_ref.astype(np.float32))
        self.body_ref = array(controller.body_ref.astype(np.float32))
        self.box_ref = array(np.zeros(7, dtype=np.float32))
        self.q_box_diag = array(np.zeros(4, dtype=np.float32))
        self.seed = array(np.array([1], dtype=np.int32), dtype=wp.int32)

        q_matrix = controller.Q if cost_mode == "locomotion" else controller.Q_robot
        self.q_robot_diag = array(np.diag(q_matrix).astype(np.float32))
        self.r_diag = array(np.diag(controller.R).astype(np.float32))
        self.act_min = array(controller.act_min.astype(np.float32))
        self.act_max = array(controller.act_max.astype(np.float32))
        self.joint_qpos_indices = array(
            controller.joint_qpos_indices.astype(np.int32), dtype=wp.int32
        )
        self.joint_qvel_indices = array(
            controller.model.jnt_dofadr[
                controller.model.actuator_trnid[:, 0]
            ].astype(np.int32),
            dtype=wp.int32,
        )

        self.actions = wp.zeros(
            (self.n_samples, self.horizon, self.act_dim),
            dtype=wp.float32,
            device=self.device,
        )
        self.costs = wp.zeros(self.n_samples, dtype=wp.float32, device=self.device)
        self.weights = wp.zeros(self.n_samples, dtype=wp.float32, device=self.device)
        self.diagnostics = wp.zeros(2, dtype=wp.float32, device=self.device)
        self.plan = wp.zeros(
            (self.horizon, self.act_dim), dtype=wp.float32, device=self.device
        )

        if self.sample_type == "cubic":
            self.knot_indices_host = np.round(
                np.linspace(0, self.horizon - 1, num=controller.n_knots)
            ).astype(np.int32)
            basis = CubicSpline(
                self.knot_indices_host,
                np.eye(controller.n_knots),
                axis=0,
            )(np.arange(self.horizon))
            self.knot_indices = array(self.knot_indices_host, dtype=wp.int32)
            self.spline_basis = array(basis.astype(np.float32))
            self.knots = wp.zeros(
                (self.n_samples, controller.n_knots, self.act_dim),
                dtype=wp.float32,
                device=self.device,
            )
            self.knot_count = controller.n_knots

        self.push_box = int(cost_mode == "push_box")
        self.robot_qpos_adr = 7 if self.push_box else 0
        self.robot_qvel_adr = 6 if self.push_box else 0
        self._host_seed = int(
            controller.random_generator.bit_generator._seed_seq.entropy
        )

        # Compile all kernels once, then capture the complete fixed-shape MPPI
        # update. Every graph replay starts by resetting the dynamic state.
        self._launch_pipeline()
        wp.synchronize()
        wp.capture_begin(device=self.device)
        self._launch_pipeline()
        self.graph = wp.capture_end(device=self.device)

    def _launch_pipeline(self):
        wp.launch(
            _reset_qpos,
            dim=(self.n_samples, self.nq),
            inputs=(self.observation, self.data.qpos),
            device=self.device,
        )
        wp.launch(
            _reset_qvel,
            dim=(self.n_samples, self.nv),
            inputs=(
                self.observation,
                self.data.qvel,
                self.data.qacc_warmstart,
                self.data.qfrc_applied,
                self.nq,
            ),
            device=self.device,
        )
        wp.launch(
            _reset_world,
            dim=self.n_samples,
            inputs=(self.data.time, self.costs),
            device=self.device,
        )

        if self.sample_type == "normal":
            wp.launch(
                _sample_normal,
                dim=(self.n_samples, self.horizon, self.act_dim),
                inputs=(
                    self.seed,
                    self.trajectory,
                    self.noise_sigma,
                    self.act_min,
                    self.act_max,
                    self.actions,
                    self.horizon,
                    self.act_dim,
                ),
                device=self.device,
            )
        else:
            wp.launch(
                _sample_knots,
                dim=(self.n_samples, self.knot_count, self.act_dim),
                inputs=(
                    self.seed,
                    self.trajectory,
                    self.noise_sigma,
                    self.knot_indices,
                    self.knots,
                    self.knot_count,
                    self.act_dim,
                ),
                device=self.device,
            )
            wp.launch(
                _interpolate_actions,
                dim=(self.n_samples, self.horizon, self.act_dim),
                inputs=(
                    self.knots,
                    self.spline_basis,
                    self.act_min,
                    self.act_max,
                    self.actions,
                    self.knot_count,
                ),
                device=self.device,
            )

        for step in range(self.horizon):
            wp.launch(
                _set_control,
                dim=(self.n_samples, self.act_dim),
                inputs=(self.actions, self.data.ctrl, step),
                device=self.device,
            )
            mjw.step(self.model, self.data)
            wp.launch(
                _accumulate_cost,
                dim=self.n_samples,
                inputs=(
                    self.data.qpos,
                    self.data.qvel,
                    self.actions,
                    self.joints_ref,
                    self.body_ref,
                    self.box_ref,
                    self.q_robot_diag,
                    self.q_box_diag,
                    self.r_diag,
                    self.joint_qpos_indices,
                    self.joint_qvel_indices,
                    self.costs,
                    step,
                    self.push_box,
                    self.robot_qpos_adr,
                    self.robot_qvel_adr,
                ),
                device=self.device,
            )

        wp.launch(
            _compute_weights,
            dim=1,
            inputs=(
                self.costs,
                self.weights,
                self.diagnostics,
                self.n_samples,
                self.temperature,
            ),
            device=self.device,
        )
        wp.launch(
            _reduce_plan,
            dim=(self.horizon, self.act_dim),
            inputs=(
                self.actions,
                self.weights,
                self.diagnostics,
                self.act_min,
                self.act_max,
                self.plan,
                self.n_samples,
            ),
            device=self.device,
        )

    def update(
        self,
        observation,
        trajectory,
        noise_sigma,
        joints_ref,
        body_ref,
        box_ref=None,
        q_box=None,
    ):
        if box_ref is None:
            box_ref = np.zeros(7, dtype=np.float32)
        if q_box is None:
            q_box_diag = np.zeros(4, dtype=np.float32)
        else:
            q_box_diag = np.diag(q_box).astype(np.float32, copy=False)

        started = time.perf_counter()
        self._host_seed = (self._host_seed + 1) & 0x7FFFFFFF
        self.observation.assign(np.asarray(observation, dtype=np.float32))
        self.trajectory.assign(np.asarray(trajectory, dtype=np.float32))
        self.noise_sigma.assign(np.asarray(noise_sigma, dtype=np.float32))
        self.joints_ref.assign(np.asarray(joints_ref, dtype=np.float32))
        self.body_ref.assign(np.asarray(body_ref, dtype=np.float32))
        self.box_ref.assign(np.asarray(box_ref, dtype=np.float32))
        self.q_box_diag.assign(q_box_diag)
        self.seed.assign(np.asarray([self._host_seed], dtype=np.int32))

        wp.capture_launch(self.graph)
        wp.synchronize()
        plan = self.plan.numpy()
        weights = self.weights.numpy()
        diagnostics = self.diagnostics.numpy()
        self.last_update_ms = (time.perf_counter() - started) * 1000.0
        return plan, weights, float(diagnostics[0])
