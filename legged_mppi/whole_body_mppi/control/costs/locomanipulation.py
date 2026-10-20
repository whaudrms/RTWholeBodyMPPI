"""Locomanipulation objective; task weights and reference semantics are unchanged."""
import numpy as np
from whole_body_mppi.utils.transforms import batch_world_to_local_velocity


class LocomanipulationCost:
    def quaternion_distance_np(self, q1, q2):
        dot_products = np.einsum('ij,ij->i', q1, q2)
        return 1 - np.abs(dot_products)

    def quaternion_distance_1D_np(self, q1, q2):
        # Broadcasting q2 to match the shape of q1
        dot_products = np.einsum('ij,j->i', q1, q2)
        # Reshape to (N, 1) for the output
        return (1 - np.abs(dot_products)).reshape(-1, 1)

    def quadruped_cost_np(self, x_robot, x_box, u, x_robot_ref, x_box_ref):
        kp = 50
        kd = 3

        # ROBOT COST
        # Compute the error terms
        x_error = x_robot - x_robot_ref

        quat_robot_dist = self.quaternion_distance_np(x_robot[:, 3:7], x_robot_ref[:, 3:7])
        x_error[:, 3] = quat_robot_dist
        x_error[:, 4] = 0
        x_error[:, 5] = 0
        x_error[:, 6] = 0

        x_joint = x_robot[:, 7:19]
        v_joint = x_robot[:, 25:]
        u_error = kp * (u - x_joint) - kd * v_joint

        # Compute cost using einsum for precise matrix operations
        # Apply the matrix Q to x_error and R to u_error, sum over appropriate dimensions
        x_error[:, :3] = 0
        x_pos_error = x_robot[:,:3] - x_robot_ref[:,:3]
        L1_norm_pos_cost = np.abs(np.dot(x_pos_error, self.Q_robot[:3,:3])).sum(axis=1)

        # BOX COST
        #quat_box_dist = self.quaternion_distance_1D_np(x_box[:, 3:7], x_box_ref[3:7])
        pos_error_box = x_box[:, :3] - x_box_ref[:3]
        #x_error_box = np.concatenate([pos_error_box, quat_box_dist], axis=1)
        box_cost = np.abs(np.dot(pos_error_box, self.Q_box[:3,:3])).sum(axis=1)

        robot_cost = np.einsum('ij,ik,jk->i', x_error, x_error, self.Q_robot) + np.einsum('ij,ik,jk->i', u_error, u_error, self.R) + L1_norm_pos_cost
        return robot_cost + box_cost

    def calculate_total_cost(self, robot_state, box_state, actions, joints_ref, body_ref):
        num_samples = robot_state.shape[0]
        num_pairs = robot_state.shape[1]

        traj_body_ref = np.repeat(body_ref[np.newaxis,:], num_samples*num_pairs, axis=0)

        # Flatten robot_state and actions to two dimensions, treating all pairs per sample as a batch
        robot_state = robot_state.reshape(-1, robot_state.shape[2])
        box_state = box_state.reshape(-1, box_state.shape[2])
        actions = actions.reshape(-1, actions.shape[2])

        joints_ref = joints_ref.T
        joints_ref = np.tile(joints_ref, (num_samples, 1, 1))
        joints_ref = joints_ref.reshape(-1, joints_ref.shape[2])

        x_robot_ref = np.concatenate([traj_body_ref[:,:7], joints_ref[:,:12],
                                traj_body_ref[:,7:], joints_ref[:,12:]],
                                axis=1)
        # Compute batch costs
        rotated_ref = batch_world_to_local_velocity(robot_state[:,3:7], robot_state[:,19:22])
        robot_state[:,19:22] = rotated_ref
        costs = self.quadruped_cost_np(robot_state, box_state, actions, x_robot_ref, self.x_box_ref)
        # Sum costs for each sample
        total_costs = costs.reshape(num_samples, num_pairs).sum(axis=1)
        return total_costs

