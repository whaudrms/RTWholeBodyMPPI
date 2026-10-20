"""Locomotion objective; task weights and reference semantics are unchanged."""
import numpy as np
from whole_body_mppi.utils.transforms import batch_world_to_local_velocity


class LocomotionCost:
    def quaternion_distance_np(self, q1, q2):
        """
        Compute the distance between two sets of quaternions.

        Args:
            q1 (np.ndarray): Array of quaternions (N x 4).
            q2 (np.ndarray): Array of quaternions (N x 4).

        Returns:
            np.ndarray: Array of distances between the quaternions.
        """
        # Compute dot product between corresponding quaternions
        dot_products = np.einsum('ij,ij->i', q1, q2)
        # Compute distance as 1 - absolute dot product
        return 1 - np.abs(dot_products)

    def quadruped_cost_np(self, x, u, x_ref):
        """
        Compute the cost for quadruped motion based on state and action errors.

        Args:
            x (np.ndarray): Current states (N x state_dim).
            u (np.ndarray): Current actions (N x action_dim).
            x_ref (np.ndarray): Reference states (N x state_dim).

        Returns:
            np.ndarray: Computed cost for each sample.
        """
        kp = 50  # Proportional gain for joint error
        kd = 3   # Derivative gain for joint velocity error

        # Compute state error relative to the reference
        x_error = x - x_ref

        # Compute quaternion distance for orientation error
        q_dist = self.quaternion_distance_np(x[:, 3:7], x_ref[:, 3:7])
        x_error[:, 3] = q_dist
        x_error[:, 4] = q_dist
        x_error[:, 5] = q_dist
        x_error[:, 6] = q_dist

        # Compute joint and velocity errors
        x_joint = x[:, 7:19]
        v_joint = x[:, 25:]
        u_error = kp * (u - x_joint) - kd * v_joint

        # Compute positional cost (L1 norm for positional error)
        x_error[:, :3] = 0  # Ignore positional error for simplicity
        x_pos_error = x[:, :3] - x_ref[:, :3]
        L1_norm_pos_cost = np.abs(np.dot(x_pos_error, self.Q[:3, :3])).sum(axis=1)

        # Compute total cost
        cost = (
            np.einsum('ij,ik,jk->i', x_error, x_error, self.Q) +
            np.einsum('ij,ik,jk->i', u_error, u_error, self.R) +
            L1_norm_pos_cost
        )
        return cost

    def calculate_total_cost(self, states, actions, joints_ref, body_ref):
        """
        Calculate the total cost for all rollouts.

        Args:
            states (np.ndarray): Rollout states (samples x time steps x state_dim).
            actions (np.ndarray): Rollout actions (samples x time steps x action_dim).
            joints_ref (np.ndarray): Reference joint positions (time steps x joint_dim).
            body_ref (np.ndarray): Reference body state (state_dim).

        Returns:
            np.ndarray: Total cost for each sample.
        """
        num_samples = states.shape[0]
        num_pairs = states.shape[1]

        # Repeat body reference for all samples and time steps
        traj_body_ref = np.repeat(body_ref[np.newaxis, :], num_samples * num_pairs, axis=0)

        # Flatten states and actions for batch processing
        states = states.reshape(-1, states.shape[2])
        actions = actions.reshape(-1, actions.shape[2])

        # Repeat and reshape joint references for batch processing
        joints_ref = joints_ref.T
        joints_ref = np.tile(joints_ref, (num_samples, 1, 1))
        joints_ref = joints_ref.reshape(-1, joints_ref.shape[2])

        # Concatenate body and joint references for full reference state
        x_ref = np.concatenate(
            [traj_body_ref[:, :7], joints_ref[:, :12], traj_body_ref[:, 7:], joints_ref[:, 12:]],
            axis=1
        )

        # Rotate velocity vectors to the local frame
        rotated_ref = batch_world_to_local_velocity(states[:, 3:7], states[:, 19:22])
        states[:, 19:22] = rotated_ref

        # Compute cost for each rollout
        costs = self.quadruped_cost_np(states, actions, x_ref)

        # Sum costs across time steps for each sample
        total_costs = costs.reshape(num_samples, num_pairs).sum(axis=1)
        return total_costs

