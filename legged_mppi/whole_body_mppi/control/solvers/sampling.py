"""Gaussian action sampling and iteration/horizon covariance annealing."""
import numpy as np
from scipy.interpolate import CubicSpline


class ActionSampling:
    def sample_delta_u(self):
        if self.sample_type == 'normal':
            size = (self.n_samples, self.horizon, self.act_dim)
            return self.generate_noise(size)
        elif self.sample_type == 'cubic':
            indices = np.arange(self.n_knots)*self.horizon//self.n_knots
            size = (self.n_samples, self.n_knots, self.act_dim)
            knot_points = self.generate_noise(size)
            cubic_spline = CubicSpline(indices, knot_points, axis=1)
            return cubic_spline(np.arange(self.horizon))

    def perturb_action(self, iteration=0, n_samples=None):
        count = self.n_samples if n_samples is None else n_samples
        if self.sample_type == 'normal':
            size = (count, self.horizon, self.act_dim)
            actions = self.trajectory + self.generate_noise(size, iteration)
            actions = np.clip(actions, self.act_min, self.act_max)
            return actions

        elif self.sample_type == 'cubic':
            indices_float = np.linspace(0, self.horizon - 1, num=self.n_knots)
            indices = np.round(indices_float).astype(int)
            size = (count, self.n_knots, self.act_dim)
            noise = self.generate_noise(size, iteration)
            knot_points = self.trajectory[indices] + noise
            #knot_points[:, 0, :] = self.trajectory[0]
            cubic_spline = CubicSpline(indices, knot_points, axis=1)
            actions = cubic_spline(np.arange(self.horizon))
            actions = np.clip(actions, self.act_min, self.act_max)
            return actions

    def generate_noise(self, size, iteration=0):
        """
        Generate noise for sampling actions.

        Args:
            size (tuple): Shape of the noise array.

        Returns:
            np.ndarray: Generated noise scaled by `noise_sigma`.
        """
        # Apply the square root of covariance annealing to standard deviation.
        # Scale knots before cubic interpolation.
        points = size[1]
        scale = np.exp(-0.5 * (
            iteration / (self.beta_iteration * self.n_iterations)
            + (points - np.arange(points)) / (self.beta_horizon * points)))
        return (self.random_generator.normal(size=size) * self.noise_sigma
                * scale[None, :, None])

