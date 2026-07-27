"""Cross-entropy optimization for low-dimensional base-pose references."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from mani_mppi.control.collision import ArmTorsoCollision
from mani_mppi.control.kinematics import ArmKinematics


@dataclass(frozen=True)
class CEMResult:
    """Best sample retained during a complete CEM solve."""

    solution: np.ndarray
    cost: float
    metrics: dict[str, float | bool]
    evaluations: int
    iterations: int


@dataclass(frozen=True)
class BasePosePlan:
    """Absolute base reference and the CEM diagnostics that selected it."""

    base_xy: np.ndarray
    base_height: float
    cem_result: CEMResult


EvaluationFunction = Callable[
    [np.ndarray],
    tuple[np.ndarray, Mapping[str, np.ndarray]],
]


class CrossEntropyOptimizer:
    """Bounded CEM that never re-evaluates its retained best sample."""

    def __init__(
        self,
        *,
        lower: np.ndarray,
        upper: np.ndarray,
        num_samples: int,
        num_iterations: int,
        elite_fraction: float,
        smoothing: float,
        min_std: np.ndarray,
        seed: int,
    ) -> None:
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.min_std = np.asarray(min_std, dtype=float)
        if (
            self.lower.ndim != 1
            or self.upper.shape != self.lower.shape
            or self.min_std.shape != self.lower.shape
        ):
            raise ValueError("CEM bounds and min_std must have matching shapes")
        if (
            not np.isfinite(self.lower).all()
            or not np.isfinite(self.upper).all()
            or not np.isfinite(self.min_std).all()
            or np.any(self.upper <= self.lower)
            or np.any(self.min_std <= 0.0)
        ):
            raise ValueError("CEM bounds and min_std must be finite and valid")

        self.num_samples = int(num_samples)
        self.num_iterations = int(num_iterations)
        self.elite_fraction = float(elite_fraction)
        self.smoothing = float(smoothing)
        if self.num_samples < 4 or self.num_iterations < 1:
            raise ValueError("CEM requires at least four samples and one iteration")
        if not 0.0 < self.elite_fraction <= 0.5:
            raise ValueError("CEM elite_fraction must be in (0, 0.5]")
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError("CEM smoothing must be in [0, 1)")
        self.num_elites = max(
            2,
            int(np.ceil(self.elite_fraction * self.num_samples)),
        )
        self.random_generator = np.random.default_rng(int(seed))

    def optimize(
        self,
        *,
        mean: np.ndarray,
        std: np.ndarray,
        evaluate: EvaluationFunction,
        project: Callable[[np.ndarray], np.ndarray] | None = None,
        seeds: np.ndarray | None = None,
    ) -> CEMResult:
        """Optimize one distribution and retain evaluated diagnostics.

        ``evaluate`` is invoked exactly once per iteration. The winning
        candidate and its metrics are saved when they are first evaluated, so
        there is no redundant final objective call.
        """
        mean = np.asarray(mean, dtype=float).copy()
        std = np.asarray(std, dtype=float).copy()
        dimension = len(self.lower)
        if mean.shape != (dimension,) or std.shape != (dimension,):
            raise ValueError("CEM mean/std must match the optimizer dimension")
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("CEM mean/std must be finite")
        if np.any(std <= 0.0):
            raise ValueError("CEM std must be positive")

        if seeds is None:
            seeds = np.empty((0, dimension), dtype=float)
        else:
            seeds = np.asarray(seeds, dtype=float)
            if seeds.ndim != 2 or seeds.shape[1] != dimension:
                raise ValueError("CEM seeds must have shape (N, dimension)")
            if not np.isfinite(seeds).all():
                raise ValueError("CEM seeds must be finite")

        best_solution = None
        best_cost = np.inf
        best_metrics: dict[str, float | bool] = {}
        evaluations = 0

        for iteration in range(self.num_iterations):
            samples = self.random_generator.normal(
                mean,
                std,
                size=(self.num_samples, dimension),
            )
            samples = np.clip(samples, self.lower, self.upper)
            samples[0] = np.clip(mean, self.lower, self.upper)
            if iteration == 0 and len(seeds):
                seed_count = min(len(seeds), self.num_samples - 1)
                samples[1:1 + seed_count] = np.clip(
                    seeds[:seed_count],
                    self.lower,
                    self.upper,
                )
            if project is not None:
                samples = np.asarray(project(samples), dtype=float)
                if samples.shape != (self.num_samples, dimension):
                    raise ValueError("CEM projection changed the sample shape")

            costs, metrics = evaluate(samples)
            costs = np.asarray(costs, dtype=float)
            if costs.shape != (self.num_samples,):
                raise ValueError("CEM evaluator must return one cost per sample")
            costs = np.where(np.isfinite(costs), costs, np.inf)
            normalized_metrics: dict[str, np.ndarray] = {}
            for name, values in metrics.items():
                values = np.asarray(values)
                if values.shape != (self.num_samples,):
                    raise ValueError(
                        f"CEM metric '{name}' must contain one value per sample"
                    )
                normalized_metrics[name] = values

            evaluations += self.num_samples
            iteration_best = int(np.argmin(costs))
            if costs[iteration_best] < best_cost:
                best_cost = float(costs[iteration_best])
                best_solution = samples[iteration_best].copy()
                best_metrics = {
                    name: (
                        bool(values[iteration_best])
                        if values.dtype == np.bool_
                        else float(values[iteration_best])
                    )
                    for name, values in normalized_metrics.items()
                }

            elite_indices = np.argpartition(
                costs,
                self.num_elites - 1,
            )[:self.num_elites]
            elites = samples[elite_indices]
            elite_mean = np.mean(elites, axis=0)
            elite_std = np.std(elites, axis=0)
            mean = (
                self.smoothing * mean
                + (1.0 - self.smoothing) * elite_mean
            )
            std = np.maximum(
                self.smoothing * std
                + (1.0 - self.smoothing) * elite_std,
                self.min_std,
            )

        if best_solution is None or not np.isfinite(best_cost):
            raise RuntimeError("CEM failed to produce a finite candidate")
        return CEMResult(
            solution=best_solution,
            cost=best_cost,
            metrics=best_metrics,
            evaluations=evaluations,
            iterations=self.num_iterations,
        )


class KinematicBasePoseCEMPlanner:
    """Private-model CEM planner safe to execute outside the control thread.

    The planner owns its MuJoCo model, IK data, collision helper, and random
    generator. It performs no dynamic rollout and never accesses controller
    model/data, so the main MPPI loop can continue concurrently.
    """

    def __init__(
        self,
        model_path: str | Path,
        config: dict,
        *,
        ee_site_name: str,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.arm_kinematics = ArmKinematics(
            self.model,
            config,
            ee_site_name=ee_site_name,
        )
        self.arm_collision = ArmTorsoCollision(self.model, config)

        self.stand_base_height = float(
            config.get("stand_base_height", 0.543542)
        )
        self.min_base_height = float(config.get("min_base_height", 0.35))
        self.residual_threshold = float(
            config.get("reachability_residual_threshold", 0.03)
        )
        self.max_xy_shift = float(
            config.get("reachability_max_xy_shift", 0.40)
        )
        self.ik_iterations = int(
            config.get("reachability_ik_iterations", 40)
        )
        self.initial_xy_std = float(
            config.get("cem_initial_xy_std", 0.20)
        )
        self.initial_height_std = float(
            config.get("cem_initial_height_std", 0.08)
        )
        self.infeasible_penalty = float(
            config.get("cem_infeasible_penalty", 1000.0)
        )
        self.reachability_weight = float(
            config.get("cem_reachability_weight", 100.0)
        )
        self.residual_weight = float(
            config.get("cem_residual_weight", 1.0)
        )
        self.xy_weight = float(config.get("cem_xy_weight", 6.0))
        self.height_weight = float(config.get("cem_height_weight", 3.0))
        self.clearance_weight = float(
            config.get("cem_clearance_weight", 2.0)
        )
        self.joint_limit_weight = float(
            config.get("cem_joint_limit_weight", 1.0)
        )
        self.joint_limit_margin = float(
            config.get("cem_joint_limit_margin", 0.10)
        )
        self.collision_penalty = float(
            config.get("cem_collision_penalty", 1e6)
        )
        self._validate_config()

        self.optimizer = CrossEntropyOptimizer(
            lower=np.array(
                [
                    -self.max_xy_shift,
                    -self.max_xy_shift,
                    self.min_base_height,
                ]
            ),
            upper=np.array(
                [
                    self.max_xy_shift,
                    self.max_xy_shift,
                    self.stand_base_height,
                ]
            ),
            num_samples=int(config.get("cem_num_samples", 48)),
            num_iterations=int(config.get("cem_num_iterations", 3)),
            elite_fraction=float(config.get("cem_elite_fraction", 0.20)),
            smoothing=float(config.get("cem_smoothing", 0.15)),
            min_std=np.array(
                [
                    float(config.get("cem_min_xy_std", 0.01)),
                    float(config.get("cem_min_xy_std", 0.01)),
                    float(config.get("cem_min_height_std", 0.005)),
                ]
            ),
            seed=int(config.get("cem_seed", config.get("seed", 42) + 1000)),
        )

    def _validate_config(self) -> None:
        if (
            not 0.0 < self.min_base_height < self.stand_base_height
            or self.residual_threshold <= 0.0
            or self.max_xy_shift <= 0.0
            or self.ik_iterations < 1
            or self.initial_xy_std <= 0.0
            or self.initial_height_std <= 0.0
            or not 0.0 < self.joint_limit_margin < 0.5
        ):
            raise ValueError("Invalid kinematic CEM planner configuration")
        weights = np.asarray(
            [
                self.infeasible_penalty,
                self.reachability_weight,
                self.residual_weight,
                self.xy_weight,
                self.height_weight,
                self.clearance_weight,
                self.joint_limit_weight,
                self.collision_penalty,
            ],
            dtype=float,
        )
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("CEM cost weights must be finite and non-negative")

    def _project_samples(self, samples: np.ndarray) -> np.ndarray:
        projected = np.asarray(samples, dtype=float).copy()
        xy = projected[:, :2]
        radii = np.linalg.norm(xy, axis=1)
        outside = radii > self.max_xy_shift
        if np.any(outside):
            xy[outside] *= (
                self.max_xy_shift / radii[outside]
            )[:, None]
        projected[:, 2] = np.clip(
            projected[:, 2],
            self.min_base_height,
            self.stand_base_height,
        )
        return projected

    def _seed_candidates(
        self,
        current_xy: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        direction = np.asarray(target[:2], dtype=float) - current_xy
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-9:
            direction /= direction_norm
        else:
            direction = np.array([-1.0, 0.0])

        height_span = self.stand_base_height - self.min_base_height
        heights = (
            self.stand_base_height,
            self.stand_base_height - 0.50 * height_span,
            self.stand_base_height - 0.75 * height_span,
        )
        distances = (
            0.0,
            0.375 * self.max_xy_shift,
            0.75 * self.max_xy_shift,
        )
        return np.asarray(
            [
                [*(distance * direction), height]
                for height in heights
                for distance in distances
            ],
            dtype=float,
        )

    def _evaluate_candidates(
        self,
        observation: np.ndarray,
        target: np.ndarray,
        current_xy: np.ndarray,
        candidates: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        candidates = np.asarray(candidates, dtype=float)
        num_candidates = len(candidates)
        states = np.zeros(
            (num_candidates, self.model.nq + self.model.nv),
            dtype=float,
        )
        arm_positions = np.empty((num_candidates, 4, 3), dtype=float)
        arm_q_samples = np.empty((num_candidates, 4), dtype=float)
        residuals = np.empty(num_candidates, dtype=float)
        initial_qpos = np.asarray(
            observation[:self.model.nq],
            dtype=float,
        )

        for index, candidate in enumerate(candidates):
            qpos = initial_qpos.copy()
            qpos[:2] = current_xy + candidate[:2]
            qpos[2] = candidate[2]
            qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
            arm_q = self.arm_kinematics.solve_ik(
                target,
                qpos,
                max_iterations=self.ik_iterations,
                strict=False,
            )
            residuals[index] = self.arm_kinematics.last_ik_residual
            arm_q_samples[index] = arm_q
            qpos[self.arm_kinematics.arm_qpos_indices] = arm_q
            states[index, :self.model.nq] = qpos

            # solve_ik already ended with mj_forward for this candidate.
            data = self.arm_kinematics.ik_data
            arm_positions[index, :3] = data.xpos[
                self.arm_kinematics.arm_fk_body_ids
            ]
            arm_positions[index, 3] = data.site_xpos[
                self.arm_kinematics.ee_site_id
            ]

        if self.arm_collision.enabled:
            collision_result = self.arm_collision.evaluate(
                states,
                arm_positions,
            )
            clearances = collision_result.clearance
            collision_valid = np.all(
                collision_result.segment_valid,
                axis=1,
            )
            minimum_clearance = np.min(clearances, axis=1)
            clearance_scale = max(self.arm_collision.safe_distance, 1e-6)
            clearance_deficit = np.maximum(
                self.arm_collision.safe_distance - clearances,
                0.0,
            ) / clearance_scale
            clearance_cost = self.clearance_weight * np.sum(
                clearance_deficit**2,
                axis=1,
            )
        else:
            collision_valid = np.ones(num_candidates, dtype=bool)
            minimum_clearance = np.full(num_candidates, np.inf)
            clearance_cost = np.zeros(num_candidates, dtype=float)

        arm_range = (
            self.arm_kinematics.arm_joint_upper
            - self.arm_kinematics.arm_joint_lower
        )
        lower_margin = (
            arm_q_samples - self.arm_kinematics.arm_joint_lower
        ) / arm_range
        upper_margin = (
            self.arm_kinematics.arm_joint_upper - arm_q_samples
        ) / arm_range
        joint_margin = np.min(
            np.minimum(lower_margin, upper_margin),
            axis=1,
        )
        joint_margin_deficit = np.maximum(
            self.joint_limit_margin - joint_margin,
            0.0,
        ) / self.joint_limit_margin

        residual_ratio = residuals / self.residual_threshold
        reachability_violation = np.maximum(residual_ratio - 1.0, 0.0)
        xy_ratio = (
            np.linalg.norm(candidates[:, :2], axis=1)
            / self.max_xy_shift
        )
        height_ratio = (
            (self.stand_base_height - candidates[:, 2])
            / (self.stand_base_height - self.min_base_height)
        )
        feasible = (
            collision_valid
            & (residuals <= self.residual_threshold)
        )
        costs = (
            self.residual_weight * residual_ratio**2
            + self.reachability_weight * reachability_violation**2
            + self.infeasible_penalty * (~feasible)
            + self.xy_weight * xy_ratio**2
            + self.height_weight * height_ratio**2
            + clearance_cost
            + self.joint_limit_weight * joint_margin_deficit**2
            + self.collision_penalty * (~collision_valid)
        )
        return costs, {
            "residual": residuals,
            "collision_valid": collision_valid,
            "minimum_clearance": minimum_clearance,
            "joint_margin": joint_margin,
            "feasible": feasible,
        }

    def plan(
        self,
        observation: np.ndarray,
        target: np.ndarray,
    ) -> BasePosePlan:
        """Return an absolute base plan using only private kinematic state."""
        observation = np.asarray(observation, dtype=float).copy()
        target = np.asarray(target, dtype=float).copy()
        expected_observation = self.model.nq + self.model.nv
        if observation.shape != (expected_observation,):
            raise ValueError(
                "CEM observation must have shape "
                f"({expected_observation},), got {observation.shape}"
            )
        if target.shape != (3,):
            raise ValueError("CEM target must have shape (3,)")

        current_xy = observation[:2].copy()
        result = self.optimizer.optimize(
            mean=np.array([0.0, 0.0, self.stand_base_height]),
            std=np.array(
                [
                    self.initial_xy_std,
                    self.initial_xy_std,
                    self.initial_height_std,
                ]
            ),
            evaluate=lambda candidates: self._evaluate_candidates(
                observation,
                target,
                current_xy,
                candidates,
            ),
            project=self._project_samples,
            seeds=self._seed_candidates(current_xy, target),
        )
        return BasePosePlan(
            base_xy=current_xy + result.solution[:2],
            base_height=float(result.solution[2]),
            cem_result=result,
        )
