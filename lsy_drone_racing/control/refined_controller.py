"""Experimental TOPPRA retiming, obstacle clearance, and tracking-error governor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _TimingPath(ta.interpolator.AbstractGeometricPath):
    """Exact polynomial path augmented by clock and tilt-cone projections."""

    def __init__(self, curve: BPoly):
        self.curve = curve
        angles = np.arange(8) * np.pi / 4
        tangent = np.tan(np.deg2rad(45.190478572755005)) * np.cos(np.pi / 8)
        self.projection = np.c_[np.cos(angles), np.sin(angles), -np.full(8, tangent)]

    def __call__(self, coordinates: float | np.ndarray, order: int = 0) -> np.ndarray:
        coordinates = np.asarray(coordinates)
        xyz = self.curve.derivative(order)(coordinates)
        clock = coordinates if order == 0 else np.ones_like(coordinates) * (order == 1)
        return np.concatenate([xyz, clock[..., None], xyz @ self.projection.T], axis=-1)

    @property
    def dof(self) -> int:
        return 12

    @property
    def path_interval(self) -> np.ndarray:
        return self.curve.x[[0, -1]]


class RetimedController(Controller):
    """Experimental obstacle-aware trajectory; not the adopted race controller.

    Sampled clearance projection is a heuristic, not a continuous collision
    guarantee. It does not model gate frames or rotor thrust limits.
    """

    segment_scales = np.ones(13)
    acceleration_limit = 6.5014035883674275
    speed_limit = 3.220762476963001
    clock_limit = 2.576672575245518
    governor_gain = 4.322454373989269
    error_gain = 0.5223136598081135

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Create the Level 2 route using only configuration and permitted observations."""
        super().__init__(obs, info, config)
        self.dt = 1 / config.env.freq
        self.time = 0.0
        self.rate = 1.0
        self.rate_acc = 0.0
        self.rate_jerk = 0.0
        self.planner_failures = 0
        self.constraint_fallbacks = 0
        self.obstacles = np.asarray(obs["obstacles_pos"])[:, :2].copy()
        self.obstacles_seen = np.asarray(obs["obstacles_visited"], dtype=bool).copy()
        self.clearance = 0.18
        self.uncertainty = 0.14
        self._speed = 1.0
        self._feedforward = 1.0
        self._integral_gain = 3.0
        self._preview = 0.0
        self._rates = 1.0
        self._kp = 16.0
        self._kd = 5.0
        self.integral = np.zeros(3)
        self.times = np.array(
            [0, 0.85, 2.1, 2.8, 3.6, 4.3, 5.3, 5.8, 6.5, 7.0, 7.6, 8.35, 9.3, 9.9]
        )
        # Experiment-only per-segment duration multipliers. Keep the official
        # course and simulator settings unchanged.
        self.times = np.r_[0.0, np.cumsum(np.diff(self.times) * self.segment_scales)]
        self.positions = np.array(
            [
                obs["pos"],
                [-0.8, 0.45, 0.65],
                [0.5, 0.25, 0.7],
                [1.38, -0.18, 0.95],
                [1.05, 0.75, 1.2],
                [-0.05, 0.43, 1.05],
                [-1, -0.25, 0.7],
                [-1.23, -0.35, 0.85],
                [-1.1, -1.2, 1.2],
                [-0.4, -1.15, 1.2],
                [0, -0.75, 1.2],
                [1.2, -0.2, 1.2],
                [1.05, 0.75, 1.2],
                [0.6, 1.15, 1.2],
            ],
            dtype=float,
        )
        self.velocities = np.array(
            [
                [0, 0, 0],
                [1, 0.0, 0.5],
                [0.8, -0.8, 0],
                [0.35, 0.7, 0.25],
                [-0.8, 0.8, 0],
                [-1.4, -0.6, -0.4],
                [-0.8, 0, 0],
                [0, -0.65, 0.4],
                [1, 0, 0.1],
                [0.8, 0.3, 0],
                [1.1, 0, 0],
                [0.4, 0.8, 0],
                [-0.8, 0.8, 0],
                [0, 0, 0],
            ],
            dtype=float,
        )
        self.accelerations = np.zeros_like(self.positions)
        self.gate_nodes = [2, 4, 6, 10, 12]
        self.sequence = np.asarray(obs["gate_sequence"], dtype=int)
        if len(self.sequence) != len(self.gate_nodes):
            raise ValueError("This controller is designed for the five-passage Level 2 course")
        self.direction = np.asarray(obs["gate_sequence_direction"])
        self.seen = np.zeros(len(obs["gates_pos"]), dtype=bool)
        self._update_nodes(obs)
        self._build(0, None)

    def _update_nodes(self, obs: dict):
        """Place each gate knot at its observed center with normal crossing velocity."""
        for passage, node in enumerate(self.gate_nodes):
            gate = self.sequence[passage]
            normal = (
                Rotation.from_quat(obs["gates_quat"][gate]).apply([1, 0, 0])
                * self.direction[passage]
            )
            self.positions[node] = obs["gates_pos"][gate]
            self.velocities[node] = normal * (0.65 if passage == 2 else 1.05)
            if passage == 2:
                self.positions[node + 1] = self.positions[node] + normal * 0.23 + [0, -0.08, 0.15]

    def _build(self, time: float, state: list | None):
        """Fit a quintic route, minimizing sampled squared jerk over free derivatives.

        Gate positions and velocities are fixed; other positions may be shifted
        for clearance. Free velocities and interior accelerations minimize jerk.
        Replanning fixes the current
        position, velocity and acceleration so observation updates cannot jump them.
        All basis responses are evaluated in one batch to fit the 50 Hz budget.
        """
        indices = np.flatnonzero(self.times > time + 0.05)
        if state is None:
            indices = np.arange(len(self.times))
            times = self.times
            values = [
                [self.positions[i], self.velocities[i], self.accelerations[i]] for i in indices
            ]
        else:
            if not len(indices):
                return
            times = np.r_[time, self.times[indices]]
            values = [state] + [
                [self.positions[i], self.velocities[i], self.accelerations[i]] for i in indices
            ]
        values = np.asarray(values)
        free = []
        for k in range(1, len(times) - 1):
            original = indices[k] if state is None else indices[k - 1]
            if original not in self.gate_nodes:
                free.append((k, 1))
            free.append((k, 2))
        for k, derivative in free:
            values[k, derivative] = 0
        samples = np.concatenate(
            [np.linspace(a, b, 7)[1:-1] for a, b in zip(times[:-1], times[1:])]
        )
        weights = np.repeat(np.sqrt(np.diff(times)), 5)
        movable = [
            k
            for k in range(1, len(times) - 1)
            if (indices[k] if state is None else indices[k - 1]) not in self.gate_nodes
        ]
        # Solve minimum-jerk derivative responses for both the route and unit
        # waypoint displacements. This keeps gate centers/normals and the
        # current p/v/a exact while allowing smooth obstacle avoidance.
        outputs = 3 + len(movable)
        basis = np.zeros((len(times), 3, outputs + len(free)))
        basis[:, :, :3] = values
        for column, k in enumerate(movable):
            basis[k, 0, column + 3] = 1
        for column, (k, derivative) in enumerate(free):
            basis[k, derivative, column + outputs] = 1
        response = BPoly.from_derivatives(times, basis).derivative(3)(samples) * weights[:, None]
        target, matrix = response[:, :outputs], response[:, outputs:]
        fitted = basis[:, :, :outputs].copy()
        if free:
            solution = np.linalg.lstsq(matrix, -target, rcond=None)[0]
            for (k, derivative), value in zip(free, solution):
                fitted[k, derivative] = value
        # Keep gate accelerations from the unmodified route as well. This
        # isolates clearance corrections between gates so a late obstacle
        # cannot distort takeoff or the approach to an earlier gate.
        if movable:
            local_free = [(j, k, d) for j, (k, d) in enumerate(free) if k in movable]
            fitted[:, :, 3:] = basis[:, :, 3:outputs]
            columns = [j for j, _, _ in local_free]
            correction = np.linalg.lstsq(matrix[:, columns], -target[:, 3:], rcond=None)[0]
            for (_, k, derivative), value in zip(local_free, correction):
                fitted[k, derivative, 3:] = value
        family = BPoly.from_derivatives(times, fitted)
        shifts = np.zeros((len(movable), 2))
        if movable and len(self.obstacles):
            grid = np.linspace(times[0], times[-1], 240)
            paths = family(grid)
            influence = paths[:, 3:]
            norm2 = np.sum(influence**2, axis=1)
            radii = self.clearance + self.uncertainty * (~self.obstacles_seen)
            for _ in range(100):
                xy = paths[:, :2] + influence @ shifts
                delta = xy[:, None, :] - self.obstacles[None, :, :]
                distance = np.linalg.norm(delta, axis=2)
                violation = radii[None, :] - distance
                violation[norm2 < 0.03] = -1
                row, obstacle = np.unravel_index(np.argmax(violation), violation.shape)
                if violation[row, obstacle] <= 0.001:
                    break
                direction = delta[row, obstacle] / max(distance[row, obstacle], 1e-6)
                shifts += (0.7 * violation[row, obstacle] / norm2[row]) * (
                    influence[row, :, None] * direction
                )
                shifts = np.clip(shifts, -0.4, 0.4)
        coefficients = family.c[:, :, :3].copy()
        coefficients[:, :, :2] += family.c[:, :, 3:] @ shifts
        self.curve = BPoly(coefficients, times)
        self._plan_timing()

    def _plan_timing(self):
        """Optimize speed along the current path with TOPPRA's Seidel solver.

        These are translational proxy constraints, not a full rotor/attitude
        dynamics model. Runtime feedback and online observations can invalidate
        a plan; failures are recorded and use a bounded slowdown fallback.
        """
        path = _TimingPath(self.curve)
        grid = np.unique(np.r_[np.linspace(*path.path_interval, 180), self.curve.x])

        def velocity_limits(s: float) -> np.ndarray:
            cap = min(self.clock_limit, 1.0 + max(0.0, s - 0.6022783540711624) * 1.5955034469282965)
            cap = min(cap,
                self.clock_limit - max(0.0, self.clock_limit - 1.0) * np.exp(-(((s - 3.3727512571601324) / 0.45) ** 4)),
                self.clock_limit - max(0.0, self.clock_limit - 2.3456483667264827) * np.exp(-(((s - 5.444625406228474) / 0.65) ** 4)),
                self.clock_limit - max(0.0, self.clock_limit - 1.2314689617565773) * np.exp(-(((s - 7.317766989395945) / 0.35) ** 4)),
            )
            limits = np.r_[self.speed_limit, self.speed_limit, 1.8, cap, np.full(8, 1000.0)]
            return np.c_[-limits, limits]

        tilt_bound = np.tan(np.deg2rad(45.190478572755005)) * np.cos(np.pi / 8) * 9.81
        limits = np.array(
            [[-self.acceleration_limit, self.acceleration_limit]] * 2
            + [[-3.0, 4.0], [-1.2, 1.2]]
            + [[-1000.0, tilt_bound]] * 8
        )
        problem = algo.TOPPRA(
            [
                constraint.JointVelocityConstraintVarying(velocity_limits),
                constraint.JointAccelerationConstraint(limits),
            ],
            path,
            gridpoints=grid,
            solver_wrapper="seidel",
        )
        try:
            _, speed, _ = problem.compute_parameterization(self.rate, 1.0)
        except (RuntimeError, ValueError):
            speed = None
        self.timing_grid = grid
        if speed is None or not np.isfinite(speed).all():
            self.planner_failures += 1
            self.timing_speed = np.ones_like(grid)
        else:
            self.timing_speed = speed

    def _govern(
        self, obs: dict, pos: np.ndarray, first: np.ndarray, second: np.ndarray, third: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Smooth the path clock, account for tracking error, and limit commands."""
        target = float(np.interp(self.time + 0.25 * self.rate, self.timing_grid, self.timing_speed))
        passage = min(int(obs["n_gates_passed"]), len(self.sequence) - 1)
        gate = self.sequence[passage]
        if not bool(obs["gates_visited"][gate]):
            distance = np.linalg.norm(obs["gates_pos"][gate] - obs["pos"])
            if distance < 1.4:
                target = min(target, 1.838140604797607 / max(np.linalg.norm(first), 0.1))
        error = float(np.linalg.norm(pos - obs["pos"]))
        target /= 1.0 + self.error_gain * max(0.0, error - 0.04)
        requested = np.clip(self.governor_gain * (target - self.rate), -1.2, 1.2)
        # Bound clock jerk to avoid sudden changes in acceleration feedforward.
        chosen = np.clip(requested, self.rate_acc - 6.347144140814202 * self.dt, self.rate_acc + 6.347144140814202 * self.dt)
        chosen = np.clip(
            chosen, (0.25 - self.rate) / self.dt, (self.clock_limit - self.rate) / self.dt
        )
        self.rate_jerk = (chosen - self.rate_acc) / self.dt
        self.rate_acc = float(chosen)
        velocity = first * self.rate
        acceleration = second * self.rate**2 + first * self.rate_acc
        jerk = (
            third * self.rate**3 + 3 * second * self.rate * self.rate_acc + first * self.rate_jerk
        )
        return velocity, acceleration, jerk

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Track the smooth route with feedback, feedforward and bounded body rates."""
        time = min(self.time, self.times[-1])
        seen = np.asarray(obs["gates_visited"], dtype=bool)
        obstacles_seen = np.asarray(obs["obstacles_visited"], dtype=bool)
        if np.any(seen & ~self.seen) or np.any(obstacles_seen & ~self.obstacles_seen):
            self.obstacles = np.asarray(obs["obstacles_pos"])[:, :2].copy()
            self.obstacles_seen = obstacles_seen.copy()
            state = [
                self.curve(time),
                self.curve.derivative(1)(time),
                self.curve.derivative(2)(time),
            ]
            self._update_nodes(obs)
            self._build(time, state)
            self.seen = seen.copy()
        pos = self.curve(time)
        vel, acc, jerk = self._govern(
            obs,
            pos,
            self.curve.derivative(1)(time),
            self.curve.derivative(2)(time),
            self.curve.derivative(3)(time),
        )
        if self.time >= self.times[-1]:
            vel, acc, jerk = np.zeros(3), np.zeros(3), np.zeros(3)
        self.integral = np.clip(self.integral + (pos - obs["pos"]) * self.dt, -0.6, 0.6)
        acc += self._integral_gain * self.integral
        acc += self._kp * (pos - obs["pos"]) + self._kd * (vel - obs["vel"])
        acc += self._preview * jerk
        force = acc + [0, 0, 9.81]
        original_force = force.copy()
        force[2] = np.clip(force[2], 3.0, 14.0)
        lateral = np.linalg.norm(force[:2])
        allowed = force[2] * np.tan(np.deg2rad(50.921903802535454))
        if lateral > allowed:
            force[:2] *= allowed / lateral
        norm = np.linalg.norm(force)
        if norm > 17.36835732528887:
            force *= 17.36835732528887 / norm
        self.constraint_fallbacks += int(np.linalg.norm(force - original_force) > 1e-6)
        acc = force - [0, 0, 9.81]
        magnitude = max(float(np.linalg.norm(force)), 1e-6)
        z = force / magnitude
        y = np.cross(z, [1, 0, 0])
        y /= max(float(np.linalg.norm(y)), 1e-6)
        x = np.cross(y, z)
        rates = np.array([-jerk @ y, jerk @ x, 0]) / magnitude
        rates = np.clip(rates, -4, 4) * self._rates * min(1.0, time / 0.5)
        return np.r_[
            pos, vel * self._feedforward, acc * self._feedforward, [0, 0, 0, 1], rates
        ].astype(np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance simulation time; only the environment decides whether gates were passed."""
        self.time += self.dt * self.rate + 0.5 * self.rate_acc * self.dt**2
        self.rate += self.rate_acc * self.dt
        return int(obs["n_gates_passed"]) == len(self.sequence)
