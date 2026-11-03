"""Fast racing-line controller for Level 2 (team Barbie Drone).

The idea, in plain words:

1.  GEOMETRY. For every remaining gate we place three points on the gate's centre line (line-up,
    centre, exit) plus hand-designed "via points" that choose the side on which we pass each
    obstacle pole. Via points are defined *relative to the measured* poles/gates, so they move with
    the randomized track. A cubic spline through these points, parametrised by distance travelled
    (arc length), gives the racing line. Because the line is parametrised by distance and not by
    time, its shape does not depend on how fast we fly.

2.  SPEED PROFILE. Along the line we compute how fast we may fly: in a curve of radius r the drone
    needs v^2 / r of sideways acceleration, so v <= sqrt(A_LAT * r). A forward pass limits how
    quickly we can speed up (A_ACC), a backward pass how late we must brake (A_DEC).

3.  REFERENCE. Every step the reference point moves along the line at the profiled speed. If the
    drone falls behind (tracking error), the reference slows down so it never runs away from it.

4.  RE-PLANNING. Exact gate/pole positions are only revealed within 0.7 m. Whenever a measured
    position changes, the line is rebuilt starting at the current reference point, with the current
    reference direction and speed, so the command stays smooth.

5.  TRACKING. Our own position controller (PID + planned acceleration as feedforward) turns the
    reference into roll, pitch and collective thrust ("attitude" control mode).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


# Via points flown *before* lining up for the k-th gate of the race (gate_order = [1, 2, 3, 4, 2]).
# Each point is (anchor, index, dx, dy, z):
#   "pole":       dx/dy relative to the measured position of obstacle pole `index`,
#   "gate_local": dx/dy in the frame of gate `index` (dx = through the gate, dy = sideways),
#   z is the absolute flying height [m].
VIA_POINTS = [
    [],
    [("pole", 1, -0.15, -0.42, 0.8), ("pole", 1, 0.38, -0.12, 1.1)],
    [("gate_local", 0, -0.25, -0.32, 1.2)],
    [("pole", 2, 0.2, -0.45, 0.7), ("pole", 3, -0.1, -0.35, 1.1)],
    [("pole", 1, 0.38, -0.15, 1.2)],
]

# Angle [deg] between our straight line through each gate and the gate's centre line (race order).
APPROACH_ANGLES = [20.0, 20.0, 0.0, 35.0, 0.0]


# ---- Model predictive tracking controller (MPC) ----------------------------------------------
# The MPC predicts the drone 0.5 s ahead with the identified "so_rpy" model, which includes the
# lag of the drone's attitude response, and picks roll/pitch/thrust so the predicted path follows
# the racing line. Building the solver compiles C code, so we do it once and reuse it.
MPC_N = 25  # Prediction steps (of one control period each).
MPC_Q = [60.0, 60.0, 300.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0]  # pos, rpy, vel, drpy
MPC_R = [5.0, 5.0, 5.0, 20.0]  # Weights on deviating from the planned roll, pitch, yaw, thrust.
MPC_MAX_TILT = 0.9  # Max. roll/pitch [rad].
_MPC_CACHE: dict = {}


def _mpc_solver(dt: float, drone: str):  # noqa: ANN202
    """Create (or reuse) the acados solver for the tracking MPC."""
    key = (dt, drone, MPC_N, tuple(MPC_Q), tuple(MPC_R), MPC_MAX_TILT)
    if key in _MPC_CACHE:
        return _MPC_CACHE[key]
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    from crazyflow.dynamics.so_rpy import symbolic_dynamics_euler

    params = load_params(Dynamics.so_rpy, drone)
    x_dot, x, u, _ = symbolic_dynamics_euler(
        mass=params["mass"],
        gravity_vec=params["gravity_vec"],
        acc_coef=params["acc_coef"],
        cmd_f_coef=params["cmd_f_coef"],
        rpy_coef=params["rpy_coef"],
        rpy_rates_coef=params["rpy_rates_coef"],
        cmd_rpy_coef=params["cmd_rpy_coef"],
    )
    model = AcadosModel()
    model.name = "racing_line_mpc"
    model.f_expl_expr, model.x, model.u = x_dot, x, u
    ocp = AcadosOcp()
    ocp.model = model
    nx, nu = x.rows(), u.rows()
    ocp.solver_options.N_horizon = MPC_N
    ocp.cost.cost_type = ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(np.diag(MPC_Q), np.diag(MPC_R))
    ocp.cost.W_e = np.diag(MPC_Q)
    ocp.cost.Vx = np.vstack([np.eye(nx), np.zeros((nu, nx))])
    ocp.cost.Vu = np.vstack([np.zeros((nx, nu)), np.eye(nu)])
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref, ocp.cost.yref_e = np.zeros(nx + nu), np.zeros(nx)
    tilt = MPC_MAX_TILT
    ocp.constraints.lbu = np.array([-tilt, -tilt, -0.5, 4 * params["thrust_min"]])
    ocp.constraints.ubu = np.array([tilt, tilt, 0.5, 4 * params["thrust_max"]])
    ocp.constraints.idxbu = np.arange(4)
    ocp.constraints.x0 = np.zeros(nx)
    opts = ocp.solver_options
    opts.qp_solver = "FULL_CONDENSING_HPIPM"
    opts.hessian_approx = "GAUSS_NEWTON"
    opts.integrator_type = "ERK"
    opts.nlp_solver_type = "SQP"
    opts.nlp_solver_max_iter = 10
    opts.qp_solver_iter_max = 30
    opts.qp_solver_warm_start = 1
    opts.tf = MPC_N * dt
    out = os.environ.get("RACING_LINE_MPC_DIR", "c_generated_code/racing_line_mpc")
    ocp.code_export_directory = out
    solver = AcadosOcpSolver(ocp, json_file=f"{out}.json", verbose=False)
    _MPC_CACHE[key] = (solver, params)
    return solver, params


def _tilt_and_thrust(acc: np.ndarray, mass: float) -> tuple[np.ndarray, float]:
    """Roll/pitch/yaw (yaw = 0) and collective thrust that produce acceleration `acc`."""
    force = mass * (acc + np.array([0.0, 0.0, 9.81]))
    z_des = force / np.linalg.norm(force)
    y_des = np.cross(z_des, [1.0, 0.0, 0.0])
    y_des /= np.linalg.norm(y_des)
    x_des = np.cross(y_des, z_des)
    return R.from_matrix(np.vstack([x_des, y_des, z_des]).T).as_euler("xyz"), float(
        np.linalg.norm(force)
    )


class RacingLineController(Controller):
    """Arc-length racing line with a curvature-limited speed profile and a tracking governor."""

    # ---- Speed profile ----------------------------------------------------------------------
    V_MAX = 2.5  # Top speed [m/s].
    A_LAT = 6.0  # Max. sideways (centripetal) acceleration in curves [m/s^2].
    A_ACC = 5.0  # Max. forward acceleration [m/s^2].
    A_DEC = 5.0  # Max. braking [m/s^2].
    V_MIN = 0.3  # The reference always moves at least this fast (needed to start from rest).
    V_GATE = 10.0  # Max. speed while crossing a gate (between line-up and exit point) [m/s].
    # ---- Geometry ---------------------------------------------------------------------------
    APPROACH_DIST = 0.3  # Line-up point in front of the gate centre [m].
    EXIT_DIST = 0.3  # Exit point behind the gate centre [m].
    MIN_GATE_DIST = 0.15  # Line-up/exit points may move this close if a pole is in the way [m].
    POLE_CLEARANCE = 0.3  # Line-up/exit points must be at least this far from any pole [m].
    DS = 0.02  # Sampling distance of the racing line [m].
    # ---- Reference governor -----------------------------------------------------------------
    ERR_FREE = 0.05  # Tracking error [m] that is tolerated without slowing down.
    ERR_GAIN = 3.0  # How strongly the reference slows down for larger errors [1/m].
    # ---- Tracking controller (attitude mode) ------------------------------------------------
    TRACKER = "mpc"  # "pid" (simple, fast) or "mpc" (model predictive control, more precise).
    KP = np.array([12.0, 12.0, 20.0])
    KD = np.array([10.0, 10.0, 12.0])
    KI = np.array([1.0, 1.0, 4.0])
    FF_LEAD = 0.12  # Acceleration feedforward is taken this far ahead in time [s].
    ACC_FF = 1.0  # Fraction of the planned acceleration used as feedforward.
    MAX_TILT_DEG = 50.0
    STATE_KI = np.array([0.5, 0.5, 2.0])  # Integral correction in state mode [1/s].
    OFFSET_TAU = 0.25  # Time constant for fading out the reference jump after a re-plan [s].
    OFFSET_GATE_MARGIN = 0.3  # ... and it is completely gone this far before the gate centre [m].
    CHANGE_TOL = 1e-3  # Change in a measured position that triggers a re-plan [m].

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Set up the controller and plan the first racing line."""
        super().__init__(obs, info, config)
        self._dt = 1.0 / config.env.freq
        self._n_gates = len(obs["gate_sequence"])
        self._mass = load_params(config.sim.dynamics, config.sim.drone)["mass"]
        self._i_error = np.zeros(3)
        self._mode = config.env.control_mode  # "attitude" (own PID) or "state" (onboard control)
        self._finished = False
        self._start = obs["pos"].copy()
        self._s_ref = 0.0
        # After a re-plan the reference continues where it was and this offset fades out: over
        # time (OFFSET_TAU) and over distance, so it is gone shortly before the next gate centre.
        self._offset = np.zeros(3)
        self._off_s0, self._off_s1 = 0.0, 1.0
        self._plan(obs)
        if self.TRACKER == "mpc" and self._mode == "attitude":
            self._mpc, self._mpc_params = _mpc_solver(self._dt, config.sim.drone)
            self._mpc.reset()

    # ---- Waypoints --------------------------------------------------------------------------
    def _near_pole(self, p: np.ndarray, obs: dict) -> bool:
        d = np.linalg.norm(obs["obstacles_pos"][:, :2] - p[:2], axis=1)
        return bool(np.any(d < self.POLE_CLEARANCE))

    def _gate_geometry(self, obs: dict, k: int) -> tuple[np.ndarray, ...]:
        """Centre, crossing direction, line-up point and exit point of the k-th gate of the race."""
        gate_id = int(obs["gate_sequence"][k])
        direction = float(obs["gate_sequence_direction"][k])
        centre = obs["gates_pos"][gate_id]
        angle = np.deg2rad(APPROACH_ANGLES[k]) if k < len(APPROACH_ANGLES) else 0.0
        normal = R.from_quat(obs["gates_quat"][gate_id]).apply([np.cos(angle), np.sin(angle), 0])
        normal = normal * direction
        d_pre = self.APPROACH_DIST
        while d_pre > self.MIN_GATE_DIST and self._near_pole(centre - d_pre * normal, obs):
            d_pre -= 0.05
        d_post = self.EXIT_DIST
        while d_post > self.MIN_GATE_DIST and self._near_pole(centre + d_post * normal, obs):
            d_post -= 0.05
        return centre, normal, centre - d_pre * normal, centre + d_post * normal

    def _via_points(self, obs: dict, k: int) -> list[np.ndarray]:
        points = []
        for anchor, idx, dx, dy, z in VIA_POINTS[k] if k < len(VIA_POINTS) else []:
            if anchor == "pole":
                base, offset = obs["obstacles_pos"][idx], np.array([dx, dy, 0.0])
            else:
                base = obs["gates_pos"][idx]
                offset = R.from_quat(obs["gates_quat"][idx]).apply([dx, dy, 0.0])
            points.append(np.array([base[0] + offset[0], base[1] + offset[1], z]))
        return points

    def _waypoints(self, obs: dict) -> tuple[list[np.ndarray], list[int]]:
        """All waypoints of the lap; flag 1 = part of a gate passage, 2 = gate centre."""
        points, in_gate = [], []
        for k in range(self._n_gates):
            _, _, pre, post = self._gate_geometry(obs, k)
            centre = obs["gates_pos"][int(obs["gate_sequence"][k])]
            vias = self._via_points(obs, k)
            points += vias + [pre, centre, post]
            in_gate += [0] * len(vias) + [1, 2, 1]  # 2 marks the gate centre
        return points, in_gate

    # ---- Racing line + speed profile ---------------------------------------------------------
    def _plan(self, obs: dict):
        """(Re-)build the racing line of the whole lap (from the start position) and its speeds."""
        start = self._start
        points, in_gate = self._waypoints(obs)
        pts = np.vstack([start, *points])
        flags = np.array([0, *in_gate])
        # Remove (nearly) duplicate points, they break the spline.
        keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 0.02]
        pts, flags = pts[keep], flags[keep]
        if len(pts) < 2:
            pts = np.vstack([pts, pts[-1] + [0.0, 0.0, 0.05]])
            flags = np.r_[flags, 0]
        u = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        spline = CubicSpline(u, pts, bc_type="natural")

        # Dense samples and arc length.
        uu = np.linspace(0.0, u[-1], max(int(u[-1] / self.DS), 2) + 1)
        p, d1, d2 = spline(uu), spline(uu, 1), spline(uu, 2)
        speed_u = np.maximum(np.linalg.norm(d1, axis=1), 1e-6)
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        tangent = d1 / speed_u[:, None]
        # Curvature vector (points to the centre of the curve, length = 1 / radius).
        curv = (d2 - np.sum(d2 * tangent, axis=1)[:, None] * tangent) / speed_u[:, None] ** 2
        kappa = np.linalg.norm(curv, axis=1)

        v_lim = np.minimum(self.V_MAX, np.sqrt(self.A_LAT / np.maximum(kappa, 1e-6)))
        # Slow gate passages (between line-up and exit point) if requested.
        seg = np.clip(np.searchsorted(u, uu, side="right") - 1, 0, len(u) - 2)
        gate_seg = (flags[seg + 1] > 0) & (flags[seg] > 0)
        v_lim = np.where(gate_seg, np.minimum(v_lim, self.V_GATE), v_lim)
        v = v_lim.copy()
        v[0] = 0.0  # We start from rest.
        ds = np.diff(s)
        for i in range(len(v) - 1):  # Forward pass: acceleration limit.
            v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * self.A_ACC * ds[i]))
        for i in range(len(v) - 2, -1, -1):  # Backward pass: braking limit.
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * self.A_DEC * ds[i]))

        self._s, self._p, self._t, self._curv, self._v = s, p, tangent, curv, v
        self._gate_s = np.interp(u[flags == 2], uu, s)  # Path distance of each gate centre.
        self._dvds = np.gradient(v, s) if len(s) > 1 else np.zeros_like(v)
        self._known_gates = obs["gates_pos"].copy()
        self._known_obstacles = obs["obstacles_pos"].copy()

    def _ref(self, s_ref: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Reference position, velocity, acceleration and speed at path distance s_ref."""
        s = self._s
        i = int(np.clip(np.searchsorted(s, s_ref) - 1, 0, len(s) - 2))
        w = float(np.clip((s_ref - s[i]) / max(s[i + 1] - s[i], 1e-9), 0.0, 1.0))

        def lerp(a: np.ndarray) -> np.ndarray:
            return (1 - w) * a[i] + w * a[i + 1]

        pos, tangent, curv = lerp(self._p), lerp(self._t), lerp(self._curv)
        speed = float(lerp(self._v))
        tangent = tangent / max(np.linalg.norm(tangent), 1e-9)
        vel = speed * tangent
        acc = speed**2 * curv + speed * float(lerp(self._dvds)) * tangent
        return pos, vel, acc, speed

    def _needs_replan(self, obs: dict) -> bool:
        gates = np.max(np.abs(obs["gates_pos"] - self._known_gates)) > self.CHANGE_TOL
        poles = np.max(np.abs(obs["obstacles_pos"] - self._known_obstacles)) > self.CHANGE_TOL
        return bool(gates or poles)

    # ---- Control loop -----------------------------------------------------------------------
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Advance the reference along the racing line and track it."""
        if self._needs_replan(obs):
            old = self._ref(self._s_ref)[0] + self._offset
            self._plan(obs)
            # Continue at the point of the new line that is closest to the old reference point.
            near = np.abs(self._s - self._s_ref) < 0.6
            idx = np.flatnonzero(near)[np.argmin(np.linalg.norm(self._p[near] - old, axis=1))]
            self._s_ref = float(self._s[idx])
            self._offset = old - self._p[idx]
            nxt = self._gate_s[self._gate_s > self._s_ref + 0.05]
            s_gate = nxt[0] - self.OFFSET_GATE_MARGIN if len(nxt) else self._s[-1]
            self._off_s0, self._off_s1 = self._s_ref, max(s_gate, self._s_ref + 0.05)
        self._offset *= np.exp(-self._dt / self.OFFSET_TAU)

        pos, vel, acc, speed = self._ref(self._s_ref)
        pos = pos + self._offset_at(self._s_ref)
        err = float(np.linalg.norm(pos - obs["pos"]))
        slow = 1.0 / (1.0 + self.ERR_GAIN * max(0.0, err - self.ERR_FREE))
        self._s_ref = min(self._s_ref + max(speed, self.V_MIN) * slow * self._dt, self._s[-1])
        # Command the reference at the *new* point: the drone needs one step to react.
        if self.TRACKER == "mpc" and self._mode == "attitude":
            return self._mpc_control(obs, slow)
        pos, vel, acc, _ = self._ref(self._s_ref)
        if self.FF_LEAD > 0:  # The drone needs time to tilt: use the acceleration from ahead.
            acc = self._ref(min(self._s_ref + speed * self.FF_LEAD, self._s[-1]))[2]
        pos = pos + self._offset_at(self._s_ref)
        if self._mode == "state":
            return self._state_command(obs, pos, vel * slow, acc * slow**2)
        return self._attitude_control(obs, pos, vel * slow, acc * slow**2)

    def _offset_at(self, s: float, offset: np.ndarray | None = None) -> np.ndarray:
        """Re-plan offset at path distance s (fades linearly to zero before the next gate)."""
        offset = self._offset if offset is None else offset
        f = np.clip((self._off_s1 - s) / (self._off_s1 - self._off_s0), 0.0, 1.0)
        return offset * f

    def _mpc_control(self, obs: dict, slow: float) -> NDArray[np.floating]:
        """Track the next 0.5 s of the racing line with the MPC."""
        solver, mass = self._mpc, float(self._mpc_params["mass"])
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], rpy, obs["vel"], drpy))
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)
        s, offset = self._s_ref, self._offset.copy()
        decay = np.exp(-self._dt / self.OFFSET_TAU)
        for j in range(MPC_N + 1):
            pos, vel, acc, speed = self._ref(s)
            ref_rpy, thrust = _tilt_and_thrust(acc * slow**2, mass)
            y = np.concatenate((pos + self._offset_at(s, offset), ref_rpy, vel * slow, np.zeros(3)))
            if j < MPC_N:
                solver.set(j, "yref", np.concatenate((y, ref_rpy, [thrust])))
            else:
                solver.set(j, "yref", y)
            s = min(s + max(speed, self.V_MIN) * slow * self._dt, self._s[-1])
            offset = offset * decay
        solver.solve()
        return np.asarray(solver.get(0, "u"), dtype=np.float32)

    def _state_command(
        self, obs: dict, des_pos: np.ndarray, des_vel: np.ndarray, des_acc: np.ndarray
    ) -> NDArray[np.floating]:
        """Full-state setpoint for the onboard controller, with an integral correction.

        The onboard controller has small steady-state errors (e.g. it hovers a few cm too high),
        so we integrate the position error and shift the commanded position by it.
        """
        self._i_error = np.clip(
            self._i_error + self.STATE_KI * (des_pos - obs["pos"]) * self._dt, -0.3, 0.3
        )
        cmd_pos = des_pos + self._i_error
        quat = np.array([0.0, 0.0, 0.0, 1.0])  # yaw = 0
        return np.concatenate(
            (cmd_pos, des_vel, self.ACC_FF * des_acc, quat, np.zeros(3)), dtype=np.float32
        )

    def _attitude_control(
        self, obs: dict, des_pos: np.ndarray, des_vel: np.ndarray, des_acc: np.ndarray
    ) -> NDArray[np.floating]:
        """PID position controller with acceleration feedforward -> [roll, pitch, yaw, thrust]."""
        pos_err = des_pos - obs["pos"]
        vel_err = des_vel - obs["vel"]
        self._i_error = np.clip(self._i_error + pos_err * self._dt, -0.5, 0.5)
        acc = self.ACC_FF * des_acc + self.KP * pos_err + self.KD * vel_err + self.KI * self._i_error
        force = self._mass * (acc + np.array([0.0, 0.0, 9.81]))
        horiz = np.linalg.norm(force[:2])
        max_horiz = np.tan(np.deg2rad(self.MAX_TILT_DEG)) * force[2]
        if horiz > max_horiz > 0:
            force[:2] *= max_horiz / horiz
        z_body = R.from_quat(obs["quat"]).as_matrix()[:, 2]
        thrust = float(force @ z_body)
        z_des = force / np.linalg.norm(force)
        y_des = np.cross(z_des, [1.0, 0.0, 0.0])
        y_des /= np.linalg.norm(y_des)
        x_des = np.cross(y_des, z_des)
        rpy = R.from_matrix(np.vstack([x_des, y_des, z_des]).T).as_euler("xyz")
        return np.concatenate([rpy, [thrust]], dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Stop once every gate has been passed."""
        self._finished = int(obs["n_gates_passed"]) >= self._n_gates
        return self._finished

    def episode_callback(self):
        """Nothing to reset between episodes (a new instance is created)."""

    def render_callback(self, sim: Sim):
        """Draw the racing line (green) and the current reference point (red)."""
        from crazyflow.sim.visualize import draw_line, draw_points

        draw_points(sim, self._ref(self._s_ref)[0].reshape(1, -1), rgba=(1, 0, 0, 1), size=0.02)
        draw_line(sim, self._p[:: max(len(self._p) // 100, 1)], rgba=(0, 1, 0, 1))
