"""Racing-line controller for Level 2 (team Barbie Drone).

How it works, in plain words:

1.  WAYPOINTS. For every gate we still have to pass, we put three waypoints on the gate's centre
    line: a "line-up" point in front of the gate, the gate centre, and an "exit" point behind it.
    This makes the drone fly straight through the 40 cm opening instead of clipping the frame.

2.  RACING LINE. Between two gates we add a few "via points" that decide on which side of each
    obstacle pole we fly (see VIA_POINTS below). They were designed by hand for this track,
    but they are defined *relative to the measured pole/gate positions*, so they move along
    when the track is randomized.

3.  SMOOTH PATH. All waypoints are connected by a smooth curve (cubic spline). The time between
    two waypoints is their distance divided by a target speed, so the curve also tells us
    *when* the drone should be where. Each control step we send the drone the position,
    velocity and acceleration of the curve at the current time.

4.  RE-PLANNING. In Level 2 the gates and poles are moved randomly. The drone only sees their
    exact position when it gets within 0.7 m. Whenever a measured position changes, we build a
    new curve from where we currently are, through the updated waypoints.

5.  HEIGHT CORRECTION. In the simulator the drone hovers ~10 cm higher than commanded. We
    measure this error while flying and subtract it from the command (integral control).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.dynamics import load_params
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


# Via points for the racing line, one list per gate in the race order (gate_order = [1, 2, 3, 4, 2]
# in level2.toml). Entry k lists the points flown *before* lining up for the k-th gate.
# Each point is (anchor, index, dx, dy, z):
#   anchor "pole": dx/dy are relative to the measured position of obstacle pole `index`,
#   anchor "gate_local": dx/dy are in the frame of gate `index` (dx = through the gate,
#                        dy = sideways), so the point moves and rotates with the gate,
#   z is the absolute flying height [m].
VIA_POINTS = [
    # Start -> gate 1: nothing needed (we come in at an angle that passes pole 1).
    [],
    # Gate 1 -> gate 2: gate 2 is right next to gate 1 and pole 2 is in between,
    # so we go around pole 2 on its south/east side and approach gate 2 from the south-east.
    [("pole", 1, 0.0, -0.38, 0.85), ("pole", 1, 0.38, -0.12, 1.1)],
    # Gate 2 -> gate 3: stay high (above gate 1's frame), then descend towards gate 3.
    # (0.25 m in front of gate 1 and 0.32 m to its side, at 1.2 m height.)
    [("gate_local", 0, -0.25, -0.32, 1.2)],
    # Gate 3 -> gate 4: pole 3 is only 0.5 m behind gate 3, so we turn left right behind the
    # gate (but clear of its frame), pass pole 4 on its south side, then come in at an angle.
    [("gate_local", 2, 0.3, 0.55, 0.7), ("pole", 3, -0.1, -0.35, 1.1)],
    # Gate 4 -> gate 2 (second time): pass pole 2 on its east side again.
    [("pole", 1, 0.38, -0.15, 1.2)],
]


# Angle [deg] between our straight flight line through each gate and the gate's centre line, one
# per gate in race order. Positive = we come in from the gate's local -y side (its "right").
# Used where a pole stands on the centre line, so the straight approach can pass beside it.
APPROACH_ANGLES = [30.0, 20.0, 0.0, 35.0, 0.0]


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two vectors in radians (0 = same direction)."""
    cos = np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9)
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


class GateSplineController(Controller):
    """Follows a smooth racing line through the measured gates and re-plans when they move."""

    # ---- Tuning knobs -------------------------------------------------------
    SPEED = 1.3  # Flight speed between gates [m/s]. Lower = safer, higher = faster.
    GATE_SPEED = 0.8  # Speed from the line-up point, through the gate, to the exit point [m/s].
    APPROACH_DIST = 0.45  # Line-up point: this far in front of the gate centre [m].
    EXIT_DIST = 0.3  # Exit point: this far behind the gate centre [m].
    MIN_GATE_DIST = 0.15  # If a pole is in the way, line-up/exit points move closer, down to this.
    POLE_CLEARANCE = 0.3  # Line-up/exit points must be at least this far from any pole [m].
    TURN_SLOWDOWN = 0.5  # Speed factor for a full U-turn (1.0 = no slow-down in corners).
    ACC_MAX = 100.0  # Max. acceleration the planned path may ask for [m/s^2] (100 = off).
    MIN_SEG_TIME = 0.1  # Minimum time between two waypoints [s].
    Z_BIAS_INIT = 0.08  # Initial guess of the simulator's height error (from hover tests) [m].
    Z_KI = 3.0  # Gain of the height correction [1/s]. Higher = corrects faster.
    ACC_FF = 1.0  # How much of the planned acceleration we pass on to the drone (0..1).
    # Gains of our own position controller (attitude mode only), in m/s^2 per m and per m/s.
    KP = np.array([12.0, 12.0, 20.0])
    KD = np.array([6.0, 6.0, 8.0])
    KI = np.array([1.0, 1.0, 4.0])
    MAX_TILT_DEG = 50.0  # Maximum tilt angle we command [deg].
    CHANGE_TOL = 1e-3  # How much a measured position must change to trigger a re-plan [m].

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Set up the controller and plan the first path."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._tick = 0
        self._finished = False
        self._n_gates_total = len(obs["gate_sequence"])
        self._z_bias = self.Z_BIAS_INIT
        self._last_des_z = None
        # "state": the drone's built-in controller follows our setpoints.
        # "attitude": our own position controller computes tilt + thrust (more precise when fast).
        self._mode = config.env.control_mode
        self._mass = load_params(config.sim.dynamics, config.sim.drone)["mass"]
        self._i_error = np.zeros(3)
        self._plan(obs, start_pos=obs["pos"], start_vel=np.zeros(3))

    # ---- Waypoints -------------------------------------------------------------
    def _near_pole(self, p: np.ndarray, obs: dict) -> bool:
        """True if point p is closer than POLE_CLEARANCE (horizontally) to any obstacle pole."""
        d = np.linalg.norm(obs["obstacles_pos"][:, :2] - p[:2], axis=1)
        return bool(np.any(d < self.POLE_CLEARANCE))

    def _gate_geometry(
        self, obs: dict, k: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Centre, crossing direction, line-up point and exit point of the k-th gate of the race."""
        gate_id = int(obs["gate_sequence"][k])
        direction = float(obs["gate_sequence_direction"][k])  # +1 forward, -1 reverse
        centre = obs["gates_pos"][gate_id]
        # Gates must be crossed along their local x-axis ("normal"). We may cross at an angle
        # (APPROACH_ANGLES): what matters is only that the drone is centred when it crosses.
        rot = R.from_quat(obs["gates_quat"][gate_id])
        angle = np.deg2rad(APPROACH_ANGLES[k]) if k < len(APPROACH_ANGLES) else 0.0
        normal = rot.apply([np.cos(angle), np.sin(angle), 0.0]) * direction
        # If a pole stands where the line-up/exit point would be, move the point closer.
        d_pre = self.APPROACH_DIST
        while d_pre > self.MIN_GATE_DIST and self._near_pole(centre - d_pre * normal, obs):
            d_pre -= 0.05
        d_post = self.EXIT_DIST
        while d_post > self.MIN_GATE_DIST and self._near_pole(centre + d_post * normal, obs):
            d_post -= 0.05
        return centre, normal, centre - d_pre * normal, centre + d_post * normal

    def _via_points(self, obs: dict, k: int) -> list[np.ndarray]:
        """Racing-line via points before the k-th gate, placed relative to measured objects."""
        points = []
        for anchor, idx, dx, dy, z in VIA_POINTS[k] if k < len(VIA_POINTS) else []:
            if anchor == "pole":
                offset = np.array([dx, dy, 0.0])
                base = obs["obstacles_pos"][idx]
            else:  # "gate_local": offset in the gate's own frame (rotates with the gate)
                offset = R.from_quat(obs["gates_quat"][idx]).apply([dx, dy, 0.0])
                base = obs["gates_pos"][idx]
            points.append(np.array([base[0] + offset[0], base[1] + offset[1], z]))
        return points

    def _waypoints(self, obs: dict, start_pos: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        """All waypoints from here to the finish, and the speed to fly *to* each of them."""
        waypoints, speeds = [], []
        n_passed = int(obs["n_gates_passed"])

        # If we just flew through a gate and haven't reached its exit point yet, keep that point.
        # Otherwise the new path would turn immediately and could clip the gate from behind.
        if n_passed > 0:
            centre, normal, _, post = self._gate_geometry(obs, n_passed - 1)
            past = np.dot(start_pos - centre, normal)  # How far we are behind that gate.
            if 0 <= past < np.dot(post - centre, normal) - 0.05:
                waypoints.append(post)
                speeds.append(self.GATE_SPEED)

        for k in range(n_passed, self._n_gates_total):
            centre, normal, pre, post = self._gate_geometry(obs, k)
            vias = self._via_points(obs, k)
            if k == n_passed:
                # When re-planning mid-way, drop via points we have already flown past: a via
                # point is "behind us" if we are closer to the point after it than it is.
                nxt_list = vias[1:] + [pre]
                while vias and np.linalg.norm(start_pos - nxt_list[0]) < np.linalg.norm(
                    vias[0] - nxt_list[0]
                ):
                    vias.pop(0)
                    nxt_list.pop(0)
            for v in vias:
                waypoints.append(v)
                speeds.append(self.SPEED)
            # Skip the line-up point if we are already on the straight line into the gate.
            rel = start_pos - centre
            ahead = np.dot(rel, normal)  # Negative = in front of the gate.
            sideways = np.linalg.norm(rel - ahead * normal)
            on_line = ahead > np.dot(pre - centre, normal) - 0.05 and sideways < 0.15
            if not (k == n_passed and not vias and on_line):
                waypoints.append(pre)
                speeds.append(self.SPEED)
            # A second point on the same straight line keeps the curve straight into the gate
            # (a cubic spline otherwise "bulges" after a turn at the line-up point).
            mid = centre + 0.4 * (pre - centre)
            if np.dot(start_pos - centre, normal) < np.dot(mid - centre, normal) - 0.05:
                waypoints.append(mid)
                speeds.append(self.GATE_SPEED)
            waypoints.append(centre)
            speeds.append(self.GATE_SPEED)  # Slow and straight through the gate ...
            waypoints.append(post)
            speeds.append(self.GATE_SPEED)  # ... and a bit beyond.
        return waypoints, speeds

    # ---- Path ------------------------------------------------------------------
    def _build_spline(
        self, points: np.ndarray, speeds: np.ndarray, start_vel: np.ndarray
    ) -> CubicSpline:
        """Fit a time-parametrised cubic spline through the points.

        speeds[i] is the speed for the segment from points[i] to points[i+1].
        """
        speeds = np.asarray(speeds, dtype=float).copy()
        # Optional: slow down for sharp corners (0 rad turn = full speed).
        for i in range(1, len(points) - 1):
            angle = _angle(points[i] - points[i - 1], points[i + 1] - points[i])
            factor = 1.0 - (1.0 - self.TURN_SLOWDOWN) * angle / np.pi
            speeds[i - 1] = min(speeds[i - 1], self.SPEED * factor)
            speeds[i] = min(speeds[i], self.SPEED * factor)
        dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
        seg_times = np.maximum(dists / speeds, self.MIN_SEG_TIME)
        # Make the path flyable: wherever the curve would need more acceleration than ACC_MAX,
        # give that piece more time and rebuild. (Twice the time = a quarter of the acceleration.)
        for _ in range(10):
            spline = self._fit(points, seg_times, start_vel)
            too_much = False
            for i in range(len(seg_times)):
                ts = np.linspace(spline.x[i], spline.x[i + 1], 8)
                acc = np.max(np.linalg.norm(spline(ts, 2), axis=1))
                if acc > self.ACC_MAX:
                    seg_times[i] *= min(np.sqrt(acc / self.ACC_MAX), 1.3)
                    too_much = True
            if not too_much:
                break
        return self._fit(points, seg_times, start_vel)

    @staticmethod
    def _fit(points: np.ndarray, seg_times: np.ndarray, start_vel: np.ndarray) -> CubicSpline:
        """Cubic spline through the points with the given time per segment."""
        t = np.concatenate(([0.0], np.cumsum(seg_times)))
        # Start with the current velocity (smooth re-planning), end at rest.
        return CubicSpline(t, points, bc_type=((1, start_vel), (1, np.zeros(3))))

    def _plan(self, obs: dict, start_pos: np.ndarray, start_vel: np.ndarray):
        """(Re-)plan a path from the given start state through all remaining gates."""
        waypoints, speeds = self._waypoints(obs, start_pos)
        if not waypoints:  # All gates passed: just hover where we are.
            waypoints, speeds = [start_pos + np.array([0.0, 0.0, 0.01])], [self.SPEED]
        points = np.vstack([start_pos, *waypoints])
        self._spline = self._build_spline(points, np.array(speeds), start_vel)
        self._t_plan = self._tick / self._freq  # Time at which this plan starts.
        self._known_gates = obs["gates_pos"].copy()
        self._known_obstacles = obs["obstacles_pos"].copy()

    def _needs_replan(self, obs: dict) -> bool:
        gates_moved = np.max(np.abs(obs["gates_pos"] - self._known_gates)) > self.CHANGE_TOL
        obst_moved = np.max(np.abs(obs["obstacles_pos"] - self._known_obstacles)) > self.CHANGE_TOL
        return bool(gates_moved or obst_moved)

    def _local_time(self) -> float:
        """Time since the current plan started, clipped to the plan's end."""
        return float(np.clip(self._tick / self._freq - self._t_plan, 0.0, self._spline.x[-1]))

    # ---- Control loop ----------------------------------------------------------
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return the desired state [pos, vel, acc, quat, ang_vel] for this step."""
        if self._needs_replan(obs):
            # Start the new path where the old path is now, with the same velocity (smooth).
            t = self._local_time()
            self._plan(obs, start_pos=self._spline(t), start_vel=self._spline(t, 1))

        t = self._local_time()
        des_pos = self._spline(t)
        des_vel = self._spline(t, 1)
        des_acc = self._spline(t, 2) * self.ACC_FF  # Acceleration "feedforward" (scaled).
        if self._mode == "attitude":
            return self._attitude_control(obs, des_pos, des_vel, des_acc)

        # Height correction: integrate the height error of the last step.
        if self._last_des_z is not None:
            z_err = obs["pos"][2] - self._last_des_z
            self._z_bias = float(np.clip(self._z_bias + self.Z_KI * z_err / self._freq, -0.2, 0.2))
        self._last_des_z = des_pos[2]
        des_pos = des_pos - np.array([0.0, 0.0, self._z_bias])  # Command a bit lower.
        des_quat = R.from_euler("z", 0.0).as_quat()  # Keep yaw at 0.
        return np.concatenate((des_pos, des_vel, des_acc, des_quat, np.zeros(3)), dtype=np.float32)

    def _attitude_control(
        self, obs: dict, des_pos: np.ndarray, des_vel: np.ndarray, des_acc: np.ndarray
    ) -> NDArray[np.floating]:
        """Our own position controller: returns [roll, pitch, yaw, collective thrust].

        Desired acceleration = planned acceleration + corrections for position/velocity errors
        (a PID controller). The drone can only push along its own z-axis, so we tilt that axis
        towards the desired force and set the thrust to its size.
        """
        pos_err = des_pos - obs["pos"]
        vel_err = des_vel - obs["vel"]
        self._i_error = np.clip(self._i_error + pos_err / self._freq, -0.5, 0.5)
        acc = des_acc + self.KP * pos_err + self.KD * vel_err + self.KI * self._i_error
        force = self._mass * (acc + np.array([0.0, 0.0, 9.81]))
        # Limit the tilt so the drone never tries to fly sideways with too little lift.
        horiz = np.linalg.norm(force[:2])
        max_horiz = np.tan(np.deg2rad(self.MAX_TILT_DEG)) * force[2]
        if horiz > max_horiz > 0:
            force[:2] *= max_horiz / horiz
        z_body = R.from_quat(obs["quat"]).as_matrix()[:, 2]
        thrust = float(force @ z_body)
        z_des = force / np.linalg.norm(force)
        x_c = np.array([1.0, 0.0, 0.0])  # yaw = 0
        y_des = np.cross(z_des, x_c)
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
        """Advance time; stop once every gate has been passed."""
        self._tick += 1
        self._finished = int(obs["n_gates_passed"]) >= self._n_gates_total
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self._tick = 0

    def render_callback(self, sim: Sim):
        """Draw the planned path (green) and the current setpoint (red)."""
        from crazyflow.sim.visualize import draw_line, draw_points

        setpoint = self._spline(self._local_time()).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        path = self._spline(np.linspace(0, self._spline.x[-1], 100))
        draw_line(sim, path, rgba=(0.0, 1.0, 0.0, 1.0))
