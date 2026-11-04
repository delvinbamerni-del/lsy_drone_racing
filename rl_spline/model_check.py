"""How well does the so_rpy model predict the simulator? Open-loop 0.1/0.3/0.5 s predictions (dev-only)."""
import crazyflow  # noqa: F401
import logging, warnings
from pathlib import Path
import casadi as ca, gymnasium, numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.spatial.transform import Rotation as R
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.so_rpy import symbolic_dynamics_euler
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from lsy_drone_racing.utils import load_config, load_controller
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
c = load_config(Path("config/level2.toml"))
p = load_params(Dynamics.so_rpy, c.sim.drone)
Xd, X, U, _ = symbolic_dynamics_euler(mass=p["mass"], gravity_vec=p["gravity_vec"], acc_coef=p["acc_coef"], cmd_f_coef=p["cmd_f_coef"],
    rpy_coef=p["rpy_coef"], rpy_rates_coef=p["rpy_rates_coef"], cmd_rpy_coef=p["cmd_rpy_coef"])
f = ca.Function("f", [X, U], [Xd])
def step(x, u, dt=0.02, sub=10):
    h = dt / sub
    for _ in range(sub):
        k1 = f(x, u); k2 = f(x + h/2*k1, u); k3 = f(x + h/2*k2, u); k4 = f(x + h*k3, u)
        x = x + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return x
cls = load_controller(Path("lsy_drone_racing/control/rl_controller.py"))
env = JaxToNumpy(gymnasium.make(c.env.id, freq=c.env.freq, sim_config=c.sim, sensor_range=c.env.sensor_range,
    control_mode=c.env.control_mode, track=c.env.track, disturbances=None, randomizations=None, seed=0))
obs, info = env.reset(seed=0); ctrl = cls(obs, info, c); xs, us = [], []
for i in range(250):
    a = ctrl.compute_control(obs, info)
    xs.append(np.concatenate([obs["pos"], R.from_quat(obs["quat"]).as_euler("xyz"), obs["vel"], ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])])); us.append(a)
    obs, r, term, trunc, info = env.step(a)
    if ctrl.step_callback(a, obs, r, term, trunc, info) or term or trunc: break
xs, us = np.array(xs), np.array(us)
for H in [5, 15, 25]:
    errs, atts = [], []
    for k in range(10, len(xs) - H, 5):
        x = ca.DM(xs[k])
        for j in range(H): x = step(x, ca.DM(us[k + j]))
        x = np.array(x).ravel(); errs.append(np.linalg.norm(x[:3] - xs[k + H, :3])); atts.append(np.degrees(np.abs(x[3:5] - xs[k + H, 3:5]).max()))
    print(f"horizon {H/50:.1f}s: position error mean {np.mean(errs):.3f} max {np.max(errs):.3f} m, attitude error mean {np.mean(atts):.1f} deg")
