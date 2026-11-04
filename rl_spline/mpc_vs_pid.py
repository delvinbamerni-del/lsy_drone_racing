"""Fly with the PID tracker; at each step also ask the MPC what it would command (dev-only)."""
import crazyflow  # noqa: F401
import logging, warnings
from pathlib import Path
import gymnasium, numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from lsy_drone_racing.utils import load_config, load_controller
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
c = load_config(Path("config/level2.toml"))
cls = load_controller(Path("lsy_drone_racing/control/rl_controller.py"))
env = JaxToNumpy(gymnasium.make(c.env.id, freq=c.env.freq, sim_config=c.sim, sensor_range=c.env.sensor_range,
    control_mode=c.env.control_mode, track=c.env.track, disturbances=None, randomizations=None, seed=0))
obs, info = env.reset(seed=0); ctrl = cls(obs, info, c)
np.set_printoptions(precision=3, suppress=True)
for i in range(60):
    a = ctrl.compute_control(obs, info)
    m = ctrl._mpc_control(obs, ctrl.t, ctrl.rate)
    if i % 6 == 0:
        print(f"t={ctrl.t:.2f} PID rpy {np.degrees(a[:3]).round(1)} T {a[3]:.3f} | MPC rpy {np.degrees(m[:3]).round(1)} T {m[3]:.3f}")
    obs, *_ = env.step(a); ctrl.step_callback(a, obs, 0, False, False, {})
