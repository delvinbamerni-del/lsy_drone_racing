"""Fly the nominal track (no randomization/disturbance) and print the tracking error along the lap (dev-only)."""
import crazyflow  # noqa: F401
import json, logging, sys, warnings
from pathlib import Path
import gymnasium, numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from lsy_drone_racing.utils import load_config, load_controller
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
c = load_config(Path("config/level2.toml"))
cls = load_controller(Path("lsy_drone_racing/control/rl_controller.py"))
for k, v in (json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}).items(): setattr(cls, k, v)
env = JaxToNumpy(gymnasium.make(c.env.id, freq=c.env.freq, sim_config=c.sim, sensor_range=c.env.sensor_range,
    control_mode=c.env.control_mode, track=c.env.track, disturbances=None, randomizations=None, seed=0))
obs, info = env.reset(seed=0); ctrl = cls(obs, info, c); rows = []
while True:
    a = ctrl.compute_control(obs, info)
    pr, vr, _ = ctrl._ref(ctrl.t)
    e = obs["pos"] - pr
    tang = vr / max(np.linalg.norm(vr), 1e-6)
    cross = e - (e @ tang) * tang
    rows.append((ctrl.t, np.linalg.norm(cross), e @ tang, ctrl.rate, int(obs["n_gates_passed"])))
    obs, r, term, trunc, info = env.step(a)
    if ctrl.step_callback(a, obs, r, term, trunc, info) or term or trunc: break
rows = np.array(rows)
print(f"end: gates {int(obs['n_gates_passed'])}, term {term}, real time {len(rows)/50:.2f}s, plan clock {ctrl.t:.2f}s")
for g in range(6):
    m = rows[:, 4] == g
    if m.any():
        print(f" leg {g}: cross-track mean {rows[m,1].mean():.3f} max {rows[m,1].max():.3f} m, along-track mean {rows[m,2].mean():+.3f} m, rate mean {rows[m,3].mean():.2f}")
