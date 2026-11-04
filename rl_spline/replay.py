"""Replay one seed; print the last steps with distances to every gate's collision boxes and every pole (dev-only)."""
import crazyflow  # noqa: F401
import logging, sys, warnings
from pathlib import Path
import gymnasium, numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.utils import load_config, load_controller
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
c = load_config(Path("config/level2.toml"))
cls = load_controller(Path("lsy_drone_racing/control/rl_controller.py"))
env = JaxToNumpy(gymnasium.make(c.env.id, freq=c.env.freq, sim_config=c.sim, sensor_range=c.env.sensor_range,
    control_mode=c.env.control_mode, track=c.env.track, disturbances=c.env.get("disturbances"), randomizations=c.env.get("randomizations"), seed=0))
BOXES = [((0, 0, 0.28), (0.01, 0.36, 0.08)), ((0, 0, -0.28), (0.01, 0.36, 0.08)),
         ((0, -0.28, 0), (0.01, 0.08, 0.36)), ((0, 0.28, 0), (0.01, 0.08, 0.36)), ((0, 0, -0.86), (0.05, 0.05, 0.5))]
def box_dist(p_local, center, half):  # distance from point to an axis-aligned box (gate frame)
    d = np.abs(np.asarray(p_local) - center) - half
    return float(np.linalg.norm(np.maximum(d, 0)) + min(d.max(), 0))
obs, info = env.reset(seed=int(sys.argv[1])); ctrl = cls(obs, info, c); hist = []
while True:
    a = ctrl.compute_control(obs, info)
    sphere = obs["pos"] + R.from_quat(obs["quat"]).apply([0, 0, 0.02])
    gd = []
    for g in range(4):
        loc = R.from_quat(obs["gates_quat"][g]).inv().apply(sphere - obs["gates_pos"][g])
        gd.append(min(box_dist(loc, np.array(cc), np.array(h)) for cc, h in BOXES))
    pd = np.linalg.norm(obs["obstacles_pos"][:, :2] - sphere[:2], axis=1) - 0.015
    hist.append((ctrl.t, obs["pos"].round(2), int(obs["n_gates_passed"]), np.round(gd, 3), pd.round(3)))
    obs, r, term, trunc, info = env.step(a)
    if ctrl.step_callback(a, obs, r, term, trunc, info) or term or trunc: break
print("collision sphere radius 0.086; distances are surface-to-centre of sphere")
for t, p, g, gd, pd in hist[-6:]:
    print(f"t={t:.2f} pos={p} gates={g}  gate-box dist {gd}  pole dist {pd}")
