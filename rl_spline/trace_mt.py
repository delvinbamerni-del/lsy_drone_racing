"""Trace the min-time controller: reference vs actual, per step (dev-only)."""
import logging, sys, warnings
from pathlib import Path
import gymnasium, numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from lsy_drone_racing.utils import load_config, load_controller
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
repo = Path(__file__).resolve().parents[1]
c = load_config(repo / "config/level2.toml")
cls = load_controller(repo / "lsy_drone_racing/control/rl_controller.py")
env = JaxToNumpy(gymnasium.make(c.env.id, freq=c.env.freq, sim_config=c.sim, sensor_range=c.env.sensor_range,
    control_mode=c.env.control_mode, track=c.env.track, disturbances=c.env.get("disturbances"),
    randomizations=c.env.get("randomizations"), seed=0))
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
every = int(sys.argv[2]) if len(sys.argv) > 2 else 5
obs, info = env.reset(seed=seed); ctrl = cls(obs, info, c)
np.set_printoptions(precision=2, suppress=True)
i = 0
while True:
    a = ctrl.compute_control(obs, info)
    pr, vr, _ = ctrl._ref(ctrl.t); o, _, _ = ctrl._offset(ctrl.t)
    if i % every == 0:
        print(f"t={ctrl.t:4.2f} ref={pr+o} pos={obs['pos']} err={np.linalg.norm(pr+o-obs['pos']):.2f} "
              f"|v|ref={np.linalg.norm(vr):.1f} |v|={np.linalg.norm(obs['vel']):.1f} rpy={np.degrees(a[:3]).round(0)} T={a[3]:.2f} gates={obs['n_gates_passed']}")
    obs, r, term, trunc, info = env.step(a); i += 1
    if ctrl.step_callback(a, obs, r, term, trunc, info) or term or trunc:
        print(f"END t={ctrl.t:.2f} term={term} gates={obs['n_gates_passed']} pos={obs['pos']}"); break
