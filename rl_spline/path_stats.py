"""Fly one nominal-track lap and report path length, speed and where time is spent (dev-only)."""
import logging, warnings
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
for seed in range(6):
    obs, info = env.reset(seed=seed); ctrl = cls(obs, info, c)
    P, V, R, G, S = [], [], [], [], []
    while True:
        a = ctrl.compute_control(obs, info)
        P.append(obs["pos"].copy()); V.append(np.linalg.norm(obs["vel"])); R.append(ctrl.rate); G.append(int(obs["n_gates_passed"])); S.append(ctrl._saturated)
        obs, r, term, trunc, info = env.step(a)
        if ctrl.step_callback(a, obs, r, term, trunc, info) or term or trunc: break
    if int(obs["n_gates_passed"]) != 5: print(f"seed {seed}: crashed after {obs['n_gates_passed']} gates"); continue
    P, V, G = np.array(P), np.array(V), np.array(G)
    L = np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))
    print(f"seed {seed}: lap {len(P)/c.env.freq:.2f}s, flown length {L:.2f} m, avg speed {V.mean():.2f} m/s, max {V.max():.2f} m/s, "
          f"rate avg {np.mean(R):.2f} max {np.max(R):.2f}, saturated {np.mean(S):.0%} of steps")
    t = [np.argmax(G >= k) / c.env.freq for k in range(1, 5)] + [len(P) / c.env.freq]
    print("   time at gates 1-5:", np.round(t, 2), " segment s:", np.round(np.diff([0] + t), 2))
    break
