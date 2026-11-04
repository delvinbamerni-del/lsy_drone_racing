"""Time one episode of the race controller (no rendering)."""
import time, warnings
from pathlib import Path
import gymnasium, numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from lsy_drone_racing.utils import load_config, load_controller
warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)
repo = Path(__file__).resolve().parents[1]
config = load_config(repo / "config/level2.toml")
cls = load_controller(repo / "lsy_drone_racing/control/rl_controller.py")
t0 = time.perf_counter()
env = JaxToNumpy(gymnasium.make(config.env.id, freq=config.env.freq, sim_config=config.sim,
    sensor_range=config.env.sensor_range, control_mode=config.env.control_mode, track=config.env.track,
    disturbances=config.env.get("disturbances"), randomizations=config.env.get("randomizations"), seed=0))
print(f"env build {time.perf_counter()-t0:.1f}s")
for ep in range(4):
    t0 = time.perf_counter(); obs, info = env.reset(); c = cls(obs, info, config); n = 0; tc = 0
    while True:
        t1 = time.perf_counter(); a = c.compute_control(obs, info); tc += time.perf_counter() - t1
        obs, r, term, trunc, info = env.step(a); n += 1
        if c.step_callback(a, obs, r, term, trunc, info) or term or trunc: break
    print(f"ep {ep}: {n} steps, {time.perf_counter()-t0:.1f}s total, controller {tc:.1f}s, gates {obs['n_gates_passed']}")
