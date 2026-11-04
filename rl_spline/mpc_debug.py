"""MPC internals on the nominal track: solver status, predicted-vs-reference error, command (dev-only)."""
import crazyflow  # noqa: F401
import collections, json, logging, sys, warnings
from pathlib import Path
import gymnasium, numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from lsy_drone_racing.utils import load_config, load_controller
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
c = load_config(Path("config/level2.toml"))
cls = load_controller(Path("lsy_drone_racing/control/rl_controller.py"))
cls.use_mpc = True
for k, v in (json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}).items(): setattr(cls, k, v)
mod = sys.modules[cls.__module__]
env = JaxToNumpy(gymnasium.make(c.env.id, freq=c.env.freq, sim_config=c.sim, sensor_range=c.env.sensor_range,
    control_mode=c.env.control_mode, track=c.env.track, disturbances=None, randomizations=None, seed=0))
obs, info = env.reset(seed=0); ctrl = cls(obs, info, c); stats = collections.Counter()
np.set_printoptions(precision=2, suppress=True)
for i in range(200):
    a = ctrl.compute_control(obs, info)
    solver = next(iter(mod._MPC_CACHE.values()))[0]
    st = solver.get_status(); stats[st] += 1
    if i % 10 == 0:
        pred = np.array([solver.get(j, "x")[:3] for j in range(0, ctrl.mpc_N + 1, 5)])
        refs = np.array([ctrl._ref(ctrl.t + j * ctrl.dt * ctrl.rate)[0] for j in range(0, ctrl.mpc_N + 1, 5)])
        print(f"t={ctrl.t:.2f} status {st} iters {solver.get_stats('sqp_iter')} pred-ref err {np.linalg.norm(pred-refs,axis=1).round(3)} u {np.degrees(a[:3]).round(0)} T {a[3]:.2f}")
    obs, r, term, trunc, info = env.step(a)
    if ctrl.step_callback(a, obs, r, term, trunc, info) or term or trunc: break
print("status counts:", dict(stats), "(0 = success, 2 = max iter, 3/4 = QP/min-step failure)")
