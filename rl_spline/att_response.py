"""Step response of the attitude interface: command a roll/pitch step in hover, log actual attitude (dev-only)."""
import logging, warnings
import crazyflow  # noqa: F401  (must be imported before scipy)
from pathlib import Path
import gymnasium, numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.spatial.transform import Rotation as R
from crazyflow.dynamics import load_params
from lsy_drone_racing.utils import load_config
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
c = load_config(Path("config/level2.toml"))
env = JaxToNumpy(gymnasium.make(c.env.id, freq=c.env.freq, sim_config=c.sim, sensor_range=c.env.sensor_range,
    control_mode="attitude", track=c.env.track, disturbances=c.env.get("disturbances"), randomizations=c.env.get("randomizations"), seed=0))
p = load_params(c.sim.dynamics, c.sim.drone); m = p["mass"]
obs, info = env.reset(seed=1)
for i in range(60):  # climb to 1 m and hover
    e = 1.0 - obs["pos"][2]
    obs, *_ = env.step(np.array([0, 0, 0, m * (9.81 + 6 * e - 3 * obs["vel"][2])], np.float32))
print("hover z", obs["pos"][2].round(2))
for i in range(15):
    cmd = np.array([0.0, np.deg2rad(30), 0.0, m * 9.81 / np.cos(np.deg2rad(30))], np.float32)
    obs, *_ = env.step(cmd)
    rpy = R.from_quat(obs["quat"]).as_euler("xyz", degrees=True)
    print(f"t={(i+1)*0.02:.2f}s  cmd pitch 30°  actual rpy {rpy.round(1)}")
