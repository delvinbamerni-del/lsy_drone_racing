"""Evaluate controller class-attribute overrides on many randomized tracks in parallel (dev-only).

Each config is a JSON object of attribute overrides for the controller class in rl_controller.py.
All configs fly the same seeds, so they are compared on equal footing.

    pixi run python rl_spline/sweep.py --episodes 60 \
        '{}' '{"acceleration_limit": 9}' '{"time_scale": 0.85, "speed_limit": 4.5}'

Special key: "time_scale" multiplies all knot_times (shorter = faster nominal plan).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import numpy as np

import train_es as T


def _run(task):
    overrides, seed = task
    cls = T._cls
    keys = set(overrides) - {"time_scale"}
    if "time_scale" in overrides:
        keys |= {"knot_times", "slow_zones", "takeoff_s"}
    saved = {k: getattr(cls, k) for k in keys}
    try:
        for k, v in overrides.items():
            if k != "time_scale":
                setattr(cls, k, tuple(tuple(x) if isinstance(x, list) else x for x in v)
                        if isinstance(v, list) else v)
        if "time_scale" in overrides:  # compress path time; keep slow zones at the same places
            f = overrides["time_scale"]
            cls.knot_times = tuple(np.asarray(cls.knot_times) * f)
            cls.slow_zones = tuple((c * f, d, w * f) for c, d, w in cls.slow_zones)
            cls.takeoff_s = cls.takeoff_s * f
        return T.run_episode((getattr(cls, "policy_params", None), seed))
    finally:
        for k, v in saved.items():
            setattr(cls, k, v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--seed0", type=int, default=20_000)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    configs = [json.loads(c) for c in a.configs]
    seeds = range(a.seed0, a.seed0 + a.episodes)
    with mp.get_context("spawn").Pool(a.workers, initializer=T._init_worker) as pool:
        for cfg in configs:
            res = pool.map(_run, [(cfg, s) for s in seeds])
            sr, t = T.summary(res)
            gates = np.mean([g for _, _, g in res])
            print(f"{sr:6.1%}  {t:6.3f}s  gates {gates:.2f}  {json.dumps(cfg)}", flush=True)


if __name__ == "__main__":
    main()
