"""Train the residual speed policy in rl_controller.py with CMA-ES (dev-only, not part of a submission).

Each candidate parameter vector is flown on the same N randomized tracks (common random numbers,
so candidates are compared on equal footing). Episodes run in parallel worker processes.

    pixi run python rl_spline/train_es.py --run es1 --generations 60
    pixi run python rl_spline/train_es.py --run es1 --generations 120 --resume

Writes rl_spline/runs/<run>/{log.csv, state.npz, best.npy}. Embed with rl_spline/embed.py.
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_env = _cls = _config = None


def _init_worker():
    global _env, _cls, _config
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    import gymnasium
    from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

    from lsy_drone_racing.utils import load_config, load_controller

    _config = load_config(REPO / "config/level2.toml")
    _cls = load_controller(REPO / "lsy_drone_racing/control/rl_controller.py")
    c = _config
    _env = JaxToNumpy(
        gymnasium.make(c.env.id, freq=c.env.freq, sim_config=c.sim, sensor_range=c.env.sensor_range,
                       control_mode=c.env.control_mode, track=c.env.track,
                       disturbances=c.env.get("disturbances"),
                       randomizations=c.env.get("randomizations"), seed=0)
    )


def run_episode(task: tuple[np.ndarray, int]) -> tuple[bool, float, int]:
    """Fly one episode; returns (finished, flight time as sim.py reports it, gates passed)."""
    params, seed = task
    if params is not None:
        _cls.policy_params = params
    obs, info = _env.reset(seed=int(seed))
    ctrl = _cls(obs, info, _config)
    i = 0
    while True:
        action = ctrl.compute_control(obs, info)
        obs, reward, terminated, truncated, info = _env.step(action)
        done = ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        if terminated or truncated or done:
            break
        i += 1
    gates = int(obs["n_gates_passed"])
    return gates == len(obs["gate_sequence"]), i / _config.env.freq, max(gates, 0)


def score(results: list[tuple[bool, float, int]], crash_penalty: float) -> float:
    """Mean episode score: faster finishes score higher, any failure costs `crash_penalty`."""
    s = [10 * (8.5 - t) if ok else -crash_penalty + 0.5 * g for ok, t, g in results]
    return float(np.mean(s))


def summary(results):
    ok = [t for f, t, _ in results if f]
    return len(ok) / len(results), (float(np.mean(ok)) if ok else float("nan"))


class CMAES:
    """Standard (mu/mu_w, lambda)-CMA-ES (Hansen's tutorial), maximizing."""

    def __init__(self, mean, sigma, popsize):
        n = self.n = len(mean)
        self.mean, self.sigma, self.lam = np.array(mean, float), sigma, popsize
        mu = popsize // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.w = w / w.sum()
        self.mueff = 1 / np.sum(self.w**2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chin = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.B, self.D, self.C = np.eye(n), np.ones(n), np.eye(n)
        self.gen = 0

    def ask(self, rng):
        self.z = rng.standard_normal((self.lam, self.n))
        self.y = self.z * self.D @ self.B.T
        return self.mean + self.sigma * self.y

    def tell(self, fitness):
        order = np.argsort(fitness)[::-1]
        mu = len(self.w)
        y_sel = self.y[order[:mu]]
        y_w = self.w @ y_sel
        self.mean = self.mean + self.sigma * y_w
        c_inv_sqrt = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * c_inv_sqrt @ y_w
        self.gen += 1
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chin < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (y_sel.T * self.w) @ y_sel)
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chin - 1))
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        d2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(d2, 1e-20))

    def state(self):
        return {k: getattr(self, k) for k in ("mean", "sigma", "pc", "ps", "B", "D", "C", "gen")}

    def load(self, s):
        for k in ("mean", "pc", "ps", "B", "D", "C"):
            setattr(self, k, np.array(s[k]))
        self.sigma, self.gen = float(s["sigma"]), int(s["gen"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="es1")
    p.add_argument("--generations", type=int, default=60)
    p.add_argument("--popsize", type=int, default=16)
    p.add_argument("--episodes", type=int, default=32, help="tracks per candidate per generation")
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--crash-penalty", type=float, default=5.0)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--val-episodes", type=int, default=96)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    out = REPO / "rl_spline/runs" / a.run
    out.mkdir(parents=True, exist_ok=True)
    import lsy_drone_racing.control.rl_controller as rc  # only for N_PARAMS

    es = CMAES(np.zeros(rc.N_PARAMS), a.sigma, a.popsize)
    best_val = -np.inf
    if a.resume and (out / "state.npz").exists():
        s = np.load(out / "state.npz")
        es.load(s)
        best_val = float(s["best_val"])
        print(f"resumed at generation {es.gen}, sigma {es.sigma:.3f}, best val {best_val:.2f}")
    rng = np.random.default_rng(1000 + es.gen)
    val_seeds = np.arange(10_000, 10_000 + a.val_episodes)  # fixed, never used for training

    log_path = out / "log.csv"
    new_log = not log_path.exists()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init_worker) as pool, open(log_path, "a", newline="") as f:
        log = csv.writer(f)
        if new_log:
            log.writerow(["gen", "sigma", "best_fit", "mean_fit", "pop_success", "val_score",
                          "val_success", "val_time", "seconds"])
        if es.gen == 0:  # baseline: the plain spline controller on the validation tracks
            r = pool.map(run_episode, [(np.zeros(rc.N_PARAMS), s) for s in val_seeds])
            sr, t = summary(r)
            print(f"baseline (zero policy): val success {sr:.0%}, time {t:.3f}s, score {score(r, a.crash_penalty):.2f}")
            log.writerow([0, a.sigma, "", "", "", score(r, a.crash_penalty), sr, t, 0])
        while es.gen < a.generations:
            t0 = time.time()
            pop = es.ask(rng)
            seeds = rng.integers(0, 2**31 - 1, a.episodes)
            res = pool.map(run_episode, [(x, s) for x in pop for s in seeds])
            res = [res[i * a.episodes:(i + 1) * a.episodes] for i in range(a.popsize)]
            fit = np.array([score(r, a.crash_penalty) for r in res])
            pop_sr = np.mean([summary(r)[0] for r in res])
            es.tell(fit)
            row = [es.gen, es.sigma, fit.max(), fit.mean(), pop_sr, "", "", "", 0]
            if es.gen % a.val_every == 0 or es.gen == a.generations:
                r = pool.map(run_episode, [(es.mean, s) for s in val_seeds])
                v, (sr, t) = score(r, a.crash_penalty), summary(r)
                row[5:8] = [v, sr, t]
                improved = v > best_val
                if improved:
                    best_val = v
                    np.save(out / "best.npy", es.mean)
                print(f"  val: score {v:.2f}, success {sr:.0%}, time {t:.3f}s{'  (new best)' if improved else ''}")
            row[8] = round(time.time() - t0, 1)
            log.writerow(row)
            f.flush()
            np.savez(out / "state.npz", best_val=best_val, **es.state())
            print(f"gen {es.gen}: sigma {es.sigma:.3f}, fit best {fit.max():.2f} mean {fit.mean():.2f}, "
                  f"pop success {pop_sr:.0%}, {row[8]}s", flush=True)


if __name__ == "__main__":
    main()
