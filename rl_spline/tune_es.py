"""Learn a per-passage braking schedule (approach window and rate for each of the 5 gate passages)
with CMA-ES (dev-only). Same scoring and common-random-number setup as train_es.py.

    pixi run python rl_spline/tune_es.py --run brake1 --generations 25
Writes rl_spline/runs/<run>/{log.csv, best.json}.
"""
import argparse, csv, json, multiprocessing as mp, time
from pathlib import Path
import numpy as np
import train_es as T

N = 5  # gate passages
W_LO, W_HI, R_LO, R_HI = 0.2, 1.0, 0.45, 1.0

def decode(x):
    s = 1 / (1 + np.exp(-np.asarray(x)))
    return {"approach_window": (W_LO + (W_HI - W_LO) * s[:N]).round(3).tolist(),
            "approach_rate": (R_LO + (R_HI - R_LO) * s[N:]).round(3).tolist()}

def encode(window, rate):
    w = (np.full(N, window) - W_LO) / (W_HI - W_LO); r = (np.full(N, rate) - R_LO) / (R_HI - R_LO)
    return np.log(np.r_[w, r] / (1 - np.r_[w, r]))

def _run(task):
    over, seed = task
    cls = T._cls
    for k, v in over.items(): setattr(cls, k, np.array(v))
    return T.run_episode((None, seed))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="brake1"); ap.add_argument("--generations", type=int, default=25)
    ap.add_argument("--popsize", type=int, default=10); ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--sigma", type=float, default=0.6); ap.add_argument("--crash-penalty", type=float, default=5.0)
    ap.add_argument("--val-every", type=int, default=5); ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--init-window", type=float, default=0.65); ap.add_argument("--init-rate", type=float, default=0.7)
    ap.add_argument("--init-json", default="", help="start from a best.json of an earlier run")
    a = ap.parse_args()
    out = Path(__file__).resolve().parent / "runs" / a.run; out.mkdir(parents=True, exist_ok=True)
    if a.init_json:
        j = json.load(open(a.init_json)); x0 = encode(np.array(j["approach_window"]), np.array(j["approach_rate"]))
    else:
        x0 = encode(a.init_window, a.init_rate)
    es = T.CMAES(x0, a.sigma, a.popsize)
    rng = np.random.default_rng(7); val_seeds = range(40_000, 40_000 + a.val_episodes); best = -np.inf
    with mp.get_context("spawn").Pool(8, initializer=T._init_worker) as pool, open(out / "log.csv", "w", newline="") as f:
        log = csv.writer(f); log.writerow(["gen", "sigma", "best_fit", "mean_fit", "val_score", "val_success", "val_time", "params"])
        def validate(x):
            r = pool.map(_run, [(decode(x), s) for s in val_seeds]); return T.score(r, a.crash_penalty), *T.summary(r)
        v, sr, t = validate(es.mean)
        print(f"start: val success {sr:.0%} time {t:.3f}s score {v:.2f}  {decode(es.mean)}", flush=True)
        log.writerow([0, es.sigma, "", "", v, sr, t, json.dumps(decode(es.mean))]); best = v
        json.dump(decode(es.mean) | {"val_success": sr, "val_time": t}, open(out / "best.json", "w"))
        while es.gen < a.generations:
            t0 = time.time(); pop = es.ask(rng); seeds = rng.integers(0, 2**31 - 1, a.episodes)
            res = pool.map(_run, [(decode(x), s) for x in pop for s in seeds])
            fit = np.array([T.score(res[i * a.episodes:(i + 1) * a.episodes], a.crash_penalty) for i in range(a.popsize)])
            es.tell(fit); row = [es.gen, es.sigma, fit.max(), fit.mean(), "", "", "", ""]
            if es.gen % a.val_every == 0 or es.gen == a.generations:
                v, sr, t = validate(es.mean); row[4:8] = [v, sr, t, json.dumps(decode(es.mean))]
                tag = ""
                if v > best:
                    best = v; tag = "  (new best)"
                    json.dump(decode(es.mean) | {"val_success": sr, "val_time": t}, open(out / "best.json", "w"))
                print(f"  val: success {sr:.0%} time {t:.3f}s score {v:.2f}{tag}  {decode(es.mean)}", flush=True)
            log.writerow(row); f.flush()
            print(f"gen {es.gen}: fit best {fit.max():.2f} mean {fit.mean():.2f}, sigma {es.sigma:.2f}, {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
