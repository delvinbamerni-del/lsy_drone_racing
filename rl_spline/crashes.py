"""Where and why does the min-time controller fail? (dev-only)

    pixi run python rl_spline/crashes.py 60 '{"ff_lead": 0.1}'
"""
import json, multiprocessing as mp, sys
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import train_es as T

def run(task):
    over, seed = task
    cls = T._cls
    for k, v in over.items(): setattr(cls, k, np.array(v) if isinstance(v, list) else v)
    obs, info = T._env.reset(seed=seed); c = cls(obs, info, T._config)
    last, maxerr, sat = obs["pos"].copy(), 0.0, 0
    while True:
        a = c.compute_control(obs, info)
        pr, _, _ = c._ref(c.t); o, _, _ = c._offset(c.t)
        maxerr = max(maxerr, float(np.linalg.norm(pr + o - obs["pos"])))
        last, lastvel, lastrpy = obs["pos"].copy(), obs["vel"].copy(), a[:3].copy()
        obs, r, term, trunc, info = T._env.step(a)
        if c.step_callback(a, obs, r, term, trunc, info): return dict(ok=True, t=c.t)
        if term or trunc:
            poles = np.asarray(obs["obstacles_pos"])[:, :2]
            pole_d = np.linalg.norm(poles - last[:2], axis=1)
            gates = np.asarray(obs["gates_pos"])
            gate_d = np.linalg.norm(gates - last, axis=1)
            near = f"pole{int(np.argmin(pole_d))} {pole_d.min():.2f}m" if pole_d.min() < gate_d.min() else f"gate{int(np.argmin(gate_d))} {gate_d.min():.2f}m"
            from scipy.spatial.transform import Rotation as Rot
            gi = int(np.argmin(gate_d)); Rg = Rot.from_quat(obs["gates_quat"][gi]).as_matrix()
            local = (last - gates[gi]) @ Rg  # (normal, lateral, vertical) in the gate frame
            shift = gates[gi] - np.array([g["pos"] for g in T._config.env.track.gates])[gi]
            return dict(ok=False, seed=seed, near=near, local=local.round(2).tolist(), shift_lat=float((shift @ Rg)[1]), shift_up=float((shift @ Rg)[2]), pole_min=float(pole_d.min()), gate_min=float(gate_d.min()), z=float(last[2]), gates=int(obs["n_gates_passed"]), pos=last.tolist(), speed=float(np.linalg.norm(lastvel)),
                        tilt=float(np.degrees(np.max(np.abs(lastrpy[:2])))), maxerr=maxerr, t=c.t, trunc=bool(trunc))

def detail(fails):
    for r in fails:
        if r["near"].startswith("gate") and r["gate_min"] < 0.5:
            print(f"   {r['near']:<14} seed {r['seed']} pole_min {r['pole_min']:.2f} z {r['z']:.2f} after {r['gates']} gates, local (normal, lat, up) {r['local']}, gate shift lat {r['shift_lat']:+.2f} up {r['shift_up']:+.2f}")

if __name__ == "__main__":
    n = int(sys.argv[1]); over = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    with mp.get_context("spawn").Pool(8, initializer=T._init_worker) as pool:
        res = pool.map(run, [(over, s) for s in range(20000, 20000 + n)])
    fails = [r for r in res if not r["ok"]]
    if "-v" in sys.argv: detail(fails)
    print(f"success {sum(r['ok'] for r in res)}/{n}")
    from collections import Counter
    print("failures by gates passed:", dict(sorted(Counter(r["gates"] for r in fails).items())))
    for g in sorted(set(r["gates"] for r in fails)):
        rs = [r for r in fails if r["gates"] == g]
        P = np.array([r["pos"] for r in rs])
        print(f" after {g} gates ({len(rs)}): mean pos {P.mean(0).round(2)}, spread {P.std(0).round(2)}, t {np.mean([r['t'] for r in rs]):.2f}s, "
              f"nearest: {Counter(r['near'].split()[0] for r in rs).most_common(3)}, speed {np.mean([r['speed'] for r in rs]):.1f}, tilt {np.mean([r['tilt'] for r in rs]):.0f}°, max track err {np.mean([r['maxerr'] for r in rs]):.2f} m")
