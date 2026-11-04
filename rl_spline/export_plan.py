"""Turn a min_time.py solution (.npz) into the compact PLAN literal used by rl_controller.py (dev-only).

    pixi run python rl_spline/export_plan.py rl_spline/runs/mintime_tilt30.npz
"""
import sys
import numpy as np

d = np.load(sys.argv[1])
T, P, V = d["T"], d["P"], d["V"]
n = (P.shape[1] - 1) // len(T)
dts = np.repeat(T / n, n)
A = np.diff(V, axis=1) / dts - np.array([0, 0, -9.81])[:, None]  # thrust acceleration per step
# check: integrating A from the start state reproduces P
p, v = P[:, 0].copy(), V[:, 0].copy()
for k, dt in enumerate(dts):
    a = A[:, k] + [0, 0, -9.81]
    p, v = p + v * dt + 0.5 * a * dt**2, v + a * dt
print(f"# reconstruction error {np.abs(p - P[:, -1]).max():.2e} m, lap {T.sum():.3f} s, n {n}")
print("PLAN_T = np.array([" + ", ".join(f"{t:.6f}" for t in T) + "])")
print("PLAN_A = np.array([")
for k in range(A.shape[1]):
    print("    [" + ", ".join(f"{x:.5f}" for x in A[:, k]) + "],")
print("])")
if "U" in d.files:
    print("PLAN_U = np.array([")
    for k in range(d["U"].shape[1]):
        print("    [" + ", ".join(f"{x:.5f}" for x in d["U"][:, k]) + "],")
    print("])")
else:
    print("PLAN_U = None")
print("PLAN_P0 = np.array([" + ", ".join(f"{x:.4f}" for x in P[:, 0]) + "])")
