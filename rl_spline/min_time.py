"""Minimum-time lap for the nominal Level 2 track (point-mass model, CasADi + IPOPT; dev-only).

Answers: how fast could a lap be, given the drone's thrust and a tilt limit, passing through
each gate opening in the right direction and keeping clear of the poles?

    pixi run python rl_spline/min_time.py --tilt 50 --thrust-frac 0.85 --margin 0.08
"""

from __future__ import annotations

import argparse

import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation

GATES = {1: ([0.5, 0.25, 0.7], -0.78), 2: ([1.05, 0.75, 1.2], 2.35),
         3: ([-1.0, -0.25, 0.7], 3.14), 4: ([0.0, -0.75, 1.2], 0.0)}
ORDER = [1, 2, 3, 4, 2]
POLES = np.array([[0.0, 0.75], [1.0, 0.25], [-1.5, -0.25], [-0.5, -0.75]])
START = np.array([-1.5, 0.75, 0.01])
G = np.array([0, 0, -9.81])
GUESS_SPEED = 1.5


def frame_points(spacing: float = 0.1, gates=None) -> np.ndarray:
    """Keep-out points on every gate's frame square (half-size 0.28 m, from the collision boxes
    in envs/assets/gate.xml) and down its stand to the floor."""
    pts = []
    s = np.arange(-0.36, 0.36 + 1e-9, spacing)
    ring = np.r_[np.c_[s, np.full_like(s, 0.28)], np.c_[s, np.full_like(s, -0.28)],
                 np.c_[np.full_like(s, 0.28), s], np.c_[np.full_like(s, -0.28), s]]
    for gid in (GATES if gates is None else gates):
        center, yaw = GATES[gid]
        R = Rotation.from_euler("z", yaw).as_matrix()
        c = np.array(center)
        pts += [c + R[:, 1] * lat + R[:, 2] * up for lat, up in ring]
        pts += [c + [0, 0, -z] for z in np.arange(0.36, c[2] + 1e-9, spacing)]  # stand
    return np.array(pts)


def solve(tilt_deg=50.0, thrust_frac=0.85, margin=0.08, pole_clear=0.25, cone_deg=45.0, n=30,
          a_max=18.43, floor=0.05, frame_clear=0.0, init=None, jerk=0.0, lag=0.0):
    opti = ca.Opti()
    phases = len(ORDER)
    T = opti.variable(phases)
    P = [opti.variable(3, n + 1) for _ in range(phases)]
    V = [opti.variable(3, n + 1) for _ in range(phases)]
    A = [opti.variable(3, n) for _ in range(phases)]  # thrust acceleration (world frame)
    # lag > 0: A is the drone's *actual* thrust acceleration, following the commanded U with a
    # first-order lag (implicit Euler): A_k = (lag * A_{k-1} + dt * U_k) / (lag + dt)
    U = [opti.variable(3, n) for _ in range(phases)] if lag > 0 else A
    opti.minimize(ca.sum1(T) + 1e-4 * sum(ca.sumsqr(ca.diff(a, 1, 1)) for a in A))
    opti.subject_to(T >= 0.1)
    cos_tilt = np.cos(np.deg2rad(tilt_deg))
    for i in range(phases):
        near = {ORDER[i]} | ({ORDER[i - 1]} if i else set())  # gate left behind and gate ahead
        if init is not None:  # plus every gate the previous path came close to on this leg
            leg = init[1][:, i * n:(i + 1) * n + 1].T
            near |= {g for g, (c, _) in GATES.items()
                     if np.min(np.linalg.norm(leg - np.array(c), axis=1)) < 1.2}
        frames = frame_points(gates=sorted(near)) if frame_clear > 0 else np.zeros((0, 3))
        dt = T[i] / n
        for k in range(n):
            a = A[i][:, k]
            u = U[i][:, k]
            if lag > 0:
                prev = A[i][:, k - 1] if k else (A[i - 1][:, -1] if i else -G)
                opti.subject_to(a * (lag + dt) == lag * prev + dt * u)
            opti.subject_to(P[i][:, k + 1] == P[i][:, k] + V[i][:, k] * dt + 0.5 * (a + G) * dt**2)
            opti.subject_to(V[i][:, k + 1] == V[i][:, k] + (a + G) * dt)
            opti.subject_to(ca.sumsqr(u) <= (thrust_frac * a_max) ** 2)
            opti.subject_to(u[2] >= cos_tilt * ca.norm_2(u))  # thrust within the tilt cone
            if jerk > 0:  # the attitude can only turn so fast: limit the change of the command
                prev_u = U[i][:, k - 1] if k else (U[i - 1][:, -1] if i else -G)
                opti.subject_to(ca.sumsqr(u - prev_u) <= (jerk * dt) ** 2)
        for k in range(n + 1):
            p = P[i][:, k]
            # ground: the drone starts at z = 0.01 at rest, so allow ground level on the first
            # nodes (a fixed 0.05 m floor there forces a minimum step length, see git log)
            opti.subject_to(opti.bounded(0.0 if (i == 0 and k < n // 3) else floor, p[2], 1.9))
            for pole in POLES:
                opti.subject_to(ca.sumsqr(p[:2] - pole) >= pole_clear**2)
            for q in frames:  # gate frames and stands
                opti.subject_to(ca.sumsqr(p - q) >= frame_clear**2)
        if i:  # continuity between phases
            opti.subject_to(P[i][:, 0] == P[i - 1][:, -1])
            opti.subject_to(V[i][:, 0] == V[i - 1][:, -1])
        # gate crossing at the end of the phase: inside the opening, moving along the normal
        center, yaw = GATES[ORDER[i]]
        R = Rotation.from_euler("z", yaw).as_matrix()
        normal, lateral, up = R[:, 0], R[:, 1], R[:, 2]
        d = P[i][:, -1] - np.array(center)
        opti.subject_to(ca.dot(d, normal) == 0)
        opti.subject_to(opti.bounded(-margin, ca.dot(d, lateral), margin))
        opti.subject_to(opti.bounded(-margin, ca.dot(d, up), margin))
        v = V[i][:, -1]
        opti.subject_to(ca.dot(v, normal) >= np.cos(np.deg2rad(cone_deg)) * ca.norm_2(v) + 0.1)
    opti.subject_to(P[0][:, 0] == START)
    opti.subject_to(V[0][:, 0] == 0)
    # initial guess: straight lines at 2 m/s
    prev = START
    for i in range(phases):
        nxt = np.array(GATES[ORDER[i]][0])
        dist = np.linalg.norm(nxt - prev)
        opti.set_initial(T[i], dist / GUESS_SPEED)
        opti.set_initial(P[i], np.linspace(prev, nxt, n + 1).T)
        opti.set_initial(V[i], np.tile((nxt - prev) / (dist / GUESS_SPEED), (n + 1, 1)).T)
        opti.set_initial(A[i], np.tile(-G, (n, 1)).T)
        if lag > 0:
            opti.set_initial(U[i], np.tile(-G, (n, 1)).T)
        prev = nxt
    if init is not None:  # warm start from a previous solution (T, P, V)
        t0, p0, v0 = init
        for i in range(phases):
            sl = slice(i * n, (i + 1) * n + 1)
            opti.set_initial(T[i], t0[i])
            opti.set_initial(P[i], p0[:, sl])
            opti.set_initial(V[i], v0[:, sl])
            a0 = np.diff(v0[:, sl], axis=1) / (t0[i] / n) - G[:, None]
            opti.set_initial(A[i], a0)
            if lag > 0:
                opti.set_initial(U[i], a0)
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "max_iter": 3000})
    sol = opti.solve()
    print("status:", sol.stats()["return_status"], "iters:", sol.stats()["iter_count"])
    t = sol.value(T)
    p = np.hstack([sol.value(P[i])[:, :-1] for i in range(phases)] + [sol.value(P[-1])[:, -1:]])
    v = np.hstack([sol.value(V[i])[:, :-1] for i in range(phases)] + [sol.value(V[-1])[:, -1:]])
    if lag > 0:
        solve.last_U = np.hstack([sol.value(U[i]) for i in range(phases)])
    return t, p, v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tilt", type=float, default=50)
    ap.add_argument("--thrust-frac", type=float, default=0.85)
    ap.add_argument("--margin", type=float, default=0.08)
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    t, p, v = solve(a.tilt, a.thrust_frac, a.margin)
    speed = np.linalg.norm(v, axis=0)
    length = np.sum(np.linalg.norm(np.diff(p, axis=1), axis=0))
    print(f"tilt {a.tilt}°, thrust {a.thrust_frac:.0%}, gate margin {a.margin} m: "
          f"lap {t.sum():.2f} s, legs {np.round(t, 2)}, length {length:.2f} m, "
          f"avg {length / t.sum():.2f} m/s, max {speed.max():.2f} m/s")
    if a.save:
        np.savez(a.save, T=t, P=p, V=v)
