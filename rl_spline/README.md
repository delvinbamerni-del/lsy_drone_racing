# rl_spline: dev tools for `rl_controller.py` (not part of a submission)

The race controller is **only** `lsy_drone_racing/control/rl_controller.py` (attitude mode).
Everything in this folder is for building and tuning it. Run all commands from the repo root.

## How the controller works
1. **Plan:** a minimum-time trajectory for the nominal track, solved offline with CasADi
   (point mass, tilt/thrust/jerk limits, gate openings, gate frames + stands, poles). It is
   embedded between the `PLAN START` / `PLAN END` markers as leg durations + accelerations.
2. **Gate correction:** when a gate is revealed (0.7 m), a cubic offset path shifts the reference
   to the true gate position by the planned crossing time.
3. **Tracking:** plan acceleration feedforward + PID -> thrust and roll/pitch.
4. **Governor:** the reference clock slows if the drone falls behind (accepting ~5 cm lag), and
   brakes before gates that are unseen or just revealed, with a per-gate schedule
   (`approach_window`, `approach_rate`) learned by CMA-ES.

## Pipeline
| Step | Command |
|---|---|
| Solve a plan | see `min_time.py` (`solve(...)`, warm start with `init=`); examples below |
| Export it | `pixi run python rl_spline/export_plan.py rl_spline/runs/mintime_t40.npz > rl_spline/runs/plan_t40.txt` |
| Put it in the controller | `pixi run python rl_spline/insert_plan.py rl_spline/runs/plan_t40.txt` |
| Compare settings (same tracks) | `pixi run python rl_spline/sweep.py --episodes 60 '{}' '{"approach_rate": 0.7}'` |
| Where/why it crashes | `pixi run python rl_spline/crashes.py 60 '{}' -v` |
| Learn the braking schedule | `pixi run python rl_spline/tune_es.py --run brake1 --generations 25` |
| Official check | `pixi run evaluate` (20 runs) |

Plan solve example (warm-started from an existing plan to keep its route):

```python
import sys; sys.path.insert(0, "rl_spline"); import numpy as np, min_time as M
d = np.load("rl_spline/runs/mintime_h23.npz")
t, p, v = M.solve(tilt_deg=40, thrust_frac=0.8, margin=0.04, pole_clear=0.31, cone_deg=35,
                  floor=0.25, frame_clear=0.23, jerk=45, init=(d["T"], d["P"], d["V"]))
np.savez("rl_spline/runs/mintime_new.npz", T=t, P=p, V=v)
```

## Diagnostics
- `nominal_err.py`: tracking error on the nominal track (no randomization).
- `replay.py <seed>`: last steps before a crash, with exact distances to every gate box and pole.
- `trace_mt.py <seed>`: reference vs actual, step by step.
- `att_response.py`: attitude step response (about 0.1 s to half of a tilt change).

## Results (official `evaluate.py`, 20 runs each)
| Commit | Version | Result |
|---|---|---|
| a7527d4 | 30 deg plan, offset correction, governor | 6.85 s / 55% |
| c679d42 | + lag bias, braking before new gates | 6.67 s / 60%, 6.68 s / 60% |
| 65489e2 | 40 deg plan + learned per-gate braking | 6.43 s / 85%, 6.41 s / 75% |

## Older / unrelated files
- `train_es.py`: CMA-ES core and worker setup (used by `sweep.py`, `tune_es.py`); its own main
  trained the residual speed policy of the earlier spline controller (run `es1`, no gain).
- `embed.py`, `bench.py`, `path_stats.py`, `backup_spline_rl_controller.py`: belong to that earlier
  spline + RL attempt (`faster-laps` commit 080250f is the clean spline version).

**Do not push to `main`: that submits to the leaderboard.**
