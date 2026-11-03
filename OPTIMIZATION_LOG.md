# Optimization log — branch `faster-laps`

## 1. How the competition works

The upstream `competition` branch (learnsyslab/lsy_drone_racing) was fetched into the local ref
`upstream/competition`. It was **not** merged and nothing was pushed.

- **Trigger:** pushing to `main` of the team fork runs `.github/workflows/competition.yml`. That
  workflow sends the commit to the central repo. Since we never push, nothing gets submitted.
- **Anti-cheat (`scripts/check_cheat.py`):** `scripts/evaluate.py`, `scripts/sim.py` and
  `.github/workflows/competition.yml` must match upstream `main` byte for byte.
  `config/level2.toml` must also match, except **line 7** (`[controller] file = ...`) and
  **line 48** (`control_mode = ...`), and its line count must not change. Our branch only
  changes those two lines.
- **Evaluation:** `pixi run evaluate` runs `scripts/evaluate.py`: **20 runs of `config/level2.toml`**,
  with `seed = -1` (a new random seed every time) and no rendering. It runs in the student's
  locked pixi environment, so any new package must also be added to `pixi.lock`.
- **Level 2 track:** 4 gates flown in the order `[1, 2, 3, 4, 2]`, so 5 gate passes. There are
  4 poles. Each run randomizes gate positions by ±0.15 m (x, y) and ±0.1 m (z), gate yaw by
  ±0.2 rad, pole positions by ±0.15 m, plus small changes to the drone's mass, inertia, start
  position and start attitude, and action/dynamics noise. The exact gate and pole positions
  are only revealed within 0.7 m. Steps run at 50 Hz and the sim at 500 Hz.
- **Score:** `evaluation.csv` = (mean lap time of the *successful* runs, success rate).
  A submission is valid only if success ≥ 50%. **Ranking is by time only.** A new submission
  only replaces the team's entry if its time beats the team's previous best.
- **Leaderboard (fetched 2026-11-04):**
  1. BreakFaster **3.268 s** (65%)
  2. Narges and Eram 3.321 s (85%)
  3. Nova-YJ 3.328 s (75%)
  4. Überflieger 3.343 s (80%)
  5. SkyX 4.011 s (55%)

  Our team, *Barbie Drone* (`delvinbamerni-del`), is 11th with **7.375 s (85%)**. The Summer 2026
  winner had 3.419 s (100%).

## 2. Method

- I only change the controller file and lines 7/48 of `config/level2.toml`. The simulator,
  environments, track, `sim.py` and `evaluate.py` stay untouched.
- **Screening:** a scratch harness runs the exact same env/config/loop as `sim.py` and scores
  the same way (mean successful time, success rate), over 20 runs with a **fixed seed**. Every
  variant then flies the same 20 random tracks, so differences come from the change rather
  than from luck.
- **Confirmation:** before a commit, every candidate best is re-checked with the official
  `pixi run evaluate` (20 runs, random seeds).
- **Rule:** keep a change only if the time improves and success stays ≥ 50% (target 70%+).

## 3. Baseline

Controller `gate_spline_controller.py` in attitude mode (commit `a5b93ae`).

| Eval | Time [s] | Success |
|---|---|---|
| official `evaluate.py` | 11.677 | 90% |
| harness, seed 1 | 11.771 | 65% |

Planned path: about 12 m long, at about 1 m/s average. Cruise speed is 1.5 m/s and gate
passages are 0.6–0.8 m/s.

## 4. Attempts

| # | Change (one at a time) | Time [s] | Success | Kept? |
|---|---|---|---|---|
| 1 | `SPEED` 1.5 → 1.8 | 10.72 | 40% | reverted |
| 2 | `SPEED` 1.5 → 2.1 | 10.06 | 5% | reverted |
| 3 | `SPEED` 1.5 → 2.4 / 2.8 | – | 0% | reverted |

| 4 | `GATE_SPEED` 0.8 → 1.2 | 10.45 | 60% | reverted (success drop) |
| 5 | `ACC_MAX` 100 → 15 | 12.04 | 80% | reverted (slower) |
| 6 | `control_mode` state | 11.97 | 50% | reverted |
| 7 | higher gains KP 20/25, KD 8/10 | 11.79 | 70% | no time gain |
| 8 | exploratory combinations (speeds + `ACC_MAX` + gains) | ≥10.4 | 0–45% | reverted |

Diagnosis after attempts 1–3: at higher speed the drone crashes within 0.1–0.3 m of a gate
centre while 10–35 cm off the planned path. So it is clipping gate frames because tracking
breaks down when the speed changes abruptly between cruise and the slow gate segments, not
because the line itself is wrong.

**Conclusion:** `gate_spline_controller.py` gives each segment a fixed time and fits a cubic
spline in time. Speeding up any segment causes acceleration spikes, and the clock-driven
reference runs away from the drone. Knob tuning stalled above 10 s, so I switched approach.

### New approach: `racing_line_controller.py`

- Same racing line topology (via points around the poles).
- The line is a cubic spline in **arc length**, so its shape does not depend on speed.
- Speed comes from a **curvature-limited profile**: sqrt(A_LAT·r), with forward and backward
  passes for acceleration and braking.
- The reference advances along the line through a **governor** that slows down when the
  tracking error grows.
- **Re-planning** rebuilds the whole lap with the updated gate and pole positions. The reference
  continues at the closest point of the new line, and the jump fades out.

From here on, screening uses fixed seeds 1 and 2 (20 runs each, 40 runs combined).

| # | Change (vs. previous kept state) | Time [s] | Success | Kept? |
|---|---|---|---|---|
| 9 | first version (V_MAX 2.5, A_LAT 6, PID) | 7.27 | 20% (seed 1) | base |
| 10 | V_MAX 1.5 / 2.0 / 3.0 | 7.7–9.0 | 5–15% | – |
| 11 | re-plan whole lap + fading offset (fixes hooks in the line) | 7.27 | 20% | kept |
| 12 | `OFFSET_TAU` 0.08 / 0.02 | – | 0% | reverted |
| 13 | `V_GATE` 1.5 / 1.0 | 8.1 / 9.2 | 10% / 20% | reverted |
| 14 | `A_LAT` 3.5 / 2.0 | 8.5 / 10.2 | 30% / 60% | reverted (slow) |
| 15 | governor off / weaker, `ACC_FF` 0.7, tilt limit 35° | – | 0–10% | reverted |
| 16 | state mode for the new line | 7.9 | 0–5% | reverted |
| 17 | **acceleration feedforward from 0.12 s ahead** (`FF_LEAD`) | 6.96 | 37.5% (40 runs) | **kept** |
| – | official `evaluate.py` with #17 | – | 30% | not committed |
| 18 | `FF_LEAD` 0.16, `A_LAT` 4.5, `APPROACH_DIST` 0.45 | 6.98–7.53 | 37–42% | reverted |
| 19 | KP 18/25 + KD 8/10 | 6.84 | 27.5% | reverted |
| 20 | `V_GATE` 1.8, `OFFSET_TAU` 0.12 | 7.02–7.09 | 42.5% | reverted |
| 21 | **KD 6/8 → 8/10** | 6.83 | 52.5% | **kept** |
| 22 | **KD → 10/12** | 6.78 | 57.5% | **kept** |
| 23 | KD 12/12, KP 9/20, no xy integrator | 6.82–6.86 | 47–50% | reverted |
| 24 | via point south of pole 1 on the start leg (0.35 / 0.4 m) | 7.0 | 40–42% | reverted |
| 25 | first-gate approach angle 30° / 10° | 6.8–6.9 | 42–45% | reverted |
| 26 | MPC tracker (acados, so_rpy model) instead of PID | 7.15 | 52.5% | not yet |
| 27 | MPC weight variants | 7.04–7.40 | 20–50% | reverted |
| 28 | offset fades out by distance, gone 0.15 m before the gate (PID) | 6.72 | 50% | kept (needed for #29) |
| 29 | **MPC + offset gone 0.3 m before the gate** | 7.13 | 62.5% | **kept** |
| – | official `evaluate.py` with #29, 4 runs | 7.09–7.15 | 40 / 55 / 65 / 45% (≈51%) | checkpoint commit |

Main failure points so far: the first gate is revealed only about 0.3 s before the drone reaches
it (so the re-plan offset was still in the reference at the gate), and tracking errors of
0.1 m or more against a lateral tolerance of only about ±0.12 m.
