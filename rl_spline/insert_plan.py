"""Insert an exported plan (export_plan.py output) between the PLAN markers of rl_controller.py (dev-only).

    pixi run python rl_spline/insert_plan.py rl_spline/runs/plan_tilt30.txt
"""
import re, sys
from pathlib import Path
target = Path(__file__).resolve().parents[1] / "lsy_drone_racing/control/rl_controller.py"
plan = Path(sys.argv[1]).read_text().strip()
src = target.read_text()
pat = re.compile(r"(# --- PLAN START[^\n]*\n).*?(# --- PLAN END)", re.S)
new, n = pat.subn(lambda m: m.group(1) + f"# source: {Path(sys.argv[1]).name}\n" + plan + "\n" + m.group(2), src)
assert n == 1, "markers not found"
target.write_text(new)
print("inserted", sys.argv[1])
