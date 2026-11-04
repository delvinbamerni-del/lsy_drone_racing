"""Write trained policy weights into rl_controller.py as a literal array (dev-only).

    pixi run python rl_spline/embed.py rl_spline/runs/es1/best.npy
    pixi run python rl_spline/embed.py --zero        # back to the plain spline controller
"""

import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "lsy_drone_racing/control/rl_controller.py"
PATTERN = re.compile(r"^POLICY_PARAMS = .*?(?=\n\n\n)", re.S | re.M)

if sys.argv[1] == "--zero":
    line = "POLICY_PARAMS = np.zeros(N_PARAMS)"
else:
    p = np.load(sys.argv[1])
    body = ",\n".join(
        "    " + ", ".join(f"{v:.6f}" for v in p[i:i + 8]) for i in range(0, len(p), 8)
    )
    line = f"POLICY_PARAMS = np.array([\n{body},\n])  # from {Path(sys.argv[1]).relative_to(REPO)}"
src = TARGET.read_text()
new, n = PATTERN.subn(lambda _: line, src)
assert n == 1, "POLICY_PARAMS line not found"
TARGET.write_text(new)
print(f"wrote {line.splitlines()[0]} ... into {TARGET.relative_to(REPO)}")
