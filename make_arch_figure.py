"""Network-architecture diagram for the reported knee model.

45 inputs -> 32 -> 18 -> 54 -> 18 -> 1  (GELU), 4,101 params.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, VERM = "#0072B2", "#D55E00"
INK, MUTED, GRID = "#1a1a1a", "#5b5b5b", "#d9d9d9"

r = json.load(open("results_phase/results.json"))
widths = r["knee"]["arch"]["widths"]           # [32, 18, 54, 18]
nparams = r["final"]["n_params"]
layers = [("Input", 45, "45 J₂ features")] + \
         [("Hidden", w, "") for w in widths] + \
         [("Output", 1, "volume")]

fig, ax = plt.subplots(figsize=(11.2, 4.4))
ax.set_xlim(0, len(layers)); ax.set_ylim(0, 10)
ax.axis("off")

# column x-centers
xs = np.linspace(0.6, len(layers) - 0.6, len(layers))
# visual heights: sqrt-scaled so 45 and 1 are both readable
def vh(w):
    return 1.2 + 6.4 * (np.sqrt(w) / np.sqrt(45))
heights = [vh(w) for _, w, _ in layers]
cy = 5.4

# connecting bands (draw first, behind)
for i in range(len(layers) - 1):
    x0, x1 = xs[i] + 0.42, xs[i + 1] - 0.42
    h0, h1 = heights[i] / 2, heights[i + 1] / 2
    ax.fill([x0, x1, x1, x0],
            [cy + h0, cy + h1, cy - h1, cy - h0],
            color=GRID, alpha=0.55, zorder=1, edgecolor="none")

for i, ((role, w, lab), x, h) in enumerate(zip(layers, xs, heights)):
    is_io = role in ("Input", "Output")
    color = BLUE if is_io else VERM
    box = FancyBboxPatch((x - 0.34, cy - h / 2), 0.68, h,
                         boxstyle="round,pad=0.02,rounding_size=0.12",
                         linewidth=0, facecolor=color, alpha=0.92, zorder=3)
    ax.add_patch(box)
    # width count centered in the box
    ax.text(x, cy, str(w), ha="center", va="center", color="white",
            fontsize=17 if not is_io else 15, fontweight="bold", zorder=4)
    # role label above
    ax.text(x, cy + h / 2 + 0.55, role, ha="center", va="bottom",
            color=MUTED, fontsize=12, zorder=4)
    # description below
    ax.text(x, cy - h / 2 - 0.55, lab, ha="center", va="top",
            color=INK, fontsize=11, zorder=4)

# arrow strip label
ax.text(xs.mean(), 0.35,
        "fully-connected · GELU activation · depth 4",
        ha="center", va="center", color=MUTED, fontsize=12, style="italic")

fig.suptitle(f"Reported model — Pareto knee ({nparams:,} parameters)",
             fontsize=16, fontweight="bold", color=INK, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("figures/architecture.png", dpi=200, bbox_inches="tight")
print(f"saved figures/architecture.png  widths={widths} params={nparams}")
