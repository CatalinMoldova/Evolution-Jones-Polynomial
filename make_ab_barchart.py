"""A/B accuracy bar chart (ensemble test accuracy, matched budget)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_BASE, C_PHASE = "#0072B2", "#D55E00"
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.8, "text.color": INK,
                     "xtick.color": INK, "ytick.color": MUTED})

b = json.load(open("results_matched_42feat.json"))["final"]
p = json.load(open("results_phase/results.json"))["final"]

# accuracy = 1 - MRE, ensemble; also keep mean+/-std for error bars
acc = [1 - b["test_mre_ensemble"], 1 - p["test_mre_ensemble"]]
acc_mean = [1 - b["test_mre_mean"], 1 - p["test_mre_mean"]]
err = [b["test_mre_std"], p["test_mre_std"]]
labels = ["42 features\n($J_2$ only)", "45 features\n(+ phase)"]
colors = [C_BASE, C_PHASE]

fig, ax = plt.subplots(figsize=(6.6, 5.4), constrained_layout=True)
x = np.arange(2)
ax.bar(x, acc, width=0.55, color=colors, zorder=3)
# mean+/-std markers overlaid for honesty
ax.errorbar(x, acc_mean, yerr=err, fmt="D", ms=7, color=INK,
            capsize=5, lw=1.4, zorder=4, label="7-seed mean $\\pm$ std")

for xi, a in zip(x, acc):
    ax.text(xi, a + 0.0006, f"{a*100:.2f}%", ha="center", va="bottom",
            fontsize=15, fontweight="bold", color=colors[xi])

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("test accuracy  (1 $-$ MRE, ensemble)")
ax.set_ylim(0.965, 0.9745)
ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.1f}%")
ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower right", fontsize=10)
ax.set_title("Same NAS budget, one added feature:\n~6% relative error reduction",
             fontsize=13, fontweight="bold")
fig.savefig("figures/ab_accuracy.png", dpi=200, bbox_inches="tight")
print("ensemble acc:", [f"{a*100:.2f}%" for a in acc],
      "mean:", [f"{a*100:.2f}%" for a in acc_mean])
