"""Headline J2/J3 bar chart: three matched-budget NAS runs on the same knots.

Reads results_j2j3_{j2,j3,j2j3}/results.json — identical NSGA-II budget
(pop 40 / gens 40 / 4 search seeds / 7 final seeds x 500 epochs), identical
1,419-knot split — only the input polynomial differs.  Bars = 7-seed ensemble
accuracy; diamonds = per-seed mean +/- std; the knee size is annotated in
each bar.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_J2, C_J3, C_BOTH = "#9aa3ab", "#0072B2", "#D55E00"
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.8, "text.color": INK,
                     "xtick.color": INK, "ytick.color": MUTED})

runs = {k: json.load(open(f"results_j2j3_{k}/results.json"))
        for k in ("j2", "j3", "j2j3")}

order = ["j2", "j3", "j2j3"]
colors = [C_J2, C_J3, C_BOTH]
labels = ["$J_2$ only\n(fundamental)", "$J_3$ only\n(adjoint)",
          "$J_2 + J_3$\n(joined)"]

acc_ens = [1 - runs[k]["final"]["test_mre_ensemble"] for k in order]
acc_mean = [1 - runs[k]["final"]["test_mre_mean"] for k in order]
err = [runs[k]["final"]["test_mre_std"] for k in order]
params = [runs[k]["final"]["n_params"] for k in order]
knees = [runs[k]["knee"]["arch"]["widths"] for k in order]

x = np.arange(3)
fig, ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)

ax.bar(x, acc_ens, width=0.55, color=colors, zorder=3)
ax.errorbar(x, acc_mean, yerr=err, fmt="D", ms=7, color=INK, capsize=5,
            lw=1.4, zorder=4, label="7-seed mean $\\pm$ std")

lo = min(acc_mean) - 3 * max(err)
for xi, a, p, w, c in zip(x, acc_ens, params, knees, colors):
    ax.text(xi, a + 0.0012, f"{a*100:.2f}%", ha="center", va="bottom",
            fontsize=15, fontweight="bold", color=c)
    ax.text(xi, lo + 0.0035, f"knee {w}\n{p:,} params", ha="center",
            va="bottom", fontsize=10, color="white", fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("test accuracy  (1 $-$ MRE, ensemble)")
ax.set_ylim(lo, max(acc_ens) + 0.006)
ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.1f}%")
ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower right", fontsize=10)
ax.set_title("Same 1,419 knots, same split, same NAS budget —\n"
             "only the input polynomial differs",
             fontsize=13, fontweight="bold")
fig.savefig("figures/j2j3_accuracy.png", dpi=200, bbox_inches="tight")
print("wrote figures/j2j3_accuracy.png")
