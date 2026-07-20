"""Ablation bar chart: three MEASURED bars from the random-feature control.

All three variants (results_control.json) share the *identical* network — the
phase-run knee [32, 18, 54, 18] GELU — and the identical retrain protocol
(7 seeds, 500 epochs, log-target).  Only the content of the three extra
feature columns differs:

    zero    : columns zeroed        == J2-only reference
    random  : 3 Gaussian-noise cols == "any 3 extra inputs" control
    phase   : real V(e^{3 pi i/4})  == the theory-picked feature

So any difference between bars is purely the information in those columns.
Result: phase > zero > random — the gain is the phase signal, and noise
columns actually hurt slightly.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_CTRL, C_ZERO, C_PHASE = "#9aa3ab", "#0072B2", "#D55E00"
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.8, "text.color": INK,
                     "xtick.color": INK, "ytick.color": MUTED})

ctrl = json.load(open("results_control.json"))["results"]

order = ["random", "zero", "phase"]
colors = [C_CTRL, C_ZERO, C_PHASE]
labels = ["3 random cols\n(noise control)", "3 zeroed cols\n($J_2$ only)",
          "3 phase cols\n$V(e^{3\\pi i/4})$"]

acc_ens = [ctrl[k]["acc_ensemble"] for k in order]
acc_mean = [ctrl[k]["acc_mean"] for k in order]
err = [ctrl[k]["test_mre_std"] for k in order]

x = np.arange(3)
fig, ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)

ax.bar(x, acc_ens, width=0.55, color=colors, zorder=3)
ax.errorbar(x, acc_mean, yerr=err, fmt="D", ms=7, color=INK, capsize=5,
            lw=1.4, zorder=4, label="7-seed mean $\\pm$ std")

for xi, a, c in zip(x, acc_ens, colors):
    ax.text(xi, a + 0.0006, f"{a*100:.2f}%", ha="center", va="bottom",
            fontsize=15, fontweight="bold", color=c)

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("test accuracy  (1 $-$ MRE, ensemble)")
lo = min(acc_mean) - 3 * max(err)
ax.set_ylim(lo, max(acc_ens) + 0.002)
ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.1f}%")
ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower right", fontsize=10)
ax.set_title("Control (measured): identical net, only the 3 extra columns vary\n"
             "knee arch [32, 18, 54, 18] GELU fixed, 7 seeds $\\times$ 500 epochs",
             fontsize=13, fontweight="bold")
fig.savefig("figures/ab_ablation.png", dpi=200, bbox_inches="tight")
print("wrote figures/ab_ablation.png")
