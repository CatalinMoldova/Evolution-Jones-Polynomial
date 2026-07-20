"""Ablation bar chart for the J2+J3 run (results_control_j2j3.json).

All three variants share the identical j2j3 knee [27, 27] GELU and retrain
protocol (7 seeds x 500 epochs); only the six phase-evaluation columns
([Re, Im, |V(q0)|] per polynomial, q0 = e^{13 pi i/16} for J2 and
e^{5 pi i/9} for J3) differ: real values vs zeros vs Gaussian noise.
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

ctrl = json.load(open("results_control_j2j3.json"))["results"]

order = ["random", "zero", "phase"]
colors = [C_CTRL, C_ZERO, C_PHASE]
labels = ["6 random cols\n(noise control)", "6 zeroed cols\n(no phase eval)",
          "6 phase cols\n$V(q_0)$ for $J_2$, $J_3$"]

acc_ens = [ctrl[k]["acc_ensemble"] for k in order]
acc_mean = [ctrl[k]["acc_mean"] for k in order]
err = [ctrl[k]["test_mre_std"] for k in order]

x = np.arange(3)
fig, ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)

ax.bar(x, acc_ens, width=0.55, color=colors, zorder=3)
ax.errorbar(x, acc_mean, yerr=err, fmt="D", ms=7, color=INK, capsize=5,
            lw=1.4, zorder=4, label="7-seed mean $\\pm$ std")

for xi, a, c in zip(x, acc_ens, colors):
    ax.text(xi, a + 0.0008, f"{a*100:.2f}%", ha="center", va="bottom",
            fontsize=15, fontweight="bold", color=c)

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("test accuracy  (1 $-$ MRE, ensemble)")
lo = min(acc_mean) - 3 * max(err)
ax.set_ylim(lo, max(acc_ens) + 0.003)
ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.1f}%")
ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower right", fontsize=10)
ax.set_title("Control (measured): identical net, only the 6 phase columns vary\n"
             "j2j3 knee [27, 27] GELU fixed, 7 seeds $\\times$ 500 epochs",
             fontsize=13, fontweight="bold")
fig.savefig("figures/ab_ablation_j2j3.png", dpi=200, bbox_inches="tight")
print("wrote figures/ab_ablation_j2j3.png")
