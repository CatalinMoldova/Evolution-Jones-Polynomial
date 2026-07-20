"""Generate two presentation figures for the phase-feature experiment.

  figures/feature_correlation.png : log|V(e^{3pi i/4})| vs log volume (strong)
                                    next to L1 norm vs log volume (weak).
  figures/pareto_overlay.png      : 42-feat vs 45-feat+phase Pareto fronts.

Run:  ~/.conda/envs/nsga2/bin/python make_presentation_plots.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parse_data import load_raw, build_features

DATA = "jones+volume.txt"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito colorblind-safe pair (fixed order: baseline, then treatment).
C_BASE = "#0072B2"   # blue  -> 42-feat baseline
C_PHASE = "#D55E00"  # vermillion -> 45-feat + phase
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"

plt.rcParams.update({
    "font.size": 12,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8,
    "axes.titlesize": 13,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.dpi": 130,
})


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------- figure 1
def feature_correlation():
    _, C, y = load_raw(DATA)
    X = build_features(C)
    logvol = np.log(y)
    abs_q0 = X[:, 10]          # |V(q0)| summary feature
    l1 = X[:, 4]               # L1 norm summary feature
    log_absq0 = np.log(abs_q0)

    r_phase = pearson(log_absq0, logvol)
    r_l1 = pearson(l1, logvol)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

    panels = [
        (axes[0], log_absq0, r"$\log\,|V(e^{3\pi i/4})|$", r_phase, C_PHASE,
         "Phase feature (theory-motivated)"),
        (axes[1], l1, r"$L_1$ norm  $\sum_k |c_k|$", r_l1, MUTED,
         "Previous best feature"),
    ]
    for ax, xvals, xlabel, r, color, sub in panels:
        ax.scatter(xvals, logvol, s=5, c=color, alpha=0.18, linewidths=0,
                   rasterized=True)
        # least-squares guide line
        m, b = np.polyfit(xvals, logvol, 1)
        xs = np.linspace(xvals.min(), xvals.max(), 100)
        ax.plot(xs, m * xs + b, color=INK, lw=1.6, alpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(sub, color=MUTED, fontsize=11, pad=8)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(0.04, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="top", color=color)
    axes[0].set_ylabel(r"$\log$ (hyperbolic volume)")
    fig.suptitle("A single phase evaluation of $J_2$ nearly tracks the volume",
                 fontsize=15, fontweight="bold")
    path = os.path.join(OUT, "feature_correlation.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] r(log|V(q0)|)={r_phase:.4f}  r(L1)={r_l1:.4f} -> {path}")


# ---------------------------------------------------------------- figure 2
def pareto_overlay():
    def load_front(path):
        r = json.load(open(path))
        pf = sorted(r["pareto_front"], key=lambda d: d["n_params"])
        p = np.array([d["n_params"] for d in pf], float)
        v = np.array([d["f1_val_mre"] for d in pf], float)
        knee = r["knee"]
        return p, v, knee["n_params"] if "n_params" in knee else \
            10 ** knee["f2_log10_params"], knee["f1_val_mre"]

    p42, v42, kp42, kv42 = load_front("results/results.json")
    p45, v45, kp45, kv45 = load_front("results_phase/results.json")

    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for p, v, color, label in [
        (p42, v42, C_BASE, "42 features ($J_2$ only)"),
        (p45, v45, C_PHASE, "45 features (+ phase)"),
    ]:
        ax.plot(p, v, "-o", color=color, lw=2, ms=6, alpha=0.95, label=label,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.6)

    # mark knees
    for kp, kv, color in [(kp42, kv42, C_BASE), (kp45, kv45, C_PHASE)]:
        ax.scatter([kp], [kv], s=180, facecolors="none", edgecolors=color,
                   linewidths=2.2, zorder=5)
    ax.annotate("knee", (kp45, kv45), textcoords="offset points",
                xytext=(8, 10), color=C_PHASE, fontsize=11, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("model size (parameters, log scale)")
    ax.set_ylabel("validation MRE")
    ax.grid(True, which="both", color=GRID, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=12)
    ax.set_title("Adding the phase feature lowers the whole Pareto front\n"
                 "(front stays flat: the $J_2$ information ceiling, not the model)",
                 fontsize=13, fontweight="bold")
    path = os.path.join(OUT, "pareto_overlay.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] 42-feat floor={v42.min():.4f}  45-feat floor={v45.min():.4f} -> {path}")


if __name__ == "__main__":
    feature_correlation()
    pareto_overlay()
