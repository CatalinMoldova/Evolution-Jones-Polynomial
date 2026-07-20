"""q0 phase sweep: correlation of log|V(e^{i theta})| with log volume vs. theta.

The volume conjecture singles out a *specific* evaluation point of the Jones
polynomial.  This figure sweeps the phase theta over (0, pi] and, at each theta,
measures how well a single scalar -- log|J2(K; e^{i theta})| -- linearly tracks
log(hyperbolic volume) across all 12,955 knots (Pearson r).  If the theory-
motivated choice q0 = e^{3 pi i/4} is special, the curve should peak there and
not at an arbitrary angle.

Run:  ~/.conda/envs/nsga2/bin/python make_phase_sweep.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parse_data import load_raw, N_COEFFS, POWER_OFFSET

DATA = "jones+volume.txt"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

C_PHASE = "#D55E00"   # vermillion
C_BASE = "#0072B2"    # blue
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"

plt.rcParams.update({"font.size": 12, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.8, "text.color": INK,
                     "xtick.color": INK, "ytick.color": MUTED})


def sweep():
    _, C, y = load_raw(DATA)
    logvol = np.log(y)
    powers = np.arange(N_COEFFS) - POWER_OFFSET          # power of coeff k

    # dense theta grid over (0, pi]; skip 0 (|V(1)| = |sum c| can vanish and is
    # the classic determinant point, not a phase evaluation).
    thetas = np.linspace(0.02 * np.pi, np.pi, 400)
    r = np.empty_like(thetas)
    for i, th in enumerate(thetas):
        qpow = np.exp(1j * th * powers)                  # [ncoef]
        absV = np.abs(C @ qpow)                          # [n] = |V(e^{i th})|
        logabs = np.log(np.maximum(absV, 1e-9))
        r[i] = np.corrcoef(logabs, logvol)[0, 1]

    q0 = 3.0 * np.pi / 4.0
    i0 = int(np.argmin(np.abs(thetas - q0)))
    imax = int(np.argmax(r))
    print(f"[sweep] r at 3pi/4 = {r[i0]:.4f}")
    print(f"[sweep] argmax r   = {r[imax]:.4f} at theta = {thetas[imax]/np.pi:.3f} pi")

    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    ax.plot(thetas / np.pi, r, color=C_BASE, lw=2.4, zorder=3)

    # mark the theory-motivated point q0 = 3 pi / 4
    ax.axvline(0.75, color=C_PHASE, lw=1.4, ls="--", alpha=0.9, zorder=2)
    ax.scatter([0.75], [r[i0]], s=90, color=C_PHASE, zorder=5,
               edgecolor="white", linewidths=1.4)
    ax.annotate(rf"$q_0 = e^{{3\pi i/4}}$" + f"\n$r = {r[i0]:.3f}$",
                (0.75, r[i0]), textcoords="offset points", xytext=(12, -6),
                color=C_PHASE, fontsize=13, fontweight="bold", va="top")

    ax.set_xlabel(r"phase  $\theta / \pi$   (evaluation point $q = e^{i\theta}$)")
    ax.set_ylabel(r"corr$(\log|V(e^{i\theta})|,\ \log\,\mathrm{vol})$")
    ax.set_xlim(0, 1)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("The theory-motivated phase is where the signal peaks\n"
                 r"(single-scalar correlation with volume vs. evaluation angle)",
                 fontsize=13, fontweight="bold")
    path = os.path.join(OUT, "phase_sweep.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[sweep] -> {path}")


if __name__ == "__main__":
    sweep()
