"""q0 phase sweep on the colored-Jones dataset: J2 vs J3 curves.

For each r in {1 (J2), 2 (J3)} sweep theta over (0, pi] and measure the Pearson
correlation of log|J(e^{i theta})| with log(hyperbolic volume) across the 1,419
hyperbolic knots of data_j2+j3.  Marks the per-r phases used by
parse_j2j3.EVAL_PHASES (13pi/16 and 5pi/9) and the literature J2 phase 3pi/4.

Run:  ~/.conda/envs/nsga2/bin/python make_phase_sweep_j2j3.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parse_j2j3 import EVAL_PHASES, TORUS_KNOTS, load_rows, load_volumes

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

C_J2 = "#0072B2"      # blue
C_J3 = "#009E73"      # green
C_MARK = "#D55E00"    # vermillion
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"

plt.rcParams.update({"font.size": 12, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.8, "text.color": INK,
                     "xtick.color": INK, "ytick.color": MUTED})


def sweep_r(rows, vols, r_wanted, thetas):
    polys = [np.asarray(row["canon"], dtype=np.float64) for row in rows
             if row["r"] == r_wanted and row["knot"] in vols
             and row["knot"] not in TORUS_KNOTS]
    logvol = np.log([vols[row["knot"]] for row in rows
                     if row["r"] == r_wanted and row["knot"] in vols
                     and row["knot"] not in TORUS_KNOTS])
    r = np.empty_like(thetas)
    for i, th in enumerate(thetas):
        absV = np.array([abs((c * np.exp(1j * th * np.arange(len(c)))).sum())
                         for c in polys])
        logabs = np.log(np.maximum(absV, 1e-9))
        r[i] = np.corrcoef(logabs, logvol)[0, 1]
    return r


def main():
    rows = load_rows()
    vols = load_volumes()
    thetas = np.linspace(0.02 * np.pi, np.pi, 400)

    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    for rr, color, label in [(1, C_J2, r"$J_2$ (classical Jones)"),
                             (2, C_J3, r"$J_3$ (adjoint)")]:
        r = sweep_r(rows, vols, rr, thetas)
        ax.plot(thetas / np.pi, r, color=color, lw=2.4, zorder=3, label=label)

        th0 = EVAL_PHASES[rr]
        i0 = int(np.argmin(np.abs(thetas - th0)))
        imax = int(np.nanargmax(r))  # corr is nan where |J| has zero variance (theta=pi)
        print(f"[sweep] r={rr}: corr at chosen phase {th0/np.pi:.4f}pi = {r[i0]:.4f}; "
              f"argmax {r[imax]:.4f} at {thetas[imax]/np.pi:.4f}pi")
        ax.scatter([th0 / np.pi], [r[i0]], s=90, color=C_MARK, zorder=5,
                   edgecolor="white", linewidths=1.4)
        frac = r"13\pi/16" if rr == 1 else r"5\pi/9"
        ax.annotate(rf"$q_0 = e^{{i\,{frac}}}$" + f"\n$r = {r[i0]:.3f}$",
                    (th0 / np.pi, r[i0]), textcoords="offset points",
                    xytext=(10, -10), color=C_MARK, fontsize=12,
                    fontweight="bold", va="top")

    # literature J2 phase for reference
    ax.axvline(0.75, color=MUTED, lw=1.2, ls="--", alpha=0.8, zorder=2)
    ax.annotate(r"lit. $3\pi/4$", (0.75, ax.get_ylim()[0]),
                textcoords="offset points", xytext=(4, 8),
                color=MUTED, fontsize=11)

    ax.set_xlabel(r"phase  $\theta / \pi$   (evaluation point $q = e^{i\theta}$)")
    ax.set_ylabel(r"corr$(\log|J(e^{i\theta})|,\ \log\,\mathrm{vol})$")
    ax.set_xlim(0, 1)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", frameon=False)
    ax.set_title("Per-polynomial evaluation phases on the colored-Jones dataset\n"
                 "(1,419 hyperbolic knots; markers = phases used as features)",
                 fontsize=13, fontweight="bold")
    path = os.path.join(OUT, "phase_sweep_j2j3.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[sweep] -> {path}")


if __name__ == "__main__":
    main()
