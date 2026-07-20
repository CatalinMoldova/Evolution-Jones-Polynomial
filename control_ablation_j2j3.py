"""Random-feature control for the J2+J3 run (matched architecture + protocol).

Question (README §3 step 4): is the J2+J3 gain carried by the *phase-evaluation
signal*, or would any extra columns do?  We fix the j2j3 run's knee architecture
and the full retrain protocol (7 seeds, 500 epochs, log-target, split seed 0)
and vary ONLY the six phase-evaluation columns — [Re, Im, |V(q0)|] inside each
per-r summary block (q0 = e^{13πi/16} for J2, e^{5πi/9} for J3):

    phase   : the real evaluations                     (expect best)
    random  : 6 Gaussian-noise columns                 (control)
    zero    : 6 dead (zeroed) columns                  (no-phase reference)

All three share the identical 68-input network and training, so any difference
is purely the content of those six columns.  Writes results_control_j2j3.json.

Run (compute node):  sbatch submit_ablation_j2j3.slurm
"""
from __future__ import annotations

import json
import time

import numpy as np

from parse_data import DataBundle, make_splits
from parse_j2j3 import N_SUMMARY, build_matrix, load_rows, poly_widths
from nas import genome_to_config
from run_nas import final_retrain

FEATURE_SET = "j2j3"
R_WANTED = [1, 2]
SEEDS = list(range(7))
EPOCHS, PATIENCE = 500, 50


def phase_columns():
    """Indices of [Re, Im, |V(q0)|] within the concatenated j2j3 vector."""
    widths = poly_widths(load_rows())
    cols, start = [], 0
    for r in R_WANTED:
        cols += [start + 6, start + 7, start + 8]
        start += N_SUMMARY + widths[r]
    return cols


def make_bundle(X, y, tr, va, te):
    xm, xs = X[tr].mean(0), X[tr].std(0)
    xs[xs < 1e-8] = 1.0
    std = lambda a: (a - xm) / xs
    return DataBundle(
        std(X[tr]).astype("float32"), std(X[va]).astype("float32"),
        std(X[te]).astype("float32"),
        y[tr].astype("float64"), y[va].astype("float64"), y[te].astype("float64"),
        float(np.log(y[tr]).mean()), float(np.log(y[tr]).std() + 1e-8),
        xm, xs, log_target=True)


def main():
    knee = json.load(open("results_j2j3_j2j3/results.json"))["knee"]
    cfg = genome_to_config(knee["genome"])
    print(f"[arch] j2j3 knee: depth={cfg['depth']} widths={cfg['widths']} "
          f"{cfg['activation']}")

    _, X0, y = build_matrix(FEATURE_SET)
    cols = phase_columns()
    print(f"[data] X {X0.shape}, phase cols {cols}")
    tr, va, te = make_splits(len(X0), seed=0)

    rng = np.random.default_rng(0)
    variants = {
        "phase": X0.copy(),
        "random": X0.copy(),
        "zero": X0.copy(),
    }
    variants["random"][:, cols] = rng.standard_normal((len(X0), len(cols)))
    variants["zero"][:, cols] = 0.0

    out = {"arch": cfg, "phase_cols": cols, "seeds": SEEDS, "epochs": EPOCHS,
           "results": {}}
    for name, X in variants.items():
        d = make_bundle(X, y, tr, va, te)
        t0 = time.time()
        r = final_retrain(cfg, d, epochs=EPOCHS, seeds=SEEDS,
                          patience=PATIENCE, device="cpu")
        r.pop("ens_test", None)
        acc_ens = 1 - r["test_mre_ensemble"]
        acc_mean = 1 - r["test_mre_mean"]
        out["results"][name] = {
            "test_mre_mean": r["test_mre_mean"],
            "test_mre_std": r["test_mre_std"],
            "test_mre_ensemble": r["test_mre_ensemble"],
            "acc_ensemble": acc_ens,
            "acc_mean": acc_mean,
        }
        print(f"[{name:>6}] ensemble acc={acc_ens*100:.2f}%  "
              f"mean={acc_mean*100:.2f}% +/- {r['test_mre_std']*100:.2f}  "
              f"({time.time()-t0:.0f}s)")

    with open("results_control_j2j3.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("[done] -> results_control_j2j3.json")


if __name__ == "__main__":
    main()
