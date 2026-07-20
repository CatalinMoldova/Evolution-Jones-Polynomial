"""Random-feature control (matched architecture + protocol).

Question: is the +phase gain the *phase signal*, or would any 3 extra inputs
help?  We fix the phase-run's knee architecture and the full retrain protocol
(7 seeds, 500 epochs, log-target, split seed 0) and vary ONLY the three extra
feature columns (summary indices 8,9,10 = [Re, Im, |V(q0)|]):

    phase   : the real phase evaluation V(e^{3 pi i/4})   (expect best)
    random  : 3 Gaussian-noise columns                    (control)
    zero    : 3 dead (zeroed) columns == J2-only          (reference)

All three share the identical 45-input network and training, so any difference
is purely the content of those three columns.  Writes results_control.json.

Run:  ~/.conda/envs/nsga2/bin/python control_ablation.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from parse_data import load_raw, build_features, make_splits, DataBundle
from nas import genome_to_config
from run_nas import final_retrain

DATA = "jones+volume.txt"
PHASE_COLS = [8, 9, 10]   # Re, Im, |V(q0)| within the 45-dim feature vector
SEEDS = list(range(7))
EPOCHS, PATIENCE = 500, 50


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
    knee = json.load(open("results_phase/results.json"))["knee"]
    cfg = genome_to_config(knee["genome"])
    print(f"[arch] phase knee: depth={cfg['depth']} widths={cfg['widths']} "
          f"{cfg['activation']}")

    _, C, y = load_raw(DATA)
    X0 = build_features(C)                      # [n,45], real phase in cols 8-10
    tr, va, te = make_splits(len(X0), seed=0)

    rng = np.random.default_rng(0)
    variants = {
        "phase": X0.copy(),
        "random": X0.copy(),
        "zero": X0.copy(),
    }
    variants["random"][:, PHASE_COLS] = rng.standard_normal((len(X0), 3))
    variants["zero"][:, PHASE_COLS] = 0.0

    out = {"arch": cfg, "seeds": SEEDS, "epochs": EPOCHS, "results": {}}
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

    with open("results_control.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("[done] -> results_control.json")


if __name__ == "__main__":
    main()
