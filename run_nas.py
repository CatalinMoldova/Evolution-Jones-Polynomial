"""Run the NSGA-II neural architecture search end to end.

Pipeline:
  1. parse + split the data (train/val/test, fixed seed; test held out),
  2. run NSGA-II over the mixed-variable genome, evaluating each candidate
     under a cheap multi-fidelity budget on train+val only,
  3. plot the (val MRE vs log10 params) Pareto front and pick the knee,
  4. retrain the knee architecture at full budget over several seeds and
     report mean-relative-error on the untouched TEST set.

Example:
    python run_nas.py --data jones+volume.txt --pop 24 --gens 20 --n-jobs 24
"""
from __future__ import annotations

# Cap thread pools *before* importing numpy/torch so no worker ever grabs one
# thread per core (that oversubscribes CPU-parallel evaluation and, on a login
# node, can exhaust the per-user process limit).
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless / compute-node safe
import matplotlib.pyplot as plt

import torch

from parse_data import prepare
from nas import (
    Budget, EvalConfig, NASProblem, count_params, fit, predict,
    genome_to_config, set_data, evaluate_genome, MAX_DEPTH,
)

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import (
    MixedVariableSampling, MixedVariableMating, MixedVariableDuplicateElimination,
)
from pymoo.parallelization import StarmapParallelization
from pymoo.optimize import minimize
from pymoo.termination import get_termination


# ---------------------------------------------------------------------------
# Knee-point selection
# ---------------------------------------------------------------------------
def knee_index(F: np.ndarray) -> int:
    """Index of the Pareto-front knee (max distance from the extremes line).

    Objectives are min-max normalized to [0,1]; the knee is the point with the
    greatest perpendicular distance from the chord joining the two extreme
    solutions -- the classic "most bang for the buck" trade-off point.
    """
    F = np.asarray(F, dtype=float)
    if len(F) == 1:
        return 0
    order = np.argsort(F[:, 0])
    Fs = F[order]

    span = Fs.max(axis=0) - Fs.min(axis=0)
    span[span < 1e-12] = 1.0
    N = (Fs - Fs.min(axis=0)) / span

    p0, p1 = N[0], N[-1]
    line = p1 - p0
    L = np.linalg.norm(line)
    if L < 1e-12:
        return int(order[0])
    # perpendicular distance of every point to the p0->p1 chord.  2-D cross
    # product written out explicitly (np.cross on 2-vectors is deprecated).
    v = N - p0
    d = np.abs(line[0] * v[:, 1] - line[1] * v[:, 0]) / L
    return int(order[int(np.argmax(d))])


def genome_from_X(x: dict) -> dict:
    """A plain-Python copy of a pymoo individual's variable dict."""
    g = {}
    for k, v in x.items():
        if hasattr(v, "item"):
            v = v.item()
        g[k] = v
    return g


# ---------------------------------------------------------------------------
# Final full-budget retrain on the knee architecture
# ---------------------------------------------------------------------------
def final_retrain(cfg: dict, data, epochs: int, seeds, patience: int, device: str,
                  weights_path: str | None = None):
    """Retrain from scratch at full budget over several seeds.

    Returns a dict with per-seed and ensemble test MRE, plus val MRE.  The
    ensemble simply averages the seed predictions.  The best single-seed model
    (lowest val MRE) is saved to ``weights_path`` if given, and the ensemble
    test prediction is returned under ``ens_test``/``yte`` for plotting.
    """
    full = Budget(epochs=epochs, subsample=None, patience=patience, device=device)
    lt = data.log_target
    per_seed = []
    test_preds, val_preds = [], []
    n_params = None
    best_val = float("inf")
    best_state = None
    for sd in seeds:
        model, ym, ys, _, _ = fit(cfg, data, full, sd)
        n_params = count_params(model)
        pte = predict(model, data.Xte, ym, ys, device, lt)
        pva = predict(model, data.Xva, ym, ys, device, lt)
        test_preds.append(pte)
        val_preds.append(pva)
        vmre = float(np.mean(np.abs(pva - data.yva) / data.yva))
        if vmre < best_val:
            best_val = vmre
            best_state = {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "cfg": cfg, "y_mean": ym, "y_std": ys, "log_target": lt,
                "x_mean": data.x_mean, "x_std": data.x_std, "seed": int(sd),
            }
        per_seed.append({
            "seed": int(sd),
            "test_mre": float(np.mean(np.abs(pte - data.yte) / data.yte)),
            "val_mre": vmre,
        })

    if weights_path is not None and best_state is not None:
        import torch as _torch
        _torch.save(best_state, weights_path)

    ens_test = np.mean(test_preds, axis=0)
    ens_val = np.mean(val_preds, axis=0)
    test_mres = np.array([s["test_mre"] for s in per_seed])
    return {
        "n_params": int(n_params),
        "per_seed": per_seed,
        "test_mre_mean": float(test_mres.mean()),
        "test_mre_std": float(test_mres.std()),
        "test_mre_ensemble": float(np.mean(np.abs(ens_test - data.yte) / data.yte)),
        "val_mre_ensemble": float(np.mean(np.abs(ens_val - data.yva) / data.yva)),
        "ens_test": ens_test,  # for plotting (not JSON-serialized)
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_pareto(F, knee_i, path):
    F = np.asarray(F, dtype=float)
    order = np.argsort(F[:, 1])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(F[order, 1], F[order, 0], "-", color="0.7", zorder=1)
    ax.scatter(F[:, 1], F[:, 0], s=45, color="#2b6cb0", zorder=2,
               label="Pareto front")
    ax.scatter(F[knee_i, 1], F[knee_i, 0], s=160, marker="*",
               color="#dd6b20", edgecolor="k", zorder=3, label="knee")
    ax.set_xlabel(r"model complexity  $f_2 = \log_{10}(\#\mathrm{params})$")
    ax.set_ylabel(r"validation MRE  $f_1$")
    ax.set_title("NSGA-II Pareto front: accuracy vs. complexity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_predictions(y_true, y_pred, path):
    """Predicted-vs-true scatter + relative-error-by-volume-decile bars."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rel = np.abs(y_pred - y_true) / y_true

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    lo, hi = float(y_true.min()), float(y_true.max())
    ax1.scatter(y_true, y_pred, s=8, alpha=0.35, color="#2b6cb0")
    ax1.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
    ax1.set_xlabel("true volume")
    ax1.set_ylabel("predicted volume")
    ax1.set_title("Ensemble prediction on TEST set")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # mean relative error within each true-volume decile
    edges = np.quantile(y_true, np.linspace(0, 1, 11))
    edges[-1] += 1e-9
    bins = np.clip(np.digitize(y_true, edges) - 1, 0, 9)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mre_bin = [rel[bins == b].mean() if np.any(bins == b) else 0.0 for b in range(10)]
    ax2.bar(range(10), mre_bin, color="#dd6b20")
    ax2.set_xticks(range(10))
    ax2.set_xticklabels([f"{c:.1f}" for c in centers], rotation=45, ha="right")
    ax2.set_xlabel("true-volume decile (bin center)")
    ax2.set_ylabel("mean relative error")
    ax2.set_title("Relative error across the volume range")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="path to jones+volume.txt")
    ap.add_argument("--feature-set", default=None, choices=["j2", "j3", "j2j3"],
                    help="use the data_j2+j3 colored-Jones dataset (parse_j2j3) "
                         "with this feature set instead of --data")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--log-target", action="store_true",
                    help="train on log(volume) instead of raw volume")

    # search budget
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=60, help="epochs per candidate")
    ap.add_argument("--subsample", type=int, default=6000,
                    help="train rows per candidate (0 => all)")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--search-seeds", type=int, default=3,
                    help="seeds averaged for f1 (val MRE)")
    ap.add_argument("--gap-constraint", type=float, default=None,
                    help="if set, constrain (val-train) MRE gap below this")

    # parallelism / device
    ap.add_argument("--n-jobs", type=int, default=max(1, mp.cpu_count() // 2))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    # final retrain
    ap.add_argument("--final-epochs", type=int, default=400)
    ap.add_argument("--final-seeds", type=int, default=5)
    ap.add_argument("--final-patience", type=int, default=40)

    ap.add_argument("--nsga-seed", type=int, default=1)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny settings for a fast end-to-end sanity run")
    args = ap.parse_args()

    if args.smoke:
        args.pop, args.gens, args.epochs = 6, 3, 15
        args.subsample, args.search_seeds = 2000, 2
        args.final_epochs, args.final_seeds = 40, 2

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable; falling back to cpu")
        args.device = "cpu"

    os.makedirs(args.outdir, exist_ok=True)
    cache_dir = os.path.join(args.outdir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # ---- data (test set is created here and never touched until the end) --
    if (args.feature_set is None) == (args.data is None):
        ap.error("give exactly one of --data or --feature-set")
    if args.feature_set is not None:
        from parse_j2j3 import prepare as prepare_j2j3
        data = prepare_j2j3(args.feature_set, seed=args.split_seed,
                            log_target=args.log_target)
    else:
        data = prepare(args.data, seed=args.split_seed, log_target=args.log_target)
    set_data(data)  # set BEFORE the pool forks so workers inherit it
    print(f"[data] features={data.n_features}  "
          f"train/val/test={len(data.Xtr)}/{len(data.Xva)}/{len(data.Xte)}  "
          f"log_target={data.log_target}")

    budget = Budget(epochs=args.epochs,
                    subsample=(None if args.subsample == 0 else args.subsample),
                    patience=args.patience, device=args.device)
    ecfg = EvalConfig(budget=budget, n_seeds=args.search_seeds,
                      seeds=tuple(range(args.search_seeds)), cache_dir=cache_dir)

    # ---- parallel runner --------------------------------------------------
    pool = None
    runner = None
    if args.n_jobs > 1 and args.device == "cpu":
        ctx = mp.get_context("fork")  # fork => workers inherit the global data
        pool = ctx.Pool(args.n_jobs)
        runner = StarmapParallelization(pool.starmap)
        print(f"[search] parallel over {args.n_jobs} workers")
    else:
        print("[search] serial evaluation")

    problem_kwargs = {} if runner is None else {"elementwise_runner": runner}
    problem = NASProblem(ecfg, gap_threshold=args.gap_constraint, **problem_kwargs)

    algorithm = NSGA2(
        pop_size=args.pop,
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(
            eliminate_duplicates=MixedVariableDuplicateElimination()),
        eliminate_duplicates=MixedVariableDuplicateElimination(),
    )

    t0 = time.time()
    res = minimize(
        problem, algorithm, get_termination("n_gen", args.gens),
        seed=args.nsga_seed, verbose=True, save_history=False,
    )
    print(f"[search] done in {time.time() - t0:.1f}s")
    if pool is not None:
        pool.close()
        pool.join()

    # ---- Pareto front -----------------------------------------------------
    F = np.atleast_2d(res.F)
    Xs = res.X if isinstance(res.X, (list, np.ndarray)) else [res.X]
    if isinstance(Xs, np.ndarray) and Xs.dtype == object:
        Xs = list(Xs)
    elif isinstance(Xs, dict):
        Xs = [Xs]

    front = []
    for i in range(len(F)):
        g = genome_from_X(Xs[i])
        cfg = genome_to_config(g)
        front.append({
            "f1_val_mre": float(F[i, 0]),
            "f2_log10_params": float(F[i, 1]),
            "n_params": int(round(10 ** F[i, 1])),
            "arch": {"depth": cfg["depth"], "widths": cfg["widths"],
                     "activation": cfg["activation"], "dropout": round(cfg["dropout"], 4),
                     "alpha": cfg["alpha"], "lr": cfg["lr"],
                     "batch_size": cfg["batch_size"]},
            "genome": g,
        })
    front.sort(key=lambda r: r["f2_log10_params"])

    ki = knee_index(F)
    knee_genome = genome_from_X(Xs[ki])
    knee_cfg = genome_to_config(knee_genome)
    print("\n[pareto] front (sorted by complexity):")
    for r in front:
        mark = "  <== KNEE" if abs(r["f1_val_mre"] - float(F[ki, 0])) < 1e-12 \
            and abs(r["f2_log10_params"] - float(F[ki, 1])) < 1e-12 else ""
        print(f"   val_MRE={r['f1_val_mre']:.4f}  params={r['n_params']:>7}  "
              f"depth={r['arch']['depth']} widths={r['arch']['widths']} "
              f"{r['arch']['activation']}{mark}")

    plot_path = os.path.join(args.outdir, "pareto_front.png")
    plot_pareto(F, ki, plot_path)
    print(f"[pareto] plot -> {plot_path}")

    # ---- final full-budget retrain on the knee ---------------------------
    print(f"\n[final] retraining knee at full budget "
          f"({args.final_epochs} epochs, {args.final_seeds} seeds, all train data)")
    print(f"[final] arch: depth={knee_cfg['depth']} widths={knee_cfg['widths']} "
          f"act={knee_cfg['activation']} dropout={knee_cfg['dropout']:.3f} "
          f"alpha={knee_cfg['alpha']:.2e} lr={knee_cfg['lr']:.2e} "
          f"bs={knee_cfg['batch_size']}")
    weights_path = os.path.join(args.outdir, "knee_model.pt")
    final = final_retrain(
        knee_cfg, data, epochs=args.final_epochs,
        seeds=list(range(args.final_seeds)), patience=args.final_patience,
        device=args.device, weights_path=weights_path)

    print(f"\n[final] TEST MRE = {final['test_mre_mean']:.4f} "
          f"+/- {final['test_mre_std']:.4f}  (mean over {args.final_seeds} seeds)")
    print(f"[final] TEST MRE (ensemble) = {final['test_mre_ensemble']:.4f}")
    print(f"[final] params = {final['n_params']}")

    # ---- diagnostic plot on the untouched test set -----------------------
    ens_test = final.pop("ens_test")  # numpy array, not JSON-serializable
    pred_path = os.path.join(args.outdir, "test_predictions.png")
    plot_predictions(data.yte, ens_test, pred_path)
    print(f"[final] best-seed weights -> {weights_path}")
    print(f"[final] prediction plot   -> {pred_path}")

    # ---- persist everything ----------------------------------------------
    out = {
        "args": vars(args),
        "data_sizes": {"train": len(data.Xtr), "val": len(data.Xva),
                       "test": len(data.Xte)},
        "pareto_front": front,
        "knee": {"arch": {"depth": knee_cfg["depth"], "widths": knee_cfg["widths"],
                          "activation": knee_cfg["activation"],
                          "dropout": knee_cfg["dropout"], "alpha": knee_cfg["alpha"],
                          "lr": knee_cfg["lr"], "batch_size": knee_cfg["batch_size"]},
                 "genome": knee_genome,
                 "f1_val_mre": float(F[ki, 0]),
                 "f2_log10_params": float(F[ki, 1])},
        "final": final,
        "search_seconds": time.time() - t0,
    }
    res_path = os.path.join(args.outdir, "results.json")
    with open(res_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[done] results -> {res_path}")


if __name__ == "__main__":
    main()
