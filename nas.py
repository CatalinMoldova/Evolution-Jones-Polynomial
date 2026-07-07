"""NSGA-II neural architecture search core.

Contents:
  * an MLP builder driven by a genome dict,
  * a differentiable training loop (Adam + Huber loss, early stopping) used to
    *fit* each candidate under a cheap multi-fidelity budget,
  * a genome evaluator that returns the two (non-differentiable) search
    objectives -- validation mean-relative-error and log10(param count) --
    averaged over a few seeds and memoized on disk,
  * a pymoo mixed-variable ``ElementwiseProblem`` wiring it all together.

The training loss (Huber on the standardized target) is what backprop
minimizes; it is deliberately *not* the same quantity NSGA-II optimizes.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from parse_data import DataBundle

# ---------------------------------------------------------------------------
# Global data handle.
#
# The dataset arrays are large-ish and identical for every evaluation, so we
# stash them in a module global rather than on the Problem instance.  Worker
# processes forked *after* set_data() inherit these arrays for free, which
# keeps the per-individual pickle payload (just the genome) tiny under
# StarmapParallelization.
# ---------------------------------------------------------------------------
_DATA: DataBundle | None = None


def set_data(bundle: DataBundle) -> None:
    global _DATA
    _DATA = bundle


ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
ACTIVATION_CHOICES = list(ACTIVATIONS)
BATCH_CHOICES = [128, 256, 512]
MAX_DEPTH = 6


# ---------------------------------------------------------------------------
# Genome <-> model
# ---------------------------------------------------------------------------
def genome_to_config(g: dict) -> dict:
    """Normalize a pymoo genome dict into plain Python scalars.

    Only the first ``depth`` widths (w0..w{depth-1}) are meaningful; the rest
    are ignored so that architecturally identical genomes hash identically.
    """
    depth = int(g["depth"])
    widths = [int(g[f"w{i}"]) for i in range(depth)]
    return {
        "depth": depth,
        "widths": widths,
        "activation": str(g["activation"]),
        "alpha": 10.0 ** float(g["log_alpha"]),
        "lr": 10.0 ** float(g["log_lr"]),
        "dropout": float(g.get("dropout", 0.0)),
        "batch_size": int(g.get("batch_size", 256)),
    }


def build_mlp(in_dim: int, cfg: dict) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    act = ACTIVATIONS[cfg["activation"]]
    for w in cfg["widths"]:
        layers.append(nn.Linear(d, w))
        layers.append(act())
        if cfg["dropout"] > 0:
            layers.append(nn.Dropout(cfg["dropout"]))
        d = w
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Training one candidate
# ---------------------------------------------------------------------------
@dataclass
class Budget:
    """Multi-fidelity training budget."""
    epochs: int = 60
    subsample: int | None = 6000  # rows of train used (None => all)
    patience: int = 12
    device: str = "cpu"


def _mre(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - y) / y))


def predict(model: nn.Module, X: np.ndarray, ym: float, ys: float,
            device, log_target: bool = False) -> np.ndarray:
    """Forward pass on standardized features -> volumes on the natural scale.

    When ``log_target`` is set the network output destandardizes to log(volume),
    so we exponentiate to recover the natural volume scale.
    """
    model.eval()
    with torch.no_grad():
        out = model(torch.as_tensor(X, device=device)).squeeze(-1)
    z = out.cpu().numpy() * ys + ym
    return np.exp(z) if log_target else z


def fit(cfg: dict, data: DataBundle, budget: Budget, seed: int):
    """Train one MLP with Adam + Huber loss and early stopping.

    The differentiable objective is Huber loss on the standardized target.
    Early stopping tracks the internal validation loss and the best-val weights
    are restored before returning.  This single routine backs both the cheap
    search evaluations and the final full-budget retrain (which passes
    ``subsample=None`` to use all training rows).

    Returns (model, ym, ys, Xtr_used, ytr_used).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)  # cooperate with process-level parallelism
    device = torch.device(budget.device)

    # ---- multi-fidelity subsample of the training rows -------------------
    Xtr, ytr = data.Xtr, data.ytr
    if budget.subsample and budget.subsample < len(Xtr):
        sel = np.random.default_rng(seed).choice(
            len(Xtr), budget.subsample, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]

    ym, ys = data.y_mean, data.y_std  # standardize target with train stats
    # In log-target mode the loss is defined on log(volume); y_mean/y_std
    # already describe that scale, so map the raw volumes through log first.
    ytr_lo = np.log(ytr) if data.log_target else ytr
    yva_lo = np.log(data.yva) if data.log_target else data.yva
    Xtr_t = torch.as_tensor(Xtr, device=device)
    ytr_std = torch.as_tensor((ytr_lo - ym) / ys, dtype=torch.float32, device=device)
    Xva_t = torch.as_tensor(data.Xva, device=device)
    yva_std = torch.as_tensor((yva_lo - ym) / ys, dtype=torch.float32, device=device)

    model = build_mlp(data.n_features, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["alpha"])
    loss_fn = nn.HuberLoss(delta=1.0)  # robust to the high-volume tail

    bs = cfg["batch_size"]
    n = len(Xtr_t)
    rng = np.random.default_rng(seed + 1)

    best_val = float("inf")
    best_state = None
    bad = 0
    for _ in range(budget.epochs):
        model.train()
        order = rng.permutation(n)
        for s in range(0, n, bs):
            b = order[s:s + bs]
            opt.zero_grad()
            out = model(Xtr_t[b]).squeeze(-1)
            loss = loss_fn(out, ytr_std[b])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xva_t).squeeze(-1), yva_std).item()
        if vloss < best_val - 1e-5:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= budget.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, ym, ys, Xtr, ytr


def train_one(cfg: dict, data: DataBundle, budget: Budget, seed: int):
    """Fit a candidate and score it; return (val_mre, train_mre, n_params).

    Errors are mean-relative-error on the natural (un-standardized) volume
    scale, which is what NSGA-II's f1 objective consumes.
    """
    model, ym, ys, Xtr, ytr = fit(cfg, data, budget, seed)
    n_params = count_params(model)
    lt = data.log_target
    val_mre = _mre(predict(model, data.Xva, ym, ys, budget.device, lt), data.yva)
    tr_mre = _mre(predict(model, Xtr, ym, ys, budget.device, lt), ytr)
    return val_mre, tr_mre, n_params


# ---------------------------------------------------------------------------
# Genome evaluation (multi-seed) + disk cache
# ---------------------------------------------------------------------------
def genome_hash(cfg: dict) -> str:
    key = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()


@dataclass
class EvalConfig:
    budget: Budget = field(default_factory=Budget)
    n_seeds: int = 3
    seeds: tuple = (0, 1, 2)
    cache_dir: str | None = None


def evaluate_genome(g: dict, ecfg: EvalConfig, data: DataBundle | None = None) -> dict:
    """Return {f1, f2, gap, n_params} for a genome, averaged over seeds.

    f1 = mean validation relative error (averaged over seeds)
    f2 = log10(parameter count)
    gap = val_mre - train_mre  (available for an optional constraint)
    Results are memoized on disk keyed by the architectural hash.
    """
    if data is None:
        data = _DATA
    cfg = genome_to_config(g)
    h = genome_hash(cfg)

    cache_file = None
    if ecfg.cache_dir:
        cache_file = os.path.join(ecfg.cache_dir, f"{h}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass  # corrupt/partial cache -> recompute

    seeds = ecfg.seeds[:ecfg.n_seeds]
    val_mres, tr_mres, n_params = [], [], None
    for sd in seeds:
        v, t, npar = train_one(cfg, data, ecfg.budget, sd)
        val_mres.append(v)
        tr_mres.append(t)
        n_params = npar

    f1 = float(np.mean(val_mres))
    f2 = float(np.log10(max(n_params, 1)))
    gap = float(np.mean(val_mres) - np.mean(tr_mres))
    result = {"f1": f1, "f2": f2, "gap": gap, "n_params": int(n_params),
              "val_mre_std": float(np.std(val_mres))}

    if cache_file:
        tmp = f"{cache_file}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as fh:
                json.dump(result, fh)
            os.replace(tmp, cache_file)  # atomic
        except OSError:
            pass
    return result


# ---------------------------------------------------------------------------
# pymoo Problem
# ---------------------------------------------------------------------------
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Choice, Integer, Real


def make_variables() -> dict:
    """The genome search space as pymoo mixed variables."""
    v: dict = {
        "depth": Integer(bounds=(1, MAX_DEPTH)),
        "activation": Choice(options=ACTIVATION_CHOICES),
        "log_alpha": Real(bounds=(-6.0, -1.0)),  # L2:  alpha = 10**log_alpha
        "log_lr": Real(bounds=(-4.0, -2.0)),     # Adam lr = 10**log_lr
        "dropout": Real(bounds=(0.0, 0.5)),
        "batch_size": Choice(options=BATCH_CHOICES),
    }
    for i in range(MAX_DEPTH):
        v[f"w{i}"] = Integer(bounds=(16, 512))
    return v


class NASProblem(ElementwiseProblem):
    """Two-objective architecture search: (val MRE, log10 params).

    An optional constraint keeps the train/val relative-error gap below a
    threshold (per the brief, overfitting is expressed as a *constraint*, never
    as an objective -- a too-weak net also has ~0 gap and would pollute the
    Pareto front).
    """

    def __init__(self, ecfg: EvalConfig, gap_threshold: float | None = None,
                 **kwargs):
        self.ecfg = ecfg
        self.gap_threshold = gap_threshold
        super().__init__(
            vars=make_variables(),
            n_obj=2,
            n_ieq_constr=1 if gap_threshold is not None else 0,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        res = evaluate_genome(x, self.ecfg)
        out["F"] = [res["f1"], res["f2"]]
        if self.gap_threshold is not None:
            out["G"] = [res["gap"] - self.gap_threshold]
