"""Parsing and feature engineering for the Jones-polynomial -> hyperbolic-volume task.

The input file is a single Mathematica association written as::

    {{i,1}->{34 integer coefficients}, {i,2}->volume, ...}

Knot ``i`` therefore contributes two entries: ``{i,1}`` maps to its 34 Jones
coefficients and ``{i,2}`` maps to its hyperbolic volume.  We parse both with
regular expressions and pair them by the shared index ``i``.

Features (45-dim): eleven polynomial-summary features followed by the 34 raw
coefficients.  The power associated with coefficient index ``k`` (0-based) is
``k - 15``.  The summary block is
[min nonzero power, max nonzero power, degree span, #nonzero terms,
 L1 norm, L2 norm, V(1) = sum of coeffs, V(-1) = alternating sum,
 Re V(q0), Im V(q0), |V(q0)|]  with q0 = exp(3*pi*i/4).
These are cheap, knot-theoretically motivated descriptors.  Degree span and the
evaluations at +/-1 are classically informative; the complex phase evaluation
V(e^{3*pi*i/4}) is the one Jejjala et al. found to track the hyperbolic volume
almost as well as a full network -- so |V(q0)| in particular is a strong,
volume-correlated signal for a tiny MLP.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# {i,1}->{c0,c1,...,c33}
_COEFF_RE = re.compile(r"\{(\d+),\s*1\}\s*->\s*\{([^}]*)\}")
# {i,2}->volume   (float, possibly with sign / scientific notation)
_VOL_RE = re.compile(r"\{(\d+),\s*2\}\s*->\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")

N_COEFFS = 34
POWER_OFFSET = 15  # power of coefficient at index k is (k - POWER_OFFSET)
# Best J2 phase for the hyperbolic volume (Jejjala et al.; cf. 2502.18575 eq.
# discussion of |J2(K; e^{3*pi*i/4})|).  Coefficients are powers of this variable.
EVAL_PHASE = 3.0 * np.pi / 4.0
N_SUMMARY = 11     # min/max power, span, #nonzero, L1, L2, V(1), V(-1),
                   # Re V(q0), Im V(q0), |V(q0)|  with q0 = exp(i*EVAL_PHASE)
N_FEATURES = N_COEFFS + N_SUMMARY


def load_raw(path: str):
    """Parse the file; return (indices, coeffs[N,34] int, volumes[N] float).

    Only knots for which *both* a coefficient list and a volume are present are
    kept, paired by their shared index and returned in ascending index order.
    """
    with open(path, "r") as fh:
        txt = fh.read()

    coeffs = {}
    for i, body in _COEFF_RE.findall(txt):
        vals = [int(x) for x in body.split(",")]
        if len(vals) != N_COEFFS:
            raise ValueError(f"knot {i}: expected {N_COEFFS} coeffs, got {len(vals)}")
        coeffs[int(i)] = vals

    vols = {int(i): float(v) for i, v in _VOL_RE.findall(txt)}

    idx = sorted(set(coeffs) & set(vols))
    if not idx:
        raise ValueError(f"no paired knots parsed from {path!r}")

    C = np.asarray([coeffs[i] for i in idx], dtype=np.float64)
    y = np.asarray([vols[i] for i in idx], dtype=np.float64)
    return np.asarray(idx, dtype=np.int64), C, y


def build_features(C: np.ndarray) -> np.ndarray:
    """[N,34] integer coefficients -> [N,45] features.

    Summary block (11): [min nonzero power, max nonzero power, degree span,
    #nonzero terms, L1 norm, L2 norm, V(1)=sum coeffs, V(-1)=alternating sum,
    Re V(q0), Im V(q0), |V(q0)|] with q0 = exp(i*EVAL_PHASE).
    Followed by the 34 raw coefficients.
    """
    C = np.asarray(C, dtype=np.float64)
    n, ncoef = C.shape
    if ncoef != N_COEFFS:
        raise ValueError(f"expected {N_COEFFS} coeffs, got {ncoef}")

    powers = np.arange(ncoef) - POWER_OFFSET  # index k -> power k-15
    mask = C != 0

    # min/max nonzero power via masked extrema (all-zero rows fall back to 0)
    min_pow = np.where(mask, powers[None, :], np.inf).min(axis=1)
    max_pow = np.where(mask, powers[None, :], -np.inf).max(axis=1)
    all_zero = ~mask.any(axis=1)
    min_pow[all_zero] = 0.0
    max_pow[all_zero] = 0.0

    span = max_pow - min_pow                       # breadth (degree span)
    n_nonzero = mask.sum(axis=1).astype(np.float64)
    l1 = np.abs(C).sum(axis=1)                      # sum |c|
    l2 = np.sqrt((C ** 2).sum(axis=1))             # Euclidean norm
    v_at_1 = C.sum(axis=1)                          # V(1) = sum of coeffs
    v_at_neg1 = (C * ((-1.0) ** powers)[None, :]).sum(axis=1)  # V(-1), det-related

    # V(q0) at the volume-correlated phase q0 = exp(i*EVAL_PHASE).
    qpow = np.exp(1j * EVAL_PHASE * powers)          # [ncoef] complex
    v_at_q0 = (C * qpow[None, :]).sum(axis=1)        # [n] complex
    re_q0, im_q0, abs_q0 = v_at_q0.real, v_at_q0.imag, np.abs(v_at_q0)

    summary = np.stack(
        [min_pow, max_pow, span, n_nonzero, l1, l2, v_at_1, v_at_neg1,
         re_q0, im_q0, abs_q0], axis=1)
    return np.concatenate([summary, C], axis=1)


@dataclass
class DataBundle:
    """Standardized features + raw volumes for the three splits.

    ``X*`` are standardized with statistics fit on the *training* split only.
    ``y*`` are the raw (positive) volumes -- relative error is always computed
    on this natural scale.  ``y_mean``/``y_std`` are the training-target
    standardization used only to stabilize the differentiable training loss;
    predictions are mapped back before any error is reported.
    """

    Xtr: np.ndarray
    Xva: np.ndarray
    Xte: np.ndarray
    ytr: np.ndarray
    yva: np.ndarray
    yte: np.ndarray
    y_mean: float
    y_std: float
    x_mean: np.ndarray
    x_std: np.ndarray
    log_target: bool = False  # if True, y_mean/y_std describe log(volume)

    @property
    def n_features(self) -> int:
        return self.Xtr.shape[1]


def make_splits(n: int, seed: int = 0, fracs=(0.70, 0.15, 0.15)):
    """Return (train_idx, val_idx, test_idx) index arrays for a fixed shuffle."""
    if abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError("fracs must sum to 1")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def prepare(path: str, seed: int = 0, fracs=(0.70, 0.15, 0.15),
            log_target: bool = False) -> DataBundle:
    """Full pipeline: parse -> features -> split -> standardize.

    ``y*`` are always the raw (positive) volumes -- relative error is reported
    on that natural scale.  When ``log_target`` is set, ``y_mean``/``y_std``
    standardize ``log(volume)`` instead of the volume itself; the training loop
    then learns in log-space and exponentiates before scoring.
    """
    _, C, y = load_raw(path)
    X = build_features(C)

    tr, va, te = make_splits(len(X), seed=seed, fracs=fracs)

    x_mean = X[tr].mean(axis=0)
    x_std = X[tr].std(axis=0)
    x_std[x_std < 1e-8] = 1.0  # guard constant columns

    def std(a):
        return (a - x_mean) / x_std

    ytr = y[tr]
    ytr_t = np.log(ytr) if log_target else ytr  # target the loss standardizes
    return DataBundle(
        Xtr=std(X[tr]).astype(np.float32),
        Xva=std(X[va]).astype(np.float32),
        Xte=std(X[te]).astype(np.float32),
        ytr=ytr.astype(np.float64),
        yva=y[va].astype(np.float64),
        yte=y[te].astype(np.float64),
        y_mean=float(ytr_t.mean()),
        y_std=float(ytr_t.std() + 1e-8),
        x_mean=x_mean,
        x_std=x_std,
        log_target=log_target,
    )


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "jones+volume.txt"
    d = prepare(path)
    print(f"features: {d.n_features}   "
          f"train/val/test = {len(d.Xtr)}/{len(d.Xva)}/{len(d.Xte)}")
    print(f"volume  mean={d.y_mean:.3f}  std={d.y_std:.3f}  "
          f"range=[{d.ytr.min():.2f}, {d.ytr.max():.2f}]")
