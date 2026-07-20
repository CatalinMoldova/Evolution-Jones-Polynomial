"""Loader + feature engineering for the colored-Jones dataset ``data_j2+j3/``.

Source table (``dataset.numbers``, cached to ``dataset_cache.json``): one row
per (knot, r), r = 1..11, holding the colored Jones polynomial J_{r+1} (r=1 is
the classical Jones "J2", r=2 the adjoint "J3").  1,426 knots, 3-12 crossings.

Data quirks this loader handles (see README §1.6):
  * ``exp_var`` is mixed 'q'/'t' (t = q^(1/2) with a framing twist).  We use
    the ``canon`` column throughout: the polynomial reduced to a consecutive
    integer q-power grid.  Validated: every r=1 canon satisfies |V(1)| = 1
    (classical Jones normalization) and canon reproduces the known Jones
    polynomials of 3_1 / 4_1 / 6_3.
  * The absolute power offset is framing-polluted for t-rows, so ALL features
    are shift-invariant: polynomials are aligned to start at power 0, and
    evaluations whose sign depends on the offset are reported as magnitudes.
  * The ``determinant`` column is wrong for 47 rows -> never used.
  * Seven non-hyperbolic torus knots are dropped (incl. 11a_367 = T(2,11),
    which SnapPy flags as flat/volume-0); volumes come from ``volumes.json``
    (built once by ``compute_volumes.py`` with SnapPy).

Per-polynomial feature block (9 + W_r):
  [span, #nonzero, L1, L2, V(1), |V(-1)|, Re V(q0), Im V(q0), |V(q0)|]
  followed by the canon coefficients padded to the family-wide max width W_r.
Re/Im are computed on the aligned polynomial (first coefficient at power 0),
which makes them well-defined canonical values; |V(q0)| is fully
shift-invariant.  q0 = exp(i * phase), one phase per r (see EVAL_PHASES).
"""
from __future__ import annotations

import json
import os

import numpy as np

from parse_data import DataBundle, make_splits

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_j2+j3")
CACHE = os.path.join(DATA_DIR, "dataset_cache.json")
VOLUMES = os.path.join(DATA_DIR, "volumes.json")

TORUS_KNOTS = {"3_1", "5_1", "7_1", "9_1", "11a_367", "8_19", "10_124"}

# Volume-correlated evaluation phase per r (angle of q0 on the unit circle).
# Empirical peaks of corr(log|V(e^{i*theta})|, log volume) measured on this
# dataset (1,419 hyperbolic knots, 2026-07-18 sweep):
#   r=1: peak at 0.814*pi, corr 0.943 -> use 13*pi/16 (0.8125*pi, corr 0.943).
#        NOTE: the literature phase 3*pi/4 (arXiv:2211.01404, 2502.18575)
#        only reaches corr 0.886 on the aligned canon grid here.
#   r=2: peak at 0.558*pi, corr 0.959 -> use 5*pi/9 (0.5556*pi, corr 0.959).
EVAL_PHASES = {1: 13.0 * np.pi / 16.0, 2: 5.0 * np.pi / 9.0}

N_SUMMARY = 9


def load_rows():
    """Return the cached table rows: [{knot, r, min_exp, canon, var, ...}]."""
    with open(CACHE) as fh:
        return json.load(fh)


def load_volumes():
    """Return {knot_name: hyperbolic volume} (torus knots absent)."""
    with open(VOLUMES) as fh:
        return json.load(fh)


def poly_widths(rows):
    """Max canon length per r over the whole table (fixed padding widths)."""
    w = {}
    for row in rows:
        w[row["r"]] = max(w.get(row["r"], 0), len(row["canon"]))
    return w


def features_for_poly(canon: list, width: int, phase: float) -> np.ndarray:
    """Shift-invariant feature block for one polynomial: 9 summary + padded coeffs."""
    c = np.asarray(canon, dtype=np.float64)
    nz = np.nonzero(c)[0]
    span = float(nz[-1] - nz[0]) if len(nz) else 0.0
    n_nonzero = float(len(nz))
    l1 = float(np.abs(c).sum())
    l2 = float(np.sqrt((c ** 2).sum()))
    v1 = float(c.sum())                                   # V(1), offset-free
    v_neg1 = abs(float((c * (-1.0) ** np.arange(len(c))).sum()))  # |V(-1)|
    q0 = np.exp(1j * phase * np.arange(len(c)))           # aligned at power 0
    v_q0 = complex((c * q0).sum())
    feats = np.empty(N_SUMMARY + width, dtype=np.float64)
    feats[:N_SUMMARY] = [span, n_nonzero, l1, l2, v1, v_neg1,
                         v_q0.real, v_q0.imag, abs(v_q0)]
    feats[N_SUMMARY:] = 0.0
    feats[N_SUMMARY:N_SUMMARY + len(c)] = c
    return feats


def build_matrix(feature_set: str = "j2j3", phases: dict | None = None):
    """Return (knot_names, X, y) for hyperbolic knots with known volumes.

    feature_set: 'j2' (r=1 block), 'j3' (r=2 block), or 'j2j3' (both).
    """
    r_wanted = {"j2": [1], "j3": [2], "j2j3": [1, 2]}[feature_set]
    phases = phases or EVAL_PHASES
    rows = load_rows()
    vols = load_volumes()
    widths = poly_widths(rows)

    by_knot = {}
    for row in rows:
        if row["r"] in r_wanted:
            by_knot.setdefault(row["knot"], {})[row["r"]] = row["canon"]

    names, X, y = [], [], []
    for name in sorted(by_knot):
        if name in TORUS_KNOTS or name not in vols:
            continue
        blocks = by_knot[name]
        if set(blocks) != set(r_wanted):
            continue
        feats = np.concatenate([
            features_for_poly(blocks[r], widths[r], phases[r]) for r in r_wanted])
        names.append(name)
        X.append(feats)
        y.append(vols[name])
    return names, np.asarray(X), np.asarray(y, dtype=np.float64)


def prepare(feature_set: str = "j2j3", seed: int = 0,
            fracs=(0.70, 0.15, 0.15), log_target: bool = False) -> DataBundle:
    """Features -> split -> standardize; same protocol as parse_data.prepare."""
    _, X, y = build_matrix(feature_set)
    tr, va, te = make_splits(len(X), seed=seed, fracs=fracs)

    x_mean = X[tr].mean(axis=0)
    x_std = X[tr].std(axis=0)
    x_std[x_std < 1e-8] = 1.0

    def std(a):
        return (a - x_mean) / x_std

    ytr = y[tr]
    ytr_t = np.log(ytr) if log_target else ytr
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
    for fs in ("j2", "j3", "j2j3"):
        names, X, y = build_matrix(fs)
        print(f"{fs:>5}: {X.shape[0]} knots x {X.shape[1]} features   "
              f"volume range [{y.min():.3f}, {y.max():.3f}]")
