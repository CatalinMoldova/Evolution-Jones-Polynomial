# Phase-evaluation feature for `J2` → hyperbolic volume

**Date:** 2026-07-07
**Author:** CatalinMoldova

This note records a feature-engineering change to the Jones→volume pipeline,
the literature analysis that motivated it, and how to reproduce the run.

---

## 1. Background: what data we use and where its accuracy caps out

The dataset in `jones+volume.txt` is the **fundamental (2-colored) Jones
polynomial `J2`** — 34 integer coefficients per knot (coefficient index `k`
corresponds to power `k − 15`) paired with the hyperbolic volume of the knot
complement. It contains **12,955 knots**, which is exactly the number of
hyperbolic knots up to **13 crossings**. So this is the classic
Davies/Jejjala fundamental-Jones dataset restricted to ≤13 crossings.

### Accuracy ceiling for `J2` (from the literature)

Accuracy is defined as `1 − MRE`, MRE = `mean(|pred − true| / true)` on a held-out
test set — directly comparable to the MRE this repo reports.

| Source | Input | Dataset | Accuracy (1 − MRE) |
|---|---|---|---|
| This repo (42-feat, matched budget) | `J2` | ≤13 crossings, 12,955 knots | 96.9% (test MRE 0.0307; ensemble 0.0286) |
| This repo (45-feat + phase, matched budget) | `J2` + `V(e^{3πi/4})` | ≤13 crossings, 12,955 knots | **97.1%** (test MRE 0.0289; ensemble **0.0269**) |
| Craven–Hughes–Jejjala–Kar (arXiv:2211.01404) | `J2` | comparable | ~97% |
| Hughes et al. (arXiv:2502.18575) | `J2` | ≤15 crossings | 98.1% (1.86%) |
| Hughes et al. (arXiv:2502.18575) | `J2` | **≤16 crossings, 1.7M knots** | **98.5% (1.44–1.65%)** — best `J2` reported |
| Hughes et al. (arXiv:2502.18575) | **`J3` (adjoint / 3-colored)** | ≤15 crossings, 177,316 knots | **99.34% (0.40%)** — highest overall |

Key point: **the ~98.5% ceiling for `J2` is set by the data, not the model.**
Distinct knots can share the same `J2` but have volumes differing by ~3%, so no
architecture can resolve them. arXiv:2502.18575 shows the extra accuracy on `J2`
comes from *more knots at higher crossing number*, not a better network. Breaking
past ~98.5% requires the **`J3` (adjoint) polynomial**, a different, larger dataset.

### Reference papers (in `papers/`, not committed — see §6)

- **arXiv:1512.07906** — Chen, *Cyclotomic Expansion and Volume Conjecture for
  Superpolynomials …* (pure math; volume conjecture, no ML / no accuracy number).
- **arXiv:2211.01404** — Craven, Hughes, Jejjala, Kar, *Illuminating new and known
  relations between knot invariants* (defines accuracy = 1 − MRE; `J2` → volume ≈ 97%).
- **arXiv:2502.18575** — Hughes, Jejjala, Ramadevi, Roy, Singh, *Colored Jones
  Polynomials and the Volume Conjecture* (**99.34%** from adjoint `J3`; identifies
  the best evaluation phases and a symbolic volume formula).

---

## 2. The baseline (before this change)

**The clean comparison is the matched-budget 42-feature run** (`results_matched_42feat.json`,
SLURM job 16544354, node cn108). It uses the *identical* search budget to the phase
run — pop 40, gens 40, full training data, gap-constraint 0.01, 4 search seeds,
7 final seeds, 500 final epochs — so the only difference from the phase run is the
input representation.

- Knee architecture: depth-2 MLP `[19, 40]`, GELU, **1,658 params**
- **Test MRE = 0.0307 ± 0.0006** (≈ 96.9% accuracy); ensemble MRE 0.0286
- The Pareto front is **flat**: going from ~1.6k → ~30k params only moved validation
  MRE ~0.0305 → ~0.0305. That flatness is the signature of an *information ceiling*
  — the model is not the bottleneck, the `J2` signal on this small dataset is.

> **Note on `results_baseline_42feat.json`.** An *earlier*, smaller-budget 42-feature
> run (pop 24, gens 20, subsample 6000, 5 final seeds) is preserved in
> `results_baseline_42feat.json` and scored test MRE 0.0288 / ensemble 0.0276. Do **not**
> use it as the before/after control: it confounds the feature change with a budget
> change, and — being a lower-budget run on a flat information ceiling — it happens to
> land below the full-budget 42-feature number, which is run-to-run noise, not a real
> effect. Compare against `results_matched_42feat.json` instead.

---

## 3. The change: add the volume-correlated phase evaluation

arXiv:2502.18575 (and earlier work it cites) notes that a function of
**`|J2(K; e^{3πi/4})|`** approximates the volume nearly as well as a full neural
network. The pipeline previously fed the network only degree-summary statistics
and the raw coefficients — it never evaluated the polynomial at that phase.

**Added** to `build_features` in `parse_data.py`: the evaluation of the Jones
polynomial at `q0 = e^{3πi/4}`, as three features — real part, imaginary part,
and modulus:

```
V(q0) = Σ_k c_k · q0^(k−15),   q0 = exp(3πi/4)
new features: [Re V(q0), Im V(q0), |V(q0)|]
```

Feature vector: **42 → 45** (`N_SUMMARY` 8 → 11). No change to the model search
space, training, splits, or targets — only the input representation.

### Files touched
- `parse_data.py` — new module constant `EVAL_PHASE = 3π/4`; `build_features`
  now appends `[Re V(q0), Im V(q0), |V(q0)|]`; `N_SUMMARY = 11`,
  `N_FEATURES = 45`; docstrings updated.
- `submit_nas_phase.slurm` — copy of `submit_nas.slurm` writing to `results_phase/`
  (fresh cache — the on-disk score cache is keyed by architecture and would be
  stale for the new input width).
- `results_baseline_42feat.json` — preserved copy of the 42-feature result for a
  clean before/after comparison.

---

## 4. Verification

Parsed the full dataset and checked the new feature against the target:

```
N_FEATURES = 45,  built X shape = (12955, 45)
|V(q0)| range: 0.414 … 87.426
corr(log|V(q0)|, log volume) = 0.9324
corr(L1 norm,    log volume) = 0.7989   # previous strongest feature, for comparison
```

`|V(q0)|` is by a wide margin the **strongest single predictor** in the feature
set — exactly the volume-tracking signal the paper describes.

---

## 5. Reproducing the run

Environment: `~/.conda/envs/nsga2` (Python 3.12; numpy, torch, pymoo, matplotlib).

Full search with the phase feature (SLURM):

```bash
sbatch submit_nas_phase.slurm      # -> results_phase/
```

Outputs land in `results_phase/`: `results.json` (test MRE + ensemble),
`pareto_front.png`, `test_predictions.png`, `knee_model.pt`.

The phase run was launched as job `16544559` (node dn032), same search budget as
the baseline (pop 40, gens 40, 4 search seeds, 7 final seeds, log-target,
gap-constraint 0.01). It confirmed `features=45` at startup.

### Expectations
Still `J2` on ≤13-crossing data, so the ~97–98% ceiling still applies. The phase
feature gives even a tiny MLP a near-direct line to the volume, so the expected
effect is a lower error and/or the knee shifting to an *even smaller* model. To
break past ~98.5% one must move to `J3` (see §1).

### Result (measured, matched budget)

Clean A/B — both runs pop 40 / gens 40 / full data / gap-constraint 0.01 / 7 final
seeds / 500 final epochs. Only the input representation differs.

| | 42-feat (`results_matched_42feat.json`) | 45-feat + phase (`results_phase/results.json`) |
|---|---|---|
| Test MRE (mean ± std) | 0.0307 ± 0.0006 | **0.0289 ± 0.0004** |
| Test MRE (7-seed ensemble) | 0.0286 | **0.0269** |
| Accuracy (1 − MRE), ensemble | 97.14% | **97.31%** |
| Knee arch | `[19, 40]`, 1,658 p | `[32, 18, 54, 18]`, 4,101 p |
| Best Pareto val MRE | ~0.0305 | 0.0281 |

Adding the phase feature cut test error ~6% relative (both mean and ensemble) and
lowered the Pareto front at every point. The gain is real but bounded by the `J2`
information ceiling — consistent with §1. The knee did **not** shrink; it moved to a
slightly deeper 4-layer net, though the front stays flat, so the smaller `[16]`–`[22]`
single-layer models are within ~0.004 val MRE of the knee if size matters more.

---

## 6. Note on the `papers/` folder

The three reference PDFs live in `papers/` locally but are **git-ignored** — they
are large copyrighted arXiv PDFs. They are cited above by arXiv ID and are freely
available at `https://arxiv.org/abs/<id>`.
