# Jones Polynomial → Hyperbolic Volume — NSGA-II NAS + phase features

Predict a knot's **hyperbolic volume** from its **Jones polynomial** with a
multi-objective **neural architecture search** (NSGA-II over MLPs, accuracy vs.
model size), then retrain the Pareto-knee at full budget and report test error.
The task is the knot-theory regression popularized by Davies et al. (Nature
2021). This README records everything done so far, in order.

MRE = `mean(|pred − true| / true)` on the natural volume scale; accuracy = 1 − MRE.

---

## 1. What has been done (chronological)

### 1.1 Baseline NAS pipeline (commit `0a9665b`, jobs 16511944/16511953/16544345)
- `parse_data.py` parses `jones+volume.txt` — the Davies/Jejjala fundamental
  (2-colored) Jones dataset `J₂`: 34 integer coefficients (power = index − 15)
  + hyperbolic volume for **12,955 hyperbolic knots ≤13 crossings**.
- **42 features:** 8 summary stats (min/max power, span, #nonzero, L1, L2,
  V(1), V(−1)) + the 34 raw coefficients. Fixed-seed 70/15/15 split
  (9068/1943/1944), train-only standardization, test untouched until the end.
- `nas.py` + `run_nas.py`: NSGA-II over genome {depth 1–6, widths 16–512,
  activation relu/tanh/gelu, dropout 0–0.5, L2 1e-6–1e-1, lr 1e-4–1e-2,
  batch 128/256/512}. Objectives: validation MRE and log₁₀(params);
  overfitting gap (val−train < 0.01) as a **constraint**. Adam + Huber on
  log-volume, early stopping, multi-fidelity, per-architecture disk memoization.
  Knee = max perpendicular distance on the Pareto front, retrained from
  scratch (500 epochs × several seeds) before test evaluation.
- Early full run (pop 24 / gens 20 / subsample 6000): knee `[37, 25]` GELU,
  2,345 params, test MRE 0.0288 ± 0.0002 (ensemble 0.0276). Preserved as
  `results_baseline_42feat.json` — **do not use as the A/B control** (budget
  differs from the phase run; see PHASE_FEATURE_EXPERIMENT.md §2).

### 1.2 Matched-budget 42-feature control (job 16544354 → `results_matched_42feat.json`)
Full budget: pop 40 / gens 40 / full data / gap 0.01 / 4 search seeds /
7 final seeds / 500 final epochs.
- Knee: depth-2 `[19, 40]` GELU, **1,658 params**
- **Test MRE 0.0307 ± 0.0006** (96.93% mean); **ensemble 0.0286 (97.14%)**
- Pareto front is flat (1.6k → 30k params barely moves val MRE) — an
  *information ceiling* of the J₂ signal, not a model limit.

### 1.3 Phase-evaluation feature (commit `4ec713e`, job 16544559 → `results_phase/`)
Theory-guided feature (arXiv:2211.01404, 2502.18575): evaluate the polynomial
at the volume-conjecture phase **q₀ = e^{3πi/4}** and append
`[Re V(q₀), Im V(q₀), |V(q₀)|]` → **45 features**. Nothing else changed.
`|V(q₀)|` has corr 0.93 with log-volume (previous best single feature: 0.80).
Identical search budget to §1.2 — a clean A/B:

| | 42-feat (matched) | 45-feat + phase |
|---|---|---|
| Test MRE (mean ± std) | 0.0307 ± 0.0006 | **0.0289 ± 0.0004** |
| Test MRE (ensemble) | 0.0286 | **0.0269** |
| Accuracy (ensemble) | 97.14% | **97.31%** |
| Knee | `[19, 40]`, 1,658 p | `[32, 18, 54, 18]`, 4,101 p |

**≈6% relative error reduction; the entire Pareto front shifts down.**

**Best architecture (phase-run knee):** 45→32→18→54→18→1 MLP, GELU,
dropout 0.006, lr 6.4e-3, L2 α 1.0e-6, batch 256 — 4,101 params.
Weights: `results_phase/knee_model.pt`.

### 1.4 Random-feature control ablation (`control_ablation.py` → `results_control.json`)
Is the gain the *phase signal* or just 3 extra inputs? Fix the phase-run knee
architecture + retrain protocol (7 seeds × 500 epochs); vary only the content
of the 3 extra columns:

| Variant (same net, same training) | Ensemble acc | Mean acc |
|---|---|---|
| **phase** `V(e^{3πi/4})` | **97.31%** | 97.11% |
| zero (dead cols = J₂-only) | 97.22% | 97.01% |
| random (Gaussian noise) | 97.11% | 96.86% |

Phase > zero > random: **the gain is the volume signal**; noise columns
actually hurt slightly.

### 1.5 Presentation (`Jones_phase_feature.pptx`, rebuilt 2026-07-18)
10 slides built by `make_deck.py` from `figures/` (scripts:
`make_presentation_plots.py`, `make_phase_sweep.py`, `make_ab_barchart.py`,
`make_ablation_barchart.py`, `make_arch_figure.py`). The control slide now
shows the **measured** ablation (§1.4) instead of the earlier "planned"
placeholder, and the architecture slide states the phase-run knee + hyperparams.

### 1.6 New colored-Jones dataset (`data_j2+j3/dataset.numbers`, added 2026-07-18)
Apple Numbers file (read with `numbers-parser`), one table, 15,686 rows ×
10 cols: `knot, r, min_exp, max_exp, coeffs, canon, exp_var, determinant,
source, diagram`. Validation findings:

- **1,426 knots × complete r = 1…11** (colored Jones J_{r+1}; r=1 is the
  classical Jones = "J₂", r=2 the adjoint "J₃"). Verified: r=1 rows match
  known Jones polynomials (3_1, 4_1) and |J(−1)| = knot determinant.
- **Crossing coverage 3–12**, but **NOT complete ≤10 crossings** for all prime
  knots: missing 8_18; 9_34/39/40/41/47/49; 35 of 165 ten-crossing knots
  (and most 11–12 crossing). Rows come from two diagram families
  (`source`: 2vertex 1,396 knots, pretzel 30) — complete only within those.
- **Mixed variable convention** (`exp_var`): `q`, or `t = q^{1/2}` (odd r only).
  Verified via determinant: q-rows pass at q=−1, t-rows at t=i. **Any phase
  evaluation must branch on `exp_var`.**
- **Inconsistent monomial shifts** (e.g. 4_1 stored at exps 1…5 instead of
  −2…2, a framing/writhe artifact): `|V(q₀)|` is shift-invariant and safe;
  raw Re/Im are not — prefer |·| or normalize the shift first.
- **`determinant` column is wrong for 47 rows** (non-alternating ≥10-crossing
  knots, e.g. 10_124: column 3, true det 1 — the polynomial itself is
  correct). Don't use that column as a feature without cleaning.
- **No volume column**, and **7 non-hyperbolic torus knots included**
  (3_1, 5_1, 7_1, 9_1, 11a_367 = T(2,11), 8_19, 10_124 — volume 0/undefined).
  Volumes must be joined externally (SnapPy by knot name is the clean route)
  and torus knots dropped. (11a_367 was found the hard way: SnapPy returns
  flat tetrahedra / volume 0 for it, and its Jones polynomial continues the
  exact T(2,n) coefficient pattern of 3_1/5_1/7_1/9_1.)

---

## 2. Running things

Environment: `~/.conda/envs/nsga2` (Python 3.12: numpy, torch, pymoo,
matplotlib, python-pptx, Pillow, numbers-parser). Never run searches on a
login node.

```bash
# quick pipeline sanity (~30 s)
python run_nas.py --data jones+volume.txt --outdir results_smoke --smoke --n-jobs 8

# full searches (SLURM)
sbatch submit_nas.slurm         # 42-feat  -> results/
sbatch submit_nas_phase.slurm   # 45-feat  -> results_phase/

# control ablation (fixed knee arch, ~minutes on a compute node)
~/.conda/envs/nsga2/bin/python control_ablation.py   # -> results_control.json

# J2/J3 searches + ablation on the colored-Jones set (SLURM)
sbatch submit_nas_j2j3.slurm j2|j3|j2j3   # -> results_j2j3_<fs>/
sbatch submit_ablation_j2j3.slurm         # -> results_control_j2j3.json

# figures + deck
~/.conda/envs/nsga2/bin/python make_presentation_plots.py
~/.conda/envs/nsga2/bin/python make_phase_sweep.py
~/.conda/envs/nsga2/bin/python make_ab_barchart.py
~/.conda/envs/nsga2/bin/python make_ablation_barchart.py
~/.conda/envs/nsga2/bin/python make_arch_figure.py
~/.conda/envs/nsga2/bin/python make_phase_sweep_j2j3.py
~/.conda/envs/nsga2/bin/python make_j2j3_barchart.py
~/.conda/envs/nsga2/bin/python make_ablation_barchart_j2j3.py
~/.conda/envs/nsga2/bin/python make_arch_figure_j2j3.py
~/.conda/envs/nsga2/bin/python make_deck.py          # -> Jones_phase_feature.pptx (16 slides)
```

Note: the NAS score cache is keyed by architecture only — **use a fresh
`--outdir` whenever the feature set (input width) changes.**

---

## 3. Next steps (J₃ roadmap)

The ~98.5% J₂ ceiling is set by the data (distinct knots share a J₂ but differ
~3% in volume); published J₃ result is 99.34% (arXiv:2502.18575). With the new
dataset:

1. ✅ **Done 2026-07-18** — `parse_j2j3.py` (loader) + `compute_volumes.py`
   (SnapPy volumes → `data_j2+j3/volumes.json`, run from the dedicated
   `/scratch/cb5330/snappy-env` venv). Uses the `canon` column (consecutive
   integer q-power grid) so the q/t mix never enters; all features are
   shift-invariant or computed on the power-0-aligned canon. Validated:
   - every r=1 canon has |V(1)| = 1; canon reproduces the known Jones
     polynomials of 3_1 / 4_1 / 6_3; |V(−1)| = determinant for 15 knots with
     known determinants;
   - **all 1,419 hyperbolic r=1 polynomials match the old `jones+volume.txt`
     dataset up to mirror, and every joined volume agrees** (validates both
     the t→q canon conversion and the SnapPy name mapping);
   - spot-checked volumes against literature values (4_1, 5_2, 6_1, 6_2, 6_3,
     7_2) to ≤1e-6.
   Final count: 1,426 − 7 torus = **1,419 knots**; `j2`/`j3`/`j2j3` feature
   sets = 22/46/68 dims.
2. Sweep the evaluation phase for J₃ (adapt `make_phase_sweep.py`) instead of
   assuming 3π/4 transfers; add `[|J₃(q₀)|, …]` features at the empirical peak.
   *Partially done during step-1 verification:* on this dataset the peaks of
   corr(log|V(e^{iθ})|, log vol) are **θ ≈ 13π/16 for J₂ (corr 0.943;
   3π/4 only gives 0.886)** and **θ ≈ 5π/9 for J₃ (corr 0.959)** —
   `EVAL_PHASES` in `parse_j2j3.py` is set to these. ✅ **Done 2026-07-18** —
   `make_phase_sweep_j2j3.py` → `figures/phase_sweep_j2j3.png` (both curves,
   markers at 13π/16 / 5π/9, literature 3π/4 for reference).
3. ✅ **Done 2026-07-18** — identical NSGA-II budget (pop 40 / gens 40 / 4+7
   seeds) for J₂-only / J₃-only / J₂+J₃ on the *same knot set*, fresh outdirs:
   `run_nas.py --feature-set {j2,j3,j2j3}` via `submit_nas_j2j3.slurm <fs>`,
   jobs 16760125/26/27. Same split seed 0; all three feature sets cover the
   identical 1,419 knots, so the splits coincide across runs. **Results:**

   | Feature set | Knee | Params | Test MRE (mean ± std) | Test MRE (ens.) | Acc (ens.) |
   |---|---|---|---|---|---|
   | j2 (22 feat) | `[20, 19]` tanh | 879 | 0.0525 ± 0.0012 | 0.0516 | 94.84% |
   | j3 (46 feat) | `[16, 19]` GELU | 1,095 | 0.0269 ± 0.0038 | 0.0233 | 97.67% |
   | j2j3 (68 feat) | `[27, 27]` GELU | 2,647 | 0.0246 ± 0.0013 | **0.0208** | **97.92%** |

   J₃ alone cuts J₂'s error 2.2×; joined J₂+J₃ is best — the same ordering
   Hughes et al. (arXiv:2502.18575) measure on 177k knots with ~180k-param
   5-layer MLPs (their same-set comparison: J₂ 1.85% / J₃ 0.62% / J₂+J₃
   0.40% MRE), reproduced here with ~70–200× fewer parameters on ~125× less
   data.
4. ✅ **Done 2026-07-20** (job 16773089) — `control_ablation_j2j3.py` →
   `results_control_j2j3.json`: fixed j2j3 knee `[27, 27]` GELU, only the 6
   phase-evaluation columns ([Re, Im, |V(q₀)|] per polynomial) vary:
   **phase 97.92% > zeroed 97.56% > Gaussian noise 96.46%** ensemble accuracy
   (means 97.54 / 97.17 / 95.91). The J₂+J₃ gain rides on the phase signal;
   noise columns actively hurt. Figures `make_j2j3_barchart.py` /
   `make_ablation_barchart_j2j3.py`; deck rebuilt with a measured Part II
   (15 slides).

Caveat: this is a different knot population (1,426 knots ≤12 crossings vs
12,955 ≤13) — numbers are not directly comparable to §1.2/1.3; re-derive
splits and re-run the J₂ baseline on the new set.

---

## 4. File inventory

| File | Purpose |
|------|---------|
| `parse_data.py` | Parse `jones+volume.txt`, build 45-dim features (incl. phase eval), split & standardize |
| `nas.py` / `run_nas.py` | Search core / end-to-end driver |
| `control_ablation.py` | Random/zero/phase 3-column ablation at fixed knee arch |
| `submit_smoke.slurm`, `submit_nas.slurm`, `submit_nas_phase.slurm` | SLURM jobs |
| `jones+volume.txt` | J₂ + volume dataset (12,955 knots ≤13 crossings) |
| `data_j2+j3/dataset.numbers` | Colored Jones J₂…J₁₂ (r=1–11), 1,426 knots ≤12 crossings, **no volumes** |
| `parse_j2j3.py` | Loader/features for `data_j2+j3` (canon-based, shift-invariant, per-r phase eval) |
| `compute_volumes.py` | One-off SnapPy volume computation → `data_j2+j3/volumes.json` (1,419 knots) |
| `results_matched_42feat.json` | Matched-budget 42-feat A/B control (job 16544354) |
| `results_phase/` | 45-feat phase run: results.json, knee_model.pt, plots (job 16544559) |
| `results_baseline_42feat.json` | Earlier low-budget 42-feat run (do not use as control) |
| `results_control.json` | Measured random-feature ablation |
| `results_j2j3_{j2,j3,j2j3}/` | Matched-budget NAS on the colored-Jones set (jobs 16760125/26/27) |
| `control_ablation_j2j3.py` / `submit_ablation_j2j3.slurm` | Phase-column ablation at the fixed j2j3 knee (job 16773089) |
| `results_control_j2j3.json` | Measured j2j3 ablation: phase > zero > noise |
| `make_*.py` | Figures + PPTX deck |
| `PHASE_FEATURE_EXPERIMENT.md` | Full write-up of the phase-feature experiment + literature |
| `Jones_phase_feature.pptx` | 16-slide presentation (Part I: phase feature; Part II: J₂/J₃) |

`papers/` holds the reference PDFs (git-ignored; arXiv 1512.07906, 2211.01404,
2502.18575).
