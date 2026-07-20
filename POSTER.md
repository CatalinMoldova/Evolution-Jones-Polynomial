# Poster source — Jones polynomial → hyperbolic volume

Content for a research poster in three sections. Figure files referenced by
name live in `figures/` (plus two in `results_j2j3_j2j3/`). Author:
CatalinMoldova, 2026.

**Title:** Theory-guided features and architecture search for predicting
hyperbolic volume from Jones polynomials
**Subtitle:** A phase-evaluation feature + breaking the J₂ ceiling with the
adjoint J₃ — with networks under 3,000 parameters

---

## Motivation and Objectives

- **Task.** Predict a knot's hyperbolic volume — a geometric invariant,
  computed exactly by SnapPy from the ideal triangulation of the knot
  complement (rigid by Mostow's theorem) — from its Jones polynomial, a
  combinatorial/quantum invariant. Accuracy of the fit is empirical evidence
  for the conjectured link between quantum invariants and hyperbolic geometry
  (the volume conjecture).
- **Why learn it: a cheap stand-in for an intractable limit.** Computing the
  Jones polynomial at low color (J₂, or J₂+J₃) is cheap, but the volume
  conjecture's actual claim only holds in the limit of infinitely high color —
  a regime that's computationally intractable to reach directly. Rather than
  chasing that limit, this approach trains a network on cheap, low-color data
  paired with independently computed ground-truth volumes. The network learns,
  empirically, whatever complex relationship links these low-level polynomial
  coefficients to the high-level geometric quantity of volume. In effect, it
  substitutes learned pattern-recognition for the intractable exact
  computation, extracting at inference time in a single forward pass what
  would otherwise require an infeasible high-color limit.
- **Objectives.** (1) Quantify, under an exactly matched search budget, the
  gain from adding the volume-conjecture phase evaluation V(q₀) as an input
  feature. (2) Prove via ablation that the gain is signal, not extra
  dimensionality. (3) Measure J₂ vs J₃ (adjoint) vs J₂+J₃ on an identical
  knot set, split, and budget — does the richer invariant break the J₂
  information ceiling?

## Methodology

- **Data.** (A) 12,955 hyperbolic knots ≤13 crossings: J₂ (34 integer
  coefficients) + volume. (B) New colored-Jones table: 1,426 knots ≤12
  crossings with J₂ and J₃; volumes computed with SnapPy; 7 non-hyperbolic
  torus knots dropped → 1,419 knots. Dataset B required cleaning: mixed q/t
  variable conventions and framing-polluted exponent shifts → all features
  built on a shift-invariant canonical form; validated against dataset A
  (all 1,419 shared polynomials and volumes agree).
- **Features.** Summary statistics (span, #nonzero, L1/L2 norms, V(1),
  |V(−1)|) + raw coefficients + the phase evaluation [Re, Im, |·|] of V(q₀)
  on the unit circle. The evaluation angle is swept per polynomial rather
  than assumed: corr(log|V(e^{iθ})|, log vol) peaks at θ = 13π/16 for J₂
  (corr 0.943) and θ = 5π/9 for J₃ (corr 0.959) — a single scalar tracks the
  volume almost as well as a trained network. [figures/phase_sweep.png,
  figures/phase_sweep_j2j3.png, figures/feature_correlation.png]
- **Search.** NSGA-II multi-objective neural architecture search over MLP
  genomes (depth 1–6, widths 16–512, activation, dropout, L2, learning rate,
  batch size). Objectives: validation MRE and log₁₀(params); overfitting gap
  as a constraint; multi-fidelity evaluation with memoization. The Pareto
  knee is retrained from scratch (7 seeds × 500 epochs) and only then
  evaluated on an untouched test set (fixed 70/15/15 split). Every
  comparison uses the identical budget: pop 40, 40 generations, 4 search
  seeds. MRE = mean(|pred − true|/true); accuracy = 1 − MRE.
- **Controls.** Fix the knee architecture and training protocol; replace the
  phase columns with zeros or Gaussian noise — any difference is purely the
  information content of those columns.

## Results

- **Phase feature (dataset A, clean A/B at matched budget):** 42-feature
  control 97.14% → 45-feature + phase 97.31% ensemble accuracy (test MRE
  0.0307 → 0.0289 mean, ≈6% relative error reduction, ~3σ); the entire
  Pareto front shifts down. Best model: 45→32→18→54→18→1 GELU MLP, 4,101
  params. [figures/ab_accuracy.png, figures/pareto_overlay.png,
  figures/architecture.png]
- **Ablation A:** phase 97.31% > zeroed 97.22% > noise 97.11% — the gain is
  the volume signal. [figures/ab_ablation.png]
- **J₃ breaks the J₂ ceiling (dataset B, same 1,419 knots, same split, same
  budget):** J₂ 94.84% → J₃ 97.67% → J₂+J₃ 97.92% ensemble accuracy (test
  MRE 5.16% → 2.33% → 2.08%). J₃ alone cuts J₂'s error 2.2×.
  [figures/j2j3_accuracy.png, results_j2j3_j2j3/test_predictions.png]
- **Best J₂+J₃ architecture:** 68→27→27→1 GELU MLP, 2,647 parameters
  (dropout 0.007, lr 5.5e-3, L2 2.0e-6, batch 128); test MRE 0.0246 ± 0.0013,
  ensemble 0.0208. [figures/architecture_j2j3.png]
- **Ablation B (fixed [27,27] knee, only the 6 phase columns vary):** phase
  97.92% > zeroed 97.56% > Gaussian noise 96.46% — the gain rides on the
  phase signal; noise actively hurts. [figures/ab_ablation_j2j3.png]
- **Context vs literature (accuracy per parameter):** the J₂ → J₃ → J₂+J₃
  error ordering of Hughes et al. 2025 (arXiv:2502.18575; 5-layer ~180k-param
  MLPs on 177k knots: 1.85% / 0.62% / 0.40% MRE) reproduces exactly with
  ~70–200× smaller models on ~125× less data. Earlier work: Jejjala–Kar–
  Parrikar 2019 (2×100 MLP, ~12k params, 2.45%), Craven et al. 2022 (3×100,
  ~22k params, ~3%). Flat Pareto fronts and sub-3k-param knees throughout:
  accuracy is limited by the information in the invariant, not model
  capacity.
- **Future work.** Scale J₃ to the ≤13-crossing census; push the substitution
  one level higher — predict the volume-correlated behavior of J₄, J₅, …
  directly from cheap low-color data, avoiding their computation entirely;
  interpret the knee networks symbolically.
