# Evolution — Jones Polynomial → Hyperbolic Volume

Multi-objective **neural architecture search (NAS)** that learns to predict a knot's
**hyperbolic volume** from its **Jones polynomial**. Instead of hand-designing one
network, we run **NSGA-II** (an evolutionary algorithm) over a space of MLP
architectures and hyperparameters, trading off accuracy against model size, then
retrain the best trade-off ("knee") point at full budget and report error on an
untouched test set.

The task is the knot-theory regression problem popularized by Davies et al.
(*"Advancing mathematics by guiding human intuition with AI"*, Nature 2021): a
surprisingly simple, low-parameter model recovers the Jones→volume relationship
very accurately.

---

## Results at a glance

From the reference full run (before the feature/target upgrades in this repo):

- Pareto front is **flat past the knee** — going from **2,345** to **421,672**
  parameters (180×) improved validation MRE only **0.0329 → 0.0316**. Tiny models
  win.
- **Knee architecture:** depth-2 MLP, widths `[37, 25]`, GELU, ~**2,345 parameters**.
- **Test MRE = 0.0288 ± 0.0002** (mean over 5 seeds); **0.0276** as a seed ensemble.

MRE = mean relative error, `mean(|pred − true| / true)`, always computed on the
natural (un-standardized) volume scale.

---

## Method

### Data (`parse_data.py`)
- **Input** `jones+volume.txt`: a Mathematica association where each knot `i`
  contributes `{i,1}->{34 integer coefficients}` (its Jones polynomial) and
  `{i,2}->volume`. The two are parsed by regex and paired on `i`.
- **Features (42-dim):** an 8-value polynomial summary — min/max nonzero power,
  degree span, number of nonzero terms, L1 & L2 coefficient norms, `V(1)` (sum of
  coefficients) and `V(-1)` (alternating sum, determinant-related) — followed by
  the 34 raw coefficients. Coefficient index `k` corresponds to power `k − 15`.
- **Split:** fixed-seed **70 / 15 / 15** train/val/test (9068 / 1943 / 1944 rows).
  Features standardized with **train-only** statistics. The **test set is carved
  out up front and never touched until the final evaluation.**
- **Optional `--log-target`:** learn on `log(volume)` (error still reported on the
  natural scale).

### Search core (`nas.py`)
- **Genome → MLP:** `depth` (1–6), per-layer `width` (16–512), `activation`
  (relu/tanh/gelu), `dropout` (0–0.5), L2 `alpha` (1e-6…1e-1), learning rate
  (1e-4…1e-2), `batch_size` (128/256/512).
- **Training** (`fit`): Adam + **Huber loss** on the standardized target, with
  **early stopping** on internal validation loss (best weights restored).
  Multi-fidelity: candidates train on a subsample for a few epochs.
- **Two objectives, both minimized:**
  1. `f1` = validation MRE (averaged over a few seeds)
  2. `f2` = `log10(parameter count)`
- **Overfitting is a constraint, not an objective:** an optional
  `--gap-constraint` keeps `val_mre − train_mre` below a threshold. (Making the gap
  an objective would reward useless under-powered nets, which also have ~0 gap.)
- Each genome's score is **memoized to disk** by architectural hash.

### Driver (`run_nas.py`)
1. Parse + split data (test held out).
2. Run NSGA-II over the genome, evaluating candidates cheaply in parallel across
   CPU workers (fork so workers inherit the data for free; math libraries pinned to
   1 thread so parallelism comes only from the worker pool).
3. Plot the **Pareto front** and select the **knee** (max perpendicular distance
   from the chord between the two extremes — "most bang for the buck").
4. **Retrain the knee architecture from scratch at full budget** over several seeds
   on all training data; report test MRE (per-seed mean ± std and an ensemble).

Outputs land in the `--outdir`:
- `pareto_front.png` — accuracy vs. complexity, knee starred
- `test_predictions.png` — predicted-vs-true scatter + error-by-volume-decile
- `knee_model.pt` — best-seed trained weights + config
- `results.json` — full front, knee arch, and test metrics

---

## Running it

### Environment
Python 3.12 with `numpy`, `torch`, `pymoo`, `matplotlib`. On the reference cluster:

```bash
conda create -n nsga2 python=3.12
conda activate nsga2
pip install numpy torch pymoo matplotlib
```

### Quick sanity check (~30 s)
```bash
python run_nas.py --data jones+volume.txt --outdir results_smoke --smoke --n-jobs 8
```

### Full search
```bash
python run_nas.py \
  --data jones+volume.txt --outdir results \
  --pop 40 --gens 40 --epochs 80 --subsample 0 --patience 15 \
  --search-seeds 4 --log-target --gap-constraint 0.01 \
  --n-jobs 32 --device cpu \
  --final-epochs 500 --final-seeds 7 --final-patience 50
```
`--subsample 0` uses all training rows.

### On a SLURM cluster
```bash
sbatch submit_smoke.slurm   # ~1 min pipeline check -> results_smoke/
sbatch submit_nas.slurm     # full run           -> results/
```
Both scripts pin `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` and run the search on a
compute node. **Do not run the full search on a login node** — each candidate
trains a PyTorch MLP, and unpinned torch grabs one thread per core.

---

## Files

| File | Purpose |
|------|---------|
| `parse_data.py`     | Parse the Mathematica file, build 42-dim features, split & standardize |
| `nas.py`            | Genome↔MLP, training loop, genome evaluation, pymoo problem |
| `run_nas.py`        | End-to-end driver: search → Pareto → knee → full retrain → test |
| `submit_smoke.slurm`| Fast end-to-end sanity job |
| `submit_nas.slurm`  | Full NAS job (full data, pop 40 × gens 40, log-target, gap constraint) |
| `jones+volume.txt`  | Dataset (Jones coefficients + hyperbolic volumes) |

`results/` (live full-run output), model weights (`*.pt`), evaluation caches, and
SLURM logs are git-ignored; `results_smoke/` sample plots are kept as examples.

---

## Ideas for going further
- Richer / knot-theoretic features (evaluations at roots of unity, signature).
- Tune or relax the gap constraint once the true train/val gap is known.
- Larger population / more generations (search is embarrassingly parallel and cached).
- Successive-halving so promising genomes get more epochs.
