# PGLA — Probe-Guided Logit Adjustment

## Overview

PGLA tests whether the Know-But-Don't-Act Gap can be closed by extracting
the internally encoded misleading signal and feeding it back to the output
distribution at inference time.  The intervention requires no input
perturbation and no double forward pass.

## Two-step workflow

### Step 1 — Train probes

```bash
python train_probes.py --model <model_name>
```

- Loads hidden states at the peak probe layer `l*` (from `config.PEAK_LAYERS`).
- Performs a stratified 25/75 train/eval split (500 train, 1500 eval samples).
- Trains a 2-layer MLP probe (256 hidden units, ReLU, Adam lr=1e-3, 100 epochs)
  to predict P(misleading | hidden_state).
- Also trains a logistic-regression linear probe for comparison.
- Computes β = mean(L_E − L_F) on standard training samples (E/F positional bias).
- Saves per-sample probe scores to `<OUTPUT_DIR>/pgla/<model>/probe_outputs.pt`.

### Step 2 — Sweep hyperparameters

```bash
python sweep.py --model <model_name>
# or with linear probe scores:
python sweep.py --model <model_name> --probe-type linear
```

- Loads `probe_outputs.pt` produced by Step 1.
- Evaluates all 162 configurations (see grid below) on the 75% eval set using
  5-fold stratified cross-validation.
- For each fold: tune on 4 folds (select config maximising balanced accuracy),
  evaluate on held-out 5th fold.
- Reports mean test performance across folds, Pareto frontier, and top configs.
- Saves results to `<OUTPUT_DIR>/pgla/<model>/sweep_results_<probe_type>.json`.

## Intervention formula

Let `g = P_mis^p` be the confidence gate and
`Δ = max(L_A..D) − max(L_E, L_F)` the content-vs-rejection logit gap.

```
L'_E = L_E + σ(γ(g − α)) · (s·Δ + δ) − β/2
L'_F = L_F + σ(γ(g − α)) · (s·Δ + δ) + β/2
```

The sigmoid gate applies near-zero adjustment when the probe is confident the
input is standard, preserving standard accuracy.  The debiasing term β corrects
for systematic positional preference between E and F.

## Hyperparameter grid — 162 configurations (3×2×3×3×3)

| Symbol | Name                  | Values             |
|--------|-----------------------|--------------------|
| γ      | `gamma` (steepness)   | 0.5, 1.0, 2.0      |
| p      | `probe_power`         | 1.0, 2.0           |
| α      | `alpha` (threshold)   | 0.3, 0.5, 1.0      |
| s      | `gap_scale`           | 0.75, 1.0, 1.5     |
| δ      | `fixed_boost`         | 5, 8, 12           |

β is not swept — it is computed directly from standard training samples.

## Metric

```
balanced_acc = 0.5 × (mean_standard_acc + mean_misleading_acc)
```

where standard_acc and misleading_acc are each the mean over the two
sub-splits (vision + audio).
