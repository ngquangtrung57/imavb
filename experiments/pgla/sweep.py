#!/usr/bin/env python3
"""PGLA hyperparameter sweep with 5-fold stratified CV (§5 + App F).

Loads probe outputs produced by train_probes.py, then exhaustively evaluates
all 162 parameter configurations (3×2×3×3×3) using 5-fold stratified CV on
the 75% evaluation set.

Paper formula (Eq. 1–2):
    g = P_mis^p                         # confidence gate
    Δ = max(L_A..D) - max(L_E, L_F)     # content-vs-rejection gap
    L'_E = L_E + σ(γ(g - α)) * (s*Δ + δ) - β/2
    L'_F = L_F + σ(γ(g - α)) * (s*Δ + δ) + β/2

Grid (162 configs = 3×2×3×3×3):
    γ (sigmoid_steepness) : [0.5, 1.0, 2.0]
    p (probe_power)       : [1.0, 2.0]
    α (confidence_threshold): [0.3, 0.5, 1.0]
    s (gap_scale)         : [0.75, 1.0, 1.5]
    δ (fixed_boost)       : [5, 8, 12]

Metric: balanced_acc = 0.5 * (mean_standard_acc + mean_misleading_acc)

Usage:
    python sweep.py --model ola
    python sweep.py --model qwen2_5_omni --probe-type linear
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS, OUTPUT_DIR, RANDOM_STATE

MISLEADING_CORRECT: dict[str, str] = {"misleading_vision": "E", "misleading_audio": "F"}

# ── Hyperparameter grid (paper App F) ────────────────────────────────────────
GRID: dict[str, list[float | int]] = {
    "gamma": [0.5, 1.0, 2.0],        # sigmoid steepness γ
    "probe_power": [1.0, 2.0],        # probe confidence power p
    "alpha": [0.3, 0.5, 1.0],         # confidence threshold α
    "gap_scale": [0.75, 1.0, 1.5],    # gap-adaptive scale s
    "fixed_boost": [5, 8, 12],        # fixed boost δ
}

N_FOLDS = 5


# ---------------------------------------------------------------------------
# Intervention
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def apply_intervention(
    choice_logits: dict[str, float],
    p_mis: float,
    ef_bias: float,
    gamma: float,
    probe_power: float,
    alpha: float,
    gap_scale: float,
    fixed_boost: float,
) -> dict[str, float]:
    """Apply PGLA confidence-gated logit adjustment (Eq. 1–2 of the paper).

    Args:
        choice_logits: Original logits dict {A: float, ..., F: float}.
        p_mis: P(misleading) from probe (before power).
        ef_bias: Beta — mean(E – F) on standard training samples.
        gamma: Sigmoid steepness γ.
        probe_power: Power p applied to p_mis.
        alpha: Confidence threshold α.
        gap_scale: Gap-adaptive scale s.
        fixed_boost: Fixed boost δ.

    Returns adjusted logits dict.
    """
    logits = dict(choice_logits)

    if not logits or len(logits) < 6:
        return logits  # no intervention if logits are incomplete

    g = float(p_mis) ** probe_power  # confidence gate
    gate = _sigmoid(gamma * (g - alpha))

    # Δ = max(L_A..D) - max(L_E, L_F)
    abcd_max = max(logits.get(c, -1e9) for c in "ABCD")
    ef_max = max(logits.get("E", -1e9), logits.get("F", -1e9))
    delta = abcd_max - ef_max

    boost = gate * (gap_scale * delta + fixed_boost)
    beta = ef_bias  # beta = mean(E – F) on std train samples

    logits["E"] = logits.get("E", 0.0) + boost - beta / 2.0
    logits["F"] = logits.get("F", 0.0) + boost + beta / 2.0
    return logits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def predict(logits: dict[str, float]) -> str:
    return max(logits, key=logits.__getitem__)


def get_correct_answer(sample: dict) -> str:
    if sample["split"] in MISLEADING_CORRECT:
        return MISLEADING_CORRECT[sample["split"]]
    return sample["correct_answer"]


def build_param_grid() -> list[dict]:
    """Return all 162 parameter configurations."""
    keys = list(GRID.keys())
    configs = [
        dict(zip(keys, combo))
        for combo in itertools.product(*[GRID[k] for k in keys])
    ]
    assert len(configs) == 162, f"Expected 162 configs, got {len(configs)}"
    return configs


def evaluate_params(
    samples: list[dict],
    params: dict,
    probe_key: str = "p_mis_mlp",
) -> dict:
    """Evaluate one parameter configuration on a list of samples.

    Returns dict with standard_acc, misleading_acc, balanced_acc, split_acc.
    """
    split_correct: dict[str, int] = {}
    split_total: dict[str, int] = {}

    for s in samples:
        split = s["split"]
        correct = get_correct_answer(s)
        p_mis = s.get(probe_key, s.get("p_mis_mlp", 0.0))
        ef_bias = s.get("ef_bias", 0.0)

        adjusted = apply_intervention(
            s["choice_logits"],
            p_mis=p_mis,
            ef_bias=ef_bias,
            gamma=params["gamma"],
            probe_power=params["probe_power"],
            alpha=params["alpha"],
            gap_scale=params["gap_scale"],
            fixed_boost=params["fixed_boost"],
        )
        pred = predict(adjusted)

        split_total[split] = split_total.get(split, 0) + 1
        if pred == correct:
            split_correct[split] = split_correct.get(split, 0) + 1

    split_acc = {
        split: split_correct.get(split, 0) / total * 100
        for split, total in split_total.items()
    }
    std_accs = [v for k, v in split_acc.items() if "standard" in k]
    mis_accs = [v for k, v in split_acc.items() if "misleading" in k]
    std_acc = float(np.mean(std_accs)) if std_accs else 0.0
    mis_acc = float(np.mean(mis_accs)) if mis_accs else 0.0
    bal_acc = (std_acc + mis_acc) / 2.0

    return {
        "params": params,
        "split_acc": split_acc,
        "standard_acc": round(std_acc, 4),
        "misleading_acc": round(mis_acc, 4),
        "balanced_acc": round(bal_acc, 4),
        "n_samples": len(samples),
    }


# ---------------------------------------------------------------------------
# 5-fold CV sweep
# ---------------------------------------------------------------------------

def run_cv_sweep(
    eval_samples: list[dict],
    param_grid: list[dict],
    probe_key: str = "p_mis_mlp",
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_STATE,
) -> dict:
    """5-fold stratified CV sweep on the eval partition.

    For each fold: tune on 4 folds, evaluate best config on held-out fold.
    Returns mean test performance across folds for each config.

    Protocol (App F):
        - Probe is already trained on fixed 25% training set.
        - CV is on the 75% eval set only.
        - Each fold: select config maximising balanced_acc on 4/5 tune folds,
          then evaluate on the 5th (test) fold.

    Returns dict with keys:
        cv_results: list of dicts (one per config) with mean_test_balanced_acc etc.
        best_config: the config with highest mean test balanced_acc
        best_cv_balanced_acc: float
        fold_details: list (one per fold) with best_params + test metrics
        pareto_frontier: Pareto-optimal configs by (standard_acc, misleading_acc)
    """
    # Build stratified label array: 0=standard, 1=misleading
    labels = np.array([1 if "misleading" in s["split"] else 0 for s in eval_samples])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Accumulate test scores per config across folds
    # config_test_scores[config_idx] = list of balanced_acc from each fold's test
    config_test_scores: dict[int, list[float]] = {i: [] for i in range(len(param_grid))}
    config_test_std: dict[int, list[float]] = {i: [] for i in range(len(param_grid))}
    config_test_mis: dict[int, list[float]] = {i: [] for i in range(len(param_grid))}
    fold_details = []

    for fold_idx, (tune_idx, test_idx) in enumerate(skf.split(np.zeros(len(eval_samples)), labels)):
        tune_samples = [eval_samples[i] for i in tune_idx]
        test_samples = [eval_samples[i] for i in test_idx]

        # Evaluate all configs on tune set to pick best
        tune_results = []
        for cfg_idx, params in enumerate(param_grid):
            r = evaluate_params(tune_samples, params, probe_key=probe_key)
            tune_results.append((cfg_idx, r["balanced_acc"]))

        best_cfg_idx = max(tune_results, key=lambda x: x[1])[0]
        best_params = param_grid[best_cfg_idx]

        # Evaluate best config on test fold
        test_r = evaluate_params(test_samples, best_params, probe_key=probe_key)

        fold_details.append({
            "fold": fold_idx,
            "best_params": best_params,
            "tune_balanced_acc": tune_results[best_cfg_idx][1],
            "test_balanced_acc": test_r["balanced_acc"],
            "test_standard_acc": test_r["standard_acc"],
            "test_misleading_acc": test_r["misleading_acc"],
            "test_split_acc": test_r["split_acc"],
            "n_tune": len(tune_samples),
            "n_test": len(test_samples),
        })

        print(
            f"  fold {fold_idx + 1}/{n_folds}: "
            f"tune_bal={tune_results[best_cfg_idx][1]:.2f}  "
            f"test_bal={test_r['balanced_acc']:.2f}  "
            f"(std={test_r['standard_acc']:.2f}, mis={test_r['misleading_acc']:.2f})"
        )

        # Also accumulate scores for every config on the test fold
        # (so we can report mean test performance across all folds per config)
        for cfg_idx, params in enumerate(param_grid):
            r = evaluate_params(test_samples, params, probe_key=probe_key)
            config_test_scores[cfg_idx].append(r["balanced_acc"])
            config_test_std[cfg_idx].append(r["standard_acc"])
            config_test_mis[cfg_idx].append(r["misleading_acc"])

    # Aggregate per-config mean test metrics across folds
    cv_results = []
    for cfg_idx, params in enumerate(param_grid):
        mean_bal = float(np.mean(config_test_scores[cfg_idx]))
        mean_std = float(np.mean(config_test_std[cfg_idx]))
        mean_mis = float(np.mean(config_test_mis[cfg_idx]))
        cv_results.append({
            "params": params,
            "mean_test_balanced_acc": round(mean_bal, 4),
            "mean_test_standard_acc": round(mean_std, 4),
            "mean_test_misleading_acc": round(mean_mis, 4),
        })

    cv_results.sort(key=lambda x: x["mean_test_balanced_acc"], reverse=True)

    best_config = cv_results[0]
    pareto = _compute_pareto(cv_results)

    # Tune–test gap: mean over folds of (tune_bal - test_bal) using fold_details
    gaps = [
        fd["tune_balanced_acc"] - fd["test_balanced_acc"] for fd in fold_details
    ]
    mean_gap = float(np.mean(gaps))
    print(f"  Mean tune–test gap: {mean_gap:.2f}pp")

    return {
        "cv_results": cv_results,
        "best_config": best_config,
        "best_cv_balanced_acc": best_config["mean_test_balanced_acc"],
        "fold_details": fold_details,
        "pareto_frontier": pareto,
        "mean_tune_test_gap": round(mean_gap, 4),
    }


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def _compute_pareto(cv_results: list[dict]) -> list[dict]:
    """Find Pareto-optimal configs maximising both standard and misleading acc."""
    pareto = []
    for r in cv_results:
        dominated = any(
            (other["mean_test_standard_acc"] >= r["mean_test_standard_acc"]
             and other["mean_test_misleading_acc"] >= r["mean_test_misleading_acc"]
             and (other["mean_test_standard_acc"] > r["mean_test_standard_acc"]
                  or other["mean_test_misleading_acc"] > r["mean_test_misleading_acc"]))
            for other in cv_results
            if other is not r
        )
        if not dominated:
            pareto.append(r)
    pareto.sort(key=lambda x: x["mean_test_balanced_acc"], reverse=True)
    return pareto


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PGLA 162-config sweep with 5-fold CV (§5 + App F)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODELS,
        help="Model name (e.g. ola, qwen2_5_omni)",
    )
    parser.add_argument(
        "--probe-type",
        type=str,
        default="mlp",
        choices=["mlp", "linear"],
        help="Which probe scores to use (default: mlp)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help=f"Random seed for CV folds (default: {RANDOM_STATE})",
    )
    parser.add_argument(
        "--probe-outputs",
        type=str,
        default=None,
        help=(
            "Path to probe_outputs.pt (default: "
            "<OUTPUT_DIR>/pgla/<model>/probe_outputs.pt)"
        ),
    )
    args = parser.parse_args()

    # Load probe outputs
    if args.probe_outputs:
        probe_path = Path(args.probe_outputs)
    else:
        probe_path = Path(OUTPUT_DIR) / "pgla" / args.model / "probe_outputs.pt"

    if not probe_path.exists():
        print(f"[error] Probe outputs not found at {probe_path}")
        print("  Run train_probes.py first.")
        sys.exit(1)

    print(f"[{args.model}] Loading probe outputs from {probe_path}")
    probe_data = torch.load(probe_path, map_location="cpu", weights_only=False)
    all_samples = probe_data["samples"]

    # Filter to eval partition only (75%)
    eval_samples = [s for s in all_samples if s.get("partition") == "eval"]
    print(
        f"[{args.model}] Eval samples: {len(eval_samples)} / {len(all_samples)} total"
    )

    if not eval_samples:
        print("[error] No eval samples found. Check probe_outputs.pt.")
        sys.exit(1)

    # Select probe key
    probe_key = "p_mis_mlp" if args.probe_type == "mlp" else "p_mis_linear"
    print(f"[{args.model}] Using probe type: {args.probe_type}  (key: {probe_key})")

    # Build 162-config grid
    param_grid = build_param_grid()
    print(f"[{args.model}] Sweeping {len(param_grid)} configurations with {N_FOLDS}-fold CV...")

    results = run_cv_sweep(
        eval_samples=eval_samples,
        param_grid=param_grid,
        probe_key=probe_key,
        n_folds=N_FOLDS,
        seed=args.seed,
    )

    # Save results
    out_dir = Path(OUTPUT_DIR) / "pgla" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sweep_results_{args.probe_type}.json"

    # Make JSON-serialisable
    def _to_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_to_json)
    print(f"[{args.model}] Saved sweep results -> {out_path}")

    # Print summary
    best = results["best_config"]
    print(f"\n=== Best config for {args.model} (probe={args.probe_type}) ===")
    print(f"  Params       : {best['params']}")
    print(f"  Mean test bal: {best['mean_test_balanced_acc']:.2f}%")
    print(f"  Mean test std: {best['mean_test_standard_acc']:.2f}%")
    print(f"  Mean test mis: {best['mean_test_misleading_acc']:.2f}%")
    print(f"  Tune–test gap: {results['mean_tune_test_gap']:.2f}pp")

    print(f"\n=== Pareto frontier ({len(results['pareto_frontier'])} configs) ===")
    print(f"  {'Bal':>7}  {'Std':>7}  {'Mis':>7}  Params")
    for r in results["pareto_frontier"][:10]:
        print(
            f"  {r['mean_test_balanced_acc']:>6.2f}%"
            f"  {r['mean_test_standard_acc']:>6.2f}%"
            f"  {r['mean_test_misleading_acc']:>6.2f}%"
            f"  {r['params']}"
        )

    print(f"\n=== Top 5 by balanced accuracy ===")
    print(f"  {'Bal':>7}  {'Std':>7}  {'Mis':>7}  Params")
    for r in results["cv_results"][:5]:
        print(
            f"  {r['mean_test_balanced_acc']:>6.2f}%"
            f"  {r['mean_test_standard_acc']:>6.2f}%"
            f"  {r['mean_test_misleading_acc']:>6.2f}%"
            f"  {r['params']}"
        )


if __name__ == "__main__":
    main()
