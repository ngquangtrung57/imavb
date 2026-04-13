#!/usr/bin/env python3
"""
Residualized Probe Analysis for IMAVB paper.

Investigates how much of the hidden-state probe signal is genuine
cross-modal encoding vs. text features encoded in representations.

Approach (nested 4-fold CV to avoid leakage):
  For each fold:
    1. Fit text projection (Ridge: hidden_states → sentence-BERT embeddings) on TRAIN
    2. Residualize TRAIN and TEST hidden states using the SAME projection
    3. Train probe on residualized TRAIN, evaluate on residualized TEST

Also computes:
  - Original probe accuracy (sanity check — should match paper Table 6)
  - TF-IDF text baseline (should match paper's 73.4% / 71.4%)
  - Sentence-BERT text baseline (stronger text control)
  - Bootstrap significance tests (probe > text, residualized > chance)

CPU-ONLY — safe to run alongside GPU processes.

Usage:
    CUDA_VISIBLE_DEVICES="" python residualized_probing.py [--models MODEL1,MODEL2]
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# Force CPU before any torch/transformers import
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config imports ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    HIDDEN_STATES_ROOT,
    MODELS,
    OUTPUT_DIR,
    PEAK_LAYERS,
    RANDOM_STATE,
    SPLITS,
    N_BOOTSTRAP,
)

# ── Constants ─────────────────────────────────────────────────────────────
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
N_FOLDS = 4
PAPER_MODELS = MODELS


# ── Data Loading ─────────────────────────────────────────────────────────

def load_hidden_states(model_name: str) -> list[dict]:
    """Load all .pt files for a model. Returns list of sample dicts."""
    model_dir = Path(HIDDEN_STATES_ROOT, model_name)
    samples = []
    for split in SPLITS:
        split_dir = model_dir / split
        if not split_dir.exists():
            print(f"  WARNING: missing split dir {split_dir}")
            continue
        for pt_file in sorted(split_dir.glob("*.pt")):
            data = torch.load(str(pt_file), map_location="cpu", weights_only=True)
            samples.append(data)
    return samples


def extract_arrays(samples: list[dict]):
    """Extract hidden states, labels, questions, and split info from samples."""
    hs_list = [s["hidden_states"].float().numpy() for s in samples]
    hs = np.stack(hs_list, axis=0)  # (N, n_layers, hidden_dim)

    is_misleading = np.array(
        [int(s["metadata"].get("is_misleading", False)) for s in samples]
    )
    questions = [s["metadata"].get("question", "") for s in samples]

    # Modality: vision vs audio (from split name)
    modalities = []
    for s in samples:
        split = s["metadata"].get("split", "")
        if "vision" in split:
            modalities.append("vision")
        else:
            modalities.append("audio")
    modalities = np.array(modalities)

    return hs, is_misleading, questions, modalities


# ── Text Features ────────────────────────────────────────────────────────

def compute_sbert_embeddings(questions: list[str]) -> np.ndarray:
    """Compute sentence-BERT embeddings (CPU only)."""
    from sentence_transformers import SentenceTransformer

    print(f"  Loading sentence-BERT model ({SBERT_MODEL_NAME}) on CPU...")
    model = SentenceTransformer(SBERT_MODEL_NAME, device="cpu")
    embeddings = model.encode(questions, show_progress_bar=False, batch_size=128)
    print(f"  Sentence-BERT embeddings: {embeddings.shape}")
    return embeddings


def run_tfidf_cv_probing(
    questions: list[str],
    y: np.ndarray,
    n_folds: int = N_FOLDS,
) -> dict:
    """Run TF-IDF probing with per-fold fit to avoid vocabulary leakage."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_accs = []
    questions_arr = np.array(questions)
    for train_idx, test_idx in skf.split(questions_arr, y):
        tfidf = TfidfVectorizer(max_features=5000)
        X_tr = tfidf.fit_transform(questions_arr[train_idx]).toarray()
        X_te = tfidf.transform(questions_arr[test_idx]).toarray()
        acc = train_and_eval_probe(X_tr, y[train_idx], X_te, y[test_idx])
        fold_accs.append(acc)
    return {
        "mean": float(np.mean(fold_accs)),
        "std": float(np.std(fold_accs, ddof=1)),
        "per_fold": [float(a) for a in fold_accs],
    }


# ── Residualization ──────────────────────────────────────────────────────

def residualize_hidden_states(
    h_train: np.ndarray,
    h_test: np.ndarray,
    text_train: np.ndarray,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove text-predictive component from hidden states via orthogonal projection.

    Method:
      1. Fit Ridge regression: h → text_embedding (on train)
      2. Compute projection via SVD of the weight matrix W
      3. Project h onto null space of W: h_residual = h - V @ V^T @ h

    The projection is fit on train data only, then applied to both splits.

    Args:
        h_train: (N_train, hidden_dim)
        h_test: (N_test, hidden_dim)
        text_train: (N_train, text_dim) sentence-BERT embeddings (train only)
        alpha: Ridge regularization strength (α=1.0, SVD threshold 10⁻⁵ × s_max)

    Returns:
        h_train_residual, h_test_residual
    """
    # Standardize hidden states (fit on train)
    scaler_h = StandardScaler()
    h_train_s = scaler_h.fit_transform(h_train)
    h_test_s = scaler_h.transform(h_test)

    # Standardize text embeddings (fit on train)
    scaler_t = StandardScaler()
    text_train_s = scaler_t.fit_transform(text_train)

    # Fit Ridge regression: h → text_embedding
    # W has shape (text_dim, hidden_dim) — maps hidden states to text space
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(h_train_s, text_train_s)
    W = ridge.coef_  # (text_dim, hidden_dim)

    # Compute projection matrix onto row space of W via SVD
    # W = U @ S @ V^T, where V columns span the text-predictive subspace
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    # Keep components with non-negligible singular values (threshold = 10⁻⁵ × s_max)
    threshold = S.max() * 1e-5
    k = int(np.sum(S > threshold))
    V_k = Vt[:k].T  # (hidden_dim, k) — basis of text-predictive subspace

    # Project onto null space: h_res = h - V_k @ V_k^T @ h
    h_train_res = h_train_s - h_train_s @ V_k @ V_k.T
    h_test_res = h_test_s - h_test_s @ V_k @ V_k.T

    return h_train_res, h_test_res


# ── Probing ──────────────────────────────────────────────────────────────

def train_and_eval_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Train logistic regression probe, return test accuracy."""
    clf = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", random_state=RANDOM_STATE
    )
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def run_cv_probing(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = N_FOLDS,
) -> dict:
    """Run stratified k-fold CV probing. Returns mean, std, per-fold accuracies."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_accs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        acc = train_and_eval_probe(X_tr, y[train_idx], X_te, y[test_idx])
        fold_accs.append(acc)
    return {
        "mean": float(np.mean(fold_accs)),
        "std": float(np.std(fold_accs, ddof=1)),
        "per_fold": [float(a) for a in fold_accs],
    }


def run_nested_residualized_probing(
    hs_at_layer: np.ndarray,
    y: np.ndarray,
    text_embeddings: np.ndarray,
    n_folds: int = N_FOLDS,
) -> dict:
    """
    Nested CV: residualization is fitted INSIDE each fold to avoid leakage.

    For each fold:
      1. Fit text projection on train fold
      2. Residualize train and test
      3. Train probe on residualized train, evaluate on residualized test
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_accs = []
    for train_idx, test_idx in skf.split(hs_at_layer, y):
        h_train = hs_at_layer[train_idx]
        h_test = hs_at_layer[test_idx]
        text_train = text_embeddings[train_idx]

        # Residualize (fit projection on train only)
        h_train_res, h_test_res = residualize_hidden_states(
            h_train, h_test, text_train
        )

        # Re-standardize after projection for fair comparison with original probe
        scaler = StandardScaler()
        h_train_res = scaler.fit_transform(h_train_res)
        h_test_res = scaler.transform(h_test_res)

        # Train probe on residualized hidden states
        acc = train_and_eval_probe(h_train_res, y[train_idx], h_test_res, y[test_idx])
        fold_accs.append(acc)

    return {
        "mean": float(np.mean(fold_accs)),
        "std": float(np.std(fold_accs, ddof=1)),
        "per_fold": [float(a) for a in fold_accs],
    }


# ── Bootstrap Significance ───────────────────────────────────────────────

def bootstrap_paired_test(
    accs_a: list[float],
    accs_b: list[float],
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    """
    Bootstrap test for whether method A > method B.
    Uses the per-fold accuracies to compute confidence interval on the difference.

    For more power, we also do a sample-level bootstrap when fold predictions
    are not available — here we use fold-level bootstrap (conservative but clean).
    """
    rng = np.random.RandomState(RANDOM_STATE)
    diffs = np.array(accs_a) - np.array(accs_b)
    observed_diff = float(np.mean(diffs))

    # Bootstrap the mean difference
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(diffs), size=len(diffs), replace=True)
        boot_diffs.append(float(np.mean(diffs[idx])))
    boot_diffs = np.array(boot_diffs)

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    # One-sided p-value: proportion of bootstrap samples where diff <= 0
    p_raw = float(np.mean(boot_diffs <= 0))
    # Cap at 1/n_bootstrap for reporting (can't resolve below this)
    p_value = max(p_raw, 1.0 / n_bootstrap)

    return {
        "observed_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value_one_sided": p_value,
        "p_display": f"p < {1.0/n_bootstrap:.4f}" if p_raw == 0 else f"p = {p_value:.4f}",
    }


def bootstrap_above_chance(
    hs_at_layer: np.ndarray,
    y: np.ndarray,
    text_embeddings: np.ndarray,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Bootstrap test: is residualized probe accuracy significantly above 50% (chance)?
    Resamples data with replacement, runs full nested CV each time.
    Limited to 1000 iterations for speed (each iteration runs 4-fold CV).
    """
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y)
    boot_accs = []

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        result = run_nested_residualized_probing(
            hs_at_layer[idx], y[idx], text_embeddings[idx], n_folds=N_FOLDS
        )
        boot_accs.append(result["mean"])

    boot_accs = np.array(boot_accs)
    ci_lower = float(np.percentile(boot_accs, 2.5))
    ci_upper = float(np.percentile(boot_accs, 97.5))
    p_value = float(np.mean(boot_accs <= 0.5))

    return {
        "mean": float(np.mean(boot_accs)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_above_chance": p_value,
    }


# ── Per-Model Analysis ───────────────────────────────────────────────────

def analyze_model(
    model_name: str,
    sbert_embeddings: np.ndarray | None = None,
    do_trajectory: bool = False,
    do_bootstrap_above_chance: bool = False,
) -> dict:
    """Run full residualized probe analysis for one model."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # Load data
    samples = load_hidden_states(model_name)
    if not samples:
        print(f"  ERROR: No samples found for {model_name}")
        return {}

    hs, y, questions, modalities = extract_arrays(samples)
    n, n_layers, hidden_dim = hs.shape
    print(f"  Samples: {n}, Layers: {n_layers}, Hidden dim: {hidden_dim}")
    print(f"  Misleading: {y.sum()}/{n} ({100*y.mean():.1f}%)")

    # Get peak layer
    peak_layer = PEAK_LAYERS.get(model_name)
    if peak_layer is None or peak_layer >= n_layers:
        # Find peak by running original probe at all layers
        print("  Finding peak layer via original probes...")
        best_acc, best_layer = 0, 0
        for l in range(n_layers):
            result = run_cv_probing(hs[:, l, :], y)
            if result["mean"] > best_acc:
                best_acc = result["mean"]
                best_layer = l
        peak_layer = best_layer
        print(f"  Peak layer found: {peak_layer} (acc={best_acc:.3f})")

    hs_peak = hs[:, peak_layer, :]
    print(f"  Using peak layer: {peak_layer}")

    # ── Compute text features ────────────────────────────────────────
    if sbert_embeddings is None:
        sbert_embeddings = compute_sbert_embeddings(questions)

    # ── Masks for modality-specific analysis ─────────────────────────
    # All standard samples as negatives + modality-specific misleading as positives
    # (1:2 ratio, matching linear_probing.py and paper §4.2)
    is_standard = y == 0
    vis_mask = is_standard | ((y == 1) & (modalities == "vision"))
    aud_mask = is_standard | ((y == 1) & (modalities == "audio"))

    results = {
        "model": model_name,
        "n_samples": n,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "peak_layer": peak_layer,
    }

    # ── 1. Original probe (sanity check) ─────────────────────────────
    print("\n  [1/5] Original probe (all, vision, audio)...")
    results["original_probe"] = {
        "all": run_cv_probing(hs_peak, y),
        "vision": run_cv_probing(hs_peak[vis_mask], y[vis_mask]),
        "audio": run_cv_probing(hs_peak[aud_mask], y[aud_mask]),
    }
    print(f"    All:    {results['original_probe']['all']['mean']:.3f} ± {results['original_probe']['all']['std']:.3f}")
    print(f"    Vision: {results['original_probe']['vision']['mean']:.3f} ± {results['original_probe']['vision']['std']:.3f}")
    print(f"    Audio:  {results['original_probe']['audio']['mean']:.3f} ± {results['original_probe']['audio']['std']:.3f}")

    # ── 2. TF-IDF text baseline (per-fold fit to avoid vocabulary leakage)
    print("\n  [2/5] TF-IDF text baseline (per-fold fit)...")
    vis_questions = [q for q, m in zip(questions, modalities) if m == "vision"]
    aud_questions = [q for q, m in zip(questions, modalities) if m == "audio"]
    results["tfidf_baseline"] = {
        "all": run_tfidf_cv_probing(questions, y),
        "vision": run_tfidf_cv_probing(vis_questions, y[vis_mask]),
        "audio": run_tfidf_cv_probing(aud_questions, y[aud_mask]),
    }
    print(f"    All:    {results['tfidf_baseline']['all']['mean']:.3f}")
    print(f"    Vision: {results['tfidf_baseline']['vision']['mean']:.3f}")
    print(f"    Audio:  {results['tfidf_baseline']['audio']['mean']:.3f}")

    # ── 3. Sentence-BERT text baseline ───────────────────────────────
    print("\n  [3/5] Sentence-BERT text baseline...")
    results["sbert_baseline"] = {
        "all": run_cv_probing(sbert_embeddings, y),
        "vision": run_cv_probing(sbert_embeddings[vis_mask], y[vis_mask]),
        "audio": run_cv_probing(sbert_embeddings[aud_mask], y[aud_mask]),
    }
    print(f"    All:    {results['sbert_baseline']['all']['mean']:.3f}")
    print(f"    Vision: {results['sbert_baseline']['vision']['mean']:.3f}")
    print(f"    Audio:  {results['sbert_baseline']['audio']['mean']:.3f}")

    # ── 4. Residualized probe ────────────────────────────────────────
    print("\n  [4/5] Residualized probe (nested CV)...")
    results["residualized_probe"] = {
        "all": run_nested_residualized_probing(hs_peak, y, sbert_embeddings),
        "vision": run_nested_residualized_probing(
            hs_peak[vis_mask], y[vis_mask], sbert_embeddings[vis_mask]
        ),
        "audio": run_nested_residualized_probing(
            hs_peak[aud_mask], y[aud_mask], sbert_embeddings[aud_mask]
        ),
    }
    print(f"    All:    {results['residualized_probe']['all']['mean']:.3f} ± {results['residualized_probe']['all']['std']:.3f}")
    print(f"    Vision: {results['residualized_probe']['vision']['mean']:.3f} ± {results['residualized_probe']['vision']['std']:.3f}")
    print(f"    Audio:  {results['residualized_probe']['audio']['mean']:.3f} ± {results['residualized_probe']['audio']['std']:.3f}")

    # ── 5. Bootstrap significance ────────────────────────────────────
    print("\n  [5/5] Bootstrap significance tests...")

    # Test: original probe > SBERT baseline (per-fold paired test)
    results["bootstrap"] = {}
    results["bootstrap"]["original_vs_sbert"] = bootstrap_paired_test(
        results["original_probe"]["all"]["per_fold"],
        results["sbert_baseline"]["all"]["per_fold"],
    )
    print(f"    Original vs SBERT: diff={results['bootstrap']['original_vs_sbert']['observed_diff']:.3f}, "
          f"p={results['bootstrap']['original_vs_sbert']['p_value_one_sided']:.4f}")

    # Test: residualized probe > chance (fold-level)
    res_folds = results["residualized_probe"]["all"]["per_fold"]
    chance_folds = [0.5] * len(res_folds)
    results["bootstrap"]["residualized_vs_chance"] = bootstrap_paired_test(
        res_folds, chance_folds
    )
    print(f"    Residualized vs chance: diff={results['bootstrap']['residualized_vs_chance']['observed_diff']:.3f}, "
          f"p={results['bootstrap']['residualized_vs_chance']['p_value_one_sided']:.4f}")

    # Test: residualized probe > SBERT baseline
    results["bootstrap"]["residualized_vs_sbert"] = bootstrap_paired_test(
        res_folds,
        results["sbert_baseline"]["all"]["per_fold"],
    )
    print(f"    Residualized vs SBERT: diff={results['bootstrap']['residualized_vs_sbert']['observed_diff']:.3f}, "
          f"p={results['bootstrap']['residualized_vs_sbert']['p_value_one_sided']:.4f}")

    # Full bootstrap for residualized above chance (expensive — optional)
    if do_bootstrap_above_chance:
        print("    Running full bootstrap (residualized > chance, 1000 iterations)...")
        results["bootstrap"]["residualized_full_bootstrap"] = bootstrap_above_chance(
            hs_peak, y, sbert_embeddings, n_bootstrap=1000
        )
        print(f"    Full bootstrap: {results['bootstrap']['residualized_full_bootstrap']['mean']:.3f} "
              f"[{results['bootstrap']['residualized_full_bootstrap']['ci_lower']:.3f}, "
              f"{results['bootstrap']['residualized_full_bootstrap']['ci_upper']:.3f}]")

    # ── Optional: layer trajectory ───────────────────────────────────
    if do_trajectory:
        print("\n  [Extra] Layer trajectory (original + residualized)...")
        traj_original = []
        traj_residualized = []
        for l in range(n_layers):
            orig = run_cv_probing(hs[:, l, :], y)
            resid = run_nested_residualized_probing(
                hs[:, l, :], y, sbert_embeddings
            )
            traj_original.append(orig["mean"])
            traj_residualized.append(resid["mean"])
            print(f"    Layer {l:2d}: original={orig['mean']:.3f}, residualized={resid['mean']:.3f}")
        results["layer_trajectory"] = {
            "original": traj_original,
            "residualized": traj_residualized,
            "sbert_baseline": results["sbert_baseline"]["all"]["mean"],
            "tfidf_baseline": results["tfidf_baseline"]["all"]["mean"],
        }

    return results


# ── Main ─────────────────────────────────────────────────────────────────

def print_summary_table(all_results: dict):
    """Print a summary table matching Table 6 format."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE — Residualized Probe Analysis")
    print("=" * 100)
    print(f"{'Model':<18} {'Orig(V)':<9} {'Orig(A)':<9} {'SBERT(V)':<9} {'SBERT(A)':<9} "
          f"{'Resid(V)':<9} {'Resid(A)':<9} {'Gap(V)':<8} {'Gap(A)':<8}")
    print("-" * 100)
    for model, r in all_results.items():
        if not r:
            continue
        ov = r["original_probe"]["vision"]["mean"]
        oa = r["original_probe"]["audio"]["mean"]
        sv = r["sbert_baseline"]["vision"]["mean"]
        sa = r["sbert_baseline"]["audio"]["mean"]
        rv = r["residualized_probe"]["vision"]["mean"]
        ra = r["residualized_probe"]["audio"]["mean"]
        gv = rv - sv  # residualized minus SBERT = genuine multimodal signal
        ga = ra - sa
        print(f"{model:<18} {ov:<9.1%} {oa:<9.1%} {sv:<9.1%} {sa:<9.1%} "
              f"{rv:<9.1%} {ra:<9.1%} {gv:+<8.1%} {ga:+<8.1%}")
    print("=" * 100)

    print("\nKey:")
    print("  Orig(V/A)  = Original hidden-state probe (should match paper Table 6)")
    print("  SBERT(V/A) = Sentence-BERT text-only baseline (stronger than TF-IDF)")
    print("  Resid(V/A) = Residualized probe (text-predictive component removed)")
    print("  Resid > 50% = genuine multimodal signal survives residualization")
    print("  Resid > SBERT = residualized hidden states still beat text features")


def main():
    parser = argparse.ArgumentParser(description="Residualized Probe Analysis")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(PAPER_MODELS),
        help="Comma-separated model names (default: all paper models)",
    )
    parser.add_argument(
        "--trajectory",
        action="store_true",
        help="Compute full layer trajectory (slower, ~2min per model)",
    )
    parser.add_argument(
        "--full-bootstrap",
        action="store_true",
        help="Run full bootstrap for residualized > chance (much slower)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(OUTPUT_DIR, "residualized_probing.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    print(f"Models to analyze: {models}")
    print(f"Trajectory: {args.trajectory}")
    print(f"Full bootstrap: {args.full_bootstrap}")
    print(f"Output: {args.output}")

    # Pre-compute sentence-BERT embeddings (shared across models with same questions)
    # Load one model to get question texts
    print("\nPre-computing text features from first model's questions...")
    first_samples = load_hidden_states(models[0])
    _, _, questions, _ = extract_arrays(first_samples)
    sbert_embeddings = compute_sbert_embeddings(questions)

    # Verify questions are the same across models (they should be — same 2000 samples)
    # We'll reuse sbert_embeddings but verify per model

    all_results = {}
    for model in models:
        # Load this model's samples to check question alignment
        samples = load_hidden_states(model)
        _, _, model_questions, _ = extract_arrays(samples)

        # Check if questions match (same ordering)
        if len(model_questions) == len(questions) and model_questions == questions:
            model_sbert = sbert_embeddings
        else:
            print(f"  Questions differ for {model} — recomputing embeddings")
            model_sbert = compute_sbert_embeddings(model_questions)

        result = analyze_model(
            model,
            sbert_embeddings=model_sbert,
            do_trajectory=args.trajectory,
            do_bootstrap_above_chance=args.full_bootstrap,
        )
        all_results[model] = result

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary
    print_summary_table(all_results)


if __name__ == "__main__":
    main()
