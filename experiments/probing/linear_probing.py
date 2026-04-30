#!/usr/bin/env python3
"""
Linear Probing Analysis for IMAVB paper (§4.2 — The Know-But-Don't-Act Gap).

Trains a logistic regression probe at each transformer layer to predict whether
a hidden state comes from a standard or misleading input. Uses 4-fold stratified
cross-validation over all 2,000 samples per model.

Also trains modality-specific probes (vision-only, audio-only) and computes a
TF-IDF text-only baseline per fold to establish the text-confound ceiling.

Results match Table 3 (tab:know-gap) in the paper.

Usage:
    python linear_probing.py --model qwen2_5_omni
    python linear_probing.py --model all
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config imports ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    HIDDEN_STATES_ROOT,
    MODELS,
    OUTPUT_DIR,
    PRETTY_NAMES,
    RANDOM_STATE,
    SPLITS,
)

# ── Constants ──────────────────────────────────────────────────────────────
N_FOLDS = 4
C = 1.0
SOLVER = "lbfgs"
MAX_ITER = 1000
TFIDF_MAX_FEATURES = 5000


# ── Data loading ───────────────────────────────────────────────────────────

def load_samples(model_name: str) -> list[dict]:
    """Load all .pt files for a model across all splits."""
    model_dir = Path(HIDDEN_STATES_ROOT) / model_name
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


def extract_arrays(
    samples: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Extract hidden states, binary labels, modality flags, and question texts.

    Returns:
        hs:           (N, n_layers, hidden_dim) float32
        y:            (N,) int  — 0=standard, 1=misleading
        modalities:   (N,) str array — 'vision' or 'audio'
        questions:    list of N question strings
    """
    hs_list = []
    y_list = []
    modality_list = []
    question_list = []
    video_id_list = []

    # Track per-sample layer counts for safety
    layer_counts = []
    for s in samples:
        hs_list.append(s["hidden_states"].float().numpy())
        layer_counts.append(s["hidden_states"].shape[0])
        y_list.append(int(s["metadata"].get("is_misleading", False)))
        split = s["metadata"].get("split", "")
        modality_list.append("vision" if "vision" in split else "audio")
        question_list.append(s["metadata"].get("question", ""))
        video_id_list.append(s["metadata"].get("video_id", ""))

    # Truncate all samples to the minimum layer count (handles variable-depth models)
    min_layers = min(layer_counts)
    hs_trunc = [h[:min_layers] for h in hs_list]

    hs = np.stack(hs_trunc, axis=0)  # (N, n_layers, hidden_dim)
    y = np.array(y_list, dtype=int)
    modalities = np.array(modality_list)
    video_ids = np.array(video_id_list)

    return hs, y, modalities, question_list, video_ids


# ── Probe helpers ──────────────────────────────────────────────────────────

def _run_cv_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """
    4-fold stratified group CV probe (grouped by video). StandardScaler fit on train per fold.

    Returns:
        mean, std, per_fold accuracies, peak layer index (caller resolves).
    """
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_accs = []
    for train_idx, test_idx in sgkf.split(X, y, groups=groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=C, solver=SOLVER, max_iter=MAX_ITER, random_state=RANDOM_STATE)
        clf.fit(X_tr, y[train_idx])
        fold_accs.append(float(clf.score(X_te, y[test_idx])))
    return {
        "mean": float(np.mean(fold_accs)),
        "std": float(np.std(fold_accs, ddof=1)),
        "per_fold": fold_accs,
    }


def run_layerwise_probe(hs: np.ndarray, y: np.ndarray, groups: np.ndarray) -> list[dict]:
    """Train probe at every layer. Returns list of result dicts (one per layer)."""
    n_layers = hs.shape[1]
    results = []
    for layer_idx in range(n_layers):
        result = _run_cv_probe(hs[:, layer_idx, :], y, groups)
        results.append(result)
    return results


def run_tfidf_baseline(questions: list[str], y: np.ndarray, groups: np.ndarray) -> dict:
    """
    TF-IDF text-only baseline. TfidfVectorizer fit per fold to avoid leakage.

    Per §4.2: TfidfVectorizer(max_features=5000), fit inside each fold.
    """
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    questions_arr = np.array(questions)
    fold_accs = []
    for train_idx, test_idx in sgkf.split(questions_arr, y, groups=groups):
        tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
        X_tr = tfidf.fit_transform(questions_arr[train_idx]).toarray()
        X_te = tfidf.transform(questions_arr[test_idx]).toarray()
        clf = LogisticRegression(C=C, solver=SOLVER, max_iter=MAX_ITER, random_state=RANDOM_STATE)
        clf.fit(X_tr, y[train_idx])
        fold_accs.append(float(clf.score(X_te, y[test_idx])))
    return {
        "mean": float(np.mean(fold_accs)),
        "std": float(np.std(fold_accs, ddof=1)),
        "per_fold": fold_accs,
    }


# ── Per-model analysis ─────────────────────────────────────────────────────

def analyze_model(model_name: str) -> dict:
    """
    Full linear probing analysis for one model.

    Probes trained:
      - Global probe: all standard (std_v + std_a) as negatives vs all misleading
        (mis_v + mis_a) as positives. Probed at every layer → peak layer l*.
      - Vision probe: all standard as negatives vs mis_v as positives (1:2 ratio).
        Evaluated at l*.
      - Audio probe: all standard as negatives vs mis_a as positives (1:2 ratio).
        Evaluated at l*.
      - TF-IDF baseline (vision questions, 4-fold CV).
      - TF-IDF baseline (audio questions, 4-fold CV).
    """
    pretty = PRETTY_NAMES.get(model_name, model_name)
    print(f"\n{'='*60}")
    print(f"Model: {pretty} ({model_name})")
    print(f"{'='*60}")

    samples = load_samples(model_name)
    if not samples:
        print(f"  ERROR: no samples found — skipping")
        return {}

    hs, y, modalities, questions, video_ids = extract_arrays(samples)
    n, n_layers, hidden_dim = hs.shape
    print(f"  Loaded: {n} samples, {n_layers} layers, hidden_dim={hidden_dim}")
    print(f"  Misleading: {y.sum()}/{n}  Standard: {(1-y).sum()}/{n}")

    # ── 1. Global probe at every layer ────────────────────────────────
    print(f"\n  [1/4] Global layerwise probe ({N_FOLDS}-fold CV)...")
    global_layer_results = run_layerwise_probe(hs, y, video_ids)
    peak_layer = int(np.argmax([r["mean"] for r in global_layer_results]))
    peak_acc = global_layer_results[peak_layer]["mean"]
    peak_std = global_layer_results[peak_layer]["std"]
    print(f"  Peak layer: {peak_layer}  acc={peak_acc:.3f} ± {peak_std:.3f}")

    # ── 2. Modality-specific probes at peak layer ─────────────────────
    # Vision probe: all standard (1000 samples) vs mis_v (500 samples) → 1:2 ratio
    # Audio probe:  all standard (1000 samples) vs mis_a (500 samples) → 1:2 ratio
    std_mask = y == 0  # all standard samples
    vis_misleading_mask = (y == 1) & (modalities == "vision")
    aud_misleading_mask = (y == 1) & (modalities == "audio")

    vis_mask = std_mask | vis_misleading_mask
    aud_mask = std_mask | aud_misleading_mask

    hs_peak = hs[:, peak_layer, :]

    print(f"\n  [2/4] Vision-specific probe at layer {peak_layer}...")
    print(f"    Samples: std={std_mask.sum()}, mis_v={vis_misleading_mask.sum()}")
    vision_result = _run_cv_probe(hs_peak[vis_mask], y[vis_mask], video_ids[vis_mask])
    print(f"    acc={vision_result['mean']:.3f} ± {vision_result['std']:.3f}")

    print(f"\n  [3/4] Audio-specific probe at layer {peak_layer}...")
    print(f"    Samples: std={std_mask.sum()}, mis_a={aud_misleading_mask.sum()}")
    audio_result = _run_cv_probe(hs_peak[aud_mask], y[aud_mask], video_ids[aud_mask])
    print(f"    acc={audio_result['mean']:.3f} ± {audio_result['std']:.3f}")

    # ── 3. TF-IDF baselines ───────────────────────────────────────────
    print(f"\n  [4/4] TF-IDF text baselines ({N_FOLDS}-fold CV)...")

    vis_questions = [q for q, m in zip(questions, modalities) if m == "vision"]
    vis_y = y[modalities == "vision"]
    vis_video_ids = video_ids[modalities == "vision"]
    aud_questions = [q for q, m in zip(questions, modalities) if m == "audio"]
    aud_y = y[modalities == "audio"]
    aud_video_ids = video_ids[modalities == "audio"]

    tfidf_vision = run_tfidf_baseline(vis_questions, vis_y, vis_video_ids)
    tfidf_audio = run_tfidf_baseline(aud_questions, aud_y, aud_video_ids)
    print(f"    TF-IDF vision: {tfidf_vision['mean']:.3f} ± {tfidf_vision['std']:.3f}")
    print(f"    TF-IDF audio:  {tfidf_audio['mean']:.3f} ± {tfidf_audio['std']:.3f}")

    # ── Assemble result ───────────────────────────────────────────────
    result = {
        "model": model_name,
        "n_samples": n,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        # Global probe
        "global_probe_layers": [
            {"layer": i, "cv_acc": r["mean"], "cv_std": r["std"]}
            for i, r in enumerate(global_layer_results)
        ],
        "peak_layer": peak_layer,
        "peak_cv_acc": round(peak_acc, 4),
        "peak_cv_std": round(peak_std, 4),
        # Modality-specific probes (at peak layer)
        "vision_probe": {
            "layer": peak_layer,
            "cv_acc": round(vision_result["mean"], 4),
            "cv_std": round(vision_result["std"], 4),
            "per_fold": vision_result["per_fold"],
            "n_std": int(std_mask.sum()),
            "n_mis_v": int(vis_misleading_mask.sum()),
        },
        "audio_probe": {
            "layer": peak_layer,
            "cv_acc": round(audio_result["mean"], 4),
            "cv_std": round(audio_result["std"], 4),
            "per_fold": audio_result["per_fold"],
            "n_std": int(std_mask.sum()),
            "n_mis_a": int(aud_misleading_mask.sum()),
        },
        # TF-IDF baselines
        "tfidf_vision": {
            "cv_acc": round(tfidf_vision["mean"], 4),
            "cv_std": round(tfidf_vision["std"], 4),
            "per_fold": tfidf_vision["per_fold"],
        },
        "tfidf_audio": {
            "cv_acc": round(tfidf_audio["mean"], 4),
            "cv_std": round(tfidf_audio["std"], 4),
            "per_fold": tfidf_audio["per_fold"],
        },
    }
    return result


# ── Summary printing ───────────────────────────────────────────────────────

def print_summary(all_results: dict) -> None:
    """Print summary table matching Table 3 (tab:know-gap) format."""
    print("\n" + "=" * 95)
    print("LINEAR PROBING SUMMARY — Table 3 (tab:know-gap)")
    print("=" * 95)
    header = (
        f"{'Model':<22} {'Peak L':>7} {'HS Probe(V)':>12} {'HS Probe(A)':>12} "
        f"{'TF-IDF(V)':>10} {'TF-IDF(A)':>10}"
    )
    print(header)
    print("-" * 95)
    for model, r in all_results.items():
        if not r:
            continue
        name = PRETTY_NAMES.get(model, model)
        vp = r["vision_probe"]["cv_acc"] * 100
        vs = r["vision_probe"]["cv_std"] * 100
        ap = r["audio_probe"]["cv_acc"] * 100
        as_ = r["audio_probe"]["cv_std"] * 100
        tv = r["tfidf_vision"]["cv_acc"] * 100
        ta = r["tfidf_audio"]["cv_acc"] * 100
        print(
            f"{name:<22} {r['peak_layer']:>7} "
            f"{vp:>8.1f}±{vs:<3.1f} "
            f"{ap:>8.1f}±{as_:<3.1f} "
            f"{tv:>10.1f} {ta:>10.1f}"
        )
    print("=" * 95)

    # Means
    valid = [r for r in all_results.values() if r]
    if valid:
        mean_vp = np.mean([r["vision_probe"]["cv_acc"] for r in valid]) * 100
        mean_ap = np.mean([r["audio_probe"]["cv_acc"] for r in valid]) * 100
        mean_tv = np.mean([r["tfidf_vision"]["cv_acc"] for r in valid]) * 100
        mean_ta = np.mean([r["tfidf_audio"]["cv_acc"] for r in valid]) * 100
        print(
            f"{'MEAN':<22} {'':>7} "
            f"{mean_vp:>12.1f} {mean_ap:>12.1f} "
            f"{mean_tv:>10.1f} {mean_ta:>10.1f}"
        )
    print("=" * 95)
    print("\nNote: HS Probe = hidden-state probe at peak layer l* (modality-specific).")
    print("      TF-IDF baseline should be ~73.4% (vision) / ~71.4% (audio) per paper.")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Linear probing analysis for IMAVB §4.2 (Table 3)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help=(
            "Model name (e.g. qwen2_5_omni) or 'all' to run all models. "
            f"Available: {', '.join(MODELS)}"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(OUTPUT_DIR, "linear_probing.json"),
        help="Path to output JSON file (default: OUTPUT_DIR/linear_probing.json)",
    )
    args = parser.parse_args()

    if args.model == "all":
        models = MODELS
    elif args.model in MODELS:
        models = [args.model]
    else:
        parser.error(
            f"Unknown model '{args.model}'. Available: {', '.join(MODELS)}"
        )

    print(f"Linear Probing — {N_FOLDS}-fold stratified group CV (grouped by video)")
    print(f"Models: {models}")
    print(f"Output: {args.output}")

    all_results: dict = {}

    # Load existing results if output file already exists (allows resuming)
    if os.path.exists(args.output):
        with open(args.output) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results from {args.output}")

    for model in models:
        if model in all_results and all_results[model]:
            print(f"\nSkipping {model} (already computed)")
            continue
        result = analyze_model(model)
        all_results[model] = result

        # Save after each model
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved intermediate results to {args.output}")

    print_summary(all_results)
    print(f"\nDone. Full results saved to {args.output}")


if __name__ == "__main__":
    main()
