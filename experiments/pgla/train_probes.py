#!/usr/bin/env python3
"""Train MLP and linear probes for PGLA (§5 + App F of the paper).

Workflow:
    1. Load hidden states at the peak probe layer l* for a given model.
    2. Split 25% train / 75% eval (stratified, fixed seed).
    3. Train a 2-layer MLP probe (binary: standard vs. misleading).
    4. Train a logistic regression linear probe (binary).
    5. Compute beta (E–F positional bias from standard training samples).
    6. Save per-sample probe scores + metadata to <OUTPUT_DIR>/pgla/<model>/probe_outputs.pt

Usage:
    python train_probes.py --model ola
    python train_probes.py --model qwen2_5_omni --layer 17
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Resolve config relative to this file's location
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    HIDDEN_DIMS,
    HIDDEN_STATES_ROOT,
    MODELS,
    OUTPUT_DIR,
    PGLA_PEAK_LAYERS,
    RANDOM_STATE,
    SPLITS,
)

MISLEADING_CORRECT = {"misleading_vision": "E", "misleading_audio": "F"}
TRAIN_FRAC = 0.25  # 500 train / 1500 eval out of 2000 total


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class MLPProbe(nn.Module):
    """2-layer MLP probe: Input -> Linear(hidden_dim, 256) -> ReLU -> Linear(256, n_classes)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hidden_states_at_layer(model: str, layer: int) -> list[dict]:
    """Load all samples for a model, returning hidden state at the given layer.

    Supports two directory formats:
      (a) Per-sample .pt files: HIDDEN_STATES_ROOT/<model>/<split>/<video_id>.pt
          Each contains {"hidden_states": Tensor(n_layers, hidden_dim), "metadata": {...}}
      (b) Aggregated file: HIDDEN_STATES_ROOT/<model>/<split>/hidden_states.pt
          Contains a list of dicts with "hidden_states" and optionally "choice_logits"

    Returns list of dicts with keys: video_id, split, correct_answer, choice_logits, hs, is_misleading
    """
    hs_root = Path(HIDDEN_STATES_ROOT)
    samples = []
    for split in SPLITS:
        split_dir = hs_root / model / split
        if not split_dir.exists():
            print(f"  [warn] missing {split_dir}, skipping split {split}")
            continue

        # Try aggregated file first, then per-sample .pt files
        agg_path = split_dir / "hidden_states.pt"
        if agg_path.exists():
            raw = torch.load(agg_path, map_location="cpu", weights_only=False)
            items = raw if isinstance(raw, list) else [raw]
        else:
            items = []
            for pt_file in sorted(split_dir.glob("*.pt")):
                data = torch.load(str(pt_file), map_location="cpu", weights_only=False)
                # Normalize per-sample format to match aggregated format
                meta = data.get("metadata", {})
                items.append({
                    "hidden_states": data["hidden_states"],
                    "video_id": meta.get("video_id", pt_file.stem),
                    "correct_answer": meta.get("correct_answer", "A"),
                    "choice_logits": meta.get("choice_logits", {}),
                })
            if not items:
                print(f"  [warn] no .pt files in {split_dir}, skipping")
                continue

        for s in items:
            hs_tensor = s["hidden_states"]  # (n_layers, hidden_dim)
            hs_vec = hs_tensor[layer].float().numpy()

            logits = s.get("choice_logits", {})
            if isinstance(logits, torch.Tensor):
                choices = list("ABCDEF")
                logits = {c: float(logits[i]) for i, c in enumerate(choices) if i < len(logits)}

            samples.append({
                "video_id": s.get("video_id", ""),
                "split": split,
                "correct_answer": s.get("correct_answer", "A"),
                "choice_logits": logits,
                "hs": hs_vec,
                "is_misleading": 1 if "misleading" in split else 0,
            })
    return samples


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_probes(
    model: str,
    layer: int | None = None,
    hidden_dim: int = 256,
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int = RANDOM_STATE,
) -> dict:
    """Train MLP + linear probes; return per-sample scores + metadata.

    Args:
        model: Model name from config.MODELS.
        layer: Layer index. Defaults to PGLA_PEAK_LAYERS[model].
        hidden_dim: MLP hidden dimension (paper: 256).
        epochs: Training epochs (paper: 100).
        lr: Adam learning rate (paper: 1e-3).
        seed: Random seed.

    Returns dict with keys:
        samples: list of dicts (video_id, split, correct_answer, choice_logits,
                  partition, p_mis_mlp, p_mis_linear, ef_bias)
        mlp_train_acc, mlp_eval_acc: float
        linear_train_acc, linear_eval_acc: float
        layer: int
        n_train, n_eval, n_total: int
        ef_bias: float
    """
    if layer is None:
        layer = PGLA_PEAK_LAYERS[model]

    print(f"[{model}] Loading hidden states at layer {layer}...")
    all_samples = load_hidden_states_at_layer(model, layer)
    if not all_samples:
        raise RuntimeError(f"No hidden states found for model '{model}'")

    n_total = len(all_samples)
    print(f"[{model}] Loaded {n_total} samples across {len(SPLITS)} splits")

    # Build arrays
    X = np.array([s["hs"] for s in all_samples])          # (N, hidden_dim)
    y = np.array([s["is_misleading"] for s in all_samples])  # (N,)

    # Stratified 25/75 split (fixed seed for reproducibility)
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=1.0 - TRAIN_FRAC, random_state=seed
    )
    train_idx, eval_idx = next(sss.split(X, y))
    train_set = set(train_idx.tolist())

    partitions = ["train" if i in train_set else "eval" for i in range(n_total)]
    train_mask = np.array([p == "train" for p in partitions])
    eval_mask = ~train_mask

    X_train, X_eval = X[train_mask], X[eval_mask]
    y_train, y_eval = y[train_mask], y[eval_mask]
    print(f"[{model}] Split: {train_mask.sum()} train / {eval_mask.sum()} eval")

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_eval_s = scaler.transform(X_eval)
    X_all_s = scaler.transform(X)

    # ── Compute beta: mean(E - F) on standard training samples ───────────────
    ef_diffs = []
    for i in train_idx:
        s = all_samples[i]
        if "standard" in s["split"]:
            logits = s["choice_logits"]
            ef_diffs.append(logits.get("E", 0.0) - logits.get("F", 0.0))
    ef_bias = float(np.mean(ef_diffs)) if ef_diffs else 0.0
    print(f"[{model}] E–F positional bias (beta) = {ef_bias:.4f}")

    # ── Train binary MLP probe ────────────────────────────────────────────────
    torch.manual_seed(seed)
    input_dim = X.shape[1]
    mlp = MLPProbe(input_dim, hidden_dim, n_classes=2)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    mlp.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logit_out = mlp(X_t)
        loss = loss_fn(logit_out, y_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"[{model}] MLP epoch {epoch + 1}/{epochs}  loss={loss.item():.4f}")

    mlp.eval()
    with torch.no_grad():
        train_preds = mlp(X_t).argmax(dim=1).numpy()
        mlp_train_acc = float(np.mean(train_preds == y_train))

        eval_preds = mlp(torch.tensor(X_eval_s, dtype=torch.float32)).argmax(dim=1).numpy()
        mlp_eval_acc = float(np.mean(eval_preds == y_eval))

        all_probs = torch.softmax(
            mlp(torch.tensor(X_all_s, dtype=torch.float32)), dim=1
        ).numpy()  # (N, 2)  col 1 = P(misleading)

    print(f"[{model}] MLP train_acc={mlp_train_acc:.3f}  eval_acc={mlp_eval_acc:.3f}")

    # ── Train binary linear (logistic regression) probe ──────────────────────
    linear = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=seed)
    linear.fit(X_train_s, y_train)
    linear_train_acc = float(linear.score(X_train_s, y_train))
    linear_eval_acc = float(linear.score(X_eval_s, y_eval))
    linear_probs = linear.predict_proba(X_all_s)  # (N, 2)  col 1 = P(misleading)
    print(f"[{model}] LR  train_acc={linear_train_acc:.3f}  eval_acc={linear_eval_acc:.3f}")

    # ── Attach scores to sample dicts ─────────────────────────────────────────
    output_samples = []
    for i, s in enumerate(all_samples):
        output_samples.append({
            "video_id": s["video_id"],
            "split": s["split"],
            "correct_answer": s["correct_answer"],
            "choice_logits": s["choice_logits"],
            "partition": partitions[i],
            "p_mis_mlp": float(all_probs[i, 1]),
            "p_mis_linear": float(linear_probs[i, 1]),
            "ef_bias": ef_bias,
        })

    return {
        "samples": output_samples,
        "mlp_train_acc": mlp_train_acc,
        "mlp_eval_acc": mlp_eval_acc,
        "linear_train_acc": linear_train_acc,
        "linear_eval_acc": linear_eval_acc,
        "layer": layer,
        "n_train": int(train_mask.sum()),
        "n_eval": int(eval_mask.sum()),
        "n_total": n_total,
        "ef_bias": ef_bias,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PGLA probes for a given model (§5 + App F)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODELS,
        help="Model name (e.g. ola, qwen2_5_omni)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index (default: PGLA_PEAK_LAYERS[model] from config.py)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="MLP hidden dimension (paper: 256)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (paper: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate (paper: 1e-3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help=f"Random seed (default: {RANDOM_STATE})",
    )
    args = parser.parse_args()

    results = train_probes(
        model=args.model,
        layer=args.layer,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )

    # Save outputs
    out_dir = Path(OUTPUT_DIR) / "pgla" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "probe_outputs.pt"
    torch.save(results, out_path)
    print(f"[{args.model}] Saved probe outputs -> {out_path}")

    # Summary
    print(f"\n=== Probe summary for {args.model} ===")
    print(f"  Layer      : {results['layer']}")
    print(f"  Train / Eval: {results['n_train']} / {results['n_eval']}")
    print(f"  MLP  acc   : train={results['mlp_train_acc']:.3f}  eval={results['mlp_eval_acc']:.3f}")
    print(f"  LR   acc   : train={results['linear_train_acc']:.3f}  eval={results['linear_eval_acc']:.3f}")
    print(f"  Beta (E-F) : {results['ef_bias']:.4f}")


if __name__ == "__main__":
    main()
