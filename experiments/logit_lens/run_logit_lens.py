#!/usr/bin/env python3
"""
Logit Lens analysis (§4.3): project each layer's hidden state through the final
LM head (with RMSNorm) to obtain P(correct answer token) at each layer.

Equations from the paper:
    z_l = W_unembed * h_l                          (Eq. 3)
    P_l(correct) = softmax(z_l)[correct_token]     (Eq. 4)

W_unembed is the LM head weight matrix. h_l is first passed through a final
RMSNorm (x / sqrt(mean(x^2) + eps) * weight, eps=1e-6) before projection.

Usage:
    # Step 1: extract LM head weights for your model
    python extract_lm_weights.py --model qwen2_5_omni

    # Step 2: run logit lens
    python run_logit_lens.py --model qwen2_5_omni
    python run_logit_lens.py --model qwen2_5_omni --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add experiments directory to path so config is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    HIDDEN_STATES_ROOT,
    LM_WEIGHTS_ROOT,
    MODELS,
    OUTPUT_DIR,
    PRETTY_NAMES,
    SPLITS,
)

# MCQ answer labels — answer tokens A through F
MCQ_LABELS = ["A", "B", "C", "D", "E", "F"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_hidden_states(model_dir: str) -> list[dict]:
    """Load all hidden-state .pt files for a model across all splits."""
    samples = []
    total_files = sum(
        len(list(Path(model_dir, s).glob("*.pt")))
        for s in SPLITS
        if Path(model_dir, s).exists()
    )
    logger.info(f"Loading {total_files} .pt files from {model_dir}")
    for split in SPLITS:
        split_dir = Path(model_dir, split)
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue
        for pt_file in sorted(split_dir.glob("*.pt")):
            data = torch.load(str(pt_file), map_location="cpu", weights_only=True)
            samples.append(data)
    return samples


def load_lm_head_weights(weights_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load norm_weights.pt and lm_head_weights.pt from weights_dir.

    Returns:
        norm_weight: (hidden_dim,) float32 — RMSNorm scale parameter.
        lm_weight:   (vocab_size, hidden_dim) float32 — unembedding matrix.
    """
    norm_path = os.path.join(weights_dir, "norm_weights.pt")
    lm_head_path = os.path.join(weights_dir, "lm_head_weights.pt")

    norm_sd = torch.load(norm_path, map_location="cpu", weights_only=True)
    norm_weight = norm_sd["weight"].float()
    logger.info(f"Loaded norm weights: shape={tuple(norm_weight.shape)} from {norm_path}")

    lm_sd = torch.load(lm_head_path, map_location="cpu", weights_only=True)
    lm_weight = lm_sd["weight"].float()
    logger.info(f"Loaded lm_head weights: shape={tuple(lm_weight.shape)} from {lm_head_path}")

    return norm_weight, lm_weight


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm: x / sqrt(mean(x^2) + eps) * weight."""
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return (x / rms) * weight


def build_projection_fn(
    norm_weight: torch.Tensor,
    lm_weight: torch.Tensor,
    device: str = "cpu",
) -> tuple:
    """Build a projection function h -> logits using RMSNorm + LM head.

    Args:
        norm_weight: (hidden_dim,) RMSNorm scale.
        lm_weight:   (vocab_size, hidden_dim) unembedding matrix.
        device:      Torch device string.

    Returns:
        (project_fn, vocab_size) where project_fn maps (N, hidden_dim) -> (N, vocab_size).
    """
    vocab_size = lm_weight.shape[0]
    hidden_dim = lm_weight.shape[1]
    logger.info(f"LM head: hidden_dim={hidden_dim}, vocab_size={vocab_size}")

    norm_w = norm_weight.to(device)
    lm_w = lm_weight.to(device)

    def project(hidden: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm then unembedding: (N, hidden_dim) -> (N, vocab_size)."""
        normed = rms_norm(hidden, norm_w)
        return normed @ lm_w.T

    return project, vocab_size


# ---------------------------------------------------------------------------
# Token ID mapping
# ---------------------------------------------------------------------------


def get_mcq_token_ids(vocab_size: int, tokenizer_name: str | None = None) -> dict[str, list[int]]:
    """Return candidate token IDs for each MCQ answer label (A-F).

    For Qwen2-family tokenizers (vocab_size 151670 or 152064), uses empirically
    identified IDs for A-F with various prefix contexts. For other architectures,
    loads the actual tokenizer to resolve correct token IDs.

    Args:
        vocab_size: Model vocabulary size.
        tokenizer_name: HuggingFace tokenizer name/path. If provided and model is
            not Qwen2-family, used to resolve token IDs via the actual tokenizer.
    """
    if vocab_size in (151670, 152064):
        # Qwen2-family: IDs empirically identified for A-F with various prefixes
        token_ids: dict[str, list[int]] = {
            "A": [32, 330, 362, 317, 65],
            "B": [33, 347, 363, 318, 66],
            "C": [34, 356, 364, 327, 67],
            "D": [35, 360, 365, 334, 68],
            "E": [36, 353, 366, 337, 69],
            "F": [37, 370, 367, 344, 70],
        }
    elif tokenizer_name:
        # Resolve via actual tokenizer for non-Qwen models
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        token_ids = {}
        for label in MCQ_LABELS:
            ids = set()
            # Try multiple encoding contexts to capture all variants
            for prefix in ["", " ", "\n"]:
                encoded = tokenizer.encode(f"{prefix}{label}", add_special_tokens=False)
                if encoded:
                    ids.add(encoded[-1])  # last token is the label
            # Also try the bare label
            bare = tokenizer.encode(label, add_special_tokens=False)
            if bare:
                ids.add(bare[-1])
            token_ids[label] = sorted(ids)
        logger.info(f"Tokenizer-resolved IDs: { {k: v for k, v in token_ids.items()} }")
    else:
        # Last resort: plain ASCII (A=65, B=66, …, F=70)
        logger.warning("No tokenizer provided for non-Qwen model — using ASCII fallback. "
                       "Pass --tokenizer to get accurate token IDs.")
        token_ids = {label: [ord(label)] for label in MCQ_LABELS}

    # Filter IDs that fall within the actual vocabulary
    return {k: [tid for tid in v if tid < vocab_size] for k, v in token_ids.items()}


# ---------------------------------------------------------------------------
# Per-layer P(correct) computation
# ---------------------------------------------------------------------------


def compute_p_correct_per_layer(
    samples: list[dict],
    project_fn,
    mcq_token_ids: dict[str, list[int]],
    split_name: str,
    device: str = "cpu",
) -> np.ndarray | None:
    """Compute P(correct answer token) at each layer for all samples in a split.

    Args:
        samples:       All loaded samples (metadata.split field used to filter).
        project_fn:    Function (N, hidden_dim) -> (N, vocab_size).
        mcq_token_ids: Mapping from label to candidate token IDs.
        split_name:    Which split to process.
        device:        Torch device.

    Returns:
        Array of shape (N, n_layers) with P(correct) values, or None if split empty.
    """
    split_samples = [s for s in samples if s["metadata"]["split"] == split_name]
    if not split_samples:
        return None

    n = len(split_samples)
    n_layers = split_samples[0]["hidden_states"].shape[0]

    # Stack: (N, n_layers, hidden_dim)
    all_hs = torch.stack([s["hidden_states"].float() for s in split_samples])

    # Correct answer label per sample
    correct_ids_list = [
        mcq_token_ids.get(s["metadata"]["correct_answer"], [])
        for s in split_samples
    ]

    p_correct_all = np.zeros((n, n_layers), dtype=np.float32)

    with torch.no_grad():
        for layer_idx in range(n_layers):
            h_batch = all_hs[:, layer_idx, :].to(device)  # (N, hidden_dim)
            logits = project_fn(h_batch)                   # (N, vocab_size)
            probs = F.softmax(logits, dim=-1).cpu()        # (N, vocab_size)

            for i, correct_ids in enumerate(correct_ids_list):
                if correct_ids:
                    p_correct_all[i, layer_idx] = sum(
                        probs[i, tid].item()
                        for tid in correct_ids
                        if tid < probs.shape[1]
                    )

    return p_correct_all


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_logit_lens_split(
    mean_p: np.ndarray,
    std_p: np.ndarray,
    model_name: str,
    split_name: str,
    out_path: str,
) -> None:
    """Plot mean ± std P(correct) across layers for a single split."""
    x = list(range(len(mean_p)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, mean_p, marker="o", linewidth=2, markersize=4)
    ax.fill_between(x, mean_p - std_p, mean_p + std_p, alpha=0.2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean P(correct)")
    pretty = PRETTY_NAMES.get(model_name, model_name)
    ax.set_title(f"{pretty} — Logit Lens: {split_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    logger.info(f"Saved plot: {out_path}")


def plot_logit_lens_compare(
    results_by_split: dict,
    model_name: str,
    modality: str,
    out_path: str,
) -> None:
    """Plot standard vs. misleading P(correct) curves for one modality."""
    std_key = f"standard_{modality}"
    mis_key = f"misleading_{modality}"
    if std_key not in results_by_split or mis_key not in results_by_split:
        return

    std_mean = results_by_split[std_key]["mean_p_correct"]
    mis_mean = results_by_split[mis_key]["mean_p_correct"]
    x = list(range(len(std_mean)))
    pretty = PRETTY_NAMES.get(model_name, model_name)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, std_mean, marker="o", linewidth=2, markersize=4, label=f"standard_{modality}")
    ax.plot(
        x, mis_mean, marker="s", linewidth=2, markersize=4,
        linestyle="--", label=f"misleading_{modality}",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean P(correct)")
    ax.set_title(f"{pretty} — Logit Lens: {modality} comparison")
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    logger.info(f"Saved comparison plot: {out_path}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_logit_lens_for_model(
    model_name: str,
    samples: list[dict],
    weights_dir: str,
    output_dir: str,
    device: str = "cpu",
    tokenizer_name: str | None = None,
) -> dict:
    """Run logit lens analysis for one model.

    Args:
        model_name:  Model key (matches MODELS in config.py).
        samples:     All hidden-state samples for this model.
        weights_dir: Directory containing norm_weights.pt and lm_head_weights.pt.
        output_dir:  Where to write JSON results and PNG plots.
        device:      Torch device.

    Returns:
        Result dict with per-split peak P(correct) and full layer curves.
    """
    logger.info(f"{model_name}: {len(samples)} samples (device={device})")

    norm_weight, lm_weight = load_lm_head_weights(weights_dir)
    project_fn, vocab_size = build_projection_fn(norm_weight, lm_weight, device=device)
    mcq_token_ids = get_mcq_token_ids(vocab_size, tokenizer_name=tokenizer_name)
    logger.info(f"vocab_size={vocab_size}, MCQ token IDs: {mcq_token_ids}")

    split_counts = {
        split: sum(1 for s in samples if s["metadata"]["split"] == split)
        for split in SPLITS
    }
    logger.info(f"Split counts: {split_counts}")

    results_by_split: dict[str, dict] = {}
    for split in SPLITS:
        count = split_counts[split]
        logger.info(f"  Processing {split} ({count} samples)...")
        p_correct = compute_p_correct_per_layer(
            samples, project_fn, mcq_token_ids, split, device=device
        )
        if p_correct is None or p_correct.shape[0] == 0:
            logger.warning(f"  {split}: no samples, skipping")
            continue

        mean_p = p_correct.mean(axis=0)
        std_p = p_correct.std(axis=0)
        best_layer = int(np.argmax(mean_p))
        peak_p = float(mean_p[best_layer])

        logger.info(f"  {split}: best_layer={best_layer} peak_P={peak_p:.4f}")

        results_by_split[split] = {
            "n_samples": count,
            "best_layer": best_layer,
            "peak_p_correct": round(peak_p, 4),
            "mean_p_correct": [round(float(v), 4) for v in mean_p],
            "std_p_correct": [round(float(v), 4) for v in std_p],
        }

        plot_logit_lens_split(
            mean_p, std_p, model_name, split,
            os.path.join(output_dir, f"{model_name}_logit_lens_{split}.png"),
        )

    for modality in ("vision", "audio"):
        plot_logit_lens_compare(
            results_by_split, model_name, modality,
            os.path.join(output_dir, f"{model_name}_logit_lens_compare_{modality}.png"),
        )

    result = {
        "model": model_name,
        "n_samples_total": len(samples),
        "splits": results_by_split,
    }
    out_path = os.path.join(output_dir, f"{model_name}_logit_lens_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved results: {out_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logit lens analysis: project hidden states through LM head at each layer."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=MODELS,
        help="Model name (must match a key in config.MODELS)",
    )
    parser.add_argument(
        "--hidden_states_root",
        default=HIDDEN_STATES_ROOT,
        help="Root directory containing per-model hidden-state .pt files",
    )
    parser.add_argument(
        "--lm_weights_root",
        default=LM_WEIGHTS_ROOT,
        help="Root directory containing per-model norm_weights.pt and lm_head_weights.pt",
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help="Output directory for JSON results and PNG plots",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HuggingFace tokenizer name/path for resolving MCQ token IDs "
             "(required for non-Qwen models; auto-detected for Qwen2 family)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    hidden_model_dir = os.path.join(args.hidden_states_root, args.model)
    weights_model_dir = os.path.join(args.lm_weights_root, args.model)

    samples = load_hidden_states(hidden_model_dir)
    if not samples:
        logger.error(f"No samples found in {hidden_model_dir}")
        raise SystemExit(1)

    result = run_logit_lens_for_model(
        args.model,
        samples,
        weights_model_dir,
        args.output_dir,
        device=args.device,
        tokenizer_name=args.tokenizer,
    )

    logger.info("=== Summary ===")
    for split, sres in result["splits"].items():
        logger.info(
            f"  {split}: best_layer={sres['best_layer']} "
            f"peak_P={sres['peak_p_correct']:.4f} n={sres['n_samples']}"
        )


if __name__ == "__main__":
    main()
