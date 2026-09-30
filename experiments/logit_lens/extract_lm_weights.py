#!/usr/bin/env python3
"""
Extract and save LM head weights needed for logit lens (§4.3).

Saves two files to <lm_weights_root>/<model>/:
    norm_weights.pt    — final RMSNorm scale parameter {"weight": tensor(hidden_dim,)}
    lm_head_weights.pt — unembedding matrix            {"weight": tensor(vocab_size, hidden_dim)}

These files are required inputs for run_logit_lens.py.

Usage:
    python extract_lm_weights.py --model qwen2_5_omni --pretrained Qwen/Qwen2.5-Omni-7B
    python extract_lm_weights.py --model ola --pretrained /path/to/ola/checkpoint
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from loguru import logger

# Add experiments directory to path so config is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LM_WEIGHTS_ROOT, MODELS


# ---------------------------------------------------------------------------
# Architecture-specific accessor functions
# ---------------------------------------------------------------------------

# Each function receives the loaded model and returns (norm_module, lm_head_module).
# Add entries here for new architectures.
_ARCH_ACCESSORS = {
    # Qwen2.5-Omni and Qwen3-Omni: language model lives under model.thinker
    "qwen2_5_omni": lambda m: (m.thinker.model.norm, m.thinker.lm_head),
    "qwen3_omni":   lambda m: (m.thinker.model.norm, m.thinker.lm_head),
    # Other models that follow a standard LM structure
    "baichuan_omni": lambda m: (m.model.norm, m.lm_head),
    "minicpm_o":     lambda m: (m.llm.model.norm, m.llm.lm_head),
    "ola":           lambda m: (m.model.norm, m.lm_head),
    "omnivinci":     lambda m: (m.model.norm, m.lm_head),
    "uni_moe_2_omni": lambda m: (m.model.norm, m.lm_head),
    "video_salmonn_2": lambda m: (m.llama_model.model.norm, m.llama_model.lm_head),
}


def get_norm_and_lm_head(model, model_name: str) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Return (norm_module, lm_head_module) for the given model.

    Falls back to common attribute paths if no specific accessor is registered.
    Raises AttributeError with a helpful message if extraction fails.
    """
    if model_name in _ARCH_ACCESSORS:
        try:
            return _ARCH_ACCESSORS[model_name](model)
        except AttributeError as exc:
            logger.warning(
                f"Registered accessor for {model_name} failed: {exc}. "
                "Trying generic fallback paths."
            )

    # Generic fallback: try common attribute layouts
    for norm_attr, lm_attr in [
        ("model.norm", "lm_head"),
        ("thinker.model.norm", "thinker.lm_head"),
        ("llm.model.norm", "llm.lm_head"),
    ]:
        try:
            norm = _getattr_nested(model, norm_attr)
            lm_head = _getattr_nested(model, lm_attr)
            logger.info(f"Found norm at '{norm_attr}', lm_head at '{lm_attr}'")
            return norm, lm_head
        except AttributeError:
            continue

    raise AttributeError(
        f"Could not locate norm and lm_head for model '{model_name}'. "
        f"Add an entry to _ARCH_ACCESSORS in extract_lm_weights.py."
    )


def _getattr_nested(obj, dotted_path: str):
    """Traverse a dotted attribute path on an object."""
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# Extraction and saving
# ---------------------------------------------------------------------------


def extract_and_save(
    model_name: str,
    pretrained: str,
    output_dir: str,
    dtype: str = "float32",
) -> None:
    """Load model, extract LM head weights, and save to output_dir.

    Args:
        model_name:  Model key (must be in config.MODELS).
        pretrained:  HuggingFace model ID or local checkpoint path.
        output_dir:  Directory where norm_weights.pt and lm_head_weights.pt are written.
        dtype:       Cast dtype before saving ("float32" or "float16").
    """
    os.makedirs(output_dir, exist_ok=True)
    norm_out = os.path.join(output_dir, "norm_weights.pt")
    lm_out = os.path.join(output_dir, "lm_head_weights.pt")

    if os.path.exists(norm_out) and os.path.exists(lm_out):
        logger.info(f"Weights already exist in {output_dir}. Pass --force to re-extract.")
        return

    logger.info(f"Loading model '{pretrained}' for weight extraction...")
    from transformers import AutoModel, AutoModelForCausalLM  # type: ignore

    # Per-model loader: some architectures need AutoModel or custom classes
    _MODEL_LOADERS = {
        "qwen2_5_omni": AutoModel,       # ConditionalGeneration wrapper
        "qwen3_omni": AutoModel,          # MoE ConditionalGeneration wrapper
        "minicpm_o": AutoModel,           # Needs AutoModel, not CausalLM
        "uni_moe_2_omni": AutoModel,      # Custom MoE class
        "video_salmonn_2": AutoModel,     # Custom class with LoRA
    }
    loader_cls = _MODEL_LOADERS.get(model_name, AutoModelForCausalLM)
    logger.info(f"Using {loader_cls.__name__} for {model_name}")

    model = loader_cls.from_pretrained(
        pretrained,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    norm_module, lm_head_module = get_norm_and_lm_head(model, model_name)
    logger.info(f"norm: {norm_module.__class__.__name__}, lm_head: {lm_head_module.__class__.__name__}")

    cast = torch.float32 if dtype == "float32" else torch.float16

    norm_weight = norm_module.weight.detach().to(cast)
    lm_weight = lm_head_module.weight.detach().to(cast)

    logger.info(f"norm_weight shape: {tuple(norm_weight.shape)}")
    logger.info(f"lm_head weight shape: {tuple(lm_weight.shape)}")

    torch.save({"weight": norm_weight}, norm_out)
    torch.save({"weight": lm_weight}, lm_out)
    logger.info(f"Saved norm_weights.pt -> {norm_out}")
    logger.info(f"Saved lm_head_weights.pt -> {lm_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract LM head weights for logit lens (§4.3)."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=MODELS,
        help="Model key (must match config.MODELS)",
    )
    parser.add_argument(
        "--pretrained",
        required=True,
        help="HuggingFace model ID or local path to model checkpoint",
    )
    parser.add_argument(
        "--lm_weights_root",
        default=LM_WEIGHTS_ROOT,
        help="Root directory; weights saved to <lm_weights_root>/<model>/",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float16"),
        help="Save dtype for extracted weights (default: float32)",
    )
    args = parser.parse_args()

    output_dir = os.path.join(args.lm_weights_root, args.model)
    extract_and_save(args.model, args.pretrained, output_dir, dtype=args.dtype)
    logger.info("Done.")


if __name__ == "__main__":
    main()
