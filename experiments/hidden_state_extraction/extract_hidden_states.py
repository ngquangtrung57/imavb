"""Hidden state extraction for the IMAVB benchmark (paper release).

Loads a model, runs a forward pass on each benchmark sample, captures
last-token hidden states from every decoder layer via forward hooks
(register_forward_hook), and saves .pt files.

Paper reference: §4.3 + Appendix J (app:eval-impl)
  "Hidden State Extraction" — forward hooks on decoder layers, per-sample
  tensor of shape (num_layers, hidden_dim), 50 uniformly sampled frames,
  full-length audio resampled to 16 kHz mono.

Output format per file:
  {
    "hidden_states": torch.Tensor(num_layers, hidden_dim),
    "metadata": {
      "video_id": str,
      "split": str,
      "model": str,
      "correct_answer": str,
      "is_misleading": bool,
      "should_reject": bool,
      "question": str,
    }
  }

Usage:
  python extract_hidden_states.py --model qwen2_5_omni
  python extract_hidden_states.py --model baichuan_omni --limit 500
  python extract_hidden_states.py --model ola --splits standard_vision misleading_vision
"""

from __future__ import annotations

import argparse
import glob as _glob
import os
import sys
import traceback
from pathlib import Path
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger

# ---------------------------------------------------------------------------
# Shared config (paths, model list, splits)
# ---------------------------------------------------------------------------

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from config import (  # noqa: E402
    DATASET_NAME,
    HIDDEN_STATES_ROOT,
    MODELS,
    SPLITS,
)
from model_adapters import ADAPTER_REGISTRY, MODEL_DEFAULTS, get_adapter  # noqa: E402

# ---------------------------------------------------------------------------
# Video cache
# ---------------------------------------------------------------------------

VIDEO_CACHE_DIR = os.getenv(
    "IMAVB_VIDEO_CACHE",
    "<SET_PATH>/cache/huggingface/imavb_bench_videos",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_video_path(sample: dict) -> str:
    """Return local video path, falling back to HuggingFace URL."""
    video_id = sample["video_id"]
    if VIDEO_CACHE_DIR:
        exact = os.path.join(VIDEO_CACHE_DIR, f"{video_id}.mp4")
        if os.path.exists(exact):
            return exact
        matches = _glob.glob(
            os.path.join(VIDEO_CACHE_DIR, "**", f"{video_id}*.mp4"), recursive=True
        )
        if matches:
            return matches[0]
    video = sample.get("video")
    if video:
        if isinstance(video, dict) and "path" in video:
            path = video["path"]
            if os.path.exists(path):
                return path
        else:
            path = str(video)
            if os.path.exists(path):
                return path
    return (
        f"https://huggingface.co/datasets/{DATASET_NAME}"
        f"/resolve/main/videos/{video_id}.mp4"
    )


def save_lm_head_weights(adapter, output_dir: str) -> None:
    """Save final norm + lm_head weights for logit lens analysis.

    These are needed for logit lens post-hoc analysis (Appendix J).
    Saved once per model run to {output_dir}/norm_weights.pt and
    {output_dir}/lm_head_weights.pt.
    """
    norm_path = os.path.join(output_dir, "norm_weights.pt")
    lm_head_path = os.path.join(output_dir, "lm_head_weights.pt")

    if os.path.exists(norm_path) and os.path.exists(lm_head_path):
        logger.info("LM-head weights already saved, skipping.")
        return

    model_name = adapter.model_name
    model = adapter.model

    try:
        if model_name == "ola":
            torch.save(model.model.norm.state_dict(), norm_path)
            torch.save(model.lm_head.state_dict(), lm_head_path)
        elif model_name == "omnivinci":
            llm = getattr(model, "llm", model)
            norm = llm.model.norm if hasattr(llm, "model") else llm.norm
            torch.save(norm.state_dict(), norm_path)
            torch.save(llm.lm_head.state_dict(), lm_head_path)
        elif model_name in ("qwen2_5_omni", "qwen3_omni"):
            thinker = getattr(model, "thinker", model)
            llm = getattr(thinker, "model", thinker)
            norm = getattr(llm, "norm", getattr(llm, "model", llm))
            if hasattr(norm, "norm"):
                norm = norm.norm
            torch.save(norm.state_dict(), norm_path)
            lm_head = getattr(thinker, "lm_head", None) or getattr(model, "lm_head")
            torch.save(lm_head.state_dict(), lm_head_path)
        elif model_name == "baichuan_omni":
            llm = getattr(model, "model", model)
            torch.save(llm.norm.state_dict(), norm_path)
            torch.save(model.lm_head.state_dict(), lm_head_path)
        elif model_name == "uni_moe_2_omni":
            torch.save(model.model.norm.state_dict(), norm_path)
            torch.save(model.lm_head.state_dict(), lm_head_path)
        elif model_name == "minicpm_o":
            llm = adapter.model.llm
            torch.save(llm.model.norm.state_dict(), norm_path)
            torch.save(llm.lm_head.state_dict(), lm_head_path)
        elif model_name == "video_salmonn_2":
            torch.save(model.model.norm.state_dict(), norm_path)
            torch.save(model.lm_head.state_dict(), lm_head_path)
        logger.info(f"Saved norm + lm_head weights to {output_dir}")
    except Exception as e:
        logger.warning(f"Could not save lm_head weights for {model_name}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IMAVB hidden state extraction — §4.3 + Appendix J"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(ADAPTER_REGISTRY.keys()),
        help="Model registry name (one of: " + ", ".join(ADAPTER_REGISTRY.keys()) + ")",
    )
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Model checkpoint path or HuggingFace ID (default: per-model default)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=SPLITS,
        help="Dataset splits to process (default: all four splits)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum samples per split (default: 500)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Root output directory. Defaults to HIDDEN_STATES_ROOT/{model} "
            "from config.py. Files are saved as {output_dir}/{split}/{video_id}.pt"
        ),
    )
    parser.add_argument(
        "--lora_ckpt",
        default=None,
        help="LoRA checkpoint path (video_salmonn_2 only)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=50,
        help=(
            "Maximum video frames to sample (default: 50 per paper Appendix J "
            "'50 uniformly sampled video frames')"
        ),
    )
    args = parser.parse_args()

    # Resolve checkpoint
    if args.pretrained is None:
        args.pretrained = MODEL_DEFAULTS.get(args.model, args.model)

    # Resolve output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(HIDDEN_STATES_ROOT, args.model)

    model_args: dict = {"pretrained": args.pretrained}
    if args.lora_ckpt:
        model_args["lora_ckpt"] = args.lora_ckpt

    logger.info(f"Loading {args.model} from {args.pretrained}")
    adapter = get_adapter(args.model, model_args)
    adapter.load()
    logger.info("Model loaded successfully")

    os.makedirs(args.output_dir, exist_ok=True)
    save_lm_head_weights(adapter, args.output_dir)

    dataset = load_dataset(DATASET_NAME, token=True, cache_dir="video-caption-dataset")

    total_saved = 0
    total_errors = 0

    for split_name in args.splits:
        split_dir = os.path.join(args.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing split: {split_name}")
        logger.info(f"{'=' * 60}")

        split_data = dataset[split_name]
        n_samples = min(args.limit, len(split_data))

        for i in range(n_samples):
            sample = split_data[i]
            video_id = sample.get("video_id", f"sample_{i}")
            out_path = os.path.join(split_dir, f"{video_id}.pt")

            if os.path.exists(out_path):
                logger.info(f"  [{i + 1}/{n_samples}] {video_id} — already exists, skipping")
                total_saved += 1
                continue

            video_path = get_video_path(sample)
            logger.info(f"  [{i + 1}/{n_samples}] {video_id}")

            try:
                hidden_states = adapter.extract_hidden_states(sample, video_path)

                is_misleading = "misleading" in split_name
                correct_answer = sample.get("correct_answer", "")
                should_reject = correct_answer in ("E", "F")

                metadata = {
                    "video_id": video_id,
                    "split": split_name,
                    "model": args.model,
                    "correct_answer": correct_answer,
                    "is_misleading": is_misleading,
                    "should_reject": should_reject,
                    "question": sample.get("question", ""),
                }

                torch.save(
                    {
                        "hidden_states": torch.from_numpy(hidden_states),
                        "metadata": metadata,
                    },
                    out_path,
                )

                total_saved += 1
                logger.info(
                    f"    Saved shape={hidden_states.shape} "
                    f"(layers={hidden_states.shape[0]}, dim={hidden_states.shape[1]}) "
                    f"-> {out_path}"
                )

            except Exception as e:
                total_errors += 1
                logger.error(f"    Error processing {video_id}: {e}")
                logger.debug(traceback.format_exc())

            finally:
                torch.cuda.empty_cache()

    logger.info(f"\nDone. Saved: {total_saved}, Errors: {total_errors}")
    logger.info(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
