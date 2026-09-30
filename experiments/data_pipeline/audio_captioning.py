#!/usr/bin/env python3
"""
IMAVB Data Pipeline - Level-1 Audio Captioning
Uses Qwen3-Omni-30B-A3B-Captioner via vLLM to generate audio captions
for 10-second video segments.

No explicit text prompt is used. The Captioner model receives only the
audio segment with the chat template scaffold and generates a caption
by default (see IMAVB paper Appendix I, "Level-1 Audio Caption").

Audio is extracted at 16kHz mono from each 10s segment.

Usage:
    python audio_captioning.py \\
        --vision-caption-dir <SET_PATH> \\
        --video-dir <SET_PATH> \\
        --output-dir <SET_PATH>
"""

import os
import sys
import json
import argparse
import logging
import tempfile
import subprocess
import wave
import random
import traceback
from typing import List, Dict, Any, Tuple

# Must be set before importing vLLM
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor


###################################################################################################
# Defaults (override via CLI flags)
###################################################################################################

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Captioner"
SAMPLE_RATE = 16000          # 16kHz mono audio input
MIN_VIDEO_DURATION = 60.0    # 1 minute
MAX_VIDEO_DURATION = 300.0   # 5 minutes
MAX_VIDEO_LIMIT = 4000
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MAX_TOKENS = 4096
BATCH_SIZE = 64              # Segments processed in parallel per vLLM call


###################################################################################################
# Logging
###################################################################################################

def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "audio_caption_run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8", mode="a"),
        ],
    )
    return logging.getLogger(__name__)


###################################################################################################
# Audio extraction
###################################################################################################

def extract_audio_segment_from_video(
    video_path: str,
    start_s: float,
    end_s: float,
    sample_rate: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """Extract mono PCM audio for segment [start_s, end_s) at sample_rate Hz."""
    assert end_s > start_s >= 0.0
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav_name = tmp_wav.name
    tmp_wav.close()

    try:
        # First try: fast seek (-ss before -i)
        cmd_fast = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(max(start_s, 0.0)),
            "-to", str(max(end_s, start_s + 0.001)),
            "-i", video_path,
            "-ac", "1",
            "-ar", str(sample_rate),
            "-acodec", "pcm_s16le",
            tmp_wav_name,
        ]
        try:
            subprocess.run(cmd_fast, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as cpe_fast:
            # Retry: accurate seek (-ss after -i)
            cmd_accurate = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", video_path,
                "-ss", str(max(start_s, 0.0)),
                "-to", str(max(end_s, start_s + 0.001)),
                "-ac", "1",
                "-ar", str(sample_rate),
                "-acodec", "pcm_s16le",
                tmp_wav_name,
            ]
            try:
                subprocess.run(cmd_accurate, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except subprocess.CalledProcessError as cpe_acc:
                err_fast = cpe_fast.stderr.decode("utf-8", errors="ignore") if cpe_fast.stderr else ""
                err_acc = cpe_acc.stderr.decode("utf-8", errors="ignore") if cpe_acc.stderr else ""
                raise RuntimeError(
                    f"ffmpeg audio extraction failed for {video_path} [{start_s}-{end_s}].\n"
                    f"Fast seek error: {err_fast}\nAccurate seek error: {err_acc}"
                )

        with wave.open(tmp_wav_name, "rb") as wf:
            num_frames = wf.getnframes()
            sample_width = wf.getsampwidth()
            channels = wf.getnchannels()
            assert channels == 1, f"Expected mono audio after ffmpeg -ac 1, got {channels} channels"
            assert sample_width == 2, f"Expected 16-bit PCM, got {sample_width * 8}-bit"
            audio_bytes = wf.readframes(num_frames)

        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_np, sample_rate

    finally:
        try:
            os.remove(tmp_wav_name)
        except Exception:
            pass


###################################################################################################
# Video selection
###################################################################################################

def get_videos_to_process(
    vision_caption_dir: str,
    video_dir: str,
    output_dir: str,
    min_duration: float,
    max_duration: float,
    max_video_limit: int,
) -> List[Dict[str, Any]]:
    """
    Scan vision_caption_dir for eligible videos, filter by duration, and
    deterministically select up to max_video_limit videos (fixed seed=42).
    Already-processed videos are skipped.
    """
    all_eligible = []

    for caption_file in os.listdir(vision_caption_dir):
        if not caption_file.endswith(".json"):
            continue

        video_id = os.path.splitext(caption_file)[0]
        caption_path = os.path.join(vision_caption_dir, caption_file)

        try:
            with open(caption_path, "r", encoding="utf-8") as f:
                vision_data = json.load(f)

            if not vision_data:
                continue

            last_end_time = vision_data[-1].get("end_time", 0.0)
            if last_end_time < min_duration or last_end_time > max_duration:
                continue

            video_path = os.path.join(video_dir, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                continue

            all_eligible.append({
                "video_id": video_id,
                "video_path": video_path,
                "vision_data": vision_data,
                "duration": last_end_time,
            })

        except Exception as e:
            logging.warning(f"Error reading {caption_file}: {e}")
            continue

    # Sort deterministically then apply fixed-seed shuffle for reproducible selection
    all_eligible.sort(key=lambda x: x["video_id"])
    random.seed(42)
    if len(all_eligible) > max_video_limit:
        random.shuffle(all_eligible)
        target_set = all_eligible[:max_video_limit]
    else:
        target_set = all_eligible

    # Filter out already processed
    processed_ids = set()
    if os.path.isdir(output_dir):
        processed_ids = {
            os.path.splitext(f)[0]
            for f in os.listdir(output_dir)
            if f.endswith(".json")
        }

    return [v for v in target_set if v["video_id"] not in processed_ids]


def split_for_process(items: List[Any], max_processes: int, process_index: int) -> List[Any]:
    """Split items evenly across processes (0-indexed)."""
    total = len(items)
    if max_processes <= 1:
        return items
    base = total // max_processes
    rem = total % max_processes
    start = process_index * base + min(process_index, rem)
    end = start + base + (1 if process_index < rem else 0)
    return items[start:end]


###################################################################################################
# CLI
###################################################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IMAVB Level-1 Audio Captioning with Qwen3-Omni-30B-A3B-Captioner"
    )
    parser.add_argument(
        "--vision-caption-dir", type=str, required=True,
        help="Directory containing per-video Level-1 vision caption JSON files"
    )
    parser.add_argument(
        "--video-dir", type=str, required=True,
        help="Directory containing input video files (.mp4)"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to write per-video audio caption JSON files"
    )
    parser.add_argument(
        "--model-path", type=str, default=MODEL_PATH,
        help=f"HuggingFace model ID or local path (default: {MODEL_PATH})"
    )
    parser.add_argument(
        "--max-processes", type=int, default=1,
        help="Total number of parallel processes (default: 1)"
    )
    parser.add_argument(
        "--process-index", type=int, default=0,
        help="0-based index of this process (default: 0)"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=None,
        help="Number of GPUs for tensor parallelism (default: all available)"
    )
    parser.add_argument(
        "--min-duration", type=float, default=MIN_VIDEO_DURATION,
        help=f"Minimum video duration in seconds (default: {MIN_VIDEO_DURATION})"
    )
    parser.add_argument(
        "--max-duration", type=float, default=MAX_VIDEO_DURATION,
        help=f"Maximum video duration in seconds (default: {MAX_VIDEO_DURATION})"
    )
    parser.add_argument(
        "--max-videos", type=int, default=MAX_VIDEO_LIMIT,
        help=f"Maximum number of videos to process (default: {MAX_VIDEO_LIMIT})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Number of segments per vLLM generate call (default: {BATCH_SIZE})"
    )
    return parser.parse_args()


###################################################################################################
# Main
###################################################################################################

def main(args: argparse.Namespace) -> None:
    logger = setup_logging(args.output_dir)

    logger.info("=" * 80)
    logger.info("IMAVB Level-1 Audio Captioning")
    logger.info("=" * 80)
    logger.info(f"Model:             {args.model_path}")
    logger.info(f"Vision caption dir: {args.vision_caption_dir}")
    logger.info(f"Video dir:         {args.video_dir}")
    logger.info(f"Output dir:        {args.output_dir}")
    logger.info(f"Duration range:    {args.min_duration}s - {args.max_duration}s")
    logger.info(f"Max video limit:   {args.max_videos}")
    logger.info(f"Sample rate:       {SAMPLE_RATE} Hz mono")
    logger.info("Prompt:            None (Captioner model generates caption by default)")

    import shutil as _shutil
    if not _shutil.which("ffmpeg"):
        logger.error("ffmpeg not found in PATH. Please install ffmpeg.")
        sys.exit(1)

    # Gather videos
    all_videos = get_videos_to_process(
        args.vision_caption_dir,
        args.video_dir,
        args.output_dir,
        args.min_duration,
        args.max_duration,
        args.max_videos,
    )

    processed_count = 0
    if os.path.isdir(args.output_dir):
        processed_count = len([f for f in os.listdir(args.output_dir) if f.endswith(".json")])

    total_target = min(args.max_videos, len(all_videos) + processed_count)
    logger.info(
        f"Target: {total_target} | Already processed: {processed_count} | Remaining: {len(all_videos)}"
    )

    # Split across processes
    max_procs = max(1, args.max_processes)
    proc_idx = max(0, min(args.process_index, max_procs - 1))
    assigned_videos = split_for_process(all_videos, max_procs, proc_idx)
    logger.info(f"Process {proc_idx}/{max_procs} assigned {len(assigned_videos)} videos")

    if not assigned_videos:
        logger.info("Nothing to do. Exiting.")
        return

    # Initialize vLLM
    tensor_parallel_size = args.tensor_parallel_size or torch.cuda.device_count()
    logger.info(f"Initializing LLM with tensor_parallel_size={tensor_parallel_size}")

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=tensor_parallel_size,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 1},
        max_num_seqs=64,
        max_model_len=4096,
        seed=1234,
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)

    # Process each video
    with tqdm(total=len(assigned_videos), desc="Processing videos") as pbar:
        for video_info in assigned_videos:
            video_id = video_info["video_id"]
            video_path = video_info["video_path"]
            vision_data = video_info["vision_data"]

            logger.info(f"Processing video: {video_id}")

            try:
                audio_captions = []
                batch_requests = []
                batch_metadata = []

                logger.info(f"Preparing {len(vision_data)} segments for batched processing of {video_id}")

                for segment in vision_data:
                    start_time = segment["start_time"]
                    end_time = segment["end_time"]

                    try:
                        audio_np, sr = extract_audio_segment_from_video(
                            video_path, start_time, end_time, SAMPLE_RATE
                        )

                        # No explicit text prompt: Captioner model generates caption from audio only.
                        # (IMAVB paper Appendix I: "No explicit text prompt is used.")
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "audio", "audio": "placeholder"}
                                ],
                            }
                        ]
                        text = processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )

                        inputs = {
                            "prompt": text,
                            "multi_modal_data": {
                                "audio": (audio_np, sr),
                            },
                        }

                        batch_requests.append(inputs)
                        batch_metadata.append({
                            "start_time": start_time,
                            "end_time": end_time,
                        })

                    except Exception as e:
                        tb = traceback.format_exc()
                        logger.error(
                            f"Error extracting segment {start_time}-{end_time} of {video_id}: "
                            f"{e}\nTraceback:\n{tb}"
                        )
                        continue

                if not batch_requests:
                    logger.warning(f"No valid segments to process for {video_id}")
                else:
                    logger.info(f"Processing {len(batch_requests)} segments in batches of {args.batch_size}")

                    for i in range(0, len(batch_requests), args.batch_size):
                        batch_end = min(i + args.batch_size, len(batch_requests))
                        current_batch = batch_requests[i:batch_end]
                        current_metadata = batch_metadata[i:batch_end]

                        try:
                            logger.info(
                                f"Batch {i // args.batch_size + 1}/"
                                f"{(len(batch_requests) + args.batch_size - 1) // args.batch_size} "
                                f"({len(current_batch)} segments)"
                            )
                            outputs = llm.generate(current_batch, sampling_params=sampling_params)

                            for j, output in enumerate(outputs):
                                audio_caption = output.outputs[0].text
                                metadata = current_metadata[j]
                                audio_captions.append({
                                    "start_time": metadata["start_time"],
                                    "end_time": metadata["end_time"],
                                    "audio_caption": audio_caption,
                                })
                                logger.info(
                                    f"Completed segment {metadata['start_time']}-{metadata['end_time']} of {video_id}"
                                )

                        except Exception as e:
                            tb = traceback.format_exc()
                            logger.error(
                                f"Error processing batch {i // args.batch_size + 1} of {video_id}: "
                                f"{e}\nTraceback:\n{tb}"
                            )
                            continue

                if audio_captions:
                    output_path = os.path.join(args.output_dir, f"{video_id}.json")
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(audio_captions, f, ensure_ascii=False, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    logger.info(f"Saved audio captions for {video_id} ({len(audio_captions)} segments)")
                else:
                    logger.warning(f"No segments successfully processed for {video_id}, skipping save")

            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Error processing video {video_id}: {e}\nTraceback:\n{tb}")

            finally:
                pbar.update(1)

    logger.info("Audio captioning complete!")


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
