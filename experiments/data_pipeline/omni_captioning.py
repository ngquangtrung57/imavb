#!/usr/bin/env python3
"""
IMAVB Data Pipeline - Pass 1 Omni Captioning
Uses Qwen3-Omni-30B-A3B-Thinking via vLLM AsyncLLMEngine to generate unified
captions for 10-second video segments combining visual frames + audio.

Two prompt modes (from IMAVB paper Appendix H):
  - FIRST_SEGMENT_PROMPT:   no prior context (segment index 0)
  - CONTEXT_SEGMENT_PROMPT: injects previous segment caption for continuity

10 frames + audio are extracted per 10s segment. The previous segment caption
is passed as context for subsequent segments.

Usage:
    python omni_captioning.py \\
        --vision-caption-dir <SET_PATH> \\
        --audio-caption-dir  <SET_PATH> \\
        --video-dir          <SET_PATH> \\
        --output-dir         <SET_PATH> \\
        --segments-dir       <SET_PATH>
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import traceback
import tempfile
import asyncio
from typing import List, Dict, Any, Optional, Tuple

# Must be set before importing vLLM
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
import numpy as np
import librosa
from tqdm import tqdm
from vllm import SamplingParams
from vllm import AsyncLLMEngine, AsyncEngineArgs
import torchvision.io as tvio


###################################################################################################
# Defaults (override via CLI flags)
###################################################################################################

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
MAX_VIDEO_DURATION = 300.0   # 5 minutes
SEGMENT_DURATION = 10.0      # seconds
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MAX_TOKENS = 4096            # Higher for thinking model (thinking + actual output)
MAX_CONTEXT_CHARS = 1500     # Truncate previous caption to this length
NUM_WORKERS = 12             # Concurrent async workers
AUDIO_SAMPLE_RATE = 16000    # Hz
VIDEO_NUM_FRAMES = 10        # Frames per 10s segment at 1fps


###################################################################################################
# Prompts (from IMAVB paper Appendix H, "Pass 1")
###################################################################################################

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)

# Pass 1: First Segment (no previous context)
FIRST_SEGMENT_PROMPT = """Describe this video clip in detail.

Include:
1. What you see: people, actions, setting, objects
2. What you hear: speech (transcribe it), sounds, music

Write a natural description combining visual and audio."""

# Pass 1: Subsequent Segments (with previous caption as context)
CONTEXT_SEGMENT_PROMPT = """For context, here is what happened in the last 10s of the video:
{previous_caption}

Use this context to understand continuity (same people, ongoing
actions, conversation flow). Now describe THIS current video clip.

Include:
1. What you see: people, actions, setting, objects
2. What you hear: speech (transcribe it), sounds, music

Write a natural description combining visual and audio."""


###################################################################################################
# Logging
###################################################################################################

def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "pass1_run.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    print(f"[PASS1] Logging initialized, log file: {log_path}", flush=True)
    return logger


###################################################################################################
# Video selection
###################################################################################################

def get_videos_to_process(
    vision_caption_dir: str,
    video_dir: str,
    output_dir: str,
    max_duration: float,
    metadata_path: Optional[str] = None,
    max_videos: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Scan vision_caption_dir for eligible videos.
    Optionally restrict to video IDs present in a metadata JSON file.
    Filter by duration, skip already-processed, apply max_videos limit.
    """
    # Build allowed set from metadata file (if provided)
    allowed_ids: Optional[set] = None
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        allowed_ids = set(metadata.keys())
        print(f"[PASS1] Restricting to {len(allowed_ids)} video IDs from metadata", flush=True)

    vision_files = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(vision_caption_dir)
        if f.endswith(".json")
    }

    candidate_ids = set(vision_files.keys())
    if allowed_ids is not None:
        candidate_ids = candidate_ids & allowed_ids

    all_eligible = []
    for video_id in candidate_ids:
        try:
            vision_path = os.path.join(vision_caption_dir, vision_files[video_id])
            with open(vision_path, "r", encoding="utf-8") as f:
                vision_data = json.load(f)

            if not vision_data:
                continue

            video_duration = vision_data[-1].get("end_time", 0.0)
            if video_duration > max_duration or video_duration < SEGMENT_DURATION:
                continue

            video_path = os.path.join(video_dir, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                continue

            all_eligible.append({
                "video_id": video_id,
                "video_path": video_path,
                "video_duration": video_duration,
            })

        except Exception as e:
            logging.warning(f"Error processing {video_id}: {e}")
            continue

    # Deterministic ordering
    all_eligible.sort(key=lambda x: x["video_id"])

    # Apply max_videos limit before filtering processed
    if max_videos and max_videos > 0:
        target_videos = all_eligible[:max_videos]
    else:
        target_videos = all_eligible

    # Filter already processed
    processed_ids = set()
    if os.path.isdir(output_dir):
        processed_ids = {
            os.path.splitext(f)[0]
            for f in os.listdir(output_dir)
            if f.endswith(".json")
        }

    return [v for v in target_videos if v["video_id"] not in processed_ids]


def split_for_process(items: List[Any], max_processes: int, process_index: int) -> List[Any]:
    total = len(items)
    if max_processes <= 1:
        return items
    base = total // max_processes
    rem = total % max_processes
    start = process_index * base + min(process_index, rem)
    end = start + base + (1 if process_index < rem else 0)
    return items[start:end]


###################################################################################################
# Segment extraction helpers
###################################################################################################

def compute_10s_segments(video_duration: float) -> List[Tuple[float, float]]:
    """Compute 10s segment boundaries for a video."""
    segments = []
    current = 0.0
    while current < video_duration:
        end = min(current + SEGMENT_DURATION, video_duration)
        if end - current >= 1.0:
            segments.append((current, end))
        current = end
    return segments


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def extract_video_segment(
    video_path: str,
    start_time: float,
    end_time: float,
    video_id: str,
    segments_dir: str,
) -> Optional[str]:
    """
    Extract a 10s video segment to segments_dir using ffmpeg re-encoding.
    Returns the segment path, or None on failure.
    """
    start_int = int(start_time)
    end_int = int(end_time)
    segment_filename = f"{video_id}_{start_int}_{end_int}.mp4"
    segment_path = os.path.join(segments_dir, segment_filename)
    expected_duration = end_time - start_time

    if os.path.exists(segment_path):
        actual_duration = get_video_duration(segment_path)
        if actual_duration >= expected_duration * 0.8:
            return segment_path
        else:
            os.remove(segment_path)

    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(max(start_time, 0.0)),
            "-i", video_path,
            "-t", str(end_time - start_time),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac",
            "-avoid_negative_ts", "make_zero",
            segment_path,
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        if os.path.exists(segment_path) and os.path.getsize(segment_path) > 10000:
            return segment_path
        else:
            return None

    except subprocess.CalledProcessError:
        return None


def extract_audio_from_video(video_path: str, sample_rate: int = AUDIO_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Extract mono audio from a video file using ffmpeg, loaded via librosa."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_audio_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            tmp_audio_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        audio_signal, sr = librosa.load(tmp_audio_path, sr=sample_rate)
        return (audio_signal.astype(np.float32), sr)
    finally:
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)


def load_video_frames(video_path: str, num_frames: int = VIDEO_NUM_FRAMES) -> np.ndarray:
    """Load evenly-sampled video frames using torchvision."""
    video, _audio, _info = tvio.read_video(video_path, pts_unit="sec")
    total_frames = video.shape[0]

    if total_frames == 0:
        raise ValueError(f"No frames in video: {video_path}")

    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    return video[indices].numpy()


def prepare_video_input(
    segment_path: str,
    prompt_text: str,
    num_frames: int = VIDEO_NUM_FRAMES,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> Dict[str, Any]:
    """
    Load video frames and audio from a segment file and construct
    the vLLM input dict with the raw chat-template prompt.
    """
    video_frames = load_video_frames(segment_path, num_frames)
    audio_data = extract_audio_from_video(segment_path, sample_rate)

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        "<|audio_start|><|audio_pad|><|audio_end|>"
        "<|vision_start|><|video_pad|><|vision_end|>"
        f"{prompt_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    return {
        "prompt": prompt,
        "multi_modal_data": {
            "video": video_frames,
            "audio": audio_data,
        },
    }


###################################################################################################
# Async per-video processing
###################################################################################################

async def process_video_with_engine(
    video_info: Dict[str, Any],
    engine: AsyncLLMEngine,
    sampling_params: SamplingParams,
    segments_dir: str,
    output_dir: str,
    logger: logging.Logger,
) -> bool:
    """Process a single video asynchronously using AsyncLLMEngine."""
    video_id = video_info["video_id"]
    video_path = video_info["video_path"]
    video_duration = video_info["video_duration"]

    print(f"[PASS1] Starting video: {video_id}, duration: {video_duration}s", flush=True)

    segments = compute_10s_segments(video_duration)
    logger.info(f"Processing {video_id}: {video_duration}s -> {len(segments)} segments")

    segment_captions = []
    previous_caption = None

    for idx, (start_time, end_time) in enumerate(segments):
        segment_path = extract_video_segment(
            video_path, start_time, end_time, video_id, segments_dir
        )
        if not segment_path:
            logger.error(f"Failed to extract segment {start_time}-{end_time} for {video_id}")
            return False

        # Select prompt based on whether we have a previous caption
        if idx == 0 or previous_caption is None:
            prompt_text = FIRST_SEGMENT_PROMPT
        else:
            truncated = previous_caption
            if len(previous_caption) > MAX_CONTEXT_CHARS:
                truncated = previous_caption[:MAX_CONTEXT_CHARS] + "..."
            prompt_text = CONTEXT_SEGMENT_PROMPT.format(previous_caption=truncated)

        inputs = prepare_video_input(segment_path, prompt_text)

        try:
            request_id = f"{video_id}_{idx}"
            prompt_input = {
                "prompt": inputs["prompt"],
                "multi_modal_data": inputs.get("multi_modal_data"),
            }
            results_generator = engine.generate(prompt_input, sampling_params, request_id)

            full_output = ""
            async for request_output in results_generator:
                if request_output.finished:
                    full_output = request_output.outputs[0].text.strip()

            # Strip thinking block if present
            thinking_part = ""
            if "</think>" in full_output:
                parts = full_output.split("</think>", 1)
                thinking_part = parts[0].replace("<think>", "").strip()
                actual_caption = parts[1].strip()
            elif "<think>" in full_output:
                thinking_part = full_output.replace("<think>", "").strip()
                actual_caption = full_output
            else:
                actual_caption = full_output

            if len(actual_caption) < 10:
                logger.warning(
                    f"Short/empty caption for segment {start_time}-{end_time} of {video_id}"
                )

            segment_data: Dict[str, Any] = {
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "raw_caption": actual_caption,
                "has_context": idx > 0,
            }
            if thinking_part:
                segment_data["thinking"] = thinking_part

            segment_captions.append(segment_data)

            # Only update context if caption is substantive
            if len(actual_caption) > 50:
                previous_caption = actual_caption

        except Exception as e:
            logger.error(f"Error processing segment {start_time}-{end_time} of {video_id}: {e}")
            traceback.print_exc()
            return False

    output_data = {
        "video_id": video_id,
        "video_duration": video_duration,
        "segment_duration": SEGMENT_DURATION,
        "num_segments": len(segment_captions),
        "segment_captions": segment_captions,
    }

    output_path = os.path.join(output_dir, f"{video_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    logger.info(f"Saved {video_id} ({len(segment_captions)} segments)")
    return True


###################################################################################################
# CLI
###################################################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IMAVB Pass 1 Omni Captioning with Qwen3-Omni-30B-A3B-Thinking"
    )
    parser.add_argument(
        "--vision-caption-dir", type=str, required=True,
        help="Directory containing per-video Level-1 vision caption JSON files"
    )
    parser.add_argument(
        "--audio-caption-dir", type=str, default=None,
        help="Directory containing per-video Level-1 audio caption JSON files (informational)"
    )
    parser.add_argument(
        "--video-dir", type=str, required=True,
        help="Directory containing input video files (.mp4)"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to write per-video Pass 1 omni caption JSON files"
    )
    parser.add_argument(
        "--segments-dir", type=str, required=True,
        help="Directory for extracted 10s video segment files (will be created if absent)"
    )
    parser.add_argument(
        "--metadata-path", type=str, default=None,
        help="Optional JSON file mapping video_id -> metadata; restricts processing to these IDs"
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
        "--max-videos", type=int, default=None,
        help="Limit number of videos to process"
    )
    parser.add_argument(
        "--max-video-duration", type=float, default=MAX_VIDEO_DURATION,
        help=f"Maximum video duration in seconds (default: {MAX_VIDEO_DURATION})"
    )
    parser.add_argument(
        "--num-workers", type=int, default=NUM_WORKERS,
        help=f"Number of concurrent async workers (default: {NUM_WORKERS})"
    )
    parser.add_argument(
        "--video-ids-file", type=str, default=None,
        help="JSON file(s) with list of video IDs to process (comma-separated for multiple files)"
    )
    return parser.parse_args()


###################################################################################################
# Main
###################################################################################################

def main(args: argparse.Namespace) -> None:
    print("[PASS1] Starting main()", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.segments_dir, exist_ok=True)

    logger = setup_logging(args.output_dir)

    logger.info("=" * 80)
    logger.info("IMAVB Pass 1: Omni Captioning (10s segments with context)")
    logger.info("=" * 80)
    logger.info(f"Model:           {args.model_path}")
    logger.info(f"Segment duration: {SEGMENT_DURATION}s")
    logger.info(f"Frames per seg:  {VIDEO_NUM_FRAMES} at 1fps")
    logger.info(f"Output dir:      {args.output_dir}")
    logger.info(f"Segments dir:    {args.segments_dir}")
    if args.metadata_path:
        logger.info(f"Metadata:        {args.metadata_path}")

    # Get eligible video list
    all_videos = get_videos_to_process(
        args.vision_caption_dir,
        args.video_dir,
        args.output_dir,
        args.max_video_duration,
        metadata_path=args.metadata_path,
        max_videos=args.max_videos,
    )

    # Optionally further restrict to a video-ids file
    if args.video_ids_file:
        subset_ids: set = set()
        for fpath in args.video_ids_file.split(","):
            with open(fpath.strip()) as fh:
                subset_ids.update(json.load(fh))
        all_videos = [v for v in all_videos if v["video_id"] in subset_ids]
        logger.info(f"Filtered to {len(all_videos)} videos from --video-ids-file")
        print(f"[PASS1] Filtered to {len(all_videos)} videos from --video-ids-file", flush=True)

    processed_count = (
        len([f for f in os.listdir(args.output_dir) if f.endswith(".json")])
        if os.path.isdir(args.output_dir) else 0
    )
    target_count = (
        min(args.max_videos, processed_count + len(all_videos))
        if args.max_videos else processed_count + len(all_videos)
    )
    logger.info(
        f"Target: {target_count} | Already processed: {processed_count} | Remaining: {len(all_videos)}"
    )
    print(f"[PASS1] Target: {target_count} | Processed: {processed_count} | Remaining: {len(all_videos)}", flush=True)

    # Split across processes
    max_procs = max(1, args.max_processes)
    proc_idx = max(0, min(args.process_index, max_procs - 1))
    assigned_videos = split_for_process(all_videos, max_procs, proc_idx)

    logger.info(f"Process {proc_idx}/{max_procs} assigned {len(assigned_videos)} videos")
    print(f"[PASS1] Assigned {len(assigned_videos)} videos to process", flush=True)

    if not assigned_videos:
        logger.info("Nothing to do. Exiting.")
        print("[PASS1] No videos to process. Exiting.", flush=True)
        return

    # Initialize AsyncLLMEngine
    tensor_parallel_size = args.tensor_parallel_size or torch.cuda.device_count()
    logger.info(f"Initializing AsyncLLMEngine with tensor_parallel_size={tensor_parallel_size}")
    print(f"[PASS1] Initializing AsyncLLMEngine with TP={tensor_parallel_size}", flush=True)

    engine_args = AsyncEngineArgs(
        model=args.model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=tensor_parallel_size,
        limit_mm_per_prompt={"image": 0, "video": 1, "audio": 1},
        max_num_seqs=8,
        max_model_len=16000,
        seed=1234,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
        min_tokens=5,
        repetition_penalty=1.1,
    )

    logger.info("AsyncLLMEngine initialized")
    print("[PASS1] AsyncLLMEngine initialized successfully!", flush=True)

    num_workers = args.num_workers
    print(f"[PASS1] Starting async processing for {len(assigned_videos)} videos with {num_workers} workers", flush=True)

    async def process_video_async(video_info, semaphore, pbar, results):
        async with semaphore:
            video_id = video_info["video_id"]
            try:
                result = await process_video_with_engine(
                    video_info, engine, sampling_params,
                    args.segments_dir, args.output_dir, logger
                )
                if result:
                    results["successful"] += 1
                    print(f"[PASS1] Video {video_id} completed successfully", flush=True)
                else:
                    results["failed"] += 1
                    print(f"[PASS1] Video {video_id} failed", flush=True)
            except Exception as e:
                results["failed"] += 1
                print(f"[PASS1] Exception processing {video_id}: {e}", flush=True)
                traceback.print_exc()
            finally:
                pbar.update(1)

    async def run_all():
        semaphore = asyncio.Semaphore(num_workers)
        results = {"successful": 0, "failed": 0}
        pbar = tqdm(total=len(assigned_videos), desc="Pass 1: Omni Captioning")
        tasks = [process_video_async(v, semaphore, pbar, results) for v in assigned_videos]
        await asyncio.gather(*tasks)
        pbar.close()
        return results["successful"], results["failed"]

    successful, failed = asyncio.run(run_all())

    logger.info("=" * 80)
    logger.info(f"Pass 1 complete! Successful: {successful}, Failed: {failed}")
    logger.info("=" * 80)


if __name__ == "__main__":
    print("[PASS1] Script starting...", flush=True)
    try:
        cli_args = parse_args()
        main(cli_args)
    except Exception as e:
        print(f"[PASS1] FATAL ERROR: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
    print("[PASS1] Script finished.", flush=True)
