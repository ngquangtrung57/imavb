#!/usr/bin/env python3
"""
IMAVB Data Pipeline - Level-1 Vision Captioning
Uses Azure OpenAI GPT-4o to generate vision captions for 10-second video segments.

Usage:
    python vision_captioning.py --video-dir <SET_PATH> --output-dir <SET_PATH>

Azure credentials are read from environment variables:
    AZURE_ENDPOINT   - Azure OpenAI endpoint URL
    AZURE_API_KEY    - Azure OpenAI API key
    AZURE_DEPLOYMENT - Deployment name (default: gpt-4o)
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
import tempfile
import subprocess
import shutil
import gc
from typing import Optional, List
from dataclasses import dataclass

try:
    import av
except ImportError:
    av = None
try:
    import cv2
except ImportError:
    cv2 = None
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import torch
from openai import AzureOpenAI
from tqdm import tqdm


###################################################################################################
# Prompts (from IMAVB paper Appendix H, "Level-1 Vision Caption")
###################################################################################################

VISION_SYSTEM_PROMPT = ""

VISION_USER_PROMPT = (
    "Describe this 10-second video clip.\n\n"
    "Focus on visible content only: the people, actions, objects, "
    "setting, and any notable visual changes. Write a short, "
    "concrete caption."
)


###################################################################################################
# Data structures
###################################################################################################

@dataclass
class VisionRequest:
    video_id: str
    start_time: float
    end_time: float
    frames_b64: List[str]
    request_id: str
    segment_path: Optional[str] = None


###################################################################################################
# Azure OpenAI client
###################################################################################################

def create_vision_client():
    """Create Azure OpenAI client. Reads credentials from environment variables."""
    endpoint = os.getenv("AZURE_ENDPOINT", "<AZURE_ENDPOINT>")
    deployment = os.getenv("AZURE_DEPLOYMENT", "<AZURE_DEPLOYMENT>")
    api_key = os.getenv("AZURE_API_KEY", "<AZURE_API_KEY>")

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2025-01-01-preview",
    )


def is_content_filter_error(error_msg: str) -> bool:
    content_filter_indicators = [
        "ResponsibleAIPolicyViolation",
        "content_filter",
        "content management policy",
    ]
    return any(indicator in str(error_msg) for indicator in content_filter_indicators)


###################################################################################################
# Frame extraction (GPU-accelerated via ffmpeg NVDEC, CPU PyAV fallback)
# 10 frames per 10s segment at 1fps
###################################################################################################

def _gpu_extract_frames_ffmpeg(path, start_time, end_time, num_frames=10):
    """Extract frames using ffmpeg + NVDEC/CUDA for GPU acceleration."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    tmp_dir = tempfile.mkdtemp(prefix="frames_")
    out_pattern = os.path.join(tmp_dir, "frame_%05d.png")
    duration = end_time - start_time
    fps = num_frames / duration if duration > 0 else 1.0

    try:
        gpu_devices = []
        try:
            if torch.cuda.is_available():
                gpu_devices = list(range(torch.cuda.device_count()))
        except Exception:
            pass

        gpu_option = []
        if gpu_devices:
            gpu_idx = os.getpid() % len(gpu_devices)
            gpu_option = ["-hwaccel_device", str(gpu_idx)]

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-hwaccel", "cuda",
        ] + gpu_option + [
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", path,
            "-vf", f"fps={fps},scale=512:-1:flags=lanczos",
            "-q:v", "2",
            out_pattern,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            stderr_msg = result.stderr.decode("utf-8", errors="ignore")
            logging.getLogger(__name__).warning(f"CUDA decode failed, trying software: {stderr_msg}")
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-ss", str(start_time),
                "-t", str(duration),
                "-i", path,
                "-vf", f"fps={fps},scale=512:-1:flags=lanczos",
                "-q:v", "2",
                out_pattern,
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                stderr_msg = result.stderr.decode("utf-8", errors="ignore")
                raise RuntimeError(f"ffmpeg failed: {stderr_msg}")

        frames = sorted(
            f for f in os.listdir(tmp_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not frames:
            raise RuntimeError("No frames extracted via ffmpeg")

        frames_b64 = []
        for fname in frames:
            fpath = os.path.join(tmp_dir, fname)
            with open(fpath, "rb") as fp:
                frames_b64.append(base64.b64encode(fp.read()).decode("utf-8"))

        return frames_b64

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _cpu_extract_frames_pyav(path, start_time, end_time, num_frames=10):
    """Fallback: extract frames using PyAV + OpenCV (CPU-based)."""
    if av is None or cv2 is None:
        raise RuntimeError("PyAV or OpenCV not available for CPU fallback")

    frames_b64 = []
    container = None

    try:
        container = av.open(path)
        video_stream = container.streams.video[0]

        if video_stream.average_rate:
            video_fps = float(video_stream.average_rate)
        elif video_stream.base_rate:
            video_fps = float(video_stream.base_rate)
        else:
            video_fps = 25.0

        start_pts = int(start_time * video_fps)
        container.seek(int(start_time * 1_000_000))

        duration = end_time - start_time
        target_frame_interval = duration * video_fps / num_frames if num_frames > 1 else duration * video_fps / 2

        frame_count = 0
        extracted_count = 0

        for frame in container.decode(video=0):
            frame_time = float(frame.pts * frame.time_base) if frame.pts else frame_count / video_fps

            if frame_time < start_time:
                frame_count += 1
                continue
            if frame_time >= end_time:
                break

            if extracted_count == 0 or (frame_count - start_pts) >= extracted_count * target_frame_interval:
                img_array = frame.to_ndarray(format="bgr24")
                height, width = img_array.shape[:2]
                if width > 512:
                    new_height = int(height * 512 / width)
                    img_array = cv2.resize(img_array, (512, new_height), interpolation=cv2.INTER_LANCZOS4)
                success, buffer = cv2.imencode(".png", img_array)
                if success:
                    frames_b64.append(base64.b64encode(buffer).decode("utf-8"))
                    extracted_count += 1
                del img_array, buffer
                if extracted_count >= num_frames:
                    break

            frame_count += 1
            if frame_count % 100 == 0:
                gc.collect()

        if not frames_b64:
            raise RuntimeError(f"No frames could be extracted from {path}")

        return frames_b64

    finally:
        if container:
            try:
                container.close()
            except Exception:
                pass
        gc.collect()


def extract_frames_from_video_segment(video_path, start_time, end_time, num_frames=10):
    """Extract num_frames frames from [start_time, end_time] at ~1fps (10 frames per 10s)."""
    logger = logging.getLogger(__name__)

    try:
        try:
            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False

        if cuda_available:
            frames_b64 = _gpu_extract_frames_ffmpeg(video_path, start_time, end_time, num_frames)
            logger.debug(f"GPU frame extraction successful: {len(frames_b64)} frames")
            return frames_b64
        else:
            logger.debug("CUDA not available, using CPU fallback")

    except Exception as gpu_err:
        logger.warning(f"GPU frame extraction failed: {gpu_err}")

    try:
        frames_b64 = _cpu_extract_frames_pyav(video_path, start_time, end_time, num_frames)
        logger.debug(f"CPU frame extraction successful: {len(frames_b64)} frames")
        return frames_b64
    except Exception as cpu_err:
        raise RuntimeError(f"All frame extraction methods failed. CPU: {cpu_err}")


###################################################################################################
# Duration helper
###################################################################################################

def get_video_duration(path):
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        try:
            cmd = ["ffmpeg", "-i", path, "-f", "null", "-"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            import re
            duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
            if duration_match:
                hours, minutes, seconds = duration_match.groups()
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        except Exception:
            pass
        raise RuntimeError(f"Could not determine duration of {path}: {e}")


###################################################################################################
# Inference
###################################################################################################

def _single_frames_inference(client, segment_data, user_prompt, deployment):
    """Single-segment frames inference via Azure GPT-4o. Raises on unrecoverable failure."""
    frames_b64 = segment_data['frames_b64']
    start_time = segment_data['start_time']
    end_time = segment_data['end_time']
    logger = logging.getLogger(__name__)

    user_content = [{"type": "text", "text": user_prompt}]
    for frame_base64 in frames_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{frame_base64}",
                "detail": "high"
            }
        })

    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            messages = [{"role": "user", "content": user_content}]
            if VISION_SYSTEM_PROMPT.strip():
                messages = [
                    {"role": "system", "content": VISION_SYSTEM_PROMPT},
                ] + messages

            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
                max_tokens=800,
                temperature=0.7,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                timeout=None,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            error_str = str(e)
            last_error = e
            if is_content_filter_error(error_str):
                logger.warning(
                    f"Content filter triggered for chunk {start_time}-{end_time}s. "
                    "Returning error string."
                )
                return f"[CONTENT_FILTER_ERROR] Segment {start_time}-{end_time}s blocked by content policy"
            elif "rate_limit" in error_str.lower() or "too_many_requests" in error_str.lower():
                wait_time = (2 ** attempt) * 5
                logger.warning(f"Rate limit hit, waiting {wait_time}s")
                time.sleep(wait_time)
            elif attempt == max_retries - 1:
                logger.error(f"Azure OpenAI failed after {max_retries} attempts: {e}")
                raise
            else:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"All {max_retries} inference attempts failed") from last_error


def vision_inference_frames_batch(client, segments_data, user_prompt, deployment, batch_size=10):
    """Batch frame inference using Azure GPT-4o."""
    results = []

    for i in range(0, len(segments_data), batch_size):
        batch = segments_data[i:i + batch_size]
        batch_results = []

        with ThreadPoolExecutor(max_workers=min(batch_size, 10)) as batch_executor:
            batch_futures = []
            for segment_data in batch:
                future = batch_executor.submit(
                    _single_frames_inference,
                    client, segment_data, user_prompt, deployment
                )
                batch_futures.append((future, segment_data))

            for future, segment_data in batch_futures:
                try:
                    result = future.result(timeout=None)
                    batch_results.append({
                        'segment_data': segment_data,
                        'caption': result,
                        'success': True
                    })
                except Exception as e:
                    batch_results.append({
                        'segment_data': segment_data,
                        'caption': None,
                        'success': False,
                        'error': str(e)
                    })

        results.extend(batch_results)
        if i + batch_size < len(segments_data):
            time.sleep(0.05)

    return results


###################################################################################################
# Per-video processing
###################################################################################################

def process_video(video_path, output_dir, vision_client, deployment,
                  chunk_duration_s=10.0, num_frames=10,
                  min_length_to_process=5.0, skip_end_seconds=0.0,
                  batch_size=5):
    """
    Process a single video file:
    - Splits into 10s segments
    - Extracts 10 frames per segment at 1fps
    - Sends frames to GPT-4o for vision captioning
    - Saves per-video JSON to output_dir
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    logger = logging.getLogger(__name__)

    output_file = os.path.join(output_dir, f"{video_id}.json")
    if os.path.exists(output_file):
        logger.info(f"Skipping {video_id}, already processed.")
        return "skipped"

    try:
        total_duration = get_video_duration(video_path)
        if total_duration is None or total_duration <= 0:
            logger.error(f"[{video_id}] Invalid duration: {total_duration}")
            return "error"

        if total_duration <= min_length_to_process:
            logger.warning(f"[{video_id}] Video too short ({total_duration:.2f}s), skipping.")
            return "skipped"

        effective_duration = total_duration - skip_end_seconds
        if effective_duration < 0:
            effective_duration = total_duration

        intervals = []
        current_time = 0.0
        while current_time + chunk_duration_s <= effective_duration:
            intervals.append((current_time, current_time + chunk_duration_s))
            current_time += chunk_duration_s

        if not intervals:
            logger.warning(f"[{video_id}] No valid intervals")
            return "skipped"

        segments_data = []
        for start_time, end_time in intervals:
            try:
                frames_b64 = extract_frames_from_video_segment(
                    video_path, start_time, end_time, num_frames=num_frames
                )
                segments_data.append({
                    'frames_b64': frames_b64,
                    'start_time': start_time,
                    'end_time': end_time,
                    'video_id': video_id
                })
            except Exception as frame_err:
                logger.error(f"[{video_id}] Frame extraction failed for chunk {start_time}-{end_time}: {frame_err}")
                continue

        if not segments_data:
            logger.error(f"[{video_id}] No segments could be extracted")
            return "error"

        logger.info(f"[{video_id}] Processing {len(segments_data)} segments...")

        batch_results = vision_inference_frames_batch(
            vision_client,
            segments_data,
            VISION_USER_PROMPT,
            deployment,
            batch_size=batch_size
        )

        failed_segments = sum(1 for r in batch_results if not r['success'])
        if failed_segments > 0:
            logger.error(f"[{video_id}] {failed_segments} segments failed, not saving JSON file")
            return "error"

        captions = []
        for result in batch_results:
            captions.append({
                "start_time": result['segment_data']['start_time'],
                "end_time": result['segment_data']['end_time'],
                "vision_caption": result['caption'],
            })

        if not captions:
            logger.error(f"[{video_id}] No captions generated")
            return "error"

        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(captions, f, ensure_ascii=False, indent=2)

        logger.info(f"[{video_id}] Success: {len(captions)} chunks captioned")
        return "success"

    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(f"[{video_id}] Error: {e}")
        return "error"


###################################################################################################
# CLI
###################################################################################################

def signal_handler(signum, frame):
    logging.getLogger(__name__).info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="IMAVB Level-1 Vision Captioning with Azure OpenAI GPT-4o"
    )
    parser.add_argument(
        "--video-dir", type=str, required=True,
        help="Directory containing input video files (searched recursively)"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to write per-video JSON caption files"
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="Directory for log files (default: --output-dir/logs)"
    )
    parser.add_argument(
        "--workers", type=int, default=40,
        help="Number of concurrent video processing workers (default: 40)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=24,
        help="Batch size for frame processing per worker (default: 24)"
    )
    parser.add_argument(
        "--num-frames", type=int, default=10,
        help="Frames to extract per 10s segment at 1fps (default: 10)"
    )
    parser.add_argument(
        "--chunk-duration", type=float, default=10.0,
        help="Segment duration in seconds (default: 10.0)"
    )
    return parser.parse_args()


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = args.log_dir or os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"vision_captioning_{run_timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8", mode="a"),
        ],
    )
    logger = logging.getLogger(__name__)

    # Resolve Azure deployment name
    deployment = os.getenv("AZURE_DEPLOYMENT", "gpt-4o")

    logger.info("=" * 80)
    logger.info("IMAVB Level-1 Vision Captioning")
    logger.info("=" * 80)
    logger.info(f"Video dir:   {args.video_dir}")
    logger.info(f"Output dir:  {args.output_dir}")
    logger.info(f"Log dir:     {log_dir}")
    logger.info(f"Deployment:  {deployment}")
    logger.info(f"Frames/seg:  {args.num_frames} at 1fps per {args.chunk_duration}s segment")
    logger.info(f"Workers:     {args.workers}, Batch size: {args.batch_size}")

    # Log GPU info
    try:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"CUDA available with {num_gpus} GPUs")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            torch.cuda.empty_cache()
        else:
            logger.warning("CUDA not available - will use CPU fallback only")
    except Exception as e:
        logger.warning(f"Error checking CUDA environment: {e}")

    # Create Azure OpenAI client
    vision_client = create_vision_client()
    try:
        test_response = vision_client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
            timeout=None
        )
        logger.info("Successfully connected to Azure OpenAI")
    except Exception as e:
        logger.warning(f"Could not test Azure OpenAI connection: {e}")

    # Find all video files
    video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    all_files = []
    if not os.path.isdir(args.video_dir):
        logger.error(f"Video directory does not exist: {args.video_dir}")
        sys.exit(1)

    for root, _, files in os.walk(args.video_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in video_extensions:
                all_files.append(os.path.join(root, file))
    all_files = sorted(all_files)

    # Skip already processed videos
    processed_ids = set()
    if os.path.isdir(args.output_dir):
        processed_ids = {
            os.path.splitext(f)[0]
            for f in os.listdir(args.output_dir)
            if f.endswith(".json")
        }
    if processed_ids:
        before_skip = len(all_files)
        all_files = [
            vp for vp in all_files
            if os.path.splitext(os.path.basename(vp))[0] not in processed_ids
        ]
        logger.info(f"Skipping {before_skip - len(all_files)} already-processed videos.")

    logger.info(f"Found {len(all_files)} videos to process.")

    actual_workers = min(args.workers, len(all_files)) if all_files else 1

    results = {"success": 0, "skipped": 0, "error": 0}
    try:
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {
                executor.submit(
                    process_video, video_path, args.output_dir, vision_client,
                    deployment, args.chunk_duration, args.num_frames,
                    batch_size=args.batch_size
                ): video_path
                for video_path in all_files
            }
            with tqdm(total=len(all_files), desc="Processing videos") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results[result] += 1
                    except KeyboardInterrupt:
                        logger.info("Received interrupt signal, cancelling remaining tasks...")
                        for f in futures:
                            f.cancel()
                        break
                    except Exception as exc:
                        logger.error(f"{futures[future]} generated an exception: {exc}")
                        results["error"] += 1
                    pbar.update(1)
                    pbar.set_postfix(results)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")

    logger.info("Level-1 vision captioning complete.")
    logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()
