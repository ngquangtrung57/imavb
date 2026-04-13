#!/usr/bin/env python3
"""
IMAVB Annotation Pipeline - Pass 2: Detail Enhancement

Qwen3.5-27B integrates all three caption streams (Omni/Vision/Audio) from Pass 1
into a unified enhanced caption per 10-second segment.

Trust hierarchy (paper Section 3.2): Omni > Vision > Audio
- Omni captions capture both modalities accurately (PRIMARY source of truth)
- Vision captions provide reliable visual detail
- Audio captions may contain errors (inferred visual events without visual context)

Input:
  --pass1-dir   : Pass 1 output directory (unified_10s_raw JSON files)
  --vision-dir  : Level-1 vision caption directory
  --audio-dir   : Level-1 audio caption directory
  --output-dir  : Output directory for enhanced captions

Usage:
  python pass2_enhancement.py \
      --pass1-dir   <SET_PATH>/unified_10s_raw \
      --vision-dir  <SET_PATH>/level1_vision \
      --audio-dir   <SET_PATH>/level1_audio \
      --output-dir  <SET_PATH>/unified_10s_enhanced
"""

import os
import sys
import json
import argparse
import logging
import asyncio
from typing import List, Dict, Any

from tqdm import tqdm
from vllm import SamplingParams
from vllm import AsyncLLMEngine, AsyncEngineArgs
from transformers import AutoTokenizer


###################################################################################################
# Model settings
###################################################################################################

MODEL_PATH = "Qwen/Qwen3.5-27B"
MAX_MODEL_LEN = 32768

# Sampling parameters (Qwen 3.5 style)
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
MIN_P = 0.0
PRESENCE_PENALTY = 1.5
MAX_TOKENS = 2000

# Concurrency - 128 workers for text-only processing
NUM_WORKERS = 128


###################################################################################################
# Enhancement Prompt - paper Appendix H (ground truth)
# Trust hierarchy: Omni (Primary) > Vision > Audio
###################################################################################################

ENHANCE_SYSTEM_PROMPT = (
    "You are an expert video caption enhancer. Add specific details "
    "while maintaining the primary source's accuracy."
)

ENHANCE_PROMPT = """You are enhancing a video segment caption by combining details from multiple sources. Your output must be a clean, natural description of what happens in the video - NEVER mention the sources or any conflicts between them.

## Primary Caption (MOST RELIABLE):
{primary_caption}

## Vision Caption (RELIABLE for visual details):
{vision_caption}

## Audio Caption (may have errors - lacks visual context):
{audio_caption}

## Your Task:
Combine these sources into ONE enhanced caption:
1. Primary Caption = TRUTH - Start with this as your base
2. Add visual details from Vision Caption
3. Add audio details ONLY if they match the scene
4. Resolve conflicts internally - Trust Primary > Vision > Audio

## CRITICAL RULES:
- Never reference models, captions, sources, or conflicts
- Write ONLY what a viewer would see and hear
- If sources conflict, choose the most reliable source silently

Enhanced caption:"""


###################################################################################################
# Utilities
###################################################################################################

def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "pass2_run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8", mode='a'),
        ],
    )
    return logging.getLogger(__name__)


def load_10s_captions(video_id: str, vision_dir: str, audio_dir: str) -> Dict[str, Dict]:
    """Load 10s vision and audio captions for a video, indexed by time range."""
    result = {"vision": {}, "audio": {}}

    # Load vision captions
    vision_path = os.path.join(vision_dir, f"{video_id}.json")
    if os.path.exists(vision_path):
        with open(vision_path, 'r', encoding='utf-8') as f:
            captions = json.load(f)
            for cap in captions:
                key = (cap.get("start_time", 0), cap.get("end_time", 0))
                result["vision"][key] = cap.get("vision_caption", "")

    # Load audio captions
    audio_path = os.path.join(audio_dir, f"{video_id}.json")
    if os.path.exists(audio_path):
        with open(audio_path, 'r', encoding='utf-8') as f:
            captions = json.load(f)
            for cap in captions:
                key = (cap.get("start_time", 0), cap.get("end_time", 0))
                result["audio"][key] = cap.get("audio_caption", "")

    return result


def get_matching_caption(start_time: float, end_time: float, captions_dict: Dict) -> str:
    """Get caption matching the given time range (with integer-key fallback)."""
    key = (start_time, end_time)
    if key in captions_dict:
        return captions_dict[key]

    key_int = (int(start_time), int(end_time))
    for k, v in captions_dict.items():
        if (int(k[0]), int(k[1])) == key_int:
            return v

    return "(No matching caption available)"


def get_videos_to_process(pass1_dir: str, output_dir: str) -> List[Dict[str, Any]]:
    """Return Pass 1 videos that have not yet been enhanced."""
    all_videos = []

    if not os.path.isdir(pass1_dir):
        logging.error(f"Pass 1 output directory not found: {pass1_dir}")
        return []

    processed_ids = set()
    if os.path.isdir(output_dir):
        processed_ids = {
            os.path.splitext(f)[0]
            for f in os.listdir(output_dir)
            if f.endswith(".json")
        }

    for filename in os.listdir(pass1_dir):
        if not filename.endswith(".json"):
            continue

        video_id = os.path.splitext(filename)[0]

        if video_id in processed_ids:
            continue

        filepath = os.path.join(pass1_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not data.get("segment_captions"):
                continue

            all_videos.append({
                "video_id": video_id,
                "filepath": filepath,
                "data": data
            })

        except Exception as e:
            logging.warning(f"Error loading {filename}: {e}")
            continue

    all_videos.sort(key=lambda x: x["video_id"])
    return all_videos


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
# Main Processing
###################################################################################################

async def process_video(
    video_info: Dict[str, Any],
    engine: AsyncLLMEngine,
    tokenizer: Any,
    sampling_params: SamplingParams,
    vision_dir: str,
    audio_dir: str,
    output_dir: str,
    logger: logging.Logger
) -> bool:
    """Enhance all 10s segment captions for one video by fusing three caption streams."""
    video_id = video_info["video_id"]
    data = video_info["data"]
    segment_captions = data["segment_captions"]

    logger.info(f"Enhancing {len(segment_captions)} segments for {video_id}")

    captions_10s = load_10s_captions(video_id, vision_dir, audio_dir)

    enhanced_segments = []

    for idx, seg in enumerate(segment_captions):
        start_time = seg["start_time"]
        end_time = seg["end_time"]
        raw_caption = seg["raw_caption"]

        vision_caption = get_matching_caption(start_time, end_time, captions_10s["vision"])
        audio_caption = get_matching_caption(start_time, end_time, captions_10s["audio"])

        prompt = ENHANCE_PROMPT.format(
            primary_caption=raw_caption,
            vision_caption=vision_caption,
            audio_caption=audio_caption
        )

        messages = [
            {"role": "system", "content": ENHANCE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        try:
            request_id = f"{video_id}_{idx}"
            results_generator = engine.generate(formatted_prompt, sampling_params, request_id)

            enhanced_caption = ""
            async for request_output in results_generator:
                if request_output.finished:
                    enhanced_caption = request_output.outputs[0].text.strip()

            # Normalize output prefix
            if enhanced_caption.startswith("final"):
                enhanced_caption = enhanced_caption[5:].lstrip()

            enhanced_segments.append({
                "start_time": start_time,
                "end_time": end_time,
                "duration": seg.get("duration", end_time - start_time),
                "raw_caption": raw_caption,
                "enhanced_caption": enhanced_caption,
                "has_context": seg.get("has_context", False)
            })

            logger.debug(f"  Segment {idx+1}: {start_time}-{end_time}s enhanced")

        except Exception as e:
            logger.error(f"Error enhancing segment {start_time}-{end_time} of {video_id}: {e}")
            enhanced_segments.append({
                "start_time": start_time,
                "end_time": end_time,
                "duration": seg.get("duration", end_time - start_time),
                "raw_caption": raw_caption,
                "enhanced_caption": raw_caption,  # fallback to raw
                "has_context": seg.get("has_context", False),
                "enhancement_failed": True
            })

    output_data = {
        "video_id": video_id,
        "video_duration": data.get("video_duration", 0),
        "segment_duration": data.get("segment_duration", 10),
        "num_segments": len(enhanced_segments),
        "segment_captions": enhanced_segments
    }

    output_path = os.path.join(output_dir, f"{video_id}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    logger.info(f"Saved enhanced captions for {video_id}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IMAVB Pass 2: Detail Enhancement — fuses Omni/Vision/Audio captions per segment"
    )
    parser.add_argument(
        "--pass1-dir", required=True,
        help="Directory containing Pass 1 output JSON files (unified_10s_raw)"
    )
    parser.add_argument(
        "--vision-dir", required=True,
        help="Directory containing level-1 vision caption JSON files"
    )
    parser.add_argument(
        "--audio-dir", required=True,
        help="Directory containing level-1 audio caption JSON files"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write enhanced caption JSON files"
    )
    parser.add_argument("--max-processes", type=int, default=1)
    parser.add_argument("--process-index", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument(
        "--max-videos", type=int, default=None,
        help="Limit number of videos to process (for testing)"
    )
    parser.add_argument(
        "--video-ids-file", type=str, default=None,
        help="JSON file(s) with list of video IDs to process (comma-separated for multiple files)"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logger = setup_logging(args.output_dir)

    logger.info("=" * 80)
    logger.info("IMAVB Pass 2: Detail Enhancement (Omni caption is PRIMARY)")
    logger.info("Trust hierarchy: Omni > Vision > Audio")
    logger.info("=" * 80)
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Input dir (Pass 1): {args.pass1_dir}")
    logger.info(f"Vision caption dir: {args.vision_dir}")
    logger.info(f"Audio caption dir:  {args.audio_dir}")
    logger.info(f"Output dir: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_videos = get_videos_to_process(args.pass1_dir, args.output_dir)

    if args.video_ids_file:
        subset_ids = set()
        for f in args.video_ids_file.split(","):
            with open(f.strip()) as fh:
                subset_ids.update(json.load(fh))
        all_videos = [v for v in all_videos if v["video_id"] in subset_ids]
        logger.info(f"Filtered to {len(all_videos)} videos from --video-ids-file")

    pass1_count = (
        len([f for f in os.listdir(args.pass1_dir) if f.endswith(".json")])
        if os.path.isdir(args.pass1_dir) else 0
    )
    processed_count = (
        len([f for f in os.listdir(args.output_dir) if f.endswith(".json")])
        if os.path.isdir(args.output_dir) else 0
    )

    logger.info(
        f"Pass 1 outputs: {pass1_count} | Already enhanced: {processed_count} | "
        f"Remaining: {len(all_videos)}"
    )

    if args.max_videos and args.max_videos > 0:
        all_videos = all_videos[:args.max_videos]
        logger.info(f"Limited to {len(all_videos)} videos (--max-videos={args.max_videos})")

    max_procs = max(1, args.max_processes)
    proc_idx = max(0, min(args.process_index, max_procs - 1))
    assigned_videos = split_for_process(all_videos, max_procs, proc_idx)

    logger.info(f"Process {proc_idx}/{max_procs} assigned {len(assigned_videos)} videos")

    if not assigned_videos:
        logger.info("Nothing to do. Exiting.")
        return

    logger.info(f"Initializing AsyncLLMEngine with tensor_parallel_size={args.tensor_parallel_size}")

    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.95,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=64,
        max_model_len=args.max_model_len,
        seed=42,
        enable_chunked_prefill=True,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        min_p=MIN_P,
        presence_penalty=PRESENCE_PENALTY,
        max_tokens=MAX_TOKENS,
    )

    logger.info("AsyncLLMEngine initialized")
    print(f"[PASS2] Starting async processing with {NUM_WORKERS} concurrent workers", flush=True)

    async def process_video_async(video_info, semaphore, pbar, results):
        async with semaphore:
            video_id = video_info['video_id']
            try:
                result = await process_video(
                    video_info, engine, tokenizer, sampling_params,
                    args.vision_dir, args.audio_dir, args.output_dir, logger
                )
                if result:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
            except Exception as e:
                results['failed'] += 1
                print(f"[PASS2] Exception processing {video_id}: {e}", flush=True)
            finally:
                pbar.update(1)

    async def run_all():
        semaphore = asyncio.Semaphore(NUM_WORKERS)
        results = {'successful': 0, 'failed': 0}
        pbar = tqdm(total=len(assigned_videos), desc="Pass 2: Enhancement")
        tasks = [process_video_async(v, semaphore, pbar, results) for v in assigned_videos]
        await asyncio.gather(*tasks)
        pbar.close()
        return results['successful'], results['failed']

    successful, failed = asyncio.run(run_all())

    logger.info("=" * 80)
    logger.info(f"Pass 2 complete! Successful: {successful}, Failed: {failed}")
    logger.info(f"Next: Run Pass 3 (pass3_merge.py) to merge into unified narrative")
    logger.info("=" * 80)


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
