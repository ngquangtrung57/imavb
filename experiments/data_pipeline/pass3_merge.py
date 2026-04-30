#!/usr/bin/env python3
"""
IMAVB Annotation Pipeline - Pass 3: Sequential Narrative Unification

Qwen3.5-27B performs a single-shot global merge of all enhanced segment captions
(from Pass 2) into one unified, deduplicated, timestamped narrative per video,
resolving remaining inter-segment inconsistencies (paper Section 3.2).

Deduplication contract:
  - Opening [0s-10s] segment establishes setting, character appearances, and
    ambient sounds once and for all.
  - Continuation segments describe only what is NEW: actions, new dialogue,
    changes to scene. Characters are referred to by pronouns or names -- never
    by clothing or hair after the first description.

Input:
  --pass2-dir   : Pass 2 output directory (unified_10s_enhanced JSON files)
  --output-dir  : Output directory for unified narrative JSON files

Usage:
  python pass3_merge.py \
      --pass2-dir  <SET_PATH>/unified_10s_enhanced \
      --output-dir <SET_PATH>/unified_final
"""

import os
import sys
import json
import re
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
MAX_MODEL_LEN = 80000

# Sampling parameters
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
MIN_P = 0.0
PRESENCE_PENALTY = 1.5
MAX_TOKENS = 40000

# Concurrency
NUM_WORKERS = 48


###################################################################################################
# Pass 3: Merge Prompt
###################################################################################################

MERGE_SYSTEM_PROMPT = (
    "You are a novelist adapting a video into prose. Your cardinal rule: describe everything "
    "ONCE, then never again. The input segments contain massive repetition because they were "
    "captioned independently - your job is to deduplicate them. Setting details go in the "
    "opening [0s-10s] only. Each character's appearance is described once on first appearance, "
    "then only pronouns or names. Ambient sounds noted once, repeated only if they genuinely "
    "change. Dialogue lines appear ONCE (they often repeat across adjacent input segments - "
    "include each line only in its first occurrence). You never reference cameras, microphones, "
    "audio equipment, recording techniques, sound engineering terms (reverb, fidelity, stereo "
    "field, sub-bass, frequency, HVAC, soundscape, etc.), captions, models, or any technical "
    "process."
)

MERGE_PROMPT = """You are writing a chapter of a novel based on a video. The video has been captioned in {num_segments} overlapping 10-second segments. **IMPORTANT**: These segments were captioned independently by a vision+audio model, which means:

1. **Each segment describes the FULL scene** as if the model had never seen prior segments
2. **Settings are re-described** in nearly every segment (walls, furniture, lighting, room layout)
3. **Character appearances are re-described** every time they appear (clothing, hair, features)
4. **Ambient sounds are re-introduced** repeatedly (music, hum, traffic, chatter)
5. **Dialogue lines may appear in 2-3 adjacent segments** because the model doesn't know what was already captured

**Your job is to deduplicate these redundant descriptions into a flowing narrative.**

## SEGMENT CAPTIONS (WITH EXTENSIVE REPETITION):
{all_segments}

## YOUR TASK:
Rewrite these as a single flowing narrative with [0s-10s] through [{last_start}s-{last_end}s] timestamp markers. Remove ALL repeated descriptions while preserving the timeline.

**CRITICAL STRUCTURE -- THE OPENING vs CONTINUATION RULE:**
- **[0s-10s] (OPENING)**: Establish everything once and for all:
  - **Setting**: Full room/location description (walls, floor, furniture, lighting, windows, decor). This is the ONLY segment where you describe the space.
  - **Characters**: Every person's complete appearance (clothing from head to toe, hair color/style, distinguishing features, accessories). This is the ONLY segment where you describe what they're wearing.
  - **Ambient sounds**: Background audio that persists (music genre, traffic hum, machine noise, crowd chatter). These stay constant unless explicitly changed later.

- **[10s-20s] through [{last_start}s-{last_end}s] (CONTINUATION)**: Write ONLY what is NEW:
  - **Actions and movements** (walking, gesturing, picking up objects)
  - **New dialogue** (each spoken line appears ONCE, in the segment where it first occurs)
  - **Changes to the scene** (new character enters -> describe them once; light turns off -> mention it; music stops -> mention it)
  - **Use pronouns/names for characters**: "he", "she", "they", character names, or role labels ("the host", "the driver"). NEVER use clothing or hair as identifiers.

**DEDUPLICATION EXAMPLES:**

**BAD (repeating setting):**
"[0s-10s] The couple sits in a dimly lit living room with cream-colored walls and a burgundy sofa. [10s-20s] They continue talking in the dimly lit living room with cream-colored walls. [20s-30s] The conversation unfolds in the same dimly lit space with burgundy furniture."

**GOOD (setting described once):**
"[0s-10s] The couple sits in a dimly lit living room with cream-colored walls and a burgundy sofa. [10s-20s] They continue talking, leaning closer. [20s-30s] She gestures toward the window as he nods."

---

**BAD (repeating character appearance):**
"[0s-10s] A woman in a red jacket and blonde hair speaks. [10s-20s] The woman in the red jacket nods. [20s-30s] The blonde woman in the red jacket smiles."

**GOOD (appearance described once, pronouns after):**
"[0s-10s] A woman in a red jacket, her blonde hair pulled back, speaks into the microphone. [10s-20s] She nods, leaning forward. [20s-30s] She smiles and glances at the audience."

---

**BAD (repeating ambient sound):**
"[0s-10s] Piano music plays softly in the background. [10s-20s] The piano continues its gentle melody. [20s-30s] Soft piano notes fill the air. [30s-40s] The piano music persists."

**GOOD (ambient sound mentioned once):**
"[0s-10s] Piano music plays softly in the background. [10s-20s] She opens the envelope and pulls out a letter. [20s-30s] She reads silently, her expression shifting. [30s-40s] She sets the letter down and exhales."

---

**BAD (repeating dialogue):**
"[20s-30s] He says, 'Where did you go?' [30s-40s] 'Where did you go?' he asks again."

**GOOD (dialogue appears once):**
"[20s-30s] He says, 'Where did you go?' [30s-40s] She hesitates, then looks away without answering."

**OTHER RULES:**
1. ALL {num_segments} timestamp markers must appear: [0s-10s], [10s-20s], ..., [{last_start}s-{last_end}s]. Never skip or combine.
2. Each segment includes both visual AND audio details.
3. Each segment continues from where the previous ended -- flowing narrative, not independent paragraphs.
4. NEVER mention technical terms: "the audio model", "the vision model", "the caption", "recording equipment", audio engineering vocabulary (reverb, fidelity, stereo field, sub-bass, frequency, HVAC, soundscape, etc.)

Begin writing with [0s-10s]:"""


###################################################################################################
# Utilities
###################################################################################################

def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "pass3_run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8", mode='a'),
        ],
    )
    return logging.getLogger(__name__)


def get_videos_to_process(pass2_dir: str, output_dir: str) -> List[Dict[str, Any]]:
    """Return Pass 2 videos that have not yet been unified."""
    all_videos = []

    if not os.path.isdir(pass2_dir):
        logging.error(f"Pass 2 output directory not found: {pass2_dir}")
        return []

    processed_ids = set()
    if os.path.isdir(output_dir):
        processed_ids = {
            os.path.splitext(f)[0]
            for f in os.listdir(output_dir)
            if f.endswith(".json")
        }

    for filename in os.listdir(pass2_dir):
        if not filename.endswith(".json"):
            continue

        video_id = os.path.splitext(filename)[0]

        if video_id in processed_ids:
            continue

        filepath = os.path.join(pass2_dir, filename)
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
# Processing helpers
###################################################################################################

def format_segments_for_prompt(segment_captions: List[Dict]) -> str:
    """Format all segment captions as structured input for the merge prompt."""
    lines = []
    for seg in segment_captions:
        st = int(seg["start_time"])
        et = int(seg["end_time"])
        caption = seg.get("enhanced_caption", seg.get("raw_caption", ""))
        lines.append(f"[{st}s-{et}s]:\n{caption}")
    return "\n\n".join(lines)


async def generate_single(
    engine: AsyncLLMEngine,
    tokenizer: Any,
    sampling_params: SamplingParams,
    messages: List[Dict],
    request_id: str,
) -> str:
    """Single async generation call."""
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    response = ""
    results_generator = engine.generate(formatted_prompt, sampling_params, request_id)
    async for request_output in results_generator:
        if request_output.finished:
            response = request_output.outputs[0].text.strip()

    if response.startswith("final"):
        response = response[5:].lstrip()

    return response


###################################################################################################
# Main Processing -- single-shot global merge
###################################################################################################

async def process_video(
    video_info: Dict[str, Any],
    engine: AsyncLLMEngine,
    tokenizer: Any,
    sampling_params: SamplingParams,
    output_dir: str,
    logger: logging.Logger
) -> bool:
    """Merge all enhanced segment captions into one unified timestamped narrative."""
    video_id = video_info["video_id"]
    data = video_info["data"]
    segment_captions = data["segment_captions"]
    num_segments = len(segment_captions)
    video_duration = data.get("video_duration", num_segments * 10)

    logger.info(f"Processing {video_id}: {num_segments} segments ({video_duration}s)")

    if not segment_captions:
        logger.warning(f"No segments for {video_id}")
        return False

    last_start = int(segment_captions[-1]["start_time"])
    last_end = int(segment_captions[-1]["end_time"])

    try:
        all_segments_text = format_segments_for_prompt(segment_captions)

        merge_prompt = MERGE_PROMPT.format(
            all_segments=all_segments_text,
            num_segments=num_segments,
            last_start=last_start,
            last_end=last_end,
        )

        merge_messages = [
            {"role": "system", "content": MERGE_SYSTEM_PROMPT},
            {"role": "user", "content": merge_prompt},
        ]

        unified_caption = await generate_single(
            engine, tokenizer, sampling_params,
            merge_messages, f"{video_id}_merge",
        )

        logger.info(
            f"  {video_id}: Merge complete, len={len(unified_caption)}"
        )

        # Strip any preamble before first timestamp marker
        first_marker = "[0s-10s]"
        if first_marker in unified_caption:
            marker_pos = unified_caption.index(first_marker)
            if marker_pos > 50:
                unified_caption = unified_caption[marker_pos:].strip()

    except Exception as e:
        logger.error(f"Error processing {video_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

    output_data = {
        "video_id": video_id,
        "video_duration": video_duration,
        "num_segments": num_segments,
        "unified_caption": unified_caption,
        "segment_captions": segment_captions,
    }

    output_path = os.path.join(output_dir, f"{video_id}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    logger.info(f"Saved {video_id}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IMAVB Pass 3: Single-shot global merge into unified narrative"
    )
    parser.add_argument(
        "--pass2-dir", required=True,
        help="Directory containing Pass 2 output JSON files (unified_10s_enhanced)"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write unified narrative JSON files"
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
    logger.info("IMAVB Pass 3: Sequential Narrative Unification (single-shot global merge)")
    logger.info("=" * 80)
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Input dir (Pass 2): {args.pass2_dir}")
    logger.info(f"Output dir: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_videos = get_videos_to_process(args.pass2_dir, args.output_dir)

    if args.video_ids_file:
        subset_ids = set()
        for f in args.video_ids_file.split(","):
            with open(f.strip()) as fh:
                subset_ids.update(json.load(fh))
        all_videos = [v for v in all_videos if v["video_id"] in subset_ids]
        logger.info(f"Filtered to {len(all_videos)} videos from --video-ids-file")

    pass2_count = (
        len([f for f in os.listdir(args.pass2_dir) if f.endswith(".json")])
        if os.path.isdir(args.pass2_dir) else 0
    )
    processed_count = (
        len([f for f in os.listdir(args.output_dir) if f.endswith(".json")])
        if os.path.isdir(args.output_dir) else 0
    )

    logger.info(
        f"Pass 2 outputs: {pass2_count} | Already unified: {processed_count} | "
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
        gpu_memory_utilization=0.90,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=16,
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
    print(f"[PASS3] Starting with {NUM_WORKERS} concurrent workers", flush=True)

    async def process_video_async(video_info, semaphore, pbar, results):
        async with semaphore:
            video_id = video_info['video_id']
            try:
                result = await process_video(
                    video_info, engine, tokenizer, sampling_params,
                    args.output_dir, logger
                )
                if result:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
            except Exception as e:
                results['failed'] += 1
                print(f"[PASS3] Exception {video_id}: {e}", flush=True)
            finally:
                pbar.update(1)

    async def run_all():
        semaphore = asyncio.Semaphore(NUM_WORKERS)
        results = {'successful': 0, 'failed': 0}
        pbar = tqdm(total=len(assigned_videos), desc="Pass 3: Merge")
        tasks = [process_video_async(v, semaphore, pbar, results) for v in assigned_videos]
        await asyncio.gather(*tasks)
        pbar.close()
        return results['successful'], results['failed']

    successful, failed = asyncio.run(run_all())

    logger.info("=" * 80)
    logger.info(f"Pass 3 complete! Successful: {successful}, Failed: {failed}")
    logger.info(f"Final outputs in: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
