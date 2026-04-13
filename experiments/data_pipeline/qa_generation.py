#!/usr/bin/env python3
"""
IMAVB QA Generation — single-pass generation.

Generates 4 question variants per video in a 2x2 design:
  - Q_std_v  : standard vision question (correct premise, visible answer)
  - Q_mis_v  : misleading vision question (one wrong visual detail in premise)
  - Q_std_a  : standard audio question (correct premise, audible answer)
  - Q_mis_a  : misleading audio question (one wrong audio detail in premise)

Each question has a PREMISE describing a specific moment and a QUERY about one
detail from that moment. Standard variants use correct premises (answer = A-D).
Misleading variants copy the standard question and swap exactly one premise
detail; the correct answer is E or F (auto-appended to every choice set).

Model: Qwen3.5-27B via vLLM AsyncLLMEngine
Paper: §3.3 "Question Design"
"""

import os
import sys
import json
import re
import argparse
import logging
import asyncio
import random
from typing import List, Dict, Any, Optional

from tqdm import tqdm
from vllm import SamplingParams
from vllm import AsyncLLMEngine, AsyncEngineArgs
from transformers import AutoTokenizer


###################################################################################################
# Settings
###################################################################################################

# Model
MODEL_PATH = "Qwen/Qwen3.5-27B"
MAX_MODEL_LEN = 48000

# Sampling parameters
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.8
GEN_TOP_K = 20
GEN_PRESENCE_PENALTY = 1.5
GEN_MAX_TOKENS = 8000

# Concurrency
NUM_WORKERS = 64

# Auto-appended choices for misleading questions (paper §3.3)
CHOICE_E = "The visual detail in the question is incorrect"
CHOICE_F = "The audio detail in the question is incorrect"


###################################################################################################
# Categories (paper §3.3)
###################################################################################################

# 9 vision misleading subcategories
VISION_MISLEADING_CATEGORIES = [
    "person_identity",
    "person_appearance",
    "person_action",
    "person_position",
    "object_type",
    "object_attribute",
    "object_location",
    "location_setting",
    "location_detail",
]

# 9 audio misleading subcategories
AUDIO_MISLEADING_CATEGORIES = [
    "speech_content",
    "speech_speaker",
    "speech_tone",
    "speech_context",
    "sound_type",
    "sound_source",
    "sound_intensity",
    "background_music",
    "ambient_sound",
]

# 8 reasoning categories assigned per question
QUESTION_FOCUS_CATEGORIES = [
    "temporal", "causal", "plot", "cross_modality",
    "emotional", "time_order", "existence", "scene_description",
]


###################################################################################################
# Prompts — Generator (paper §3.3 and Appendix H)
###################################################################################################

SYSTEM_PROMPT = """\
You create benchmark questions to test whether video understanding models truly \
watch the video or just guess from text patterns.

QUESTION FORMAT — each question is one sentence with two parts:
  PREMISE (describes a specific scene moment) + QUESTION (asks one detail)

Example: "When the man in the red jacket sits down at the table, what does he pick up first?"
         ^^^^^^^^^^^^^^^^ premise ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^ question ^^^^^^^^^^^^

THE 4 VARIANTS you must create:
  Q_std_v — correct premise + vision question (answer = something you SEE)
  Q_mis_v — premise with ONE wrong visual detail + SAME vision question
  Q_std_a — correct premise + audio question (answer = something you HEAR)
  Q_mis_a — premise with ONE wrong audio detail + SAME audio question

RULES (violating ANY of these makes the output invalid):

1) MODALITY — the question part determines the modality.
   Vision = answer is visible: gesture, position, object, expression, movement, color, clothing.
   Audio  = answer is audible: spoken words, sounds, music, tone of voice, volume.
   The premise CAN mention both audio+visual to set context.
   WRONG vision Q: "what does he shout?" → shouting is audio.
   WRONG audio Q: "what gesture does he make?" → gesture is visual.

2) MISLEADING = SIMPLE SWAP.
   Write Q_std first, then COPY it to Q_mis and change exactly ONE detail in the premise.
   The question part (everything after the premise) must be WORD-FOR-WORD IDENTICAL.
   Do NOT rewrite the sentence. Do NOT add or remove words in the question part.

   Good:
     Q_std_v: "When the man in the RED jacket sits at the table, what does he pick up?"
     Q_mis_v: "When the man in the BLUE jacket sits at the table, what does he pick up?"
     (Only RED→BLUE changed. Rest is identical.)

   Bad — question part changed:
     Q_std_a: "After the glass shatters, what sound cuts through the air?"
     Q_mis_a: "After the glass shatters, what soft piano melody cuts through the air?"
     (WRONG: "sound" was replaced with "soft piano melody" in the question part.)

3) PREMISE SPECIFICITY — must pinpoint ONE moment.
   Include: timestamp [Xs-Ys], character-identifying details, specific action.
   Bad: "When the man walks" — too vague.
   Good: "At [20s-30s], when the tall man in the gray suit pauses at the doorway"

4) TIMESTAMPS — use [Xs-Ys] from the caption's 10-second segments.
   answer_timestamp = the exact segment where the ANSWER appears in the caption.
   Double-check against the caption text.

5) CHOICES — all 4 choices (A/B/C/D) must be UNIQUE and DIFFERENT from each other.
   No two choices can have the same text. The correct answer must directly match
   something stated in the caption — do not ask for details the caption does not describe.

6) ANSWER GROUNDING — the correct answer must be a fact DIRECTLY stated in the caption.
   Do NOT ask about details the caption does not mention (e.g., do not ask "what color"
   if the caption only says "large" without naming a color).

7) NO ANSWER IN PREMISE — the standard question's premise must NOT contain the answer.
   The premise sets the scene; the question asks for a DIFFERENT detail from that moment.
   Bad: premise "man in red jacket" + question "what color is his jacket?" (answer in premise)
   Good: premise "man in red jacket sits at table" + question "what does he pick up?" """


QA_GENERATION_PROMPT = """\
=== VIDEO CAPTION ===
{unified_caption}
=== END CAPTION ===

Create 4 question variants. Correct answer at position {correct_position}.
Vision misleading category (pick one): {vision_categories}
Audio misleading category (pick one): {audio_categories}

CRITICAL RULES — your output will be rejected if any are violated:
- Q_mis_v must be a COPY of Q_std_v with exactly ONE detail swapped in the premise. \
The question part must be word-for-word identical.
- Q_mis_a must be a COPY of Q_std_a with exactly ONE detail swapped in the premise. \
The question part must be word-for-word identical.
- Vision questions: the answer must be something VISIBLE. Never ask about sounds/speech.
- Audio questions: the answer must be something AUDIBLE. Never ask about appearance/position.
- All 4 choices (A/B/C/D) must have UNIQUE, DIFFERENT text. No duplicates.
- The correct answer must be a fact DIRECTLY stated in the caption.

Output ONLY valid JSON:
{{
  "shared_intro": "Brief video description",
  "visual_element": {{
    "correct_detail": "the real visual detail from caption",
    "wrong_detail": "the swapped-in wrong visual detail",
    "timestamp_range": "[Xs-Ys]"
  }},
  "audio_element": {{
    "correct_detail": "the real audio detail from caption",
    "wrong_detail": "the swapped-in wrong audio detail",
    "timestamp_range": "[Xs-Ys]"
  }},
  "Q_std_v": "premise with correct visual detail + vision question",
  "Q_mis_v": "SAME sentence with ONE visual detail swapped",
  "Q_std_a": "premise with correct audio detail + audio question",
  "Q_mis_a": "SAME sentence with ONE audio detail swapped",
  "vision_choices": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "audio_choices": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "correct_answer": "{correct_position}",
  "vision_answer_timestamp": "[Xs-Ys]",
  "audio_answer_timestamp": "[Xs-Ys]",
  "vision_misleading": {{"category": "from list", "description": "what was swapped"}},
  "audio_misleading": {{"category": "from list", "description": "what was swapped"}}
}}"""


###################################################################################################
# Utilities
###################################################################################################

def setup_logging(output_dir: str, verbose: bool = False) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "qa_generation.log")
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8", mode="a"),
        ],
    )
    return logging.getLogger(__name__)


def get_videos_to_process(
    caption_dir: str, output_dir: str, max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    if not os.path.isdir(caption_dir):
        logging.error(f"Caption directory not found: {caption_dir}")
        return []

    processed_ids: set = set()
    if os.path.isdir(output_dir):
        processed_ids = {
            os.path.splitext(f)[0]
            for f in os.listdir(output_dir)
            if f.endswith(".json")
        }

    all_videos = []
    for filename in os.listdir(caption_dir):
        if not filename.endswith(".json"):
            continue
        video_id = os.path.splitext(filename)[0]
        if video_id in processed_ids:
            continue
        filepath = os.path.join(caption_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            caption = data.get("unified_caption", "")
            if not caption:
                continue
            all_videos.append({
                "video_id": video_id,
                "filepath": filepath,
                "unified_caption": caption,
                "video_duration": data.get("video_duration", 0),
                "num_segments": data.get("num_segments", 0),
            })
        except Exception as e:
            logging.warning(f"Error loading {filename}: {e}")

    all_videos.sort(key=lambda x: x["video_id"])
    if max_samples and len(all_videos) > max_samples:
        all_videos = all_videos[:max_samples]
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


def sanitize_json_string(json_str: str) -> str:
    result = []
    in_string = False
    escape_next = False
    for char in json_str:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
        if char == "\\" and in_string:
            result.append(char)
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            result.append(char)
            continue
        if in_string:
            if char == "\n":
                result.append("\\n")
            elif char == "\r":
                result.append("\\r")
            elif char == "\t":
                result.append("\\t")
            elif ord(char) < 32:
                result.append(f"\\u{ord(char):04x}")
            else:
                result.append(char)
        else:
            result.append(char)
    return "".join(result)


def parse_json_response(response: str) -> Dict[str, Any]:
    start_idx = response.find("{")
    if start_idx == -1:
        raise ValueError("No JSON found in response")

    brace_count = 0
    end_idx = -1
    in_string = False
    escape_next = False
    for i, char in enumerate(response[start_idx:], start=start_idx):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break

    if end_idx == -1:
        json_str = response[start_idx:] + "}" * brace_count
    else:
        json_str = response[start_idx:end_idx + 1]

    for strategy in [
        lambda s: json.loads(s),
        lambda s: json.loads(sanitize_json_string(s)),
        lambda s: json.loads(re.sub(r",\s*([}\]])", r"\1", sanitize_json_string(s))),
    ]:
        try:
            return strategy(json_str)
        except (json.JSONDecodeError, re.error):
            continue

    raise ValueError(f"Failed to parse JSON. First 200 chars: {json_str[:200]}")


def validate_qa_structure(qa: Dict[str, Any]) -> bool:
    """Check all required fields exist and basic structure is valid."""
    required = [
        "Q_std_v", "Q_mis_v", "Q_std_a", "Q_mis_a",
        "vision_choices", "audio_choices", "correct_answer",
        "vision_misleading", "audio_misleading",
    ]
    for field in required:
        if field not in qa:
            return False
    for key in ["vision_choices", "audio_choices"]:
        if not all(k in qa.get(key, {}) for k in ["A", "B", "C", "D"]):
            return False
    if qa.get("correct_answer") not in ["A", "B", "C", "D"]:
        return False
    if qa.get("Q_std_v") == qa.get("Q_mis_v"):
        return False
    if qa.get("Q_std_a") == qa.get("Q_mis_a"):
        return False
    # No duplicate choices within a set
    for key in ["vision_choices", "audio_choices"]:
        vals = list(qa.get(key, {}).values())
        if len(vals) != len(set(vals)):
            return False
    return True


def format_categories(cats: List[str]) -> str:
    return ", ".join(cats)


def programmatic_swap(qa: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce identical question parts by replacing correct_detail with wrong_detail
    in Q_std to produce Q_mis. Falls back to model-generated Q_mis when the
    correct_detail substring is not found verbatim in Q_std.
    """
    qa = dict(qa)

    # Vision swap
    ve = qa.get("visual_element", {})
    correct_v = ve.get("correct_detail", "")
    wrong_v = ve.get("wrong_detail", "")
    if correct_v and wrong_v and correct_v in qa.get("Q_std_v", ""):
        qa["Q_mis_v"] = qa["Q_std_v"].replace(correct_v, wrong_v, 1)

    # Audio swap
    ae = qa.get("audio_element", {})
    correct_a = ae.get("correct_detail", "")
    wrong_a = ae.get("wrong_detail", "")
    if correct_a and wrong_a and correct_a in qa.get("Q_std_a", ""):
        qa["Q_mis_a"] = qa["Q_std_a"].replace(correct_a, wrong_a, 1)

    return qa


###################################################################################################
# Core: LLM call via vLLM AsyncLLMEngine
###################################################################################################

async def llm_generate(
    engine: AsyncLLMEngine,
    tokenizer: Any,
    system: str,
    user: str,
    sampling_params: SamplingParams,
    request_id: str,
) -> str:
    """Single LLM generation call through the async engine."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    response = ""
    async for out in engine.generate(formatted, sampling_params, request_id):
        if out.finished:
            response = out.outputs[0].text.strip()
    return response


###################################################################################################
# Generate step
###################################################################################################

async def generate_qa(
    engine: AsyncLLMEngine,
    tokenizer: Any,
    gen_params: SamplingParams,
    video_id: str,
    caption: str,
    correct_position: str,
    vision_cats: List[str],
    audio_cats: List[str],
    req_suffix: str = "",
) -> Optional[Dict[str, Any]]:
    """Run the generator and return a parsed, validated QA dict, or None on failure."""
    prompt = QA_GENERATION_PROMPT.format(
        unified_caption=caption,
        correct_position=correct_position,
        vision_categories=format_categories(vision_cats),
        audio_categories=format_categories(audio_cats),
    )
    rid = f"{video_id}_gen{req_suffix}"
    raw = await llm_generate(engine, tokenizer, SYSTEM_PROMPT, prompt, gen_params, rid)
    try:
        qa = parse_json_response(raw)
        qa = programmatic_swap(qa)
        if not validate_qa_structure(qa):
            return None
        return qa
    except Exception:
        return None


###################################################################################################
# Process one video
###################################################################################################

async def process_video(
    video_info: Dict[str, Any],
    engine: AsyncLLMEngine,
    tokenizer: Any,
    gen_params: SamplingParams,
    logger: logging.Logger,
    output_dir: str,
    verbose: bool = False,
) -> bool:
    video_id = video_info["video_id"]
    caption = video_info["unified_caption"]

    correct_position = random.choice(["A", "B", "C", "D"])
    vision_cats = random.sample(VISION_MISLEADING_CATEGORIES, k=3)
    audio_cats = random.sample(AUDIO_MISLEADING_CATEGORIES, k=3)
    question_focus = random.choice(QUESTION_FOCUS_CATEGORIES)

    qa = await generate_qa(
        engine, tokenizer, gen_params, video_id, caption,
        correct_position, vision_cats, audio_cats,
    )
    if qa is None:
        logger.warning(f"{video_id}: Generation failed (no valid JSON)")
        return False

    if verbose:
        print(f"\n{'='*60}\n[{video_id}] GENERATED\n{'='*60}")
        print(f"  Q_std_v: {qa.get('Q_std_v', '')[:120]}")
        print(f"  Q_mis_v: {qa.get('Q_mis_v', '')[:120]}")
        print(f"  Q_std_a: {qa.get('Q_std_a', '')[:120]}")
        print(f"  Q_mis_a: {qa.get('Q_mis_a', '')[:120]}")

    # Attach question_focus metadata to element dicts
    visual_element = qa.get("visual_element", {})
    visual_element["question_focus"] = question_focus
    audio_element = qa.get("audio_element", {})
    audio_element["question_focus"] = question_focus

    # Build output record.
    # vision_choices and audio_choices each have A-D from the model; E and F are auto-appended
    # per paper §3.3: "options E and F are appended automatically."
    output_data = {
        "video_id": video_id,
        "video_duration": video_info.get("video_duration", 0),
        "num_segments": video_info.get("num_segments", 0),
        "shared_intro": qa.get("shared_intro", ""),
        "visual_element": visual_element,
        "audio_element": audio_element,
        "variants": {
            "Q_std_v": {
                "question": qa["Q_std_v"],
                "type": "vision_standard",
                "premise": "correct",
                "correct_answer": qa["correct_answer"],
                "answer_timestamp": qa.get("vision_answer_timestamp", ""),
            },
            "Q_mis_v": {
                "question": qa["Q_mis_v"],
                "type": "vision_misleading",
                "premise": "wrong",
                "correct_answer": None,   # correct response is E (auto-appended)
                "answer_timestamp": qa.get("vision_answer_timestamp", ""),
                "misleading_category": qa.get("vision_misleading", {}).get("category", ""),
                "misleading_description": qa.get("vision_misleading", {}).get("description", ""),
            },
            "Q_std_a": {
                "question": qa["Q_std_a"],
                "type": "audio_standard",
                "premise": "correct",
                "correct_answer": qa["correct_answer"],
                "answer_timestamp": qa.get("audio_answer_timestamp", ""),
            },
            "Q_mis_a": {
                "question": qa["Q_mis_a"],
                "type": "audio_misleading",
                "premise": "wrong",
                "correct_answer": None,   # correct response is F (auto-appended)
                "answer_timestamp": qa.get("audio_answer_timestamp", ""),
                "misleading_category": qa.get("audio_misleading", {}).get("category", ""),
                "misleading_description": qa.get("audio_misleading", {}).get("description", ""),
            },
        },
        "vision_choices": {**qa["vision_choices"], "E": CHOICE_E, "F": CHOICE_F},
        "audio_choices": {**qa["audio_choices"], "E": CHOICE_E, "F": CHOICE_F},
        "correct_answer": qa["correct_answer"],
        "vision_answer_timestamp": qa.get("vision_answer_timestamp", ""),
        "audio_answer_timestamp": qa.get("audio_answer_timestamp", ""),
        "vision_misleading": qa.get("vision_misleading", {}),
        "audio_misleading": qa.get("audio_misleading", {}),
        "requested_correct_position": correct_position,
    }

    output_path = os.path.join(output_dir, f"{video_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    logger.info(f"{video_id}: saved to {output_path}")
    return True


###################################################################################################
# Main
###################################################################################################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IMAVB QA Generation — single-pass")
    p.add_argument(
        "--caption-dir",
        default="<SET_PATH>",
        help="Directory of unified caption JSON files (output of Pass 3)",
    )
    p.add_argument(
        "--output-dir",
        default="<SET_PATH>",
        help="Directory where per-video QA JSON files are written",
    )
    p.add_argument("--model", default=MODEL_PATH, help="HuggingFace model name or local path")
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Process at most N videos (default: all)")
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                   help="Max concurrent async generation tasks")
    p.add_argument("--max-processes", type=int, default=1,
                   help="Total number of parallel processes (for multi-node sharding)")
    p.add_argument("--process-index", type=int, default=0,
                   help="Index of this process (0-based)")
    p.add_argument("--dryrun", type=int, default=0,
                   help="Process only N samples with verbose output (0 = off)")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    verbose = args.dryrun > 0
    output_dir = args.output_dir
    if verbose:
        output_dir = os.path.join(args.output_dir, "dryrun")

    logger = setup_logging(output_dir, verbose=verbose)
    logger.info("=" * 70)
    logger.info("IMAVB QA Generation — single-pass")
    logger.info("=" * 70)
    logger.info(f"Model      : {args.model}")
    logger.info(f"Caption dir: {args.caption_dir}")
    logger.info(f"Output dir : {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    max_samples = args.dryrun if args.dryrun > 0 else args.max_samples
    all_videos = get_videos_to_process(args.caption_dir, output_dir, max_samples)

    total_captions = (
        len([f for f in os.listdir(args.caption_dir) if f.endswith(".json")])
        if os.path.isdir(args.caption_dir) else 0
    )
    already_done = (
        len([f for f in os.listdir(output_dir) if f.endswith(".json")])
        if os.path.isdir(output_dir) else 0
    )
    logger.info(
        f"Total captions: {total_captions} | Already done: {already_done} | "
        f"Remaining: {len(all_videos)}"
    )

    assigned = split_for_process(
        all_videos, max(1, args.max_processes), args.process_index
    )
    logger.info(f"This process: {len(assigned)} videos")

    if not assigned:
        logger.info("Nothing to do.")
        return

    # --- Initialize vLLM engine ---
    logger.info(f"Loading vLLM engine (tp={args.tensor_parallel_size}) ...")
    engine_args = AsyncEngineArgs(
        model=args.model,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.95,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=48,
        max_model_len=args.max_model_len,
        seed=42,
        enable_chunked_prefill=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    gen_params = SamplingParams(
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        top_k=GEN_TOP_K,
        presence_penalty=GEN_PRESENCE_PENALTY,
        max_tokens=GEN_MAX_TOKENS,
    )
    logger.info("Engine ready.")

    # --- Run async processing ---
    num_workers = min(args.num_workers, len(assigned))
    if verbose:
        num_workers = 1  # sequential for readability

    async def run_one(
        video_info: Dict[str, Any],
        semaphore: asyncio.Semaphore,
        pbar: tqdm,
        results: Dict[str, int],
    ) -> None:
        async with semaphore:
            try:
                ok = await process_video(
                    video_info, engine, tokenizer, gen_params,
                    logger, output_dir, verbose=verbose,
                )
                if ok:
                    results["ok"] += 1
                else:
                    results["fail"] += 1
            except Exception as e:
                results["fail"] += 1
                logger.error(f"{video_info['video_id']}: {e}")
            finally:
                pbar.update(1)

    async def run_all() -> Dict[str, int]:
        sem = asyncio.Semaphore(num_workers)
        results: Dict[str, int] = {"ok": 0, "fail": 0}
        pbar = tqdm(total=len(assigned), desc="QA generation")
        tasks = [run_one(v, sem, pbar, results) for v in assigned]
        await asyncio.gather(*tasks)
        pbar.close()
        return results

    results = asyncio.run(run_all())

    logger.info("=" * 70)
    logger.info(f"Done. OK: {results['ok']} | Failed: {results['fail']}")
    logger.info("=" * 70)

    if verbose:
        print(f"\n{'='*70}")
        print("DRYRUN SUMMARY")
        print(f"{'='*70}")
        for fname in sorted(os.listdir(output_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(output_dir, fname)) as fh:
                d = json.load(fh)
            print(f"  {d['video_id']}: {d['variants']['Q_std_v']['question'][:80]}")


if __name__ == "__main__":
    main(parse_args())
