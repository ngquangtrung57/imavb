"""
LLM-as-Judge verification for A5 (BinExplain) outputs (Appendix F).

Evaluates whether each model's TRUE/FALSE prediction and its accompanying
explanation are both correct, using Qwen3.5-27B in non-thinking mode via
vLLM AsyncLLMEngine.

Metrics (Table tab:judge in paper):
  P-Acc  — prediction accuracy (same as A2 binary accuracy)
  E-Acc  — explanation accuracy (judge says reasoning is correct)
  R+R    — both prediction and explanation correct
  R+W    — right prediction, wrong explanation (dissociation signal)

All 8 IMAVB models are evaluated. 16,000 entries total (8 × 2,000).

Features:
  - Incremental JSONL save: each result flushed immediately
  - Resume/continue: skips already-completed entries on rerun
  - Failed tracking: failed_ids.json retried automatically on rerun

Usage:
    python verify_binary_explain.py                           # full run
    python verify_binary_explain.py --models baichuan_omni   # one model
    python verify_binary_explain.py --dry-run                # 5 entries/model
    python verify_binary_explain.py --tp 4                   # custom TP
    python verify_binary_explain.py --input-base /path/to/catA --output-base /path/to/out
"""

import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

# ── shared config ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BASELINE_OUTPUT_ROOT, MODELS, OUTPUT_DIR, PRETTY_NAMES

# ── judge model ────────────────────────────────────────────────────────
JUDGE_MODEL = "Qwen/Qwen3.5-27B"
MAX_CONCURRENT = 72

# Default output: sibling of BASELINE_OUTPUT_ROOT so paths stay together
_DEFAULT_OUTPUT_BASE = str(Path(OUTPUT_DIR) / "judge_results")
_DEFAULT_INPUT_BASE = BASELINE_OUTPUT_ROOT


# =============================================================================
# Unique key for resume logic
# =============================================================================

def entry_key(model_name: str, entry: dict) -> str:
    """Stable unique key for an entry (used to track completion)."""
    acc = entry.get("accuracy", {})
    video_id = acc.get("video_id", "")
    doc_id = entry.get("doc_id", -1)
    modality = acc.get("modality", "unknown")
    return f"{model_name}|{video_id}|{doc_id}|{modality}"


# =============================================================================
# Step 1: Programmatic extraction check (fast, no LLM needed)
# =============================================================================

def verify_extraction_programmatic(entry: dict) -> dict:
    """Check whether the model's pred matches its explanation text."""
    acc = entry.get("accuracy", {})
    explanation = (acc.get("explanation") or "").strip()
    pred = (acc.get("pred") or "").upper()

    if not explanation:
        return {"correct": False, "reason": "empty_response", "needs_llm": False}
    if not pred:
        return {"correct": False, "reason": "empty_pred", "needs_llm": False}

    first_word = explanation.split()[0].strip(".,!:;\"'()").upper()
    if first_word in ("TRUE", "FALSE"):
        if first_word == pred:
            return {"correct": True, "reason": "matches_first_word", "needs_llm": False}
        return {"correct": False, "reason": f"first_word={first_word}_but_pred={pred}", "needs_llm": True}

    first_sentence = explanation.split(".")[0].upper()
    has_true = "TRUE" in first_sentence
    has_false = "FALSE" in first_sentence

    if has_true and not has_false:
        if pred == "TRUE":
            return {"correct": True, "reason": "true_in_first_sentence", "needs_llm": False}
        return {"correct": False, "reason": f"first_sentence_says_TRUE_but_pred={pred}", "needs_llm": True}
    if has_false and not has_true:
        if pred == "FALSE":
            return {"correct": True, "reason": "false_in_first_sentence", "needs_llm": False}
        return {"correct": False, "reason": f"first_sentence_says_FALSE_but_pred={pred}", "needs_llm": True}

    return {"correct": None, "reason": "ambiguous", "needs_llm": True}


# =============================================================================
# Step 2: Judge prompt
# =============================================================================

def build_explanation_judge_prompt(entry: dict) -> str:
    """Build judge prompt for a single A5 entry."""
    acc = entry.get("accuracy", {})
    question = acc.get("question", "")
    gold = acc.get("gold", "")
    pred = acc.get("pred", "")
    explanation = acc.get("explanation", "")
    modality = acc.get("modality", "unknown")
    misleading_cat = acc.get("misleading_category", "N/A")

    if gold == "FALSE":
        ground_truth_context = (
            f"GROUND TRUTH: The description in the question is INACCURATE (FALSE).\n"
            f"The question contains a misleading {modality} premise of type: {misleading_cat}.\n"
            f"This means the {modality} event described in the question does NOT actually occur in the video."
        )
    else:
        ground_truth_context = (
            f"GROUND TRUTH: The description in the question is ACCURATE (TRUE).\n"
            f"The {modality} event described in the question actually occurs in the video."
        )

    return (
        f"You are a judge evaluating whether a model's explanation for a TRUE/FALSE question is correct.\n"
        f"\n"
        f"TASK: A model was shown a video and asked whether a description in a question matches "
        f"what's in the video. The model answered {pred} and gave an explanation. "
        f"You need to judge if the explanation is correct.\n"
        f"\n"
        f"QUESTION GIVEN TO MODEL:\n"
        f'"{question}"\n'
        f"\n"
        f"{ground_truth_context}\n"
        f"\n"
        f"MODEL'S RESPONSE:\n"
        f'"{explanation}"\n'
        f"\n"
        f"EVALUATION CRITERIA:\n"
        f'1. extraction_correct: Does the model\'s response clearly state "{pred}" as its answer? (true/false)\n'
        f"2. explanation_correct: Is the model's reasoning valid and consistent with the ground truth? Consider:\n"
        f"   - If ground truth is FALSE (misleading): Does the model correctly identify that the described "
        f"event is inaccurate or doesn't occur? The model doesn't need to identify the exact misleading "
        f"category, but its reasoning should align with the ground truth.\n"
        f"   - If ground truth is TRUE (accurate): Does the model correctly validate that the described event occurs?\n"
        f"   - A model that gives the right answer (TRUE/FALSE) but with wrong reasoning should be marked "
        f"as explanation_correct=false.\n"
        f"   - A model that gives the wrong answer but explains something reasonable about the content "
        f"should still be marked as explanation_correct=false (since the conclusion is wrong).\n"
        f"   - If the explanation is empty or nonsensical, mark explanation_correct=false.\n"
        f"\n"
        f"Respond ONLY with a JSON object. Keep judge_reasoning under 15 words:\n"
        f'{{"extraction_correct": true/false, "explanation_correct": true/false, "judge_reasoning": "short reason"}}'
    )


# =============================================================================
# Step 3: JSON response parsing
# =============================================================================

def _validate_judge_json(parsed: object) -> bool:
    if not isinstance(parsed, dict):
        return False
    if "extraction_correct" not in parsed or "explanation_correct" not in parsed:
        return False
    ec = parsed["extraction_correct"]
    xc = parsed["explanation_correct"]
    if ec is not None and not isinstance(ec, bool):
        return False
    if xc is not None and not isinstance(xc, bool):
        return False
    return True


def _try_fix_truncated_json(text: str) -> dict | None:
    """Recover from truncated JSON (missing closing brace/quote)."""
    if not text.startswith("{"):
        idx = text.find("{")
        if idx == -1:
            return None
        text = text[idx:]

    for suffix in ['"}', '}', '"', '"}}', '}}']:
        try:
            parsed = json.loads(text + suffix)
            if _validate_judge_json(parsed):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    ec_match = re.search(r'"extraction_correct"\s*:\s*(true|false|null)', text, re.IGNORECASE)
    xc_match = re.search(r'"explanation_correct"\s*:\s*(true|false|null)', text, re.IGNORECASE)
    if ec_match and xc_match:
        def _to_bool(s: str) -> bool | None:
            s = s.lower()
            if s == "true":
                return True
            if s == "false":
                return False
            return None
        jr_match = re.search(r'"judge_reasoning"\s*:\s*"([^"]*)', text)
        reasoning = (jr_match.group(1) if jr_match else "truncated") + " [truncated]"
        return {
            "extraction_correct": _to_bool(ec_match.group(1)),
            "explanation_correct": _to_bool(xc_match.group(1)),
            "judge_reasoning": reasoning,
        }
    return None


def parse_judge_response(text: str) -> dict:
    """Extract structured JSON from judge response, handling markdown/truncation."""
    if not text or not text.strip():
        return {
            "extraction_correct": None,
            "explanation_correct": None,
            "judge_reasoning": "PARSE_ERROR: empty_response",
        }

    text = text.strip()

    try:
        parsed = json.loads(text)
        if _validate_judge_json(parsed):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if md_match:
        try:
            parsed = json.loads(md_match.group(1))
            if _validate_judge_json(parsed):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group())
            if _validate_judge_json(parsed):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group())
            if _validate_judge_json(parsed):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    recovered = _try_fix_truncated_json(text)
    if recovered:
        return recovered

    return {
        "extraction_correct": None,
        "explanation_correct": None,
        "judge_reasoning": "PARSE_ERROR: " + text[:200],
    }


# =============================================================================
# Step 4: Data loading
# =============================================================================

def load_model_entries(model_name: str, input_base: str) -> list[dict]:
    """Load all A5 binary_explain JSONL entries for a model (latest timestamp)."""
    model_dir = Path(input_base) / model_name / "binary_explain"
    if not model_dir.exists():
        print(f"  [SKIP] {model_name}: directory not found at {model_dir}")
        return []

    jsonl_files = sorted(model_dir.rglob("*.jsonl"))
    if not jsonl_files:
        print(f"  [SKIP] {model_name}: no JSONL files found")
        return []

    # Use latest timestamp prefix
    timestamps: set[str] = set()
    for f in jsonl_files:
        match = re.match(r"(\d{8}_\d{6})_", f.name)
        if match:
            timestamps.add(match.group(1))

    if timestamps:
        latest_ts = sorted(timestamps)[-1]
        jsonl_files = [f for f in jsonl_files if f.name.startswith(latest_ts)]
        print(f"  Using latest timestamp: {latest_ts} ({len(jsonl_files)} files)")
    else:
        print(f"  No timestamp pattern found, using all {len(jsonl_files)} files")

    entries: list[dict] = []
    for jsonl_file in jsonl_files:
        with open(jsonl_file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "accuracy" in entry and "explanation" in entry.get("accuracy", {}):
                        entries.append(entry)
                    else:
                        print(f"  [WARN] {jsonl_file.name}:{line_num}: missing accuracy/explanation fields")
                except json.JSONDecodeError as e:
                    print(f"  [WARN] {jsonl_file.name}:{line_num}: JSON parse error: {e}")

    return entries


# =============================================================================
# Step 5: Resume logic
# =============================================================================

def load_completed_keys(output_base: str) -> dict:
    """Load keys of already-completed entries from previous runs."""
    completed: dict = {}
    out_base = Path(output_base)
    if not out_base.exists():
        return completed

    for jsonl_file in out_base.rglob("judge_results.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    key = f"{r['model']}|{r['video_id']}|{r['doc_id']}|{r['modality']}"
                    has_valid_result = (
                        r.get("extraction_correct") is not None
                        and r.get("explanation_correct") is not None
                        and "error" not in r
                        and not str(r.get("judge_reasoning", "")).startswith("PARSE_ERROR")
                    )
                    if has_valid_result:
                        completed[key] = r
                except (json.JSONDecodeError, KeyError):
                    continue

    return completed


def load_failed_keys(output_base: str) -> set[str]:
    failed_path = Path(output_base) / "failed_ids.json"
    if not failed_path.exists():
        return set()
    try:
        with open(failed_path) as f:
            data = json.load(f)
        return set(data.get("failed_keys", []))
    except (json.JSONDecodeError, KeyError):
        return set()


def save_failed_keys(failed_keys: set[str], output_base: str) -> None:
    out_path = Path(output_base) / "failed_ids.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "failed_keys": sorted(failed_keys),
                "count": len(failed_keys),
                "updated": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )


# =============================================================================
# Step 6: Incremental JSONL writer
# =============================================================================

class IncrementalWriter:
    """Append-mode per-model JSONL writer with immediate flush."""

    def __init__(self, output_base: str) -> None:
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)
        self._handles: dict = {}
        self._counts: dict = {}

    def write(self, result: dict) -> None:
        model = result["model"]
        if model not in self._handles:
            out_dir = self.output_base / model
            out_dir.mkdir(parents=True, exist_ok=True)
            self._handles[model] = open(out_dir / "judge_results.jsonl", "a")
            self._counts[model] = 0
        self._handles[model].write(json.dumps(result) + "\n")
        self._handles[model].flush()
        self._counts[model] += 1

    def close(self) -> None:
        for h in self._handles.values():
            h.close()
        self._handles.clear()

    def get_counts(self) -> dict:
        return dict(self._counts)


# =============================================================================
# Step 7: Async vLLM inference
# =============================================================================

async def run_judge_async(
    pending_items: list[tuple],
    args: argparse.Namespace,
) -> tuple[list[dict], set[str]]:
    """Run judge using vLLM AsyncLLMEngine with bounded concurrency."""
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

    engine_args = AsyncEngineArgs(
        model=JUDGE_MODEL,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
        dtype="auto",
        enable_chunked_prefill=True,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    print(f"Loading tokenizer for {JUDGE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
    )

    print(f"Applying chat template to {len(pending_items)} prompts ...")
    tokenized_prompts: list[str] = []
    for _, _, entry, _ in tqdm(pending_items, desc="Tokenizing"):
        prompt_text = build_explanation_judge_prompt(entry)
        messages = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokenized_prompts.append(formatted)

    writer = IncrementalWriter(args.output_base)
    semaphore = asyncio.Semaphore(args.max_concurrent)
    results: list[dict] = []
    failed_keys: set[str] = set()
    progress = tqdm(total=len(pending_items), desc="Judging")

    async def process_one(idx: int) -> None:
        key, model_name, entry, prog_result = pending_items[idx]
        prompt = tokenized_prompts[idx]
        acc = entry.get("accuracy", {})
        request_id = f"judge-{uuid.uuid4().hex[:12]}"

        async with semaphore:
            try:
                final_output = None
                async for request_output in engine.generate(prompt, sampling_params, request_id):
                    final_output = request_output

                if final_output is None or not final_output.outputs:
                    raise RuntimeError("Empty output from engine")

                result_text = final_output.outputs[0].text.strip()
                parsed = parse_judge_response(result_text)

                is_parse_error = str(parsed.get("judge_reasoning", "")).startswith("PARSE_ERROR")
                result = {
                    "key": key,
                    "video_id": acc.get("video_id", ""),
                    "doc_id": entry.get("doc_id", -1),
                    "model": model_name,
                    "modality": acc.get("modality", "unknown"),
                    "category": acc.get("category", "unknown"),
                    "is_misleading": acc.get("is_misleading", False),
                    "misleading_category": acc.get("misleading_category", "N/A"),
                    "gold": acc.get("gold", ""),
                    "pred": acc.get("pred", ""),
                    "original_correct": acc.get("overall", 0),
                    "explanation": acc.get("explanation", ""),
                    "question": acc.get("question", ""),
                    "extraction_correct": parsed.get("extraction_correct"),
                    "explanation_correct": parsed.get("explanation_correct"),
                    "judge_reasoning": parsed.get("judge_reasoning", ""),
                    "raw_judge_response": result_text,
                    "programmatic_extraction": prog_result,
                    "status": "parse_error" if is_parse_error else "success",
                }
                if is_parse_error:
                    failed_keys.add(key)

            except Exception as e:
                result = {
                    "key": key,
                    "video_id": acc.get("video_id", ""),
                    "doc_id": entry.get("doc_id", -1),
                    "model": model_name,
                    "modality": acc.get("modality", "unknown"),
                    "category": acc.get("category", "unknown"),
                    "is_misleading": acc.get("is_misleading", False),
                    "misleading_category": acc.get("misleading_category", "N/A"),
                    "gold": acc.get("gold", ""),
                    "pred": acc.get("pred", ""),
                    "original_correct": acc.get("overall", 0),
                    "explanation": acc.get("explanation", ""),
                    "question": acc.get("question", ""),
                    "extraction_correct": None,
                    "explanation_correct": None,
                    "judge_reasoning": "",
                    "raw_judge_response": "",
                    "programmatic_extraction": prog_result,
                    "error": str(e),
                    "status": "error",
                }
                failed_keys.add(key)

            finally:
                writer.write(result)
                results.append(result)
                progress.update(1)

    tasks = [asyncio.create_task(process_one(i)) for i in range(len(pending_items))]
    await asyncio.gather(*tasks)

    progress.close()
    writer.close()

    if hasattr(engine, "shutdown_background_loop"):
        engine.shutdown_background_loop()
    elif hasattr(engine, "shutdown"):
        engine.shutdown()

    return results, failed_keys


# =============================================================================
# Step 8: Summary (P-Acc, E-Acc, R+R, R+W per model)
# =============================================================================

def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "N/A"
    return f"{num / denom * 100:.1f}%"


def generate_summary(
    all_results: list[dict],
    run_timestamp: str,
    output_base: str,
) -> dict:
    """Compute P-Acc, E-Acc, R+R, R+W per model and save JSON + Markdown."""
    out_base = Path(output_base)
    out_base.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    for r in all_results:
        model = r["model"]
        if model not in summary:
            summary[model] = {
                "total": 0,
                "errors": 0,
                "parse_errors": 0,
                "pred_correct": 0,
                "extraction_correct": 0,
                "explanation_correct": 0,
                # paper metrics
                "r_plus_r": 0,  # right pred + right explanation
                "r_plus_w": 0,  # right pred + wrong explanation
                "w_plus_r": 0,
                "w_plus_w": 0,
                "by_modality": {},
                "by_misleading_cat": {},
                "by_category": {},
            }

        s = summary[model]
        s["total"] += 1

        status = r.get("status", "success")
        if status == "error":
            s["errors"] += 1
            continue
        if status == "parse_error":
            s["parse_errors"] += 1
            continue

        pred_ok = bool(r.get("original_correct"))
        ext_ok = r.get("extraction_correct") is True
        exp_ok = r.get("explanation_correct") is True

        if pred_ok:
            s["pred_correct"] += 1
        if ext_ok:
            s["extraction_correct"] += 1
        if exp_ok:
            s["explanation_correct"] += 1

        if pred_ok and exp_ok:
            s["r_plus_r"] += 1
        elif pred_ok and not exp_ok:
            s["r_plus_w"] += 1
        elif not pred_ok and exp_ok:
            s["w_plus_r"] += 1
        else:
            s["w_plus_w"] += 1

        # by_modality
        mod = r.get("modality", "unknown")
        if mod not in s["by_modality"]:
            s["by_modality"][mod] = {"total": 0, "pred_correct": 0, "extraction_correct": 0, "explanation_correct": 0}
        m = s["by_modality"][mod]
        m["total"] += 1
        if pred_ok:
            m["pred_correct"] += 1
        if ext_ok:
            m["extraction_correct"] += 1
        if exp_ok:
            m["explanation_correct"] += 1

        # by_misleading_cat
        mcat = r.get("misleading_category", "N/A")
        if mcat not in s["by_misleading_cat"]:
            s["by_misleading_cat"][mcat] = {"total": 0, "pred_correct": 0, "extraction_correct": 0, "explanation_correct": 0}
        mc = s["by_misleading_cat"][mcat]
        mc["total"] += 1
        if pred_ok:
            mc["pred_correct"] += 1
        if ext_ok:
            mc["extraction_correct"] += 1
        if exp_ok:
            mc["explanation_correct"] += 1

        # by_category
        cat = r.get("category", "unknown")
        if cat not in s["by_category"]:
            s["by_category"][cat] = {"total": 0, "pred_correct": 0, "extraction_correct": 0, "explanation_correct": 0}
        c = s["by_category"][cat]
        c["total"] += 1
        if pred_ok:
            c["pred_correct"] += 1
        if ext_ok:
            c["extraction_correct"] += 1
        if exp_ok:
            c["explanation_correct"] += 1

    with open(out_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON summary: {out_base / 'summary.json'}")

    md_lines = _generate_markdown(summary, all_results, run_timestamp)
    with open(out_base / "summary.md", "w") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown summary: {out_base / 'summary.md'}")

    return summary


def _generate_markdown(
    summary: dict,
    all_results: list[dict],
    run_timestamp: str,
) -> list[str]:
    n_total = len(all_results)
    n_success = sum(1 for r in all_results if r.get("status") == "success")
    n_errors = sum(1 for r in all_results if r.get("status") == "error")
    n_parse = sum(1 for r in all_results if r.get("status") == "parse_error")

    lines = [
        "# LLM-as-Judge Results (Appendix F)",
        "",
        f"**Judge model:** {JUDGE_MODEL} (non-thinking mode, AsyncLLMEngine)",
        f"**Run timestamp:** {run_timestamp}",
        f"**Total entries:** {n_total} (success: {n_success}, parse_error: {n_parse}, error: {n_errors})",
        "",
        "## Per-Model Summary (paper Table tab:judge)",
        "",
        "| Model | Total | P-Acc | E-Acc | R+R | R+W | Errors | Parse Errors |",
        "|-------|------:|:-----:|:-----:|:---:|:---:|-------:|:------------:|",
    ]

    for model_name in MODELS:
        s = summary.get(model_name)
        if s is None:
            continue
        t = s["total"]
        pretty = PRETTY_NAMES.get(model_name, model_name)
        lines.append(
            f"| {pretty} | {t} "
            f"| {_pct(s['pred_correct'], t)} "
            f"| {_pct(s['explanation_correct'], t)} "
            f"| {_pct(s['r_plus_r'], t)} "
            f"| {_pct(s['r_plus_w'], t)} "
            f"| {s['errors']} | {s['parse_errors']} |"
        )
    lines.append("")

    # Aggregate row
    agg_t = sum(s["total"] for s in summary.values())
    agg_p = sum(s["pred_correct"] for s in summary.values())
    agg_e = sum(s["explanation_correct"] for s in summary.values())
    agg_rr = sum(s["r_plus_r"] for s in summary.values())
    agg_rw = sum(s["r_plus_w"] for s in summary.values())
    lines += [
        "## Aggregate",
        "",
        f"- Total entries: {agg_t}",
        f"- Overall P-Acc: {_pct(agg_p, agg_t)}",
        f"- Overall E-Acc: {_pct(agg_e, agg_t)}",
        f"- Overall R+R: {_pct(agg_rr, agg_t)}",
        f"- Overall R+W: {_pct(agg_rw, agg_t)}",
        "",
    ]

    # By modality
    all_mods = sorted({m for s in summary.values() for m in s["by_modality"]})
    lines += ["## By Modality", ""]
    for mod in all_mods:
        lines += [
            f"### {mod.capitalize()}",
            "",
            "| Model | Total | P-Acc | E-Acc |",
            "|-------|------:|:-----:|:-----:|",
        ]
        for model_name in MODELS:
            m = summary.get(model_name, {}).get("by_modality", {}).get(mod)
            if m:
                t = m["total"]
                pretty = PRETTY_NAMES.get(model_name, model_name)
                lines.append(f"| {pretty} | {t} | {_pct(m['pred_correct'], t)} | {_pct(m['explanation_correct'], t)} |")
        lines.append("")

    return lines


# =============================================================================
# Step 9: Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge verification for A5 binary_explain outputs (Appendix F)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Models to evaluate (default: all 8 from config.py)",
    )
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT,
        help=f"Max in-flight requests (default: {MAX_CONCURRENT})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Use only 5 entries per model")
    parser.add_argument(
        "--input-base",
        default=_DEFAULT_INPUT_BASE,
        help="Root dir for A5 binary_explain JSONL outputs",
    )
    parser.add_argument("--output-base", default=_DEFAULT_OUTPUT_BASE)
    parser.add_argument("--no-resume", action="store_true", help="Rerun from scratch")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graph capture")
    parser.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help="Disable custom all-reduce kernel",
    )
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 60)
    print("  LLM-as-Judge: Binary Explain Verification (Appendix F)")
    print("=" * 60)
    print(f"  Judge model:      {JUDGE_MODEL}")
    print(f"  Tensor parallel:  {args.tp} GPUs")
    print(f"  Max model len:    {args.max_model_len}")
    print(f"  Max concurrent:   {args.max_concurrent}")
    print(f"  Models:           {', '.join(args.models)}")
    print(f"  Input dir:        {args.input_base}")
    print(f"  Output dir:       {args.output_base}")
    print(f"  Dry run:          {args.dry_run}")
    print(f"  Resume:           {not args.no_resume}")
    print(f"  Timestamp:        {run_timestamp}")
    print("=" * 60)

    # Resume logic
    if not args.no_resume:
        print("\nLoading previous results for resume ...")
        completed = load_completed_keys(args.output_base)
        prev_failed = load_failed_keys(args.output_base)
        print(f"  Previously completed: {len(completed)}")
        print(f"  Previously failed (will retry): {len(prev_failed)}")
    else:
        completed = {}
        prev_failed = set()
        out_base = Path(args.output_base)
        for jsonl_file in out_base.rglob("judge_results.jsonl"):
            jsonl_file.unlink()
            print(f"  Removed: {jsonl_file}")

    # Load entries
    all_entries: list[tuple] = []
    completed_results = list(completed.values())

    for model_name in args.models:
        print(f"\n--- Loading {model_name} ({PRETTY_NAMES.get(model_name, model_name)}) ---")
        entries = load_model_entries(model_name, args.input_base)
        if not entries:
            continue

        if args.dry_run:
            entries = entries[:5]
            print(f"  [DRY RUN] Using {len(entries)} entries")

        n_skip = 0
        n_pending = 0
        for entry in entries:
            key = entry_key(model_name, entry)
            prog_result = verify_extraction_programmatic(entry)
            if key in completed:
                n_skip += 1
                continue
            all_entries.append((key, model_name, entry, prog_result))
            n_pending += 1

        print(f"  Total: {len(entries)}, Skip (completed): {n_skip}, Pending: {n_pending}")

    if not all_entries:
        print("\nAll entries already completed.")
        if completed_results:
            generate_summary(completed_results, run_timestamp, output_base=args.output_base)
        return

    print(f"\n{'='*60}")
    print(f"  Pending prompts:  {len(all_entries)}")
    print(f"  Already done:     {len(completed_results)}")
    print(f"{'='*60}")

    start_time = time.time()
    new_results, new_failed = asyncio.run(run_judge_async(all_entries, args))
    elapsed = time.time() - start_time

    n_success = sum(1 for r in new_results if r.get("status") == "success")
    n_errors = sum(1 for r in new_results if r.get("status") == "error")
    n_parse = sum(1 for r in new_results if r.get("status") == "parse_error")

    print(f"\n{'='*60}")
    print(f"  Inference time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  New results:    {len(new_results)} (success: {n_success}, parse_error: {n_parse}, error: {n_errors})")
    if elapsed > 0:
        print(f"  Throughput:     {len(new_results) / elapsed:.1f} entries/sec")
    print(f"{'='*60}")

    save_failed_keys(new_failed, args.output_base)
    if new_failed:
        print(f"\nFailed entries: {len(new_failed)} — rerun to retry automatically.")

    generate_summary(completed_results + new_results, run_timestamp, output_base=args.output_base)


if __name__ == "__main__":
    main()
