## Overview

| Step | Script | Model | Access |
|------|--------|-------|--------|
| Pass 1 — vision captions | `vision_captioning.py` | GPT-4o | Azure OpenAI API |
| Pass 1 — audio captions | `audio_captioning.py` | Qwen3-Omni-30B-A3B-Captioner | vLLM (GPU) |
| Pass 1 — omni captions | `omni_captioning.py` | Qwen3-Omni-30B-A3B-Thinking | vLLM (GPU) |
| Pass 2 — detail enhancement | `pass2_enhancement.py` | Qwen3.5-27B | vLLM (GPU) |
| Pass 3 — narrative unification | `pass3_merge.py` | Qwen3.5-27B | vLLM (GPU) |
| QA generation | `qa_generation.py` | Qwen3.5-27B | vLLM (GPU) |

---

## Pipeline Execution Order

Run the six steps in sequence. Each step reads the output of the previous step.

### Step 1a — Vision Captions (GPT-4o, Azure API)

```bash
python vision_captioning.py \
  --video-dir <SET_PATH>/videos \
  --output-dir <SET_PATH>/captions/level1_vision
```

Produces one JSON per video containing per-segment vision-only captions.
GPT-4o receives 10 frames (1 fps) per 10-second segment.

Prompt (paper Appendix H, "Level-1 Vision Caption"):
```
Describe this 10-second video clip.

Focus on visible content only: the people, actions, objects,
setting, and any notable visual changes. Write a short,
concrete caption.
```

### Step 1b — Audio Captions (Qwen3-Omni-30B-A3B-Captioner, vLLM)

```bash
python audio_captioning.py \
  --video-dir <SET_PATH>/videos \
  --vision-caption-dir <SET_PATH>/captions/level1_vision \
  --output-dir <SET_PATH>/captions/level1_audio
```

Produces per-segment audio-only captions. The model receives only the mono
audio track for each 10-second segment; no explicit text prompt is used
(paper Appendix H, "Level-1 Audio Caption").

### Step 1c — Omni Captions (Qwen3-Omni-30B-A3B-Thinking, vLLM)

```bash
python omni_captioning.py \
  --video-dir <SET_PATH>/videos \
  --vision-caption-dir <SET_PATH>/captions/level1_vision \
  --audio-caption-dir <SET_PATH>/captions/level1_audio \
  --segments-dir <SET_PATH>/segments \
  --output-dir <SET_PATH>/captions/unified_10s_raw
```

Produces per-segment omnimodal captions. Each segment receives both 10 video
frames and audio, plus the previous segment's caption as continuity context.
Output: `<output-dir>/<video_id>.json` with a list of per-segment captions.

### Step 2 — Detail Enhancement (Qwen3.5-27B, vLLM)

```bash
python pass2_enhancement.py \
  --pass1-dir <SET_PATH>/captions/unified_10s_raw \
  --vision-dir <SET_PATH>/captions/level1_vision \
  --audio-dir <SET_PATH>/captions/level1_audio \
  --output-dir <SET_PATH>/captions/unified_10s_enhanced
```

Fuses the three caption streams per segment. Trust hierarchy:
Omni > Vision > Audio (paper §3.2).
Output: one JSON per video with enhanced per-segment captions.

### Step 3 — Narrative Unification (Qwen3.5-27B, vLLM)

```bash
python pass3_merge.py \
  --pass2-dir <SET_PATH>/captions/unified_10s_enhanced \
  --output-dir <SET_PATH>/captions/unified_final
```

Merges all per-segment enhanced captions into a single deduplicated,
timestamped narrative per video.
Output: one JSON per video with a `unified_caption` string.

### Step 4 — QA Generation (Qwen3.5-27B, vLLM)

```bash
python qa_generation.py \
  --caption-dir <SET_PATH>/captions/unified_final \
  --output-dir <SET_PATH>/qa_benchmark \
  --tensor-parallel-size 8
```

Generates four question variants per video from the unified caption.
Dry-run mode (processes N samples with verbose output):

```bash
python qa_generation.py \
  --caption-dir <SET_PATH>/captions/unified_final \
  --output-dir <SET_PATH>/qa_benchmark \
  --dryrun 5
```

Multi-node sharding (split work across P processes):

```bash
python qa_generation.py \
  --caption-dir <SET_PATH>/captions/unified_final \
  --output-dir <SET_PATH>/qa_benchmark \
  --max-processes P \
  --process-index I   # 0 .. P-1
```

---

## Input / Output Flow

```
videos/
  <video_id>.mp4
        |
        v
[Step 1a] vision_captioning.py      (GPT-4o, 10 frames/segment)
        |
        v
captions/level1_vision/<video_id>.json
        |
[Step 1b] audio_captioning.py       (Qwen3-Omni-Captioner, audio only)
        |
        v
captions/level1_audio/<video_id>.json
        |
[Step 1c] omni_captioning.py        (Qwen3-Omni-Thinking, frames+audio+context)
        |
        v
captions/unified_10s_raw/<video_id>.json   — list of per-segment omni captions
        |
[Step 2]  pass2_enhancement.py      (Qwen3.5-27B, fuse three streams)
        |
        v
captions/unified_10s_enhanced/<video_id>.json   — enhanced per-segment captions
        |
[Step 3]  pass3_merge.py            (Qwen3.5-27B, global merge)
        |
        v
captions/unified_final/<video_id>.json   — single unified_caption string per video
        |
[Step 4]  qa_generation.py          (Qwen3.5-27B, 2×2 QA design)
        |
        v
qa_benchmark/<video_id>.json   — 4 question variants with choices and metadata
```

---

## QA Output Schema

Each `qa_benchmark/<video_id>.json` contains:

```jsonc
{
  "video_id": "...",
  "shared_intro": "Brief description of the video",
  "visual_element": {
    "correct_detail": "...",   // real visual detail from caption
    "wrong_detail": "...",     // swapped-in wrong detail for Q_mis_v
    "timestamp_range": "[Xs-Ys]",
    "question_focus": "..."    // one of 8 reasoning categories
  },
  "audio_element": { ... },   // same structure as visual_element
  "variants": {
    "Q_std_v": { "question": "...", "type": "vision_standard",   "correct_answer": "A-D", ... },
    "Q_mis_v": { "question": "...", "type": "vision_misleading",  "correct_answer": null,  ... },
    "Q_std_a": { "question": "...", "type": "audio_standard",    "correct_answer": "A-D", ... },
    "Q_mis_a": { "question": "...", "type": "audio_misleading",   "correct_answer": null,  ... }
  },
  "vision_choices": { "A": "...", "B": "...", "C": "...", "D": "...", "E": "The visual detail in the question is incorrect", "F": "The audio detail in the question is incorrect" },
  "audio_choices":  { "A": "...", "B": "...", "C": "...", "D": "...", "E": "...", "F": "..." },
  "correct_answer": "A-D",
  "vision_answer_timestamp": "[Xs-Ys]",
  "audio_answer_timestamp": "[Xs-Ys]",
  "vision_misleading": { "category": "...", "description": "..." },
  "audio_misleading":  { "category": "...", "description": "..." }
}
```

For misleading variants (`Q_mis_v`, `Q_mis_a`), the correct response is E or F
respectively — these are auto-appended to every choice set and are not among the
A–D options generated by the model.

---

## Script Descriptions

| Script | Description |
|--------|-------------|
| `vision_captioning.py` | Calls GPT-4o via Azure OpenAI API to produce vision-only captions for each 10-second segment at 1 fps. |
| `audio_captioning.py` | Runs Qwen3-Omni-30B-A3B-Captioner via vLLM to produce audio-only captions per segment. |
| `omni_captioning.py` | Runs Qwen3-Omni-30B-A3B-Thinking via vLLM AsyncLLMEngine for omnimodal per-segment captions with continuity context. |
| `pass2_enhancement.py` | Runs Qwen3.5-27B to fuse the three caption streams per segment using the Omni > Vision > Audio trust hierarchy. |
| `pass3_merge.py` | Runs Qwen3.5-27B to merge all enhanced segment captions into one unified, deduplicated, timestamped narrative per video. |
| `qa_generation.py` | Runs Qwen3.5-27B to generate four QA variants per video (Q_std_v, Q_mis_v, Q_std_a, Q_mis_a) with programmatic misleading-swap enforcement and resume support. |
