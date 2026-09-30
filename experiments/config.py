"""
Shared configuration for IMAVB experiment scripts.

All paths are placeholders — set them before running any script.
"""

# ── Paths (SET THESE BEFORE RUNNING) ─────────────────────────────────────
HIDDEN_STATES_ROOT = "<SET_PATH>/hidden_states"            # .pt files per model per split
BASELINE_OUTPUT_ROOT = "<SET_PATH>/lmms_eval_output/catA"  # A1 JSONL baseline results
LM_WEIGHTS_ROOT = "<SET_PATH>/lm_weights"                  # norm_weights.pt + lm_head_weights.pt per model
DATASET_NAME = "ngqtrung/IMAVB"                             # HuggingFace dataset (public)
OUTPUT_DIR = "<SET_PATH>/analysis_outputs"                 # Where results are saved

# ── Models ────────────────────────────────────────────────────────────────
MODELS = [
    "baichuan_omni",
    "minicpm_o",
    "ola",
    "omnivinci",
    "qwen2_5_omni",
    "uni_moe_2_omni",
    "video_salmonn_2",
    "qwen3_omni",
]

PRETTY_NAMES = {
    "baichuan_omni": "Baichuan-Omni-1.5",
    "minicpm_o": "MiniCPM-o 2.6",
    "ola": "OLA",
    "omnivinci": "OmniVinci",
    "qwen2_5_omni": "Qwen2.5-Omni",
    "uni_moe_2_omni": "Uni-MoE-2.0-Omni",
    "video_salmonn_2": "Video-SALMONN-2",
    "qwen3_omni": "Qwen3-Omni",
}

# ── Splits ────────────────────────────────────────────────────────────────
SPLITS = ["standard_vision", "standard_audio", "misleading_vision", "misleading_audio"]

SHORT_SPLIT_NAMES = {
    "standard_vision": "std_v",
    "standard_audio": "std_a",
    "misleading_vision": "mis_v",
    "misleading_audio": "mis_a",
}

# ── Per-model peak probe layers (from 4-fold CV probing, §4.3) ───────────
PEAK_LAYERS = {
    "baichuan_omni": 2,
    "minicpm_o": 17,
    "ola": 14,
    "omnivinci": 14,
    "qwen2_5_omni": 17,
    "uni_moe_2_omni": 16,
    "video_salmonn_2": 15,
    "qwen3_omni": 30,
}

# ── PGLA probe layers
PGLA_PEAK_LAYERS = {
    "baichuan_omni": 2,
    "minicpm_o": 17,
    "ola": 14,
    "omnivinci": 14,
    "qwen2_5_omni": 17,
    "uni_moe_2_omni": 16,
    "video_salmonn_2": 15,
    "qwen3_omni": 30,
}

# ── Per-model hidden dimensions ──────────────────────────────────────────
HIDDEN_DIMS = {
    "baichuan_omni": 3584,
    "minicpm_o": 3584,
    "ola": 3584,
    "omnivinci": 3584,
    "qwen2_5_omni": 3584,
    "uni_moe_2_omni": 3584,
    "video_salmonn_2": 3584,
    "qwen3_omni": 2048,
}

# ── Per-model layer counts ───────────────────────────────────────────────
NUM_LAYERS = {
    "baichuan_omni": 28,
    "minicpm_o": 28,
    "ola": 29,
    "omnivinci": 29,
    "qwen2_5_omni": 28,
    "uni_moe_2_omni": 29,
    "video_salmonn_2": 29,
    "qwen3_omni": 48,
}

# ── Shared constants ─────────────────────────────────────────────────────
RANDOM_STATE = 42
N_SAMPLES = 2000         # Total samples per model (500 per split)
N_BOOTSTRAP = 10_000     # Bootstrap resamples for 95% CIs
