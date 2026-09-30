# Hidden State Extraction

Extracts last-token hidden states from all transformer decoder layers for every
benchmark sample and saves them as `.pt` files. These tensors are used for the
linear probing and logit lens analyses described in the paper.

For each sample in the IMAVB benchmark, the script:

1. Loads a model via one of the eight model adapters in `model_adapters.py`
2. Prepares inputs: 50 uniformly sampled video frames + full audio at 16 kHz mono
3. Runs a forward/generate pass with forward hooks registered on every decoder
   layer via `register_forward_hook()` (per paper Appendix J)
4. Captures the last-token hidden state `h[:, -1, :]` from each layer during the
   prefill pass
5. Saves one `.pt` file per sample with shape `(num_layers, hidden_dim)`

### Output format

Each `.pt` file contains a dict:

```python
{
    "hidden_states": torch.Tensor(num_layers, hidden_dim),  # float32, CPU
    "metadata": {
        "video_id": str,
        "split": str,          # e.g. "standard_vision"
        "model": str,          # e.g. "qwen2_5_omni"
        "correct_answer": str, # "A"-"F"
        "is_misleading": bool, # True for misleading_* splits
        "should_reject": bool, # True when correct_answer in {E, F}
        "question": str,
    }
}
```

Files are organized as:
```
{HIDDEN_STATES_ROOT}/{model}/{split}/{video_id}.pt
```

Also saves `norm_weights.pt` and `lm_head_weights.pt` to `{model}/` for logit
lens projections (Appendix J: "Logit lens projections apply RMSNorm followed by
the LM head at each layer").

---

## How to run

```bash
# Basic — extracts all 4 splits, 500 samples each
python extract_hidden_states.py --model qwen2_5_omni

# Specific splits only
python extract_hidden_states.py --model baichuan_omni \
    --splits standard_vision misleading_vision

# Custom output directory
python extract_hidden_states.py --model ola \
    --output_dir ./outputs/hidden_states/ola

# Custom HuggingFace checkpoint
python extract_hidden_states.py --model qwen3_omni \
    --pretrained Qwen/Qwen3-Omni-30B-A3B-Instruct

# Video-SALMONN-2 with LoRA checkpoint
python extract_hidden_states.py --model video_salmonn_2 \
    --lora_ckpt tsinghua-ee/video-SALMONN-2_plus_7B

# Limit samples for testing
python extract_hidden_states.py --model minicpm_o --limit 10
```
