# Probing Experiments

## Scripts

### `linear_probing.py`

Trains logistic regression probes on hidden states to detect misleading inputs.

**Protocol (matches paper exactly):**
- 4-fold stratified CV (`StratifiedKFold(n_splits=4, shuffle=True, random_state=42)`)
- `LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)`
- `StandardScaler` fit on train fold, applied to test fold
- Binary target: 0 = standard (std_v + std_a), 1 = misleading (mis_v + mis_a)
- Peak layer l\* = layer with highest mean CV accuracy (global probe)

**Probes:**
- **Global probe**: all 2,000 samples per model, all layers → identifies peak layer l\*
- **Vision probe**: all standard (1,000) vs mis_v (500) at l\* — 1:2 class ratio
- **Audio probe**: all standard (1,000) vs mis_a (500) at l\* — 1:2 class ratio
- **TF-IDF baseline**: `TfidfVectorizer(max_features=5000)` fit per fold, vision and audio questions separately

**Expected baseline values (paper §4.2):** TF-IDF ~73.4% (vision), ~71.4% (audio).

**Usage:**
```bash
# Single model
python linear_probing.py --model qwen2_5_omni

# All 8 models
python linear_probing.py --model all

# Custom output path
python linear_probing.py --model all --output /path/to/results.json
```

**Output:** JSON file with per-layer CV accuracies, peak layer, modality-specific probe
results, and TF-IDF baselines. Saved to `OUTPUT_DIR/linear_probing.json` by default
(see `../config.py`). Intermediate results are saved after each model, so the run
can be resumed if interrupted.

### `residualized_probing.py`

Residualized probe analysis (Appendix — Table 6). Projects out text-predictive
features via Ridge regression + orthogonal projection (nested 4-fold CV) then
retrains the probe on residual hidden states to confirm genuine cross-modal signal.

**Usage:**
```bash
python residualized_probing.py [--models MODEL1,MODEL2] [--trajectory] [--full-bootstrap]
```
