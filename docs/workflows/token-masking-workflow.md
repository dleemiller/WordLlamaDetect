# Token Masking Workflow

This document describes the complete workflow for training a model with token masking applied to zero-weight under-represented tokens.

## Overview

Token masking allows you to zero-weight tokens with insufficient training representation, based on the hypothesis that under-represented tokens may contribute unreliable signals.

**Key Points**:
- Mask is created **before training** from token frequency counts
- Masked tokens are initialized to weight=0 during model initialization
- Masked tokens can still learn during training (they start at 0 but are not frozen)
- Impact on accuracy should be validated empirically

## Complete Workflow

### Step 1: Count Tokens in Training Data

```bash
uv run python scripts/analysis/token_frequency_counter.py \
  --training-config configs/training/gemma3-27b.yaml \
  --output artifacts/token_counts.npy \
  --batch-size 512 \
  --num-workers 4
```

**Output**: `artifacts/token_counts.npy` - shape (vocab_size,), dtype int64

**What it does**:
- Loads the training dataset specified in the config
- Tokenizes all examples
- Counts occurrences of each token
- Saves counts as numpy array

**Example output**:
```
Using vocab_size=262208 from embeddings (tokenizer reports 262145)
Loading dataset laurievb/OpenLID-v2...
  Train: 124902000 examples
Filtering dataset to 148 languages...
  Kept 121798076 / 124902000 examples
Counting tokens: 100%|██████████| 237887/237887 [1:23:45<00:00, 47.32it/s]
Saved token counts to artifacts/token_counts.npy
  (shape=(262208, 1), dtype=int64, total_tokens=4,945,738,333)
```

### Step 2: Create Weight Mask

```bash
uv run python scripts/analysis/create_weight_mask.py \
  --token-counts artifacts/token_counts.npy \
  --threshold 10 \
  --output artifacts/token_mask.npy
```

**Output**: `artifacts/token_mask.npy` - shape (vocab_size,), dtype bool

**Parameters**:
- `--threshold`: Minimum token count (tokens below this are masked)
  - Recommended: 10 (tokens appearing <10 times are likely noise)
  - Conservative: 5
  - Aggressive: 50

**What it does**:
- Loads token counts
- Creates boolean mask: True = keep (count >= threshold), False = zero weight
- Saves mask as numpy array
- Reports statistics on masked vs kept tokens

**Example output**:
```
Loading token counts from artifacts/token_counts.npy...
  Vocab size:    262,208
  Total tokens:  4,945,738,333

Threshold: 10 occurrences
  Tokens kept (count >= 10):  185,432 (70.72%)
  Tokens masked (count < 10): 76,776 (29.28%)

Masked token statistics:
  Total occurrences: 123,456
  % of training data: 0.0025%
  Mean count: 1.61
  Max count:  9

Kept token statistics:
  Total occurrences: 4,945,614,877
  % of training data: 99.9975%
  Mean count: 26,672.4
  Min count:  10

Saving mask to artifacts/token_mask.npy...
  Saved: 256.5 KB
```

### Step 3: Train Model with Mask

Update your training config to apply the mask:

```yaml
# configs/training/gemma3-27b.yaml
training:
  batch_size: 512
  learning_rate: 0.001
  epochs: 1

  # Token masking
  token_mask_path: artifacts/token_mask.npy  # Add this line
```

Then train:

```bash
uv run wldetect train --config configs/training/gemma3-27b.yaml
```

**What happens during training**:
1. Model loads the token mask
2. Initializes `token_weights` to 1.0 for all tokens
3. Sets `token_weights[masked_tokens] = 0.0` (where mask is False)
4. Training proceeds normally
5. Masked tokens can still learn (they start at 0 but are trainable parameters)

**Training output (Step 5)**:
```
Step 5: Initialize model
  Converting embeddings to torch tensor...
  Embeddings tensor: torch.Size([262208, 2304]), 1.99 GB
  Loading token mask from artifacts/token_mask.npy...
    Mask shape: torch.Size([262208])
    Tokens to zero: 76,776 (29.28%)
    Tokens to train: 185,432
  Model parameters: 341,032
```

### Step 4: Evaluate and Save E3M4 Lookup Table

This happens automatically at the end of training:

```
Step 8: Save artifacts
Step 8b: Generate fp8 lookup tables
  Saving E4M3FN format (backward compatibility)...
  E4M3FN saved: lookup_table_fp8_e4m3fn.safetensors (37.8 MB)
  Saving E3M4 format (30% better precision)...
  E3M4 saved: lookup_table_fp8_e3m4.safetensors (37.8 MB)
```

### Step 5: Analyze Results

After training, compare FLORES results with and without masking:

```bash
# Baseline (no masking)
uv run wldetect eval --model-path artifacts/baseline/

# With masking
uv run wldetect eval --model-path artifacts/masked/
```

Compare `flores_dev_results.json` to see impact on accuracy.

## Configuration Options

### Threshold Selection

The `--threshold` parameter determines how aggressively to filter tokens:

| Threshold | Tokens Masked (%) | Use Case |
|-----------|-------------------|----------|
| 1 | ~5% | Only zero tokens with 0 occurrences |
| 5 | ~15% | Conservative filtering |
| 10 | ~30% | Recommended default |
| 50 | ~50% | Aggressive filtering |
| 100 | ~60% | Very aggressive |

**Recommendation**: Start with threshold=10, then experiment with 5 and 50 to find optimal trade-off.

### Optional: Visualize Mask Coverage

```bash
uv run python scripts/analysis/visualize_mask.py \
  --token-counts artifacts/token_counts.npy \
  --token-mask artifacts/token_mask.npy \
  --output artifacts/mask_coverage.png
```

This shows:
- Cumulative distribution of masked vs kept tokens
- % of training data represented by masked tokens
- Validation that masked tokens represent <0.01% of training data

## Expected Impact

Based on the hypothesis that under-represented tokens contribute unreliable signals:

**Expected**:
- Minimal accuracy impact (<0.5% drop) because masked tokens represent <0.01% of training data
- Possibly improved accuracy on rare languages by reducing noise
- Faster training (fewer effective parameters)

**Validation Required**:
- Compare FLORES accuracy with/without masking
- Check per-language metrics to ensure no language is disproportionately affected
- Verify masked tokens are indeed noise (not legitimate rare language tokens)

## Troubleshooting

### Too Many Tokens Masked

**Symptom**: >50% of tokens masked

**Cause**: Threshold too high or dataset too small

**Fix**:
- Lower threshold (try 5 instead of 10)
- Increase training data
- Check token_counts.npy statistics

### Significant Accuracy Drop

**Symptom**: >1% accuracy drop on FLORES

**Cause**: Masking removed important tokens

**Fix**:
- Lower threshold
- Analyze which tokens were masked (create visualization)
- Check if masked tokens include language-specific characters

### Mask Shape Mismatch

**Symptom**: `Token mask shape doesn't match vocab_size`

**Cause**: Token counts generated with wrong vocab size (using tokenizer length instead of embeddings shape)

**Fix**:
- Regenerate token_counts.npy with updated script (uses embeddings.shape[0])
- Ensure mask is created from correct token counts

## Files and Artifacts

| File | Description | Size |
|------|-------------|------|
| `artifacts/token_counts.npy` | Token frequency counts (vocab_size,) | ~2 MB |
| `artifacts/token_mask.npy` | Boolean mask (vocab_size,) | ~256 KB |
| `artifacts/lookup_table_fp8_e3m4.safetensors` | Quantized lookup table with masking applied | ~38 MB |
| `artifacts/flores_dev_results.json` | Per-language FLORES evaluation metrics | ~50 KB |

## Next Steps

- [Weight Distribution Analysis](../analysis/weight-distribution.md) - Analyze learned weights
- [Training Guide](../training.md) - Complete training documentation
- [Architecture](../architecture.md) - Understand token weighting mechanism
