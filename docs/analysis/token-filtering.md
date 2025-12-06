# Token Filtering Analysis

This document explains the token filtering strategy for WLDetect.

## Overview

**Token filtering** zero-weights tokens with insufficient training representation. The hypothesis is that under-represented tokens may contribute spurious signals.

## Motivation

### The Problem

Not all tokens in the vocabulary appear equally in training data:

- **Common tokens**: Appear millions of times (e.g., "the", "is", "и")
- **Rare tokens**: Appear < 10 times or never
- **Distribution**: Highly skewed (power law)

**Issue**: Tokens with few occurrences have limited signal for learning language associations.

### The Solution

Set `token_weights[rare_tokens] = 0` to exclude them from language detection:

```python
# Identify rare tokens
zero_weight_mask = token_counts < min_count_threshold

# Zero their weights
filtered_token_weights = token_weights.copy()
filtered_token_weights[zero_weight_mask] = 0.0

# Compute lookup table with filtered weights
lookup_table = (embeddings * filtered_token_weights) @ projection.T + bias
```

**Result**: Only tokens with sufficient training representation contribute to predictions.

## Analysis Workflow

### 1. Count Token Frequencies

Count how many times each token appears in training data:

```bash
uv run python scripts/analysis/token_frequency_counter.py \
  --config configs/training/default.yaml \
  --output artifacts/token_counts.npy
```

**Output**: `token_counts.npy` - shape (vocab_size,) with count per token

### 2. Analyze Weight Distribution

Correlate token weights with token frequencies:

```bash
uv run python scripts/analysis/weight_analysis.py \
  --checkpoint artifacts/gemma3-27b/checkpoints/checkpoint_epoch_1.pt \
  --token-counts artifacts/token_counts.npy \
  --output-dir artifacts/ \
  --threshold 10
```

**Outputs**:
- `token_weight_frequency_analysis.png` - 6-panel visualization
- `token_filtering_recommendations.txt` - Threshold recommendations

### 3. Visualizations

The analysis generates 6 plots:

#### Plot 1: Scatter (Log-Log)
- X-axis: log(token count + 1)
- Y-axis: token weight
- Colors: tokens below threshold (red) vs above (blue)

**Insight**: Identifies correlation between frequency and weight

#### Plot 2: Token Count Distribution
- Histogram of log10(count + 1)
- Red line: suggested threshold

**Insight**: Shows vocabulary distribution (most tokens are rare)

#### Plot 3: Weight Distribution
- Histogram of learned token weights

**Insight**: Shows if weights are concentrated or spread out

#### Plot 4: Coverage Curve (Pareto)
- Cumulative token coverage by frequency rank
- Shows what % of tokens cover 95%, 99% of training data

**Insight**: Token frequency follows a power law distribution

#### Plot 5: Weight by Frequency Bin
- Bar chart: average weight for tokens in bins (0, 1-9, 10-99, etc.)

**Insight**: Shows relationship between token frequency and average weight

#### Plot 6: Outlier Detection
- Scatter with highlighted outliers (high weight + low frequency)

**Insight**: Shows tokens with unusual weight-frequency relationships

### 4. Create Filtered Model

Generate E3M4 lookup table with filtered weights:

```bash
uv run python scripts/analysis/create_filtered_e3m4_model.py \
  --checkpoint artifacts/gemma3-27b/checkpoints/checkpoint_epoch_1.pt \
  --embeddings artifacts/embeddings/embeddings_a26b8f6b3226_150langs.safetensors \
  --token-counts artifacts/token_counts.npy \
  --output artifacts/gemma3-27b/lookup_table_fp8_e3m4_filtered.safetensors \
  --threshold 10
```

**Output**: E3M4 lookup table with zero-weighted rare tokens

## Threshold Selection

### Typical Thresholds

| Threshold | Tokens Zeroed | Use Case |
|-----------|---------------|----------|
| 0 | None | Baseline (no filtering) |
| 1 | ~30-40% | Aggressive (zero tokens never seen) |
| 10 | ~50-60% | **Recommended** (balance noise vs coverage) |
| 100 | ~70-80% | Conservative (only keep very common tokens) |

### Choosing a Threshold

Consider:

1. **Token count distribution**: Run analysis to see percentiles
2. **Accuracy vs coverage trade-off**: Higher threshold = fewer tokens contribute
3. **Language diversity**: Rare languages may need lower threshold
4. **Validation accuracy**: Test on FLORES with different thresholds

**Recommended approach**:
1. Start with threshold = 10
2. Evaluate on FLORES
3. Adjust based on accuracy changes

## Running the Analysis

Use the analysis scripts to examine your actual training data and learned weights. The scripts will generate statistics and visualizations specific to your model.

## Impact on Accuracy

### Validation

Validate on FLORES after filtering:

```bash
uv run wldetect eval --model-path artifacts/gemma3-27b/
```

Compare accuracy before/after filtering to determine impact.

## Outlier Analysis

**Outliers** are tokens with:
- High learned weight (e.g., > 75th percentile)
- Low frequency (e.g., < threshold)

### Why Outliers Occur

1. **Discriminative rare tokens**: Unique to a language but rare overall
   - Example: Specialized scripts or characters
2. **Overfitting**: Model assigns high weight due to limited examples
3. **Data quality issues**: Mislabeled examples

### Handling Outliers

**Option 1**: Manual review
- Examine tokens in outlier set
- Decide case-by-case whether to filter

**Option 2**: Keep all outliers
- Use a modified threshold: `token_counts < threshold AND weight < weight_threshold`

**Option 3**: Ignore (recommended)
- If outliers are few (< 1% of vocab), impact is minimal

## Reproducibility

Filtered models save metadata for reproducibility:

```python
metadata = {
    "zero_weight_mask": zero_weight_mask,  # Boolean mask (vocab_size,)
    "min_count_threshold": 10,              # Threshold used
}
```

This allows:
- Understanding which tokens were filtered
- Reproducing the filtering process
- Analyzing filtered vs non-filtered tokens

## Best Practices

1. **Always count token frequencies** from actual training data (not external corpus)
2. **Analyze before filtering** - understand the distribution first
3. **Test multiple thresholds** - validate on FLORES
4. **Document your choice** - include threshold in model metadata
5. **Version control** - save both filtered and non-filtered models

## Limitations

1. **Language-agnostic**: Same threshold for all languages (rare languages may suffer)
2. **Static**: Threshold set once (cannot adapt to input)
3. **Binary**: Token either contributes or doesn't (no soft filtering)

## Future Improvements

Potential enhancements:

1. **Per-language thresholds**: Different thresholds for common vs rare languages
2. **Soft filtering**: Reduce weight instead of zeroing
3. **Adaptive filtering**: Adjust based on input text characteristics
4. **Multi-stage filtering**: Different thresholds at training vs inference

## References

- `scripts/analysis/token_frequency_counter.py` - Count token occurrences
- `scripts/analysis/weight_analysis.py` - Analyze weight-frequency correlation
- `scripts/analysis/create_filtered_e3m4_model.py` - Generate filtered models
- `docs/architecture.md` - Token weighting mechanism

## Next Steps

- [Weight Distribution](weight-distribution.md) - CDF analysis of weights
- [Architecture](../architecture.md) - Token weighting mechanism
- [Training Guide](../training.md) - Train custom models
