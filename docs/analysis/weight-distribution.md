# Weight Distribution Analysis

This document explains how to analyze the distribution of learned token weights.

## Overview

Token weights are learned parameters that determine how much each token contributes to language detection. Analyzing their distribution provides insights into:

- Which tokens are most important for language detection
- Whether weights are concentrated or spread out
- Potential issues with training (e.g., all weights near zero)

## Generating Weight CDF

The **Cumulative Distribution Function (CDF)** shows what percentage of tokens have weight ≤ x.

### Running the Analysis

```bash
uv run python scripts/analysis/weight_cdf.py \
  --checkpoint artifacts/gemma3-27b/checkpoints/checkpoint_epoch_1.pt \
  --output artifacts/token_weights_cdf.png
```

**Output**: `token_weights_cdf.png` with two subplots:
1. Full range CDF
2. Zoomed CDF (5th-95th percentile)

### Interpreting the CDF

#### Actual Results (Gemma3-27B, epoch 1)

![Token Weight CDF](../../artifacts/token_weights_cdf.png)

```
Token Weight Statistics:
  Mean:   0.998239
  Median: 0.982546
  Std:    0.710033
  Min:    -12.913287
  Max:    12.991529

Percentiles:
   1st:  -0.710587
   5th:   0.164636
  10th:   0.272308
  25th:   0.612911
  50th:   0.982546
  75th:   1.144689
  90th:   1.887186
  95th:   2.367052
  99th:   3.362391
```

**Observations**:
- Median near 1.0 (close to initialization)
- Weights learned a distribution (std ~0.71)
- Some outliers exist (min: -12.9, max: 13.0)

#### Issues to Watch For

**Problem 1: All weights near 1.0**
```
50th percentile: 0.998
95th percentile: 1.002
```
**Diagnosis**: Model didn't learn (weights barely changed from initialization)

**Problem 2: Extreme outliers**
```
99th percentile: 15.6
99.9th percentile: 234.7
```
**Diagnosis**: Possible overfitting or gradient explosion

**Problem 3: Many near-zero weights**
```
25th percentile: 0.001
50th percentile: 0.003
```
**Diagnosis**: Too much regularization or learning rate too low

## Combining with Token Count CDF

Compare weight distribution with token frequency distribution:

### Token Count CDF

![Token Count CDF](../../artifacts/token_counts_cdf.png)

```bash
uv run python scripts/analysis/plot_token_count_cdf.py \
  --counts artifacts/token_counts.npy \
  --output artifacts/token_counts_cdf.png
```

**Total training tokens**: 4,945,738,333

**Shows**: Cumulative tokens by frequency rank (sorted ascending - lowest frequency at origin)

### Joint Analysis

Plot both CDFs side-by-side to understand:

1. **Do high-frequency tokens have high weights?**
   - Expected: Yes (model should weight common tokens)
   - If no: Model may be overfitting to rare tokens

2. **Do rare tokens have low weights?**
   - Expected: Yes (model should down-weight unreliable tokens)
   - If no: May need token filtering

## Weight Statistics

### Basic Statistics

Run the weight CDF script to get statistics for your checkpoint:

```bash
uv run python scripts/analysis/weight_cdf.py \
  --checkpoint artifacts/gemma3-27b/checkpoints/checkpoint_epoch_1.pt \
  --output artifacts/token_weights_cdf.png
```

This will output:
- Mean, median, std, min, max
- Percentiles (1st, 5th, 10th, ..., 99th)
- CDF plots (full range + zoomed)

### Percentiles

Percentiles help identify the distribution shape:

```python
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = np.percentile(token_weights, p)
    print(f"{p:2d}th: {val:.6f}")
```

**Use cases**:
- **1st percentile**: Identify minimum non-zero weight
- **50th percentile (median)**: Central tendency
- **99th percentile**: Identify potential outliers

## Weight Histogram

In addition to CDF, a histogram shows the density:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(token_weights.flatten(), bins=100, edgecolor='black', alpha=0.7)
plt.xlabel("Token Weight")
plt.ylabel("Number of Tokens")
plt.title("Distribution of Token Weights")
plt.grid(True, alpha=0.3)
plt.savefig("token_weights_histogram.png", dpi=150)
```

**Expected shape**:
- **Gaussian-like**: Most tokens near mean, few outliers
- **Bimodal**: Two peaks (e.g., rare vs common tokens)
- **Uniform**: Wide spread (many weights at different values)

**Unhealthy shapes**:
- **Delta spike**: All weights the same (no learning)
- **Heavy tail**: Many extreme outliers (instability)

## Comparing Across Checkpoints

Track weight evolution during training:

```python
checkpoints = [
    "checkpoint_step_1000.pt",
    "checkpoint_step_5000.pt",
    "checkpoint_epoch_1.pt",
]

for ckpt in checkpoints:
    weights = load_weights(ckpt)
    print(f"{ckpt}: mean={weights.mean():.4f}, std={weights.std():.4f}")
```

**Expected trend**:
- **Mean**: Stays relatively stable (around initialization)
- **Std**: Increases (weights diverge as model learns)

## Weight Sparsity

Measure how many weights are effectively zero (or near-zero):

```python
threshold = 0.01  # Consider < 0.01 as "zero"
sparsity = (np.abs(token_weights) < threshold).sum() / len(token_weights)
print(f"Sparsity: {sparsity:.2%}")
```

**Interpretation**:
- **0-10% sparsity**: Dense model (most tokens contribute)
- **10-50% sparsity**: Moderate (some tokens ignored)
- **>50% sparsity**: Sparse model (many tokens ignored)

**Note**: With explicit token filtering, sparsity can be >50% (by design).

## Weight Magnitude Distribution

Analyze the distribution of **absolute values**:

```python
abs_weights = np.abs(token_weights)
print(f"Mean magnitude: {abs_weights.mean():.6f}")
print(f"Max magnitude:  {abs_weights.max():.6f}")
```

**Why it matters**:
- Large magnitude weights → strong contribution
- Small magnitude weights → weak contribution

## Top/Bottom Tokens by Weight

Identify most/least important tokens:

```python
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-pt")

# Sort by weight
sorted_indices = np.argsort(token_weights.flatten())

# Top 10 highest weights
print("Top 10 tokens (highest weight):")
for i in sorted_indices[-10:][::-1]:
    token = tokenizer.decode([i])
    weight = token_weights[i]
    print(f"  {i:6d} | {weight:.6f} | {repr(token)}")

# Bottom 10 lowest weights
print("\nBottom 10 tokens (lowest weight):")
for i in sorted_indices[:10]:
    token = tokenizer.decode([i])
    weight = token_weights[i]
    print(f"  {i:6d} | {weight:.6f} | {repr(token)}")
```

**Expected patterns**:
- **High weights**: Language-specific characters, function words
- **Low weights**: Numbers, punctuation, rare technical terms

## Combining Weight and Frequency Analysis

Use `weight_analysis.py` for comprehensive correlation analysis:

```bash
uv run python scripts/analysis/weight_analysis.py \
  --checkpoint artifacts/gemma3-27b/checkpoints/checkpoint_epoch_1.pt \
  --token-counts artifacts/token_counts.npy \
  --output-dir artifacts/
```

This generates:
1. **Scatter plot**: Weight vs frequency (log-log)
2. **Histograms**: Separate distributions
3. **Coverage curve**: Token frequency Pareto
4. **Binned weights**: Average weight by frequency bin
5. **Outlier detection**: High weight + low frequency

## Practical Applications

### 1. Debugging Training

**Symptom**: Validation accuracy not improving

**Check**: Weight distribution CDF

**Diagnosis**:
- Flat CDF (all weights same) → No learning, check learning rate
- Extreme outliers → Gradient explosion, reduce learning rate
- All near zero → Too much regularization

### 2. Model Compression

**Goal**: Reduce model size

**Strategy**: Prune low-weight tokens

**Implementation**:
1. Plot weight CDF
2. Choose threshold (e.g., 10th percentile)
3. Zero weights below threshold
4. Validate accuracy impact

### 3. Interpretability

**Goal**: Understand what model learned

**Approach**:
1. Identify high-weight tokens per language
2. Verify they are linguistically meaningful
3. Check for unexpected patterns (potential issues)

## Comparison with Token Count Distribution

### Correlation Analysis

Expected: **Positive correlation** between frequency and weight

```python
from scipy.stats import pearsonr

correlation, p_value = pearsonr(np.log1p(token_counts), token_weights)
print(f"Correlation: {correlation:.4f} (p={p_value:.4e})")
```

**Interpretation**:
- **correlation > 0.5**: Strong positive correlation (expected)
- **correlation < 0.2**: Weak correlation (may indicate issues)
- **correlation < 0**: Negative correlation (unexpected, investigate)

### Divergence Analysis

Tokens where weight and frequency disagree:

```python
# Normalize both to [0, 1]
norm_counts = (token_counts - token_counts.min()) / (token_counts.max() - token_counts.min())
norm_weights = (token_weights - token_weights.min()) / (token_weights.max() - token_weights.min())

# Find divergent tokens
divergence = np.abs(norm_counts - norm_weights)
divergent_indices = np.argsort(divergence)[-100:]  # Top 100 divergent

# Investigate
for i in divergent_indices:
    token = tokenizer.decode([i])
    print(f"{repr(token)}: count={token_counts[i]}, weight={token_weights[i]:.4f}")
```

## Visualization Best Practices

### CDF Plot

**Good practices**:
- Show full range + zoomed view (5th-95th percentile)
- Mark median and key percentiles
- Use log scale if distribution spans orders of magnitude

**Example code**: See `scripts/analysis/weight_cdf.py`

### Histogram

**Good practices**:
- Use 50-100 bins for smooth distribution
- Add vertical lines for mean/median
- Consider log scale for x-axis if needed

### Scatter (Weight vs Frequency)

**Good practices**:
- Use log-log scale (both axes)
- Color-code by language or filtering status
- Add regression line to show correlation

## References

- `scripts/analysis/weight_cdf.py` - Generate weight CDF
- `scripts/analysis/plot_token_count_cdf.py` - Generate count CDF
- `scripts/analysis/weight_analysis.py` - Comprehensive correlation analysis
- `docs/architecture.md` - Token weighting mechanism

## Next Steps

- [Token Filtering](token-filtering.md) - Filter rare tokens
- [Architecture](../architecture.md) - Understand token weighting
- [Training Guide](../training.md) - Train custom models
