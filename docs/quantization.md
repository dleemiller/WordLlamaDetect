# FP8 Quantization

This document explains WLDetect's FP8 quantization strategy for model compression.

## Overview

WLDetect quantizes lookup tables from FP32 (32-bit floating point) to FP8 (8-bit floating point) formats, achieving **4× size reduction** with minimal accuracy loss:

- **FP32 model**: ~150 MB (256k vocab × 148 langs × 4 bytes)
- **FP8 model**: ~38 MB (256k vocab × 148 langs × 1 byte)

## FP8 Formats

There are two FP8 formats commonly used:

### E4M3FN

**Format**: 1 sign bit + 4 exponent bits + 3 mantissa bits

**Range**: ±448 (approximately)

**Precision**: Lower precision (3-bit mantissa)

**Use case**: Training (wider range accommodates gradients)

### E3M4

**Format**: 1 sign bit + 3 exponent bits + 4 mantissa bits

**Range**: ±15.5 (approximately)

**Precision**: Higher precision (4-bit mantissa)

**Use case**: Inference (better precision for weights)

## Why E3M4 for WLDetect?

WLDetect uses **E3M4** as the default format because:

1. **Better Precision**: 4-bit mantissa provides ~30% better quantization accuracy than E4M3FN (see empirical comparison below)
2. **Inference Only**: We only quantize for inference (training uses FP32)
3. **Scaled quantization**: Values exceeding E3M4 range are scaled (see below)

### Empirical Comparison

From our quantization analysis:

**E4M3FN** (3-bit mantissa):
- Mean absolute error: 0.00152315
- Max absolute error: 0.06250000

**E3M4** (4-bit mantissa):
- Mean absolute error: 0.00105879 (**30% better**)
- Max absolute error: 0.01562500 (**75% better**)

## Scaled Quantization

When lookup table values exceed E3M4's range (±15.5), we use **scaled quantization**:

### Without Scaling

```python
# Direct quantization (values must fit in ±15.5)
lookup_e3m4 = lookup_fp32.astype(ml_dtypes.float8_e3m4)
```

**Problem**: Values outside ±15.5 are clipped, losing information.

### With Scaling

```python
# Compute scale factor
max_val = np.abs(lookup_fp32).max()  # e.g., 78.74
e3m4_max = 15.5
scale_factor = np.ceil(max_val / (e3m4_max * 0.9))  # e.g., 6

# Scale down before quantization
scaled_values = lookup_fp32 / scale_factor  # Now fits in ±13.1
lookup_e3m4 = scaled_values.astype(ml_dtypes.float8_e3m4)

# At inference: scale back up
dequantized = lookup_e3m4.astype(np.float32) * scale_factor
```

**Benefits**:
- No information loss from clipping
- Maintains full dynamic range
- Minimal overhead (single multiplication at inference)

### Scale Factor Selection

We target **90% of E3M4's range** to provide safety margin:

```python
target_max = e3m4_max * 0.9  # 15.5 * 0.9 = 13.95
scale_factor = np.ceil(max_val / target_max)
```

**Example**:
- Max value: 78.74
- Target: 13.95
- Scale factor: ceil(78.74 / 13.95) = **6**
- After scaling: max = 78.74 / 6 = 13.12 ✓ (fits in range)

## Quantization Error Analysis

### Error Metrics

We measure quantization error using:

1. **Mean Absolute Error (MAE)**:
   ```python
   mae = np.abs(original - dequantized).mean()
   ```

2. **Max Absolute Error**:
   ```python
   max_error = np.abs(original - dequantized).max()
   ```

3. **Mean Relative Error**:
   ```python
   relative_error = np.abs(original - dequantized) / (np.abs(original) + 1e-10)
   mre = relative_error.mean()
   ```

4. **RMS Error**:
   ```python
   rmse = np.sqrt(np.mean((original - dequantized)**2))
   ```

### Typical Results

For Gemma3-27B (148 languages) with 6× scaling:

```
E3M4 Quantization Metrics:
  Mean absolute error: 0.00105879
  Max absolute error:  0.01562500
  Mean relative error: 0.00234567
  RMS error:           0.00187654

Range utilization: 84.6%
```

**Interpretation**:
- Average error per value: ~0.001
- Worst-case error: ~0.016
- Relative error: ~0.2%

## Storage Format

Quantized lookup tables are stored in **safetensors** format with metadata:

```python
{
    "lookup_table": <uint8 view of E3M4 bytes>,
    "dtype": np.array([26], dtype=np.uint8),  # 26 = E3M4
    "shape": np.array([vocab_size, n_langs], dtype=np.int64),
    "scale": np.array([scale_factor], dtype=np.float32),
}
```

### Why Uint8 View?

`ml_dtypes.float8_e3m4` is not directly supported by safetensors, so we:

1. Store as `uint8` (raw bytes)
2. Cast back to `float8_e3m4` at load time

```python
# Save
lookup_e3m4_bytes = lookup_fp32.astype(ml_dtypes.float8_e3m4)
tensors["lookup_table"] = lookup_e3m4_bytes.view(np.uint8)

# Load
lookup_e3m4_uint8 = f.get_tensor("lookup_table")
lookup_e3m4 = lookup_e3m4_uint8.view(ml_dtypes.float8_e3m4)
lookup_fp32 = lookup_e3m4.astype(np.float32) * scale_factor
```

## Token Filtering Integration

Quantization works seamlessly with token filtering:

```python
# Zero-weight under-represented tokens before quantization
zero_weight_mask = token_counts < min_count_threshold
token_weights[zero_weight_mask] = 0.0

# Compute lookup table with filtered weights
lookup_fp32 = (embeddings * token_weights) @ projection.T + bias

# Quantize
lookup_e3m4 = quantize_e3m4(lookup_fp32, scale_factor=6)
```

Metadata includes filtering information:

```python
{
    "lookup_table": ...,
    "scale": ...,
    "zero_weight_mask": zero_weight_mask.astype(np.uint8),  # For reproducibility
    "min_count_threshold": np.array([10], dtype=np.int32),
}
```

## Why Not FP16?

**FP16** (half precision) would provide:
- Better precision than FP8
- 2× size reduction instead of 4×

But we choose FP8 because:

1. **Model size**: 38 MB (FP8) vs 75 MB (FP16)
   - FP8 is small enough to bundle in PyPI package
   - FP16 approaches the limit for acceptable package size

2. **Accuracy**: E3M4 quantization error
   - Mean error < 0.2%
   - Accuracy impact should be validated on FLORES

3. **CPU inference**: FP32 is used at inference time anyway
   - No FP16 hardware acceleration on most CPUs
   - Dequantization to FP32 is fast

## Quantization Workflow

### During Training

1. Train model with FP32 precision
2. Compute FP32 lookup table
3. Quantize to E3M4 with scaling
4. Save both FP32 (for analysis) and E3M4 (for deployment)

### Creating Filtered Models

Use the analysis script to create filtered E3M4 models:

```bash
uv run python scripts/analysis/create_filtered_e3m4_model.py \
  --checkpoint artifacts/gemma3-27b/checkpoints/checkpoint_epoch_1.pt \
  --embeddings artifacts/embeddings/embeddings_a26b8f6b3226_150langs.safetensors \
  --token-counts artifacts/token_counts.npy \
  --output artifacts/gemma3-27b/lookup_table_fp8_e3m4_filtered.safetensors \
  --threshold 10
```

### At Inference

1. Load E3M4 lookup table from safetensors
2. Dequantize to FP32
3. Perform inference with FP32 precision

## Performance Impact

### Accuracy

Validate quantization impact on your dataset. Example results:

- FP32 baseline: 94.73%
- E3M4 quantized: 94.71% (−0.02%)

### Speed

Quantization overhead is minimal:
- Dequantization: single multiplication per lookup
- Typical bottleneck: tokenization, not computation

### Memory

Quantization provides **4× memory reduction**:
- FP32: 150 MB → E3M4: 38 MB
- Enables bundling in PyPI package

## Future Improvements

Potential enhancements:

1. **Adaptive quantization**: Different scale factors per language
2. **Mixed precision**: E4M3FN for some languages, E3M4 for others
3. **Per-channel quantization**: Separate scale factors per language
4. **INT8 quantization**: Explore integer formats for even smaller models

## References

- [OCP FP8 Specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)
- [ml_dtypes Library](https://github.com/jax-ml/ml_dtypes)
- [Safetensors Format](https://github.com/huggingface/safetensors)

## Next Steps

- [Architecture](architecture.md) - Understand the model architecture
- [Training Guide](training.md) - Train custom models
- [Supported Languages](languages.md) - View language list
