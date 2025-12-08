# Training Guide

This guide covers training custom language detection models with WLDetect.

## Overview

WLDetect uses a lightweight training architecture:
- **Frozen embeddings** from pre-trained LLMs (Gemma3, Qwen3, etc.)
- **Learned projection** layer (hidden_dim → n_languages)
- **Token weighting** to emphasize discriminative tokens

## Installation

Install training dependencies:

```bash
# CPU training
uv sync --extra training

# GPU training (CUDA 12.8)
uv sync --extra cu128
```

## Quick Start

1. **Configure your model** in `configs/models/`:

```yaml
# configs/models/my-model.yaml
model:
  name: google/gemma-3-27b-pt
  hidden_dim: 2304
  embedding_layer_name: model.embed_tokens

languages:
  eng_Latn: 0
  fra_Latn: 1
  # ... more languages
```

2. **Configure training** in `configs/training/`:

```yaml
# configs/training/my-training.yaml
model_config_path: configs/models/my-model.yaml

dataset:
  name: laurievb/OpenLID-v2
  split: train
  max_samples_per_language: 100000
  min_samples_per_language: 1000

training:
  batch_size: 512
  learning_rate: 0.001
  num_epochs: 1
  projection:
    dropout: 0.1

output:
  artifacts_dir: artifacts/my-model/
```

3. **Train the model**:

```bash
uv run wldetect train --config configs/training/my-training.yaml
```

## Training Pipeline

The training pipeline follows these steps:

1. **Extract Embeddings**: Download and extract embedding tensors from HuggingFace
2. **Load Tokenizer**: Load tokenizer for text preprocessing
3. **Prepare Dataset**: Load OpenLID-v2, filter languages, balance samples
4. **Create Data Loader**: PyTorch DataLoader with lazy tokenization
5. **Initialize Model**: Create LanguageDetectionModel with frozen embeddings
6. **Train**: Train projection layer and token weights
7. **Evaluate on FLORES**: Final evaluation on FLORES+ dev set
8. **Save Artifacts**: Save projection matrix, lookup tables, model config

## Configuration Options

### Dataset Configuration

```yaml
dataset:
  name: laurievb/OpenLID-v2          # HuggingFace dataset name
  split: train                        # Dataset split
  max_samples_per_language: 100000   # Cap per language
  min_samples_per_language: 1000     # Minimum samples required
  balance_languages: true             # Balance dataset
```

### Training Configuration

```yaml
training:
  batch_size: 512                     # Batch size
  learning_rate: 0.001                # Learning rate
  num_epochs: 1                       # Number of epochs
  num_workers: 4                      # DataLoader workers
  save_checkpoints: true              # Save checkpoints
  checkpoint_interval: 5000           # Save every N steps

  projection:
    dropout: 0.1                      # Dropout probability
```

### Inference Configuration

```yaml
inference:
  max_sequence_length: 512            # Max tokens per sequence
  pooling: logsumexp                  # Pooling method (logsumexp, max, average)
```

## Multi-Model Training

WLDetect supports concatenating embeddings from multiple models **before training** to enrich the feature representation while keeping the final artifact size unchanged.

```yaml
# configs/models/multi-model.yaml
models:
  - name: google/gemma-3-27b-pt
    type: gemma3
    hidden_dim: 2304
    embedding_layer_name: model.embed_tokens.weight

  - name: google/gemma-3-4b-pt
    type: gemma3
    hidden_dim: 4096
    embedding_layer_name: model.embed_tokens.weight

# Combined hidden_dim = 2304 + 4096 = 6400
# Final lookup table: still (n_vocab, n_languages) - same size as single-model
```

**How it works**:
1. Embeddings are concatenated **before training**: (n_vocab, 2304+4096) = (n_vocab, 6400)
2. Projection is trained on concatenated embeddings: (6400, n_languages)
3. Lookup table is precomputed: (n_vocab, 6400) @ (6400, n_languages) = (n_vocab, n_languages)
4. **Final artifact is always (n_vocab, n_languages)** - same size as single-model!

**CRITICAL Requirements**:
- All models must be from the **same family** (same `type` field)
- Models must use the **exact same tokenizer** (identical vocabularies)
- Vocabulary sizes must match exactly
- Token IDs must map to the same tokens across all models

**Valid**: Gemma3-27B + Gemma3-4B (both `type: gemma3`)
**Invalid**: Gemma3 + Qwen3 (different types, different vocabularies)

The model config validator checks this at load time and will raise an error if types don't match.

## Output Artifacts

After training, the following artifacts are saved to `artifacts/`:

- **`projection.safetensors`**: Projection weights and token weights
- **`lookup_table_fp8_e4m3fn.safetensors`**: E4M3FN lookup table (backward compatibility)
- **`lookup_table_fp8_e3m4.safetensors`**: E3M4 lookup table (new default, 30% better precision)
- **`model_config.yaml`**: Model configuration
- **`flores_dev_results.json`**: FLORES evaluation results
- **`checkpoints/`**: Training checkpoints

## Advanced Topics

### Token Filtering

Zero-weight tokens with insufficient training representation:

1. Generate token counts:
```bash
uv run python scripts/analysis/token_frequency_counter.py
```

2. Analyze weight distribution:
```bash
uv run python scripts/analysis/weight_analysis.py
```

3. Create filtered E3M4 model:
```bash
uv run python scripts/analysis/create_filtered_e3m4_model.py --threshold 10
```

### Custom Datasets

To use a custom dataset instead of OpenLID-v2, modify the dataset configuration:

```yaml
dataset:
  name: my-org/my-language-dataset
  split: train
  text_column: text         # Column containing text
  language_column: language # Column containing language codes
```

### Evaluation

Evaluate on FLORES+ after training:

```bash
uv run wldetect eval --model-path artifacts/my-model/
```

## Troubleshooting

### Out of Memory

Reduce batch size or use gradient accumulation:

```yaml
training:
  batch_size: 256  # Reduce from 512
```

### Language Imbalance

Adjust balancing parameters:

```yaml
dataset:
  balance_languages: true
  max_samples_per_language: 50000  # Reduce maximum
  min_samples_per_language: 5000   # Increase minimum
```

### Poor Accuracy

- Increase training epochs
- Adjust learning rate
- Add more training data
- Check language confusion matrix in FLORES results

## Next Steps

- [Supported Languages](languages.md) - View language list and FLORES scores
- [Architecture](architecture.md) - Understand the model architecture
- [Quantization](quantization.md) - Learn about FP8 quantization
