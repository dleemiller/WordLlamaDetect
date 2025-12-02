# LangToken

Language detection using static LLM embeddings with learned projection.

## Overview

LangToken is a lightweight language detection library that uses static embeddings from large language models combined with a simple learned projection layer. The key innovation is separating training (PyTorch) from inference (NumPy-only), enabling fast, lightweight deployment without deep learning framework dependencies.

## Features

- **NumPy-only inference**: No PyTorch dependency for production use
- **Simple architecture**: Linear projection + max pooling + softmax
- **Efficient embedding extraction**: Download only embedding tensor shards, not full models
- **FLORES+ evaluation**: Standardized multilingual benchmarking via HuggingFace
- **CLI interface**: Train, evaluate, and detect via command line

## Installation

### For inference only (NumPy-only):
```bash
git clone git@github.com:dleemiller/LangToken.git
cd langtoken
uv sync
```

### For training (includes PyTorch):
```bash
# CPU or default CUDA version
uv sync --extra training

# With CUDA 12.8 (recommended for GPU training)
uv sync --extra cu128
```

### For development (includes testing tools):
```bash
uv sync --extra training --extra dev

# Or with CUDA 12.8
uv sync --extra cu128 --extra dev
```

## Quick Start

### Training a model

1. Configure your model in `configs/models/`:
```yaml
# configs/models/gemma3-27b.yaml
model:
  name: google/gemma-3-27b-pt
  type: gemma3
  hidden_dim: 5376
  shard_pattern: model-00001-of-00012.safetensors
  embedding_layer_name: language_model.model.embed_tokens.weight

languages:
  eng_Latn: 0
  spa_Latn: 1
  fra_Latn: 2
  deu_Latn: 3
  # ... add more languages

inference:
  max_sequence_length: 512
  pooling: logsumexp
```

2. Configure training in `configs/training/`:
```yaml
# configs/training/gemma3-27b.yaml
model_config_path: "configs/models/gemma3-27b.yaml"

dataset:
  name: "laurievb/OpenLID-v2"
  filter_languages: true

training:
  batch_size: 1536
  learning_rate: 0.002
  epochs: 2
  optimizer: "adamw"

evaluation:
  flores_hf_dataset: "openlanguagedata/flores_plus"
```

3. Train:
```bash
uv run langtoken train --config configs/training/gemma3-27b.yaml
```

Artifacts saved to `artifacts/`:
- `projection.safetensors` - Projection matrix and weights
- `model_config.yaml` - Model configuration
- `model.pt` - Full PyTorch checkpoint (optional)

### Using a trained model for inference

```python
from langtoken.inference.detector import LanguageDetector

# Load detector (NumPy-only, no PyTorch needed)
detector = LanguageDetector("artifacts/")

# Detect language
text = "Hello, how are you today?"
lang = detector.get_top_language(text)
probs = detector.detect(text)

print(f"Detected: {lang}")
print(f"Probabilities: {probs}")
```

### CLI detection

```bash
# Detect from text
uv run langtoken detect --model-path artifacts/ --text "Bonjour le monde"

# Detect from file
uv run langtoken detect --model-path artifacts/ --file input.txt
```

## Evaluation

### Evaluate on test set

```bash
uv run langtoken eval \
  --config configs/training/gemma3-27b.yaml \
  --split test \
  --output artifacts/evaluation_metrics.json
```

### Evaluate on FLORES+

```bash
uv run langtoken eval \
  --config configs/training/gemma3-27b.yaml \
  --split dev \
  --output artifacts/flores_metrics.json
```

The FLORES+ dataset from HuggingFace (`openlanguagedata/flores_plus`) provides standardized multilingual evaluation across 200+ languages.

## Architecture

### Training Pipeline

1. **Embedding Extraction**: Download only embedding tensor shards from HuggingFace
2. **Dataset Preparation**: Load OpenLID-v2, filter by model's supported languages
3. **Model Training**: Simple linear projection (hidden_dim → N_lang) with pooling
4. **Artifact Export**: Save projection matrix (safetensors) and config

### Inference Pipeline (NumPy-only)

1. **Tokenize**: Use HuggingFace fast tokenizer (Rust-based, no PyTorch)
2. **Lookup**: Get static embeddings for tokens
3. **Project**: Apply learned projection matrix
4. **Pool**: Max pooling or logsumexp over sequence
5. **Softmax**: Convert to language probabilities

### Why This Works

- **Static embeddings capture linguistic patterns**: Even without context-dependent computation, token embeddings encode language-specific information
- **Simple projection is sufficient**: Language detection doesn't require complex reasoning
- **Pooling captures salient features**: Strongest language signals dominate

## Configuration

See `configs/` directory for examples:

- `configs/models/gemma3-27b.yaml` - Gemma3 27B model (150 languages)
- `configs/models/gemma3-4b.yaml` - Gemma3 4B model
- `configs/models/gemma3-1b.yaml` - Gemma3 1B model
- `configs/training/default.yaml` - Default training config

## Project Structure

```
langtoken/
├── src/langtoken/
│   ├── config/          # YAML config loading and validation
│   ├── embeddings/      # Extract embeddings from HF shards
│   ├── data/            # Dataset loading and tokenization
│   ├── training/        # PyTorch training pipeline
│   ├── inference/       # NumPy-only inference
│   └── cli/             # Command-line interface
├── configs/             # Model and training configurations
├── tests/               # Test suite
└── artifacts/           # Trained models (gitignored)
```

## Development

### Running tests
```bash
uv run pytest                              # All tests
uv run pytest tests/test_config.py         # Single file
uv run pytest --cov=src                    # With coverage
```

### Linting and formatting
```bash
ruff check .                               # Check for issues
ruff check . --fix                         # Auto-fix issues
ruff format .                              # Format code
```

### Pre-commit hooks
```bash
uv run pre-commit install                  # Install hooks
uv run pre-commit run --all-files          # Run manually
```

## Performance

- **Inference speed**: ~1-5ms per text (NumPy-only, CPU)
- **Memory footprint**: ~10GB for Gemma3-27B embeddings + projection
- **Training time**: ~2-4 hours on GPU for 2 epochs (150 languages, 5000 samples/lang)

## Supported Languages

Languages depend on your configuration. The OpenLID-v2 dataset supports 200+ languages. The FLORES+ dataset provides evaluation data for 200+ language varieties using ISO 639-3 and ISO 15924 script codes (e.g., `eng_Latn`, `cmn_Hans`, `arb_Arab`).

## Limitations

- **No context modeling**: Uses static embeddings, not contextualized
- **Vocabulary-dependent**: Limited to model's tokenizer vocabulary
- **Single language detection**: Doesn't handle code-switching or multilingual texts

## License

MIT License

## Acknowledgments

- OpenLID-v2 dataset: [laurievb/OpenLID-v2](https://huggingface.co/datasets/laurievb/OpenLID-v2)
- FLORES+ dataset: [openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus)
- HuggingFace transformers and tokenizers libraries
- Google Gemma model team
