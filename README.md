# WordLlama Detect

Fast, lightweight language detection using static LLM embeddings with learned projection.

<p align="center">
  <img src="assets/wordllamadetect.jpeg" alt="WordLlamaDetect" width="90%">
</p>

## Overview

WordLlama Detect is a language detection library that uses static embeddings from large language models combined with a simple learned projection layer. This detector is fast, accurate and targets CPU & numpy-only inference.

**Features:**
- NumPy-only inference with no PyTorch dependency
- Pre-trained model (148 languages)
- Sparse exp lookup table (13MB, 91% size reduction)
- Fast inference: ~30k texts/s single thread
- Simple API

## Installation

```bash
pip install wldetect
```

Or from source:
```bash
git clone https://github.com/dleemiller/wldetect.git
cd wldetect
uv sync
```

## Quick Start

### Python API

```python
from wldetect import WLDetect

# Load bundled model (no path needed)
wld = WLDetect.load()

# Detect language for single text
lang, confidence = wld.predict("Hello, how are you today?")
print(f"Detected: {lang} (confidence: {confidence:.2%})")
# Output: Detected: eng_Latn (confidence: 99.84%)

# Detect language for multiple texts (uses batch tokenization)
texts = ["Hello world", "Bonjour le monde", "Hola mundo", "你好世界"]
results = wld.predict(texts)
for text, (lang, conf) in zip(texts, results):
    print(f"{text} → {lang} ({conf:.2%})")
```

### CLI Usage

```bash
# Detect from text
uv run wldetect detect --text "Bonjour le monde"

# Detect from file
uv run wldetect detect --file input.txt

# Use custom model
uv run wldetect detect --model-path /path/to/model --text "Hello"
```

## Bundled Model

WLDetect ships with a pre-trained model based on concatenated Gemma3-27B + Gemma3-4B token embeddings:
- **Languages**: 148 (from OpenLID-v2 dataset)
- **Model size**: 13MB (sparse exp lookup table, 97.15% sparsity)
- **Accuracy**: 92.92% on FLORES+ dev set
- **F1 (macro)**: 92.74%
- **Language codes**: ISO 639-3 + ISO 15924 script (e.g., `eng_Latn`, `cmn_Hans`, `arb_Arab`)

The model loads automatically with `WLDetect.load()` - no separate download needed.

See [docs/languages.md](docs/languages.md) for the complete list of supported languages with performance metrics.

## Architecture

### Simple Inference Pipeline (NumPy-only)

1. **Tokenize**: Use HuggingFace fast tokenizer (Rust-based, no PyTorch)
2. **Lookup**: Index into pre-computed exp lookup table (vocab_size × n_languages)
3. **Pool**: LogSumExp pooling over token sequence
4. **Softmax**: Convert to language probabilities

The lookup table is pre-trained using: `exp((embeddings * token_weights) @ projection.T + bias)`,
where embeddings are frozen token embeddings from Gemma3, trained with focal loss on OpenLID-v2.

**Key optimization**: We pre-compute `exp(logits)` before saving, enabling efficient LogSumExp pooling:

```python
# Lookup pre-exponentiated values
exp_values = lookup_table[token_ids]  # O(seq_len) lookup
# Complete logsumexp: log(sum(exp(logits)))
pooled = log(sum(exp_values, axis=0))  # O(seq_len * n_langs)
# Get probabilities
probs = softmax(pooled)                # O(n_langs)
```

This avoids runtime `exp()` calls while maintaining numerical stability.

### Sparse Storage

The lookup table uses sparse COO (Coordinate) format with configurable sparsification threshold:
- **Original size**: ~148MB (dense fp32)
- **Sparse size**: 13MB (threshold=10.0, default)
- **Sparsity**: 97.15% (values below threshold set to zero)
- **Format**: COO (row, col, data) indices stored as int32, values as fp32
- **Performance impact**: Negligible (0.003% accuracy loss)

Sparsification is applied after exp() pre-computation, removing small exp values that contribute minimally to LogSumExp pooling. The threshold is configurable during model creation.

## Performance

### FLORES+ Benchmark Results

Evaluated on FLORES+ dev set (148 languages, 1,012 sentences per language):

| Metric         | Score  |
|----------------|--------|
| Accuracy       | 92.92% |
| F1 (macro)     | 92.74% |
| F1 (weighted)  | 92.75% |

**Top performing languages**: 31 languages achieve 100% accuracy including `asm_Beng`, `ben_Beng`, `cmn_Hant`, `dzo_Tibt`, `ell_Grek`, `jpn_Jpan`, `kor_Hang`, and more.

**Challenging languages**: `arz_Arab` (27.9%), `bho_Deva` (35.7%), `knc_Arab` (45.8%) - primarily confusion with closely related language variants.

For detailed per-language performance, see [docs/languages.md](docs/languages.md).

### Inference Speed

- **Single-threaded**: ~30,000 texts/second
- **Batch processing**: Uses fast tokenization for optimal throughput

## Supported Languages

The bundled model supports 148 languages from the OpenLID-v2 dataset. Languages use ISO 639-3 language codes with ISO 15924 script codes (e.g., `eng_Latn`, `cmn_Hans`, `arb_Arab`).

See [model_config.yaml](src/wldetect/models/model_config.yaml) for the complete list of supported languages.

## Training

### Installation for Training

```bash
# CPU or default CUDA version
uv sync --extra training

# With CUDA 12.8 (recommended for GPU training)
uv sync --extra cu128
```

### Training Pipeline

1. **Configure model** in `configs/models/custom-config.yaml`:
```yaml
model:
  name: google/gemma-3-27b-pt
  hidden_dim: 5376
  shard_pattern: model-00001-of-00012.safetensors
  embedding_layer_name: language_model.model.embed_tokens.weight

languages:
  eng_Latn: 0
  spa_Latn: 1
  fra_Latn: 2
  # ... add more languages

inference:
  max_sequence_length: 512
  pooling: logsumexp
```

2. **Configure training** in `configs/training/custom-training.yaml`:
```yaml
model_config_path: "configs/models/custom-model.yaml"

dataset:
  name: "laurievb/OpenLID-v2"
  filter_languages: true

training:
  batch_size: 1536
  learning_rate: 0.002
  epochs: 2
```

3. **Train**:
```bash
uv run wldetect train --config configs/training/custom-training.yaml
```

Artifacts saved to `artifacts/`:
- `lookup_table_exp.safetensors` - Sparse exp lookup table (for inference)
- `projection.safetensors` - Projection matrix (fp32, for fine-tuning)
- `model_config.yaml` - Model configuration
- `model.pt` - Full PyTorch checkpoint

### Training Commands

```bash
# Train model
uv run wldetect train --config configs/training/gemma3-27b.yaml

# Evaluate on FLORES+
uv run wldetect eval --model-path artifacts/ --split dev

# Generate sparse lookup table from checkpoint (default: threshold=10.0)
uv run wldetect create-lookup \
  --checkpoint artifacts/checkpoints/checkpoint_step_100000.pt \
  --config configs/training/gemma3-27b.yaml \
  --output-dir artifacts/

# Custom sparsification threshold
uv run wldetect create-lookup \
  --checkpoint artifacts/checkpoints/checkpoint_step_100000.pt \
  --config configs/training/gemma3-27b.yaml \
  --output-dir artifacts/ \
  --threshold 5.0

# Force dense storage (no sparsification)
uv run wldetect create-lookup \
  --checkpoint artifacts/checkpoints/checkpoint_step_100000.pt \
  --config configs/training/gemma3-27b.yaml \
  --output-dir artifacts/ \
  --dense
```

### Training Details

- **Embedding extraction**: Downloads only embedding tensor shards from HuggingFace (not full models)
- **Dataset**: OpenLID-v2 with configurable language filtering and balancing
- **Model**: Simple linear projection (hidden_dim → n_languages) with dropout
- **Pooling**: LogSumExp or max pooling over token sequences
- **Training time**: ~2-4 hours on GPU for 2 epochs (150 languages, 5000 samples/language)
- **Evaluation**: Automatic FLORES+ evaluation after training

## License

Apache 2.0 License

## Acknowledgments

- OpenLID-v2 dataset: [laurievb/OpenLID-v2](https://huggingface.co/datasets/laurievb/OpenLID-v2)
- FLORES+ dataset: [openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus)
- HuggingFace transformers and tokenizers libraries
- Google Gemma model team
