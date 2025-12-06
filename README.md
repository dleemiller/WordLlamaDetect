# WordLlama Detect

Fast, lightweight language detection using static LLM embeddings with learned projection.

<p align="center">
  <img src="assets/wordllamadetect.jpg" alt="WordLlamaDetect" width="90%">
</p>

## Overview

WordLlama Detect is a language detection library that uses static embeddings from large language models combined with a simple learned projection layer. This detector is fast, accurate and targets CPU & numpy-only inference.

**Features:**
- NumPy-only inference with no PyTorch dependency
- Pre-trained model (150 languages)
- FP8 quantized lookup table (38MB)
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

WLDetect ships with a pre-trained model based on Gemma3-27B token embeddings:
- **Languages**: 150 (from OpenLID-v2 dataset)
- **Model size**: 38MB (fp8-quantized lookup table)
- **Accuracy**: XX% on FLORES+ dev set
- **Language codes**: ISO 639-3 + ISO 15924 script (e.g., `eng_Latn`, `cmn_Hans`, `arb_Arab`)

The model loads automatically with `WLDetect.load()` - no separate download needed.

## Architecture

### Simple Inference Pipeline (NumPy-only)

1. **Tokenize**: Use HuggingFace fast tokenizer (Rust-based, no PyTorch)
2. **Lookup**: Index into pre-computed lookup table (vocab_size × n_languages)
3. **Pool**: Pool token sequence (1 x n_languages)
4. **Softmax**: Convert to language probabilities

The lookup table is trained using: `(embeddings * token_weights) @ projection.T + bias = lookup_table`,
where the embeddings are frozen token embeddings extracted from Gemma3 27B using focal loss. Training
is done using the OpenLIDv2 dataset.

This means inference is just:
```python
logits = lookup_table[token_ids]  # O(seq_len) lookup
pooled = logsumexp_pool(logits)   # O(seq_len * n_langs)
probs = softmax(pooled)           # O(n_langs)
```

### Quantization

The lookup table is quantized from fp32 to fp8_e4m3fn format:
- **Original size**: ~150MB (fp32)
- **Quantized size**: 38MB (fp8_e4m3fn)

## Performance

TODO: BENCHMARK AND ACCURACY

## Supported Languages

TODO: LINK TO MODEL CONFIG

Languages use ISO 639-3 language codes with ISO 15924 script codes.

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
- `lookup_table_fp8_e4m3fn.safetensors` - Quantized lookup table (for inference)
- `projection.safetensors` - Projection matrix (fp32, for fine-tuning)
- `model_config.yaml` - Model configuration
- `model.pt` - Full PyTorch checkpoint

### Training Commands

```bash
# Train model
uv run wldetect train --config configs/training/gemma3-27b.yaml

# Evaluate on FLORES+
uv run wldetect eval --config configs/training/gemma3-27b.yaml

# Generate lookup table from checkpoint
uv run wldetect create-lookup \
  --checkpoint artifacts/checkpoints/checkpoint_step_100000.pt \
  --config configs/training/gemma3-27b.yaml \
  --output-dir artifacts/

# Curate languages (filter by accuracy threshold)
uv run wldetect curate \
  --config configs/training/gemma3-27b.yaml \
  --accuracy-threshold 0.8 \
  --output-config configs/models/curated.yaml
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
