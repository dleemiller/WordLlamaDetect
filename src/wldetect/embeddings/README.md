# EmbeddingsManager

Class for downloading, extracting, caching, and loading LLM embeddings.

## Public Methods

### `__init__(model_config, cache_dir="artifacts/embeddings", hf_cache_dir=None)`
Initialize the manager with configuration.
- **Used in:** All CLI commands (train, evaluate, generate_lookup_table, sweep), training/evaluator.py

### `extract_embeddings(use_cache=True) -> np.ndarray`
Download embeddings from HuggingFace and cache locally. Returns (vocab_size, hidden_dim) array.
- **Used in:** CLI train, evaluate, generate_lookup_table, sweep commands
- **Behavior:** Downloads shards → extracts tensors → concatenates (if multi-model) → saves to cache

### `load_cached_embeddings() -> np.ndarray`
Load embeddings from cache (no download). Returns (vocab_size, hidden_dim) array.
- **Used in:** training/evaluator.py (lookup table generation), CLI sweep command
- **Raises:** FileNotFoundError if cache doesn't exist

### `load_as_memmap() -> np.ndarray`
Load embeddings as memory-mapped array for multi-worker efficiency.
- **Used in:** CLI train, evaluate, generate_lookup_table, sweep commands
- **Behavior:** Creates .npy file if needed, returns read-only memory-mapped array

## Private Methods (Internal Use Only)

- `_get_cache_path()` - Generate cache filename from model hash
- `_load_single_model_embeddings()` - Download and extract embeddings from one model
- `_download_shard()` - Download specific safetensors shard from HuggingFace
- `_find_shard()` - Find shard file containing embeddings using pattern matching
- `_load_from_shard()` - Extract embedding tensor from safetensors file
- `_concatenate_embeddings()` - Concatenate embeddings from multiple models
- `_save_to_cache()` - Save embeddings to safetensors cache
- `_get_model_hash()` - Generate deterministic hash from model names

## Usage Pattern

```python
from wldetect.embeddings import EmbeddingsManager
from wldetect.config.loader import load_model_config

# Initialize
config = load_model_config("configs/models/qwen.yaml")
manager = EmbeddingsManager(config)

# Training: Download once, use memmap for multi-worker
manager.extract_embeddings()  # Downloads and caches
embeddings = manager.load_as_memmap()  # Fast shared access

# Inference: Load from cache
embeddings = manager.load_cached_embeddings()  # Direct load
```

## Cache Location

Embeddings cached at: `artifacts/embeddings/embeddings_<hash>_<n_langs>langs.safetensors`

Example: `embeddings_a1b2c3d4e5f6_201langs.safetensors`
