# Architecture

This document explains the technical architecture of WLDetect.

## Overview

WLDetect uses a **precomputed lookup table** approach for language detection:

1. Extract static embeddings from pre-trained LLMs
2. Train a lightweight projection layer (hidden_dim → n_languages)
3. Precompute lookup table for all vocabulary tokens
4. At inference: token lookup → pooling → softmax → language probabilities

**Key Insight**: By precomputing the projection for all tokens, we eliminate the need to run the full LLM at inference time.

## Training Architecture

### Components

```
Input Text
    ↓
Tokenizer
    ↓
Token IDs [batch_size, seq_len]
    ↓
Embedding Lookup (frozen)
    ↓
Token Embeddings [batch_size, seq_len, hidden_dim]
    ↓
Token Weighting (learned)
    ↓
Weighted Embeddings [batch_size, seq_len, hidden_dim]
    ↓
Linear Projection (learned)
    ↓
Logits [batch_size, seq_len, n_languages]
    ↓
LogSumExp Pooling
    ↓
Sequence Logits [batch_size, n_languages]
    ↓
Softmax
    ↓
Language Probabilities [batch_size, n_languages]
```

### Learned Parameters

1. **Token Weights**: `(vocab_size, 1)`
   - Learned weight for each token
   - Allows model to emphasize discriminative tokens
   - Zero-weighted tokens have no contribution

2. **Projection Matrix**: `(n_languages, hidden_dim)`
   - Linear transformation from embedding space to language space
   - Single dense layer, no hidden layers

3. **Projection Bias**: `(n_languages,)`
   - Learned bias term for each language

**Total Parameters**: `vocab_size + (n_languages × hidden_dim) + n_languages`

For Gemma3-27B (256k vocab, 148 langs, 2304 dim):
- Token weights: 256,000 parameters
- Projection: 148 × 2304 = 340,992 parameters
- Bias: 148 parameters
- **Total: ~597k parameters** (vs 27B in base model)

## Inference Architecture

At inference, we use a **precomputed lookup table** instead of embedding lookups and projection:

```
Input Text
    ↓
Tokenizer
    ↓
Token IDs [seq_len]
    ↓
Lookup Table Indexing
    ↓
Token Logits [seq_len, n_languages]
    ↓
LogSumExp Pooling
    ↓
Sequence Logits [n_languages]
    ↓
Softmax
    ↓
Language Probabilities [n_languages]
```

### Lookup Table Computation

The lookup table is precomputed as:

```python
lookup_table = (embeddings * token_weights) @ projection.T + bias
```

Where:
- `embeddings`: (vocab_size, hidden_dim) - frozen from LLM
- `token_weights`: (vocab_size, 1) - learned
- `projection`: (n_languages, hidden_dim) - learned
- `bias`: (n_languages,) - learned

**Result**: `lookup_table` is (vocab_size, n_languages)

At inference:
```python
token_logits = lookup_table[token_ids]  # (seq_len, n_languages)
seq_logits = logsumexp(token_logits, dim=0)  # (n_languages,)
probs = softmax(seq_logits)
```

**Advantages**:
- No matrix multiplication at inference (just indexing)
- No PyTorch dependency
- Extremely fast (pure NumPy operations)
- Small model size (~38MB for 148 languages)

## Pooling Methods

WLDetect supports three pooling methods:

### 1. LogSumExp Pooling (Default)

```python
seq_logits = logsumexp(token_logits, dim=0)
```

**Rationale**: Aggregates information from multiple tokens rather than relying on a single token decision boundary. Considers all tokens with emphasis on high-scoring ones.

**Characteristics**: More dynamic than max pooling, more discriminative than average pooling

### 2. Max Pooling

```python
seq_logits = token_logits.max(dim=0)
```

**Rationale**: Uses only the single most discriminative token for each language.

**Characteristics**: Sharp decision boundaries, sensitive to single-token signals

### 3. Average Pooling

```python
seq_logits = token_logits.mean(dim=0)
```

**Rationale**: Equal weight to all tokens in the sequence.

**Characteristics**: Most stable, considers full sequence uniformly

## Token Weighting Mechanism

Token weights allow the model to learn which tokens are most useful for language detection.

**Examples of high-weight tokens**:
- Language-specific characters (e.g., ñ, ü, ж)
- Function words (articles, prepositions)
- Common words unique to a language

**Examples of low-weight tokens**:
- Numbers, punctuation (language-agnostic)
- Rare technical terms (insufficient training data)
- Borrowed words (appear in multiple languages)

### Token Filtering

Tokens with insufficient training representation can be **zero-weighted**:

1. Count token occurrences in training data
2. Identify tokens below threshold (e.g., < 10 occurrences)
3. Set `token_weights[rare_tokens] = 0`

This approach is based on the hypothesis that under-represented tokens may contribute unreliable signals. Impact on accuracy should be validated empirically.

## Multi-Model Concatenation

WLDetect supports concatenating embeddings from multiple models **before training** to enrich the feature representation while keeping the final artifact size unchanged.

### How It Works

```python
# Step 1: Load embeddings from multiple models
embeddings_1 = extract_embeddings("google/gemma-3-27b-pt")  # (262208, 2304)
embeddings_2 = extract_embeddings("google/gemma-3-4b-pt")   # (262208, 4096)

# Step 2: Concatenate along hidden dimension BEFORE training
embeddings_combined = np.concatenate([embeddings_1, embeddings_2], axis=1)
# Shape: (262208, 2304 + 4096) = (262208, 6400)

# Step 3: Train projection on concatenated embeddings
projection = train_projection(
    embeddings=embeddings_combined,     # (262208, 6400)
    n_languages=148
)
# Projection shape: (6400, 148)

# Step 4: Precompute lookup table
lookup_table = (embeddings_combined * token_weights) @ projection.T + bias
# Lookup table shape: (262208, 148) - SAME SIZE as single-model!
```

**Key Point**: The **final lookup table is always (n_vocab, n_languages)** regardless of how many models are concatenated. Multi-model concatenation only affects training, not the deployed artifact size.

### CRITICAL Requirements

Models **MUST** have **identical vocabularies**:

**Valid**: Gemma3-27B + Gemma3-4B
- Same model family (Gemma3)
- Same tokenizer
- Same vocab size (262,208)
- Token ID 12345 maps to the same token in both models

**Invalid**: Gemma3 + Qwen3
- Different model families
- Different tokenizers
- Different vocabularies
- Token ID 12345 might be "hello" in Gemma3 but "世界" in Qwen3
- Would produce meaningless concatenated embeddings

The model config validates this at load time by checking all models have the same `type` field.

### Why Concatenate Embeddings?

Different model sizes learn different representations:
- **Larger models** (27B params): Broader context, richer semantics
- **Smaller models** (4B params): More focused, task-specific features

Concatenating embeddings combines these complementary representations without increasing the final model size.

### Trade-offs

**Pros**:
- Richer feature representation
- No increase in deployed model size (lookup table still n_vocab × n_languages)
- Can leverage different model capacities

**Cons**:
- Larger projection matrix during training: (sum_of_hidden_dims, n_languages)
- Longer training time
- Requires downloading multiple embedding sets
- Impact on accuracy must be validated empirically

## Training Strategy

### Frozen Embeddings

Embeddings are **frozen during training**:
- Prevents catastrophic forgetting
- Reduces memory requirements
- Faster training (only update projection)
- Enables pre-extraction of embeddings

### Loss Function

Cross-entropy loss:
```python
loss = CrossEntropyLoss(logits, targets)
```

With optional class weighting for imbalanced datasets.

### Optimization

- **Optimizer**: Adam
- **Learning Rate**: 0.001 (default)
- **Batch Size**: 512 (default)
- **Epochs**: 1-2 (adjust based on validation performance)

### Data Augmentation

None currently.

## Inference Optimizations

### Quantization

Lookup tables are quantized to FP8 formats (E4M3FN or E3M4) to reduce model size:
- FP32 → FP8: 4× size reduction
- 148 languages × 256k vocab × 4 bytes = 150 MB (FP32)
- 148 languages × 256k vocab × 1 byte = **38 MB (FP8)**

See [Quantization](quantization.md) for details.

### Fast Tokenization

Uses `tokenizers` library (Rust-based) instead of transformers:
- 10-100× faster tokenization
- No PyTorch dependency
- Smaller package size

### NumPy-Only Inference

All inference operations use NumPy:
- No PyTorch or transformers at inference
- Smaller deployment footprint (~1GB savings)
- Easier integration in production systems

## Design Decisions

### Why Static Embeddings?

**Alternatives considered**:
1. Full LLM forward pass (too slow)
2. Fine-tuned embeddings (risk of catastrophic forgetting)
3. Static embeddings ✓ (best trade-off)

Static embeddings still capture rich lexical and linguistic information sufficient for language detection.

### Why Single Linear Layer?

**Alternatives considered**:
1. Deep MLP (overfitting risk, longer inference)
2. Attention mechanism (too complex)
3. Single linear layer ✓ (simple, interpretable, fast)

Language detection is a relatively simple classification task that doesn't require deep architectures.

### Why LogSumExp Pooling?

**Alternatives considered**:
1. Average pooling (uniform weighting, may lack discriminative power)
2. Attention pooling (adds complexity and learnable parameters)
3. Max pooling (single-token decision boundary)
4. LogSumExp pooling ✓ (default choice)

LogSumExp was chosen as the default because it avoids relying on single-token decision boundaries (as max pooling does) while still being more discriminative than uniform average pooling. The choice of pooling method is empirical and may vary based on the task.

## Limitations

1. **Token-level detection only**: Cannot detect sub-token languages or mixed scripts within a single token
2. **Vocabulary bound**: Can only detect languages represented in training data
3. **No context modeling**: Each token contributes independently (no sequential dependencies)
4. **Single language output**: Designed for monolingual text (not code-switching)

## Future Improvements

Potential enhancements:
- Multi-label output for code-switching detection
- Confidence calibration for uncertain predictions
- Sub-word pooling for script detection
- Adaptive token weighting based on input text

## Next Steps

- [Quantization](quantization.md) - Learn about FP8 quantization
- [Training Guide](training.md) - Train custom models
- [Supported Languages](languages.md) - View language list
