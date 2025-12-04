"""Tests for lookup table generation."""

import numpy as np

from wldetect.training.lookup_table import compute_lookup_table


def test_compute_lookup_table_shapes():
    """Test lookup table computation produces correct shape."""
    vocab_size = 1000
    hidden_dim = 128
    n_langs = 10

    embeddings = np.random.randn(vocab_size, hidden_dim).astype(np.float32)
    token_weights = np.random.randn(vocab_size, 1).astype(np.float32)
    projection_weight = np.random.randn(n_langs, hidden_dim).astype(np.float32)
    projection_bias = np.random.randn(n_langs).astype(np.float32)

    lookup_table = compute_lookup_table(
        embeddings=embeddings,
        token_weights=token_weights,
        projection_weight=projection_weight,
        projection_bias=projection_bias,
    )

    assert lookup_table.shape == (vocab_size, n_langs)
    assert lookup_table.dtype == np.float32


def test_compute_lookup_table_manual():
    """Test lookup table computation with manual verification."""
    embeddings = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float32,
    )

    token_weights = np.array(
        [
            [0.5],
            [1.0],
            [2.0],
        ],
        dtype=np.float32,
    )

    projection_weight = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    projection_bias = np.array([0.1, 0.2], dtype=np.float32)

    lookup_table = compute_lookup_table(
        embeddings=embeddings,
        token_weights=token_weights,
        projection_weight=projection_weight,
        projection_bias=projection_bias,
    )

    # Manual calculation for first token:
    # weighted_emb[0] = [1.0, 2.0] * 0.5 = [0.5, 1.0]
    # logits[0] = [0.5, 1.0] @ [[1.0, 0.0], [0.0, 1.0]].T + [0.1, 0.2]
    #           = [0.5, 1.0] + [0.1, 0.2] = [0.6, 1.2]
    expected_0 = np.array([0.6, 1.2])
    np.testing.assert_array_almost_equal(lookup_table[0], expected_0)


def test_lookup_table_equivalence_to_inference():
    """Test that lookup table gives same results as original inference."""
    vocab_size = 100
    hidden_dim = 64
    n_langs = 5
    seq_len = 10

    # Create random parameters
    embeddings = np.random.randn(vocab_size, hidden_dim).astype(np.float32)
    token_weights = np.random.randn(vocab_size, 1).astype(np.float32)
    projection_weight = np.random.randn(n_langs, hidden_dim).astype(np.float32)
    projection_bias = np.random.randn(n_langs).astype(np.float32)

    # Generate lookup table
    lookup_table = compute_lookup_table(
        embeddings=embeddings,
        token_weights=token_weights,
        projection_weight=projection_weight,
        projection_bias=projection_bias,
    )

    # Sample token IDs
    token_ids = np.random.randint(0, vocab_size, size=(seq_len,))

    # Method 1: Original inference (embeddings → projection)
    token_embeddings = embeddings[token_ids]  # (seq_len, hidden_dim)
    weighted_embeddings = token_embeddings * token_weights[token_ids]  # (seq_len, hidden_dim)
    logits_original = (
        weighted_embeddings @ projection_weight.T + projection_bias
    )  # (seq_len, n_langs)

    # Method 2: Lookup table
    logits_lookup = lookup_table[token_ids]  # (seq_len, n_langs)

    # Should be identical
    np.testing.assert_array_almost_equal(logits_original, logits_lookup, decimal=5)
