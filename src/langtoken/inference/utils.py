"""NumPy utility functions for inference."""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax values for array x.

    Args:
        x: Input array
        axis: Axis along which to compute softmax

    Returns:
        Softmax probabilities
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def max_pool(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Max pooling along specified axis.

    Args:
        x: Input array (seq_len, hidden_dim)
        axis: Axis to pool over

    Returns:
        Max-pooled array
    """
    return np.max(x, axis=axis)


def avg_pool(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Average pooling along specified axis.

    Args:
        x: Input array (seq_len, hidden_dim)
        axis: Axis to pool over

    Returns:
        Average-pooled array
    """
    return np.mean(x, axis=axis)


def apply_projection(
    embeddings: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Apply linear projection (embeddings @ weight.T + bias).

    Args:
        embeddings: Input embeddings (seq_len, hidden_dim) or (hidden_dim,)
        weight: Projection weight matrix (n_languages, hidden_dim)
        bias: Projection bias (n_languages,)

    Returns:
        Projected values (seq_len, n_languages) or (n_languages,)
    """
    # embeddings @ weight.T + bias
    return embeddings @ weight.T + bias
