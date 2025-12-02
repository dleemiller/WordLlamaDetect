"""Tests for inference utilities."""

import numpy as np

from langtoken.inference.utils import apply_projection, avg_pool, max_pool, softmax


def test_softmax():
    """Test softmax function."""
    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)

    # Check probabilities sum to 1
    assert np.isclose(result.sum(), 1.0)

    # Check all values are positive
    assert np.all(result > 0)

    # Check largest input has largest probability
    assert np.argmax(result) == 2


def test_softmax_numerical_stability():
    """Test softmax with large values."""
    x = np.array([1000.0, 1001.0, 1002.0])
    result = softmax(x)

    assert np.isclose(result.sum(), 1.0)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_max_pool():
    """Test max pooling."""
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.5, 1.5, 2.5],
        ]
    )

    result = max_pool(x, axis=0)
    expected = np.array([4.0, 5.0, 6.0])

    np.testing.assert_array_equal(result, expected)


def test_avg_pool():
    """Test average pooling."""
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0],
        ]
    )

    result = avg_pool(x, axis=0)
    expected = np.array([3.0, 4.0, 5.0])

    np.testing.assert_array_almost_equal(result, expected)


def test_apply_projection():
    """Test linear projection."""
    embeddings = np.array([[1.0, 2.0, 3.0]])  # (1, 3)
    weight = np.array(
        [
            [0.5, 0.5, 0.5],
            [1.0, 0.0, -1.0],
        ]
    )  # (2, 3)
    bias = np.array([0.1, 0.2])  # (2,)

    result = apply_projection(embeddings, weight, bias)

    # Expected: embeddings @ weight.T + bias
    # [[1, 2, 3]] @ [[0.5, 1.0], [0.5, 0.0], [0.5, -1.0]] + [0.1, 0.2]
    # [[3.0, -2.0]] + [0.1, 0.2] = [[3.1, -1.8]]
    expected = np.array([[3.1, -1.8]])

    np.testing.assert_array_almost_equal(result, expected)


def test_apply_projection_sequence():
    """Test projection on sequence of embeddings."""
    embeddings = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )  # (3, 2)
    weight = np.array([[1.0, 1.0]])  # (1, 2)
    bias = np.array([0.0])  # (1,)

    result = apply_projection(embeddings, weight, bias)

    # Each row sums: [3.0], [7.0], [11.0]
    expected = np.array([[3.0], [7.0], [11.0]])

    np.testing.assert_array_almost_equal(result, expected)
