"""Tests for inference utilities."""

import numpy as np

from wldetect.softmax import softmax


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


def test_softmax_2d():
    """Test softmax on 2D array."""
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = softmax(x, axis=-1)

    # Check each row sums to 1
    assert np.allclose(result.sum(axis=-1), 1.0)

    # Check all values are positive
    assert np.all(result > 0)

    # Check largest values have largest probabilities
    assert np.argmax(result[0]) == 2
    assert np.argmax(result[1]) == 2
