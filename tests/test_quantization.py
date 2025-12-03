"""Tests for quantization utilities."""

import numpy as np

from langtoken.inference.quantization import (
    dequantize_fp8,
    quantize_fp8_e4m3fn,
    quantize_fp8_e5m2,
    quantize_fp16,
)


def test_quantize_fp16():
    """Test fp16 quantization."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x_fp16 = quantize_fp16(x)

    assert x_fp16.dtype == np.float16
    np.testing.assert_array_almost_equal(x_fp16.astype(np.float32), x, decimal=3)


def test_quantize_fp8_e4m3fn():
    """Test fp8_e4m3fn quantization."""
    import ml_dtypes

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x_fp8 = quantize_fp8_e4m3fn(x)

    assert x_fp8.dtype == ml_dtypes.float8_e4m3fn

    # Dequantize and check approximate equality
    x_recovered = dequantize_fp8(x_fp8)
    np.testing.assert_array_almost_equal(x_recovered, x, decimal=1)


def test_quantize_fp8_e5m2():
    """Test fp8_e5m2 quantization."""
    import ml_dtypes

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x_fp8 = quantize_fp8_e5m2(x)

    assert x_fp8.dtype == ml_dtypes.float8_e5m2

    # Dequantize and check approximate equality
    x_recovered = dequantize_fp8(x_fp8)
    np.testing.assert_array_almost_equal(x_recovered, x, decimal=1)


def test_quantize_dequantize_roundtrip():
    """Test quantization/dequantization preserves approximate values."""
    x = np.random.randn(100, 50).astype(np.float32)

    # fp16
    x_fp16 = quantize_fp16(x)
    x_recovered_fp16 = x_fp16.astype(np.float32)
    assert np.allclose(x, x_recovered_fp16, rtol=1e-3, atol=1e-3)

    # fp8_e4m3fn
    x_fp8_e4m3fn = quantize_fp8_e4m3fn(x)
    x_recovered_fp8_e4m3fn = dequantize_fp8(x_fp8_e4m3fn)
    assert np.allclose(x, x_recovered_fp8_e4m3fn, rtol=0.1, atol=0.1)

    # fp8_e5m2
    x_fp8_e5m2 = quantize_fp8_e5m2(x)
    x_recovered_fp8_e5m2 = dequantize_fp8(x_fp8_e5m2)
    assert np.allclose(x, x_recovered_fp8_e5m2, rtol=0.1, atol=0.1)
