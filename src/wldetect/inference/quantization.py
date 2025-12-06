"""Quantization utilities for language detection inference."""

import numpy as np


def quantize_fp8_e4m3fn(array: np.ndarray) -> np.ndarray:
    """Quantize fp32 array to float8_e4m3fn format.

    Args:
        array: Input array (fp32)

    Returns:
        Quantized array (float8_e4m3fn dtype)
    """
    import ml_dtypes

    return array.astype(ml_dtypes.float8_e4m3fn)


def quantize_fp8_e5m2(array: np.ndarray) -> np.ndarray:
    """Quantize fp32 array to float8_e5m2 format.

    Args:
        array: Input array (fp32)

    Returns:
        Quantized array (float8_e5m2 dtype)
    """
    import ml_dtypes

    return array.astype(ml_dtypes.float8_e5m2)


def dequantize_fp8(array: np.ndarray) -> np.ndarray:
    """Dequantize fp8 array back to fp32.

    Args:
        array: fp8 array (float8_e4m3fn or float8_e5m2)

    Returns:
        fp32 array
    """
    return array.astype(np.float32)


def quantize_fp16(array: np.ndarray) -> np.ndarray:
    """Quantize fp32 array to fp16.

    Args:
        array: Input array (fp32)

    Returns:
        Quantized array (fp16 dtype)
    """
    return array.astype(np.float16)
