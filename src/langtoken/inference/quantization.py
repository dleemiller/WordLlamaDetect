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


def quantize_float4_e2m1fn(array: np.ndarray) -> np.ndarray:
    """Quantize fp32 array to float4_e2m1fn format (MX).

    Args:
        array: Input array (fp32)

    Returns:
        Quantized array (float4_e2m1fn dtype)
    """
    import ml_dtypes

    return array.astype(ml_dtypes.float4_e2m1fn)


def quantize_float6_e2m3fn(array: np.ndarray) -> np.ndarray:
    """Quantize fp32 array to float6_e2m3fn format (MX).

    Args:
        array: Input array (fp32)

    Returns:
        Quantized array (float6_e2m3fn dtype)
    """
    import ml_dtypes

    return array.astype(ml_dtypes.float6_e2m3fn)


def quantize_float6_e3m2fn(array: np.ndarray) -> np.ndarray:
    """Quantize fp32 array to float6_e3m2fn format (MX).

    Args:
        array: Input array (fp32)

    Returns:
        Quantized array (float6_e3m2fn dtype)
    """
    import ml_dtypes

    return array.astype(ml_dtypes.float6_e3m2fn)


def dequantize(array: np.ndarray) -> np.ndarray:
    """Dequantize any quantized array back to fp32.

    Works with fp8, float4, float6, and fp16 formats.

    Args:
        array: Quantized array (any ml_dtypes or fp16)

    Returns:
        fp32 array
    """
    return array.astype(np.float32)
