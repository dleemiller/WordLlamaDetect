"""CLI utilities and common helpers."""

import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup rich logging for CLI commands.

    Args:
        level: Logging level (default: INFO)

    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger("wldetect")


def ensure_training_deps(logger: logging.Logger) -> bool:
    """Check if training dependencies are installed.

    Args:
        logger: Logger instance

    Returns:
        True if dependencies available, False otherwise
    """
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer  # noqa: F401

        return True
    except ImportError as e:
        logger.error("Error: Training dependencies not installed. Run: uv sync --extra training")
        logger.error(f"Details: {e}")
        return False


def print_header(logger: logging.Logger, title: str) -> None:
    """Print formatted header.

    Args:
        logger: Logger instance
        title: Header title
    """
    logger.info("\n" + "=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def ensure_output_dir(path: str | Path) -> Path:
    """Ensure output directory exists.

    Args:
        path: Directory path

    Returns:
        Path object with created directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
