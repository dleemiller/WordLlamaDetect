"""Detect command - language detection on text."""

from wldetect import WLDetect
from wldetect.cli.utils import setup_logging


def run(args) -> int:
    """Execute detection command.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger = setup_logging()

    model_desc = "bundled model" if args.model_path is None else args.model_path
    logger.info(f"Loading model from {model_desc}...")
    detector = WLDetect.load(args.model_path)

    # Get text from args or file
    if args.text:
        text = args.text
        source = f"Text: {args.text}"
    elif args.file:
        with open(args.file) as f:
            text = f.read()
        source = f"File: {args.file}"
    else:
        logger.error("Error: Either --text or --file must be provided")
        return 1

    # Detect language
    result = detector.predict(text)

    if result is None:
        logger.info(f"\n{source}")
        logger.info("Detected language: None (empty text)")
    else:
        top_lang, confidence = result
        logger.info(f"\n{source}")
        logger.info(f"Detected language: {top_lang} (confidence: {confidence:.2%})")

    return 0
