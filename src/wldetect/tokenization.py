"""Tokenizer utilities."""

from typing import Any


def disable_chat_template(tokenizer: Any):
    """Disable chat templates or default system prompts if present."""
    if hasattr(tokenizer, "chat_template"):
        tokenizer.chat_template = None
    if hasattr(tokenizer, "use_default_system_prompt"):
        try:
            tokenizer.use_default_system_prompt = False
        except Exception:
            pass
    return tokenizer


__all__ = ["disable_chat_template"]
