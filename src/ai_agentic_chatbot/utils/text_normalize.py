"""Prompt normalization for prompt-cache key generation."""


def normalize_prompt(text: str) -> str:
    """Lowercase, trim, and collapse whitespace.

    Deliberately no punctuation stripping or synonym handling — v1 ships
    simple; "top 10" and "top ten" are different cache keys.
    """
    return " ".join(text.strip().lower().split())