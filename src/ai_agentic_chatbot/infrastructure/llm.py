"""Legacy LLM module - use llm package for new implementations."""

import warnings
from .llm import get_llm, LLMProvider


# Backward compatibility
def get_azure_llm():
    """Legacy function - use get_llm() instead."""
    warnings.warn(
        "get_azure_llm() is deprecated. Use get_llm() from llm package instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_llm(provider=LLMProvider.AZURE).get_client()
