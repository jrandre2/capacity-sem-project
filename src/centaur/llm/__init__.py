"""
LLM provider package for the vendored CENTAUR framework.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..config import LLM_MAX_TOKENS, LLM_MODELS, LLM_PROVIDER, LLM_TEMPERATURE

if TYPE_CHECKING:
    from .base import LLMProvider


def get_provider(provider_name: Optional[str] = None) -> "LLMProvider":
    """
    Get an LLM provider instance using the vendored CENTAUR configuration.
    """
    provider_name = provider_name or LLM_PROVIDER
    model = LLM_MODELS.get(provider_name)

    if provider_name == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(
            model=model,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

    if provider_name == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(
            model=model,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

    raise ValueError(
        f"Unknown provider: {provider_name}. Supported providers: 'anthropic', 'openai'"
    )


__all__ = ["get_provider"]
