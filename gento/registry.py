from dataclasses import dataclass
from typing import Dict, Optional, Type

from .adapters.ark import ArkAdapter
from .adapters.base import BaseAdapter
from .adapters.gemini import GeminiAdapter
from .adapters.openai import OpenAIAdapter
from .exceptions import UnsupportedModelError
from .models import ModelCapabilities


@dataclass(frozen=True)
class ModelDefinition:
    adapter_class: Type[BaseAdapter]
    capabilities: ModelCapabilities


DEFAULT_CAPABILITIES = ModelCapabilities(
    structured_output=True,
    tool_calling=True,
    streaming=True,
    vision=True,
    search=True,
)

# Known model mapping registry for fast lookup
MODEL_REGISTRY: Dict[str, ModelDefinition] = {
    # Gemini models
    "google/gemini-3.5-flash": ModelDefinition(GeminiAdapter, DEFAULT_CAPABILITIES),
    "gemini-3.5-flash": ModelDefinition(GeminiAdapter, DEFAULT_CAPABILITIES),
    "google/gemini-2.5-flash": ModelDefinition(GeminiAdapter, DEFAULT_CAPABILITIES),
    "gemini-2.5-flash": ModelDefinition(GeminiAdapter, DEFAULT_CAPABILITIES),
    "google/gemini-2.5-pro": ModelDefinition(GeminiAdapter, DEFAULT_CAPABILITIES),
    "gemini-2.5-pro": ModelDefinition(GeminiAdapter, DEFAULT_CAPABILITIES),
    # OpenAI models
    "openai/gpt-4o": ModelDefinition(OpenAIAdapter, DEFAULT_CAPABILITIES),
    "gpt-4o": ModelDefinition(OpenAIAdapter, DEFAULT_CAPABILITIES),
    "openai/gpt-4o-mini": ModelDefinition(OpenAIAdapter, DEFAULT_CAPABILITIES),
    "gpt-4o-mini": ModelDefinition(OpenAIAdapter, DEFAULT_CAPABILITIES),
    "gpt-5": ModelDefinition(OpenAIAdapter, DEFAULT_CAPABILITIES),
    "o3-mini": ModelDefinition(OpenAIAdapter, DEFAULT_CAPABILITIES),
    "o1": ModelDefinition(OpenAIAdapter, DEFAULT_CAPABILITIES),
    # Ark / Volcengine models
    "volcengine/doubao-1.5-pro-32k": ModelDefinition(ArkAdapter, DEFAULT_CAPABILITIES),
    "ark/doubao-1.5-pro-32k": ModelDefinition(ArkAdapter, DEFAULT_CAPABILITIES),
    "doubao-1.5-pro-32k": ModelDefinition(ArkAdapter, DEFAULT_CAPABILITIES),
    "doubao-pro-32k": ModelDefinition(ArkAdapter, DEFAULT_CAPABILITIES),
}


def resolve_model_definition(model: str) -> ModelDefinition:
    """Resolve model name to its ModelDefinition using registry or prefix pattern matching."""
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model]

    lower_model = model.lower()

    # Prefix and pattern matching for dynamic / future models
    if (
        lower_model.startswith("google/")
        or lower_model.startswith("gemini/")
        or "gemini" in lower_model
    ):
        return ModelDefinition(GeminiAdapter, DEFAULT_CAPABILITIES)

    if (
        lower_model.startswith("openai/")
        or lower_model.startswith("gpt-")
        or lower_model.startswith("o1")
        or lower_model.startswith("o3")
    ):
        return ModelDefinition(OpenAIAdapter, DEFAULT_CAPABILITIES)

    if (
        lower_model.startswith("volcengine/")
        or lower_model.startswith("ark/")
        or lower_model.startswith("ep-")
        or "doubao" in lower_model
    ):
        return ModelDefinition(ArkAdapter, DEFAULT_CAPABILITIES)

    raise UnsupportedModelError(
        f"Unsupported or unknown model '{model}'. "
        f"Supported provider prefixes include 'google/', 'openai/', 'volcengine/', 'ark/'."
    )


def create_adapter(
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> BaseAdapter:
    """Factory function to build an adapter for a given model."""
    definition = resolve_model_definition(model)
    adapter_cls = definition.adapter_class

    kwargs: dict = {"capabilities": definition.capabilities}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    return adapter_cls(**kwargs)
