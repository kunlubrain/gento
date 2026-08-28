import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type

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
    provider: str
    api_key_env_vars: Tuple[str, ...]
    base_url_env_vars: Tuple[str, ...] = ()
    default_base_url: Optional[str] = None

    def resolve_api_key(self) -> Optional[str]:
        """Resolve API key from explicit argument or known environment variables."""
        for env_var in self.api_key_env_vars:
            val = os.getenv(env_var)
            if val:
                return val
        return None

    def resolve_base_url(
        self, explicit_base_url: Optional[str] = None
    ) -> Optional[str]:
        """Resolve base URL from explicit argument, known environment variables, or default."""
        if explicit_base_url:
            return explicit_base_url
        for env_var in self.base_url_env_vars:
            val = os.getenv(env_var)
            if val:
                return val
        return self.default_base_url


DEFAULT_CAPABILITIES = ModelCapabilities(
    structured_output=True,
    tool_calling=True,
    streaming=True,
    vision=True,
    search=True,
)

# Provider constants and default configurations
GOOGLE_PROVIDER = "google"
OPENAI_PROVIDER = "openai"
VOLCENGINE_PROVIDER = "volcengine"

GEMINI_MODEL_DEF = ModelDefinition(
    adapter_class=GeminiAdapter,
    capabilities=DEFAULT_CAPABILITIES,
    provider=GOOGLE_PROVIDER,
    api_key_env_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    base_url_env_vars=("GEMINI_BASE_URL", "GOOGLE_BASE_URL"),
)

OPENAI_MODEL_DEF = ModelDefinition(
    adapter_class=OpenAIAdapter,
    capabilities=DEFAULT_CAPABILITIES,
    provider=OPENAI_PROVIDER,
    api_key_env_vars=("OPENAI_API_KEY",),
    base_url_env_vars=("OPENAI_BASE_URL",),
)

ARK_MODEL_DEF = ModelDefinition(
    adapter_class=ArkAdapter,
    capabilities=DEFAULT_CAPABILITIES,
    provider=VOLCENGINE_PROVIDER,
    api_key_env_vars=("VOLC_API_KEY", "ARK_API_KEY", "VOLCENGINE_API_KEY"),
    base_url_env_vars=("VOLC_BASE_URL", "ARK_BASE_URL", "VOLCENGINE_BASE_URL"),
    default_base_url="https://ark.cn-beijing.volces.com/api/v3",
)

MODEL_REGISTRY: Dict[str, ModelDefinition] = {
    "google/gemini-3.5-flash": GEMINI_MODEL_DEF,
    "google/gemini-3.6-flash": GEMINI_MODEL_DEF,
    "google/gemini-3.7-flash": GEMINI_MODEL_DEF,
    "openai/gpt-4o": OPENAI_MODEL_DEF,
    "openai/gpt-4o-mini": OPENAI_MODEL_DEF,
    "openai/gpt-5": OPENAI_MODEL_DEF,
    "openai/gpt-4o-mini": OPENAI_MODEL_DEF,
    "openai/gpt-4o": OPENAI_MODEL_DEF,
    "volcengine/doubao-1.5-pro-32k": ARK_MODEL_DEF,
    "volcengine/doubao-pro-32k": ARK_MODEL_DEF,
}


def resolve_model_definition(model: str) -> ModelDefinition:
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model]
    raise UnsupportedModelError(f"Unsupported {model=}")


def create_adapter(
    model: str,
) -> BaseAdapter:
    """Factory function to build an adapter for a given model, resolving env vars automatically."""
    definition = resolve_model_definition(model)
    adapter_cls = definition.adapter_class

    resolved_api_key = definition.resolve_api_key()
    resolved_base_url = definition.resolve_base_url()

    if not resolved_api_key:
        raise ValueError(
            f"Missing API key for {model=}"
            f"Set one of the environment variables: {definition.api_key_env_vars}"
        )

    kwargs: dict = {
        "capabilities": definition.capabilities,
        "api_key": resolved_api_key,
    }
    if resolved_base_url is not None:
        kwargs["base_url"] = resolved_base_url

    return adapter_cls(**kwargs)
