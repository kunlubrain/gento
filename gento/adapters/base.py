from abc import ABC, abstractmethod

from ..models import GenerateRequest, GenerateResponse, ModelCapabilities


class BaseAdapter(ABC):
    """Abstract base class for provider-specific LLM adapters."""

    def __init__(self, capabilities: ModelCapabilities | None = None):
        self.capabilities = capabilities or ModelCapabilities()

    @abstractmethod
    async def generate(
        self,
        request: GenerateRequest,
    ) -> GenerateResponse:
        """Execute a text generation or structured parsing request against the provider."""
        raise NotImplementedError
