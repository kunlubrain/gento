import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from .adapters.base import BaseAdapter
from .exceptions import SchemaParseError
from .infrastructure.rate_limiter import RateLimiterManager
from .infrastructure.retry import execute_with_retry
from .models import GenerateRequest, GenerateResponse
from .registry import create_adapter
from .tools import Tool

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("gento.client")


class LLMClient:
    """Unified LLM Client supporting Gemini, OpenAI, and Volcengine/Ark models."""

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        rate_limit_rps: Optional[float] = None,
        max_retries: int = 3,
    ):
        """Initialize LLMClient.

        API keys and base URLs are automatically resolved from environment variables
        (e.g., loaded via dotenv or shell environment) using the known model registry mapping.

        :param model: Default model name for subsequent generation calls.
                      Defaults to "google/gemini-3.5-flash" if None.
        :param rate_limit_rps: Optional rate limit in requests per second.
        :param max_retries: Default retry count for failed schema validation or rate limits.
        """
        self.default_model = model or "google/gemini-3.5-flash"
        self.rate_limiter = RateLimiterManager(default_rps=rate_limit_rps)
        self.default_max_retries = max_retries
        self._adapters: Dict[str, BaseAdapter] = {}

    def _get_adapter(self, model: str) -> BaseAdapter:
        if model in self._adapters:
            return self._adapters[model]

        adapter = create_adapter(model=model)
        self._adapters[model] = adapter
        return adapter

    async def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        response_schema: Optional[Type[T]] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        temperature: Optional[float] = None,
        enable_search: bool = False,
        retry_count: Optional[int] = None,
        rate_limit_rps: Optional[float] = None,
    ) -> GenerateResponse:
        """Asynchronously generate content or structured output from an LLM model."""
        target_model = model or self.default_model
        adapter = self._get_adapter(target_model)
        retries = retry_count if retry_count is not None else self.default_max_retries

        await self.rate_limiter.acquire(target_model, rps=rate_limit_rps)

        request = GenerateRequest(
            model=target_model,
            prompt=prompt,
            response_schema=response_schema,
            system_instruction=system_instruction,
            tools=tools or [],
            temperature=temperature,
            enable_search=enable_search,
        )

        return await execute_with_retry(
            adapter.generate,
            request,
            retry_count=retries,
        )

    async def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        *,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        temperature: Optional[float] = None,
        enable_search: bool = False,
        retry_count: Optional[int] = None,
        rate_limit_rps: Optional[float] = None,
    ) -> T:
        """Generate structured output validated against a Pydantic model."""
        target_model = model or self.default_model
        response = await self.generate(
            prompt,
            model=target_model,
            response_schema=response_schema,
            system_instruction=system_instruction,
            tools=tools,
            temperature=temperature,
            enable_search=enable_search,
            retry_count=retry_count,
            rate_limit_rps=rate_limit_rps,
        )

        if response.parsed is None:
            raise SchemaParseError(
                f"Model '{target_model}' failed to return structured output conforming to {response_schema}."
            )

        return response.parsed  # type: ignore

    async def generate_dict(
        self,
        prompt: str,
        response_schema: Type[T],
        *,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        temperature: Optional[float] = None,
        enable_search: bool = False,
        retry_count: Optional[int] = None,
        rate_limit_rps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate structured output and return it as a python dictionary."""
        structured = await self.generate_structured(
            prompt,
            response_schema,
            model=model,
            system_instruction=system_instruction,
            tools=tools,
            temperature=temperature,
            enable_search=enable_search,
            retry_count=retry_count,
            rate_limit_rps=rate_limit_rps,
        )
        return structured.model_dump()

    def generate_sync(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> GenerateResponse:
        """Synchronous wrapper for generate."""
        return asyncio.run(self.generate(prompt, **kwargs))

    def generate_structured_sync(
        self,
        prompt: str,
        response_schema: Type[T],
        **kwargs: Any,
    ) -> T:
        """Synchronous wrapper for generate_structured."""
        return asyncio.run(self.generate_structured(prompt, response_schema, **kwargs))

    def generate_dict_sync(
        self,
        prompt: str,
        response_schema: Type[T],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for generate_dict."""
        return asyncio.run(self.generate_dict(prompt, response_schema, **kwargs))
