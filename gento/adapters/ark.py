
from openai import AsyncOpenAI

from ..exceptions import APIError, SchemaParseError, UnsupportedCapabilityError
from ..models import (
    GenerateRequest,
    GenerateResponse,
    ModelCapabilities,
    ToolCall,
)
from .base import BaseAdapter


class ArkAdapter(BaseAdapter):
    """Adapter for Volcengine (Ark / Doubao) models via OpenAI-compatible SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        capabilities: ModelCapabilities | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        if not self.api_key:
            raise ValueError(
                "Missing API keys: VOLC_API_KEY, ARK_API_KEY, or VOLCENGINE_API_KEY."
            )

        super().__init__(
            capabilities
            or ModelCapabilities(
                structured_output=True,
                tool_calling=True,
                streaming=True,
                vision=True,
                search=True,
            )
        )

    def _get_client(self) -> AsyncOpenAI:
        key = self.api_key
        if not key:
            raise ValueError(
                "VOLC_API_KEY, ARK_API_KEY, or VOLCENGINE_API_KEY must be set in environment or passed to client."
            )

        url = self.base_url
        return AsyncOpenAI(api_key=key, base_url=url)

    @property
    def client(self) -> AsyncOpenAI:
        """Dynamically get client bound to active event loop."""
        return self._get_client()

    @staticmethod
    def normalize_model_name(raw_model: str) -> str:
        """Strip provider prefix if present (e.g., volcengine/doubao-1.5-pro-32k -> doubao-1.5-pro-32k)."""
        if raw_model.startswith("volcengine/"):
            return raw_model[len("volcengine/") :]
        if raw_model.startswith("ark/"):
            return raw_model[len("ark/") :]
        return raw_model

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        if request.response_schema and not self.capabilities.structured_output:
            raise UnsupportedCapabilityError(
                f"Model '{request.model}' does not support structured output."
            )

        if request.tools and not self.capabilities.tool_calling:
            raise UnsupportedCapabilityError(
                f"Model '{request.model}' does not support tool calling."
            )

        model_name = self.normalize_model_name(request.model)
        messages = []

        if request.system_instruction:
            messages.append({"role": "system", "content": request.system_instruction})

        messages.append({"role": "user", "content": request.prompt})

        kwargs: dict = {
            "model": model_name,
            "messages": messages,
        }

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]

        if request.response_schema:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            client = self._get_client()
            response = await client.chat.completions.create(**kwargs)
        except Exception as e:
            raise APIError(f"Volcengine/Ark API error: {str(e)}") from e

        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None
        content = message.content if message else None
        parsed = None

        if request.response_schema:
            if not content:
                raise SchemaParseError(
                    f"Volcengine/Ark response was empty for schema request on model '{request.model}'."
                )
            try:
                parsed = request.response_schema.model_validate_json(content)
            except Exception as parse_err:
                raise SchemaParseError(
                    f"Failed to parse Volcengine/Ark response into schema {request.response_schema}: {parse_err}"
                ) from parse_err

        tool_calls = self._extract_tool_calls(message)

        return GenerateResponse(
            content=content,
            parsed=parsed,
            tool_calls=tool_calls,
            raw_response=response,
        )

    @staticmethod
    def _extract_tool_calls(message) -> list[ToolCall]:
        if not message:
            return []
        tool_calls: list[ToolCall] = []
        raw_calls = getattr(message, "tool_calls", None) or []
        for call in raw_calls:
            call_id = getattr(call, "id", "")
            function = getattr(call, "function", None)
            if not function:
                continue
            name = getattr(function, "name", "")
            args_str = getattr(function, "arguments", "")
            import json

            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except Exception:
                args = {}
            tool_calls.append(ToolCall(id=call_id, name=name, arguments=args or {}))
        return tool_calls
