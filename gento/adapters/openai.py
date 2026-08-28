import os
from typing import Optional

from openai import AsyncOpenAI

from ..exceptions import APIError, SchemaParseError, UnsupportedCapabilityError
from ..models import (
    GenerateRequest,
    GenerateResponse,
    ModelCapabilities,
    ToolCall,
)
from .base import BaseAdapter


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI models via official openai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        capabilities: Optional[ModelCapabilities] = None,
    ):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY must be set in environment or passed to client."
            )

        url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = AsyncOpenAI(api_key=key, base_url=url)
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

    @staticmethod
    def normalize_model_name(raw_model: str) -> str:
        """Strip provider prefix if present (e.g., openai/gpt-4o -> gpt-4o)."""
        if raw_model.startswith("openai/"):
            return raw_model[len("openai/") :]
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

        try:
            if request.response_schema:
                kwargs["response_format"] = request.response_schema
                response = await self.client.beta.chat.completions.parse(**kwargs)
            else:
                response = await self.client.chat.completions.create(**kwargs)
        except Exception as e:
            raise APIError(f"OpenAI API error: {str(e)}") from e

        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None
        content = message.content if message else None

        parsed = getattr(message, "parsed", None)
        refusal = getattr(message, "refusal", None)

        if refusal:
            raise SchemaParseError(f"Model refused generation: {refusal}")

        if request.response_schema and parsed is None:
            if content:
                try:
                    parsed = request.response_schema.model_validate_json(content)
                except Exception as parse_err:
                    raise SchemaParseError(
                        f"Failed to parse OpenAI response into schema {request.response_schema}: {parse_err}"
                    ) from parse_err
            else:
                raise SchemaParseError(
                    f"OpenAI response did not conform to schema for model '{request.model}'."
                )

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
