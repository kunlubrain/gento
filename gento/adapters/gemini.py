from google import genai
from google.genai import types as gemini_types

from ..exceptions import APIError, SchemaParseError, UnsupportedCapabilityError
from ..models import (
    GenerateRequest,
    GenerateResponse,
    ModelCapabilities,
    ToolCall,
)
from .base import BaseAdapter


class GeminiAdapter(BaseAdapter):
    """Adapter for Google Gemini models via google-genai SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        capabilities: ModelCapabilities | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY!")

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

    def _get_client(self) -> genai.Client:
        return genai.Client(api_key=self.api_key)

    @property
    def client(self) -> genai.Client:
        """Dynamically get client bound to active event loop."""
        return self._get_client()

    @staticmethod
    def normalize_model_name(raw_model: str) -> str:
        """Strip provider prefix if present (e.g., google/gemini-3.5-flash -> gemini-3.5-flash)."""
        if raw_model.startswith("google/"):
            return raw_model[len("google/") :]
        if raw_model.startswith("gemini/"):
            return raw_model[len("gemini/") :]
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

        if request.enable_search and not self.capabilities.search:
            raise UnsupportedCapabilityError(
                f"Model '{request.model}' does not support search/grounding."
            )

        model_name = self.normalize_model_name(request.model)
        config_kwargs: dict = {}

        if request.system_instruction:
            config_kwargs["system_instruction"] = request.system_instruction

        if request.temperature is not None:
            config_kwargs["temperature"] = request.temperature

        if request.response_schema:
            json_schema = request.response_schema.model_json_schema()
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = json_schema
            # config_kwargs["temperature"] = 0.2

        tools_list = []
        if request.tools:
            tools_list.append(
                gemini_types.Tool(
                    function_declarations=[
                        gemini_types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.parameters,
                        )
                        for tool in request.tools
                    ]
                )
            )

        if request.enable_search:
            tools_list.append(gemini_types.Tool(google_search=gemini_types.GoogleSearch()))

        if tools_list:
            config_kwargs["tools"] = tools_list

        config = gemini_types.GenerateContentConfig(**config_kwargs)

        try:
            client = self._get_client()
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=request.prompt,
                config=config,
            )
        except Exception as e:
            raise APIError(f"Gemini API error: {str(e)}") from e

        parsed = getattr(response, "parsed", None)
        text_content = getattr(response, "text", None)

        if request.response_schema and text_content:
            try:
                parsed = request.response_schema.model_validate_json(text_content)
            except Exception as parse_err:
                raise SchemaParseError(
                    f"Failed to parse Gemini response into schema {request.response_schema}: {parse_err}"
                ) from parse_err

        tool_calls = self._extract_tool_calls(response)

        return GenerateResponse(
            content=text_content,
            parsed=parsed,
            tool_calls=tool_calls,
            raw_response=response,
        )

    @staticmethod
    def _extract_tool_calls(response) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None) or []
            for index, part in enumerate(parts):
                function_call = getattr(part, "function_call", None)
                if function_call is None:
                    continue
                name = getattr(function_call, "name", None)
                if not name:
                    continue
                arguments = getattr(function_call, "args", None) or {}
                tool_calls.append(
                    ToolCall(
                        id=f"call_{index}",
                        name=name,
                        arguments=dict(arguments),
                    )
                )
        return tool_calls
