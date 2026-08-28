from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from .tools import Tool


@dataclass(frozen=True)
class ModelCapabilities:
    """Capabilities supported by a specific LLM model or adapter."""

    structured_output: bool = True
    tool_calling: bool = True
    streaming: bool = True
    vision: bool = True
    search: bool = True


@dataclass
class GenerateRequest:
    """Request payload for model generation."""

    model: str
    prompt: str
    response_schema: type[BaseModel] | None = None
    system_instruction: str | None = None
    tools: list[Tool] = field(default_factory=list)
    temperature: float | None = None
    enable_search: bool = False
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """Structured tool invocation returned by an LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class GenerateResponse:
    """Unified response object returned by LLMClient."""

    content: str | None = None
    parsed: BaseModel | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_response: Any = None
