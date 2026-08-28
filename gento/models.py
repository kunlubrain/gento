from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
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
    response_schema: Optional[Type[BaseModel]] = None
    system_instruction: Optional[str] = None
    tools: List[Tool] = field(default_factory=list)
    temperature: Optional[float] = None
    enable_search: bool = False
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """Structured tool invocation returned by an LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class GenerateResponse:
    """Unified response object returned by LLMClient."""

    content: Optional[str] = None
    parsed: Optional[BaseModel] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw_response: Any = None
