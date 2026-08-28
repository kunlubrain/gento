from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Tool:
    """Definition for custom function/tool declarations passed to LLMs."""

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Awaitable[Any]] | None = None
