from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional


@dataclass
class Tool:
    """Definition for custom function/tool declarations passed to LLMs."""

    name: str
    description: str
    parameters: Dict[str, Any]
    function: Optional[Callable[..., Awaitable[Any]]] = None
