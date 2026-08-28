"""Custom exceptions for Gento."""


class GentoError(Exception):
    """Base exception for all Gento errors."""


class SchemaParseError(GentoError):
    """Raised when a model response cannot be parsed into the requested Pydantic schema."""


class UnsupportedModelError(GentoError):
    """Raised when a model name is invalid or cannot be resolved to an adapter."""


class UnsupportedCapabilityError(GentoError):
    """Raised when a model does not support a requested capability."""


class RateLimitError(GentoError):
    """Raised when rate limit is exceeded."""


class APIError(GentoError):
    """Raised when an underlying provider API fails."""
