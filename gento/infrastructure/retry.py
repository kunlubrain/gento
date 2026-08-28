import logging
from collections.abc import Callable, Coroutine
from typing import Any

from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ..exceptions import APIError, RateLimitError, SchemaParseError

logger = logging.getLogger("gento.retry")


def build_async_retryer(retry_count: int = 3) -> AsyncRetrying:
    """Build an AsyncRetrying instance configured with exponential backoff and jitter."""
    return AsyncRetrying(
        stop=stop_after_attempt(1 + max(0, retry_count)),
        wait=wait_exponential_jitter(initial=1.0, max=10.0),
        retry=retry_if_exception_type((SchemaParseError, RateLimitError, APIError)),
        before_sleep=before_sleep_log(logger, logging.WARNING, exc_info=False),
        reraise=True,
    )


async def execute_with_retry(
    func: Callable[..., Coroutine[Any, Any, Any]],
    *args: Any,
    retry_count: int = 3,
    **kwargs: Any,
) -> Any:
    """Execute an async function wrapped with tenacity retry logic."""
    retryer = build_async_retryer(retry_count=retry_count)
    return await retryer(func, *args, **kwargs)
