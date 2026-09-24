import logging
import re
from collections.abc import Callable, Coroutine
from typing import Any

from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_chain,
    wait_random,
)

from ..exceptions import APIError, RateLimitError, SchemaParseError

logger = logging.getLogger("gento.retry")


def is_retryable_exception(exc: BaseException) -> bool:
    """Determine whether an exception should be retried.

    Do not retry 429 (rate limit) or 503 (service unavailable) errors.
    """
    if isinstance(exc, RateLimitError):
        return False

    if not isinstance(exc, (SchemaParseError, APIError)):
        return False

    current: BaseException | None = exc
    while current is not None:
        status_code = getattr(current, "status_code", None) or getattr(current, "code", None)
        if status_code in (429, 503):
            return False

        msg = str(current)
        if re.search(r"\b(429|503)\b", msg):
            return False
        if re.search(r"resource[ _-]?exhausted", msg, re.IGNORECASE):
            return False
        if re.search(r"rate[ _-]?limit", msg, re.IGNORECASE):
            return False
        if re.search(r"service[ _-]?unavailable", msg, re.IGNORECASE):
            return False

        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)

    return True


def build_async_retryer(retry_count: int = 3) -> AsyncRetrying:
    """Build an AsyncRetrying instance configured with custom wait jitter and exception filter.

    Wait logic:
    - Attempt 1 wait: random jitter between 2.0s and 6.0s (average 4.0s).
    - Attempt 2+ wait: random jitter between 5.0s and 15.0s (average 10.0s, max 15.0s).
    """
    return AsyncRetrying(
        stop=stop_after_attempt(1 + max(0, retry_count)),
        wait=wait_chain(
            wait_random(2.0, 6.0),
            wait_random(5.0, 15.0),
        ),
        retry=retry_if_exception(is_retryable_exception),
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
