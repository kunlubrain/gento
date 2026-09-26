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


NON_RETRYABLE_STATUS_CODES = {
    400,  # Bad Request (invalid prompt, schema, parameters)
    401,  # Unauthorized (invalid API key)
    403,  # Forbidden / Permission Denied
    404,  # Not Found (model does not exist, bad endpoint)
    422,  # Unprocessable Entity
    429,  # Rate limit exceeded / Resource exhausted
    503,  # Service Unavailable / Overloaded
}


def is_retryable_exception(exc: BaseException) -> bool:
    """Determine whether an exception should be retried.

    Do not retry errors that will not succeed on immediate retry:
    - 400 (bad request), 401 (unauthorized), 403 (forbidden), 404 (not found),
      422 (unprocessable), 429 (rate limit), 503 (service unavailable).
    - RateLimitError.
    """
    if isinstance(exc, RateLimitError):
        return False

    if not isinstance(exc, (SchemaParseError, APIError)):
        return False

    current: BaseException | None = exc
    while current is not None:
        status_code = getattr(current, "status_code", None) or getattr(current, "code", None)
        if status_code in NON_RETRYABLE_STATUS_CODES:
            return False

        msg = str(current)
        if re.search(r"\b(400|401|403|404|422|429|503)\b", msg):
            return False
        if re.search(r"resource[ _-]?exhausted", msg, re.IGNORECASE):
            return False
        if re.search(r"rate[ _-]?limit", msg, re.IGNORECASE):
            return False
        if re.search(r"service[ _-]?unavailable", msg, re.IGNORECASE):
            return False
        if re.search(r"not[ _-]?found", msg, re.IGNORECASE):
            return False
        if re.search(r"permission[ _-]?denied|unauthorized|forbidden", msg, re.IGNORECASE):
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
