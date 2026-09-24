from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from gento.exceptions import SchemaParseError
from gento.infrastructure.rate_limiter import RateLimiterManager
from gento.infrastructure.retry import execute_with_retry


class SampleSchema(BaseModel):
    val: int


@pytest.mark.asyncio
async def test_rate_limiter_manager():
    manager = RateLimiterManager(default_rps=10)
    limiter = manager.get_limiter("gemini-3.5-flash")
    assert limiter is not None
    await manager.acquire("gemini-3.5-flash")


@pytest.mark.asyncio
async def test_tenacity_retry_on_schema_parse_error():
    mock_func = AsyncMock()
    # Fail twice with SchemaParseError, then succeed on 3rd attempt
    mock_func.side_effect = [
        SchemaParseError("Attempt 1 failed"),
        SchemaParseError("Attempt 2 failed"),
        "success_result",
    ]

    result = await execute_with_retry(mock_func, retry_count=3)
    assert result == "success_result"
    assert mock_func.call_count == 3


@pytest.mark.asyncio
async def test_tenacity_retry_exhausted_raises():
    mock_func = AsyncMock()
    mock_func.side_effect = SchemaParseError("Parsing error")

    with pytest.raises(SchemaParseError):
        await execute_with_retry(mock_func, retry_count=2)

    assert mock_func.call_count == 3


@pytest.mark.asyncio
async def test_no_retry_on_rate_limit_error():
    from gento.exceptions import RateLimitError

    mock_func = AsyncMock()
    mock_func.side_effect = RateLimitError("Rate limit reached")

    with pytest.raises(RateLimitError):
        await execute_with_retry(mock_func, retry_count=3)

    # Should NOT retry at all
    assert mock_func.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_factory",
    [
        lambda: APIError("Gemini API error: 429 ResourceExhausted"),
        lambda: APIError("OpenAI API error: 503 Service Unavailable"),
        lambda: APIError("rate limit exceeded"),
        lambda: APIError("Service Unavailable"),
        lambda: APIError("429 Too Many Requests"),
        lambda: APIError("503 Backend Error"),
    ],
)
async def test_no_retry_on_429_and_503_api_errors(error_factory):
    mock_func = AsyncMock()
    mock_func.side_effect = error_factory()

    with pytest.raises(APIError):
        await execute_with_retry(mock_func, retry_count=3)

    # Should NOT retry at all for 429 and 503
    assert mock_func.call_count == 1


@pytest.mark.asyncio
async def test_retry_on_retryable_api_error():
    mock_func = AsyncMock()
    # 500 error or other transient error should retry
    mock_func.side_effect = [
        APIError("Temporary connection reset"),
        "recovered",
    ]

    result = await execute_with_retry(mock_func, retry_count=3)
    assert result == "recovered"
    assert mock_func.call_count == 2


def test_retry_wait_jitter_bouncing():
    from tenacity import RetryCallState
    from gento.infrastructure.retry import build_async_retryer

    retryer = build_async_retryer(retry_count=3)
    wait_strategy = retryer.wait

    state = RetryCallState(None, None, None, None)

    # Attempt 1: average 4 seconds (between 2s and 6s)
    state.attempt_number = 1
    attempt_1_waits = [wait_strategy(state) for _ in range(100)]
    assert all(2.0 <= w <= 6.0 for w in attempt_1_waits)
    avg_1 = sum(attempt_1_waits) / len(attempt_1_waits)
    assert 3.5 <= avg_1 <= 4.5

    # Attempt 2+: average 10 seconds, no more than 15 seconds (between 5s and 15s)
    state.attempt_number = 2
    attempt_2_waits = [wait_strategy(state) for _ in range(100)]
    assert all(5.0 <= w <= 15.0 for w in attempt_2_waits)
    avg_2 = sum(attempt_2_waits) / len(attempt_2_waits)
    assert 9.0 <= avg_2 <= 11.0

    state.attempt_number = 3
    attempt_3_waits = [wait_strategy(state) for _ in range(100)]
    assert all(5.0 <= w <= 15.0 for w in attempt_3_waits)

