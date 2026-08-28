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
