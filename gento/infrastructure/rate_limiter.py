
from aiolimiter import AsyncLimiter


class RateLimiterManager:
    """Rate limiter manager utilizing AsyncLimiter for rate control."""

    def __init__(self, default_rps: float | None = None):
        self.default_rps = default_rps
        self._limiters: dict[str, AsyncLimiter] = {}

    def get_limiter(
        self, key: str, rps: float | None = None
    ) -> AsyncLimiter | None:
        effective_rps = rps if rps is not None else self.default_rps
        if effective_rps is None or effective_rps <= 0:
            return None

        if key not in self._limiters:
            max_rate = max(1, int(effective_rps))
            self._limiters[key] = AsyncLimiter(max_rate=max_rate, time_period=1.0)

        return self._limiters[key]

    async def acquire(self, key: str, rps: float | None = None) -> None:
        """Acquire a token from the rate limiter if configured."""
        limiter = self.get_limiter(key, rps)
        if limiter:
            await limiter.acquire()
