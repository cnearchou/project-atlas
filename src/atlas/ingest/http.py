"""Shared HTTP helper with polite rate limiting and retries.

Several public APIs set request limits (SEC EDGAR allows about 10 requests per second and requires
a User-Agent with a contact email; Technical Reference, Part 2.2). This helper enforces a minimum
gap between requests and retries temporary failures with exponential back-off.
"""

from __future__ import annotations

import time

import requests


class RateLimitedSession:
    def __init__(self, min_interval: float = 0.12, headers: dict | None = None, retries: int = 3):
        self.session = requests.Session()
        self.session.headers.update(headers or {})
        self.min_interval = min_interval
        self.retries = retries
        self._last = 0.0

    def get(self, url: str, **kwargs) -> requests.Response:
        for attempt in range(self.retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            resp = self.session.get(url, timeout=kwargs.pop("timeout", 30), **kwargs)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError("unreachable")
