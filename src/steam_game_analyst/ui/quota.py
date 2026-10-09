"""Daily question caps for the public demo. The hosted LLM's free allowance (Workers AI: 10,000
neurons a day, ~100 questions) is shared by every visitor, so one visitor gets a slice of it
and the total stops just short of it: past that, the endpoint would fail mid-answer.

Counts live in memory (a restart resets them) and roll over at 00:00 UTC, when Cloudflare
resets the allowance."""

from __future__ import annotations

import datetime
import threading
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal


def utc_today() -> datetime.date:
    return datetime.datetime.now(datetime.UTC).date()


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: Literal["ok", "visitor", "total"]


class DailyQuota:
    """`take(visitor)` counts one question if both caps allow it (None = no cap)."""

    def __init__(
        self,
        per_visitor: int | None,
        total: int | None,
        today: Callable[[], datetime.date] = utc_today,
    ) -> None:
        self.per_visitor = per_visitor
        self.total = total
        self._today = today
        self._lock = threading.Lock()
        self._day = today()
        self._counts: Counter[str] = Counter()

    def take(self, visitor: str) -> Verdict:
        with self._lock:
            day = self._today()
            if day != self._day:
                self._day, self._counts = day, Counter()
            if self.total is not None and self._counts.total() >= self.total:
                return Verdict(False, "total")
            if self.per_visitor is not None and self._counts[visitor] >= self.per_visitor:
                return Verdict(False, "visitor")
            self._counts[visitor] += 1
            return Verdict(True, "ok")


def visitor_id(headers: Mapping[str, str], ip: object) -> str:
    """The client address: the first X-Forwarded-For entry behind Caddy (which replaces any
    value the client sent), else the socket's peer address."""
    forwarded = headers.get("X-Forwarded-For", "").split(",")[0].strip()
    return forwarded or (ip if isinstance(ip, str) and ip else "local")
