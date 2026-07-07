"""In-memory activity feed — plain-language narration of what the bot is doing.

A bounded ring buffer (no disk): the dashboard "bot chat" polls
/live?ev_after=<id> once a second and appends whatever is new. A restart
clears the feed by design — it is live narration, not a journal (trades,
analyses and shadow labels are already persisted in SQLite).
"""
from __future__ import annotations

import collections
from datetime import datetime, timezone


class ActivityFeed:
    def __init__(self, maxlen: int = 400):
        self._buf: collections.deque = collections.deque(maxlen=maxlen)
        self._last_id = 0

    @property
    def top(self) -> int:
        """Highest event id so far (0 = empty). Lets the client detect a
        server restart: if top < the id it already has, it resets."""
        return self._last_id

    def say(self, kind: str, text: str, symbol: str | None = None) -> None:
        self._last_id += 1
        self._buf.append({
            "id": self._last_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "symbol": symbol,
            "text": text,
        })

    def tail(self, after: int = 0, limit: int = 100) -> list:
        """Events with id > after, oldest first, capped at `limit`."""
        evs = [e for e in self._buf if e["id"] > after]
        return evs[-limit:]
