"""Run the agent in a worker thread so the page can tick a timer while it works.

The worker never touches Streamlit: it only puts (node, update) events on a queue. The script
thread polls the queue about once a second and does all the drawing."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any

OnStep = Callable[[str, dict[str, Any]], None]
Event = tuple[str, dict[str, Any]]


class BackgroundRun:
    def __init__(self, fn: Callable[[OnStep], dict[str, Any]]) -> None:
        """`fn(on_step)` runs the agent and returns its final state."""
        self._fn = fn
        self._events: queue.Queue[Event] = queue.Queue()
        self._thread = threading.Thread(target=self._work, daemon=True, name="sga-agent")
        self.result: dict[str, Any] | None = None
        self.error: BaseException | None = None

    def _work(self) -> None:
        try:
            self.result = self._fn(lambda node, update: self._events.put((node, update)))
        except BaseException as e:  # noqa: BLE001 — re-raised in the script thread
            self.error = e

    def start(self) -> BackgroundRun:
        self._thread.start()
        return self

    def poll(self, timeout: float = 1.0) -> list[Event]:
        """Events since the last poll; waits up to `timeout` s for the first one."""
        try:
            got = [self._events.get(timeout=timeout)]
        except queue.Empty:
            return []
        while True:
            try:
                got.append(self._events.get_nowait())
            except queue.Empty:
                return got

    def finished(self) -> bool:
        return not self._thread.is_alive() and self._events.empty()
