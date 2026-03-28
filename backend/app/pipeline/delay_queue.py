from __future__ import annotations

import asyncio
import heapq
import time

from app.models import ScheduledSubtitle, TranslationResult


class DelayQueue:
    def __init__(self) -> None:
        self._heap: list[ScheduledSubtitle] = []
        self._lock = asyncio.Lock()
        self._wake_event = asyncio.Event()

    async def put(self, payload: TranslationResult, emit_at: float) -> None:
        async with self._lock:
            heapq.heappush(
                self._heap,
                ScheduledSubtitle(
                    emit_at=emit_at,
                    sequence_id=payload.chunk_id,
                    payload=payload,
                ),
            )
            self._wake_event.set()

    async def get_ready(self) -> TranslationResult:
        while True:
            await self._wait_until_ready()
            async with self._lock:
                if not self._heap:
                    continue
                item = self._heap[0]
                if item.emit_at <= time.monotonic():
                    heapq.heappop(self._heap)
                    return item.payload

    async def _wait_until_ready(self) -> None:
        async with self._lock:
            if not self._heap:
                next_wait = None
            else:
                next_wait = max(0.0, self._heap[0].emit_at - time.monotonic())

        if next_wait is None:
            self._wake_event.clear()
            await self._wake_event.wait()
            return
        if next_wait == 0.0:
            return

        self._wake_event.clear()
        try:
            await asyncio.wait_for(self._wake_event.wait(), timeout=next_wait)
        except TimeoutError:
            return
