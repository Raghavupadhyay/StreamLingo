from __future__ import annotations

import time

import pytest

from app.models import TranslationResult
from app.pipeline.delay_queue import DelayQueue


def _translation(chunk_id: int, text: str) -> TranslationResult:
    return TranslationResult(
        chunk_id=chunk_id,
        start_ts=0.0,
        end_ts=1.0,
        source_text=text,
        translated_text=text,
        source_language="hi",
        target_language="en",
        metadata={},
    )


@pytest.mark.asyncio
async def test_delay_queue_emits_items_by_ready_time() -> None:
    queue = DelayQueue()
    now = time.monotonic()

    await queue.put(_translation(1, "first"), emit_at=now + 0.01)
    await queue.put(_translation(2, "second"), emit_at=now + 0.02)

    out1 = await queue.get_ready()
    out2 = await queue.get_ready()

    assert out1.chunk_id == 1
    assert out2.chunk_id == 2
