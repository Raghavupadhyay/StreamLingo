from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from app.models import TranscriptSegment


@dataclass
class ContextItem:
    chunk_id: int
    text: str
    start_ts: float
    end_ts: float
    language: str


class ContextBuffer:
    def __init__(self, max_sentences: int = 6) -> None:
        self.max_sentences = max(1, max_sentences)
        self._items: deque[ContextItem] = deque(maxlen=self.max_sentences)

    def add(self, segment: TranscriptSegment) -> None:
        if not segment.text.strip():
            return
        self._items.append(
            ContextItem(
                chunk_id=segment.chunk_id,
                text=segment.text.strip(),
                start_ts=segment.start_ts,
                end_ts=segment.end_ts,
                language=segment.language,
            )
        )

    def get_text_window(self, exclude_chunk_id: int | None = None) -> str:
        lines: list[str] = []
        for item in self._items:
            if exclude_chunk_id is not None and item.chunk_id == exclude_chunk_id:
                continue
            lines.append(item.text)
        return "\n".join(lines)
