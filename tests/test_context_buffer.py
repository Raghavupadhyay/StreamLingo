from __future__ import annotations

from app.context.buffer import ContextBuffer
from app.models import TranscriptSegment


def _segment(chunk_id: int, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        chunk_id=chunk_id,
        start_ts=chunk_id * 1.0,
        end_ts=chunk_id * 1.0 + 0.8,
        text=text,
        language="hi",
    )


def test_context_buffer_holds_only_max_sentences() -> None:
    buffer = ContextBuffer(max_sentences=2)
    buffer.add(_segment(1, "namaste"))
    buffer.add(_segment(2, "aap kaise ho"))
    buffer.add(_segment(3, "main theek hoon"))

    window = buffer.get_text_window()
    assert window == "aap kaise ho\nmain theek hoon"


def test_context_buffer_excludes_current_chunk() -> None:
    buffer = ContextBuffer(max_sentences=3)
    buffer.add(_segment(1, "yeh pehla vakya hai"))
    buffer.add(_segment(2, "yeh doosra vakya hai"))

    context_text = buffer.get_text_window(exclude_chunk_id=2)
    assert context_text == "yeh pehla vakya hai"


def test_context_buffer_ignores_empty_segments() -> None:
    buffer = ContextBuffer(max_sentences=3)
    buffer.add(_segment(1, ""))
    buffer.add(_segment(2, "   "))
    buffer.add(_segment(3, "sahi text"))

    assert buffer.get_text_window() == "sahi text"
