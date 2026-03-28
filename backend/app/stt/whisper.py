from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.models import AudioChunk, TranscriptSegment

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - optional dependency
    WhisperModel = None  # type: ignore[assignment]


@dataclass
class WhisperConfig:
    model_name: str
    device: str = "auto"
    compute_type: str = "int8"
    beam_size: int = 3


class WhisperTranscriber:
    def __init__(self, cfg: WhisperConfig) -> None:
        self.cfg = cfg
        self._model: Any = None
        self._is_mock = WhisperModel is None
        if not self._is_mock:
            self._model = WhisperModel(
                cfg.model_name,
                device=cfg.device,
                compute_type=cfg.compute_type,
            )

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    def transcribe(self, chunk: AudioChunk, source_language: str) -> TranscriptSegment:
        if self._is_mock:
            return TranscriptSegment(
                chunk_id=chunk.sequence_id,
                start_ts=chunk.start_ts,
                end_ts=chunk.end_ts,
                text="",
                language=source_language,
                confidence=None,
            )

        audio = np.array(chunk.samples, dtype=np.float32)
        segments, info = self._model.transcribe(
            audio,
            language=source_language,
            vad_filter=True,
            beam_size=max(1, self.cfg.beam_size),
        )
        joined = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        return TranscriptSegment(
            chunk_id=chunk.sequence_id,
            start_ts=chunk.start_ts,
            end_ts=chunk.end_ts,
            text=joined,
            language=info.language or source_language,
            confidence=info.language_probability,
        )
