from __future__ import annotations

import math
from dataclasses import dataclass

from app.models import AudioChunk, AudioFrame


@dataclass
class ChunkerConfig:
    chunk_seconds: float
    sample_rate_hz: int
    vad_rms_threshold: float


class AudioChunker:
    def __init__(self, cfg: ChunkerConfig) -> None:
        self.cfg = cfg
        self._buffer: list[float] = []
        self._frame_count = 0
        self._voiced_frames = 0
        self._chunk_id = 0
        self._chunk_start_ts: float | None = None

    def _is_voiced(self, samples: list[float]) -> bool:
        if not samples:
            return False
        square_sum = sum(x * x for x in samples)
        rms = math.sqrt(square_sum / len(samples))
        return rms >= self.cfg.vad_rms_threshold

    def feed(self, frame: AudioFrame) -> AudioChunk | None:
        if self._chunk_start_ts is None:
            self._chunk_start_ts = frame.timestamp

        self._frame_count += 1
        if self._is_voiced(frame.samples):
            self._voiced_frames += 1

        self._buffer.extend(frame.samples)

        target_size = int(self.cfg.chunk_seconds * self.cfg.sample_rate_hz)
        if len(self._buffer) < target_size:
            return None

        chunk_samples = self._buffer[:target_size]
        self._buffer = self._buffer[target_size:]
        voiced_ratio = (
            self._voiced_frames / self._frame_count if self._frame_count > 0 else 0.0
        )
        chunk = AudioChunk(
            sequence_id=self._chunk_id,
            start_ts=self._chunk_start_ts,
            end_ts=frame.timestamp,
            sample_rate_hz=self.cfg.sample_rate_hz,
            samples=chunk_samples,
            voiced_ratio=voiced_ratio,
        )
        self._chunk_id += 1
        self._frame_count = 0
        self._voiced_frames = 0
        self._chunk_start_ts = frame.timestamp
        return chunk
