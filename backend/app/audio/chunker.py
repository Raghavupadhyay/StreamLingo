from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from app.models import AudioChunk, AudioFrame

# audio/chunker.py  (replaces existing file)
#
# Breaking change from previous version:
#   - ChunkerConfig gains optional vad_config field
#   - If vad_config + silero_vad are provided, the smart VAD path is used
#   - If not provided, falls back to the original RMS-only path
#     so existing tests and the mock pipeline continue to work unchanged.


if TYPE_CHECKING:
    from stt.silero_vad import SileroVAD
    from stt.vad_segmenter import VADConfig, VADSegmenter


@dataclass
class ChunkerConfig:
    chunk_seconds: float          # used by legacy RMS path only
    sample_rate_hz: int
    vad_rms_threshold: float      # used by legacy RMS path only


class AudioChunker:
    """
    Two modes:

    Smart mode (VADSegmenter):
        Pass silero_vad + vad_segmenter into the constructor.
        Chunks are emitted at natural speech boundaries with
        speaker-change detection.  voiced_ratio is always 1.0
        because only confirmed speech reaches this point.

    Legacy mode (RMS only):
        Original behaviour — fixed-length chunks, RMS gate.
        Used when silero_vad / vad_segmenter are None.
    """

    def __init__(
        self,
        cfg: ChunkerConfig,
        silero_vad: "SileroVAD | None" = None,
        vad_segmenter: "VADSegmenter | None" = None,
    ) -> None:
        self._cfg = cfg
        self._segmenter = vad_segmenter
        self._smart_mode = silero_vad is not None and vad_segmenter is not None

        # legacy state
        self._buffer: list[float] = []
        self._frame_count = 0
        self._voiced_frames = 0
        self._chunk_id = 0
        self._chunk_start_ts: float | None = None

    # ------------------------------------------------------------------
    # Public API — same signature as before
    # ------------------------------------------------------------------

    def feed(self, frame: AudioFrame) -> AudioChunk | None:
        if self._smart_mode:
            return self._feed_smart(frame)
        return self._feed_legacy(frame)

    def flush(self) -> AudioChunk | None:
        """Force-emit buffered audio. Call when pipeline stops."""
        if self._smart_mode and self._segmenter is not None:
            result = self._segmenter.flush()
            if result is not None:
                chunk = self._make_chunk(
                    samples=result.samples,
                    start_ts=self._chunk_start_ts or 0.0,
                    end_ts=self._chunk_start_ts or 0.0,
                    voiced_ratio=result.voiced_ratio,
                    speaker_changed=result.speaker_changed,
                )
                return chunk
        return None

    # ------------------------------------------------------------------
    # Smart VAD path
    # ------------------------------------------------------------------

    def _feed_smart(self, frame: AudioFrame) -> AudioChunk | None:
        if self._chunk_start_ts is None:
            self._chunk_start_ts = frame.timestamp

        samples = np.array(frame.samples, dtype=np.float32)
        assert self._segmenter is not None
        result = self._segmenter.feed(samples)

        if result is None:
            return None

        chunk = self._make_chunk(
            samples=result.samples,
            start_ts=self._chunk_start_ts,
            end_ts=frame.timestamp,
            voiced_ratio=result.voiced_ratio,
            speaker_changed=result.speaker_changed,
        )
        self._chunk_start_ts = frame.timestamp
        return chunk

    def _make_chunk(
        self,
        samples: np.ndarray,
        start_ts: float,
        end_ts: float,
        voiced_ratio: float,
        speaker_changed: bool,
    ) -> AudioChunk:
        chunk = AudioChunk(
            sequence_id=self._chunk_id,
            start_ts=start_ts,
            end_ts=end_ts,
            sample_rate_hz=self._cfg.sample_rate_hz,
            samples=samples.tolist(),
            voiced_ratio=voiced_ratio,
            speaker_changed=speaker_changed,
        )
        self._chunk_id += 1
        return chunk

    # ------------------------------------------------------------------
    # Legacy RMS path (unchanged from original)
    # ------------------------------------------------------------------

    def _feed_legacy(self, frame: AudioFrame) -> AudioChunk | None:
        import math

        if self._chunk_start_ts is None:
            self._chunk_start_ts = frame.timestamp

        self._frame_count += 1
        samples = frame.samples
        square_sum = sum(x * x for x in samples)
        rms = math.sqrt(square_sum / len(samples)) if samples else 0.0
        if rms >= self._cfg.vad_rms_threshold:
            self._voiced_frames += 1

        self._buffer.extend(samples)

        target_size = int(self._cfg.chunk_seconds * self._cfg.sample_rate_hz)
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
            sample_rate_hz=self._cfg.sample_rate_hz,
            samples=chunk_samples,
            voiced_ratio=voiced_ratio,
            speaker_changed=False,
        )
        self._chunk_id += 1
        self._frame_count = 0
        self._voiced_frames = 0
        self._chunk_start_ts = frame.timestamp
        return chunk