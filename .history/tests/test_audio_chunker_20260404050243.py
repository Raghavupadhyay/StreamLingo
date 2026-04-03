from __future__ import annotations
import numpy as np
import pytest

from app.audio.chunker import AudioChunker, ChunkerConfig
from backend.app.models import AudioFrame

# tests/test_audio_chunker.py  (updated)
# Changes from original:
#   - AudioChunk now has speaker_changed: bool = False
#   - Legacy RMS path is now accessed via AudioChunker(cfg) with no vad args
#   - Added assertion that legacy path always sets speaker_changed=False
#   - Smart VAD path tested with a mock VADSegmenter



def _make_cfg() -> ChunkerConfig:
    return ChunkerConfig(
        chunk_seconds=1.0,
        sample_rate_hz=10,
        vad_rms_threshold=0.1,
    )


# ── Legacy RMS path (unchanged behaviour) ─────────────────────────────

def test_audio_chunker_emits_chunk_when_target_samples_reached() -> None:
    chunker = AudioChunker(_make_cfg())
    frame1 = AudioFrame(timestamp=1.0, sample_rate_hz=10, samples=[0.2] * 5)
    frame2 = AudioFrame(timestamp=2.0, sample_rate_hz=10, samples=[0.2] * 5)

    assert chunker.feed(frame1) is None
    chunk = chunker.feed(frame2)

    assert chunk is not None
    assert chunk.sequence_id == 0
    assert chunk.start_ts == 1.0
    assert chunk.end_ts == 2.0
    assert len(chunk.samples) == 10
    assert chunk.voiced_ratio == 1.0


def test_audio_chunker_tracks_voiced_ratio() -> None:
    chunker = AudioChunker(
        ChunkerConfig(chunk_seconds=1.0, sample_rate_hz=4, vad_rms_threshold=0.5)
    )
    quiet = AudioFrame(timestamp=1.0, sample_rate_hz=4, samples=[0.01, 0.01])
    loud = AudioFrame(timestamp=2.0, sample_rate_hz=4, samples=[1.0, 1.0])

    assert chunker.feed(quiet) is None
    chunk = chunker.feed(loud)

    assert chunk is not None
    assert chunk.voiced_ratio == 0.5


def test_legacy_path_speaker_changed_always_false() -> None:
    """Legacy RMS path never sets speaker_changed — no VAD to detect it."""
    chunker = AudioChunker(_make_cfg())
    frame1 = AudioFrame(timestamp=1.0, sample_rate_hz=10, samples=[0.2] * 5)
    frame2 = AudioFrame(timestamp=2.0, sample_rate_hz=10, samples=[0.2] * 5)
    chunker.feed(frame1)
    chunk = chunker.feed(frame2)
    assert chunk is not None
    assert chunk.speaker_changed is False


# ── Smart VAD path ────────────────────────────────────────────────────

class _FakeSegmentResult:
    def __init__(self, speaker_changed: bool = False, voiced_ratio: float = 1.0):
        self.samples = np.ones(480, dtype=np.float32)
        self.emit_reason = None
        self.speaker_changed = speaker_changed
        self.voiced_ratio = voiced_ratio
        self.duration_ms = 480.0


class _FakeVADSegmenter:
    """Returns a result on the second feed call, None on the first."""
    def __init__(self, speaker_changed: bool = False):
        self._calls = 0
        self._speaker_changed = speaker_changed

    def feed(self, _samples: np.ndarray):
        self._calls += 1
        if self._calls >= 2:
            return _FakeSegmentResult(speaker_changed=self._speaker_changed)
        return None

    def flush(self):
        return None


class _FakeVAD:
    pass


def _make_smart_chunker(speaker_changed: bool = False) -> AudioChunker:
    cfg = ChunkerConfig(chunk_seconds=2.0, sample_rate_hz=16000, vad_rms_threshold=0.01)
    segmenter = _FakeVADSegmenter(speaker_changed=speaker_changed)
    return AudioChunker(cfg, silero_vad=_FakeVAD(), vad_segmenter=segmenter)  # type: ignore


def test_smart_chunker_emits_on_segmenter_result() -> None:
    chunker = _make_smart_chunker()
    frame = AudioFrame(timestamp=1.0, sample_rate_hz=16000, samples=[0.1] * 480)

    assert chunker.feed(frame) is None   # first call → segmenter returns None
    chunk = chunker.feed(frame)           # second call → segmenter returns result
    assert chunk is not None
    assert chunk.voiced_ratio == 1.0


def test_smart_chunker_propagates_speaker_changed_true() -> None:
    chunker = _make_smart_chunker(speaker_changed=True)
    frame = AudioFrame(timestamp=1.0, sample_rate_hz=16000, samples=[0.1] * 480)
    chunker.feed(frame)
    chunk = chunker.feed(frame)
    assert chunk is not None
    assert chunk.speaker_changed is True


def test_smart_chunker_propagates_speaker_changed_false() -> None:
    chunker = _make_smart_chunker(speaker_changed=False)
    frame = AudioFrame(timestamp=1.0, sample_rate_hz=16000, samples=[0.1] * 480)
    chunker.feed(frame)
    chunk = chunker.feed(frame)
    assert chunk is not None
    assert chunk.speaker_changed is False


def test_smart_chunker_increments_sequence_id() -> None:
    chunker = _make_smart_chunker()
    frame = AudioFrame(timestamp=1.0, sample_rate_hz=16000, samples=[0.1] * 480)

    chunker.feed(frame)
    chunk1 = chunker.feed(frame)

    chunker._segmenter._calls = 0  # type: ignore[attr-defined]
    chunker.feed(frame)
    chunk2 = chunker.feed(frame)

    assert chunk1 is not None
    assert chunk2 is not None
    assert chunk2.sequence_id == chunk1.sequence_id + 1