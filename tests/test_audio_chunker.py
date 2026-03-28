from __future__ import annotations

from app.audio.chunker import AudioChunker, ChunkerConfig
from app.models import AudioFrame


def test_audio_chunker_emits_chunk_when_target_samples_reached() -> None:
    chunker = AudioChunker(
        ChunkerConfig(
            chunk_seconds=1.0,
            sample_rate_hz=10,
            vad_rms_threshold=0.1,
        )
    )
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
        ChunkerConfig(
            chunk_seconds=1.0,
            sample_rate_hz=4,
            vad_rms_threshold=0.5,
        )
    )
    quiet = AudioFrame(timestamp=1.0, sample_rate_hz=4, samples=[0.01, 0.01])
    loud = AudioFrame(timestamp=2.0, sample_rate_hz=4, samples=[1.0, 1.0])

    assert chunker.feed(quiet) is None
    chunk = chunker.feed(loud)

    assert chunk is not None
    assert chunk.voiced_ratio == 0.5
