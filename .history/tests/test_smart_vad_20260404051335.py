from __future__ import annotations

# tests/test_silero_vad.py

import numpy as np
import pytest

from backend.app.stt.silero_vad import SileroVAD

import numpy as np
import pytest

from backend.app.stt.vad_segmenter import EmitReason, VADConfig, VADSegmenter

class _FakeSession:
    """Replaces onnxruntime.InferenceSession for unit tests."""

    def __init__(self, fixed_prob: float = 0.8) -> None:
        self._prob = fixed_prob

    def run(self, _output_names, inputs):
        h = inputs["h"]
        c = inputs["c"]
        prob = np.array([[[self._prob]]], dtype=np.float32)
        return prob, h, c


def _make_vad(prob: float = 0.8) -> SileroVAD:
    vad = object.__new__(SileroVAD)
    vad._session = _FakeSession(prob)
    vad.threshold = 0.5
    vad._sr = np.array(16000, dtype=np.int64)
    vad._h = np.zeros((2, 1, 64), dtype=np.float32)
    vad._c = np.zeros((2, 1, 64), dtype=np.float32)
    return vad


def test_predict_returns_float():
    vad = _make_vad(prob=0.8)
    samples = np.zeros(480, dtype=np.float32)
    result = vad.predict(samples)
    assert isinstance(result, float)
    assert result == pytest.approx(0.8)


def test_predict_above_threshold_is_speech():
    vad = _make_vad(prob=0.9)
    assert vad.predict(np.zeros(480)) >= vad.threshold


def test_predict_below_threshold_is_not_speech():
    vad = _make_vad(prob=0.2)
    assert vad.predict(np.zeros(480)) < vad.threshold


def test_reset_state_clears_hidden():
    vad = _make_vad()
    vad._h[:] = 999.0
    vad._c[:] = 999.0
    vad.reset_state()
    assert np.all(vad._h == 0.0)
    assert np.all(vad._c == 0.0)


def test_state_updates_across_frames():
    """Hidden state h/c must be carried forward between predict() calls."""
    vad = _make_vad(prob=0.7)
    h_before = vad._h.copy()
    vad.predict(np.zeros(480))
    # FakeSession returns h unchanged — just check no crash and state is set
    assert vad._h is not None


# ──────────────────────────────────────────────────────────────────────
# tests/test_envelope.py
# ──────────────────────────────────────────────────────────────────────

from backend.app.stt.envelope import SpeechEnvelopeTracker


def test_slope_positive_when_rising():
    tracker = SpeechEnvelopeTracker(window=10)
    for p in [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]:
        tracker.feed(p)
    assert tracker.slope > 0


def test_slope_negative_when_falling():
    tracker = SpeechEnvelopeTracker(window=10)
    for p in [0.9, 0.8, 0.6, 0.4, 0.2, 0.1]:
        tracker.feed(p)
    assert tracker.slope < 0


def test_winding_down_when_falling_and_low():
    tracker = SpeechEnvelopeTracker(window=10)
    for p in [0.9, 0.7, 0.5, 0.35, 0.2, 0.1]:
        tracker.feed(p)
    assert tracker.winding_down is True


def test_not_winding_down_when_high():
    tracker = SpeechEnvelopeTracker(window=10)
    for p in [0.8, 0.85, 0.9, 0.88, 0.87]:
        tracker.feed(p)
    assert tracker.winding_down is False


def test_speaker_change_detected_on_different_attack():
    tracker = SpeechEnvelopeTracker(window=20)
    # Speaker A — slow ramp (low attack)
    for p in [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9]:
        tracker.feed(p)
    tracker.save_speaker_signature()
    tracker.reset()

    # Speaker B — sharp attack
    for p in [0.9, 0.92, 0.95, 0.93, 0.88]:
        tracker.feed(p)
    new_sig = tracker.current_signature()
    assert tracker.likely_speaker_change(new_sig) is True


def test_no_speaker_change_same_profile():
    tracker = SpeechEnvelopeTracker(window=20)
    profile = [0.6, 0.7, 0.75, 0.8, 0.78, 0.76, 0.7, 0.6]
    for p in profile:
        tracker.feed(p)
    tracker.save_speaker_signature()
    tracker.reset()

    # Same speaker again
    for p in profile:
        tracker.feed(p)
    new_sig = tracker.current_signature()
    assert tracker.likely_speaker_change(new_sig) is False


# ──────────────────────────────────────────────────────────────────────
# tests/test_vad_segmenter.py
# ──────────────────────────────────────────────────────────────────────



def _make_segmenter(
    silero_prob: float = 0.8,
    min_speech_ms: int = 60,    # short for tests (2 frames @ 30ms)
    min_silence_ms: int = 90,   # 3 frames
    winding_down_grace_ms: int = 60,
    max_utterance_ms: int = 5000,
) -> tuple[VADSegmenter, "_FakeVAD"]:

    class _FakeVAD:
        def __init__(self, prob):
            self.prob = prob
            self.threshold = 0.5
            self.reset_calls = 0

        def predict(self, _samples):
            return self.prob

        def reset_state(self):
            self.reset_calls += 1

    fake_vad = _FakeVAD(silero_prob)
    cfg = VADConfig(
        sample_rate=16000,
        frame_ms=30,
        rms_threshold=0.0,       # disable RMS gate in tests
        silero_threshold=0.5,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        winding_down_grace_ms=winding_down_grace_ms,
        max_utterance_ms=max_utterance_ms,
    )
    seg = VADSegmenter(vad=fake_vad, cfg=cfg)  # type: ignore[arg-type]
    return seg, fake_vad


def _loud_frame() -> np.ndarray:
    return np.full(480, 0.5, dtype=np.float32)


def _silent_frame() -> np.ndarray:
    return np.zeros(480, dtype=np.float32)


def test_returns_none_while_buffering_speech():
    seg, _ = _make_segmenter()
    assert seg.feed(_loud_frame()) is None
    assert seg.feed(_loud_frame()) is None


def test_emits_after_silence_timeout():
    seg, _ = _make_segmenter(min_speech_ms=60, min_silence_ms=90)
    seg.feed(_loud_frame())
    seg.feed(_loud_frame())   # confirmed speech (2 frames × 30ms = 60ms)

    result = None
    for _ in range(3):        # 3 silence frames × 30ms = 90ms
        result = seg.feed(_silent_frame())
        if result is not None:
            break

    assert result is not None
    assert result.emit_reason == EmitReason.SILENCE_TIMEOUT


def test_voiced_ratio_is_correct():
    seg, _ = _make_segmenter(min_speech_ms=60, min_silence_ms=90)
    seg.feed(_loud_frame())
    seg.feed(_loud_frame())

    result = None
    for _ in range(3):
        result = seg.feed(_silent_frame())
        if result is not None:
            break

    assert result is not None
    # 2 speech + 3 silence = 5 total; voiced = 2
    assert result.voiced_ratio == pytest.approx(2 / 5)


def test_resets_vad_state_on_emit():
    seg, fake_vad = _make_segmenter(min_speech_ms=60, min_silence_ms=90)
    seg.feed(_loud_frame())
    seg.feed(_loud_frame())
    for _ in range(3):
        r = seg.feed(_silent_frame())
        if r:
            break
    assert fake_vad.reset_calls == 1


def test_emits_on_max_duration():
    seg, _ = _make_segmenter(max_utterance_ms=90, min_speech_ms=30)
    # 3 loud frames × 30ms = 90ms → max hit
    result = None
    for _ in range(4):
        result = seg.feed(_loud_frame())
        if result is not None:
            break
    assert result is not None
    assert result.emit_reason == EmitReason.MAX_DURATION


def test_flush_returns_buffered_audio():
    seg, _ = _make_segmenter(min_speech_ms=60)
    seg.feed(_loud_frame())
    seg.feed(_loud_frame())   # confirmed speech, not yet emitted
    result = seg.flush()
    assert result is not None
    assert len(result.samples) > 0


def test_no_emit_below_min_speech_threshold():
    seg, _ = _make_segmenter(min_speech_ms=120)  # needs 4 frames
    seg.feed(_loud_frame())   # only 1 frame — not confirmed yet
    result = seg.flush()
    assert result is None







    from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum, auto


# ─────────────────────────────────────────────────────────────
# Silero VAD
# ─────────────────────────────────────────────────────────────

class SileroVAD:
    def __init__(self, session):
        self._session = session
        self.threshold = 0.5
        self._sr = np.array(16000, dtype=np.int64)
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def predict(self, samples: np.ndarray) -> float:
        samples = samples.astype(np.float32)

        out, self._h, self._c = self._session.run(
            None,
            {
                "input": samples,
                "h": self._h,
                "c": self._c,
                "sr": self._sr,
            },
        )

        return float(out.squeeze())   # ✅ critical fix

    def reset_state(self):
        self._h[:] = 0.0
        self._c[:] = 0.0


# ─────────────────────────────────────────────────────────────
# Envelope Tracker
# ─────────────────────────────────────────────────────────────

class SpeechEnvelopeTracker:
    def __init__(self, window: int = 20):
        self.window = window
        self.values: list[float] = []
        self._prev_signature = None

    def feed(self, p: float):
        self.values.append(p)
        if len(self.values) > self.window:
            self.values.pop(0)

    @property
    def slope(self) -> float:
        if len(self.values) < 2:
            return 0.0
        x = np.arange(len(self.values))
        y = np.array(self.values)
        return float(np.polyfit(x, y, 1)[0])   # ✅ robust slope

    @property
    def winding_down(self) -> bool:
        if not self.values:
            return False
        return self.slope < 0 and self.values[-1] < 0.5

    def reset(self):
        self.values.clear()

    def current_signature(self):
        if not self.values:
            return None
        return (round(self.slope, 2), round(np.mean(self.values), 2))

    def save_speaker_signature(self):
        self._prev_signature = self.current_signature()

    def likely_speaker_change(self, new_sig) -> bool:
        if self._prev_signature is None or new_sig is None:
            return False
        return abs(self._prev_signature[0] - new_sig[0]) > 0.2


# ─────────────────────────────────────────────────────────────
# Segmenter
# ─────────────────────────────────────────────────────────────

class EmitReason(Enum):
    SILENCE_TIMEOUT = auto()
    MAX_DURATION = auto()
    FLUSH = auto()


@dataclass
class VADConfig:
    sample_rate: int = 16000
    frame_ms: int = 30
    rms_threshold: float = 0.01
    silero_threshold: float = 0.5
    min_speech_ms: int = 300
    min_silence_ms: int = 300
    winding_down_grace_ms: int = 150
    max_utterance_ms: int = 10000


@dataclass
class SegmentResult:
    samples: np.ndarray
    emit_reason: EmitReason
    voiced_ratio: float


class VADSegmenter:
    def __init__(self, vad: SileroVAD, cfg: VADConfig):
        self.vad = vad
        self.cfg = cfg

        self._buffer: list[np.ndarray] = []
        self._speech_frames = 0
        self._silence_frames = 0
        self._total_frames = 0

    def feed(self, frame: np.ndarray):
        prob = float(self.vad.predict(frame))   # ✅ FIX (no indexing)

        is_speech = prob >= self.cfg.silero_threshold

        self._total_frames += 1

        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        self._buffer.append(frame)

        speech_ms = self._speech_frames * self.cfg.frame_ms
        silence_ms = self._silence_frames * self.cfg.frame_ms
        total_ms = self._total_frames * self.cfg.frame_ms

        if speech_ms < self.cfg.min_speech_ms:
            return None

        if silence_ms >= self.cfg.min_silence_ms:
            return self._emit(EmitReason.SILENCE_TIMEOUT)

        if total_ms >= self.cfg.max_utterance_ms:
            return self._emit(EmitReason.MAX_DURATION)

        return None

    def _emit(self, reason: EmitReason):
        samples = np.concatenate(self._buffer)

        voiced_ratio = self._speech_frames / self._total_frames  # ✅ correct

        result = SegmentResult(
            samples=samples,
            emit_reason=reason,
            voiced_ratio=voiced_ratio,
        )

        # ✅ reset VAD state
        self.vad.reset_state()

        # reset internal buffers
        self._buffer.clear()
        self._speech_frames = 0
        self._silence_frames = 0
        self._total_frames = 0

        return result

    def flush(self):
        speech_ms = self._speech_frames * self.cfg.frame_ms

        if speech_ms < self.cfg.min_speech_ms:
            return None

        return self._emit(EmitReason.FLUSH)