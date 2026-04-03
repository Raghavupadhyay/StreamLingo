from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from backend.appstt.envelope import SpeechEnvelopeTracker
from stt.silero_vad import SileroVAD


class EmitReason(Enum):
    SILENCE_TIMEOUT = auto()     # speaker paused long enough
    WINDING_DOWN = auto()        # energy slope confirmed end of utterance
    SPEAKER_CHANGE = auto()      # new voice detected after gap
    MAX_DURATION = auto()        # hard cap hit — force emit


@dataclass
class VADConfig:
    sample_rate: int = 16000
    frame_ms: int = 30

    # gate 1 — RMS pre-filter
    rms_threshold: float = 0.010
    rms_adapt_rate: float = 0.995

    # gate 2 — silero
    silero_threshold: float = 0.50

    # gate 3 — turn boundary
    min_speech_ms: int = 250        # ignore blips shorter than this
    min_silence_ms: int = 400       # pause this long → end of turn
    winding_down_grace_ms: int = 200  # extra wait after winding_down detected
    max_utterance_ms: int = 15000   # hard cap

    # envelope tracker
    envelope_window: int = 20


@dataclass
class SegmentResult:
    samples: np.ndarray
    emit_reason: EmitReason
    speaker_changed: bool
    voiced_ratio: float
    duration_ms: float


class _AdaptiveRMSGate:
    def __init__(self, base_threshold: float, adapt_rate: float) -> None:
        self._noise_floor = base_threshold
        self._adapt_rate = adapt_rate

    def check(self, samples: np.ndarray) -> tuple[bool, float]:
        rms = math.sqrt(float(np.mean(samples ** 2)))
        if rms < self._noise_floor:
            self._noise_floor = (
                self._adapt_rate * self._noise_floor
                + (1 - self._adapt_rate) * rms
            )
        threshold = max(self._noise_floor * 3.0, 0.005)
        return rms >= threshold, rms

    def reset(self) -> None:
        pass  # noise floor intentionally persists across utterances


class VADSegmenter:
    """
    Three-gate VAD segmenter for two-speaker conversations.

    Feed 30ms audio frames one at a time via feed().
    Returns a SegmentResult when a complete utterance is detected,
    or None while still buffering.

    Gates:
      1. Adaptive RMS  — kills silence cheaply, zero model cost
      2. Silero VAD    — confirms speech vs noise, ~1–3ms/frame
      3. Turn boundary — slope + silence timer + speaker change detection
    """

    def __init__(self, vad: SileroVAD, cfg: VADConfig) -> None:
        self._vad = vad
        self._cfg = cfg
        self._rms_gate = _AdaptiveRMSGate(cfg.rms_threshold, cfg.rms_adapt_rate)
        self._envelope = SpeechEnvelopeTracker(window=cfg.envelope_window)

        self._frame_ms = cfg.frame_ms
        self._min_speech_frames = cfg.min_speech_ms // cfg.frame_ms
        self._min_silence_frames = cfg.min_silence_ms // cfg.frame_ms
        self._winding_down_grace_frames = cfg.winding_down_grace_ms // cfg.frame_ms
        self._max_frames = cfg.max_utterance_ms // cfg.frame_ms

        self._reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, samples: np.ndarray) -> SegmentResult | None:
        """
        Process one audio frame (30ms of float32 samples).
        Returns SegmentResult when a turn boundary is detected, else None.
        """
        # ── Gate 1: RMS pre-filter ──────────────────────────────────────
        passes_rms, _ = self._rms_gate.check(samples)
        if not passes_rms:
            if self._in_speech:
                # silence during an active utterance — count it
                return self._handle_silence_frame(samples)
            return None

        # ── Gate 2: Silero VAD ──────────────────────────────────────────
        prob = self._vad.predict(samples)
        self._envelope.feed(prob)
        is_speech = prob >= self._cfg.silero_threshold

        if is_speech:
            return self._handle_speech_frame(samples)
        else:
            if self._in_speech:
                return self._handle_silence_frame(samples)
            return None

    def flush(self) -> SegmentResult | None:
        """Force-emit whatever is buffered. Call when pipeline stops."""
        if self._buffer and self._voiced_frame_count >= self._min_speech_frames:
            return self._emit(EmitReason.SILENCE_TIMEOUT)
        return None

    # ------------------------------------------------------------------
    # Internal frame handlers
    # ------------------------------------------------------------------

    def _handle_speech_frame(self, samples: np.ndarray) -> SegmentResult | None:
        self._total_frames += 1
        self._voiced_frame_count += 1
        self._silence_frames = 0
        self._winding_down_frames = 0
        self._buffer.append(samples)

        if not self._in_speech:
            if self._voiced_frame_count >= self._min_speech_frames:
                self._in_speech = True
                # check speaker change on the leading edge of a new utterance
                sig = self._envelope.current_signature()
                self._pending_speaker_change = self._envelope.likely_speaker_change(sig)

        # hard cap
        if len(self._buffer) >= self._max_frames:
            return self._emit(EmitReason.MAX_DURATION)

        return None

    def _handle_silence_frame(self, samples: np.ndarray) -> SegmentResult | None:
        self._total_frames += 1
        self._silence_frames += 1
        self._buffer.append(samples)   # keep padding frames

        # winding-down detection via slope
        if self._envelope.winding_down:
            self._winding_down_frames += 1
            if self._winding_down_frames >= self._winding_down_grace_frames:
                return self._emit(EmitReason.WINDING_DOWN)

        # silence timeout
        if self._silence_frames >= self._min_silence_frames:
            return self._emit(EmitReason.SILENCE_TIMEOUT)

        return None

    # ------------------------------------------------------------------
    # Emit + reset
    # ------------------------------------------------------------------

    def _emit(self, reason: EmitReason) -> SegmentResult:
        utterance = np.concatenate(self._buffer)
        voiced_ratio = (
            self._voiced_frame_count / self._total_frames
            if self._total_frames > 0 else 0.0
        )
        duration_ms = self._total_frames * self._frame_ms
        speaker_changed = self._pending_speaker_change

        # snapshot this speaker's signature before resetting
        self._envelope.save_speaker_signature()

        # reset all state
        self._vad.reset_state()
        self._envelope.reset()
        self._reset()

        return SegmentResult(
            samples=utterance,
            emit_reason=reason,
            speaker_changed=speaker_changed,
            voiced_ratio=voiced_ratio,
            duration_ms=duration_ms,
        )

    def _reset(self) -> None:
        self._buffer: list[np.ndarray] = []
        self._in_speech = False
        self._voiced_frame_count = 0
        self._silence_frames = 0
        self._total_frames = 0
        self._winding_down_frames = 0
        self._pending_speaker_change = False