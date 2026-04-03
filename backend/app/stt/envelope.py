from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class EnvelopeSignature:
    """
    Compact fingerprint of a speaker's energy envelope over a short window.
    Used to detect speaker change after a silence gap.
    """
    mean: float
    variance: float
    peak: float
    attack: float      # average energy in the first 5 frames — how fast energy rose
    slope: float       # linear trend over the window (positive = rising)


class SpeechEnvelopeTracker:
    """
    Tracks the rolling Silero probability stream and exposes:
      - current slope  (positive = still speaking / energy rising)
      - winding_down   (slope dropped below threshold, energy falling)
      - likely_speaker_change  (new envelope after a gap looks different)

    Designed for two-speaker conversations.  With only two distinct
    voice profiles the contrast between signatures is strong enough that
    simple variance + attack comparison is reliable.
    """

    # slope below this while prob < 0.4 → utterance is winding down
    WINDING_DOWN_SLOPE: float = -0.10
    WINDING_DOWN_PROB: float = 0.40

    # speaker-change thresholds (tuned for two-speaker interviews)
    ATTACK_DIFF_THRESHOLD: float = 0.22
    VARIANCE_DIFF_THRESHOLD: float = 0.07

    def __init__(self, window: int = 20) -> None:
        self._window = window
        self._probs: deque[float] = deque(maxlen=window)
        self._last_signature: EnvelopeSignature | None = None

    # ------------------------------------------------------------------
    # Feed
    # ------------------------------------------------------------------

    def feed(self, prob: float) -> None:
        self._probs.append(prob)

    # ------------------------------------------------------------------
    # Derived signals
    # ------------------------------------------------------------------

    @property
    def slope(self) -> float:
        """Linear slope of the probability window.  Positive = rising."""
        arr = np.array(self._probs)
        if len(arr) < 2:
            return 0.0
        xs = np.arange(len(arr), dtype=np.float32)
        return float(np.polyfit(xs, arr, 1)[0])

    @property
    def winding_down(self) -> bool:
        """True when energy is actively falling and below threshold."""
        if not self._probs:
            return False
        current_prob = self._probs[-1]
        return self.slope < self.WINDING_DOWN_SLOPE and current_prob < self.WINDING_DOWN_PROB

    def current_signature(self) -> EnvelopeSignature:
        arr = np.array(self._probs) if self._probs else np.zeros(1)
        attack_window = list(self._probs)[:5]
        return EnvelopeSignature(
            mean=float(arr.mean()),
            variance=float(arr.var()),
            peak=float(arr.max()),
            attack=float(np.mean(attack_window)) if attack_window else 0.0,
            slope=self.slope,
        )

    def likely_speaker_change(self, new_sig: EnvelopeSignature) -> bool:
        """
        Compare new_sig against the saved signature from the previous speaker.
        Returns True if the difference suggests a different speaker.
        Only meaningful after save_speaker_signature() has been called at
        least once (i.e. after the first completed utterance).
        """
        if self._last_signature is None:
            return False
        old = self._last_signature
        attack_diff = abs(new_sig.attack - old.attack)
        variance_diff = abs(new_sig.variance - old.variance)
        return (
            attack_diff > self.ATTACK_DIFF_THRESHOLD
            or variance_diff > self.VARIANCE_DIFF_THRESHOLD
        )

    def save_speaker_signature(self) -> None:
        """
        Call at the end of each confirmed utterance to snapshot the current
        speaker's envelope.  Used as the reference for the next speaker-change check.
        """
        self._last_signature = self.current_signature()

    def reset(self) -> None:
        """Clear rolling window — call at turn boundaries."""
        self._probs.clear()