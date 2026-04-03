from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]

_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
)


def _download_model(dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_MODEL_URL, dest)


class SileroVAD:
    """
    Thin wrapper around the Silero VAD ONNX model.

    One instance per pipeline — the ONNX session is NOT thread-safe
    for concurrent calls.  Hidden state (h, c) persists across frames
    within a single utterance; call reset_state() at every turn boundary.
    """

    def __init__(
        self,
        model_path: str | None = None,
        threshold: float = 0.5,
        sample_rate: int = 16000,
    ) -> None:
        if ort is None:
            raise RuntimeError(
                "onnxruntime is required for SileroVAD. "
                "Install it with: pip install onnxruntime"
            )

        path = Path(model_path) if model_path else Path("models/silero_vad.onnx")
        if not path.exists():
            _download_model(path)

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1  # small model — threads add overhead
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3    # suppress ONNX runtime info logs

        self._session: Any = ort.InferenceSession(str(path), sess_options=opts)
        self.threshold = threshold
        self._sr = np.array(sample_rate, dtype=np.int64)
        self._reset_state()

    def _reset_state(self) -> None:
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def reset_state(self) -> None:
        """Call at every turn boundary so state does not bleed across utterances."""
        self._reset_state()

    def predict(self, samples: np.ndarray) -> float:
        """
        Return speech probability in [0.0, 1.0] for a single audio frame.
        samples must be float32, shape (N,), N = sample_rate * frame_ms / 1000.
        """
        x = samples.reshape(1, -1).astype(np.float32)
        out, self._h, self._c = self._session.run(
            None,
            {
                "input": x,
                "h": self._h,
                "c": self._c,
                "sr": self._sr,
            },
        )
        return float(out[0][0])

    @property
    def is_available(self) -> bool:
        return True