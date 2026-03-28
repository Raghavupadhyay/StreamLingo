from __future__ import annotations

import asyncio
import queue
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.models import AudioFrame

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional dependency at import time
    sd = None  # type: ignore[assignment]


@dataclass
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    default_samplerate: float


def list_input_devices() -> list[AudioDevice]:
    if sd is None:
        return []
    devices: list[dict[str, Any]] = sd.query_devices()  # type: ignore[assignment]
    found: list[AudioDevice] = []
    for idx, dev in enumerate(devices):
        max_in = int(dev.get("max_input_channels", 0))
        if max_in <= 0:
            continue
        found.append(
            AudioDevice(
                index=idx,
                name=str(dev.get("name", f"Device {idx}")),
                max_input_channels=max_in,
                default_samplerate=float(dev.get("default_samplerate", 0.0)),
            )
        )
    return found


class AudioCapture:
    def __init__(
        self,
        sample_rate_hz: int,
        frame_ms: int,
        input_device_index: int | None = None,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.frame_ms = frame_ms
        self.input_device_index = input_device_index
        self.samples_per_frame = max(1, int(sample_rate_hz * frame_ms / 1000))
        self._queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=200)
        self._stream = None
        self._running = False

    def start(self) -> None:
        if sd is None:
            raise RuntimeError(
                "sounddevice is not available. Install dependencies first."
            )
        if self._running:
            return

        def callback(indata: np.ndarray, frames: int, *_args: Any) -> None:
            if frames <= 0:
                return
            mono = indata[:, 0] if indata.ndim > 1 else indata
            frame = AudioFrame(
                timestamp=time.monotonic(),
                sample_rate_hz=self.sample_rate_hz,
                samples=mono.astype(np.float32).tolist(),
            )
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                # Drop oldest to preserve recency during overload.
                _ = self._queue.get_nowait()
                self._queue.put_nowait(frame)

        self._stream = sd.InputStream(
            samplerate=self.sample_rate_hz,
            channels=1,
            blocksize=self.samples_per_frame,
            device=self.input_device_index,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def frames(self):
        while self._running:
            frame = await asyncio.to_thread(self._queue.get)
            yield frame
