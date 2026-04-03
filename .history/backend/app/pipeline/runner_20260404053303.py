from __future__ import annotations

# app/pipeline/runner.py  (replace existing)
# Changes from original:
#   - Builds SileroVAD + VADSegmenter when use_smart_vad=True
#   - Passes them into AudioChunker
#   - Removes the voiced_ratio <= 0.05 drop gate (VADSegmenter handles this)
#   - Propagates speaker_changed through TranscriptSegment → SubtitleEvent

import asyncio
import json
from backend.app.stt.silero_vad import SileroVAD
from app.stt.vad_segmenter import VADConfig, VADSegmenter
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from app.audio.capture import AudioCapture
from app.audio.chunker import AudioChunker, ChunkerConfig
from app.config import Settings
from app.context.buffer import ContextBuffer
from app.logging.exporter import PipelineLogger
from app.models import SubtitleEvent, TranslationResult
from app.pipeline.delay_queue import DelayQueue
from app.stt.whisper import WhisperConfig, WhisperTranscriber
from app.translate.llm import LLMTranslator, TranslatorConfig

SubtitleCallback = Callable[[SubtitleEvent], Awaitable[None]]


@dataclass
class PipelineState:
    running: bool = False
    last_error: str | None = None


def _build_chunker(settings: Settings) -> AudioChunker:
    cfg = ChunkerConfig(
        chunk_seconds=settings.chunk_seconds,
        sample_rate_hz=settings.sample_rate_hz,
        vad_rms_threshold=settings.vad_rms_threshold,
    )

    if not settings.use_smart_vad:
        return AudioChunker(cfg)

    # Smart VAD path — import here so onnxruntime is optional
    
    vad = SileroVAD(
        model_path=settings.silero_model_path,
        threshold=settings.silero_threshold,
        sample_rate=settings.sample_rate_hz,
    )
    vad_cfg = VADConfig(
        sample_rate=settings.sample_rate_hz,
        frame_ms=settings.frame_ms,
        rms_threshold=settings.vad_rms_threshold,
        silero_threshold=settings.silero_threshold,
        min_speech_ms=settings.vad_min_speech_ms,
        min_silence_ms=settings.vad_min_silence_ms,
        winding_down_grace_ms=settings.vad_winding_down_grace_ms,
        max_utterance_ms=settings.vad_max_utterance_ms,
        envelope_window=settings.vad_envelope_window,
    )
    segmenter = VADSegmenter(vad=vad, cfg=vad_cfg)
    return AudioChunker(cfg, silero_vad=vad, vad_segmenter=segmenter)


class PipelineRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = PipelineState()
        self._capture = AudioCapture(
            sample_rate_hz=settings.sample_rate_hz,
            frame_ms=settings.frame_ms,
            input_device_index=settings.input_device_index,
        )
        self._chunker = _build_chunker(settings)
        self._stt = WhisperTranscriber(
            WhisperConfig(
                model_name=settings.whisper_model_name,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
                beam_size=settings.whisper_beam_size,
            )
        )
        self._context = ContextBuffer(max_sentences=settings.context_max_sentences)
        self._translator = LLMTranslator(
            TranslatorConfig(
                provider=settings.translation_provider,
                openai_model=settings.openai_model,
                ollama_model=settings.ollama_model,
                ollama_base_url=settings.ollama_base_url,
                ollama_timeout_seconds=settings.ollama_timeout_seconds,
            )
        )
        self._delay = DelayQueue()
        self._log = PipelineLogger(settings.logs_dir) if settings.save_logs else None
        self._tasks: list[asyncio.Task] = []
        self._callbacks: set[SubtitleCallback] = set()

    @property
    def provider_state(self) -> dict[str, str | bool]:
        return {
            "stt_mock": self._stt.is_mock,
            "translator_mock": self._translator.is_mock,
            "translation_provider": self.settings.translation_provider,
            "smart_vad": self.settings.use_smart_vad,
        }

    def add_subscriber(self, callback: SubtitleCallback) -> None:
        self._callbacks.add(callback)

    def remove_subscriber(self, callback: SubtitleCallback) -> None:
        self._callbacks.discard(callback)

    async def start(self) -> None:
        if self.state.running:
            return
        self.state = PipelineState(running=True, last_error=None)
        self._capture.start()
        self._tasks = [
            asyncio.create_task(self._ingest_loop(), name="ingest-loop"),
            asyncio.create_task(self._emit_loop(), name="emit-loop"),
        ]

    async def stop(self) -> None:
        if not self.state.running:
            return
        self.state.running = False
        self._capture.stop()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _ingest_loop(self) -> None:
        try:
            async for frame in self._capture.frames():
                if not self.state.running:
                    break
                chunk = self._chunker.feed(frame)

                # Smart VAD: VADSegmenter already filtered silence — no extra gate.
                # Legacy VAD: keep the original 5% voiced_ratio gate.
                if chunk is None:
                    continue
                if not self.settings.use_smart_vad and chunk.voiced_ratio <= 0.05:
                    continue

                if self._log:
                    self._log.log_chunk(chunk)

                segment = await asyncio.to_thread(
                    self._stt.transcribe,
                    chunk,
                    self.settings.source_language,
                )
                # propagate speaker_changed into the segment
                segment.speaker_changed = chunk.speaker_changed

                if self._log:
                    self._log.log_transcript(segment)
                self._context.add(segment)
                context_text = self._context.get_text_window(
                    exclude_chunk_id=segment.chunk_id
                )

                translated = await asyncio.to_thread(
                    self._translator.translate,
                    segment,
                    context_text,
                    self.settings.source_language,
                    self.settings.target_language,
                )
                # carry speaker_changed into metadata for logging + UI
                translated.metadata["speaker_changed"] = str(chunk.speaker_changed)

                if self._log:
                    self._log.log_translation(translated)

                emit_at = time.monotonic() + self.settings.target_delay_seconds
                await self._delay.put(translated, emit_at)

        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.running = False
            self._capture.stop()

    async def _emit_loop(self) -> None:
        while self.state.running:
            result = await self._delay.get_ready()
            speaker_changed = result.metadata.get("speaker_changed", "False") == "True"
            event = SubtitleEvent(
                chunk_id=result.chunk_id,
                start_ts=result.start_ts,
                end_ts=result.end_ts,
                source_text=result.source_text,
                translated_text=result.translated_text,
                source_language=result.source_language,
                target_language=result.target_language,
                speaker_changed=speaker_changed,
            )
            dead_callbacks: list[SubtitleCallback] = []
            for callback in self._callbacks:
                try:
                    await callback(event)
                except Exception:
                    dead_callbacks.append(callback)
            for callback in dead_callbacks:
                self._callbacks.discard(callback)

    async def replay_transcripts(self, jsonl_path: str) -> list[TranslationResult]:
        rows: list[dict] = []
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {jsonl_path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "transcript":
                rows.append(row)

        translated: list[TranslationResult] = []
        for row in rows:
            text = row.get("text", "").strip()
            if not text:
                continue
            from app.models import TranscriptSegment

            segment = TranscriptSegment(
                chunk_id=int(row["chunk_id"]),
                start_ts=float(row["start_ts"]),
                end_ts=float(row["end_ts"]),
                text=text,
                language=str(row.get("language", self.settings.source_language)),
            )
            context_text = self._context.get_text_window(
                exclude_chunk_id=segment.chunk_id
            )
            result = await asyncio.to_thread(
                self._translator.translate,
                segment,
                context_text,
                self.settings.source_language,
                self.settings.target_language,
            )
            self._context.add(segment)
            translated.append(result)
        return translated