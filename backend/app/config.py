from __future__ import annotations

# app/config.py  (replace existing)

import os
from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    sample_rate_hz: int = int(os.getenv("SL_SAMPLE_RATE_HZ", "16000"))
    frame_ms: int = int(os.getenv("SL_FRAME_MS", "30"))
    chunk_seconds: float = float(os.getenv("SL_CHUNK_SECONDS", "2.0"))
    vad_rms_threshold: float = float(os.getenv("SL_VAD_RMS_THRESHOLD", "0.010"))
    target_delay_seconds: float = float(os.getenv("SL_TARGET_DELAY_SECONDS", "7.0"))
    context_max_sentences: int = int(os.getenv("SL_CONTEXT_MAX_SENTENCES", "6"))
    whisper_model_name: str = os.getenv("SL_WHISPER_MODEL", "base")
    whisper_device: str = os.getenv("SL_WHISPER_DEVICE", "auto")
    whisper_compute_type: str = os.getenv("SL_WHISPER_COMPUTE_TYPE", "int8")
    whisper_beam_size: int = int(os.getenv("SL_WHISPER_BEAM_SIZE", "3"))
    source_language: str = os.getenv("SL_SOURCE_LANGUAGE", "ja")
    target_language: str = os.getenv("SL_TARGET_LANGUAGE", "en")
    input_device_index: int | None = (
        int(os.getenv("SL_INPUT_DEVICE_INDEX"))
        if os.getenv("SL_INPUT_DEVICE_INDEX") is not None
        else None
    )
    translation_provider: str = os.getenv("SL_TRANSLATION_PROVIDER", "mock").lower()
    openai_model: str = os.getenv("SL_OPENAI_MODEL", "gpt-4o-mini")
    ollama_model: str = os.getenv("SL_OLLAMA_MODEL", "qwen2.5:7b-instruct")
    ollama_base_url: str = os.getenv("SL_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_timeout_seconds: float = float(os.getenv("SL_OLLAMA_TIMEOUT_SECONDS", "10"))
    save_logs: bool = _get_bool("SL_SAVE_LOGS", True)
    logs_dir: str = os.getenv("SL_LOGS_DIR", "runs")

    # ── Smart VAD settings (new) ────────────────────────────────────────
    # Set SL_USE_SMART_VAD=true to enable Silero + envelope-based segmentation.
    # When false the original RMS-only chunker is used (default, no extra deps).
    use_smart_vad: bool = _get_bool("SL_USE_SMART_VAD", False)

    # Path to silero_vad.onnx — downloaded automatically if missing.
    silero_model_path: str = os.getenv("SL_SILERO_MODEL_PATH", "models/silero_vad.onnx")

    # Silero speech-probability threshold (0.0–1.0).
    silero_threshold: float = float(os.getenv("SL_SILERO_THRESHOLD", "0.50"))

    # Minimum voiced audio before a chunk is considered real speech (ms).
    vad_min_speech_ms: int = int(os.getenv("SL_VAD_MIN_SPEECH_MS", "250"))

    # Silence duration that marks end of turn (ms).
    vad_min_silence_ms: int = int(os.getenv("SL_VAD_MIN_SILENCE_MS", "400"))

    # Extra wait after winding-down slope detected before emitting (ms).
    vad_winding_down_grace_ms: int = int(os.getenv("SL_VAD_WINDING_DOWN_GRACE_MS", "200"))

    # Hard cap on utterance length — force emit regardless (ms).
    vad_max_utterance_ms: int = int(os.getenv("SL_VAD_MAX_UTTERANCE_MS", "15000"))

    # Rolling window size for envelope tracker (frames).
    vad_envelope_window: int = int(os.getenv("SL_VAD_ENVELOPE_WINDOW", "20"))


settings = Settings()