from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from app.models import TranscriptSegment, TranslationResult

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]


@dataclass
class TranslatorConfig:
    provider: str = "mock"
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_timeout_seconds: float = 10.0


class LLMTranslator:
    def __init__(self, cfg: TranslatorConfig) -> None:
        self.cfg = cfg
        self._provider = cfg.provider
        self._is_mock = cfg.provider not in {"openai", "ollama"}
        self._client: Any = None
        if cfg.provider == "openai" and not self._is_mock:
            if OpenAI is None:
                self._is_mock = True
                self._provider = "mock"
                return
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self._is_mock = True
                self._provider = "mock"
            else:
                self._client = OpenAI(api_key=api_key)

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    def _build_prompt(
        self,
        source_language: str,
        target_language: str,
        context_text: str,
        text: str,
    ) -> str:
        return (
            "You are a translation assistant for high-stakes conversations.\n"
            "Instructions:\n"
            "1) Preserve meaning precisely.\n"
            "2) Use natural spoken phrasing.\n"
            "3) Keep names, numbers, dates accurate.\n"
            "4) If uncertain, choose the most faithful interpretation.\n\n"
            f"Source language: {source_language}\n"
            f"Target language: {target_language}\n\n"
            "Conversation context:\n"
            f"{context_text or '(none)'}\n\n"
            "Current utterance:\n"
            f"{text}\n\n"
            "Return only the translated text."
        )

    def _translate_with_ollama(self, prompt: str) -> str:
        response = httpx.post(
            f"{self.cfg.ollama_base_url.rstrip('/')}/api/generate",
            json={
                "model": self.cfg.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=self.cfg.ollama_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("response", "")).strip()

    def translate(
        self,
        segment: TranscriptSegment,
        context_text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        source_text = segment.text.strip()
        if not source_text:
            translated = ""
            provider = "noop"
        elif self._is_mock:
            translated = source_text
            provider = "mock"
        else:
            prompt = self._build_prompt(
                source_language=source_language,
                target_language=target_language,
                context_text=context_text,
                text=source_text,
            )
            if self._provider == "openai":
                response = self._client.responses.create(
                    model=self.cfg.openai_model,
                    input=prompt,
                    temperature=0.1,
                )
                translated = (response.output_text or "").strip()
                provider = "openai"
            elif self._provider == "ollama":
                try:
                    translated = self._translate_with_ollama(prompt)
                    provider = "ollama"
                except Exception:
                    # Keep stream alive even if local model is unavailable.
                    translated = source_text
                    provider = "ollama_error_fallback"
            else:
                translated = source_text
                provider = "mock"

        return TranslationResult(
            chunk_id=segment.chunk_id,
            start_ts=segment.start_ts,
            end_ts=segment.end_ts,
            source_text=source_text,
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
            metadata={"provider": provider},
        )
