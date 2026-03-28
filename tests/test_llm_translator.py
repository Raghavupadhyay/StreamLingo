from __future__ import annotations

from app.models import TranscriptSegment
from app.translate.llm import LLMTranslator, TranslatorConfig


def _segment(text: str) -> TranscriptSegment:
    return TranscriptSegment(
        chunk_id=7,
        start_ts=1.0,
        end_ts=2.0,
        text=text,
        language="hi",
    )


def test_mock_translator_passthroughs_text() -> None:
    translator = LLMTranslator(TranslatorConfig(provider="mock"))
    result = translator.translate(
        _segment("namaste duniya"),
        context_text="",
        source_language="hi",
        target_language="en",
    )
    assert translator.is_mock is True
    assert result.translated_text == "namaste duniya"
    assert result.metadata["provider"] == "mock"


def test_translator_returns_noop_on_empty_text() -> None:
    translator = LLMTranslator(TranslatorConfig(provider="mock"))
    result = translator.translate(
        _segment("   "),
        context_text="prior context",
        source_language="hi",
        target_language="en",
    )
    assert result.translated_text == ""
    assert result.metadata["provider"] == "noop"


def test_openai_provider_falls_back_to_mock_without_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    translator = LLMTranslator(TranslatorConfig(provider="openai"))
    result = translator.translate(
        _segment("aaj mausam achha hai"),
        context_text="",
        source_language="hi",
        target_language="en",
    )
    assert translator.is_mock is True
    assert result.metadata["provider"] == "mock"
