from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel


@dataclass
class AudioFrame:
    timestamp: float
    sample_rate_hz: int
    samples: list[float]


@dataclass
class AudioChunk:
    sequence_id: int
    start_ts: float
    end_ts: float
    sample_rate_hz: int
    samples: list[float]
    voiced_ratio: float


@dataclass
class TranscriptSegment:
    chunk_id: int
    start_ts: float
    end_ts: float
    text: str
    language: str
    confidence: float | None = None


@dataclass
class TranslationResult:
    chunk_id: int
    start_ts: float
    end_ts: float
    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(order=True)
class ScheduledSubtitle:
    emit_at: float
    sequence_id: int
    payload: TranslationResult = field(compare=False)


class SubtitleEvent(BaseModel):
    event_type: Literal["subtitle"] = "subtitle"
    chunk_id: int
    start_ts: float
    end_ts: float
    source_text: str
    translated_text: str
    source_language: str
    target_language: str


class StatusEvent(BaseModel):
    event_type: Literal["status"] = "status"
    state: Literal["idle", "running", "stopped", "error"]
    message: str
