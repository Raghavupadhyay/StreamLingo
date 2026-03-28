from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from app.models import AudioChunk, TranscriptSegment, TranslationResult


class PipelineLogger:
    def __init__(self, runs_dir: str = "runs") -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(runs_dir) / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.run_dir / "events.jsonl"
        self.srt_file = self.run_dir / "subtitles.srt"
        self._srt_index = 1

    def log_chunk(self, chunk: AudioChunk) -> None:
        self._append_jsonl("chunk", asdict(chunk))

    def log_transcript(self, segment: TranscriptSegment) -> None:
        self._append_jsonl("transcript", asdict(segment))

    def log_translation(self, item: TranslationResult) -> None:
        data = asdict(item)
        self._append_jsonl("translation", data)
        if item.translated_text.strip():
            self._append_srt(item)

    def _append_jsonl(self, event_type: str, payload: dict) -> None:
        row = {"type": event_type, "timestamp": datetime.now(UTC).isoformat(), **payload}
        with self.events_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _append_srt(self, item: TranslationResult) -> None:
        with self.srt_file.open("a", encoding="utf-8") as f:
            f.write(f"{self._srt_index}\n")
            f.write(
                f"{_to_srt_time(item.start_ts)} --> {_to_srt_time(item.end_ts)}\n"
            )
            f.write(f"{item.translated_text.strip()}\n\n")
            self._srt_index += 1


def _to_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    millis = int((seconds % 1) * 1000)
    total = int(seconds)
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"
