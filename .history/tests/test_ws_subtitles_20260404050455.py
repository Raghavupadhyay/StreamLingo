from __future__ import annotations

# tests/test_ws_subtitles.py  (updated)
# Change from original:
#   - SubtitleEvent now has speaker_changed: bool = False
#   - Added assertion that speaker_changed is present in subtitle event payload
#   - Added test that speaker_changed=True is forwarded correctly

import asyncio
import json

from fastapi.testclient import TestClient

from backend.app.main import app, runner
from backendapp.models import SubtitleEvent


def test_ws_subtitles_sends_initial_status_event() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/subtitles") as websocket:
            raw = websocket.receive_text()
            payload = json.loads(raw)

            assert payload["event_type"] == "status"
            assert payload["message"] == "ready"
            assert payload["state"] in {"idle", "running"}

            websocket.send_text("ping")


def test_ws_subtitles_can_emit_subtitle_event(monkeypatch) -> None:
    subtitle = SubtitleEvent(
        chunk_id=42,
        start_ts=2.0,
        end_ts=3.5,
        source_text="namaste dosto",
        translated_text="hello friends",
        source_language="hi",
        target_language="en",
        speaker_changed=False,
    )

    def fake_add_subscriber(callback):
        asyncio.get_running_loop().create_task(callback(subtitle))

    monkeypatch.setattr(runner, "add_subscriber", fake_add_subscriber)
    monkeypatch.setattr(runner, "remove_subscriber", lambda _callback: None)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/subtitles") as websocket:
            first = json.loads(websocket.receive_text())
            second = json.loads(websocket.receive_text())
            events = [first, second]

            status_events = [e for e in events if e.get("event_type") == "status"]
            subtitle_events = [e for e in events if e.get("event_type") == "subtitle"]

            assert len(status_events) == 1
            assert len(subtitle_events) == 1

            sub = subtitle_events[0]
            assert sub["chunk_id"] == 42
            assert sub["source_text"] == "namaste dosto"
            assert sub["translated_text"] == "hello friends"
            assert sub["speaker_changed"] is False   # NEW field present


def test_ws_subtitles_forwards_speaker_changed_true(monkeypatch) -> None:
    """speaker_changed=True must reach the browser payload."""
    subtitle = SubtitleEvent(
        chunk_id=7,
        start_ts=0.0,
        end_ts=1.0,
        source_text="aur batao",
        translated_text="tell me more",
        source_language="hi",
        target_language="en",
        speaker_changed=True,   # ← speaker switched
    )

    def fake_add_subscriber(callback):
        asyncio.get_running_loop().create_task(callback(subtitle))

    monkeypatch.setattr(runner, "add_subscriber", fake_add_subscriber)
    monkeypatch.setattr(runner, "remove_subscriber", lambda _callback: None)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/subtitles") as websocket:
            first = json.loads(websocket.receive_text())
            second = json.loads(websocket.receive_text())
            events = [first, second]
            subtitle_events = [e for e in events if e.get("event_type") == "subtitle"]

            assert subtitle_events[0]["speaker_changed"] is True