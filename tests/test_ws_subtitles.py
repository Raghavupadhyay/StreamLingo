from __future__ import annotations

import asyncio
import json

from fastapi.testclient import TestClient

from app.main import app, runner
from app.models import SubtitleEvent


def test_ws_subtitles_sends_initial_status_event() -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/subtitles") as websocket:
            raw = websocket.receive_text()
            payload = json.loads(raw)

            assert payload["event_type"] == "status"
            assert payload["message"] == "ready"
            assert payload["state"] in {"idle", "running"}

            # Keepalive-style message should be accepted by server loop.
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

            status_events = [evt for evt in events if evt.get("event_type") == "status"]
            subtitle_events = [
                evt for evt in events if evt.get("event_type") == "subtitle"
            ]
            assert len(status_events) == 1
            assert len(subtitle_events) == 1
            assert subtitle_events[0]["chunk_id"] == 42
            assert subtitle_events[0]["source_text"] == "namaste dosto"
            assert subtitle_events[0]["translated_text"] == "hello friends"
