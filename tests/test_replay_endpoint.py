from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import app


def test_replay_endpoint_translates_transcript_jsonl(tmp_path) -> None:
    events_path = tmp_path / "events.jsonl"
    rows = [
        {
            "type": "transcript",
            "chunk_id": 1,
            "start_ts": 0.0,
            "end_ts": 1.0,
            "text": "namaste",
            "language": "hi",
        },
        {
            "type": "transcript",
            "chunk_id": 2,
            "start_ts": 1.0,
            "end_ts": 2.0,
            "text": "aap kaise ho",
            "language": "hi",
        },
    ]
    events_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with TestClient(app) as client:
        response = client.post(
            "/pipeline/replay",
            json={"jsonl_path": str(events_path)},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["count"] == 2
    assert payload["items"][0]["source_text"] == "namaste"
    assert payload["items"][0]["translated_text"] == "namaste"
    assert payload["items"][1]["source_text"] == "aap kaise ho"
    assert payload["items"][1]["translated_text"] == "aap kaise ho"
