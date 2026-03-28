from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.audio.capture import list_input_devices
from app.config import settings
from app.models import StatusEvent, SubtitleEvent
from app.pipeline.runner import PipelineRunner

app = FastAPI(title="StreamLingo", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

runner = PipelineRunner(settings)
BACKEND_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = BACKEND_ROOT / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "pipeline_running": runner.state.running,
        "last_error": runner.state.last_error,
        **runner.provider_state,
    }


@app.get("/devices")
async def devices() -> dict:
    found = [d.__dict__ for d in list_input_devices()]
    return {"devices": found}


@app.post("/pipeline/start")
async def start_pipeline() -> dict:
    await runner.start()
    return {"started": True}


@app.post("/pipeline/stop")
async def stop_pipeline() -> dict:
    await runner.stop()
    return {"stopped": True}


@app.post("/pipeline/replay")
async def replay(payload: dict) -> dict:
    path = payload.get("jsonl_path")
    if not path:
        return {"ok": False, "error": "jsonl_path is required"}
    out = await runner.replay_transcripts(path)
    return {
        "ok": True,
        "count": len(out),
        "items": [item.__dict__ for item in out],
    }


@app.websocket("/ws/subtitles")
async def ws_subtitles(ws: WebSocket) -> None:
    await ws.accept()

    async def send_event(event: SubtitleEvent) -> None:
        await ws.send_text(event.model_dump_json())

    runner.add_subscriber(send_event)
    await ws.send_text(
        StatusEvent(state="running" if runner.state.running else "idle", message="ready")
        .model_dump_json()
    )
    try:
        while True:
            # Keep socket alive and optionally receive client pings.
            _ = await ws.receive_text()
    except WebSocketDisconnect:
        runner.remove_subscriber(send_event)
    except Exception as exc:
        runner.remove_subscriber(send_event)
        await ws.close(code=1011, reason=json.dumps({"error": str(exc)}))
