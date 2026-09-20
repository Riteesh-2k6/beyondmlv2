"""
BeyondML Web UI — FastAPI application.

Endpoints:
  GET  /api/health
  POST /api/upload                → UploadResponse
  POST /api/config/{session_id}   → {"ok": true}
  GET  /api/session/{session_id}  → session metadata
  WS   /ws/{session_id}           → streaming pipeline events
"""
from __future__ import annotations

import os
import uuid
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from beyondml.web.models import PipelineConfig, UploadResponse
from beyondml.web.runner import PipelineRunner

app = FastAPI(title="BeyondML Web API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store  {session_id: {path, filename, df, config}}
_sessions: Dict[str, Dict[str, Any]] = {}


# ─────────────────────────────────────────────────────────
# REST endpoints
# ─────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """
    Accept a CSV / XLSX / JSON file upload.
    Returns a session_id and basic dataset metadata.
    """
    session_id = str(uuid.uuid4())
    tmp_dir    = tempfile.mkdtemp(prefix="beyondml_")
    # Guard against path traversal
    safe_name  = Path(file.filename).name
    save_path  = os.path.join(tmp_dir, safe_name)

    content = await file.read()
    with open(save_path, "wb") as fh:
        fh.write(content)

    ext = Path(safe_name).suffix.lower()
    try:
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(save_path)
        elif ext == ".json":
            df = pd.read_json(save_path)
        else:
            df = pd.read_csv(save_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}")

    df = df.replace([np.inf, -np.inf], np.nan)

    _sessions[session_id] = {
        "path":     save_path,
        "filename": safe_name,
        "df":       df,
        "config":   None,
    }

    return UploadResponse(
        session_id=session_id,
        rows=len(df),
        cols=len(df.columns),
        columns=list(df.columns),
        filename=safe_name,
    )


@app.post("/api/config/{session_id}")
async def set_config(session_id: str, config: PipelineConfig):
    """Store pipeline configuration against an existing session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    _sessions[session_id]["config"] = config
    return {"ok": True}


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    """Return session metadata (rows, cols, columns, has_config)."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sess = _sessions[session_id]
    df: pd.DataFrame = sess["df"]
    return {
        "session_id": session_id,
        "filename":   sess["filename"],
        "rows":       len(df),
        "cols":       len(df.columns),
        "columns":    list(df.columns),
        "has_config": sess["config"] is not None,
    }


# ─────────────────────────────────────────────────────────
# WebSocket pipeline endpoint
# ─────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def pipeline_ws(websocket: WebSocket, session_id: str):
    """
    Stream pipeline events to a connected browser client.

    Protocol (server → client):
      {type: "log",          payload: {spans: [{text, cls}], timestamp}}
      {type: "stat",         payload: {key, value}}
      {type: "tree_update",  payload: {node, status, children?}}
      {type: "chart",        payload: {title, plotly_json}}
      {type: "ga_progress",  payload: {fitness: float[]}}
      {type: "input_request",payload: {prompt, options?}}
      {type: "done",         payload: {eval_result?}}
      {type: "error",        payload: {message, traceback}}

    Protocol (client → server):
      {type: "user_response", payload: {value}}
      {type: "cancel"}
    """
    await websocket.accept()

    if session_id not in _sessions:
        await websocket.send_json({
            "type":    "error",
            "payload": {"message": f"Unknown session: {session_id}", "traceback": ""},
        })
        await websocket.close()
        return

    sess   = _sessions[session_id]
    df     = sess["df"].copy()
    config = sess["config"].model_dump() if sess["config"] else {}

    async def send(event: dict):
        try:
            await websocket.send_json(event)
        except Exception:
            pass

    async def recv() -> str:
        return await websocket.receive_text()

    runner = PipelineRunner(send=send, recv=recv)
    try:
        await runner.run_pipeline(config=config, df=df)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await send({
            "type":    "error",
            "payload": {"message": str(exc), "traceback": ""},
        })


# ─────────────────────────────────────────────────────────
# Static file serving (production: mount Vite build)
# ─────────────────────────────────────────────────────────

_dist_dir = Path(__file__).parent / "ui" / "dist"
if _dist_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist_dir), html=True), name="static")
