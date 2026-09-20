"""Pydantic models for the BeyondML web API."""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class Span(BaseModel):
    text: str
    cls: str = ""


class LogEvent(BaseModel):
    type: Literal["log"] = "log"
    payload: "LogPayload"


class LogPayload(BaseModel):
    spans: List[Span]
    timestamp: str


class StatEvent(BaseModel):
    type: Literal["stat"] = "stat"
    payload: "StatPayload"


class StatPayload(BaseModel):
    key: str
    value: str


class ChartEvent(BaseModel):
    type: Literal["chart"] = "chart"
    payload: "ChartPayload"


class ChartPayload(BaseModel):
    title: str
    plotly_json: Dict[str, Any]


class TreeUpdateEvent(BaseModel):
    type: Literal["tree_update"] = "tree_update"
    payload: "TreePayload"


class TreePayload(BaseModel):
    node: str
    status: Literal["waiting", "running", "done", "error"]
    children: Optional[List[str]] = None


class InputRequestEvent(BaseModel):
    type: Literal["input_request"] = "input_request"
    payload: "InputRequestPayload"


class InputRequestPayload(BaseModel):
    prompt: str
    options: Optional[List[str]] = None


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    payload: Dict[str, Any] = Field(default_factory=dict)


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    payload: "ErrorPayload"


class ErrorPayload(BaseModel):
    message: str
    traceback: str = ""


# Client → Server
class UserResponseMessage(BaseModel):
    type: Literal["user_response"] = "user_response"
    payload: "UserResponsePayload"


class UserResponsePayload(BaseModel):
    value: str


class CancelMessage(BaseModel):
    type: Literal["cancel"] = "cancel"


# Pipeline configuration sent after file upload
class PipelineConfig(BaseModel):
    llm_choice: Literal["ollama", "groq"] = "ollama"
    path_choice: str = "autonomous"
    ga_pop: int = 10
    ga_gen: int = 5
    test_size: float = 0.20
    use_ensemble: bool = False
    description: str = ""
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_model: str = "qwen3:8b"
    target_col: str = ""


# Upload response
class UploadResponse(BaseModel):
    session_id: str
    rows: int
    cols: int
    columns: List[str]
    filename: str


# Session stored in memory
class SessionState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    session_id: str
    path: str
    filename: str
    df: Any  # pd.DataFrame — not serialisable, stored as Any
    config: Optional[PipelineConfig] = None
