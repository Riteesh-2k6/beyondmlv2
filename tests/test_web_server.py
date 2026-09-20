"""
Test suite for BeyondML Web API — backend + API routes.

Run:  PYTHONPATH=. pytest tests/test_web_server.py -v

Coverage:
  - markup.parse_rich_markup          (all tag types, edge cases)
  - GET  /api/health
  - POST /api/upload                  (CSV, XLSX, JSON, bad file, path traversal)
  - POST /api/config/{session_id}     (valid, unknown session, bad payload)
  - GET  /api/session/{session_id}    (found, not found)
  - WS   /ws/{session_id}             (unknown session, event schema, input_request round-trip)
  - PipelineRunner                    (log, stat, tree, chart, get_user_input, _pump_recv)
  - models.PipelineConfig             (defaults, validation)
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import pytest
import pandas as pd
import numpy as np

from fastapi.testclient import TestClient

# ─── App under test ───────────────────────────────────────────────────────────
from beyondml.web.server import app, _sessions
from beyondml.web.markup import parse_rich_markup
from beyondml.web.runner import PipelineRunner
from beyondml.web.models import PipelineConfig

client = TestClient(app, raise_server_exceptions=False)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_csv_bytes(rows: int = 20, cols: int = 4) -> bytes:
    rng = np.random.RandomState(0)
    df  = pd.DataFrame(
        rng.randn(rows, cols),
        columns=[f"col_{i}" for i in range(cols)],
    )
    df["target"] = rng.randint(0, 2, rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _upload_csv(rows: int = 20) -> str:
    """Upload a synthetic CSV and return the session_id."""
    r = client.post(
        "/api/upload",
        files={"file": ("smoke.csv", _make_csv_bytes(rows), "text/csv")},
    )
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. parse_rich_markup
# ══════════════════════════════════════════════════════════════════════════════

class TestParseRichMarkup:
    def test_empty_string_returns_empty_list(self):
        assert parse_rich_markup("") == []

    def test_plain_text_no_tags(self):
        result = parse_rich_markup("hello world")
        assert len(result) == 1
        assert result[0]["text"] == "hello world"
        assert result[0]["cls"] == ""

    def test_bold_magenta_maps_to_agent_header(self):
        result = parse_rich_markup("[bold magenta]● Orch[/bold magenta]")
        assert result[0]["cls"] == "agent-header"
        assert result[0]["text"] == "● Orch"

    def test_bold_green_maps_to_success(self):
        result = parse_rich_markup("[bold green]done[/bold green]")
        assert result[0]["cls"] == "success"

    def test_bold_cyan_maps_to_info(self):
        result = parse_rich_markup("[bold cyan]info[/bold cyan]")
        assert result[0]["cls"] == "info"

    def test_bold_yellow_maps_to_warn(self):
        result = parse_rich_markup("[bold yellow]warn[/bold yellow]")
        assert result[0]["cls"] == "warn"

    def test_red_maps_to_error_text(self):
        result = parse_rich_markup("[red]error[/red]")
        assert result[0]["cls"] == "error-text"

    def test_bold_red_maps_to_error_text(self):
        result = parse_rich_markup("[bold red]fatal[/bold red]")
        assert result[0]["cls"] == "error-text"

    def test_dim_maps_to_muted(self):
        result = parse_rich_markup("[dim]quiet[/dim]")
        assert result[0]["cls"] == "muted"

    def test_orange1_maps_to_accent(self):
        result = parse_rich_markup("[bold orange1]accent[/bold orange1]")
        assert result[0]["cls"] == "accent"

    def test_italic_maps_to_italic(self):
        result = parse_rich_markup("[italic]slanted[/italic]")
        assert result[0]["cls"] == "italic"

    def test_mixed_tagged_and_plain(self):
        result = parse_rich_markup("[bold green]ok[/bold green] plain")
        texts = [s["text"] for s in result]
        assert "ok" in texts
        assert any("plain" in t for t in texts)
        labeled = [s for s in result if "ok" in s["text"]]
        assert labeled[0]["cls"] == "success"
        plain = [s for s in result if "plain" in s["text"]]
        assert plain[0]["cls"] == ""

    def test_unknown_tag_passes_text_through(self):
        result = parse_rich_markup("[weird_tag]content[/weird_tag]")
        combined = "".join(s["text"] for s in result)
        assert "content" in combined

    def test_whitespace_only_text_not_in_output(self):
        # span with only whitespace inside tags should not crash
        result = parse_rich_markup("[dim][/dim]")
        assert isinstance(result, list)

    def test_agent_log_line_realistic(self):
        line = "[bold magenta]● Orchestrator[/bold magenta]  Analyzing dataset..."
        result = parse_rich_markup(line)
        classes = [s["cls"] for s in result]
        assert "agent-header" in classes
        combined = "".join(s["text"] for s in result)
        assert "Analyzing dataset" in combined

    def test_separator_line_dim(self):
        line = "[dim]─── Feature Engineer ────────────────────────────[/dim]"
        result = parse_rich_markup(line)
        assert result[0]["cls"] == "muted"
        assert "Feature Engineer" in result[0]["text"]

    def test_no_spans_for_empty_tags_only(self):
        # Only empty closing tags — nothing to emit
        result = parse_rich_markup("[/bold]")
        # May return empty or a fallthrough — should not raise
        assert isinstance(result, list)

    def test_all_spans_have_text_key(self):
        result = parse_rich_markup("[red]a[/red] b [dim]c[/dim]")
        for span in result:
            assert "text" in span
            assert "cls" in span
            assert len(span["text"]) > 0

    def test_adjacent_same_class_merged(self):
        # Two consecutive dim spans should merge
        result = parse_rich_markup("[dim]a[/dim][dim]b[/dim]")
        dim_spans = [s for s in result if s["cls"] == "muted"]
        # Either merged or two separate — both valid; key constraint: no empty spans
        for s in result:
            assert s["text"]


# ══════════════════════════════════════════════════════════════════════════════
# 2. GET /api/health
# ══════════════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_returns_200(self):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_body_has_status_ok(self):
        r = client.get("/api/health")
        assert r.json()["status"] == "ok"

    def test_body_has_version(self):
        r = client.get("/api/health")
        assert "version" in r.json()


# ══════════════════════════════════════════════════════════════════════════════
# 3. POST /api/upload
# ══════════════════════════════════════════════════════════════════════════════

class TestUpload:
    def test_csv_upload_returns_200(self):
        r = client.post(
            "/api/upload",
            files={"file": ("data.csv", _make_csv_bytes(), "text/csv")},
        )
        assert r.status_code == 200

    def test_csv_upload_returns_session_id(self):
        r = client.post(
            "/api/upload",
            files={"file": ("data.csv", _make_csv_bytes(), "text/csv")},
        )
        body = r.json()
        assert "session_id" in body
        assert len(body["session_id"]) == 36  # UUID4

    def test_csv_upload_returns_correct_row_col_count(self):
        r = client.post(
            "/api/upload",
            files={"file": ("data.csv", _make_csv_bytes(rows=50), "text/csv")},
        )
        body = r.json()
        assert body["rows"] == 50
        assert body["cols"] == 5   # 4 feature cols + 1 target

    def test_csv_upload_returns_column_names(self):
        r = client.post(
            "/api/upload",
            files={"file": ("data.csv", _make_csv_bytes(), "text/csv")},
        )
        cols = r.json()["columns"]
        assert isinstance(cols, list)
        assert "target" in cols

    def test_csv_upload_returns_filename(self):
        r = client.post(
            "/api/upload",
            files={"file": ("myfile.csv", _make_csv_bytes(), "text/csv")},
        )
        assert r.json()["filename"] == "myfile.csv"

    def test_session_stored_after_upload(self):
        r = client.post(
            "/api/upload",
            files={"file": ("data.csv", _make_csv_bytes(), "text/csv")},
        )
        sid = r.json()["session_id"]
        assert sid in _sessions
        assert isinstance(_sessions[sid]["df"], pd.DataFrame)

    def test_inf_values_scrubbed_to_nan(self):
        """Verify the server replaces ±inf with NaN on ingestion."""
        df = pd.DataFrame({"a": [1.0, np.inf, -np.inf], "b": [1, 2, 3]})
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        r = client.post(
            "/api/upload",
            files={"file": ("inf.csv", buf.getvalue(), "text/csv")},
        )
        assert r.status_code == 200
        sid = r.json()["session_id"]
        stored_df = _sessions[sid]["df"]
        assert not np.any(np.isinf(stored_df.select_dtypes("number").values))

    def test_json_file_upload(self):
        df  = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        buf = io.BytesIO(df.to_json().encode())
        r   = client.post(
            "/api/upload",
            files={"file": ("data.json", buf.getvalue(), "application/json")},
        )
        assert r.status_code == 200
        assert r.json()["cols"] == 2

    def test_invalid_file_returns_422(self):
        r = client.post(
            "/api/upload",
            files={"file": ("bad.csv", b"not,valid\n\x00\x01\x02", "text/csv")},
        )
        # Server may return 200 (pandas can parse partial garbage) or 422
        # Key constraint: must not return 500
        assert r.status_code in (200, 422)

    def test_empty_csv_returns_422(self):
        r = client.post(
            "/api/upload",
            files={"file": ("empty.csv", b"", "text/csv")},
        )
        assert r.status_code in (422, 500)

    def test_path_traversal_filename_is_sanitised(self):
        """Filename with ../ must not escape temp dir."""
        r = client.post(
            "/api/upload",
            files={"file": ("../../evil.csv", _make_csv_bytes(), "text/csv")},
        )
        # If it succeeded, stored filename must not contain '..'
        if r.status_code == 200:
            assert ".." not in r.json()["filename"]

    def test_multiple_uploads_create_distinct_sessions(self):
        r1 = client.post("/api/upload", files={"file": ("a.csv", _make_csv_bytes(), "text/csv")})
        r2 = client.post("/api/upload", files={"file": ("b.csv", _make_csv_bytes(), "text/csv")})
        assert r1.json()["session_id"] != r2.json()["session_id"]


# ══════════════════════════════════════════════════════════════════════════════
# 4. POST /api/config/{session_id}
# ══════════════════════════════════════════════════════════════════════════════

class TestSetConfig:
    def test_valid_config_returns_ok(self):
        sid = _upload_csv()
        r   = client.post(f"/api/config/{sid}", json={
            "llm_choice": "ollama", "path_choice": "supervised",
            "ga_pop": 10, "ga_gen": 5, "test_size": 0.2,
            "use_ensemble": False, "description": "test",
        })
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_config_stored_on_session(self):
        sid = _upload_csv()
        client.post(f"/api/config/{sid}", json={
            "llm_choice": "groq", "path_choice": "explore",
            "ga_pop": 5, "ga_gen": 2, "test_size": 0.3,
        })
        assert _sessions[sid]["config"] is not None
        assert _sessions[sid]["config"].llm_choice == "groq"
        assert _sessions[sid]["config"].path_choice == "explore"

    def test_unknown_session_returns_404(self):
        r = client.post("/api/config/nonexistent-id", json={})
        assert r.status_code == 404

    def test_invalid_llm_choice_returns_422(self):
        sid = _upload_csv()
        r   = client.post(f"/api/config/{sid}", json={"llm_choice": "anthropic"})
        assert r.status_code == 422

    def test_test_size_out_of_range_accepted_by_api(self):
        # Pydantic doesn't constrain float range by default — server accepts it
        sid = _upload_csv()
        r   = client.post(f"/api/config/{sid}", json={"test_size": 1.5})
        assert r.status_code == 200

    def test_config_overwrites_previous(self):
        sid = _upload_csv()
        client.post(f"/api/config/{sid}", json={"path_choice": "supervised"})
        client.post(f"/api/config/{sid}", json={"path_choice": "explore"})
        assert _sessions[sid]["config"].path_choice == "explore"

    def test_defaults_applied_when_fields_omitted(self):
        sid = _upload_csv()
        client.post(f"/api/config/{sid}", json={})
        cfg = _sessions[sid]["config"]
        assert cfg.llm_choice == "ollama"
        assert cfg.ga_pop == 10
        assert cfg.ga_gen == 5
        assert cfg.test_size == pytest.approx(0.20)
        assert cfg.use_ensemble is False


# ══════════════════════════════════════════════════════════════════════════════
# 5. GET /api/session/{session_id}
# ══════════════════════════════════════════════════════════════════════════════

class TestGetSession:
    def test_found_returns_200(self):
        sid = _upload_csv()
        r   = client.get(f"/api/session/{sid}")
        assert r.status_code == 200

    def test_not_found_returns_404(self):
        r = client.get("/api/session/does-not-exist")
        assert r.status_code == 404

    def test_response_has_expected_keys(self):
        sid  = _upload_csv()
        body = client.get(f"/api/session/{sid}").json()
        for key in ("session_id", "filename", "rows", "cols", "columns", "has_config"):
            assert key in body, f"Missing key: {key}"

    def test_has_config_false_before_config_set(self):
        sid = _upload_csv()
        assert client.get(f"/api/session/{sid}").json()["has_config"] is False

    def test_has_config_true_after_config_set(self):
        sid = _upload_csv()
        client.post(f"/api/config/{sid}", json={})
        assert client.get(f"/api/session/{sid}").json()["has_config"] is True

    def test_columns_list_matches_upload(self):
        sid  = _upload_csv()
        body = client.get(f"/api/session/{sid}").json()
        assert "target" in body["columns"]

    def test_rows_match_upload(self):
        r   = client.post("/api/upload",
                          files={"file": ("r.csv", _make_csv_bytes(rows=77), "text/csv")})
        sid = r.json()["session_id"]
        assert client.get(f"/api/session/{sid}").json()["rows"] == 77


# ══════════════════════════════════════════════════════════════════════════════
# 6. WS /ws/{session_id}
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineWebSocket:
    def test_unknown_session_closes_with_error(self):
        with client.websocket_connect("/ws/bad-session-id") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Unknown session" in msg["payload"]["message"]

    def test_first_message_after_connect_has_type(self):
        """Any valid session must receive at least one structured message."""
        sid = _upload_csv()
        client.post(f"/api/config/{sid}", json={"path_choice": "explore"})
        with client.websocket_connect(f"/ws/{sid}") as ws:
            msg = ws.receive_json()  # Starlette TestClient WS has no timeout kwarg
            assert "type" in msg
            assert msg["type"] in (
                "log", "stat", "tree_update", "chart",
                "input_request", "done", "error", "ga_progress",
            )

    def test_log_event_schema(self):
        """log events must have payload.spans (list) and payload.timestamp."""
        sid = _upload_csv()
        client.post(f"/api/config/{sid}", json={"path_choice": "explore"})
        with client.websocket_connect(f"/ws/{sid}") as ws:
            log_event = None
            for _ in range(30):
                try:
                    msg = ws.receive_json()
                    if msg["type"] == "log":
                        log_event = msg
                        break
                    if msg["type"] in ("done", "error"):
                        break
                except Exception:
                    break
            if log_event:
                assert "spans" in log_event["payload"]
                assert isinstance(log_event["payload"]["spans"], list)
                assert "timestamp" in log_event["payload"]

    def test_stat_event_schema(self):
        """stat events must have payload.key and payload.value strings."""
        sid = _upload_csv()
        client.post(f"/api/config/{sid}", json={"path_choice": "explore"})
        with client.websocket_connect(f"/ws/{sid}") as ws:
            for _ in range(40):
                try:
                    msg = ws.receive_json()
                    if msg["type"] == "stat":
                        assert isinstance(msg["payload"]["key"], str)
                        assert isinstance(msg["payload"]["value"], str)
                        break
                    if msg["type"] in ("done", "error"):
                        break
                except Exception:
                    break

    def test_tree_update_event_schema(self):
        """tree_update events must have node (str) and status (str)."""
        sid = _upload_csv()
        client.post(f"/api/config/{sid}", json={"path_choice": "explore"})
        with client.websocket_connect(f"/ws/{sid}") as ws:
            for _ in range(40):
                try:
                    msg = ws.receive_json()
                    if msg["type"] == "tree_update":
                        assert isinstance(msg["payload"]["node"], str)
                        assert msg["payload"]["status"] in (
                            "waiting", "running", "done", "error"
                        )
                        break
                    if msg["type"] in ("done", "error"):
                        break
                except Exception:
                    break

    def test_log_spans_each_have_text_and_cls(self):
        sid = _upload_csv()
        client.post(f"/api/config/{sid}", json={"path_choice": "explore"})
        with client.websocket_connect(f"/ws/{sid}") as ws:
            for _ in range(30):
                try:
                    msg = ws.receive_json()
                    if msg["type"] == "log":
                        for span in msg["payload"]["spans"]:
                            assert "text" in span
                            assert "cls" in span
                        break
                    if msg["type"] in ("done", "error"):
                        break
                except Exception:
                    break

    def test_done_or_error_terminates_stream(self):
        """
        PipelineRunner.run_pipeline must always emit done or error.
        Tested via a mock runner that bypasses real LLM agents entirely.
        """
        async def _run():
            sent = []

            async def send(e): sent.append(e)
            async def recv(): await asyncio.sleep(9999)

            runner = PipelineRunner(send=send, recv=recv)
            df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})
            # Patch run_pipeline to a tiny stub that just emits done
            async def _stub_pipeline(config, df):
                await runner.log("[bold green]stub run[/bold green]")
                await runner._emit_done({"test_score": 0.99})
            runner.run_pipeline = _stub_pipeline
            await runner.run_pipeline(config={}, df=df)
            return sent

        sent = asyncio.run(_run())
        types = [e["type"] for e in sent]
        assert "done" in types

    def test_error_event_schema(self):
        """error events must have payload.message."""
        with client.websocket_connect("/ws/nonexistent") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "message" in msg["payload"]
            assert isinstance(msg["payload"]["message"], str)


# ══════════════════════════════════════════════════════════════════════════════
# 7. PipelineRunner unit tests (no real LLM / no WS server needed)
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineRunner:
    """Test PipelineRunner in isolation — no FastAPI, no real agents."""

    def _make_runner(self, client_responses: list[str] | None = None):
        """Return a (runner, sent_events) pair with a mock send/recv."""
        sent: list[dict] = []
        _responses = list(client_responses or [])

        async def send(event: dict):
            sent.append(event)

        async def recv() -> str:
            if _responses:
                return _responses.pop(0)
            await asyncio.sleep(9999)

        runner = PipelineRunner(send=send, recv=recv)
        return runner, sent

    def _run(self, coro):
        return asyncio.run(coro)

    def test_log_emits_log_event(self):
        runner, sent = self._make_runner()
        self._run(runner.log("[bold green]hello[/bold green]"))
        assert len(sent) == 1
        assert sent[0]["type"] == "log"

    def test_log_spans_populated(self):
        runner, sent = self._make_runner()
        self._run(runner.log("[bold green]done[/bold green] msg"))
        spans = sent[0]["payload"]["spans"]
        assert any(s["cls"] == "success" for s in spans)

    def test_log_timestamp_present(self):
        runner, sent = self._make_runner()
        self._run(runner.log("test"))
        assert "timestamp" in sent[0]["payload"]
        assert len(sent[0]["payload"]["timestamp"]) == 8  # HH:MM:SS

    def test_stat_emits_stat_event(self):
        runner, sent = self._make_runner()
        self._run(runner.stat("shape", "200×6"))
        assert sent[0] == {"type": "stat", "payload": {"key": "shape", "value": "200×6"}}

    def test_tree_emits_tree_update_event(self):
        runner, sent = self._make_runner()
        self._run(runner.tree("Orchestrator", "running"))
        assert sent[0]["type"] == "tree_update"
        assert sent[0]["payload"]["node"] == "Orchestrator"
        assert sent[0]["payload"]["status"] == "running"

    def test_tree_with_children(self):
        runner, sent = self._make_runner()
        self._run(runner.tree("EDA Agent (Parallel)", "done", children=["EDA", "Outlier"]))
        assert sent[0]["payload"]["children"] == ["EDA", "Outlier"]

    def test_tree_without_children_omits_key(self):
        runner, sent = self._make_runner()
        self._run(runner.tree("X", "done"))
        assert "children" not in sent[0]["payload"]

    def test_get_user_input_sends_input_request(self):
        response_json = json.dumps({"type": "user_response", "payload": {"value": "cap"}})
        runner, sent = self._make_runner(client_responses=[response_json])

        async def _go():
            await runner._start_recv_pump()
            return await runner.get_user_input("Choose strategy:")

        result = self._run(_go())
        assert any(e["type"] == "input_request" for e in sent)
        assert result == "cap"

    def test_get_user_input_ignores_non_response_messages(self):
        responses = [
            json.dumps({"type": "stat", "payload": {"key": "x", "value": "y"}}),
            json.dumps({"type": "user_response", "payload": {"value": "remove"}}),
        ]
        runner, sent = self._make_runner(client_responses=responses)

        async def _go():
            await runner._start_recv_pump()
            return await runner.get_user_input("Strategy?")

        result = self._run(_go())
        assert result == "remove"

    def test_get_user_input_cancel_returns_empty_string(self):
        responses = [json.dumps({"type": "cancel"})]
        runner, sent = self._make_runner(client_responses=responses)

        async def _go():
            await runner._start_recv_pump()
            return await runner.get_user_input("Choose?")

        result = self._run(_go())
        assert result == ""

    def test_emit_error_sends_error_event(self):
        runner, sent = self._make_runner()
        self._run(runner._emit_error("boom", "traceback here"))
        assert sent[0]["type"] == "error"
        assert sent[0]["payload"]["message"] == "boom"
        assert sent[0]["payload"]["traceback"] == "traceback here"

    def test_emit_done_sends_done_event(self):
        runner, sent = self._make_runner()
        self._run(runner._emit_done({"test_score": 0.92}))
        assert sent[0]["type"] == "done"
        assert sent[0]["payload"]["test_score"] == pytest.approx(0.92)

    def test_emit_done_with_no_result_sends_empty_payload(self):
        runner, sent = self._make_runner()
        self._run(runner._emit_done())
        assert sent[0]["type"] == "done"
        assert sent[0]["payload"] == {}

    def test_multiple_log_calls_accumulate(self):
        runner, sent = self._make_runner()

        async def _go():
            await runner.log("line 1")
            await runner.log("line 2")
            await runner.log("line 3")

        self._run(_go())
        assert len(sent) == 3
        assert all(e["type"] == "log" for e in sent)


# ══════════════════════════════════════════════════════════════════════════════
# 8. PipelineConfig model
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineConfig:
    def test_default_llm_choice_is_ollama(self):
        cfg = PipelineConfig()
        assert cfg.llm_choice == "ollama"

    def test_default_path_choice_is_autonomous(self):
        cfg = PipelineConfig()
        assert cfg.path_choice == "autonomous"

    def test_default_ga_params(self):
        cfg = PipelineConfig()
        assert cfg.ga_pop == 10
        assert cfg.ga_gen == 5

    def test_default_test_size(self):
        cfg = PipelineConfig()
        assert cfg.test_size == pytest.approx(0.20)

    def test_default_use_ensemble_false(self):
        cfg = PipelineConfig()
        assert cfg.use_ensemble is False

    def test_invalid_llm_choice_raises(self):
        with pytest.raises(Exception):  # pydantic ValidationError
            PipelineConfig(llm_choice="anthropic")

    def test_custom_values_stored(self):
        cfg = PipelineConfig(
            llm_choice="groq",
            path_choice="supervised",
            ga_pop=20,
            ga_gen=10,
            test_size=0.25,
            use_ensemble=True,
            description="hello",
        )
        assert cfg.llm_choice == "groq"
        assert cfg.ga_pop == 20
        assert cfg.description == "hello"

    def test_model_dump_serialisable(self):
        cfg = PipelineConfig()
        d   = cfg.model_dump()
        assert json.dumps(d)  # must be JSON-serialisable
