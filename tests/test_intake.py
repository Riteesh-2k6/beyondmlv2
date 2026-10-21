import json
import random
from pathlib import Path

import pandas as pd
import pytest

from beyondml.harness.intake import run_intake
from beyondml.workspace.events import EventBus
from beyondml.workspace.project import Project

TITANIC = Path(__file__).resolve().parent.parent / "data" / "titanic.csv"


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


def _run(tmp_path):
    p = Project.init(tmp_path)
    return p, lambda: run_intake(p, EventBus(p))


def test_tabular_titanic(tmp_path):
    p, go = _run(tmp_path)
    p.add_data(TITANIC)
    [r] = go()
    assert r.modality == "tabular" and r.usable
    assert r.summary["rows"] == 2099 and r.summary["suggested_target"] == "target"


def test_timeseries_detected(tmp_path):
    p, go = _run(tmp_path)
    pd.DataFrame({"date": pd.date_range("2024-01-01", periods=50).astype(str), "value": range(50)}).to_csv(
        p.root / "data/raw/ts.csv", index=False)
    [r] = go()
    assert r.modality == "timeseries" and r.usable


def test_image_dir(tmp_path):
    p, go = _run(tmp_path)
    d = p.root / "data/raw/imgs"
    d.mkdir()
    for i in range(3):
        (d / f"{i}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 8)
    [r] = go()
    assert r.modality == "image" and r.summary["files"] == 3


def test_unusable_inputs_reported(tmp_path):
    p, go = _run(tmp_path)
    (p.root / "data/raw/empty.csv").write_bytes(b"")
    (p.root / "data/raw/garbage.csv").write_bytes(random.Random(0).randbytes(256))
    reports = go()
    assert len(reports) == 2 and all(not r.usable and r.problem for r in reports)
    text = p.read("DATA.md")
    assert "empty.csv" in text and "garbage.csv" in text and "unusable" in text


def test_intake_events(tmp_path):
    p, go = _run(tmp_path)
    go()
    evs = [json.loads(l) for l in (tmp_path / "agents/intake/events.jsonl").read_text().splitlines()]
    assert [e["data"]["status"] for e in evs if e["kind"] == "step_status"] == ["running", "done"]
