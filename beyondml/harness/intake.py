"""Intake: look at everything in data/raw/ and write DATA.md. Deterministic — no LLM."""
from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from beyondml.engine.profiler import TargetIdentifier
from beyondml.workspace.events import EventBus
from beyondml.workspace.project import Project

_EXT = {
    **dict.fromkeys((".csv", ".tsv", ".parquet", ".xlsx"), "tabular"),
    **dict.fromkeys((".txt", ".md", ".jsonl"), "text"),
    **dict.fromkeys((".wav", ".mp3", ".flac"), "audio"),
    **dict.fromkeys((".jpg", ".jpeg", ".png", ".bmp", ".webp"), "image"),
}


@dataclass
class InputReport:
    path: str
    modality: str
    usable: bool
    summary: dict = field(default_factory=dict)
    problem: str | None = None


def _read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext == ".parquet":
        return pd.read_parquet(path)
    return pd.read_excel(path)


def _is_timeseries(df: pd.DataFrame) -> bool:
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            continue  # integer ids are not dates
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce").dropna()
        if len(s) and len(parsed) >= 0.9 * len(s) and parsed.is_monotonic_increasing:
            return True
    return False


def _files(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted(f for f in path.rglob("*") if f.is_file())


def detect_modality(path: Path) -> str:
    if path.is_dir():
        exts = Counter(f.suffix.lower() for f in _files(path))
        return _EXT.get(exts.most_common(1)[0][0], "unknown") if exts else "unknown"
    modality = _EXT.get(path.suffix.lower(), "unknown")
    if modality == "tabular":
        try:
            return "timeseries" if _is_timeseries(_read_table(path)) else "tabular"
        except Exception:
            return "tabular"
    return modality


def _table_summary(df: pd.DataFrame) -> dict:
    target = TargetIdentifier(df).identify()
    return {
        "rows": len(df),
        "cols": len(df.columns),
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "missing_pct": round(float(df.isna().mean().mean() * 100), 2),
        "suggested_target": target["suggested_target"],
        "target_confidence": round(float(target["confidence_score"]), 3),
    }


def _inspect(entry: Path, rel: str) -> InputReport:
    if entry.is_file() and _EXT.get(entry.suffix.lower()) == "tabular":
        df = _read_table(entry)
        if df.empty:
            raise ValueError("table has no rows")
        modality = "timeseries" if _is_timeseries(df) else "tabular"
        return InputReport(rel, modality, True, _table_summary(df))
    files = _files(entry)
    total = sum(f.stat().st_size for f in files)
    if not files or total == 0:
        raise ValueError("no data (empty file or folder)")
    return InputReport(rel, detect_modality(entry), True, {"files": len(files), "total_bytes": total})


def _section(r: InputReport) -> str:
    lines = [f"## {r.path}", f"- modality: {r.modality}", f"- usable: {'yes' if r.usable else 'no (unusable)'}"]
    lines += [f"- {k}: {v}" for k, v in r.summary.items()]
    if r.problem:
        lines.append(f"- problem: {r.problem}")
    return "\n".join(lines)


def run_intake(project: Project, bus: EventBus) -> list[InputReport]:
    bus.emit("intake", "step_status", {"id": "intake", "status": "running"})
    reports = []
    raw = project.root / "data" / "raw"
    for entry in sorted(raw.iterdir()) if raw.is_dir() else []:
        rel = entry.relative_to(project.root).as_posix()
        try:
            reports.append(_inspect(entry, rel))
        except Exception as e:  # any unreadable input is reported, never fatal
            guess = _EXT.get(entry.suffix.lower(), "unknown")
            reports.append(InputReport(rel, guess, False, {}, f"{type(e).__name__}: {e}"))
    body = "\n\n".join(_section(r) for r in reports) or "No inputs in data/raw/."
    project.write("DATA.md", f"# Data\n\n{body}\n")
    bus.emit("intake", "step_status", {"id": "intake", "status": "done", "inputs": len(reports)})
    return reports
