"""A BeyondML project folder: files, path guard, ledger, git checkpoints, settings."""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import threading
import time
import tomllib
from pathlib import Path
from typing import Iterable

GITIGNORE = "data/raw/\ndata/processed/\n.beyondml/sessions.sqlite*\n.beyondml/runs/\n*.pkl\n*.joblib\n*.pt\n"
DEFAULT_SETTINGS = {
    "model": {"default": "ollama_chat/qwen2.5:14b", "api_base": "http://127.0.0.1:11434"},
    "run": {"mode": "auto", "max_tool_calls": 40, "budget_minutes": 60, "budget_usd": 2.0},
}
SETTINGS_TOML = (
    '[model]\ndefault = "ollama_chat/qwen2.5:14b"\napi_base = "http://127.0.0.1:11434"\n\n'
    '[run]\nmode = "auto"\nmax_tool_calls = 40\nbudget_minutes = 60\nbudget_usd = 2.0\n'
)
DIRS = ["data/raw", "data/processed", "src", "models", "reports", "agents", ".beyondml/skills"]
LEDGER = "experiments.jsonl"
GIT_ID = ["-c", "user.name=beyondml", "-c", "user.email=beyondml@localhost"]


class ProjectError(Exception): ...
class PathError(ProjectError): ...


def beyondml_home() -> Path:
    return Path(os.environ.get("BEYONDML_HOME") or Path.home() / ".beyondml")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(root: Path) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(root.resolve(), threading.Lock())


class Project:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = _lock_for(self.root)

    def _git(self, *args: str, check: bool = True) -> str:
        r = subprocess.run(["git", *args], cwd=self.root, check=check, capture_output=True, text=True)
        return r.stdout.strip()

    @classmethod
    def init(cls, root: Path, goal: str | None = None) -> "Project":
        if shutil.which("git") is None:
            raise ProjectError("git is required but was not found on PATH")
        root = Path(root)
        for d in DIRS:
            (root / d).mkdir(parents=True, exist_ok=True)
        for rel, text in {
            "PROJECT.md": f"# Project\n\n## Goal\n{goal or '(not set yet)'}\n",
            "MEMORY.md": "# Memory\n",
            ".gitignore": GITIGNORE,
            ".beyondml/settings.toml": SETTINGS_TOML,
        }.items():
            f = root / rel
            if not f.exists():
                f.write_text(text, encoding="utf-8")
        p = cls(root)
        if not (root / ".git").exists():
            p._git("init")
            p._git("add", "-A")
            p._git(*GIT_ID, "commit", "-m", "beyondml: init")
        return p

    @classmethod
    def open(cls, root: Path) -> "Project":
        if not (Path(root) / ".beyondml").is_dir():
            raise ProjectError(f"{root} is not a BeyondML project (no .beyondml folder)")
        return cls(root)

    def path(self, rel: str, *, write: bool = False) -> Path:
        if Path(rel).is_absolute() or Path(rel).anchor:
            raise PathError(f"absolute paths are not allowed: {rel}")
        root = self.root.resolve()
        p = (root / rel).resolve()
        if not p.is_relative_to(root):
            raise PathError(f"path escapes the project: {rel}")
        if write and p.is_relative_to(root / "data" / "raw"):
            raise PathError(f"data/raw is read-only: {rel}")
        if write and p.is_relative_to(root / ".git"):
            raise PathError(f"writing under .git is not allowed: {rel}")
        return p

    def read(self, rel: str) -> str:
        p = self.path(rel)
        return p.read_text(encoding="utf-8") if p.is_file() else ""

    def write(self, rel: str, text: str) -> None:
        p = self.path(rel, write=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def append_memory(self, line: str) -> None:
        lines = self.read("MEMORY.md").splitlines() or ["# Memory"]
        lines = [lines[0]] + (lines[1:] + [line])[-200:]
        self.write("MEMORY.md", "\n".join(lines) + "\n")

    def ledger_append(self, record: dict) -> None:
        with self._lock, open(self.path(LEDGER, write=True), "a", encoding="utf-8") as f:
            f.write(json.dumps({**record, "ts": time.time()}) + "\n")

    def ledger(self) -> list[dict]:
        return [json.loads(l) for l in self.read(LEDGER).splitlines() if l.strip()]

    def checkpoint(self, message: str) -> str:
        with self._lock:
            self._git("add", "-A")
            if self._git("status", "--porcelain"):
                self._git(*GIT_ID, "commit", "-m", message)
            return self._git("rev-parse", "HEAD")

    def restore(self, sha: str, paths: Iterable[str]) -> None:
        root = self.root.resolve()
        targets = []
        for rel in paths:
            p = self.path(rel, write=True)
            if p == root or p.is_relative_to(root / ".git") or p.is_relative_to(root / ".beyondml"):
                raise PathError(f"cannot restore {rel}")
            targets.append(p)
        with self._lock:
            if subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"],
                              cwd=self.root, capture_output=True).returncode != 0:
                raise ProjectError(f"unknown commit: {sha}")
            for p in targets:
                git_rel = p.relative_to(root).as_posix()
                # cat-file -e exits 128 for a missing path (same as real errors), so use ls-tree:
                # exit 0 with empty output = absent at sha, non-zero = real failure.
                r = subprocess.run(["git", "ls-tree", "--name-only", sha, "--", git_rel],
                                   cwd=self.root, capture_output=True, text=True)
                if r.returncode != 0:
                    raise ProjectError(f"git ls-tree failed (exit {r.returncode}) for {git_rel}")
                if r.stdout.strip():
                    self._git("checkout", sha, "--", git_rel)
                elif p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink(missing_ok=True)

    def settings(self) -> dict:
        out = copy.deepcopy(DEFAULT_SETTINGS)
        for f in (beyondml_home() / "settings.toml", self.root / ".beyondml" / "settings.toml"):
            if f.is_file():
                out = _merge(out, tomllib.loads(f.read_text(encoding="utf-8")))
        return out

    def add_data(self, src: Path) -> Path:
        src = Path(src)
        dest = self.root / "data" / "raw" / src.name
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        return dest
