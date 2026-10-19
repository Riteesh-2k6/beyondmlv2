"""run_python and install_package: run project scripts in a subprocess, with a ledger record per run."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Mapping

from beyondml.toolkit.env import ToolEnv
from beyondml.workspace.events import EventBus
from beyondml.workspace.project import PathError, Project

SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.I)
_NAME = re.compile(r"[A-Za-z0-9_.\[\],<>=!~-]+")
_INSTALL_LOCK = threading.Lock()
_SKIP_TOP = {".git", ".beyondml", "agents"}
TAIL = 200
_WIN = sys.platform == "win32"


def clean_env(environ: Mapping[str, str]) -> dict[str, str]:
    env = {k: v for k, v in environ.items() if not SECRET.search(k)}
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _tail(b: bytes | None) -> str:
    return "\n".join((b or b"").decode("utf-8", errors="replace").splitlines()[-TAIL:])


def _snapshot(root: Path, sub: str = ".") -> dict[str, tuple[int, int]]:
    out = {}
    base = root / sub
    for d, dirs, files in os.walk(base):
        if Path(d) == root:
            dirs[:] = [x for x in dirs if x not in _SKIP_TOP]
        for f in files:
            p = Path(d, f)
            try:
                st = p.stat()
            except OSError:
                continue
            out[p.relative_to(root).as_posix()] = (st.st_mtime_ns, st.st_size)
    return out


def _data_hash(root: Path) -> str:
    snap = _snapshot(root, "data/raw") if (root / "data/raw").is_dir() else {}
    rows = sorted((k, v[1], v[0]) for k, v in snap.items())
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def _kill_tree(proc: subprocess.Popen) -> None:
    if _WIN:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass


def _fail(error: str) -> dict:
    return {"ok": False, "error": error}


def _spawn(argv: list[str], cwd: Path, env: dict, timeout: int) -> tuple[int | None, bool, bytes, bytes, float]:
    """Run argv; on timeout kill the whole tree and wait at most 5 s more. Always returns."""
    kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if _WIN else {"start_new_session": True})
    t0 = time.time()
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, **kw)
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        timed_out = True
        out, err = e.stdout or b"", e.stderr or b""
        _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired as e2:
            out, err = e2.stdout or out, e2.stderr or err
            for f in (proc.stdout, proc.stderr):
                try:
                    f.close()
                except OSError:
                    pass
    return proc.returncode, timed_out, out, err, round(time.time() - t0, 2)


def run_script(project: Project, script: str, args: list[str], timeout: int, *, agent: str,
               bus: EventBus | None = None, python: str = sys.executable) -> dict:
    try:
        path = project.path(script)
        root = project.root.resolve()
        rel = path.relative_to(root).as_posix()
    except (PathError, OSError, ValueError) as e:
        return _fail(str(e))
    if path.suffix != ".py" or not path.is_file():
        return _fail(f"script must be an existing .py file: {script}")
    run_id = uuid.uuid4().hex[:8]
    metrics_path = root / ".beyondml" / "runs" / run_id / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**clean_env(os.environ), "BEYONDML_METRICS": str(metrics_path)}
    before = _snapshot(root)
    try:
        code, timed_out, out, err, seconds = _spawn([python, str(path), *args], root, env, max(1, min(timeout, 3600)))
    except OSError as e:
        return _fail(str(e))
    after = _snapshot(root)
    changed = sorted(k for k, v in after.items() if before.get(k) != v)
    metrics = None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    project.ledger_append({"run_id": run_id, "agent": agent, "script": rel, "args": args, "exit_code": code,
                           "timed_out": timed_out, "seconds": seconds, "files_changed": changed,
                           "metrics": metrics, "data_hash": _data_hash(root)})
    if bus:
        for c in changed:
            bus.emit(agent, "artifact", {"path": c})
    return {"ok": code == 0 and not timed_out, "exit_code": code, "timed_out": timed_out,
            "stdout_tail": _tail(out), "stderr_tail": _tail(err), "files_changed": changed,
            "metrics": metrics, "seconds": seconds, "run_id": run_id}


def python_tools(env: ToolEnv, python: str = sys.executable) -> list[Callable]:
    async def run_python(script: str, args: list[str] = [], timeout: int = 600) -> dict:
        """Run a Python script from the project and return its output.

        Args:
            script: Project-relative path of a .py file.
            args: Command-line arguments.
            timeout: Seconds before the run is killed.
        """
        try:
            r = await asyncio.to_thread(run_script, env.project, script, list(args), timeout,
                                        agent=env.agent, bus=env.bus, python=python)
        except Exception as e:  # tools never raise to the model
            return _fail(f"{type(e).__name__}: {e}")
        if r.get("files_changed") and env.on_touch:
            env.on_touch(r["files_changed"])
        return r

    async def install_package(name: str) -> dict:
        """Install a Python package with pip.

        Args:
            name: Package spec, for example "pandas" or "scikit-learn>=1.4".
        """
        if (not _NAME.fullmatch(name) or name[0] in "-." or not re.search(r"[A-Za-z0-9]", name)):
            return _fail(f"invalid package name: {name!r}")

        def go():
            with _INSTALL_LOCK:  # one pip at a time
                return _spawn([python, "-m", "pip", "install", name], env.project.root,
                              clean_env(os.environ), 600)

        try:
            code, timed_out, out, err, _ = await asyncio.to_thread(go)
        except OSError as e:
            return _fail(str(e))
        return {"ok": code == 0 and not timed_out, "exit_code": code,
                "output_tail": _tail(out + err)}

    return [run_python, install_package]
