"""File tools: read, write, list inside the project."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from beyondml.toolkit.env import ToolEnv
from beyondml.workspace.project import PathError

_SKIP = {".git", ".beyondml"}


def file_tools(env: ToolEnv) -> list[Callable]:
    def read_file(path: str, max_lines: int = 400) -> dict:
        """Read a text file in the project.

        Args:
            path: Project-relative path.
            max_lines: Maximum number of lines to return.
        """
        try:
            p = env.project.path(path)
            max_lines = max(0, max_lines)
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except (PathError, OSError) as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "content": "\n".join(lines[:max_lines]), "truncated": len(lines) > max_lines}

    def write_file(path: str, content: str) -> dict:
        """Write (create or overwrite) a text file in the project.

        Args:
            path: Project-relative path. data/raw is read-only.
            content: Full file text.
        """
        try:
            env.project.write(path, content)
            path = env.project.path(path).relative_to(env.project.root.resolve()).as_posix()
        except (PathError, OSError) as e:
            return {"ok": False, "error": str(e)}
        env.bus.emit(env.agent, "artifact", {"path": path})
        if env.on_touch:
            env.on_touch([path])
        return {"ok": True, "path": path}

    def list_files(path: str = ".", pattern: str = "*") -> dict:
        """List files under a project directory.

        Args:
            path: Project-relative directory.
            pattern: Glob pattern; recursive only if it contains '**'.
        """
        try:
            if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
                return {"ok": False, "error": f"pattern must stay inside the project: {pattern}"}
            base = env.project.path(path)
            root = env.project.root.resolve()
            files = sorted(r.relative_to(root).as_posix() for r in (p.resolve() for p in base.glob(pattern))
                           if r.is_relative_to(root) and r.is_file() and not _SKIP & set(r.relative_to(root).parts))
        except (PathError, OSError, ValueError, NotImplementedError) as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "files": files}

    return [read_file, write_file, list_files]
