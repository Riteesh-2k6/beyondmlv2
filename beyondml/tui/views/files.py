"""Files: the project tree and a preview of the selected file."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable

from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import DirectoryTree, Static

from beyondml.tui.theme import DIM, MUTED
from beyondml.workspace.project import PathError

HIDDEN = {".git", ".beyondml", "__pycache__"}
MAX_BYTES = 200_000
LEXERS = {".py": "python", ".json": "json", ".toml": "toml", ".md": "markdown", ".yaml": "yaml", ".yml": "yaml",
          ".txt": "text", ".jsonl": "json", ".html": "html", ".css": "css", ".js": "javascript", ".ts": "typescript"}


class ProjectTree(DirectoryTree):
    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return [p for p in paths if p.name not in HIDDEN]


def csv_preview(text: str, rows: int = 20) -> Table:
    reader = csv.reader(io.StringIO(text))
    table = Table(box=None, header_style=MUTED, show_edge=False, pad_edge=False)
    header = next(reader, None) or []
    for h in header[:12]:
        table.add_column(h, overflow="fold", max_width=18)
    for n, row in enumerate(reader):
        if n >= rows:
            break
        table.add_row(*[c for c in row[:12]])
    return table


class FilesView(Horizontal):
    def compose(self) -> ComposeResult:
        yield ProjectTree(self.app.session.project.root, id="tree")
        with VerticalScroll(id="preview-scroll"):
            yield Static(Text("select a file to preview it", style=DIM), id="preview")

    def on_mount(self) -> None:
        self.preview_text = self.preview_title = ""

    @property
    def tree(self) -> ProjectTree:
        return self.query_one(ProjectTree)

    def preview(self, path: Path) -> None:
        project = self.app.session.project
        try:
            rel = Path(path).resolve().relative_to(project.root.resolve()).as_posix()
            target = project.path(rel)
        except (ValueError, PathError) as e:
            self._show(rel if "rel" in locals() else str(path), Text(str(e), style="#EF4444"))
            return
        size = target.stat().st_size
        raw = target.read_bytes()[:MAX_BYTES]
        if b"\0" in raw[:4096]:
            self._show(rel, Text(f"binary file, {size:,} bytes", style=DIM))
            return
        text = raw.decode("utf-8", errors="replace")
        note = Text(f"  showing the first {MAX_BYTES // 1000} KB of {size:,} bytes", style=DIM) if size > MAX_BYTES else None
        if target.suffix.lower() in (".csv", ".tsv"):
            body = csv_preview(text)
        else:
            lexer = LEXERS.get(target.suffix.lower(), "text")
            body = Syntax(text, lexer, theme="ansi_dark", line_numbers=True, word_wrap=True, background_color="#09090B")
        self._show(rel, body, note, plain=text)

    def _show(self, rel: str, body, note: Text | None = None, plain: str = "") -> None:
        from rich.console import Group
        head = Text.assemble((rel, "bold"))
        if note:
            head.append_text(note)
        self.preview_title = rel
        self.preview_text = plain or (body.plain if isinstance(body, Text) else "")
        self.query_one("#preview", Static).update(Group(head, Text(""), body))

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self.preview(event.path)

    def on_bus_event(self, ev: dict) -> None:
        if ev["kind"] == "artifact" or (ev["kind"] == "tool_result" and ev["data"].get("name") == "run_python"):
            self.tree.reload()
