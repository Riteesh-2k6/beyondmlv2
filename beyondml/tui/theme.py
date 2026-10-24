"""Obsidian: the one palette both clients share (PRODUCT.md tone)."""
from __future__ import annotations

import time

from rich.text import Text
from textual.theme import Theme

OBSIDIAN = Theme(
    name="obsidian", primary="#F97316", secondary="#22D3EE", accent="#F97316", warning="#EAB308",
    error="#EF4444", success="#22C55E", foreground="#E4E4E7", background="#09090B", surface="#18181B",
    panel="#27272A", dark=True,
    variables={"footer-key-foreground": "#F97316", "footer-background": "#09090B", "block-cursor-background": "#27272A"},
)

MUTED = "#71717A"
DIM = "#52525B"
STATUS_GLYPH = {"pending": "○", "running": "◐", "done": "●", "failed": "✕", "skipped": "◌", "n/a": "·"}
STATUS_COLOR = {"pending": MUTED, "running": "#F97316", "done": "#22C55E", "failed": "#EF4444", "skipped": DIM,
                "n/a": DIM}


def status_text(status: str) -> Text:
    return Text(f"{STATUS_GLYPH.get(status, '?')} {status}", style=STATUS_COLOR.get(status, MUTED))


def elapsed(started: float | None, finished: float | None) -> str:
    if not started:
        return "—"
    seconds = (finished or time.time()) - started
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def fmt_num(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e4 else f"{v:.3e}"
    return str(v)
