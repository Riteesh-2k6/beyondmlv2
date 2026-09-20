"""
Convert Rich markup strings to span lists for the web UI log pane.

parse_rich_markup("[bold magenta]● Orch[/bold magenta] msg")
→ [{"text": "● Orch", "cls": "agent-header"}, {"text": " msg", "cls": ""}]
"""
from __future__ import annotations
import re
from typing import List, Dict

# Ordered longest-first so "bold magenta" matches before "bold"
_TAG_CLASS_MAP: List[tuple[str, str]] = [
    ("bold magenta",  "agent-header"),
    ("bold green",    "success"),
    ("bold cyan",     "info"),
    ("bold yellow",   "warn"),
    ("bold orange1",  "accent"),
    ("bold orange3",  "accent"),
    ("bold red",      "error-text"),
    ("bold white",    "bold"),
    ("bold",          "bold"),
    ("magenta",       "agent-header"),
    ("green",         "success"),
    ("cyan",          "info"),
    ("yellow",        "warn"),
    ("red",           "error-text"),
    ("orange1",       "accent"),
    ("orange3",       "accent"),
    ("dim",           "muted"),
    ("italic",        "italic"),
]

# Pre-build lookup dict from the ordered list (first match wins)
_TAG_TO_CLASS: Dict[str, str] = {}
for _tag, _cls in _TAG_CLASS_MAP:
    if _tag not in _TAG_TO_CLASS:
        _TAG_TO_CLASS[_tag] = _cls

_TAG_RE = re.compile(r'\[(/?)([^\]]+?)\]')


def _resolve_class(tag_body: str) -> str:
    """Return the CSS class for a Rich tag body, or '' for unknown tags."""
    normalized = tag_body.strip().lower()
    # Exact match first
    if normalized in _TAG_TO_CLASS:
        return _TAG_TO_CLASS[normalized]
    # Prefix match (handles e.g. "bold dim" → "muted bold" compound tags)
    for tag, cls in _TAG_CLASS_MAP:
        if normalized.startswith(tag):
            return cls
    return ""


def parse_rich_markup(text: str) -> List[Dict[str, str]]:
    """
    Parse a Rich markup string into a flat list of {text, cls} span dicts.

    - Unknown tags are stripped and their content passes through as plain text.
    - Nested / overlapping tags produce the innermost known class.
    - Returns [] for empty input.
    """
    if not text:
        return []

    spans: List[Dict[str, str]] = []
    style_stack: List[str] = []  # active CSS class names (innermost last)
    pos = 0

    for m in _TAG_RE.finditer(text):
        start, end = m.start(), m.end()
        closing, tag_body = bool(m.group(1)), m.group(2)

        # Emit any literal text before this tag
        if start > pos:
            current_cls = style_stack[-1] if style_stack else ""
            spans.append({"text": text[pos:start], "cls": current_cls})

        cls = _resolve_class(tag_body)

        if closing:
            # Pop the matching class from the stack (remove deepest occurrence)
            if cls and cls in style_stack:
                idx = len(style_stack) - 1 - style_stack[::-1].index(cls)
                style_stack.pop(idx)
        else:
            if cls:
                style_stack.append(cls)

        pos = end

    # Emit any trailing literal text
    if pos < len(text):
        current_cls = style_stack[-1] if style_stack else ""
        spans.append({"text": text[pos:], "cls": current_cls})

    # Collapse adjacent spans with the same class and drop empty-text spans
    merged: List[Dict[str, str]] = []
    for span in spans:
        if not span["text"]:
            continue
        if merged and merged[-1]["cls"] == span["cls"]:
            merged[-1] = {"text": merged[-1]["text"] + span["text"], "cls": span["cls"]}
        else:
            merged.append(span)

    return merged
