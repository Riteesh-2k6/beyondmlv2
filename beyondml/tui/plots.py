"""plotext charts as Rich text, in the obsidian palette."""
from __future__ import annotations

import json

import plotext as plt
from rich.text import Text

PALETTE = [(249, 115, 22), (34, 211, 238), (217, 70, 239), (34, 197, 94), (234, 179, 8), (113, 113, 122)]
MEAN = (113, 113, 122)


def _build() -> Text:
    try:
        return Text.from_ansi(plt.build())
    finally:
        plt.clf()


def _start(width: int, height: int, title: str) -> None:
    plt.clf()
    plt.plotsize(max(width, 30), max(height, 6))
    plt.theme("clear")
    plt.title(title)


def line_chart(title: str, x: list, series: dict[str, list], *, width: int, height: int = 12,
               xlabel: str = "", colors: dict[str, tuple] | None = None) -> Text:
    _start(width, height, title)
    for i, (name, ys) in enumerate(series.items()):
        n = min(len(x), len(ys))
        if n:
            plt.plot(x[:n], ys[:n], label=name, color=(colors or {}).get(name) or PALETTE[i % len(PALETTE)],
                     marker="braille")
    if xlabel:
        plt.xlabel(xlabel)
    return _build()


def bar_chart(title: str, x: list, y: list, *, width: int, height: int = 8, color: tuple = PALETTE[1]) -> Text:
    _start(width, height, title)
    n = min(len(x), len(y))
    if n:
        plt.bar(x[:n], y[:n], color=color, width=0.5)
    return _build()


def scatter_chart(title: str, x: list, series: dict[str, list], *, width: int, height: int = 12) -> Text:
    _start(width, height, title)
    for i, (name, ys) in enumerate(series.items()):
        n = min(len(x), len(ys))
        if n:
            plt.scatter(x[:n], ys[:n], label=name, color=PALETTE[i % len(PALETTE)])
    return _build()


def _numeric(values) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def figure_to_text(fig: dict, title: str, *, width: int, height: int = 14) -> Text:
    """Approximate a Plotly figure ({data, layout}) with plotext: one trace per line/scatter/bar."""
    traces = [t for t in fig.get("data", []) if isinstance(t, dict) and t.get("y")]
    if not traces:
        return Text(f"{title}: the figure has no plottable traces", style="#52525B")
    _start(width, height, title)
    labels: list | None = None
    for i, trace in enumerate(traces):
        y = [v for v in trace["y"] if v is not None]
        x = list(trace.get("x") or range(1, len(y) + 1))[:len(y)]
        y = y[:len(x)]
        if not _numeric(y) or not x:
            continue
        if not _numeric(x):  # categories: positions on the axis, names as ticks
            labels = [str(v) for v in x]
            x = list(range(1, len(x) + 1))
        name = str(trace.get("name") or f"trace {i + 1}")
        colour = PALETTE[i % len(PALETTE)]
        kind, mode = trace.get("type", "scatter"), str(trace.get("mode") or "lines")
        if kind == "bar":
            plt.bar(x, y, label=name, color=colour, width=0.5)
        elif "lines" in mode:
            plt.plot(x, y, label=name, color=colour, marker="braille")
        else:
            plt.scatter(x, y, label=name, color=colour)
    if labels:
        plt.xticks(list(range(1, len(labels) + 1)), labels)
    axis = (fig.get("layout") or {}).get("xaxis", {}).get("title")
    if isinstance(axis, dict) and axis.get("text"):
        plt.xlabel(str(axis["text"]))
    return _build()


def chart_event(data: dict, *, width: int, project=None) -> Text:
    """Render a `chart` event: {title, path} to a Plotly figure file, or inline {kind, x, series}."""
    title = str(data.get("title", "chart"))
    path = data.get("path")
    if path:
        if project is None:
            return Text(f"{title}: {path}", style="#52525B")
        try:
            fig = json.loads(project.read(str(path)))
            if not isinstance(fig, dict):
                raise ValueError("not a figure object")
        except Exception as e:  # a missing or malformed file is shown, never raised into the app
            return Text(f"{title}: {path} could not be read ({type(e).__name__}: {e})", style="#EF4444")
        return figure_to_text(fig, title, width=width)
    x, series = list(data.get("x") or []), dict(data.get("series") or {})
    if not x or not series:
        return Text(f"{title}: nothing to plot", style="#52525B")
    kind = data.get("kind", "line")
    if kind == "bar":
        name, ys = next(iter(series.items()))
        return bar_chart(f"{title} · {name}", x, list(ys), width=width)
    if kind == "scatter":
        return scatter_chart(title, x, series, width=width)
    return line_chart(title, x, series, width=width)
