"""plotext charts as Rich text, in the obsidian palette."""
from __future__ import annotations

import plotext as plt
from rich.text import Text

PALETTE = [(249, 115, 22), (34, 211, 238), (217, 70, 239), (34, 197, 94), (234, 179, 8), (113, 113, 122)]
MEAN = (113, 113, 122)


def _build() -> Text:
    try:
        return Text.from_ansi(plt.build())
    finally:
        plt.clf()


def line_chart(title: str, x: list, series: dict[str, list], *, width: int, height: int = 12,
               xlabel: str = "", colors: dict[str, tuple] | None = None) -> Text:
    plt.clf()
    plt.plotsize(max(width, 30), max(height, 6))
    plt.theme("clear")
    for i, (name, ys) in enumerate(series.items()):
        n = min(len(x), len(ys))
        if n == 0:
            continue
        colour = (colors or {}).get(name) or PALETTE[i % len(PALETTE)]
        plt.plot(x[:n], ys[:n], label=name, color=colour, marker="braille")
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    return _build()


def bar_chart(title: str, x: list, y: list, *, width: int, height: int = 8, color: tuple = PALETTE[1]) -> Text:
    plt.clf()
    plt.plotsize(max(width, 30), max(height, 5))
    plt.theme("clear")
    n = min(len(x), len(y))
    if n:
        plt.bar(x[:n], y[:n], color=color, width=0.5)
    plt.title(title)
    return _build()


def scatter_chart(title: str, x: list, series: dict[str, list], *, width: int, height: int = 12) -> Text:
    plt.clf()
    plt.plotsize(max(width, 30), max(height, 6))
    plt.theme("clear")
    for i, (name, ys) in enumerate(series.items()):
        n = min(len(x), len(ys))
        if n:
            plt.scatter(x[:n], ys[:n], label=name, color=PALETTE[i % len(PALETTE)])
    plt.title(title)
    return _build()


def chart_event(data: dict, *, width: int) -> Text:
    """Render a `chart` event: {title, kind, x, series}."""
    title, x, series = str(data.get("title", "chart")), list(data.get("x") or []), dict(data.get("series") or {})
    if not x or not series:
        return Text(f"{title}: nothing to plot", style="#52525B")
    kind = data.get("kind", "line")
    if kind == "bar":
        name, ys = next(iter(series.items()))
        return bar_chart(f"{title} · {name}", x, list(ys), width=width)
    if kind == "scatter":
        return scatter_chart(title, x, series, width=width)
    return line_chart(title, x, series, width=width)
