"""
BeyondML CLI — entry point for the terminal-native AutoML platform.

Usage:
    beyondml run          Launch the interactive TUI
    beyondml benchmark    Run PMLB benchmark suite
    beyondml --version    Show version
"""

import click
import sys
import os


@click.group()
@click.version_option(version="0.1.0", prog_name="beyondml")
def main():
    """BeyondML — AI Agent Orchestration Platform for AutoML."""
    pass


@main.command()
def run():
    """Launch the interactive TUI application."""
    # Ensure the project root is on sys.path so tui_app.py can import beyondml
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import here to avoid heavy imports on --help
    from tui_app import BeyondMLApp

    app = BeyondMLApp()
    app.run()


@main.command()
@click.option(
    "--datasets",
    "-d",
    multiple=True,
    default=None,
    help="PMLB dataset names to benchmark (e.g. -d titanic -d car). Defaults to titanic + car.",
)
def benchmark(datasets):
    """Run the PMLB benchmark suite."""
    import asyncio
    from beyondml.engine.benchmarker import PMLBRunner

    ds = list(datasets) if datasets else None
    runner = PMLBRunner(datasets=ds)
    asyncio.run(runner.run_benchmark())


@main.command()
@click.option("--port", default=7777, show_default=True, help="Port for the web server.")
@click.option("--no-browser", is_flag=True, default=False, help="Don't open browser automatically.")
def web(port, no_browser):
    """Launch the BeyondML Web UI (FastAPI + React)."""
    import threading
    import time
    import webbrowser
    import uvicorn

    def _open():
        time.sleep(1.5)
        if not no_browser:
            webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=_open, daemon=True).start()
    uvicorn.run(
        "beyondml.web.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
