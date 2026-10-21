"""
BeyondML CLI — entry point for the terminal-native AutoML platform.

Usage:
    beyondml                          Launch the interactive TUI (legacy, until the new TUI lands)
    beyondml init [DIR] --goal TEXT   Create a BeyondML project folder
    beyondml run [GOAL] --data PATH   Plan and build an ML project headless, printing every agent event
    beyondml serve                    Run the gateway headless (REST, SSE, web UI)
    beyondml benchmark                Run PMLB benchmark suite
    beyondml --version                Show version
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import click


def _launch_tui():
    # Ensure the project root is on sys.path so tui_app.py can import beyondml
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import here to avoid heavy imports on --help
    from tui_app import BeyondMLApp

    BeyondMLApp().run()


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0", prog_name="beyondml")
@click.pass_context
def main(ctx):
    """BeyondML — AI Agent Orchestration Platform for AutoML."""
    if ctx.invoked_subcommand is None:
        _launch_tui()


@main.command()
def tui():
    """Launch the interactive TUI application."""
    _launch_tui()


@main.command()
@click.argument("directory", default=".", type=click.Path(file_okay=False))
@click.option("--goal", default=None, help="What the project should achieve.")
def init(directory, goal):
    """Create a BeyondML project folder (it is also the repo you ship)."""
    from beyondml.workspace.project import Project

    project = Project.init(Path(directory), goal)
    click.echo(f"BeyondML project ready at {project.root.resolve()}")


def format_event(ev: dict) -> str | None:
    """One readable line per bus event; None for partial message deltas."""
    agent, kind, data = ev["agent"], ev["kind"], ev["data"]
    if kind == "message_delta":
        if data.get("partial"):
            return None
        short = data.get("text", "")
    elif kind == "tool_call":
        short = f"{data.get('name')}({json.dumps(data.get('args', {}), default=str)[:80]})"
    elif kind == "tool_result":
        resp = data.get("response")
        failed = isinstance(resp, dict) and (resp.get("ok") is False or "error" in resp)
        detail = (resp.get("error") or resp.get("errors")) if failed else None
        short = f"{data.get('name')} error: {str(detail)[:100]}" if failed else f"{data.get('name')} ok"
    elif kind == "step_status":
        short = f"{data.get('id')} {data.get('status')}"
    else:
        short = json.dumps(data, default=str)[:120]
    return f"[{agent}] {kind}: {short}"


def _print_event(ev: dict) -> None:
    line = format_event(ev)
    if line:
        click.echo(line)


async def _ask(question: str) -> str:
    return await asyncio.to_thread(click.prompt, f"? {question}")


async def _approve(payload: dict) -> bool:
    kind = payload["kind"]
    if kind == "plan":
        steps = payload["plan"]["steps"]
        click.echo("Plan:\n" + "\n".join(f"  {s['id']} ({s['skill']}): {s['goal']}" for s in steps))
        text = "Run this plan?"
    elif kind == "tool":
        text = f"Allow {payload['agent']} to run {payload['tool']}({json.dumps(payload['args'], default=str)[:120]})?"
    else:
        text = f"Budget reached ({payload['reason']}). Keep going?"
    return await asyncio.to_thread(click.confirm, text, default=True)


@main.command()
@click.argument("goal", required=False)
@click.option("--data", "data", multiple=True, type=click.Path(exists=True), help="Input file or folder (repeatable).")
@click.option("--project", "project_dir", default=".", show_default=True, type=click.Path(file_okay=False))
@click.option("--mode", type=click.Choice(["ask", "auto", "autonomous"]), default=None,
              help="ask: confirm every run; auto: confirm installs and network; autonomous: never ask.")
@click.option("--model", default=None, help="Model for every agent, e.g. ollama_chat/qwen2.5:14b.")
def run(goal, data, project_dir, mode, model):
    """Plan and build an ML project headless (resumes an unfinished plan when GOAL is omitted)."""
    from beyondml.harness.run import run_goal

    try:
        plan = asyncio.run(run_goal(Path(project_dir), goal, [Path(d) for d in data], mode=mode, model=model,
                                    asker=_ask, approver=_approve, on_event=_print_event))
    except ValueError as e:
        raise click.UsageError(str(e))
    click.echo("\n" + "\n".join(f"{s.id}: {s.status}" for s in plan.steps))


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


def _open_session(project_dir, **kw):
    """A Session on an existing project, or exit 2 with the init hint."""
    from beyondml.gateway.session import Session

    root = Path(project_dir)
    if not (root / ".beyondml").is_dir():
        click.echo(f"{root.resolve()} is not a BeyondML project. Run `beyondml init` there first.", err=True)
        sys.exit(2)
    return Session(root, **kw)


@main.command()
@click.option("--project", "project_dir", default=".", show_default=True, type=click.Path(file_okay=False))
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address; loopback by default.")
@click.option("--port", default=7777, show_default=True)
@click.option("--mode", type=click.Choice(["ask", "auto", "autonomous"]), default=None)
@click.option("--model", default=None, help="Model for every agent, e.g. ollama_chat/qwen2.5:14b.")
def serve(project_dir, host, port, mode, model):
    """Run the gateway headless: REST, SSE events and the web UI for one project."""
    from beyondml.gateway.app import create_app
    from beyondml.gateway.serve import run_foreground

    session = _open_session(project_dir, mode=mode, model=model)
    click.echo(f"BeyondML gateway on http://{host}:{port} for {session.project.root.resolve()}")
    run_foreground(create_app({session.name: session}), host, port)


if __name__ == "__main__":
    main()
