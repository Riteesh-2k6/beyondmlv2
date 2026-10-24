"""
BeyondML CLI — entry point for the terminal-native AutoML platform.

Usage:
    beyondml [--project DIR]          Open the TUI on a project, with the gateway in-process
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


TUI_OPTIONS = [
    click.option("--project", "project_dir", default=".", show_default=True, type=click.Path(file_okay=False)),
    click.option("--port", default=7777, show_default=True, help="Gateway port for the web UI."),
    click.option("--no-serve", is_flag=True, default=False, help="Do not start the gateway alongside the TUI."),
    click.option("--mode", type=click.Choice(["ask", "auto", "autonomous"]), default=None),
    click.option("--model", default=None, help="Model for every agent, e.g. ollama_chat/qwen2.5:14b."),
]


def tui_options(f):
    for option in reversed(TUI_OPTIONS):
        f = option(f)
    return f


def _launch_tui(project_dir, port, no_serve, mode, model):
    from beyondml.tui.app import BeyondMLApp

    session = _open_session(project_dir, mode=mode, model=model)
    server = url = None
    if not no_serve:
        from beyondml.gateway.app import create_app
        from beyondml.gateway.serve import BackgroundServer

        try:
            server = BackgroundServer(create_app({session.name: session}), port=port).start()
            url = server.url
        except RuntimeError as e:
            click.echo(f"gateway not started ({e}); the TUI runs alone", err=True)
    try:
        BeyondMLApp(session, gateway_url=url).run()
    finally:
        if server:
            server.stop()


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0", prog_name="beyondml")
@tui_options
@click.pass_context
def main(ctx, project_dir, port, no_serve, mode, model):
    """BeyondML: an ML-engineering agent harness. With no command, opens the TUI on the current project."""
    if ctx.invoked_subcommand is None:
        _launch_tui(project_dir, port, no_serve, mode, model)


@main.command()
@tui_options
def tui(project_dir, port, no_serve, mode, model):
    """Open the interactive TUI (the same as running beyondml with no command)."""
    _launch_tui(project_dir, port, no_serve, mode, model)


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
        if failed:  # a failed script has no "error": show its last stderr line, else its exit code
            tail = (resp.get("stderr_tail") or "").strip().splitlines()
            detail = resp.get("error") or resp.get("errors") or (tail[-1] if tail else f"exit {resp.get('exit_code')}")
        short = f"{data.get('name')} error: {str(detail)[:100]}" if failed else f"{data.get('name')} ok"
    elif kind == "step_status":
        short = f"{data.get('id')} {data.get('status')}" + (f" — {data['reason']}" if data.get("reason") else "")
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
    try:
        return await asyncio.to_thread(click.confirm, text, default=True)
    except click.Abort:  # no terminal answer (EOF, Ctrl-C at the prompt) means no
        return False


@main.command()
@click.argument("goal", required=False)
@click.option("--data", "data", multiple=True, type=click.Path(), help="Input file or folder (repeatable).")
@click.option("--project", "project_dir", default=".", show_default=True, type=click.Path(file_okay=False))
@click.option("--mode", type=click.Choice(["ask", "auto", "autonomous"]), default=None,
              help="ask: confirm every run; auto: confirm installs and network; autonomous: never ask.")
@click.option("--model", default=None, help="Model for every agent, e.g. ollama_chat/qwen3:8b.")
def run(goal, data, project_dir, mode, model):
    """Plan and build an ML project headless (resumes an unfinished plan when GOAL is omitted)."""
    from beyondml.harness.run import GoalRequired, run_goal

    try:
        plan = asyncio.run(run_goal(Path(project_dir), goal, [Path(d) for d in data], mode=mode, model=model,
                                    asker=_ask, approver=_approve, on_event=_print_event))
    except GoalRequired as e:
        raise click.UsageError(str(e))
    except Exception as e:  # PlanError, ProjectError, an unreachable model, ...: one line, not a traceback
        raise click.ClickException(f"{type(e).__name__}: {e}") from None
    lines = [f"{s.id}: {s.status}" + (f" — {s.last_failure}" if s.status != "done" and s.last_failure else "")
             for s in plan.steps]
    click.echo("\n" + "\n".join(lines))
    if any(s.status != "done" for s in plan.steps):
        raise SystemExit(1)


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
