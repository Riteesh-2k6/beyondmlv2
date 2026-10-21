"""Run the gateway: in the foreground for `beyondml serve`, or on a daemon thread under the TUI."""
from __future__ import annotations

import asyncio
import threading

import uvicorn
from fastapi import FastAPI


def run_foreground(app: FastAPI, host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port, log_level="warning")


class BackgroundServer:
    """The gateway on its own loop in a daemon thread, so the harness never stalls the terminal."""

    def __init__(self, app: FastAPI, host: str = "127.0.0.1", port: int = 7777) -> None:
        self.host, self.port = host, port
        self.server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
        self.thread = threading.Thread(target=self._run, name="beyondml-gateway", daemon=True)
        self.error: BaseException | None = None

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.server.serve())
        except BaseException as e:  # surfaced to the TUI, never a silent dead thread
            self.error = e
        finally:
            loop.close()

    def start(self, timeout: float = 10.0) -> "BackgroundServer":
        self.thread.start()
        deadline = threading.Event()
        for _ in range(int(timeout / 0.02)):
            if self.server.started or self.error or not self.thread.is_alive():
                break
            deadline.wait(0.02)
        if self.error:
            raise RuntimeError(f"gateway failed to start: {self.error}")
        return self

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self, timeout: float = 5.0) -> None:
        self.server.should_exit = True
        self.thread.join(timeout)
