"""Runs a plan: one dedicated agent per ready step, verified by our own checks, with retry, replan and resume."""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Awaitable, Callable

from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.runners import Runner
from google.genai import types

from beyondml.harness.adk import (APP_NAME, ModelResolver, TurnResult, confirmation_reply, ensure_session, run_turn,
                                  session_service, user_text)
from beyondml.harness.factory import build_step_agent
from beyondml.harness.skills import SkillInfo
from beyondml.toolkit.env import Asker, StepAborted, StepOutcome
from beyondml.toolkit.python import run_script
from beyondml.workspace.events import EventBus
from beyondml.workspace.plan import Plan, Step, save_plan
from beyondml.workspace.project import PathError, Project, ProjectError

Approver = Callable[[dict], Awaitable[bool]]
Replanner = Callable[[Plan, dict], Awaitable[Plan]]
NUDGE = "Call finish_step when your outputs exist, or explain what blocks you."
MAX_REPLANS = 3
# ponytail: path-scoped restore assumes parallel steps write disjoint files; a shared file written by two running
# steps can be restored over. Add per-step output ownership if that happens.
# ponytail: plan edits from the lead during a run are not merged live. Sub-project 4 adds a reload-and-merge per
# scheduling round.


class Executor:
    def __init__(self, project: Project, bus: EventBus, plan: Plan, skills: dict[str, SkillInfo], *,
                 resolver: ModelResolver, mode: str, asker: Asker, approver: Approver,
                 replanner: Replanner | None = None, max_tool_calls: int = 40,
                 budget_minutes: float | None = None, budget_usd: float | None = None) -> None:
        self.project, self.bus, self.plan, self.skills = project, bus, plan, skills
        self.resolver, self.mode, self.asker, self.approver = resolver, mode, asker, approver
        self.replanner, self.max_tool_calls = replanner, max_tool_calls
        self.budget_minutes, self.budget_usd = budget_minutes, budget_usd
        self.cost = 0.0
        self._replans = 0
        self._save_lock = threading.Lock()  # on_touch may fire from a tool's worker thread
        self._sessions = session_service(project)

    # ── persistence ────────────────────────────────────────────
    def _save(self) -> None:
        with self._save_lock:
            save_plan(self.project, self.plan)

    def _status(self, step: Step, status: str, reason: str = "") -> None:
        step.status = status
        self._save()
        self.bus.emit("executor", "step_status", {"id": step.id, "status": status, "reason": reason})

    def _touch(self, step_id: str) -> Callable[[list[str]], None]:
        def on_touch(paths: list[str]) -> None:
            with self._save_lock:
                step = self.plan.step(step_id)  # R8: look up by id, the plan object may have been replaced
                step.touched += [p for p in paths if p not in step.touched]
                save_plan(self.project, self.plan)
        return on_touch

    def _restore(self, step: Step) -> None:
        """R15: put back only what this step wrote, never a sibling's in-flight files or a done step's outputs."""
        keep = set()
        for other in self.plan.steps:
            if other.id != step.id and other.status == "running":
                keep |= set(other.touched)
            elif other.status == "done":
                keep |= set(other.outputs)
        paths = [p for p in step.touched if p not in keep] if step.checkpoint else []
        refused = []
        for p in paths:  # one path at a time, so a refused one (say under data/raw) can't block the rest
            try:
                self.project.restore(step.checkpoint, [p])
            except ProjectError as e:
                refused.append(str(e))
        if refused:
            self.bus.emit("executor", "error", {"code": "restore_failed", "message": f"{step.id}: " + "; ".join(refused)})
        step.touched = []

    # ── success check ──────────────────────────────────────────
    def _last_metrics(self, step_id: str) -> dict:
        for record in reversed(self.project.ledger()):
            if record.get("agent") == step_id and record.get("metrics"):
                return record["metrics"]
        return {}

    def check(self, step: Step) -> tuple[bool, str]:
        sc = step.success_check
        metrics = None
        if sc.run:
            r = run_script(self.project, sc.run, [], 600, agent="executor", bus=self.bus)
            if not r["ok"]:
                detail = r.get("error") or r.get("stderr_tail", "")[-500:]
                return False, f"success check {sc.run} failed (exit {r.get('exit_code')}): {detail}"
            metrics = r["metrics"]
        for rel in sc.expects:
            try:
                if not self.project.path(rel).exists():
                    return False, f"expected output missing: {rel}"
            except PathError as e:
                return False, f"expected output not allowed: {rel} ({e})"
        if sc.metric:
            value = (metrics if metrics is not None else self._last_metrics(step.id)).get(sc.metric)
            if not isinstance(value, (int, float)):
                return False, f"metric {sc.metric} not reported"
            if sc.min is not None and value < sc.min:
                return False, f"metric {sc.metric}={value} below minimum {sc.min}"
        return True, ""

    # ── one attempt of one step ────────────────────────────────
    def _add_cost(self, model, turn: TurnResult) -> None:
        if not (turn.prompt_tokens or turn.completion_tokens):
            return  # nothing reported, nothing to price (and no slow litellm import on the event loop)
        name = getattr(model, "model", model)
        try:
            import litellm
            prompt, completion = litellm.cost_per_token(model=name, prompt_tokens=turn.prompt_tokens,
                                                        completion_tokens=turn.completion_tokens)
            self.cost += prompt + completion
        except Exception:
            pass  # local and unknown models cost nothing we can price

    async def _attempt(self, step_id: str) -> tuple[StepOutcome, str | None]:
        step = self.plan.step(step_id)
        outcome = StepOutcome()
        agent = build_step_agent(self.project, self.bus, self.plan, step, self.skills, resolver=self.resolver,
                                 mode=self.mode, asker=self.asker, outcome=outcome, on_touch=self._touch(step_id))
        runner = Runner(app_name=APP_NAME, agent=agent, session_service=self._sessions)
        session_id = f"{step_id}-{step.attempts}-{uuid.uuid4().hex[:6]}"  # never reuse an old plan's conversation
        await ensure_session(self._sessions, session_id)
        message, nudged = user_text("Start your step."), False
        while True:
            try:
                turn = await run_turn(runner, session_id, message, self.bus, max_llm_calls=self.max_tool_calls + 1)
            except LlmCallsLimitExceededError:
                self.bus.emit(step_id, "budget", {"agent": step_id, "reason": "tool-call cap",
                                                  "limit": self.max_tool_calls})
                if self.mode != "autonomous" and await self.approver(
                        {"kind": "budget", "agent": step_id, "reason": "tool-call cap"}):
                    message = user_text("Continue.")
                    continue
                return outcome, "tool-call cap reached"
            except StepAborted:
                return outcome, "three tool errors in a row"
            self._add_cost(agent.model, turn)
            if outcome.consecutive_errors >= 3:  # R5: ADK may turn the tool's exception into an error response
                return outcome, "three tool errors in a row"
            if turn.confirmations and self.mode != "autonomous":
                parts = []
                for c in turn.confirmations:
                    approved = await self.approver({"kind": "tool", "agent": step_id, "tool": c["tool"],
                                                    "args": c["args"], "hint": c["hint"]})
                    parts += confirmation_reply(c["call_id"], approved).parts
                message = types.Content(role="user", parts=parts)
                continue
            if outcome.finished:
                return outcome, None
            if nudged:
                return outcome, "agent ended without finish_step"
            message, nudged = user_text(NUDGE), True

    # ── the plan loop ──────────────────────────────────────────
    def _recover(self) -> None:
        for step in self.plan.steps:
            if step.status == "running":
                self._restore(step)
                step.status = "pending"
        self._save()

    async def _settle(self, step_id: str, outcome: StepOutcome, error: str | None) -> None:
        step = self.plan.step(step_id)
        if error:
            ok, reason = False, error
        elif outcome.finished:
            ok, reason = await asyncio.to_thread(self.check, step)
        else:
            ok, reason = False, "agent ended without finish_step"
        if ok:
            step.summary = outcome.summary
            self.project.checkpoint(f"done {step_id}")
            self._status(step, "done")
        elif step.attempts < 2:
            self._restore(step)
            step.last_failure = reason
            self._status(step, "pending", reason)
        elif self.replanner and self._replans < MAX_REPLANS:
            step.last_failure = reason
            self._status(step, "failed", reason)  # failed, so merge_revision drops it unless the revision re-lists it
            self._replans += 1
            try:
                revision = await self.replanner(self.plan, {"step": step_id, "reason": reason,
                                                            "summary": outcome.summary})
            except Exception as e:
                self.bus.emit("executor", "error", {"code": "replan_failed", "message": str(e)})
                return
            self.plan = self.plan.merge_revision(revision)
            self._save()
            self.bus.emit("executor", "plan_updated", {"steps": [s.id for s in self.plan.steps]})
        else:
            step.last_failure = reason
            self._status(step, "failed", reason)

    async def _over_budget(self, started: float) -> bool:
        minutes = (time.monotonic() - started) / 60
        reasons = []
        if self.budget_minutes is not None and minutes > self.budget_minutes:
            reasons.append(f"{minutes:.1f} min > {self.budget_minutes} min")
        if self.budget_usd is not None and self.cost > self.budget_usd:
            reasons.append(f"${self.cost:.2f} > ${self.budget_usd}")
        if not reasons:
            return False
        reason = "; ".join(reasons)
        self.bus.emit("executor", "budget", {"agent": None, "reason": reason})
        if self.mode == "autonomous":
            return True  # nobody to ask: stop cleanly, pending steps stay pending for a later run
        if await self.approver({"kind": "budget", "agent": None, "reason": reason}):
            if self.budget_minutes is not None:
                self.budget_minutes += self.budget_minutes  # approval grants one more budget's worth
            if self.budget_usd is not None:
                self.budget_usd += self.budget_usd
            return False
        return True

    async def run(self) -> Plan:
        started = time.monotonic()
        self._recover()
        running: dict[asyncio.Task, str] = {}
        try:
            while not self.plan.is_finished():
                for step in self.plan.blocked():
                    step.summary = "blocked by failed dependency"
                    self._status(step, "skipped", step.summary)
                for step in self.plan.ready():
                    if step.id in running.values():
                        continue
                    step.attempts += 1
                    step.checkpoint = self.project.checkpoint(f"start {step.id}")
                    step.touched = []
                    self._status(step, "running")
                    running[asyncio.create_task(self._attempt(step.id))] = step.id
                if not running:
                    break
                done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    step_id = running.pop(task)
                    try:
                        outcome, error = task.result()
                    except Exception as e:  # an attempt must never take the whole run down
                        outcome, error = StepOutcome(), f"{type(e).__name__}: {e}"
                    await self._settle(step_id, outcome, error)
                if await self._over_budget(started):
                    break
        finally:
            for task in running:
                task.cancel()  # their steps stay "running" on disk; the next run() restores and retries them
        return self.plan
