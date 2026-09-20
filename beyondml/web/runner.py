"""
Pipeline runner — bridges BeyondML agents to a WebSocket connection.

PipelineRunner owns the async event loop side of the pipeline:
  - self.log(msg)        → emits {type:"log"} WS event
  - self.stat(key, val)  → emits {type:"stat"} WS event
  - self.tree(node, st)  → emits {type:"tree_update"} WS event
  - self.chart(title, fig) → emits {type:"chart"} WS event
  - self.get_user_input(prompt) → sends {type:"input_request"}, awaits user_response
  - self.run_pipeline(config, df) → full agent orchestration
"""
from __future__ import annotations

import asyncio
import json
import os
import traceback as tb
from datetime import datetime
from typing import Any, Callable, Awaitable, Dict, Optional

import numpy as np
import pandas as pd

from beyondml.web.markup import parse_rich_markup


class PipelineRunner:
    """Runs the BeyondML agent pipeline, emitting structured WebSocket events."""

    def __init__(
        self,
        send: Callable[[Dict], Awaitable[None]],
        recv: Callable[[], Awaitable[str]],
    ):
        self._send = send
        self._recv = recv
        self._recv_queue: asyncio.Queue[str] = asyncio.Queue()
        self._recv_task: Optional[asyncio.Task] = None

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    async def _pump_recv(self):
        """Background task: continuously read from the WebSocket and queue messages."""
        try:
            while True:
                raw = await self._recv()
                await self._recv_queue.put(raw)
        except Exception:
            pass

    async def _start_recv_pump(self):
        if self._recv_task is None or self._recv_task.done():
            self._recv_task = asyncio.create_task(self._pump_recv())

    # ─────────────────────────────────────────────
    # Outbound events
    # ─────────────────────────────────────────────

    async def log(self, msg: str):
        """Emit a log event with Rich-markup parsed spans."""
        spans = parse_rich_markup(msg)
        await self._send({
            "type": "log",
            "payload": {
                "spans": [s for s in spans],
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            },
        })

    async def stat(self, key: str, value: str):
        """Emit a dataset/run stat update."""
        await self._send({"type": "stat", "payload": {"key": key, "value": value}})

    async def tree(self, node: str, status: str, children: list[str] | None = None):
        """Emit a pipeline tree node status update."""
        payload: Dict[str, Any] = {"node": node, "status": status}
        if children is not None:
            payload["children"] = children
        await self._send({"type": "tree_update", "payload": payload})

    async def chart(self, title: str, fig):
        """Emit an inline Plotly chart event."""
        import json as _json
        raw = _json.loads(fig.to_json())
        # Apply obsidian dark theme
        raw.setdefault("layout", {}).update({
            "paper_bgcolor": "#09090B",
            "plot_bgcolor":  "#18181B",
            "font": {"color": "#E4E4E7", "family": "Geist Mono, monospace"},
            "xaxis": {"gridcolor": "#27272A", "zerolinecolor": "#27272A"},
            "yaxis": {"gridcolor": "#27272A", "zerolinecolor": "#27272A"},
            "margin": {"l": 40, "r": 16, "t": 32, "b": 40},
        })
        await self._send({"type": "chart", "payload": {"title": title, "plotly_json": raw}})

    async def get_user_input(self, prompt: str) -> str:
        """Send an input_request event and block until the client responds."""
        await self._send({"type": "input_request", "payload": {"prompt": prompt}})
        while True:
            raw = await self._recv_queue.get()
            try:
                msg = json.loads(raw)
                if msg.get("type") == "user_response":
                    return msg.get("payload", {}).get("value", "")
                if msg.get("type") == "cancel":
                    return ""
            except (json.JSONDecodeError, AttributeError):
                continue

    async def _emit_error(self, message: str, trace: str = ""):
        await self._send({"type": "error", "payload": {"message": message, "traceback": trace}})

    async def _emit_done(self, eval_result: dict | None = None):
        await self._send({"type": "done", "payload": eval_result or {}})

    # ─────────────────────────────────────────────
    # Pipeline orchestration
    # ─────────────────────────────────────────────

    async def run_pipeline(self, config: dict, df: pd.DataFrame):
        """
        Full agent pipeline — exact port of tui_app.py PipelineScreen.run_pipeline.
        Replaces all TUI widget calls with self.log / self.stat / self.tree / self.chart.
        """
        await self._start_recv_pump()

        try:
            await self.log("[bold green]Pipeline started[/bold green]")
            await self.stat("shape", f"{df.shape[0]} × {df.shape[1]}")

            # ── Configure LLM ──
            llm_choice  = config.get("llm_choice", "ollama")
            path_choice = config.get("path_choice", "autonomous")
            ga_pop      = int(config.get("ga_pop", 10))
            ga_gen      = int(config.get("ga_gen", 5))
            test_size   = float(config.get("test_size", 0.20))
            use_ensemble = bool(config.get("use_ensemble", False))
            description = config.get("description", "")
            user_target = config.get("target_col", "")

            os.environ.setdefault("LLM_PROVIDER", llm_choice)
            if llm_choice == "groq":
                if config.get("groq_api_key"):
                    os.environ["GROQ_API_KEY"] = config["groq_api_key"]
                if config.get("groq_model"):
                    os.environ["GROQ_MODEL"] = config["groq_model"]
                from beyondml.llm.groq_provider import GroqProvider
                llm = GroqProvider()
            else:
                if config.get("ollama_model"):
                    os.environ["OLLAMA_MODEL"] = config["ollama_model"]
                from beyondml.llm.ollama_provider import OllamaProvider
                llm = OllamaProvider()

            # ── STEP 0: Profile ──
            from beyondml.engine.profiler import DatasetProfiler, TargetIdentifier
            from beyondml.engine.tracing import AgentTrace

            trace = AgentTrace()
            profiler = DatasetProfiler(df)
            profile  = profiler.run()
            target_id = TargetIdentifier(df)
            target_info = target_id.identify()
            df_summary = (
                f"Shape: {df.shape}\n"
                f"Columns: {list(df.columns)}\n"
                f"Dtypes: {df.dtypes.to_dict()}\n"
                f"Sample:\n{df.head(3).to_string()}"
            )

            await self.stat("shape", f"{df.shape[0]} × {df.shape[1]}")

            # ── STEP 1: Domain Sanity + Leakage (parallel) ──
            await self.log("[dim]─── Domain Sanity + Drift ────────────────────────[/dim]")
            await self.tree("Domain Sanity (Parallel)", "running")

            from beyondml.agents.sanity_agent import SanityAgent
            from beyondml.agents.drift_agent import DriftAgent

            col_stats = profile.get("numerical_summary", {})
            sanity_agent = SanityAgent(llm)
            sanity_result, _ = await asyncio.gather(
                sanity_agent.run(df_summary, col_stats, self.log),
                asyncio.sleep(0),
            )
            await self.tree("Domain Sanity (Parallel)", "done")

            # ── STEP 2: Imputation ──
            await self.log("[dim]─── Imputation ──────────────────────────────────[/dim]")
            await self.tree("Imputation", "running")

            from beyondml.agents.imputation_agent import ImputationAgent
            imp_agent = ImputationAgent(llm)
            imp_result = await imp_agent.run(df, profile, self.log)
            df = imp_result["df"]
            await self.tree("Imputation", "done")

            # ── STEP 3: Orchestrator ──
            await self.log("[dim]─── Orchestrator ────────────────────────────────[/dim]")
            await self.tree("Orchestrator", "running")

            from beyondml.agents.orchestrator import OrchestratorAgent
            orch = OrchestratorAgent(llm)
            decision = await orch.run(
                df_summary=df_summary,
                description=description,
                target_info=target_info,
                user_path_choice=path_choice,
                log=self.log,
            )
            # Prefer explicit user_target, then LLM suggestion, then auto-identified
            target = (
                user_target
                or decision.get("suggested_target")
                or (target_info.get("candidates", [None])[0] if target_info else None)
            )
            path         = decision.get("path", "supervised")
            model_recs   = decision.get("model_recommendations", ["RandomForest"])
            task_type    = decision.get("task_type", "classification")

            await self.stat("target", str(target or "—"))
            await self.stat("task", task_type)
            await self.tree("Orchestrator", "done")

            # ── STEP 3.5: Leakage check (only if supervised target known) ──
            if target and path in ("supervised", "deep_learning"):
                await self.tree("Data Drift Analysis", "running")
                from beyondml.agents.leakage_agent import LeakageAgent
                num_cols = profile.get("numerical_summary", {})
                corrs: dict = {}
                try:
                    if target in df.columns:
                        corrs = {
                            col: float(df[col].corr(df[target]))
                            for col in df.select_dtypes("number").columns
                            if col != target
                        }
                except Exception:
                    pass

                col_descs = "\n".join(f"{c}: {df[c].dtype}" for c in df.columns)
                leakage_agent = LeakageAgent(llm)
                leakage_result = await leakage_agent.run(
                    target_column=target,
                    column_descriptions=col_descs,
                    correlations=corrs,
                    log=self.log,
                )
                await self.tree("Data Drift Analysis", "done")

            # ── STEP 4: EDA + Outlier (parallel) ──
            await self.log("[dim]─── EDA Agent ───────────────────────────────────[/dim]")
            await self.tree("EDA Agent (Parallel)", "running")
            await self.tree("Outlier Handler (Parallel)", "running")

            from beyondml.agents.eda_agent import EDAAgent
            from beyondml.agents.outlier_agent import OutlierAgent
            from beyondml.engine.profiler import DatasetProfiler

            profiler = DatasetProfiler(df, target_column=target)
            profile  = profiler.run()

            eda_agent = EDAAgent(llm)
            outlier_agent = OutlierAgent(llm)

            outlier_summary = profile.get("outlier_summary", {})
            eda_result, outlier_result = await asyncio.gather(
                eda_agent.run(df, profile=profile, target_info=target_info,
                              description=description, log=self.log),
                outlier_agent.run(df, outlier_summary, profile, self.log,
                                  self.get_user_input),
            )
            df = outlier_result["df"]
            df = df.replace([np.inf, -np.inf], np.nan)

            # Send EDA charts inline
            eda_charts = eda_result.get("charts", [])
            for chart_entry in eda_charts:
                if isinstance(chart_entry, dict) and "fig" in chart_entry:
                    await self.chart(chart_entry.get("title", "EDA Chart"),
                                     chart_entry["fig"])

            confirmed_target = eda_result.get("suggested_target") or target
            task_type        = eda_result.get("task_type", task_type)

            await self.stat("target", str(confirmed_target or "—"))
            await self.stat("task", task_type)
            n_classes = profile.get("target_analysis", {}).get("num_unique", "—")
            await self.stat("classes", str(n_classes))
            await self.stat("split", f"{1-test_size:.0%} / {test_size:.0%}")

            await self.tree("EDA Agent (Parallel)", "done")
            await self.tree("Outlier Handler (Parallel)",
                            "done")

            if path in ("explore", "dimensionality_reduction"):
                await self.log("[bold green]Exploration complete.[/bold green]")
                await self._emit_done()
                return

            # ── ITERATIVE OPTIMIZATION LOOP ──
            max_iterations   = 3
            current_iter     = 1
            best_eval_result = None
            current_insights = eda_result.get("eda_insights", [])
            model_choice     = model_recs[0] if model_recs else "RandomForest"
            current_pop      = ga_pop
            current_gens     = ga_gen

            while current_iter <= max_iterations:
                if current_iter > 1:
                    await self.log(
                        f"[bold magenta]─── OPTIMIZATION LOOP {current_iter}/{max_iterations} ───[/bold magenta]"
                    )

                # ── Feature Engineering ──
                await self.log("[dim]─── Feature Engineer ────────────────────────────[/dim]")
                await self.tree("Feature Engineer", "running")

                profiler = DatasetProfiler(df, target_column=confirmed_target)
                profile  = profiler.run()

                from beyondml.agents.feature_agent import FeatureAgent
                feat_agent   = FeatureAgent(llm)
                feat_result  = await feat_agent.run(df, profile, current_insights, self.log)
                df           = feat_result["df"]
                df           = df.replace([np.inf, -np.inf], np.nan)

                n_applied = len(feat_result.get("features_applied", []))
                await self.stat("new_features", str(n_applied))
                await self.stat("shape", f"{df.shape[0]} × {df.shape[1]}")
                await self.tree("Feature Engineer", "done")

                if path == "unsupervised":
                    await self.tree("GA Trainer (Supervised)", "waiting")
                    from beyondml.engine.unsupervised import UnsupervisedPipeline
                    unsup = UnsupervisedPipeline(df, profile)
                    results = await asyncio.to_thread(unsup.run_clustering)
                    for task_name, metrics in results.items():
                        await self.log(f"[bold cyan]{task_name}[/bold cyan]")
                        for k, v in metrics.items():
                            await self.log(f"  {k}: {v}")
                    await self.log("[bold green]Unsupervised analysis complete.[/bold green]")
                    await self._emit_done()
                    return

                prebuilt_model = None
                if path == "supervised":
                    # ── GA Trainer ──
                    await self.log("[dim]─── GA Trainer ───────────────────────────────[/dim]")
                    await self.tree("GA Trainer (Supervised)", "running")

                    profiler = DatasetProfiler(df, target_column=confirmed_target)
                    try:
                        profile = profiler.run() or {
                            "target_analysis": {"target_type": "classification"},
                            "numerical_summary": {},
                        }
                    except Exception as e:
                        await self.log(f"[yellow]Profiling warning: {e}[/yellow]")
                        profile = {
                            "target_analysis": {"target_type": "classification"},
                            "numerical_summary": {},
                        }
                    if "target_analysis" not in profile or profile["target_analysis"] is None:
                        profile["target_analysis"] = {"target_type": "classification"}

                    fitness_data: list[float] = []

                    async def on_ga_progress(gen_summary):
                        fitness_data.append(gen_summary["best_fitness"] * 100)
                        await self.stat("fitness_latest",
                                        f"{gen_summary['best_fitness']:.4f}")
                        await self._send({
                            "type": "ga_progress",
                            "payload": {"fitness": fitness_data.copy()},
                        })

                    from beyondml.agents.ga_trainer import GATrainerAgent
                    ga_agent   = GATrainerAgent(llm)
                    ga_result  = await ga_agent.run(
                        df=df,
                        target_column=confirmed_target,
                        profile=profile,
                        model_choice=model_choice,
                        log=self.log,
                        get_user_input=self.get_user_input,
                        on_ga_progress=on_ga_progress,
                        pop_size=current_pop,
                        generations=current_gens,
                    )
                    best_params = ga_result["best_params"]
                    model_type  = ga_result["model_type"]
                    await self.tree("GA Trainer (Supervised)", "done")

                    # ── Ensemble (optional) ──
                    top_genomes = ga_result.get("top_genomes", [])
                    if use_ensemble and len(top_genomes) > 1:
                        await self.tree("Ensemble", "running")
                        from beyondml.agents.ensemble_agent import EnsembleAgent
                        ensemble_agent  = EnsembleAgent(llm)
                        ensemble_result = await ensemble_agent.run(
                            df=df,
                            target_column=confirmed_target,
                            profile=profile,
                            top_genomes=top_genomes,
                            problem_type=profile["target_analysis"]["target_type"],
                            log=self.log,
                            strategy="stacking",
                        )
                        ens_score = ensemble_result["test_score"]
                        await self.tree("Ensemble", "done")
                        if ens_score > ga_result["best_cv_score"]:
                            from beyondml.engine.ensemble import EnsembleEngine
                            engine = EnsembleEngine(
                                profile["target_analysis"]["target_type"]
                            )
                            prebuilt_model = engine.build_stacking(top_genomes)
                            best_params = {"strategy": "stacking"}
                            model_type  = ensemble_result["model_type"]

                elif path == "deep_learning":
                    await self.tree("DL Trainer (Neural Net)", "running")
                    from beyondml.agents.dl_agent import DeepLearningAgent
                    dl_agent  = DeepLearningAgent(llm)
                    dl_result = await dl_agent.run(
                        df=df,
                        target_column=confirmed_target,
                        problem_type=profile["target_analysis"]["target_type"],
                        log=self.log,
                        epochs=10,
                    )
                    best_params = dl_result.get("best_params", {})
                    model_type  = dl_result["model_type"]
                    await self.tree("DL Trainer (Neural Net)", "done")

                # ── Evaluator ──
                await self.log("[dim]─── Evaluator ────────────────────────────────[/dim]")
                await self.tree("Evaluator", "running")

                from beyondml.agents.evaluator_agent import EvaluatorAgent
                eval_agent  = EvaluatorAgent(llm)
                eval_result = await eval_agent.run(
                    df=df,
                    target_column=confirmed_target,
                    profile=profile,
                    best_params=best_params,
                    model_type=model_type,
                    problem_type=profile["target_analysis"]["target_type"],
                    log=self.log,
                    prebuilt_model=prebuilt_model,
                    test_size=test_size,
                )

                test_score  = eval_result.get("test_score", 0)
                train_score = eval_result.get("train_score", 0)
                await self.stat("test_score",  f"{test_score:.4f}")
                await self.stat("train_score", f"{train_score:.4f}")
                await self.tree("Evaluator", "done")

                # Send evaluation charts
                eval_charts = eval_result.get("charts", [])
                for chart_entry in eval_charts:
                    if isinstance(chart_entry, dict) and "fig" in chart_entry:
                        await self.chart(chart_entry.get("title", "Eval Chart"),
                                         chart_entry["fig"])

                if eval_result.get("test_score", 0) > (
                    best_eval_result or {}
                ).get("test_score", -1):
                    best_eval_result = eval_result

                # ── Explainability ──
                await self.tree("Explainability", "running")
                from beyondml.agents.explainability_agent import ExplainabilityAgent
                explain_agent = ExplainabilityAgent(llm)
                try:
                    await explain_agent.run(eval_result, self.log)
                except Exception:
                    pass
                await self.tree("Explainability", "done")

                # ── Reflection ──
                await self.log("[dim]─── Reflection Agent ─────────────────────────[/dim]")
                await self.tree("Reflection", "running")

                from beyondml.agents.reflection_agent import ReflectionAgent
                reflect_agent  = ReflectionAgent(llm)
                reflect_result = await reflect_agent.run(
                    eval_result=eval_result,
                    iteration=current_iter,
                    max_iterations=max_iterations,
                    log=self.log,
                )
                await self.tree("Reflection", "done")

                if reflect_result["status"] in ("satisfied", "error") or (
                    current_iter >= max_iterations
                ):
                    break

                # Apply modifications for next iteration
                mods = reflect_result.get("modifications") or {}
                new_feats = mods.get("new_features", [])
                for feat in new_feats:
                    name = feat.get("name", "")
                    expr = feat.get("expr", "")
                    try:
                        import pandas as _pd
                        result = eval(
                            expr, {"__builtins__": {}},
                            {"df": df, "np": np, "pd": _pd},
                        )
                        if isinstance(result, _pd.Series) and len(result) == len(df):
                            df[name] = result
                    except Exception:
                        pass

                drops = mods.get("features_to_drop", [])
                for col in drops:
                    clean_col = col.strip().strip("'\"")
                    if clean_col in df.columns:
                        df = df.drop(columns=[clean_col])

                if mods.get("next_model"):
                    model_choice = mods["next_model"]
                if mods.get("next_ga_generations"):
                    current_gens = mods["next_ga_generations"]
                if mods.get("next_ga_pop_size"):
                    current_pop = mods["next_ga_pop_size"]

                current_iter += 1

            await self.log("[bold green]Pipeline complete.[/bold green]")
            await self._emit_done(best_eval_result)

        except Exception as e:
            trace_str = tb.format_exc()
            await self._emit_error(str(e), trace_str)
        finally:
            if self._recv_task and not self._recv_task.done():
                self._recv_task.cancel()
