"""
GA Trainer Agent — runs genetic algorithm optimization with TUI progress callback.

User can override the model type recommended by the AI.
"""

import asyncio
from typing import Dict, Any, Callable, Awaitable, Optional
from ..engine.genetic import GeneticModelOptimizer
from ..llm.base import LLMProvider


class GATrainerAgent:
    """Wraps GeneticModelOptimizer with async TUI integration."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def run(
        self,
        df,
        target_column: str,
        profile: Dict[str, Any],
        model_choice: str,
        log: Callable[[str], Awaitable[None]],
        get_user_input: Callable[[str], Awaitable[str]],
        on_ga_progress: Optional[Callable[[Dict], Awaitable[None]]] = None,
        pop_size: int = 10,
        generations: int = 5,
    ) -> Dict[str, Any]:
        """
        Args:
            model_choice: AI-recommended model (user can override)
            get_user_input: Human-in-the-loop input function
            on_ga_progress: Callback for per-generation progress updates
        """
        await log(f"[bold green]● GA Trainer[/bold green]  Preparing genetic optimization...")

        # Let user override model choice
        all_models = [
            "RandomForest", "LogisticRegression", "LinearRegression",
            "SVM", "DecisionTree", "KNN", "GradientBoosting",
        ]
        model_list = ", ".join(all_models)
        prompt = (
            f"  AI recommends: [bold cyan]{model_choice}[/bold cyan]\n"
            f"  Available models: {model_list}\n"
            f"  Enter model name to override (or press Enter to accept):"
        )
        await log(prompt)
        user_model = await get_user_input(f"Model override (default: {model_choice})")

        if user_model.strip():
            user_model_clean = user_model.strip().lower()
            
            # Shorthand mapping
            shorthands = {
                "rf": "RandomForest",
                "lr": "LogisticRegression",
                "logreg": "LogisticRegression",
                "linreg": "LinearRegression",
                "svm": "SVM",
                "svc": "SVM",
                "svr": "SVM",
                "dt": "DecisionTree",
                "knn": "KNN",
                "gb": "GradientBoosting",
                "gbc": "GradientBoosting"
            }
            
            # Check shorthand first
            if user_model_clean in shorthands:
                model_choice = shorthands[user_model_clean]
                await log(f"  [yellow]Mapped '{user_model_clean}' → {model_choice}[/yellow]")
            else:
                matched = False
                for vm in all_models:
                    if user_model_clean == vm.lower():
                        model_choice = vm
                        matched = True
                        break
                if not matched:
                    # Try partial match
                    for vm in all_models:
                        if user_model_clean in vm.lower():
                            model_choice = vm
                            matched = True
                            await log(f"  [yellow]Matched '{user_model_clean}' → {vm}[/yellow]")
                            break
                if not matched:
                    await log(f"  [yellow]Unknown model '{user_model_clean}', using {model_choice}[/yellow]")

        await log(f"\n  Model: [bold green]{model_choice}[/bold green] | Pop: {pop_size} | Gen: {generations}")

        # Progress callback bridge (sync -> async)
        loop = asyncio.get_event_loop()
        progress_queue = asyncio.Queue()

        def sync_callback(gen_summary):
            """Called from the GA worker thread."""
            asyncio.run_coroutine_threadsafe(progress_queue.put(gen_summary), loop)

        # Run GA in a thread
        optimizer = GeneticModelOptimizer(
            df=df,
            target_column=target_column,
            profile=profile,
            pop_size=pop_size,
            generations=generations,
            model_choice=model_choice,
            gen_callback=sync_callback,
        )

        # Start GA in background thread
        ori_score = profile.get("overfitting_risk_index", {}).get("score", 0.0)
        await log(f"  [dim]ORI Risk Index: {ori_score:.4f} (Beta-scaled penalties active)[/dim]")
        
        ga_task = asyncio.create_task(asyncio.to_thread(optimizer.evolve))

        # Process progress updates while GA runs
        ga_history = []
        while not ga_task.done():
            try:
                gen_summary = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                ga_history.append(gen_summary)
                gen = gen_summary["gen"]
                best = gen_summary["best_fitness"]
                avg = gen_summary["avg_fitness"]
                await log(f"  Gen {gen:3d}  best={best:.4f}  avg={avg:.4f}")

                if on_ga_progress:
                    await on_ga_progress(gen_summary)
            except asyncio.TimeoutError:
                continue

        # Drain remaining items from queue
        while not progress_queue.empty():
            gen_summary = await progress_queue.get()
            ga_history.append(gen_summary)
            if on_ga_progress:
                await on_ga_progress(gen_summary)

        history, best_genome = await ga_task

        # Output explicit full tracking text-tabulated trace for Gen improvement graph
        await log("\n  [bold magenta]Generational PROGRESS GRAPH[/bold magenta]")
        await log("  [color_dim]Gen[/color_dim] | [color_dim]Best Fit[/color_dim] | [color_dim]Avg Fit[/color_dim]")
        await log("  " + "─" * 25)
        for gen_data in ga_history:
            g = gen_data["gen"]
            b = gen_data["best_fitness"]
            a = gen_data["avg_fitness"]
            await log(f"  {g:3d} |   {b:5.4f} |   {a:5.4f}")
        await log("  " + "─" * 25)

        await log(f"\n  [bold green]✓ GA Complete![/bold green]")
        await log(f"  Best Fitness: [bold]{best_genome.fitness:.4f}[/bold]")
        await log(f"  Best Model: {best_genome.model_choice}")
        await log(f"  Best Params: {best_genome.hparams}")

        # Expose the population for ensemble stacking. It is already best-first and rung-aware; a re-sort by
        # raw fitness would let a screening-rung score outrank one scored on the full ladder.
        top_n = optimizer.population[:5]
        await log(f"  Top-{len(top_n)} genomes ready for ensemble stacking")

        return {
            "ga_history": ga_history or history,
            "best_params": best_genome.hparams,
            "best_cv_score": best_genome.fitness,
            "model_type": best_genome.model_choice,
            "best_metrics": best_genome.metrics,
            "top_genomes": top_n,
        }
