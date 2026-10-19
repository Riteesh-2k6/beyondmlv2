import pytest
import json
import pandas as pd
from beyondml.agents.orchestrator import OrchestratorAgent
from beyondml.agents.eda_agent import EDAAgent
from beyondml.agents.outlier_agent import OutlierAgent
from beyondml.agents.feature_agent import FeatureAgent
from beyondml.agents.ga_trainer import GATrainerAgent
from beyondml.agents.evaluator_agent import EvaluatorAgent

@pytest.mark.asyncio
async def test_full_supervised_pipeline_integration(classification_df, mock_llm):
    # 1. Setup Mock Orchestrator Response
    mock_llm.responses["default"] = json.dumps({
        "path": "supervised",
        "reasoning": "Standard classification task.",
        "suggested_target": "target",
        "confidence": "high",
        "task_type": "classification",
        "model_recommendations": ["RandomForest"]
    })
    
    async def mock_log(msg): pass
    
    # 2. Run Orchestrator
    # Prepare summary and target_info as expected by Orchestrator
    from beyondml.engine.profiler import DatasetProfiler, TargetIdentifier
    profiler = DatasetProfiler(classification_df)
    profile = profiler.run() # Full profile not strictly needed but good for summary
    
    target_id = TargetIdentifier(classification_df)
    target_info = target_id.identify()
    
    # Simple summary generator logic (like in tui_app or cli)
    df_summary = f"Rows: {len(classification_df)}, Cols: {len(classification_df.columns)}"
    
    orchestrator = OrchestratorAgent(mock_llm)
    decision = await orchestrator.run(
        df_summary=df_summary,
        description="A synthetic classification dataset.",
        user_target=None,
        user_path_choice="auto",
        log=mock_log
    )
    
    assert decision["path"] == "supervised"
    target = decision["suggested_target"]
    
    # 3. Simulate Pipeline (like TUI run_pipeline but headless)
    # EDA
    eda_agent = EDAAgent(mock_llm)
    # Mock EDA response
    mock_llm.responses["default"] = json.dumps({
        "insights": [{"finding": "Looks good", "severity": "low"}],
        "suggested_target": "target"
    })
    eda_result = await eda_agent.run(
        classification_df, 
        profile=profile, 
        target_info=target_info, 
        description="A synthetic classification dataset.",
        log=mock_log
    )
    insights = eda_result.get("eda_insights", [])
    
    # Feature Engineering
    feat_agent = FeatureAgent(mock_llm)
    # Mock Feature response
    mock_llm.responses["default"] = json.dumps({
        "reasoning": "Adding interactions.",
        "features": [{"name": "age_income", "expr": "df['age'] * df['income']", "rationale": "Better signal"}]
    })
    
    # Prepare dummy profile for FeatureAgent
    from beyondml.engine.profiler import DatasetProfiler
    profiler = DatasetProfiler(classification_df, target_column=target)
    profile = profiler.run()
    
    feat_result = await feat_agent.run(classification_df, profile, insights, log=mock_log)
    df_transformed = feat_result["df"]
    assert "age_income" in df_transformed.columns
    
    # GA Trainer
    ga_agent = GATrainerAgent(mock_llm)
    
    async def mock_get_input(prompt): return "" # Accept default
    
    # Small pop/gen for speed
    ga_result = await ga_agent.run(
        df=df_transformed,
        target_column=target,
        profile=profile,
        model_choice="RandomForest",
        log=mock_log,
        get_user_input=mock_get_input,
        pop_size=2,
        generations=1
    )
    assert "best_cv_score" in ga_result
    
    # Evaluator
    eval_agent = EvaluatorAgent(mock_llm)
    eval_result = await eval_agent.run(
        df=df_transformed,
        target_column=target,
        profile=profile,
        best_params=ga_result["best_params"],
        model_type=ga_result["model_type"],
        problem_type="classification",
        log=mock_log
    )
    assert "test_score" in eval_result
    assert eval_result["test_score"] >= 0


@pytest.mark.asyncio
async def test_deep_learning_pipeline_integration(classification_df, mock_llm):
    # 1. Setup Mock Orchestrator Response to choose deep_learning
    mock_llm.responses["default"] = json.dumps({
        "path": "deep_learning",
        "reasoning": "Complex data, choosing Neural Network.",
        "suggested_target": "target",
        "confidence": "high",
        "task_type": "classification",
        "model_recommendations": ["SimpleMLP"]
    })
    
    async def mock_log(msg): pass
    
    from beyondml.engine.profiler import DatasetProfiler, TargetIdentifier
    profiler = DatasetProfiler(classification_df)
    profile = profiler.run()
    target_id = TargetIdentifier(classification_df)
    target_info = target_id.identify()
    
    orchestrator = OrchestratorAgent(mock_llm)
    decision = await orchestrator.run(
        df_summary="Rows: 200, Cols: 6",
        description="Deep learning test.",
        user_target=None,
        user_path_choice="auto",
        log=mock_log
    )
    
    assert decision["path"] == "deep_learning"
    target = decision["suggested_target"]
    
    # 2. Run DL Agent
    from beyondml.agents.dl_agent import DeepLearningAgent
    dl_agent = DeepLearningAgent(mock_llm)
    
    dl_result = await dl_agent.run(
        df=classification_df,
        target_column=target,
        problem_type="classification",
        log=mock_log,
        epochs=1
    )
    
    assert dl_result["model_type"] == "SimpleMLP"
    assert dl_result["test_score"] >= 0
