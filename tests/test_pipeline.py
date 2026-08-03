"""
Tests for src/pipeline.py — the orchestrator, not the math.

Each stage's correctness is already covered by its own test file. This
file tests one thing: that pipeline.run() calls every stage in the right
order. Real network calls and file writes are mocked out — an actual
end-to-end run belongs in a manual or scheduled check, not a fast unit
test suite.
"""

from unittest.mock import MagicMock, patch

import pandas as pd


def test_run_calls_every_stage_in_order():
    call_order = []

    def track(name):
        def _fn(*args, **kwargs):
            call_order.append(name)
        return _fn

    fake_summary = pd.DataFrame([{
        "gbm_cvar_95": 0.3,
        "heston_cvar_95": 0.4,
        "hedge_mean_agreement_pct_diff": 1.5,
    }])

    with patch("src.pipeline.mlflow") as mock_mlflow, \
         patch("src.pipeline.extractor.extract", side_effect=track("extract")), \
         patch("src.pipeline.processor.process", side_effect=track("process")), \
         patch("src.pipeline.estimation.estimate", side_effect=track("estimate")), \
         patch("src.pipeline.optimization.optimize", side_effect=track("optimize")), \
         patch("src.pipeline.simulation.simulate", side_effect=track("simulate")), \
         patch("src.pipeline.heston.run", side_effect=track("heston")), \
         patch("src.pipeline.output.finalize_summary", return_value="fake_path.parquet"), \
         patch("src.pipeline.pd.read_parquet", return_value=fake_summary), \
         patch("src.pipeline._sharpe_of", return_value=1.2):

        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from src.pipeline import run
        run()

    assert call_order == [
        "extract", "process", "estimate", "optimize", "simulate", "heston",
    ]


def test_run_logs_to_mlflow():
    """Confirms the orchestrator actually calls mlflow's tracking functions
    — not just that they're imported, but that run() invokes them."""
    fake_summary = pd.DataFrame([{
        "gbm_cvar_95": 0.3,
        "heston_cvar_95": 0.4,
        "hedge_mean_agreement_pct_diff": 1.5,
    }])

    with patch("src.pipeline.mlflow") as mock_mlflow, \
         patch("src.pipeline.extractor.extract"), \
         patch("src.pipeline.processor.process"), \
         patch("src.pipeline.estimation.estimate"), \
         patch("src.pipeline.optimization.optimize"), \
         patch("src.pipeline.simulation.simulate"), \
         patch("src.pipeline.heston.run"), \
         patch("src.pipeline.output.finalize_summary", return_value="fake_path.parquet"), \
         patch("src.pipeline.pd.read_parquet", return_value=fake_summary), \
         patch("src.pipeline._sharpe_of", return_value=1.2):

        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        from src.pipeline import run
        run()

    mock_mlflow.set_tracking_uri.assert_called_once()
    mock_mlflow.log_params.assert_called_once()
    mock_mlflow.log_metrics.assert_called_once()
    mock_mlflow.log_artifact.assert_called_once()
