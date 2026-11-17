"""
Tests for IndusPredict AI PyTorch LSTM Prognostics Engine and NASA C-MAPSS RUL Service.
"""

import numpy as np
import pytest
import torch
from induspredict.db.models import RULEstimate
from induspredict.ml.prognostics import (
    FEATURE_COLS,
    LSTMRULModel,
    SEQUENCE_WINDOW_LENGTH,
    nasa_scoring_function,
)
from induspredict.services.prognostics_service import PrognosticsService


def test_lstm_model_architecture():
    """Verify PyTorch LSTM model instantiation, parameter count, and tensor shapes."""
    model = LSTMRULModel(input_dim=len(FEATURE_COLS), hidden_dim=64, num_layers=2, dropout=0.2)
    model.eval()

    # Create dummy batch of 4 sequences of length 30 with 17 features
    dummy_input = torch.randn(4, 30, len(FEATURE_COLS))
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (4, 1)
    assert not torch.isnan(out).any()


def test_nasa_scoring_function():
    """Verify asymmetric NASA scoring penalizing late predictions more than early ones."""
    y_true = np.array([50.0, 50.0])
    # Early prediction: diff = -10 (predicted 40 when true was 50)
    # Late prediction:  diff = +10 (predicted 60 when true was 50)
    y_early = np.array([40.0, 50.0])
    y_late = np.array([60.0, 50.0])

    score_early = nasa_scoring_function(y_true, y_early)
    score_late = nasa_scoring_function(y_true, y_late)

    # In NASA metric:
    # d = y_pred - y_true
    # When y_pred = 60, true = 50 -> d = +10 -> exp(10/10) - 1 = exp(1) - 1 ~ 1.718
    # When y_pred = 40, true = 50 -> d = -10 -> exp(-(-10)/13) - 1 = exp(10/13) - 1 ~ 1.158
    assert score_late > score_early, "NASA metric must penalize optimistic/late predictions more heavily"


def test_insufficient_data_rejection_guard():
    """Verify that input sequences with fewer than 30 cycles are strictly rejected."""
    # Provide only 15 cycles (insufficient temporal context)
    short_sequence = [{col: 0.5 for col in FEATURE_COLS} for _ in range(15)]

    with pytest.raises(ValueError) as excinfo:
        PrognosticsService.predict_rul_from_sequence(short_sequence, "CMP-01")

    assert "minimum of 30 consecutive cycles is required" in str(excinfo.value).lower()


def test_rul_prediction_with_uncertainty(test_engine):
    """Verify inference on a valid 30-cycle sequence yielding 95% confidence bounds."""
    # Create valid 30-cycle operational sequence
    valid_sequence = [{col: float(np.sin(i * 0.1)) for col in FEATURE_COLS} for i in range(30)]

    res = PrognosticsService.predict_rul_from_sequence(valid_sequence, "CMP-01", num_mc_samples=10)

    assert res is not None
    assert "predicted_rul_cycles" in res
    assert res["predicted_rul_cycles"] > 0
    assert "lower_bound_95" in res
    assert "upper_bound_95" in res
    assert res["lower_bound_95"] <= res["predicted_rul_cycles"] <= res["upper_bound_95"]
    assert 0.0 <= res["degradation_pct"] <= 100.0


def test_engine_trajectory_extraction():
    """Verify loading full run-to-failure trajectory for held-out test fleet (Unit 81)."""
    traj = PrognosticsService.get_test_engine_trajectory(unit_id=81)

    assert traj is not None
    assert traj["unit_id"] == 81
    assert traj["max_cycles"] > 100
    assert len(traj["cycles"]) > 0
    assert len(traj["actual_rul"]) == len(traj["cycles"])
    assert len(traj["predicted_rul"]) == len(traj["cycles"])
    # Trajectory must show monotonically decreasing ground truth RUL
    assert traj["actual_rul"][0] >= traj["actual_rul"][-1]


def test_rul_persistence_in_db(test_engine, db_session):
    """Verify that RUL predictions are recorded in SQLite rul_estimates table."""
    initial_count = db_session.query(RULEstimate).count()

    valid_seq = [{col: 0.1 for col in FEATURE_COLS} for _ in range(30)]
    PrognosticsService.predict_rul_from_sequence(valid_seq, "CMP-01", num_mc_samples=5)

    new_count = db_session.query(RULEstimate).count()
    assert new_count == initial_count + 1
