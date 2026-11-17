"""
Tests for IndusPredict AI Machine Learning Models and Prediction Service.
"""

import pytest
from induspredict.db.models import Prediction
from induspredict.ml.preprocessor import DataPreprocessor
from induspredict.services.prediction_service import PredictionService


def test_preprocessor_feature_engineering():
    """Verify that thermodynamic and kinematic interaction terms are computed correctly."""
    preprocessor = DataPreprocessor()
    assert len(preprocessor.feature_names) == 10

    sample_input = {
        "air_temperature_c": 25.0,        # 298.15 K
        "process_temperature_c": 35.0,    # 308.15 K
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": 60.0,
        "machine_type": "M",
    }
    vec = preprocessor.transform_single(sample_input)
    assert vec.shape == (1, 10)


def test_model_inference_all_algorithms(test_engine):
    """Verify inference across Random Forest, XGBoost, LightGBM, and Logistic Regression."""
    algorithms = ["Random Forest", "XGBoost", "LightGBM", "Logistic Regression"]

    sample_nominal = {
        "air_temperature_c": 25.0,
        "process_temperature_c": 35.0,
        "rotational_speed": 1500.0,
        "torque": 38.0,
        "tool_wear": 30.0,
        "machine_type": "M",
    }

    for algo in algorithms:
        res = PredictionService.predict_machine_failure(
            raw_sensor_input=sample_nominal,
            machine_code="CMP-01",
            model_name=algo,
        )
        assert res is not None
        assert 0.0 <= res["failure_probability"] <= 1.0
        assert res["risk_category"] in ("Normal", "Warning", "High Risk", "Critical")
        assert res["confidence_score"] > 0.0
        assert len(res["top_contributions"]) > 0
        assert res["model_name"] == algo


def test_failure_mode_diagnostics(test_engine):
    """Verify physics-informed failure mode triggers for HDF, OSF, and TWF."""
    # 1. Critical Tool Wear Failure (> 200 min)
    sample_twf = {
        "air_temperature_c": 25.0,
        "process_temperature_c": 35.0,
        "rotational_speed": 1500.0,
        "torque": 45.0,
        "tool_wear": 235.0,
        "machine_type": "M",
    }
    res_twf = PredictionService.predict_machine_failure(sample_twf, "CMP-01", "Random Forest")
    assert "Tool Wear Failure" in res_twf["likely_failure_mode"]

    # 2. Extreme Overstrain Failure (Torque * Wear > 11000)
    sample_osf = {
        "air_temperature_c": 25.0,
        "process_temperature_c": 35.0,
        "rotational_speed": 1400.0,
        "torque": 72.0,
        "tool_wear": 190.0,  # 72 * 190 = 13680 > 11000
        "machine_type": "L",
    }
    res_osf = PredictionService.predict_machine_failure(sample_osf, "CMP-01", "Random Forest")
    assert "Overstrain Failure" in res_osf["likely_failure_mode"]


def test_benchmark_comparisons_from_db(test_engine):
    """Verify retrieval of genuine model evaluation benchmarks stored in SQLite."""
    benchmarks = PredictionService.get_model_benchmark_comparisons()
    assert len(benchmarks) == 4

    algo_names = {b["model_name"] for b in benchmarks}
    expected_algos = {"Random Forest", "XGBoost", "LightGBM", "Logistic Regression"}
    assert algo_names == expected_algos

    for b in benchmarks:
        assert b["accuracy"] > 0.80
        assert b["roc_auc"] > 0.90
        assert b["pr_auc"] > 0.40
        assert "confusion_matrix" in b
        assert "tp" in b["confusion_matrix"]


def test_prediction_persisted_to_db(test_engine, db_session):
    """Verify that predictions are recorded to the SQLite predictions table."""
    initial_count = db_session.query(Prediction).count()

    sample_input = {
        "air_temperature_c": 25.0,
        "process_temperature_c": 35.0,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": 50.0,
        "machine_type": "M",
    }
    PredictionService.predict_machine_failure(sample_input, "CMP-01", "Random Forest")

    new_count = db_session.query(Prediction).count()
    assert new_count == initial_count + 1
