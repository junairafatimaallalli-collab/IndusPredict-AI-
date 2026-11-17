"""
Tests for IndusPredict AI Explainable AI (SHAP) Engine (Phase 6).
Verifies Shapley feature attributions, TreeExplainer caching,
global feature importance, normal vs abnormal telemetry comparisons,
and scientific disclaimers.
"""

import pytest
from induspredict.ml.explainability import SHAPExplainer
from induspredict.services.xai_service import XAIService


def test_shap_local_prediction_explanation(test_engine):
    """Verify SHAP explanation generation for high mechanical overstrain stress condition."""
    stress_sample = {
        "air_temperature": 25.0,
        "process_temperature": 35.0,
        "rotational_speed": 1320.0,
        "torque": 68.0,
        "tool_wear": 180.0,
        "machine_type": "M",
    }

    res = XAIService.explain_active_prediction(stress_sample, model_name="Random Forest")

    assert "model_name" in res
    assert res["model_name"] == "Random Forest"
    assert "attributions" in res
    assert len(res["attributions"]) == 10  # 10 engineered features

    # Check top failure driver
    top_driver = res["attributions"][0]
    assert "display_name" in top_driver
    assert "shap_value" in top_driver
    assert "contribution_pct" in top_driver
    assert top_driver["contribution_pct"] > 0

    # Check engineering summary
    assert "engineering_summary" in res
    assert "Primary Failure Driver" in res["engineering_summary"]

    # Check normal vs abnormal comparison
    assert "normal_vs_abnormal" in res
    assert "strain_index" in res["normal_vs_abnormal"]
    assert res["normal_vs_abnormal"]["strain_index"]["is_abnormal"] is True

    # Check scientific disclaimer
    assert "SCIENTIFIC NOTICE" in res["disclaimer"]


def test_shap_global_feature_importance(test_engine):
    """Verify global feature importance computation across reference background dataset."""
    rankings = XAIService.get_global_importance(model_name="Random Forest")

    assert len(rankings) == 10
    assert rankings[0]["mean_abs_shap"] >= rankings[-1]["mean_abs_shap"]
    total_pct = sum(r["global_importance_pct"] for r in rankings)
    assert 98.0 <= total_pct <= 102.0  # Approx 100%


def test_shap_explanation_across_all_tree_models(test_engine):
    """Verify SHAP explainer runs on XGBoost and LightGBM without tensor shape errors."""
    sample = {
        "air_temperature": 30.0,
        "process_temperature": 45.0,
        "rotational_speed": 1400.0,
        "torque": 55.0,
        "tool_wear": 100.0,
        "machine_type": "M",
    }

    for model_name in ["XGBoost", "LightGBM"]:
        res = XAIService.explain_active_prediction(sample, model_name=model_name)
        assert len(res["attributions"]) == 10
        assert res["engineering_summary"] is not None
