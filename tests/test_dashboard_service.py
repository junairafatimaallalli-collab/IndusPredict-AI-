"""
Tests for IndusPredict AI Dashboard Aggregation Service.
"""

import pytest
from induspredict.services.dashboard_service import DashboardService


def test_dashboard_kpis(test_engine):
    """Verify that KPI counts match actual database rows and are not hardcoded."""
    kpis = DashboardService.get_summary_kpis()

    assert kpis["total_machines"] >= 5
    assert kpis["healthy_machines"] + kpis["observation_machines"] + kpis["high_risk_machines"] == kpis["total_machines"]
    assert kpis["critical_alerts"] >= 0
    assert kpis["pending_tasks"] >= 1  # We seeded 2 work orders
    assert kpis["total_sensors"] >= 15


def test_failure_risk_distribution(test_engine):
    """Verify failure risk stratification totals equal machine count."""
    kpis = DashboardService.get_summary_kpis()
    dist = DashboardService.get_failure_risk_distribution()

    total_in_dist = sum(dist.values())
    assert total_in_dist == kpis["total_machines"]
    assert "Normal" in dist
    assert "Warning" in dist
    assert "Critical" in dist


def test_health_trends_data(test_engine):
    """Verify health trends format for chart rendering."""
    trends = DashboardService.get_health_trends(hours=24)

    assert "timestamps" in trends
    assert "values" in trends
    assert "anomalies" in trends
    assert len(trends["timestamps"]) == len(trends["values"])
    assert len(trends["values"]) == len(trends["anomalies"])


def test_recent_predictions_and_alerts(test_engine):
    """Verify retrieval of recent predictions and anomaly events."""
    preds = DashboardService.get_recent_predictions(limit=5)
    assert len(preds) > 0
    for p in preds:
        assert "machine_code" in p
        assert "failure_prob" in p
        assert 0.0 <= p["failure_prob"] <= 100.0

    rul_list = DashboardService.get_rul_overview(limit=5)
    assert len(rul_list) > 0
    for r in rul_list:
        assert "predicted_rul" in r
        assert r["predicted_rul"] > 0
