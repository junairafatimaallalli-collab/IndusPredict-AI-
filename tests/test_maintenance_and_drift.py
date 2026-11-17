"""
Tests for IndusPredict AI Adaptive Maintenance Planner and Concept Drift Engine (Phase 7).
Verifies ADWIN, Page-Hinkley, KS-test, work order creation, approval, completion,
maintenance history archival, and AI suggestions.
"""

import numpy as np
import pytest
from induspredict.db.models import DriftEvent, MaintenanceHistory, WorkOrder
from induspredict.ml.drift_detector import ADWINDetector, PageHinkleyDetector, compute_kolmogorov_smirnov
from induspredict.services.maintenance_service import MaintenanceService


def test_adwin_drift_detector():
    """Verify ADWIN algorithm detects distribution shift between stationary and shifted regimes."""
    adwin = ADWINDetector(delta=0.005)

    # 1. Stationary regime: mean = 10.0, variance = 0.5
    for _ in range(80):
        val = float(np.random.normal(10.0, 0.5))
        adwin.add_element(val)

    assert adwin.window_size > 30

    # 2. Injected abrupt mean shift: mean jumps to 25.0
    detected = False
    for _ in range(60):
        val = float(np.random.normal(25.0, 0.5))
        if adwin.add_element(val):
            detected = True
            break

    assert detected is True, "ADWIN must detect significant distribution mean shift"


def test_page_hinkley_drift_detector():
    """Verify Page-Hinkley sequential change detection triggers when signal exceeds threshold."""
    ph = PageHinkleyDetector(delta=0.1, threshold=20.0)

    # 1. Baseline regime
    for _ in range(50):
        ph.add_element(float(np.random.normal(0.0, 0.2)))

    # 2. Sudden continuous upward bias
    detected = False
    for _ in range(50):
        val = float(np.random.normal(5.0, 0.2))
        if ph.add_element(val):
            detected = True
            break

    assert detected is True, "Page-Hinkley must flag persistent positive bias"


def test_kolmogorov_smirnov_distribution_shift():
    """Verify KS-test distinguishes identical vs shifted distributions."""
    rng = np.random.RandomState(42)
    dist_a = rng.normal(25.0, 2.0, size=200)
    dist_b = rng.normal(25.0, 2.0, size=200)
    dist_shifted = rng.normal(38.0, 2.0, size=200)

    # Identical
    _, p_same, is_drift_same = compute_kolmogorov_smirnov(dist_a, dist_b, alpha=0.01)
    assert is_drift_same is False
    assert p_same > 0.05

    # Shifted
    _, p_diff, is_drift_diff = compute_kolmogorov_smirnov(dist_a, dist_shifted, alpha=0.01)
    assert is_drift_diff is True
    assert p_diff < 0.001


def test_work_order_lifecycle_and_history(test_engine, db_session):
    """Verify work order creation, approval, completion, and archival into MaintenanceHistory."""
    initial_wo_count = db_session.query(WorkOrder).count()
    initial_hist_count = db_session.query(MaintenanceHistory).count()

    # 1. Create Work Order
    wo = MaintenanceService.create_work_order(
        machine_code="CMP-01",
        title="Automated Test Motor Overhaul",
        description="Verify bearing clearances and stator insulation",
        priority="High",
        estimated_downtime_hours=3.5,
    )
    assert wo["id"] is not None
    assert wo["status"] == "Pending"

    # 2. Approve
    approved = MaintenanceService.approve_work_order(wo["id"])
    assert approved is True

    # 3. Complete and Archive
    completed = MaintenanceService.complete_work_order(
        work_order_id=wo["id"],
        actual_downtime_hours=3.2,
        action_taken="Replaced front bearings and re-greased",
        parts_replaced="SKF 6205-2RS1",
        technician_name="Senior Technician",
    )
    assert completed is True

    # Verify counts
    assert db_session.query(WorkOrder).count() == initial_wo_count + 1
    assert db_session.query(MaintenanceHistory).count() == initial_hist_count + 1


def test_ai_adaptive_maintenance_suggestions(test_engine):
    """Verify AI maintenance generator suggests tasks based on machine conditions."""
    suggestions = MaintenanceService.generate_adaptive_suggestions()
    assert isinstance(suggestions, list)
    # Registry has 5 machines with various hours and predictions
    if suggestions:
        s0 = suggestions[0]
        assert "machine_code" in s0
        assert "title" in s0
        assert "priority" in s0
        assert s0["suggested_downtime_hours"] > 0


def test_drift_event_recording(test_engine, db_session):
    """Verify recording and reviewing drift events."""
    initial_count = db_session.query(DriftEvent).count()

    event = MaintenanceService.record_drift_event(
        model_name="Random Forest",
        sensor_code="CMP-01-VIB",
        detector="ADWIN",
        statistic_value=0.1425,
        threshold=0.0800,
        recommendation="Vibration sensor distribution shifted. Review model calibration.",
    )
    assert event["id"] is not None
    assert event["status"] == "detected"

    # Review event
    reviewed = MaintenanceService.review_drift_event(event["id"], "reviewed")
    assert reviewed is True

    assert db_session.query(DriftEvent).count() == initial_count + 1
