"""
Tests for IndusPredict AI Digital Twin Simulation Service (Phase 5).
Verifies virtual profile generation, what-if parametric stress perturbation,
physics degradation calculations, coupled ML inference, and scenario persistence.
"""

import pytest
from induspredict.db.models import SimulationScenario
from induspredict.services.digital_twin_service import DigitalTwinService


def test_virtual_machine_profile(test_engine):
    """Verify digital twin profile assembly from machine registry and component wear."""
    profile = DigitalTwinService.get_machine_virtual_profile("CMP-01")

    assert profile["machine_code"] == "CMP-01"
    assert "baseline_health" in profile
    assert 0.0 <= profile["baseline_health"] <= 100.0
    assert "baseline_telemetry" in profile
    assert profile["baseline_telemetry"]["air_temperature"] == 25.0
    assert len(profile["components"]) > 0
    assert "SCIENTIFIC NOTICE" in profile["disclaimer"]


def test_what_if_simulation_thermal_overload(test_engine):
    """Verify coupled ML and physics response to severe thermal overload condition."""
    sim_params = {
        "air_temperature": 35.0,
        "process_temperature": 50.0,
        "rotational_speed": 1350.0,
        "torque": 45.0,
        "tool_wear": 50.0,
        "vibration": 2.5,
        "pressure": 7.0,
    }

    res = DigitalTwinService.simulate_what_if("CMP-01", sim_params, model_name="Random Forest")

    assert res["is_simulated"] is True
    assert res["simulation_tag"] == "SYNTHETIC_PARAMETRIC_WHAT_IF"
    assert "baseline" in res
    assert "simulated" in res
    assert res["simulated"]["delta_t_c"] == 15.0
    assert res["simulated"]["mechanical_power_kw"] > 0
    # Health should degrade under stress
    assert res["comparison"]["estimated_health_delta_pct"] <= 0.0


def test_what_if_simulation_heavy_overstrain(test_engine):
    """Verify coupled ML failure risk and failure mode diagnosis under heavy mechanical overstrain."""
    sim_params = {
        "air_temperature": 25.0,
        "process_temperature": 35.0,
        "rotational_speed": 1300.0,
        "torque": 75.0,
        "tool_wear": 200.0,
        "vibration": 4.5,
        "pressure": 6.8,
    }

    res = DigitalTwinService.simulate_what_if("CMP-01", sim_params, model_name="Random Forest")

    # High torque and wear should trigger risk elevation and Overstrain/Power diagnostic
    assert res["simulated"]["failure_probability"] > res["baseline"]["failure_probability"]
    assert res["comparison"]["predicted_risk_delta_pct"] > 0
    assert "Overstrain" in res["simulated"]["likely_failure_mode"] or "Power" in res["simulated"]["likely_failure_mode"]


def test_scenario_persistence_lifecycle(test_engine, db_session):
    """Verify saving, retrieving, and deleting digital twin simulation scenarios in SQLite."""
    initial_count = db_session.query(SimulationScenario).count()

    baseline_state = {"health": 92.0, "risk": 0.01}
    sim_params = {"torque": 70.0, "vibration": 3.8}

    saved = DigitalTwinService.save_scenario(
        machine_code="CMP-01",
        name="Test Overstrain Scenario",
        baseline_state=baseline_state,
        simulated_parameters=sim_params,
        predicted_risk_delta=45.2,
        estimated_health_delta=-18.5,
        notes="Automated test scenario",
    )

    assert saved["id"] is not None
    assert saved["name"] == "Test Overstrain Scenario"

    # Verify query
    scenarios = DigitalTwinService.get_saved_scenarios("CMP-01")
    assert any(s["id"] == saved["id"] for s in scenarios)

    # Verify deletion
    deleted = DigitalTwinService.delete_scenario(saved["id"])
    assert deleted is True

    new_count = db_session.query(SimulationScenario).count()
    assert new_count == initial_count
