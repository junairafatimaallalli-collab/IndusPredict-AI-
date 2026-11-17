"""
Tests for IndusPredict AI Physical Sensor Simulator.
"""

import pytest
from induspredict.iot.simulator import SensorSimulator


def test_simulator_nominal_generation():
    """Verify nominal sensor generation physics and range bounds."""
    sim = SensorSimulator(machine_code="CMP-01", sample_rate_hz=2.0)
    sim.set_mode(SensorSimulator.MODE_NOMINAL)

    frame = sim.generate_step()
    assert frame is not None
    assert frame["machine_code"] == "CMP-01"
    assert frame["source"] == "simulated"
    assert "sensors" in frame

    sensors = frame["sensors"]
    for expected in ["temperature", "vibration", "pressure", "motor_speed", "current", "torque", "operating_hours"]:
        assert expected in sensors
        assert isinstance(sensors[expected]["value"], float)

    # Nominal bounds
    assert 55.0 <= sensors["temperature"]["value"] <= 80.0
    assert 1.5 <= sensors["vibration"]["value"] <= 5.0
    assert 5.0 <= sensors["pressure"]["value"] <= 10.0


def test_thermal_overrun_fault_injection():
    """Verify that thermal runaway increases temperature and triggers alerts."""
    sim = SensorSimulator(machine_code="CMP-01", sample_rate_hz=10.0)
    sim.set_mode(SensorSimulator.MODE_THERMAL_OVERRUN)

    triggered_alerts = []
    sim.alert_triggered.connect(lambda a: triggered_alerts.append(a))

    # Run 60 simulation steps to accumulate heat
    initial_temp = sim.state["temperature"]
    for _ in range(60):
        sim.generate_step()

    final_temp = sim.state["temperature"]
    assert final_temp > initial_temp + 10.0, "Thermal runaway must increase temperature"
    assert len(triggered_alerts) > 0
    assert any(a["severity"] in ("Warning", "Critical") for a in triggered_alerts)


def test_bearing_degradation_fault_injection():
    """Verify that bearing wear causes vibration progression and shock peaks."""
    sim = SensorSimulator(machine_code="MTR-02", sample_rate_hz=5.0)
    sim.set_mode(SensorSimulator.MODE_BEARING_DEGRADATION)

    initial_vib = sim.state["vibration"]
    for _ in range(50):
        sim.generate_step()

    final_vib = sim.state["vibration"]
    assert final_vib > initial_vib, "Bearing wear must produce elevated vibration"


def test_pressure_leak_fault_injection():
    """Verify that pressure leak drops operating pressure towards zero."""
    sim = SensorSimulator(machine_code="CMP-01", sample_rate_hz=5.0)
    sim.set_mode(SensorSimulator.MODE_PRESSURE_LEAK)

    initial_press = sim.state["pressure"]
    for _ in range(30):
        sim.generate_step()

    final_press = sim.state["pressure"]
    assert final_press < initial_press, "Pressure leak must reduce line pressure"
