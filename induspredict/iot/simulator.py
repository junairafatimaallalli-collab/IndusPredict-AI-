"""
IndusPredict AI - Industrial Sensor Simulator.
Generates realistic multi-sensor physical telemetry with support for nominal running,
thermal degradation, bearing vibration wear, pressure leakages, and sensor disconnects.
All generated readings are strictly identified as source="simulated".
"""

from datetime import datetime
import math
import random
import time
from typing import Any, Dict, List, Optional
from PySide6.QtCore import QObject, QThread, Signal

from induspredict.core.config import SENSOR_SPECS
from induspredict.core.logger import get_logger
from induspredict.core.utils import utc_now

logger = get_logger("iot.simulator")


class SensorSimulator(QObject):
    """
    Physics-informed sensor telemetry simulator for registered machinery.
    Can inject specific failure modes to test condition monitoring and predictive engines.
    """

    # Signal emitted when a new multi-sensor reading frame is generated
    # Payload format: Dict[str, Any] with machine_code, timestamp, sensors: Dict[sensor_type, dict]
    frame_generated = Signal(dict)
    alert_triggered = Signal(dict)
    status_changed = Signal(bool)  # is_running

    # Simulation Modes
    MODE_NOMINAL = "Nominal Operation"
    MODE_THERMAL_OVERRUN = "Thermal Runaway"
    MODE_BEARING_DEGRADATION = "Bearing Vibration Wear"
    MODE_PRESSURE_LEAK = "Pneumatic/Hydraulic Leak"
    MODE_INTERMITTENT_DISCONNECT = "Sensor Communication Loss"

    ALL_MODES = [
        MODE_NOMINAL,
        MODE_THERMAL_OVERRUN,
        MODE_BEARING_DEGRADATION,
        MODE_PRESSURE_LEAK,
        MODE_INTERMITTENT_DISCONNECT,
    ]

    def __init__(self, machine_code: str = "CMP-01", sample_rate_hz: float = 1.0, parent=None):
        super().__init__(parent)
        self.machine_code = machine_code
        self.sample_rate_hz = sample_rate_hz
        self.mode = self.MODE_NOMINAL
        self._is_running = False
        self._step_count = 0

        # Physical machine state variables
        self.state = {
            "temperature": 68.0,      # °C
            "vibration": 2.8,          # mm/s
            "pressure": 7.2,           # bar
            "motor_speed": 2980.0,     # RPM
            "current": 64.0,           # A
            "torque": 185.0,           # Nm
            "operating_hours": 14250.0 # hrs
        }

    def set_machine(self, machine_code: str, base_hours: float = 10000.0) -> None:
        """Switch simulation target machine."""
        self.machine_code = machine_code
        self.state["operating_hours"] = base_hours
        self._step_count = 0
        self.reset_to_nominal()

    def set_mode(self, mode: str) -> None:
        """Set simulation mode (Nominal or Fault Injection)."""
        if mode in self.ALL_MODES:
            self.mode = mode
            logger.info(f"Simulator for '{self.machine_code}' switched mode to: {mode}")

    def reset_to_nominal(self) -> None:
        """Reset state variables to healthy baseline envelope."""
        self.mode = self.MODE_NOMINAL
        self.state["temperature"] = random.uniform(62.0, 70.0)
        self.state["vibration"] = random.uniform(2.1, 3.4)
        self.state["pressure"] = random.uniform(6.8, 7.6)
        self.state["motor_speed"] = random.uniform(2970.0, 2995.0)
        self.state["current"] = random.uniform(60.0, 68.0)
        self.state["torque"] = random.uniform(175.0, 195.0)

    def generate_step(self) -> Optional[Dict[str, Any]]:
        """
        Advance physical model by one discrete time step dt = 1 / sample_rate_hz.
        Returns a frame of sensor readings tagged as source='simulated'.
        """
        self._step_count += 1
        t = self._step_count * (1.0 / max(0.1, self.sample_rate_hz))
        now = utc_now()

        # Handle communication disconnect simulation
        if self.mode == self.MODE_INTERMITTENT_DISCONNECT:
            # 60% probability of dropping packet
            if random.random() < 0.60:
                logger.debug(f"Simulated packet drop for {self.machine_code}")
                return None

        # 1. Physics Engine: State Progression
        if self.mode == self.MODE_NOMINAL:
            # Healthy steady-state with white noise and mild sinusoidal cycle
            self.state["temperature"] = 68.0 + 3.0 * math.sin(0.05 * t) + random.gauss(0, 0.4)
            self.state["vibration"] = 2.8 + 0.3 * math.sin(0.2 * t) + random.gauss(0, 0.15)
            self.state["pressure"] = 7.2 + 0.2 * math.cos(0.08 * t) + random.gauss(0, 0.08)
            self.state["motor_speed"] = 2980.0 + 8.0 * math.sin(0.1 * t) + random.gauss(0, 3.0)
            self.state["current"] = 64.0 + 2.0 * math.sin(0.1 * t) + random.gauss(0, 0.8)
            self.state["torque"] = 185.0 + 5.0 * math.sin(0.1 * t) + random.gauss(0, 2.0)

        elif self.mode == self.MODE_THERMAL_OVERRUN:
            # Heat accumulation model: dT/dt = (Q_in - Q_out) / C
            # Progressive temperature runaway past warning (85°C) into critical (>102°C)
            thermal_rate = 0.45  # °C per second
            self.state["temperature"] = min(118.0, self.state["temperature"] + thermal_rate + random.gauss(0, 0.3))
            self.state["current"] += 0.25 + random.gauss(0, 0.2)  # Higher winding current with heat
            self.state["vibration"] += 0.03 + random.gauss(0, 0.1)

        elif self.mode == self.MODE_BEARING_DEGRADATION:
            # Bearing defect vibration: fundamental rotational frequency + ball pass harmonics + shock impulses
            rot_freq = 2980.0 / 60.0  # ~49.6 Hz
            decay = min(1.0, self._step_count / 100.0)
            impulse = 4.5 if random.random() < 0.15 else 0.0  # Spalling impact shocks
            self.state["vibration"] = min(24.0, 3.0 + 8.5 * decay + impulse + random.gauss(0, 0.3))
            self.state["temperature"] += 0.15 * decay + random.gauss(0, 0.1)  # Bearing friction heat
            self.state["torque"] += 0.4 * decay + random.gauss(0, 0.5)

        elif self.mode == self.MODE_PRESSURE_LEAK:
            # Pressure drops precipitously; compressor motor current surges trying to maintain load
            self.state["pressure"] = max(1.2, self.state["pressure"] - 0.22 + random.gauss(0, 0.05))
            self.state["motor_speed"] = min(3600.0, self.state["motor_speed"] + 8.0 + random.gauss(0, 4.0))
            self.state["current"] = min(140.0, self.state["current"] + 0.6 + random.gauss(0, 0.5))

        # Accumulate operating hours
        self.state["operating_hours"] += (1.0 / 3600.0) / max(0.1, self.sample_rate_hz)

        # 2. Package Multi-Sensor Reading Frame
        frame = {
            "machine_code": self.machine_code,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "iso_timestamp": now.isoformat(),
            "source": "simulated",  # Explicit unambiguous disclosure
            "simulation_mode": self.mode,
            "sensors": {},
        }

        for stype, sval in self.state.items():
            spec = SENSOR_SPECS.get(stype, {"name": stype.title(), "unit": "", "warning": 1000, "critical": 2000})
            is_warn = sval >= spec["warning"]
            is_crit = sval >= spec["critical"]

            frame["sensors"][stype] = {
                "name": spec["name"],
                "value": round(sval, 2),
                "unit": spec["unit"],
                "warning_threshold": spec["warning"],
                "critical_threshold": spec["critical"],
                "status": "critical" if is_crit else ("warning" if is_warn else "healthy"),
                "is_anomaly": is_warn or is_crit,
            }

            # Emit alert signal if critical or warning threshold is crossed
            if is_crit or is_warn:
                self.alert_triggered.emit({
                    "machine_code": self.machine_code,
                    "sensor_type": stype,
                    "value": round(sval, 2),
                    "unit": spec["unit"],
                    "severity": "Critical" if is_crit else "Warning",
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "simulated",
                })

        self.frame_generated.emit(frame)
        return frame


class SimulatorWorker(QThread):
    """Background worker thread executing real-time simulation loops."""

    def __init__(self, simulator: SensorSimulator, parent=None):
        super().__init__(parent)
        self.simulator = simulator
        self._running = False

    def run(self) -> None:
        self._running = True
        self.simulator.status_changed.emit(True)
        logger.info(f"Simulator worker thread started for '{self.simulator.machine_code}'.")

        while self._running:
            delay = 1.0 / max(0.1, self.simulator.sample_rate_hz)
            self.simulator.generate_step()
            time.sleep(delay)

        self.simulator.status_changed.emit(False)
        logger.info("Simulator worker thread terminated.")

    def stop(self) -> None:
        self._running = False
        self.wait(2000)
