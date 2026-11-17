"""
IndusPredict AI - Digital Twin Simulation Service.
Provides virtual machine representations, parametric what-if operating condition
simulations, physics-informed stress metrics, coupled ML risk forecasting,
and scenario persistence.
"""

from datetime import datetime
import json
import math
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import selectinload

from induspredict.core.logger import get_logger
from induspredict.core.utils import utc_now
from induspredict.db.models import Machine, MachineComponent, Sensor, SimulationScenario, User
from induspredict.db.session import get_db_session
from induspredict.services.prediction_service import PredictionService

logger = get_logger("services.digital_twin")


class DigitalTwinService:
    """Service providing virtual machine profiles, what-if simulations, and scenario management."""

    DISCLAIMER = (
        "SCIENTIFIC NOTICE: Digital Twin simulation represents mathematical parametric modeling "
        "and empirical ML response curves under perturbed operational stress. It does not replace "
        "direct physical sensor telemetry or finite-element metallurgical analysis."
    )

    @classmethod
    def get_machine_virtual_profile(cls, machine_code: str) -> Dict[str, Any]:
        """
        Build a comprehensive virtual digital twin profile for a registered industrial machine,
        including hardware specs, component degradation states, and baseline telemetry.
        """
        with get_db_session() as session:
            machine = (
                session.query(Machine)
                .options(
                    selectinload(Machine.components),
                    selectinload(Machine.sensors),
                    selectinload(Machine.predictions),
                )
                .filter(Machine.machine_code == machine_code)
                .first()
            )
            if not machine:
                raise ValueError(f"Machine with code '{machine_code}' not found in registry.")

            # Component wear states
            components_data = []
            total_wear_ratio = 0.0
            for comp in machine.components:
                wear_ratio = min(1.0, max(0.0, comp.current_operating_hours / max(1.0, comp.expected_lifespan_hours)))
                total_wear_ratio += wear_ratio
                components_data.append({
                    "id": comp.id,
                    "name": comp.name,
                    "type": comp.component_type,
                    "operating_hours": comp.current_operating_hours,
                    "expected_lifetime": comp.expected_lifespan_hours,
                    "health_percentage": round((1.0 - wear_ratio) * 100.0, 1),
                })

            avg_comp_wear = (total_wear_ratio / len(machine.components)) if machine.components else 0.1

            # Design life assumed 20,000 hours for industrial assets
            design_life_hours = 20000.0
            life_wear = min(1.0, machine.operating_hours / design_life_hours)

            # Nominal baseline operating parameters (calibrated to industrial equipment standard)
            baseline_telemetry = {
                "air_temperature": 25.0,        # 298.15 K standard ambient
                "process_temperature": 35.0,    # 308.15 K
                "rotational_speed": 1500.0,     # RPM
                "torque": 40.0,                 # Nm
                "tool_wear": round(avg_comp_wear * 200.0, 1),  # minutes of tool/bearing wear
                "vibration": 1.45,              # mm/s RMS
                "pressure": 6.8,                # bar
            }

            # Baseline Machine Health Index (0 - 100%)
            # Health = 100 - (35% life wear + 40% component wear + 25% vibration/thermal stress)
            base_health = 100.0 - (life_wear * 35.0 + avg_comp_wear * 40.0 + (baseline_telemetry["vibration"] / 4.5) * 25.0)
            base_health = max(5.0, min(100.0, round(base_health, 1)))

            return {
                "machine_id": machine.id,
                "machine_code": machine.machine_code,
                "machine_name": machine.name,
                "machine_type": machine.machine_type,
                "manufacturer": machine.manufacturer,
                "serial_number": machine.serial_number,
                "installation_date": machine.installation_date.isoformat() if machine.installation_date else None,
                "operating_hours": machine.operating_hours,
                "criticality": machine.criticality,
                "location": machine.location,
                "status": machine.status,
                "components": components_data,
                "baseline_health": base_health,
                "baseline_telemetry": baseline_telemetry,
                "disclaimer": cls.DISCLAIMER,
            }

    @classmethod
    def simulate_what_if(
        cls,
        machine_code: str,
        simulated_parameters: Dict[str, float],
        model_name: str = "Random Forest",
    ) -> Dict[str, Any]:
        """
        Execute parametric what-if simulation:
        1. Evaluates baseline operational condition.
        2. Applies simulated parameter adjustments (temperature, speed, torque, wear, vibration, pressure).
        3. Computes physical stress multipliers (thermal dissipation, mechanical power, overstrain).
        4. Runs coupled ML inference on baseline vs. simulated conditions.
        5. Computes predicted risk changes and estimated health degradation delta.
        """
        profile = cls.get_machine_virtual_profile(machine_code)
        baseline_params = profile["baseline_telemetry"]

        # Simulated parameters with fallback to baseline if unspecified
        sim_air_t = float(simulated_parameters.get("air_temperature", baseline_params["air_temperature"]))
        sim_proc_t = float(simulated_parameters.get("process_temperature", baseline_params["process_temperature"]))
        sim_speed = float(simulated_parameters.get("rotational_speed", baseline_params["rotational_speed"]))
        sim_torque = float(simulated_parameters.get("torque", baseline_params["torque"]))
        sim_wear = float(simulated_parameters.get("tool_wear", baseline_params["tool_wear"]))
        sim_vibration = float(simulated_parameters.get("vibration", baseline_params["vibration"]))
        sim_pressure = float(simulated_parameters.get("pressure", baseline_params["pressure"]))

        # 1. Physics stress calculations
        # Thermal delta
        delta_t = sim_proc_t - sim_air_t
        # Mechanical rotational power (Watts) = Torque (Nm) * Angular velocity (rad/s)
        angular_vel = sim_speed * (2.0 * math.pi / 60.0)
        mech_power_watts = sim_torque * angular_vel
        # Overstrain index (Nm * min)
        overstrain_idx = sim_torque * sim_wear

        # Baseline physical metrics
        base_delta_t = baseline_params["process_temperature"] - baseline_params["air_temperature"]
        base_power = baseline_params["torque"] * (baseline_params["rotational_speed"] * (2.0 * math.pi / 60.0))
        base_overstrain = baseline_params["torque"] * baseline_params["tool_wear"]

        # Health degradation calculation
        # Stress penalties
        thermal_penalty = max(0.0, (12.0 - delta_t) * 1.5) if delta_t < 8.6 else max(0.0, (delta_t - 15.0) * 1.2)
        power_penalty = max(0.0, (mech_power_watts - 6000.0) / 300.0) if mech_power_watts > 6000.0 else (
            max(0.0, (2000.0 - mech_power_watts) / 200.0) if mech_power_watts < 2500.0 else 0.0
        )
        wear_penalty = (sim_wear / 240.0) * 20.0
        vib_penalty = max(0.0, (sim_vibration - 1.8) * 8.0)

        total_sim_penalty = thermal_penalty + power_penalty + wear_penalty + vib_penalty
        simulated_health = max(0.0, min(100.0, round(profile["baseline_health"] - total_sim_penalty, 1)))
        estimated_health_delta = round(simulated_health - profile["baseline_health"], 1)

        # 2. Coupled ML Failure Risk Inference
        base_ml_input = {
            "air_temperature": baseline_params["air_temperature"],
            "process_temperature": baseline_params["process_temperature"],
            "rotational_speed": baseline_params["rotational_speed"],
            "torque": baseline_params["torque"],
            "tool_wear": baseline_params["tool_wear"],
            "machine_type": "M",
        }
        sim_ml_input = {
            "air_temperature": sim_air_t,
            "process_temperature": sim_proc_t,
            "rotational_speed": sim_speed,
            "torque": sim_torque,
            "tool_wear": sim_wear,
            "machine_type": "M",
        }

        base_pred = PredictionService.predict_machine_failure(base_ml_input, machine_code=machine_code, model_name=model_name)
        sim_pred = PredictionService.predict_machine_failure(sim_ml_input, machine_code=machine_code, model_name=model_name)

        base_risk_prob = base_pred["failure_probability"]
        sim_risk_prob = sim_pred["failure_probability"]
        risk_prob_delta = round(sim_risk_prob - base_risk_prob, 4)

        return {
            "machine_code": machine_code,
            "machine_name": profile["machine_name"],
            "model_used": model_name,
            "is_simulated": True,
            "simulation_tag": "SYNTHETIC_PARAMETRIC_WHAT_IF",
            "baseline": {
                "health_percentage": profile["baseline_health"],
                "failure_probability": base_risk_prob,
                "risk_category": base_pred["risk_category"],
                "delta_t_c": round(base_delta_t, 2),
                "mechanical_power_kw": round(base_power / 1000.0, 2),
                "overstrain_index": round(base_overstrain, 1),
                "parameters": baseline_params,
            },
            "simulated": {
                "health_percentage": simulated_health,
                "failure_probability": sim_risk_prob,
                "risk_category": sim_pred["risk_category"],
                "likely_failure_mode": sim_pred["likely_failure_mode"],
                "delta_t_c": round(delta_t, 2),
                "mechanical_power_kw": round(mech_power_watts / 1000.0, 2),
                "overstrain_index": round(overstrain_idx, 1),
                "parameters": {
                    "air_temperature": sim_air_t,
                    "process_temperature": sim_proc_t,
                    "rotational_speed": sim_speed,
                    "torque": sim_torque,
                    "tool_wear": sim_wear,
                    "vibration": sim_vibration,
                    "pressure": sim_pressure,
                },
            },
            "comparison": {
                "predicted_risk_delta_pct": round(risk_prob_delta * 100.0, 2),
                "estimated_health_delta_pct": estimated_health_delta,
                "severity_escalated": sim_risk_prob > (base_risk_prob + 0.05),
                "top_stress_factors": list(sim_pred.get("top_contributions", {}).keys())[:3],
            },
            "disclaimer": cls.DISCLAIMER,
            "timestamp": utc_now().isoformat(),
        }

    @classmethod
    def save_scenario(
        cls,
        machine_code: str,
        name: str,
        baseline_state: Dict[str, Any],
        simulated_parameters: Dict[str, Any],
        predicted_risk_delta: float,
        estimated_health_delta: float,
        notes: Optional[str] = None,
        created_by_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Persist a digital twin simulation scenario into SQLite."""
        with get_db_session() as session:
            machine = session.query(Machine).filter(Machine.machine_code == machine_code).first()
            if not machine:
                raise ValueError(f"Machine '{machine_code}' not found.")

            scenario = SimulationScenario(
                machine_id=machine.id,
                name=name,
                created_by_id=created_by_id,
                baseline_state_json=json.dumps(baseline_state),
                simulated_parameters_json=json.dumps(simulated_parameters),
                predicted_risk_delta=predicted_risk_delta,
                estimated_health_delta=estimated_health_delta,
                notes=notes,
                created_at=utc_now(),
            )
            session.add(scenario)
            session.commit()
            session.refresh(scenario)

            logger.info(f"Saved Digital Twin scenario '{name}' (ID: {scenario.id}) for machine {machine_code}")
            return {
                "id": scenario.id,
                "name": scenario.name,
                "machine_code": machine_code,
                "predicted_risk_delta": scenario.predicted_risk_delta,
                "estimated_health_delta": scenario.estimated_health_delta,
                "created_at": scenario.created_at.isoformat(),
            }

    @classmethod
    def get_saved_scenarios(cls, machine_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve list of saved simulation scenarios from database."""
        with get_db_session() as session:
            query = session.query(SimulationScenario).join(Machine)
            if machine_code:
                query = query.filter(Machine.machine_code == machine_code)

            scenarios = query.order_by(SimulationScenario.created_at.desc()).all()
            return [
                {
                    "id": s.id,
                    "machine_code": s.machine.machine_code,
                    "machine_name": s.machine.name,
                    "name": s.name,
                    "predicted_risk_delta": s.predicted_risk_delta,
                    "estimated_health_delta": s.estimated_health_delta,
                    "notes": s.notes,
                    "created_at": s.created_at.strftime("%Y-%m-%d %H:%M"),
                    "simulated_parameters": json.loads(s.simulated_parameters_json) if s.simulated_parameters_json else {},
                }
                for s in scenarios
            ]

    @classmethod
    def delete_scenario(cls, scenario_id: int) -> bool:
        """Delete a saved simulation scenario by ID."""
        with get_db_session() as session:
            scenario = session.query(SimulationScenario).filter(SimulationScenario.id == scenario_id).first()
            if not scenario:
                return False
            session.delete(scenario)
            session.commit()
            return True
