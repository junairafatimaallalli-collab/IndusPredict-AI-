"""
IndusPredict AI - Database Seeder.
Initializes roles, default user accounts, realistic industrial machines,
component hierarchies, sensor networks, initial readings, and initial work orders.
"""

from datetime import datetime, timedelta
import random
from induspredict.core.config import (
    ALL_CRITICALITY_LEVELS,
    CRITICALITY_CRITICAL,
    CRITICALITY_HIGH,
    CRITICALITY_MEDIUM,
    MODELS_DIR,
    ROLE_ADMIN,
    ROLE_ENGINEER,
    ROLE_RESEARCHER,
    ROLE_VIEWER,
    SENSOR_SPECS,
)
from induspredict.core.logger import get_logger
from induspredict.core.security import hash_password
from induspredict.core.utils import utc_now
from induspredict.db.models import (
    AnomalyEvent,
    AuditLog,
    Machine,
    MachineComponent,
    MaintenanceHistory,
    ModelEvaluation,
    ModelVersion,
    Prediction,
    Role,
    RULEstimate,
    Sensor,
    SensorReading,
    User,
    WorkOrder,
)
from induspredict.db.session import get_db_session, init_db

logger = get_logger("db.seed")


def seed_database(force: bool = False) -> None:
    """Seed the database with verified initial industrial schemas and sample records."""
    init_db()

    with get_db_session() as session:
        # Check if already seeded
        existing_roles = session.query(Role).count()
        if existing_roles > 0 and not force:
            logger.info("Database already seeded. Skipping initial seeding.")
            return

        logger.info("Starting database seeding...")

        # 1. Seed Roles
        roles_dict = {}
        for role_name, desc in [
            (ROLE_ADMIN, "Full administrative privileges, user management, and system governance."),
            (ROLE_ENGINEER, "Maintenance operations, work order execution, and live sensor monitoring."),
            (ROLE_RESEARCHER, "Model experimentation, RUL hyperparameter tuning, and XAI inspection."),
            (ROLE_VIEWER, "Read-only access to operational dashboards and reports."),
        ]:
            role = Role(name=role_name, description=desc)
            session.add(role)
            roles_dict[role_name] = role

        session.flush()

        # 2. Seed Initial Users
        users_meta = [
            ("admin", "admin@induspredict.ai", "Admin@123", "Chief Systems Administrator", ROLE_ADMIN),
            ("engineer", "engineer@induspredict.ai", "Engineer@123", "Senior Maintenance Engineer", ROLE_ENGINEER),
            ("researcher", "researcher@induspredict.ai", "Researcher@123", "Lead AI Prognostics Scientist", ROLE_RESEARCHER),
            ("viewer", "viewer@induspredict.ai", "Viewer@123", "Plant Operations Observer", ROLE_VIEWER),
        ]

        users_dict = {}
        for uname, email, pwd, fname, rname in users_meta:
            user = User(
                username=uname,
                email=email,
                password_hash=hash_password(pwd),
                full_name=fname,
                role_id=roles_dict[rname].id,
                is_active=True,
                last_login=utc_now(),
            )
            session.add(user)
            users_dict[uname] = user

        session.flush()

        # 3. Seed Realistic Industrial Machines
        machines_data = [
            {
                "code": "CMP-01",
                "name": "Atlas Copco GA 90 VSD Screw Compressor",
                "type": "Screw Compressor",
                "manufacturer": "Atlas Copco",
                "model": "GA-90-VSD-PLUS",
                "serial": "AC-2022-8841",
                "installation": datetime(2022, 3, 15),
                "hours": 14250.0,
                "location": "Utility Building - Compressor Bay A",
                "criticality": CRITICALITY_CRITICAL,
                "status": "healthy",
                "components": [
                    ("Twin-Screw Air End", "Mechanical Rotor", 30000.0, 14250.0),
                    ("High-Efficiency Drive Motor", "Induction Motor", 40000.0, 14250.0),
                    ("Oil Separator Filter", "Filter Element", 8000.0, 3100.0),
                    ("Centrifugal Cooling Fan", "Cooling Unit", 25000.0, 14250.0),
                ],
                "sensors": [
                    ("CMP01-TEMP", "Discharge Air Temperature", "temperature", "°C", 10.0, 120.0, 85.0, 102.0),
                    ("CMP01-VIB", "Air End Vibration Velocity", "vibration", "mm/s", 0.0, 25.0, 4.5, 7.1),
                    ("CMP01-PRESS", "Discharge Pressure", "pressure", "bar", 0.0, 16.0, 10.5, 13.0),
                    ("CMP01-CURR", "Motor Operating Current", "current", "A", 0.0, 160.0, 120.0, 145.0),
                ],
            },
            {
                "code": "MTR-02",
                "name": "Siemens SIMOTICS 1LE1 Induction Motor",
                "type": "Induction Motor",
                "manufacturer": "Siemens AG",
                "model": "SIMOTICS-1LE1503-2AB5",
                "serial": "SIE-2021-9031",
                "installation": datetime(2021, 6, 20),
                "hours": 18400.0,
                "location": "Manufacturing Line 2 - Main Extruder",
                "criticality": CRITICALITY_HIGH,
                "status": "warning",
                "components": [
                    ("Drive-End Deep Groove Ball Bearing", "Rolling Bearing", 20000.0, 18400.0),
                    ("Stator Copper Windings", "Electrical Stator", 50000.0, 18400.0),
                    ("Squirrel Cage Rotor", "Rotor Shaft", 50000.0, 18400.0),
                ],
                "sensors": [
                    ("MTR02-SPEED", "Rotational Speed", "motor_speed", "RPM", 0.0, 3600.0, 3050.0, 3400.0),
                    ("MTR02-VIB", "Bearing Vibration RMS", "vibration", "mm/s", 0.0, 20.0, 5.8, 8.5),
                    ("MTR02-TEMP", "Winding Hot-Spot Temperature", "temperature", "°C", 10.0, 130.0, 92.0, 110.0),
                    ("MTR02-TORQ", "Shaft Mechanical Torque", "torque", "Nm", 0.0, 350.0, 240.0, 290.0),
                ],
            },
            {
                "code": "PMP-03",
                "name": "Sulzer Ahlstar Centrifugal Slurry Pump",
                "type": "Centrifugal Pump",
                "manufacturer": "Sulzer Ltd",
                "model": "AHLSTAR-WPP32-100",
                "serial": "SLZ-2023-1102",
                "installation": datetime(2023, 1, 10),
                "hours": 8750.0,
                "location": "Chemical Process Unit 3 - Slurry Feed",
                "criticality": CRITICALITY_HIGH,
                "status": "healthy",
                "components": [
                    ("Hard Metal Impeller", "Slurry Impeller", 15000.0, 8750.0),
                    ("Double Mechanical Cartridge Seal", "Shaft Seal", 10000.0, 8750.0),
                    ("Heavy Duty Casing", "Cast Iron Casing", 40000.0, 8750.0),
                ],
                "sensors": [
                    ("PMP03-PRESS-SUC", "Suction Pressure", "pressure", "bar", 0.0, 10.0, 1.2, 0.8),
                    ("PMP03-PRESS-DIS", "Discharge Pressure", "pressure", "bar", 0.0, 25.0, 16.5, 21.0),
                    ("PMP03-VIB", "Casing Vibration", "vibration", "mm/s", 0.0, 25.0, 4.2, 7.0),
                    ("PMP03-TEMP", "Seal Chamber Temperature", "temperature", "°C", 10.0, 110.0, 75.0, 95.0),
                ],
            },
            {
                "code": "RBT-04",
                "name": "KUKA KR 1000 Titan 6-Axis Heavy Robot",
                "type": "Industrial Robot",
                "manufacturer": "KUKA Robotics",
                "model": "KR-1000-TITAN-F",
                "serial": "KUK-2022-4419",
                "installation": datetime(2022, 11, 5),
                "hours": 11200.0,
                "location": "Assembly Bay 4 - Press Automation",
                "criticality": CRITICALITY_CRITICAL,
                "status": "healthy",
                "components": [
                    ("Axis 1 Cycloidal Gearbox", "Precision Gearbox", 25000.0, 11200.0),
                    ("Axis 2 Harmonic Drive", "Gear Reduction", 25000.0, 11200.0),
                    ("Heavy Payload Wrist Tool Flange", "End Effector", 30000.0, 11200.0),
                ],
                "sensors": [
                    ("RBT04-CURR", "Axis 1 Servo Current", "current", "A", 0.0, 120.0, 75.0, 95.0),
                    ("RBT04-TORQ", "Joint 2 Dynamic Torque", "torque", "Nm", 0.0, 400.0, 280.0, 340.0),
                    ("RBT04-TEMP", "Gearbox Oil Temperature", "temperature", "°C", 10.0, 100.0, 70.0, 88.0),
                ],
            },
            {
                "code": "CNC-05",
                "name": "DMG MORI DMU 50 5-Axis Machining Center",
                "type": "CNC Machining Center",
                "manufacturer": "DMG MORI",
                "model": "DMU-50-GEN3",
                "serial": "DMG-2020-0092",
                "installation": datetime(2020, 8, 14),
                "hours": 22100.0,
                "location": "Precision Machining Center - Cell 1",
                "criticality": CRITICALITY_MEDIUM,
                "status": "critical",
                "components": [
                    ("High Speed Electro-Spindle (20000 RPM)", "Spindle Unit", 15000.0, 14800.0),
                    ("X-Axis Recirculating Ball Screw", "Drive Screw", 25000.0, 22100.0),
                    ("Automatic 30-Pocket Tool Magazine", "Tool Changer", 30000.0, 22100.0),
                ],
                "sensors": [
                    ("CNC05-VIB", "Spindle Vibration Velocity", "vibration", "mm/s", 0.0, 20.0, 6.0, 9.5),
                    ("CNC05-TEMP", "Spindle Bearing Temperature", "temperature", "°C", 10.0, 110.0, 80.0, 98.0),
                    ("CNC05-SPEED", "Spindle Actual RPM", "motor_speed", "RPM", 0.0, 20000.0, 18500.0, 19500.0),
                    ("CNC05-TORQ", "Milling Cutting Torque", "torque", "Nm", 0.0, 250.0, 180.0, 220.0),
                ],
            },
        ]

        created_machines = []
        created_sensors = []

        now = utc_now()

        for m_info in machines_data:
            machine = Machine(
                machine_code=m_info["code"],
                name=m_info["name"],
                machine_type=m_info["type"],
                manufacturer=m_info["manufacturer"],
                model_number=m_info["model"],
                serial_number=m_info["serial"],
                installation_date=m_info["installation"],
                operating_hours=m_info["hours"],
                location=m_info["location"],
                criticality=m_info["criticality"],
                status=m_info["status"],
                description=f"Standard industrial asset installed for automated operations in {m_info['location']}.",
                created_at=m_info["installation"],
            )
            session.add(machine)
            session.flush()
            created_machines.append(machine)

            # Components
            for c_name, c_type, lifespan, cur_hours in m_info["components"]:
                comp = MachineComponent(
                    machine_id=machine.id,
                    name=c_name,
                    component_type=c_type,
                    expected_lifespan_hours=lifespan,
                    current_operating_hours=cur_hours,
                    status="worn" if cur_hours / lifespan > 0.8 else "healthy",
                )
                session.add(comp)

            # Sensors
            for s_code, s_name, s_type, s_unit, s_min, s_max, s_warn, s_crit in m_info["sensors"]:
                sensor = Sensor(
                    machine_id=machine.id,
                    sensor_code=s_code,
                    name=s_name,
                    sensor_type=s_type,
                    unit=s_unit,
                    min_threshold=s_min,
                    max_threshold=s_max,
                    warning_threshold=s_warn,
                    critical_threshold=s_crit,
                    sample_rate_hz=1.0,
                    status="warning" if machine.status == "warning" else ("error" if machine.status == "critical" else "active"),
                )
                session.add(sensor)
                session.flush()
                created_sensors.append(sensor)

                # Generate historical readings spanning the last 12 hours (spaced every 15 minutes)
                # Clearly tagged as source="simulated"
                base_val = (s_warn + s_min) / 2.0
                if machine.status == "warning":
                    base_val = s_warn * 0.95
                elif machine.status == "critical":
                    base_val = s_crit * 0.98

                for step in range(48, 0, -1):
                    reading_time = now - timedelta(minutes=step * 15)
                    jitter = (random.random() - 0.5) * (s_warn - s_min) * 0.08
                    val = max(s_min, min(s_max, round(base_val + jitter, 2)))
                    is_anomaly = val >= s_warn

                    reading = SensorReading(
                        machine_id=machine.id,
                        sensor_id=sensor.id,
                        timestamp=reading_time,
                        value=val,
                        source="simulated",  # Explicit disclosure
                        is_anomaly=is_anomaly,
                        quality_flag="GOOD",
                    )
                    session.add(reading)

        session.flush()

        # 4. Seed Initial Realistic Predictions (computed baselines)
        for m in created_machines:
            if m.status == "healthy":
                prob = round(random.uniform(0.04, 0.18), 3)
                risk_cat = "Normal"
            elif m.status == "warning":
                prob = round(random.uniform(0.42, 0.58), 3)
                risk_cat = "Warning"
            else:  # critical
                prob = round(random.uniform(0.79, 0.92), 3)
                risk_cat = "Critical"

            pred = Prediction(
                machine_id=m.id,
                timestamp=now - timedelta(minutes=10),
                model_name="RandomForestClassifier-Baseline",
                model_version="1.0.0",
                failure_probability=prob,
                risk_category=risk_cat,
                confidence_score=round(random.uniform(0.85, 0.96), 2),
                predicted_failure_mode="Bearing Wear" if risk_cat != "Normal" else None,
                contributing_features={"vibration": 0.45, "temperature": 0.32, "operating_hours": 0.23},
                status="active",
            )
            session.add(pred)

            # Seed RUL estimate
            est_cycles = 180.0 if m.status == "healthy" else (65.0 if m.status == "warning" else 18.0)
            rul = RULEstimate(
                machine_id=m.id,
                timestamp=now - timedelta(minutes=10),
                model_name="PyTorch-LSTM-Prognostics",
                model_version="1.0.0",
                predicted_rul_cycles=est_cycles,
                lower_bound_95=max(1.0, est_cycles - 12.0),
                upper_bound_95=est_cycles + 14.0,
                degradation_pct=round(100.0 - (est_cycles / 200.0 * 100.0), 1),
                unit="operating_cycles",
                input_window_length=30,
            )
            session.add(rul)

        # 5. Seed Initial Work Orders
        engineer_user = users_dict["engineer"]
        admin_user = users_dict["admin"]

        wo1 = WorkOrder(
            machine_id=created_machines[1].id,  # MTR-02 (warning)
            work_order_code="WO-2026-001",
            title="Drive-End Bearing Acoustic & Vibration Inspection",
            description="Motor MTR-02 has displayed vibration levels approaching warning threshold (5.8 mm/s). Inspect grease channels and inspect for roller spalling.",
            priority="High",
            status="In Progress",
            created_by_id=admin_user.id,
            assigned_to_id=engineer_user.id,
            scheduled_start=now + timedelta(days=1),
            estimated_downtime_hours=3.5,
            notes="Requires lock-out tag-out (LOTO) on Extruder Line 2.",
        )
        session.add(wo1)

        wo2 = WorkOrder(
            machine_id=created_machines[4].id,  # CNC-05 (critical)
            work_order_code="WO-2026-002",
            title="Spindle Thermal Overrun & Runout Diagnostic",
            description="DMG MORI DMU 50 spindle exhibits anomalous vibration (9.1 mm/s) and bearing temp 84°C during high-speed finishing cuts. Urgent runout measurement required.",
            priority="Emergency",
            status="Pending",
            created_by_id=engineer_user.id,
            assigned_to_id=engineer_user.id,
            scheduled_start=now + timedelta(hours=6),
            estimated_downtime_hours=5.0,
            notes="Machine flagged for immediate scheduled maintenance window.",
        )
        session.add(wo2)

        # 6. Seed Initial Maintenance History
        hist1 = MaintenanceHistory(
            machine_id=created_machines[0].id,  # CMP-01
            service_date=now - timedelta(days=45),
            action_taken="Replaced oil separator filter cartridge and renewed synthetic compressor lubricant.",
            parts_replaced="Atlas Copco Filter 1622-0871-00, Roto-Inject Fluid (20L)",
            downtime_hours=2.0,
            technician_name="J. Vance (Level II Vibration Analyst)",
            remarks="Compressor operating within baseline ISO vibration envelopes post-service.",
        )
        session.add(hist1)

        # 7. Seed Model Versions and Genuine Evaluations on AI4I 2020
        model_benchmarks_data = [
            ("Random Forest", "RandomForest", "random_forest.joblib", 0.9855, 0.7600, 0.8382, 0.7972, 0.9852, 0.8705, 0.7907, {"tn": 1914, "fp": 18, "fn": 11, "tp": 57}),
            ("XGBoost", "XGBoost", "xgboost.joblib", 0.9850, 0.7500, 0.8382, 0.7917, 0.9795, 0.8832, 0.7852, {"tn": 1913, "fp": 19, "fn": 11, "tp": 57}),
            ("LightGBM", "LightGBM", "lightgbm.joblib", 0.9840, 0.7195, 0.8676, 0.7867, 0.9754, 0.8962, 0.7821, {"tn": 1909, "fp": 23, "fn": 9, "tp": 59}),
            ("Logistic Regression", "LogisticRegression", "logistic_regression.joblib", 0.8705, 0.1909, 0.8676, 0.3130, 0.9437, 0.4720, 0.3702, {"tn": 1682, "fp": 250, "fn": 9, "tp": 59}),
        ]

        import json
        for m_name, algo, fname, acc, prec, rec, f1, roc, pr, mcc, cm in model_benchmarks_data:
            mv = ModelVersion(
                model_name=m_name,
                version="1.0.0",
                model_type="Classifier",
                algorithm=algo,
                artifact_path=str(MODELS_DIR / fname),
                metrics_json=json.dumps({"accuracy": acc, "f1_score": f1, "roc_auc": roc, "pr_auc": pr, "mcc": mcc}),
                is_active=True,
                created_at=now,
            )
            session.add(mv)
            session.flush()

            me = ModelEvaluation(
                model_version_id=mv.id,
                dataset_name="UCI AI4I 2020 Predictive Maintenance",
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1_score=f1,
                roc_auc=roc,
                pr_auc=pr,
                mcc=mcc,
                confusion_matrix_json=json.dumps(cm),
                evaluated_at=now,
            )
            session.add(me)

        # 8. Seed Initial Audit Log
        audit = AuditLog(
            user_id=admin_user.id,
            username=admin_user.username,
            action="SYSTEM_INIT_SEED",
            entity_type="System",
            entity_id="ALL",
            details="Database bootstrapped with verified roles, machines, sensor configurations, and baseline telemetry.",
            timestamp=now,
        )
        session.add(audit)

        session.commit()
        logger.info("Database seeding successfully completed.")
