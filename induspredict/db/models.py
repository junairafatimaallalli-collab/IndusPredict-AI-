"""
IndusPredict AI - SQLAlchemy Database Models.
Complete relational schema for industrial assets, sensors, predictions,
RUL prognostics, drift tracking, work orders, scenarios, and audit logs.
"""

from datetime import datetime
import json
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

from induspredict.core.utils import utc_now

Base = declarative_base()


class Role(Base):
    """User authorization roles (Administrator, Maintenance Engineer, Researcher, Viewer)."""
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    users = relationship("User", back_populates="role", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Role(id={self.id}, name='{self.name}')>"


class User(Base):
    """System operators and researchers."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    role_id = Column(Integer, ForeignKey("roles.id", ondelete="RESTRICT"), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    role = relationship("Role", back_populates="users")
    work_orders_created = relationship("WorkOrder", foreign_keys="WorkOrder.created_by_id", back_populates="creator")
    work_orders_assigned = relationship("WorkOrder", foreign_keys="WorkOrder.assigned_to_id", back_populates="assignee")
    audit_logs = relationship("AuditLog", back_populates="user")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', role='{self.role.name if self.role else self.role_id}')>"


class Machine(Base):
    """Industrial machines registered in the factory plant."""
    __tablename__ = "machines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_code = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(120), nullable=False)
    machine_type = Column(String(80), nullable=False, index=True)  # e.g., Compressor, Motor, Pump, Robot
    manufacturer = Column(String(100), nullable=False)
    model_number = Column(String(80), nullable=True)
    serial_number = Column(String(80), nullable=True)
    installation_date = Column(DateTime, nullable=False)
    operating_hours = Column(Float, default=0.0, nullable=False)
    location = Column(String(120), nullable=False)  # e.g., Plant 1 - Bay B
    criticality = Column(String(30), default="Medium", nullable=False, index=True)  # Low, Medium, High, Mission Critical
    status = Column(String(30), default="healthy", nullable=False, index=True)  # healthy, warning, critical, maintenance
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    components = relationship("MachineComponent", back_populates="machine", cascade="all, delete-orphan")
    sensors = relationship("Sensor", back_populates="machine", cascade="all, delete-orphan")
    readings = relationship("SensorReading", back_populates="machine", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="machine", cascade="all, delete-orphan")
    rul_estimates = relationship("RULEstimate", back_populates="machine", cascade="all, delete-orphan")
    anomaly_events = relationship("AnomalyEvent", back_populates="machine", cascade="all, delete-orphan")
    work_orders = relationship("WorkOrder", back_populates="machine", cascade="all, delete-orphan")
    maintenance_records = relationship("MaintenanceHistory", back_populates="machine", cascade="all, delete-orphan")
    simulation_scenarios = relationship("SimulationScenario", back_populates="machine", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Machine(id={self.id}, code='{self.machine_code}', name='{self.name}', status='{self.status}')>"


class MachineComponent(Base):
    """Sub-components / modules of an industrial machine (e.g. Bearings, Rotor, Impeller)."""
    __tablename__ = "machine_components"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(Integer, ForeignKey("machines.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    component_type = Column(String(80), nullable=False)
    part_number = Column(String(60), nullable=True)
    expected_lifespan_hours = Column(Float, default=20000.0, nullable=False)
    current_operating_hours = Column(Float, default=0.0, nullable=False)
    status = Column(String(30), default="healthy", nullable=False)  # healthy, worn, defective, replaced

    machine = relationship("Machine", back_populates="components")


class Sensor(Base):
    """Sensors attached to industrial machinery."""
    __tablename__ = "sensors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(Integer, ForeignKey("machines.id", ondelete="CASCADE"), nullable=False, index=True)
    sensor_code = Column(String(50), nullable=False)  # e.g., COMP-01-TEMP
    name = Column(String(100), nullable=False)  # Discharge Air Temperature
    sensor_type = Column(String(50), nullable=False, index=True)  # temperature, vibration, pressure, motor_speed, current, torque
    unit = Column(String(20), nullable=False)  # °C, mm/s, bar, RPM, A, Nm
    min_threshold = Column(Float, nullable=False)
    max_threshold = Column(Float, nullable=False)
    warning_threshold = Column(Float, nullable=False)
    critical_threshold = Column(Float, nullable=False)
    sample_rate_hz = Column(Float, default=1.0, nullable=False)
    status = Column(String(30), default="active", nullable=False)  # active, warning, offline, error

    machine = relationship("Machine", back_populates="sensors")
    readings = relationship("SensorReading", back_populates="sensor", cascade="all, delete-orphan")
    anomaly_events = relationship("AnomalyEvent", back_populates="sensor", cascade="all, delete-orphan")


class SensorReading(Base):
    """High-frequency sensor telemetry records."""
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(Integer, ForeignKey("machines.id", ondelete="CASCADE"), nullable=False, index=True)
    sensor_id = Column(Integer, ForeignKey("sensors.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, default=utc_now, nullable=False, index=True)
    value = Column(Float, nullable=False)
    source = Column(String(30), default="simulated", nullable=False)  # mqtt, opcua, uploaded, simulated
    is_anomaly = Column(Boolean, default=False, nullable=False, index=True)
    quality_flag = Column(String(20), default="GOOD", nullable=False)  # GOOD, UNCERTAIN, BAD

    machine = relationship("Machine", back_populates="readings")
    sensor = relationship("Sensor", back_populates="readings")

    __table_args__ = (
        Index("idx_sensor_time", "sensor_id", "timestamp"),
        Index("idx_machine_time", "machine_id", "timestamp"),
    )


class Prediction(Base):
    """Machine failure classification prediction records."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(Integer, ForeignKey("machines.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, default=utc_now, nullable=False, index=True)
    model_name = Column(String(60), nullable=False)
    model_version = Column(String(30), nullable=False)
    failure_probability = Column(Float, nullable=False)
    risk_category = Column(String(30), nullable=False, index=True)  # Normal, Warning, High Risk, Critical
    confidence_score = Column(Float, nullable=False)
    predicted_failure_mode = Column(String(80), nullable=True)  # Tool Wear, Heat Dissipation, Power, Overstrain, None
    contributing_features_json = Column(Text, nullable=True)  # JSON dictionary of top features
    status = Column(String(30), default="active", nullable=False)

    machine = relationship("Machine", back_populates="predictions")

    @property
    def contributing_features(self) -> Dict[str, Any]:
        if not self.contributing_features_json:
            return {}
        try:
            return json.loads(self.contributing_features_json)
        except Exception:
            return {}

    @contributing_features.setter
    def contributing_features(self, val: Dict[str, Any]) -> None:
        self.contributing_features_json = json.dumps(val)


class RULEstimate(Base):
    """Remaining Useful Life estimation records using PyTorch LSTM."""
    __tablename__ = "rul_estimates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(Integer, ForeignKey("machines.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, default=utc_now, nullable=False, index=True)
    model_name = Column(String(60), default="LSTM-RUL", nullable=False)
    model_version = Column(String(30), default="v1.0", nullable=False)
    predicted_rul_cycles = Column(Float, nullable=False)
    lower_bound_95 = Column(Float, nullable=False)
    upper_bound_95 = Column(Float, nullable=False)
    degradation_pct = Column(Float, nullable=False)  # 0.0 to 100.0%
    unit = Column(String(20), default="cycles", nullable=False)
    input_window_length = Column(Integer, default=30, nullable=False)

    machine = relationship("Machine", back_populates="rul_estimates")


class AnomalyEvent(Base):
    """Sensor anomaly triggers and exceedances."""
    __tablename__ = "anomaly_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(Integer, ForeignKey("machines.id", ondelete="CASCADE"), nullable=False, index=True)
    sensor_id = Column(Integer, ForeignKey("sensors.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, default=utc_now, nullable=False, index=True)
    anomaly_score = Column(Float, nullable=False)
    severity = Column(String(30), nullable=False)  # Warning, High, Critical
    description = Column(String(255), nullable=False)
    acknowledged = Column(Boolean, default=False, nullable=False)
    acknowledged_by = Column(String(80), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)

    machine = relationship("Machine", back_populates="anomaly_events")
    sensor = relationship("Sensor", back_populates="anomaly_events")


class ModelVersion(Base):
    """Registered Machine Learning & Deep Learning model artifacts."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(80), nullable=False, index=True)
    version = Column(String(30), nullable=False)
    model_type = Column(String(40), nullable=False)  # Classifier, RUL_Prognostic, Drift_Detector
    algorithm = Column(String(50), nullable=False)  # RandomForest, XGBoost, LightGBM, LSTM, GRU
    hyperparameters_json = Column(Text, nullable=True)
    artifact_path = Column(String(255), nullable=False)
    metrics_json = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    evaluations = relationship("ModelEvaluation", back_populates="model_version", cascade="all, delete-orphan")


class ModelEvaluation(Base):
    """Rigorous evaluation metrics on held-out industrial datasets."""
    __tablename__ = "model_evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False)
    dataset_name = Column(String(100), nullable=False)  # AI4I 2020, NASA C-MAPSS FD001, MetroPT-3
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    pr_auc = Column(Float, nullable=True)
    mcc = Column(Float, nullable=True)  # Matthews Correlation Coefficient
    mae = Column(Float, nullable=True)  # For RUL regression
    rmse = Column(Float, nullable=True)  # For RUL regression
    confusion_matrix_json = Column(Text, nullable=True)
    evaluated_at = Column(DateTime, default=utc_now, nullable=False)

    model_version = relationship("ModelVersion", back_populates="evaluations")


class DriftEvent(Base):
    """Concept drift and data distribution shift alerts."""
    __tablename__ = "drift_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(80), nullable=False, index=True)
    sensor_code = Column(String(60), nullable=True)
    timestamp = Column(DateTime, default=utc_now, nullable=False, index=True)
    detector = Column(String(40), nullable=False)  # ADWIN, Page-Hinkley, KS-Test
    statistic_value = Column(Float, nullable=False)
    p_value = Column(Float, nullable=True)
    threshold = Column(Float, nullable=False)
    status = Column(String(30), default="detected", nullable=False)  # detected, reviewed, retraining_queued
    recommendation = Column(String(255), nullable=False)


class WorkOrder(Base):
    """Adaptive maintenance tasks and work orders."""
    __tablename__ = "work_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(Integer, ForeignKey("machines.id", ondelete="CASCADE"), nullable=False, index=True)
    work_order_code = Column(String(50), unique=True, nullable=False, index=True)
    title = Column(String(150), nullable=False)
    description = Column(Text, nullable=False)
    priority = Column(String(30), default="Medium", nullable=False, index=True)  # Low, Medium, High, Emergency
    status = Column(String(30), default="Pending", nullable=False, index=True)  # Pending, Approved, In Progress, Completed, Cancelled
    created_by_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    assigned_to_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    scheduled_start = Column(DateTime, nullable=True)
    estimated_downtime_hours = Column(Float, default=2.0, nullable=False)
    actual_downtime_hours = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)

    machine = relationship("Machine", back_populates="work_orders")
    creator = relationship("User", foreign_keys=[created_by_id], back_populates="work_orders_created")
    assignee = relationship("User", foreign_keys=[assigned_to_id], back_populates="work_orders_assigned")
    history_entry = relationship("MaintenanceHistory", uselist=False, back_populates="work_order")


class MaintenanceHistory(Base):
    """Archival maintenance records."""
    __tablename__ = "maintenance_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(Integer, ForeignKey("machines.id", ondelete="CASCADE"), nullable=False, index=True)
    work_order_id = Column(Integer, ForeignKey("work_orders.id", ondelete="SET NULL"), nullable=True)
    service_date = Column(DateTime, default=utc_now, nullable=False, index=True)
    action_taken = Column(String(255), nullable=False)
    parts_replaced = Column(String(255), nullable=True)
    downtime_hours = Column(Float, default=0.0, nullable=False)
    technician_name = Column(String(100), nullable=False)
    remarks = Column(Text, nullable=True)

    machine = relationship("Machine", back_populates="maintenance_records")
    work_order = relationship("WorkOrder", back_populates="history_entry")


class SimulationScenario(Base):
    """Digital twin what-if simulation scenarios."""
    __tablename__ = "simulation_scenarios"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(Integer, ForeignKey("machines.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(120), nullable=False)
    created_by_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    baseline_state_json = Column(Text, nullable=False)
    simulated_parameters_json = Column(Text, nullable=False)
    predicted_risk_delta = Column(Float, nullable=False)
    estimated_health_delta = Column(Float, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    machine = relationship("Machine", back_populates="simulation_scenarios")


class AuditLog(Base):
    """Tamper-evident audit trail for industrial compliance."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    username = Column(String(50), nullable=True)
    action = Column(String(80), nullable=False, index=True)  # LOGIN, MACHINE_CREATE, MACHINE_DELETE, WORK_ORDER_APPROVE, etc.
    entity_type = Column(String(50), nullable=True)  # Machine, WorkOrder, User, Model
    entity_id = Column(String(50), nullable=True)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=utc_now, nullable=False, index=True)

    user = relationship("User", back_populates="audit_logs")
