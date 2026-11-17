"""
IndusPredict AI - Machine Management Service.
Provides verified CRUD operations, component/sensor configurations,
and SQLite persistence with validation and audit logging.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import joinedload, selectinload

from induspredict.core.utils import utc_now
from induspredict.core.config import (
    ALL_CRITICALITY_LEVELS,
    CRITICALITY_MEDIUM,
    SENSOR_SPECS,
)
from induspredict.core.logger import get_logger
from induspredict.core.security import (
    PERM_MACHINE_DELETE,
    PERM_MACHINE_EDIT,
    PERM_MACHINE_VIEW,
    UserSession,
)
from induspredict.db.models import (
    AuditLog,
    Machine,
    MachineComponent,
    MaintenanceHistory,
    Prediction,
    Sensor,
    SensorReading,
    WorkOrder,
)
from induspredict.db.session import get_db_session

logger = get_logger("services.machine")


class MachineService:
    """Service managing industrial machines, installed components, and sensor hardware."""

    @staticmethod
    def get_all_machines(
        type_filter: Optional[str] = None,
        criticality_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve list of registered machines with summary statistics."""
        with get_db_session() as session:
            query = session.query(Machine).options(
                selectinload(Machine.components),
                selectinload(Machine.sensors),
                selectinload(Machine.predictions),
                selectinload(Machine.work_orders),
            )

            if type_filter and type_filter != "All":
                query = query.filter(Machine.machine_type == type_filter)
            if criticality_filter and criticality_filter != "All":
                query = query.filter(Machine.criticality == criticality_filter)
            if status_filter and status_filter != "All":
                query = query.filter(Machine.status == status_filter)
            if search_query:
                term = f"%{search_query.strip()}%"
                query = query.filter(
                    (Machine.name.ilike(term))
                    | (Machine.machine_code.ilike(term))
                    | (Machine.location.ilike(term))
                    | (Machine.manufacturer.ilike(term))
                )

            machines = query.order_by(Machine.id.asc()).all()

            results = []
            for m in machines:
                # Latest prediction
                latest_pred = None
                if m.predictions:
                    latest_pred = sorted(m.predictions, key=lambda x: x.timestamp, reverse=True)[0]

                # Pending work orders count
                pending_orders = sum(1 for wo in m.work_orders if wo.status in ("Pending", "In Progress"))

                results.append(
                    {
                        "id": m.id,
                        "code": m.machine_code,
                        "name": m.name,
                        "type": m.machine_type,
                        "manufacturer": m.manufacturer,
                        "model": m.model_number or "N/A",
                        "serial": m.serial_number or "N/A",
                        "installation_date": m.installation_date.strftime("%Y-%m-%d"),
                        "operating_hours": m.operating_hours,
                        "location": m.location,
                        "criticality": m.criticality,
                        "status": m.status,
                        "components_count": len(m.components),
                        "sensors_count": len(m.sensors),
                        "latest_failure_prob": latest_pred.failure_probability if latest_pred else 0.0,
                        "latest_risk_category": latest_pred.risk_category if latest_pred else "Normal",
                        "pending_work_orders": pending_orders,
                        "created_at": m.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

            return results

    @staticmethod
    def get_machine_details_by_code(machine_code: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete machine details by machine code."""
        with get_db_session() as session:
            m = session.query(Machine).filter(Machine.machine_code == machine_code).first()
            if not m:
                return None
            return MachineService.get_machine_details(m.id)

    @staticmethod
    def get_machine_details(machine_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve complete machine details including components, sensors, and recent history."""
        with get_db_session() as session:
            m = (
                session.query(Machine)
                .options(
                    selectinload(Machine.components),
                    selectinload(Machine.sensors),
                    selectinload(Machine.maintenance_records),
                    selectinload(Machine.work_orders),
                )
                .filter(Machine.id == machine_id)
                .first()
            )

            if not m:
                return None

            return {
                "id": m.id,
                "code": m.machine_code,
                "name": m.name,
                "type": m.machine_type,
                "manufacturer": m.manufacturer,
                "model": m.model_number or "",
                "serial": m.serial_number or "",
                "installation_date": m.installation_date.strftime("%Y-%m-%d"),
                "operating_hours": m.operating_hours,
                "location": m.location,
                "criticality": m.criticality,
                "status": m.status,
                "description": m.description or "",
                "components": [
                    {
                        "id": c.id,
                        "name": c.name,
                        "type": c.component_type,
                        "part_number": c.part_number or "",
                        "expected_lifespan_hours": c.expected_lifespan_hours,
                        "current_operating_hours": c.current_operating_hours,
                        "status": c.status,
                        "wear_pct": round((c.current_operating_hours / max(1.0, c.expected_lifespan_hours)) * 100, 1),
                    }
                    for c in m.components
                ],
                "sensors": [
                    {
                        "id": s.id,
                        "code": s.sensor_code,
                        "name": s.name,
                        "type": s.sensor_type,
                        "unit": s.unit,
                        "min_threshold": s.min_threshold,
                        "max_threshold": s.max_threshold,
                        "warning_threshold": s.warning_threshold,
                        "critical_threshold": s.critical_threshold,
                        "status": s.status,
                    }
                    for s in m.sensors
                ],
                "maintenance_records": [
                    {
                        "id": h.id,
                        "date": h.service_date.strftime("%Y-%m-%d"),
                        "action": h.action_taken,
                        "parts": h.parts_replaced or "None",
                        "downtime_hours": h.downtime_hours,
                        "technician": h.technician_name,
                    }
                    for h in sorted(m.maintenance_records, key=lambda x: x.service_date, reverse=True)
                ],
                "work_orders": [
                    {
                        "id": w.id,
                        "code": w.work_order_code,
                        "title": w.title,
                        "priority": w.priority,
                        "status": w.status,
                        "downtime": w.estimated_downtime_hours,
                    }
                    for w in m.work_orders
                ],
            }

    @staticmethod
    def create_machine(data: Dict[str, Any], user_session: Optional[UserSession] = None) -> Dict[str, Any]:
        """Validate and create a new industrial machine with default sensors."""
        # Validation
        code = str(data.get("code", "")).strip().upper()
        name = str(data.get("name", "")).strip()
        m_type = str(data.get("type", "")).strip()
        manufacturer = str(data.get("manufacturer", "")).strip()
        location = str(data.get("location", "")).strip()
        criticality = str(data.get("criticality", CRITICALITY_MEDIUM)).strip()

        if not code or len(code) < 2:
            return {"success": False, "error": "Machine code is required (min 2 characters)."}
        if not name or len(name) < 3:
            return {"success": False, "error": "Machine name is required (min 3 characters)."}
        if not m_type:
            return {"success": False, "error": "Machine type is required."}
        if not manufacturer:
            return {"success": False, "error": "Manufacturer is required."}
        if not location:
            return {"success": False, "error": "Location is required."}
        if criticality not in ALL_CRITICALITY_LEVELS:
            criticality = CRITICALITY_MEDIUM

        try:
            op_hours = float(data.get("operating_hours", 0.0))
            if op_hours < 0:
                return {"success": False, "error": "Operating hours must be non-negative."}
        except ValueError:
            return {"success": False, "error": "Operating hours must be a valid number."}

        install_date = data.get("installation_date")
        if isinstance(install_date, str):
            try:
                install_date = datetime.strptime(install_date, "%Y-%m-%d")
            except ValueError:
                install_date = utc_now()
        elif not isinstance(install_date, datetime):
            install_date = utc_now()

        with get_db_session() as session:
            # Check duplicate code
            if session.query(Machine).filter(Machine.machine_code == code).first():
                return {"success": False, "error": f"Machine code '{code}' already exists."}

            new_m = Machine(
                machine_code=code,
                name=name,
                machine_type=m_type,
                manufacturer=manufacturer,
                model_number=data.get("model", ""),
                serial_number=data.get("serial", ""),
                installation_date=install_date,
                operating_hours=op_hours,
                location=location,
                criticality=criticality,
                status="healthy",
                description=data.get("description", ""),
                created_at=utc_now(),
            )
            session.add(new_m)
            session.flush()

            # Attach default standard sensors for this machine
            default_sensors = [
                ("TEMP", "temperature"),
                ("VIB", "vibration"),
                ("PRESS", "pressure"),
                ("SPEED", "motor_speed"),
            ]
            for suffix, stype in default_sensors:
                spec = SENSOR_SPECS.get(stype, {"name": stype.title(), "unit": "", "min": 0, "max": 100, "warning": 75, "critical": 90})
                sensor = Sensor(
                    machine_id=new_m.id,
                    sensor_code=f"{code}-{suffix}",
                    name=f"{name} {spec['name']}",
                    sensor_type=stype,
                    unit=spec["unit"],
                    min_threshold=spec["min"],
                    max_threshold=spec["max"],
                    warning_threshold=spec["warning"],
                    critical_threshold=spec["critical"],
                    status="active",
                )
                session.add(sensor)

            # Audit log
            audit = AuditLog(
                user_id=user_session.user_id if user_session else None,
                username=user_session.username if user_session else "System",
                action="MACHINE_CREATE",
                entity_type="Machine",
                entity_id=str(new_m.id),
                details=f"Created new machine '{code}' ({name}) with default sensors.",
                timestamp=utc_now(),
            )
            session.add(audit)
            session.commit()

            logger.info(f"Machine '{code}' created successfully with ID {new_m.id}.")
            return {"success": True, "machine_id": new_m.id, "code": code}

    @staticmethod
    def update_machine(machine_id: int, data: Dict[str, Any], user_session: Optional[UserSession] = None) -> Dict[str, Any]:
        """Update existing machine properties with validation."""
        with get_db_session() as session:
            m = session.query(Machine).filter(Machine.id == machine_id).first()
            if not m:
                return {"success": False, "error": f"Machine with ID {machine_id} not found."}

            if "name" in data and str(data["name"]).strip():
                m.name = str(data["name"]).strip()
            if "type" in data and str(data["type"]).strip():
                m.machine_type = str(data["type"]).strip()
            if "manufacturer" in data and str(data["manufacturer"]).strip():
                m.manufacturer = str(data["manufacturer"]).strip()
            if "model" in data:
                m.model_number = str(data["model"]).strip()
            if "serial" in data:
                m.serial_number = str(data["serial"]).strip()
            if "location" in data and str(data["location"]).strip():
                m.location = str(data["location"]).strip()
            if "criticality" in data and data["criticality"] in ALL_CRITICALITY_LEVELS:
                m.criticality = data["criticality"]
            if "status" in data and data["status"] in ("healthy", "warning", "critical", "maintenance"):
                m.status = data["status"]
            if "description" in data:
                m.description = str(data["description"]).strip()

            if "operating_hours" in data:
                try:
                    val = float(data["operating_hours"])
                    if val >= 0:
                        m.operating_hours = val
                except ValueError:
                    pass

            m.updated_at = utc_now()

            # Audit log
            audit = AuditLog(
                user_id=user_session.user_id if user_session else None,
                username=user_session.username if user_session else "System",
                action="MACHINE_UPDATE",
                entity_type="Machine",
                entity_id=str(m.id),
                details=f"Updated machine details for '{m.machine_code}'.",
                timestamp=utc_now(),
            )
            session.add(audit)
            session.commit()

            logger.info(f"Machine ID {machine_id} ({m.machine_code}) updated.")
            return {"success": True, "machine_id": m.id}

    @staticmethod
    def delete_machine(machine_id: int, user_session: Optional[UserSession] = None) -> Dict[str, Any]:
        """Delete machine with confirmation and cascade removal of sensors/telemetry."""
        with get_db_session() as session:
            m = session.query(Machine).filter(Machine.id == machine_id).first()
            if not m:
                return {"success": False, "error": f"Machine with ID {machine_id} not found."}

            m_code = m.machine_code
            m_name = m.name

            session.delete(m)

            # Audit log
            audit = AuditLog(
                user_id=user_session.user_id if user_session else None,
                username=user_session.username if user_session else "System",
                action="MACHINE_DELETE",
                entity_type="Machine",
                entity_id=str(machine_id),
                details=f"Permanently removed machine '{m_code}' ({m_name}) and associated records.",
                timestamp=utc_now(),
            )
            session.add(audit)
            session.commit()

            logger.info(f"Machine '{m_code}' (ID {machine_id}) deleted.")
            return {"success": True, "deleted_code": m_code}

    @staticmethod
    def add_component(machine_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add sub-component to machine."""
        name = str(data.get("name", "")).strip()
        c_type = str(data.get("type", "")).strip()
        if not name or not c_type:
            return {"success": False, "error": "Component name and type are required."}

        with get_db_session() as session:
            m = session.query(Machine).filter(Machine.id == machine_id).first()
            if not m:
                return {"success": False, "error": "Machine not found."}

            comp = MachineComponent(
                machine_id=m.id,
                name=name,
                component_type=c_type,
                part_number=str(data.get("part_number", "")).strip(),
                expected_lifespan_hours=float(data.get("lifespan", 20000.0)),
                current_operating_hours=float(data.get("hours", 0.0)),
                status=data.get("status", "healthy"),
            )
            session.add(comp)
            session.commit()
            return {"success": True, "component_id": comp.id}
