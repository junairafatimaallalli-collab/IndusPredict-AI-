"""
IndusPredict AI - Adaptive Maintenance Planner & Drift Management Service.
Manages work order lifecycles (creation, engineer assignment, approval, execution, completion),
downtime modeling, maintenance history archival, and drift event tracking.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import joinedload, selectinload

from induspredict.core.logger import get_logger
from induspredict.core.utils import utc_now
from induspredict.db.models import (
    DriftEvent,
    Machine,
    MaintenanceHistory,
    Prediction,
    RULEstimate,
    User,
    WorkOrder,
)
from induspredict.db.session import get_db_session

logger = get_logger("services.maintenance")


class MaintenanceService:
    """Service providing adaptive maintenance workflows and drift alert tracking."""

    # Standard estimated downtime hours based on maintenance task classification
    DOWNTIME_ESTIMATES = {
        "Bearing Replacement / Overhaul": 4.5,
        "Thermal Dissipation / Cooler Flush": 2.5,
        "Mechanical Alignment & Torque Check": 1.5,
        "Lubrication & Seal Inspection": 0.75,
        "Emergency Vibration Inspection": 3.0,
        "Full Preventative Maintenance Overhaul": 6.0,
    }

    @classmethod
    def create_work_order(
        cls,
        machine_code: str,
        title: str,
        description: str,
        priority: str = "Medium",
        estimated_downtime_hours: float = 2.0,
        assigned_to_id: Optional[int] = None,
        created_by_id: Optional[int] = None,
        scheduled_start: Optional[datetime] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new maintenance work order."""
        with get_db_session() as session:
            machine = session.query(Machine).filter(Machine.machine_code == machine_code).first()
            if not machine:
                raise ValueError(f"Machine '{machine_code}' not found.")

            # Generate unique code: WO-YYYYMMDD-XXXX
            now = utc_now()
            count = session.query(WorkOrder).count() + 1
            wo_code = f"WO-{now.strftime('%Y%m%d')}-{count:04d}"

            wo = WorkOrder(
                machine_id=machine.id,
                work_order_code=wo_code,
                title=title,
                description=description,
                priority=priority,
                status="Pending",
                created_by_id=created_by_id,
                assigned_to_id=assigned_to_id,
                scheduled_start=scheduled_start or (now + timedelta(days=1)),
                estimated_downtime_hours=estimated_downtime_hours,
                created_at=now,
                notes=notes,
            )
            session.add(wo)
            session.commit()
            session.refresh(wo)

            logger.info(f"Created Work Order '{wo_code}' for machine {machine_code} (Priority: {priority})")
            return {
                "id": wo.id,
                "work_order_code": wo.work_order_code,
                "machine_code": machine_code,
                "title": wo.title,
                "priority": wo.priority,
                "status": wo.status,
                "estimated_downtime_hours": wo.estimated_downtime_hours,
                "created_at": wo.created_at.strftime("%Y-%m-%d %H:%M"),
            }

    @classmethod
    def get_all_work_orders(
        cls,
        status_filter: Optional[str] = None,
        priority_filter: Optional[str] = None,
        machine_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve list of work orders with filtering."""
        with get_db_session() as session:
            query = session.query(WorkOrder).options(
                joinedload(WorkOrder.machine),
                joinedload(WorkOrder.assignee),
                joinedload(WorkOrder.creator),
            )
            if status_filter and status_filter != "All":
                query = query.filter(WorkOrder.status == status_filter)
            if priority_filter and priority_filter != "All":
                query = query.filter(WorkOrder.priority == priority_filter)
            if machine_filter and machine_filter != "All":
                query = query.join(Machine).filter(Machine.machine_code == machine_filter)

            orders = query.order_by(WorkOrder.created_at.desc()).all()
            return [
                {
                    "id": wo.id,
                    "code": wo.work_order_code,
                    "machine_code": wo.machine.machine_code if wo.machine else "N/A",
                    "machine_name": wo.machine.name if wo.machine else "N/A",
                    "title": wo.title,
                    "description": wo.description,
                    "priority": wo.priority,
                    "status": wo.status,
                    "assignee": wo.assignee.full_name if wo.assignee else "Unassigned",
                    "scheduled_start": wo.scheduled_start.strftime("%Y-%m-%d") if wo.scheduled_start else "TBD",
                    "estimated_downtime_hours": wo.estimated_downtime_hours,
                    "actual_downtime_hours": wo.actual_downtime_hours,
                    "created_at": wo.created_at.strftime("%Y-%m-%d %H:%M"),
                    "completed_at": wo.completed_at.strftime("%Y-%m-%d %H:%M") if wo.completed_at else None,
                }
                for wo in orders
            ]

    @classmethod
    def approve_work_order(cls, work_order_id: int) -> bool:
        """Approve a pending maintenance work order."""
        with get_db_session() as session:
            wo = session.query(WorkOrder).filter(WorkOrder.id == work_order_id).first()
            if not wo:
                return False
            wo.status = "Approved"
            session.commit()
            logger.info(f"Work Order {wo.work_order_code} approved.")
            return True

    @classmethod
    def start_work_order(cls, work_order_id: int) -> bool:
        """Transition work order to In Progress."""
        with get_db_session() as session:
            wo = session.query(WorkOrder).filter(WorkOrder.id == work_order_id).first()
            if not wo:
                return False
            wo.status = "In Progress"
            session.commit()
            return True

    @classmethod
    def complete_work_order(
        cls,
        work_order_id: int,
        actual_downtime_hours: float,
        action_taken: str,
        parts_replaced: Optional[str] = None,
        technician_name: str = "Lead Maintenance Engineer",
        remarks: Optional[str] = None,
    ) -> bool:
        """Mark work order as Completed and archive into MaintenanceHistory."""
        now = utc_now()
        with get_db_session() as session:
            wo = session.query(WorkOrder).filter(WorkOrder.id == work_order_id).first()
            if not wo:
                return False

            wo.status = "Completed"
            wo.actual_downtime_hours = actual_downtime_hours
            wo.completed_at = now

            # Create history record
            history = MaintenanceHistory(
                machine_id=wo.machine_id,
                work_order_id=wo.id,
                service_date=now,
                action_taken=action_taken,
                parts_replaced=parts_replaced,
                downtime_hours=actual_downtime_hours,
                technician_name=technician_name,
                remarks=remarks or wo.description,
            )
            session.add(history)
            session.commit()

            logger.info(f"Work Order {wo.work_order_code} completed. Archived in maintenance history.")
            return True

    @classmethod
    def generate_adaptive_suggestions(cls) -> List[Dict[str, Any]]:
        """
        Scan machine registry, latest predictions, and RUL prognostics
        to automatically synthesize risk-prioritized maintenance recommendations.
        """
        suggestions = []
        with get_db_session() as session:
            machines = session.query(Machine).options(
                selectinload(Machine.predictions),
                selectinload(Machine.rul_estimates),
                selectinload(Machine.components),
            ).all()

            for m in machines:
                latest_pred = m.predictions[-1] if m.predictions else None
                latest_rul = m.rul_estimates[-1] if m.rul_estimates else None

                # Rule 1: High/Critical Risk
                if latest_pred and latest_pred.failure_probability >= 0.40:
                    priority = "Emergency" if m.criticality == "Critical" or latest_pred.failure_probability >= 0.70 else "High"
                    task_name = f"Proactive Inspection: {latest_pred.predicted_failure_mode}"
                    downtime = cls.DOWNTIME_ESTIMATES.get("Bearing Replacement / Overhaul", 4.0)
                    suggestions.append({
                        "machine_code": m.machine_code,
                        "machine_name": m.name,
                        "title": task_name,
                        "description": f"AI model detected {latest_pred.failure_probability*100:.1f}% failure risk ({latest_pred.risk_category}). Diagnosed mode: {latest_pred.predicted_failure_mode}.",
                        "priority": priority,
                        "suggested_downtime_hours": downtime,
                        "trigger": "Failure Risk Probability",
                    })

                # Rule 2: Low RUL (< 50 cycles)
                elif latest_rul and latest_rul.predicted_rul_cycles < 50.0:
                    priority = "High" if latest_rul.predicted_rul_cycles < 25.0 else "Medium"
                    suggestions.append({
                        "machine_code": m.machine_code,
                        "machine_name": m.name,
                        "title": "Degradation Overhaul (Low Remaining Useful Life)",
                        "description": f"PyTorch LSTM estimated Remaining Useful Life at {latest_rul.predicted_rul_cycles:.1f} cycles with {latest_rul.degradation_pct:.1f}% wear.",
                        "priority": priority,
                        "suggested_downtime_hours": 4.5,
                        "trigger": "RUL Prognostics Countdown",
                    })

                # Rule 3: High component operating hours
                else:
                    for comp in m.components:
                        if comp.current_operating_hours >= comp.expected_lifespan_hours * 0.85:
                            suggestions.append({
                                "machine_code": m.machine_code,
                                "machine_name": m.name,
                                "title": f"Component Wear Overhaul: {comp.name}",
                                "description": f"Component has logged {comp.current_operating_hours:.0f} hours ({comp.current_operating_hours/comp.expected_lifespan_hours*100:.1f}% of rated lifespan).",
                                "priority": "Medium",
                                "suggested_downtime_hours": 2.0,
                                "trigger": "Component Lifespan Exceedance",
                            })
                            break

        return suggestions

    @classmethod
    def get_maintenance_history(cls, machine_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve archival maintenance history records."""
        with get_db_session() as session:
            query = session.query(MaintenanceHistory).options(joinedload(MaintenanceHistory.machine))
            if machine_code:
                query = query.join(Machine).filter(Machine.machine_code == machine_code)

            records = query.order_by(MaintenanceHistory.service_date.desc()).all()
            return [
                {
                    "id": rec.id,
                    "machine_code": rec.machine.machine_code if rec.machine else "N/A",
                    "machine_name": rec.machine.name if rec.machine else "N/A",
                    "service_date": rec.service_date.strftime("%Y-%m-%d"),
                    "action_taken": rec.action_taken,
                    "parts_replaced": rec.parts_replaced or "-",
                    "downtime_hours": rec.downtime_hours,
                    "technician": rec.technician_name,
                    "remarks": rec.remarks or "-",
                }
                for rec in records
            ]

    # --- Drift Event Management ---

    @classmethod
    def record_drift_event(
        cls,
        model_name: str,
        sensor_code: str,
        detector: str,
        statistic_value: float,
        threshold: float,
        recommendation: str,
    ) -> Dict[str, Any]:
        """Record a concept drift event alert in SQLite."""
        with get_db_session() as session:
            event = DriftEvent(
                model_name=model_name,
                sensor_code=sensor_code,
                timestamp=utc_now(),
                detector=detector,
                statistic_value=round(statistic_value, 4),
                threshold=round(threshold, 4),
                status="detected",
                recommendation=recommendation,
            )
            session.add(event)
            session.commit()
            session.refresh(event)

            logger.warning(f"Recorded Drift Event: {detector} on {sensor_code} ({recommendation})")
            return {
                "id": event.id,
                "model_name": event.model_name,
                "sensor_code": event.sensor_code,
                "detector": event.detector,
                "statistic_value": event.statistic_value,
                "threshold": event.threshold,
                "status": event.status,
                "recommendation": event.recommendation,
                "timestamp": event.timestamp.strftime("%Y-%m-%d %H:%M"),
            }

    @classmethod
    def get_drift_events(cls, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve list of drift events."""
        with get_db_session() as session:
            query = session.query(DriftEvent)
            if status_filter and status_filter != "All":
                query = query.filter(DriftEvent.status == status_filter)

            events = query.order_by(DriftEvent.timestamp.desc()).all()
            return [
                {
                    "id": e.id,
                    "model_name": e.model_name,
                    "sensor_code": e.sensor_code or "Fleet Telemetry",
                    "detector": e.detector,
                    "statistic": e.statistic_value,
                    "threshold": e.threshold,
                    "status": e.status,
                    "recommendation": e.recommendation,
                    "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M"),
                }
                for e in events
            ]

    @classmethod
    def review_drift_event(cls, event_id: int, new_status: str = "reviewed") -> bool:
        """Mark drift event as reviewed or queued for retraining with human sign-off."""
        with get_db_session() as session:
            event = session.query(DriftEvent).filter(DriftEvent.id == event_id).first()
            if not event:
                return False
            event.status = new_status
            session.commit()
            return True
