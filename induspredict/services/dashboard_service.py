"""
IndusPredict AI - Dashboard Aggregation Service.
Computes real-time dynamic KPIs, health trends, failure risk distributions,
and active alert summaries directly from database queries.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from sqlalchemy import case, func
from sqlalchemy.orm import selectinload

from induspredict.core.utils import utc_now
from induspredict.core.logger import get_logger
from induspredict.db.models import (
    AnomalyEvent,
    Machine,
    MaintenanceHistory,
    Prediction,
    RULEstimate,
    Sensor,
    SensorReading,
    WorkOrder,
)
from induspredict.db.session import get_db_session

logger = get_logger("services.dashboard")


class DashboardService:
    """Service providing aggregated dashboard metrics and chart payloads from SQLite."""

    @staticmethod
    def get_summary_kpis(machine_id_filter: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute actual KPI counts dynamically from the database.
        Returns:
          - total_machines
          - healthy_machines
          - observation_machines
          - high_risk_machines
          - critical_alerts
          - pending_maintenance_tasks
        """
        with get_db_session() as session:
            # Machine counts
            m_query = session.query(Machine)
            if machine_id_filter:
                m_query = m_query.filter(Machine.id == machine_id_filter)

            total_machines = m_query.count()
            healthy_machines = m_query.filter(Machine.status == "healthy").count()
            observation_machines = m_query.filter(Machine.status == "warning").count()
            high_risk_machines = m_query.filter(Machine.status.in_(["critical", "maintenance"])).count()

            # Active critical alerts
            a_query = session.query(AnomalyEvent).filter(AnomalyEvent.acknowledged.is_(False))
            if machine_id_filter:
                a_query = a_query.filter(AnomalyEvent.machine_id == machine_id_filter)
            critical_alerts = a_query.count()

            # Pending maintenance work orders
            w_query = session.query(WorkOrder).filter(WorkOrder.status.in_(["Pending", "In Progress"]))
            if machine_id_filter:
                w_query = w_query.filter(WorkOrder.machine_id == machine_id_filter)
            pending_maintenance = w_query.count()

            # Active sensors count
            s_query = session.query(Sensor)
            if machine_id_filter:
                s_query = s_query.filter(Sensor.machine_id == machine_id_filter)
            total_sensors = s_query.count()
            active_sensors = s_query.filter(Sensor.status == "active").count()

            return {
                "total_machines": total_machines,
                "healthy_machines": healthy_machines,
                "observation_machines": observation_machines,
                "high_risk_machines": high_risk_machines,
                "critical_alerts": critical_alerts,
                "pending_tasks": pending_maintenance,
                "total_sensors": total_sensors,
                "active_sensors": active_sensors,
            }

    @staticmethod
    def get_failure_risk_distribution(machine_id_filter: Optional[int] = None) -> Dict[str, int]:
        """
        Calculate failure risk categorization from the latest prediction of each machine.
        """
        with get_db_session() as session:
            query = session.query(Machine).options(selectinload(Machine.predictions))
            if machine_id_filter:
                query = query.filter(Machine.id == machine_id_filter)

            machines = query.all()
            distribution = {"Normal": 0, "Warning": 0, "High Risk": 0, "Critical": 0}

            for m in machines:
                if m.predictions:
                    latest = sorted(m.predictions, key=lambda p: p.timestamp, reverse=True)[0]
                    cat = latest.risk_category
                    if cat in distribution:
                        distribution[cat] += 1
                    else:
                        distribution["Normal"] += 1
                else:
                    distribution["Normal"] += 1

            return distribution

    @staticmethod
    def get_health_trends(machine_id_filter: Optional[int] = None, hours: int = 12) -> Dict[str, Any]:
        """
        Compute chronological average sensor health and reading trends from actual telemetry.
        """
        with get_db_session() as session:
            cutoff = utc_now() - timedelta(hours=hours)

            query = session.query(
                SensorReading.timestamp,
                func.avg(SensorReading.value).label("avg_value"),
                func.sum(case((SensorReading.is_anomaly.is_(True), 1), else_=0)).label("anomaly_count"),
            ).filter(SensorReading.timestamp >= cutoff)

            if machine_id_filter:
                query = query.filter(SensorReading.machine_id == machine_id_filter)

            # Group into 30-minute intervals
            records = query.group_by(SensorReading.timestamp).order_by(SensorReading.timestamp.asc()).all()

            timestamps = [r[0].strftime("%H:%M") for r in records]
            avg_values = [round(float(r[1]), 2) if r[1] is not None else 0.0 for r in records]
            anomaly_counts = [int(r[2]) for r in records]

            # If no dense readings, provide aggregated points from recent predictions
            if not timestamps:
                p_query = session.query(Prediction).filter(Prediction.timestamp >= cutoff)
                if machine_id_filter:
                    p_query = p_query.filter(Prediction.machine_id == machine_id_filter)
                preds = p_query.order_by(Prediction.timestamp.asc()).all()
                timestamps = [p.timestamp.strftime("%H:%M") for p in preds]
                avg_values = [round((1.0 - p.failure_probability) * 100.0, 1) for p in preds]
                anomaly_counts = [1 if p.failure_probability > 0.5 else 0 for p in preds]

            return {
                "timestamps": timestamps,
                "values": avg_values,
                "anomalies": anomaly_counts,
            }

    @staticmethod
    def get_rul_overview(limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve latest RUL prognostics overview for machines."""
        with get_db_session() as session:
            machines = session.query(Machine).options(selectinload(Machine.rul_estimates)).all()
            overview = []
            for m in machines:
                if m.rul_estimates:
                    latest = sorted(m.rul_estimates, key=lambda r: r.timestamp, reverse=True)[0]
                    overview.append(
                        {
                            "machine_code": m.machine_code,
                            "machine_name": m.name,
                            "predicted_rul": round(latest.predicted_rul_cycles, 1),
                            "lower_bound": round(latest.lower_bound_95, 1),
                            "upper_bound": round(latest.upper_bound_95, 1),
                            "degradation_pct": round(latest.degradation_pct, 1),
                            "unit": latest.unit,
                            "timestamp": latest.timestamp.strftime("%Y-%m-%d %H:%M"),
                        }
                    )
            return sorted(overview, key=lambda x: x["predicted_rul"])[:limit]

    @staticmethod
    def get_recent_alerts(limit: int = 8, machine_id_filter: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch latest anomaly events and critical status changes."""
        with get_db_session() as session:
            query = (
                session.query(AnomalyEvent)
                .options(selectinload(AnomalyEvent.machine), selectinload(AnomalyEvent.sensor))
                .order_by(AnomalyEvent.timestamp.desc())
            )
            if machine_id_filter:
                query = query.filter(AnomalyEvent.machine_id == machine_id_filter)

            events = query.limit(limit).all()
            results = []
            for e in events:
                results.append(
                    {
                        "id": e.id,
                        "machine_code": e.machine.machine_code if e.machine else "N/A",
                        "machine_name": e.machine.name if e.machine else "N/A",
                        "sensor_name": e.sensor.name if e.sensor else "N/A",
                        "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M"),
                        "severity": e.severity,
                        "description": e.description,
                        "acknowledged": e.acknowledged,
                    }
                )
            return results

    @staticmethod
    def get_recent_predictions(limit: int = 8, machine_id_filter: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch latest predictions across machines."""
        with get_db_session() as session:
            query = (
                session.query(Prediction)
                .options(selectinload(Prediction.machine))
                .order_by(Prediction.timestamp.desc())
            )
            if machine_id_filter:
                query = query.filter(Prediction.machine_id == machine_id_filter)

            preds = query.limit(limit).all()
            return [
                {
                    "id": p.id,
                    "machine_code": p.machine.machine_code if p.machine else "N/A",
                    "machine_name": p.machine.name if p.machine else "N/A",
                    "timestamp": p.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "model_name": p.model_name,
                    "failure_prob": round(p.failure_probability * 100, 1),
                    "risk_category": p.risk_category,
                    "confidence": round(p.confidence_score * 100, 1),
                    "failure_mode": p.predicted_failure_mode or "None",
                }
                for p in preds
            ]
