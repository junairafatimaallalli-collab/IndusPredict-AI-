"""
IndusPredict AI - Report Aggregation Service.
Queries SQLite database for machine health, predictions, RUL trajectories,
telemetry, work orders, and model evaluations; prepares datasets for PDF,
Excel, and CSV export.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import joinedload, selectinload

from induspredict.core.logger import get_logger
from induspredict.db.models import (
    DriftEvent,
    Machine,
    ModelEvaluation,
    ModelVersion,
    Prediction,
    RULEstimate,
    Sensor,
    SensorReading,
    WorkOrder,
)
from induspredict.db.session import get_db_session
from induspredict.reports.generator import ReportGenerator

logger = get_logger("services.report")


class ReportService:
    """Service providing tabular data aggregation and report export coordination."""

    @classmethod
    def get_report_data(
        cls,
        report_type: str,
        machine_code: str = "All",
    ) -> Dict[str, Any]:
        """
        Query database and assemble summary KPIs, column definitions,
        and tabular row records for the requested report category.
        """
        with get_db_session() as session:
            if report_type == "Machine Health & Asset Inventory":
                query = session.query(Machine).options(
                    selectinload(Machine.predictions),
                    selectinload(Machine.rul_estimates),
                    selectinload(Machine.components),
                )
                if machine_code != "All":
                    query = query.filter(Machine.machine_code == machine_code)
                machines = query.all()

                kpis = {
                    "total_machines": len(machines),
                    "healthy_count": sum(1 for m in machines if m.status == "healthy"),
                    "under_observation": sum(1 for m in machines if m.status == "warning"),
                    "high_risk_assets": sum(1 for m in machines if m.status == "critical"),
                }
                columns = ["Code", "Asset Name", "Type", "Manufacturer", "Hours Logged", "Status", "Criticality"]
                rows = [
                    [m.machine_code, m.name, m.machine_type, m.manufacturer, f"{m.operating_hours:.0f} hrs", m.status.title(), m.criticality]
                    for m in machines
                ]
                dict_rows = [
                    {
                        "machine_code": m.machine_code,
                        "name": m.name,
                        "type": m.machine_type,
                        "manufacturer": m.manufacturer,
                        "operating_hours": m.operating_hours,
                        "status": m.status,
                        "criticality": m.criticality,
                    }
                    for m in machines
                ]
                recs = [
                    "Perform routine lubrication on all assets exceeding 5,000 operational hours.",
                    "Review critical CNC and compressor bearings for localized fatigue wear.",
                ]

            elif report_type == "Failure Risk & Anomaly Predictions":
                query = session.query(Prediction).options(joinedload(Prediction.machine))
                if machine_code != "All":
                    query = query.join(Machine).filter(Machine.machine_code == machine_code)
                preds = query.order_by(Prediction.timestamp.desc()).limit(100).all()

                high_risk_cnt = sum(1 for p in preds if p.risk_category in ["High Risk", "Critical"])
                kpis = {
                    "total_predictions": len(preds),
                    "high_risk_events": high_risk_cnt,
                    "avg_probability": f"{(sum(p.failure_probability for p in preds) / max(1, len(preds))) * 100:.1f}%",
                    "dominant_model": preds[0].model_name if preds else "Random Forest",
                }
                columns = ["Timestamp", "Machine", "Model", "Failure Prob", "Risk Level", "Diagnosed Mode"]
                rows = [
                    [
                        p.timestamp.strftime("%Y-%m-%d %H:%M"),
                        p.machine.machine_code if p.machine else "N/A",
                        p.model_name,
                        f"{p.failure_probability * 100:.1f}%",
                        p.risk_category,
                        p.predicted_failure_mode or "Nominal",
                    ]
                    for p in preds
                ]
                dict_rows = [
                    {
                        "timestamp": p.timestamp.strftime("%Y-%m-%d %H:%M"),
                        "machine_code": p.machine.machine_code if p.machine else "N/A",
                        "model": p.model_name,
                        "failure_probability": p.failure_probability,
                        "risk_category": p.risk_category,
                        "failure_mode": p.predicted_failure_mode,
                    }
                    for p in preds
                ]
                recs = [
                    "Schedule immediate mechanical overhaul for assets classified as High Risk or Critical.",
                    "Verify thermal sensor calibration on units flagging Heat Dissipation Failure (HDF).",
                ]

            elif report_type == "Remaining Useful Life (RUL) Prognostics":
                query = session.query(RULEstimate).options(joinedload(RULEstimate.machine))
                if machine_code != "All":
                    query = query.join(Machine).filter(Machine.machine_code == machine_code)
                ruls = query.order_by(RULEstimate.timestamp.desc()).limit(100).all()

                critical_rul_cnt = sum(1 for r in ruls if r.predicted_rul_cycles < 40.0)
                kpis = {
                    "total_rul_records": len(ruls),
                    "critical_rul_units": critical_rul_cnt,
                    "mean_rul_cycles": f"{(sum(r.predicted_rul_cycles for r in ruls) / max(1, len(ruls))):.1f} cycles",
                    "deep_model": "PyTorch LSTM (FD001)",
                }
                columns = ["Timestamp", "Machine", "Model", "Predicted RUL", "95% Confidence Bounds", "Degradation Wear"]
                rows = [
                    [
                        r.timestamp.strftime("%Y-%m-%d %H:%M"),
                        r.machine.machine_code if r.machine else "N/A",
                        r.model_name,
                        f"{r.predicted_rul_cycles:.1f} cycles",
                        f"[{r.lower_bound_95:.1f} - {r.upper_bound_95:.1f}]",
                        f"{r.degradation_pct:.1f}%",
                    ]
                    for r in ruls
                ]
                dict_rows = [
                    {
                        "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M"),
                        "machine_code": r.machine.machine_code if r.machine else "N/A",
                        "model": r.model_name,
                        "predicted_rul_cycles": r.predicted_rul_cycles,
                        "ci_lower": r.lower_bound_95,
                        "ci_upper": r.upper_bound_95,
                        "degradation_pct": r.degradation_pct,
                    }
                    for r in ruls
                ]
                recs = [
                    "Order replacement components for units with Remaining Useful Life under 50 cycles.",
                    "Flag turbine degradation trajectory trends for engineering overhaul before end-of-life cutoff.",
                ]

            elif report_type == "Maintenance Work Orders & History":
                query = session.query(WorkOrder).options(joinedload(WorkOrder.machine))
                if machine_code != "All":
                    query = query.join(Machine).filter(Machine.machine_code == machine_code)
                orders = query.order_by(WorkOrder.created_at.desc()).all()

                kpis = {
                    "total_work_orders": len(orders),
                    "pending_orders": sum(1 for o in orders if o.status == "Pending"),
                    "completed_orders": sum(1 for o in orders if o.status == "Completed"),
                    "total_downtime_hours": f"{sum(o.actual_downtime_hours or o.estimated_downtime_hours for o in orders):.1f} hrs",
                }
                columns = ["WO Code", "Asset", "Task Title", "Priority", "Status", "Est. Downtime", "Created At"]
                rows = [
                    [
                        o.work_order_code,
                        o.machine.machine_code if o.machine else "N/A",
                        o.title,
                        o.priority,
                        o.status,
                        f"{o.estimated_downtime_hours} hrs",
                        o.created_at.strftime("%Y-%m-%d"),
                    ]
                    for o in orders
                ]
                dict_rows = [
                    {
                        "wo_code": o.work_order_code,
                        "machine_code": o.machine.machine_code if o.machine else "N/A",
                        "title": o.title,
                        "priority": o.priority,
                        "status": o.status,
                        "estimated_downtime": o.estimated_downtime_hours,
                        "created_at": o.created_at.strftime("%Y-%m-%d"),
                    }
                    for o in orders
                ]
                recs = [
                    "Expedite approval of pending Emergency and High priority work orders.",
                    "Verify actual downtime logs match maintenance contractor service sheets.",
                ]

            else:  # Model Benchmarks & Research Evaluations
                evals = session.query(ModelEvaluation).options(joinedload(ModelEvaluation.model_version)).all()
                drifts = session.query(DriftEvent).all()

                kpis = {
                    "registered_models": len(evals),
                    "top_classifier": "Random Forest",
                    "top_prognostic": "PyTorch LSTM",
                    "drift_events_flagged": len(drifts),
                }
                columns = ["Algorithm", "Dataset", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
                rows = [
                    [
                        e.model_version.algorithm if e.model_version else "Unknown",
                        e.dataset_name,
                        f"{e.accuracy * 100:.1f}%" if e.accuracy else "N/A",
                        f"{e.precision * 100:.1f}%" if e.precision else "N/A",
                        f"{e.recall * 100:.1f}%" if e.recall else "N/A",
                        f"{e.f1_score:.4f}" if e.f1_score else "N/A",
                        f"{e.roc_auc:.4f}" if e.roc_auc else "N/A",
                    ]
                    for e in evals
                ]
                dict_rows = [
                    {
                        "algorithm": e.model_version.algorithm if e.model_version else "Unknown",
                        "dataset": e.dataset_name,
                        "accuracy": e.accuracy,
                        "precision": e.precision,
                        "recall": e.recall,
                        "f1_score": e.f1_score,
                        "roc_auc": e.roc_auc,
                        "pr_auc": e.pr_auc,
                        "mcc": e.mcc,
                    }
                    for e in evals
                ]
                recs = [
                    "Tree ensemble algorithms demonstrate state-of-the-art F1 and PR-AUC on imbalanced telemetry.",
                    "PyTorch LSTM achieves lower asymmetric NASA degradation error than baseline regression.",
                ]

            return {
                "report_title": report_type,
                "summary_kpis": kpis,
                "table_columns": columns,
                "table_rows": rows,
                "dict_rows": dict_rows,
                "machine_code": machine_code,
                "recommendations": recs,
            }

    @classmethod
    def export_pdf(cls, report_type: str, machine_code: str, output_path: Path) -> Path:
        """Fetch report data and compile into PDF format."""
        data = cls.get_report_data(report_type, machine_code)
        return ReportGenerator.generate_pdf(
            report_title=data["report_title"],
            summary_kpis=data["summary_kpis"],
            table_columns=data["table_columns"],
            table_rows=data["table_rows"],
            output_path=output_path,
            machine_code=machine_code,
            recommendations=data["recommendations"],
        )

    @classmethod
    def export_excel(cls, report_type: str, machine_code: str, output_path: Path) -> Path:
        """Fetch report data and compile into formatted Excel workbook."""
        data = cls.get_report_data(report_type, machine_code)
        sheets = {data["report_title"][:28]: data["dict_rows"]}
        return ReportGenerator.generate_excel(
            sheets_data=sheets,
            output_path=output_path,
            report_title=data["report_title"],
        )

    @classmethod
    def export_csv(cls, report_type: str, machine_code: str, output_path: Path) -> Path:
        """Fetch report data and compile into CSV format."""
        data = cls.get_report_data(report_type, machine_code)
        ReportGenerator.generate_csv(data["dict_rows"], output_path=output_path)
        return output_path
