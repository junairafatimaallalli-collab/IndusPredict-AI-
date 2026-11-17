"""
Tests for IndusPredict AI Report Generator & Analytics Service (Phase 8).
Verifies PDF generation with ReportLab, multi-sheet Excel workbooks with XlsxWriter,
CSV streaming, and data aggregation for all report categories.
"""

from pathlib import Path
import tempfile
import pytest
from induspredict.reports.generator import ReportGenerator
from induspredict.services.report_service import ReportService


def test_csv_generation(tmp_path):
    """Verify clean CSV export from tabular records."""
    data = [
        {"machine": "CMP-01", "risk": "Normal", "prob": 0.012},
        {"machine": "MOT-02", "risk": "Warning", "prob": 0.425},
    ]
    out_file = tmp_path / "test_report.csv"
    csv_str = ReportGenerator.generate_csv(data, output_path=out_file)

    assert out_file.exists()
    assert "CMP-01" in csv_str
    assert "MOT-02" in csv_str


def test_excel_generation(tmp_path):
    """Verify multi-sheet styled Excel workbook generation using XlsxWriter."""
    sheets_data = {
        "Fleet Assets": [
            {"code": "CMP-01", "status": "healthy", "operating_hours": 3400.0},
            {"code": "PMP-03", "status": "warning", "operating_hours": 8900.0},
        ],
        "Work Orders": [
            {"code": "WO-001", "priority": "High", "downtime": 4.5},
        ],
    }
    out_file = tmp_path / "test_report.xlsx"
    res_path = ReportGenerator.generate_excel(sheets_data, out_file, report_title="Unit Test Report")

    assert res_path.exists()
    assert res_path.stat().st_size > 1000  # Non-empty binary xlsx


def test_pdf_generation(tmp_path):
    """Verify publication-quality PDF generation with ReportLab."""
    summary_kpis = {
        "total_assets": 5,
        "healthy_count": 4,
        "critical_alerts": 1,
        "mean_rul": "84.2 cycles",
    }
    columns = ["Asset Code", "Name", "Risk Level", "Hours"]
    rows = [
        ["CMP-01", "Atlas Copco GA37", "Normal", "3400 hrs"],
        ["CNC-05", "DMG MORI NLX", "Critical", "12100 hrs"],
    ]
    recs = [
        "Immediate spindle bearing inspection required for CNC-05.",
    ]
    out_file = tmp_path / "test_report.pdf"
    res_path = ReportGenerator.generate_pdf(
        report_title="Asset Reliability Assessment",
        summary_kpis=summary_kpis,
        table_columns=columns,
        table_rows=rows,
        output_path=out_file,
        machine_code="Fleet",
        recommendations=recs,
    )

    assert res_path.exists()
    assert res_path.stat().st_size > 2000  # Non-empty valid PDF


def test_report_service_aggregation(test_engine):
    """Verify database query aggregation for all 5 enterprise report types."""
    for rep_type in ReportService.get_report_data.__code__.co_consts:
        if isinstance(rep_type, str) and len(rep_type) > 10:
            pass

    for r_type in [
        "Machine Health & Asset Inventory",
        "Failure Risk & Anomaly Predictions",
        "Remaining Useful Life (RUL) Prognostics",
        "Maintenance Work Orders & History",
        "Model Benchmarks & Research Evaluations",
    ]:
        data = ReportService.get_report_data(r_type, machine_code="All")
        assert "summary_kpis" in data
        assert "table_columns" in data
        assert "table_rows" in data
        assert len(data["table_columns"]) > 0
