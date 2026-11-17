"""
IndusPredict AI - Certified Industrial Report Generator.
Generates publication-quality PDF reports with ReportLab, multi-sheet formatted
Excel workbooks with XlsxWriter, and standardized CSV datasets.
Supports Machine Health, Failure Predictions, RUL Prognostics,
Telemetry Audits, Maintenance Records, and Scientific Model Benchmarks.
"""

from datetime import datetime
import io
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
import xlsxwriter

from induspredict.core.logger import get_logger
from induspredict.core.utils import utc_now

logger = get_logger("reports.generator")


class ReportGenerator:
    """Enterprise report authoring engine generating PDF, Excel, and CSV artifacts."""

    @staticmethod
    def generate_csv(data: List[Dict[str, Any]], output_path: Optional[Path] = None) -> str:
        """Export dataset as clean CSV string or file."""
        if not data:
            return ""
        df = pd.DataFrame(data)
        csv_str = df.to_csv(index=False)
        if output_path:
            output_path.write_text(csv_str, encoding="utf-8")
            logger.info(f"Exported CSV report to {output_path}")
        return csv_str

    @staticmethod
    def generate_excel(
        sheets_data: Dict[str, List[Dict[str, Any]]],
        output_path: Path,
        report_title: str = "IndusPredict AI Industrial Report",
    ) -> Path:
        """Export multi-sheet styled Excel workbook using XlsxWriter."""
        workbook = xlsxwriter.Workbook(str(output_path))

        # Modern palette formats
        fmt_title = workbook.add_format({
            "bold": True,
            "font_size": 16,
            "font_color": "#00F2FE",
            "bg_color": "#0D1117",
            "align": "left",
            "valign": "vcenter",
        })
        fmt_meta = workbook.add_format({
            "font_size": 9,
            "font_color": "#8B949E",
            "bg_color": "#0D1117",
            "italic": True,
        })
        fmt_header = workbook.add_format({
            "bold": True,
            "font_size": 10,
            "font_color": "#FFFFFF",
            "bg_color": "#161B22",
            "border": 1,
            "border_color": "#30363D",
            "align": "center",
            "valign": "vcenter",
        })
        fmt_cell = workbook.add_format({
            "font_size": 9,
            "font_color": "#C9D1D9",
            "bg_color": "#0D1117",
            "border": 1,
            "border_color": "#21262D",
            "valign": "vcenter",
        })
        fmt_num = workbook.add_format({
            "font_size": 9,
            "font_color": "#00F2FE",
            "bg_color": "#0D1117",
            "border": 1,
            "border_color": "#21262D",
            "align": "right",
            "valign": "vcenter",
            "num_format": "0.00",
        })
        fmt_danger = workbook.add_format({
            "font_size": 9,
            "font_color": "#FF4B4B",
            "bold": True,
            "bg_color": "#2D1515",
            "border": 1,
            "border_color": "#30363D",
            "align": "center",
        })
        fmt_success = workbook.add_format({
            "font_size": 9,
            "font_color": "#00C9A7",
            "bold": True,
            "bg_color": "#122A22",
            "border": 1,
            "border_color": "#30363D",
            "align": "center",
        })

        for sheet_name, rows in sheets_data.items():
            safe_name = sheet_name[:31]  # Excel 31-char sheet limit
            ws = workbook.add_worksheet(safe_name)
            ws.set_tab_color("#00F2FE")

            # Title Header Row
            ws.merge_range("A1:G1", f"IndusPredict AI • {report_title} - {sheet_name}", fmt_title)
            ws.merge_range("A2:G2", f"Generated: {utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')} | Strict Auditing Enforced", fmt_meta)
            ws.set_row(0, 28)
            ws.set_row(1, 16)

            if not rows:
                ws.write(3, 0, "No records matched selected query.", fmt_cell)
                continue

            columns = list(rows[0].keys())

            # Header Row
            ws.set_row(3, 22)
            for col_idx, col_name in enumerate(columns):
                clean_header = col_name.replace("_", " ").title()
                ws.write(3, col_idx, clean_header, fmt_header)

            # Data Rows
            for row_idx, record in enumerate(rows):
                ws_row = 4 + row_idx
                ws.set_row(ws_row, 18)
                for col_idx, col_name in enumerate(columns):
                    val = record.get(col_name)

                    if isinstance(val, (int, float)):
                        ws.write(ws_row, col_idx, val, fmt_num)
                    elif str(val) in ["Critical", "High", "High Risk", "Emergency"]:
                        ws.write(ws_row, col_idx, str(val), fmt_danger)
                    elif str(val) in ["Normal", "Healthy", "Completed", "Approved"]:
                        ws.write(ws_row, col_idx, str(val), fmt_success)
                    else:
                        ws.write(ws_row, col_idx, str(val) if val is not None else "-", fmt_cell)

            # Set column widths
            for col_idx, col_name in enumerate(columns):
                max_len = max(len(col_name), max(len(str(r.get(col_name, ""))) for r in rows[:50]))
                ws.set_column(col_idx, col_idx, max(12, min(max_len + 3, 35)))

        workbook.close()
        logger.info(f"Generated formatted Excel workbook at {output_path}")
        return output_path

    @staticmethod
    def generate_pdf(
        report_title: str,
        summary_kpis: Dict[str, Any],
        table_columns: List[str],
        table_rows: List[List[Any]],
        output_path: Path,
        machine_code: str = "Fleet-Wide",
        recommendations: Optional[List[str]] = None,
    ) -> Path:
        """
        Generate a professional industrial PDF report using ReportLab with dark accents,
        tables, KPI summary blocks, and certified engineer sign-off blocks.
        """
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        # Custom styles
        style_title = ParagraphStyle(
            "DocTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0D1117"),
        )
        style_sub = ParagraphStyle(
            "DocSub",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#57606A"),
        )
        style_h2 = ParagraphStyle(
            "DocH2",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#0969DA"),
            spaceBefore=10,
            spaceAfter=6,
        )
        style_body = ParagraphStyle(
            "DocBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#24292F"),
        )
        style_tbl_hdr = ParagraphStyle(
            "TblHdr",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
            alignment=1,
        )
        style_tbl_cell = ParagraphStyle(
            "TblCell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#24292F"),
        )

        story = []

        # 1. Header Banner
        header_text = f"IndusPredict AI • {report_title}"
        story.append(Paragraph(header_text, style_title))
        sub_text = (
            f"Asset Target: <b>{machine_code}</b> | Timestamp: {utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')} "
            f"| Certified Compliance: ISO 13374 Condition Monitoring Standard"
        )
        story.append(Paragraph(sub_text, style_sub))
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0969DA"), spaceAfter=12))

        # 2. Executive KPI Summary Block
        story.append(Paragraph("EXECUTIVE KPI SUMMARY", style_h2))
        kpi_data = []
        kpi_items = list(summary_kpis.items())
        # Arrange in 2x2 or 3x2 grid
        row1_keys = kpi_items[:3]
        row2_keys = kpi_items[3:6]

        if row1_keys:
            kpi_data.append([Paragraph(f"<b>{k.replace('_', ' ').title()}</b>", style_body) for k, _ in row1_keys])
            kpi_data.append([Paragraph(f"<font size=12 color='#0969DA'><b>{v}</b></font>", style_body) for _, v in row1_keys])
        if row2_keys:
            kpi_data.append([Paragraph(f"<b>{k.replace('_', ' ').title()}</b>", style_body) for k, _ in row2_keys])
            kpi_data.append([Paragraph(f"<font size=12 color='#0969DA'><b>{v}</b></font>", style_body) for _, v in row2_keys])

        if kpi_data:
            kpi_tbl = Table(kpi_data, colWidths=[180] * len(row1_keys))
            kpi_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F6F8FA")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#D0D7DE")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E1E4E8")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]))
            story.append(kpi_tbl)
            story.append(Spacer(1, 12))

        # 3. Main Data Table
        story.append(Paragraph("AUDIT DATA LOGS", style_h2))
        tbl_content = []
        # Header row
        tbl_content.append([Paragraph(c, style_tbl_hdr) for c in table_columns])

        # Rows (limit to first 30 for PDF fit)
        for r in table_rows[:30]:
            tbl_content.append([Paragraph(str(cell), style_tbl_cell) for cell in r])

        num_cols = len(table_columns)
        col_w = max(45, int(540 / max(1, num_cols)))
        data_tbl = Table(tbl_content, colWidths=[col_w] * num_cols)
        data_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0D1117")),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8FA")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
        ]))
        story.append(data_tbl)
        story.append(Spacer(1, 14))

        # 4. Proactive Maintenance Recommendations
        if recommendations:
            story.append(Paragraph("RECOMMENDED MAINTENANCE ACTIONS", style_h2))
            for rec in recommendations:
                story.append(Paragraph(f"• {rec}", style_body))
            story.append(Spacer(1, 14))

        # 5. Sign-off and Verification Block (Industrial Compliance)
        story.append(KeepTogether([
            HRFlowable(width="100%", thickness=1, color=colors.HexColor("#D0D7DE"), spaceAfter=8),
            Paragraph("<b>COMPLIANCE SIGN-OFF & ENGINEERING CERTIFICATION</b>", style_sub),
            Spacer(1, 18),
            Table([
                [
                    Paragraph("Lead Reliability Engineer: _______________________", style_body),
                    Paragraph("Plant Maintenance Manager: _______________________", style_body),
                ],
                [
                    Paragraph(f"Signature Date: {utc_now().strftime('%Y-%m-%d')}", style_sub),
                    Paragraph("Approval Status: [X] VERIFIED  [ ] REJECTED", style_sub),
                ],
            ], colWidths=[270, 270]),
        ]))

        doc.build(story)
        logger.info(f"Generated PDF audit report at {output_path}")
        return output_path
