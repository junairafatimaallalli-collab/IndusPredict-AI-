"""
Tests for IndusPredict AI CSV and Excel Telemetry Ingestion.
"""

from pathlib import Path
import tempfile
import pandas as pd
import pytest
from induspredict.core.utils import utc_now
from induspredict.db.models import SensorReading
from induspredict.iot.csv_uploader import CSVTelemetryUploader


def test_csv_uploader_parsing_and_imputation(test_engine):
    """Verify CSV dataset validation, column identification, and forward-fill imputation."""
    uploader = CSVTelemetryUploader()

    # Create temporary sample CSV with missing values
    tmp_dir = Path(tempfile.mkdtemp())
    csv_file = tmp_dir / "sample_telemetry.csv"

    data = {
        "timestamp": [
            "2026-11-17 10:00:00",
            "2026-11-17 10:01:00",
            "2026-11-17 10:02:00",
            "2026-11-17 10:03:00",
        ],
        "temperature": [65.2, None, 68.0, 71.4],  # Row 1 is missing
        "vibration": [2.4, 2.5, None, 3.1],       # Row 2 is missing
        "pressure": [7.1, 7.2, 7.0, 6.9],
    }
    df_raw = pd.DataFrame(data)
    df_raw.to_csv(csv_file, index=False)

    success, df_clean, msg = uploader.parse_and_validate_file(str(csv_file))
    assert success is True
    assert len(df_clean) == 4
    assert df_clean["temperature"].isnull().sum() == 0, "Missing values must be imputed"
    assert df_clean["vibration"].isnull().sum() == 0

    # Ingest into SQLite database for machine 'CMP-01'
    res = uploader.import_dataframe_to_db(df_clean, "CMP-01")
    assert res["success"] is True
    assert res["readings_imported"] >= 4
    assert res["source"] == "uploaded"

    csv_file.unlink()
