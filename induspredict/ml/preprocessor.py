"""
IndusPredict AI - Machine Learning Preprocessing & Feature Engineering.
Preprocesses UCI AI4I 2020 Predictive Maintenance dataset without target leakage.
Computes thermodynamic, kinematic, and mechanical stress interaction features.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from induspredict.core.config import MODELS_DIR, RAW_DATA_DIR
from induspredict.core.logger import get_logger

logger = get_logger("ml.preprocessor")

SCALER_PATH = MODELS_DIR / "ai4i_scaler.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "ai4i_feature_names.joblib"

# Base raw features from AI4I 2020
RAW_NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Engineered feature names
ENGINEERED_FEATURE_NAMES = [
    "temp_difference_k",       # Process Temp - Air Temp (Heat Dissipation indicator)
    "mechanical_power_w",      # Torque * Rotational Speed in rad/s (Power failure indicator)
    "strain_index",            # Torque * Tool wear (Overstrain failure indicator)
    "heat_dissipation_factor", # Delta T / Rotational Speed
    "type_encoded",            # L=0, M=1, H=2
]

ALL_FEATURE_NAMES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURE_NAMES


class DataPreprocessor:
    """Preprocesses raw industrial telemetry and engineers physical interaction variables."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = ALL_FEATURE_NAMES
        self.is_fitted = False

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Derive physical interaction terms from raw kinematic and thermal variables."""
        df_feat = df.copy()

        # 1. Temperature difference (Heat Dissipation Failure feature)
        # Delta T = Process Temperature - Air Temperature
        air_t = df_feat["Air temperature [K]"]
        proc_t = df_feat["Process temperature [K]"]
        df_feat["temp_difference_k"] = proc_t - air_t

        # 2. Mechanical Power: P = Torque * omega (rad/s)
        # omega = rpm * (2 * pi / 60)
        speed_rpm = df_feat["Rotational speed [rpm]"]
        torque_nm = df_feat["Torque [Nm]"]
        omega_rad_s = speed_rpm * (2.0 * np.pi / 60.0)
        df_feat["mechanical_power_w"] = torque_nm * omega_rad_s

        # 3. Mechanical Strain Index = Torque * Tool Wear
        tool_wear = df_feat["Tool wear [min]"]
        df_feat["strain_index"] = torque_nm * tool_wear

        # 4. Heat Dissipation Factor = Delta T / Rotational Speed
        df_feat["heat_dissipation_factor"] = (proc_t - air_t) / np.maximum(1.0, speed_rpm)

        # 5. Type encoding: L (Low quality) = 0, M (Medium) = 1, H (High) = 2
        type_map = {"L": 0, "M": 1, "H": 2}
        if "Type" in df_feat.columns:
            df_feat["type_encoded"] = df_feat["Type"].map(type_map).fillna(1).astype(int)
        else:
            df_feat["type_encoded"] = 1  # Default medium

        return df_feat

    def fit_transform(
        self,
        csv_path: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load UCI AI4I 2020 dataset, engineer physical features, perform stratified split,
        and fit standard scaler strictly on training data to prevent data leakage.
        """
        path = Path(csv_path) if csv_path else (RAW_DATA_DIR / "ai4i2020.csv")
        if not path.exists():
            raise FileNotFoundError(f"UCI AI4I 2020 dataset not found at: {path}")

        df = pd.read_csv(path)
        logger.info(f"Loaded raw dataset with shape: {df.shape}")

        # Target variable: 'Machine failure' (binary)
        y = df["Machine failure"].values

        # Physical feature engineering
        df_engineered = self.engineer_features(df)
        X_df = df_engineered[self.feature_names]

        # Stratified Train-Test split to preserve exact failure ratio (~3.39%)
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df, y, test_size=test_size, stratify=y, random_state=random_state
        )

        # Fit Scaler strictly on train partition
        X_train_scaled = self.scaler.fit_transform(X_train_df)
        X_test_scaled = self.scaler.transform(X_test_df)
        self.is_fitted = True

        # Save fitted scaler
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(self.feature_names, FEATURE_NAMES_PATH)
        logger.info("Fitted preprocessor scaler and feature registry saved to disk.")

        metadata = {
            "total_samples": len(df),
            "train_samples": len(X_train_scaled),
            "test_samples": len(X_test_scaled),
            "failure_rate_train": float(np.mean(y_train)),
            "failure_rate_test": float(np.mean(y_test)),
            "feature_names": self.feature_names,
        }

        return X_train_scaled, X_test_scaled, y_train, y_test, metadata

    def ensure_fitted(self) -> None:
        """Ensure scaler and feature names are loaded from disk if not yet fitted."""
        if not self.is_fitted:
            if SCALER_PATH.exists():
                self.scaler = joblib.load(SCALER_PATH)
                self.feature_names = joblib.load(FEATURE_NAMES_PATH)
                self.is_fitted = True
            else:
                raise RuntimeError("Preprocessor has not been fitted and no saved scaler was found.")

    def transform_single(self, raw_input: Dict[str, float]) -> np.ndarray:
        """
        Transform a single runtime sensor measurement dictionary into scaled feature vector.
        raw_input format:
        {
            "air_temperature_c": 25.0,        # Converted to Kelvin
            "process_temperature_c": 35.0,    # Converted to Kelvin
            "rotational_speed": 1500.0,       # RPM
            "torque": 40.0,                   # Nm
            "tool_wear": 120.0,               # min
            "machine_type": "M"               # L, M, or H
        }
        """
        self.ensure_fitted()

        # Convert temperatures to Kelvin if passed in Celsius (typical industrial range < 150)
        air_t = float(raw_input.get("air_temperature", raw_input.get("air_temperature_c", 25.0)))
        if air_t < 200.0:
            air_t_k = air_t + 273.15
        else:
            air_t_k = air_t

        proc_t = float(raw_input.get("process_temperature", raw_input.get("process_temperature_c", 35.0)))
        if proc_t < 200.0:
            proc_t_k = proc_t + 273.15
        else:
            proc_t_k = proc_t

        speed = float(raw_input.get("rotational_speed", 1500.0))
        torque = float(raw_input.get("torque", 40.0))
        tool_wear = float(raw_input.get("tool_wear", 0.0))
        m_type = str(raw_input.get("machine_type", "M")).upper()

        row_df = pd.DataFrame([{
            "Air temperature [K]": air_t_k,
            "Process temperature [K]": proc_t_k,
            "Rotational speed [rpm]": speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Type": m_type,
        }])

        engineered_df = self.engineer_features(row_df)
        X_df = engineered_df[self.feature_names]
        return self.scaler.transform(X_df)
