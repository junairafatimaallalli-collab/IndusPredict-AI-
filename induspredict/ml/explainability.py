"""
IndusPredict AI - Explainable Artificial Intelligence (XAI) Engine.
Integrates SHAP (SHapley Additive exPlanations) for tree ensemble models
(Random Forest, XGBoost, LightGBM).
Provides:
- Local prediction explanations (Shapley feature attribution values phi_i)
- Global feature importance across representative background data
- Human-readable engineering summaries
- Normal vs. Abnormal parameter distribution comparisons
- Explicit scientific disclaimer for industrial safety
"""

from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import shap

from induspredict.core.config import MODELS_DIR, RAW_DATA_DIR
from induspredict.core.logger import get_logger
from induspredict.ml.preprocessor import DataPreprocessor

logger = get_logger("ml.explainability")


class SHAPExplainer:
    """Explainable AI engine providing SHAP attributions and engineering narratives."""

    DISCLAIMER = (
        "SCIENTIFIC NOTICE: SHAP values represent statistical game-theoretic feature attributions "
        "from the trained tree ensemble model. They quantify which features shifted the mathematical "
        "prediction toward the failure threshold, but do NOT constitute metallurgically verified "
        "physical root causes. On-site engineering inspection is mandatory before maintenance."
    )

    FEATURE_FRIENDLY_NAMES = {
        "air_temperature": ("Ambient Air Temperature", "°C"),
        "process_temperature": ("Process Core Temperature", "°C"),
        "rotational_speed": ("Rotational Speed", "RPM"),
        "torque": ("Mechanical Torque / Load", "Nm"),
        "tool_wear": ("Tool / Component Wear", "min"),
        "temp_diff": ("Thermal Dissipation Delta (T_proc - T_air)", "°C"),
        "power_watts": ("Mechanical Rotational Power (τ · ω)", "W"),
        "strain_index": ("Mechanical Strain Index (Torque · Wear)", "Nm·min"),
        "dissipation_ratio": ("Heat Dissipation Ratio", "ratio"),
        "type_encoded": ("Machine Variant / Spec Tier", "tier"),
    }

    # Reference nominal values from UCI AI4I 2020 dataset
    NOMINAL_BENCHMARKS = {
        "air_temperature": {"median": 25.0, "nominal_min": 20.0, "nominal_max": 30.0, "unit": "°C"},
        "process_temperature": {"median": 35.0, "nominal_min": 30.0, "nominal_max": 42.0, "unit": "°C"},
        "rotational_speed": {"median": 1500.0, "nominal_min": 1380.0, "nominal_max": 1620.0, "unit": "RPM"},
        "torque": {"median": 40.0, "nominal_min": 25.0, "nominal_max": 55.0, "unit": "Nm"},
        "tool_wear": {"median": 30.0, "nominal_min": 0.0, "nominal_max": 150.0, "unit": "min"},
        "temp_diff": {"median": 10.0, "nominal_min": 8.6, "nominal_max": 12.5, "unit": "°C"},
        "power_watts": {"median": 6280.0, "nominal_min": 3800.0, "nominal_max": 8800.0, "unit": "W"},
        "strain_index": {"median": 1200.0, "nominal_min": 0.0, "nominal_max": 6000.0, "unit": "Nm·min"},
    }

    _explainers_cache: Dict[str, shap.TreeExplainer] = {}
    _preprocessor: Optional[DataPreprocessor] = None
    _background_df: Optional[pd.DataFrame] = None

    @classmethod
    def get_preprocessor(cls) -> DataPreprocessor:
        if cls._preprocessor is None:
            cls._preprocessor = DataPreprocessor()
        return cls._preprocessor

    @classmethod
    def get_background_data(cls, max_samples: int = 100) -> np.ndarray:
        """Load stratified background sample for global explanations."""
        if cls._background_df is not None:
            return cls._background_df

        preprocessor = cls.get_preprocessor()
        preprocessor.ensure_fitted()
        raw_csv = RAW_DATA_DIR / "ai4i2020.csv"
        if raw_csv.exists():
            df = pd.read_csv(raw_csv)
            sample_df = df.sample(min(max_samples, len(df)), random_state=42)
            eng_df = preprocessor.engineer_features(sample_df)
            X_scaled = preprocessor.scaler.transform(eng_df[preprocessor.feature_names])
            cls._background_df = X_scaled
        else:
            # Fallback synthetic grid
            grid = []
            for _ in range(50):
                s = {
                    "air_temperature": np.random.uniform(20, 30),
                    "process_temperature": np.random.uniform(30, 42),
                    "rotational_speed": np.random.uniform(1300, 1700),
                    "torque": np.random.uniform(20, 60),
                    "tool_wear": np.random.uniform(0, 180),
                    "machine_type": "M",
                }
                grid.append(preprocessor.transform_single(s)[0])
            cls._background_df = np.array(grid)

        return cls._background_df

    @classmethod
    def get_explainer(cls, model_name: str = "Random Forest") -> shap.TreeExplainer:
        """Retrieve or build a cached TreeExplainer for the specified model."""
        if model_name in cls._explainers_cache:
            return cls._explainers_cache[model_name]

        from induspredict.services.prediction_service import PredictionService
        model = PredictionService.load_model(model_name)
        explainer = shap.TreeExplainer(model)
        cls._explainers_cache[model_name] = explainer
        logger.info(f"Initialized SHAP TreeExplainer for {model_name}")
        return explainer

    @classmethod
    def explain_prediction(
        cls,
        raw_sensor_dict: Dict[str, float],
        model_name: str = "Random Forest",
    ) -> Dict[str, Any]:
        """
        Compute local SHAP attribution values for an individual prediction.
        Returns detailed feature attributions, base value, engineering summary,
        and normal vs. abnormal parameter comparison.
        """
        preprocessor = cls.get_preprocessor()
        explainer = cls.get_explainer(model_name)

        # Scale features
        X_df = preprocessor.transform_single(raw_sensor_dict)
        feature_names = preprocessor.feature_names

        # Compute raw SHAP values
        raw_shap = explainer.shap_values(X_df)

        # Extract failure class attribution
        if isinstance(raw_shap, list):
            # Binary classification list: [class_0, class_1]
            phi = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0]
        elif len(np.shape(raw_shap)) == 3:
            # Shape (1, num_features, 2)
            phi = raw_shap[0, :, 1]
        elif len(np.shape(raw_shap)) == 2:
            # Shape (1, num_features)
            phi = raw_shap[0]
        else:
            phi = np.array(raw_shap).flatten()

        # Expected base value E[f(x)]
        expected_val = explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) > 1:
            base_val = float(expected_val[1])
        elif isinstance(expected_val, (list, np.ndarray)):
            base_val = float(expected_val[0])
        else:
            base_val = float(expected_val)

        # Build feature attribution records
        total_abs_phi = float(np.sum(np.abs(phi))) or 1.0
        attributions = []

        for i, f_name in enumerate(feature_names):
            friendly_title, unit = cls.FEATURE_FRIENDLY_NAMES.get(f_name, (f_name, ""))
            actual_val = float(X_df[0, i])
            shap_val = float(phi[i])
            pct_contrib = round((abs(shap_val) / total_abs_phi) * 100.0, 1)

            attributions.append({
                "feature": f_name,
                "display_name": friendly_title,
                "unit": unit,
                "actual_value": round(actual_val, 2),
                "shap_value": round(shap_val, 4),
                "contribution_pct": pct_contrib,
                "direction": "Risk Escalator (+)" if shap_val > 0 else "Risk Mitigator (-)",
                "is_positive": shap_val > 0,
                "importance": abs(shap_val),
            })

        # Sort descending by importance
        attributions.sort(key=lambda x: x["importance"], reverse=True)

        # Generate Natural Language Engineering Summary
        top_positive = [a for a in attributions if a["is_positive"]][:3]
        top_mitigating = [a for a in attributions if not a["is_positive"]][:2]

        narratives = []
        if top_positive:
            narratives.append(f"• Primary Failure Driver: {top_positive[0]['display_name']} pushed failure probability up by {top_positive[0]['contribution_pct']}% of total attribution.")
            for pos in top_positive[1:]:
                narratives.append(f"• Elevated Stress: {pos['display_name']} ({pos['actual_value']} {pos['unit']}) contributed +{pos['contribution_pct']}% toward failure.")
        else:
            narratives.append("• Machine parameters operate within standard nominal envelopes; no significant failure drivers detected.")

        if top_mitigating:
            narratives.append(f"• Stabilizing Factors: {top_mitigating[0]['display_name']} ({top_mitigating[0]['actual_value']} {top_mitigating[0]['unit']}) helped suppress risk.")

        # Normal vs Abnormal Comparison
        comparison = {}
        for f_name, bench in cls.NOMINAL_BENCHMARKS.items():
            if f_name in raw_sensor_dict:
                val = raw_sensor_dict[f_name]
            elif f_name == "temp_diff":
                val = raw_sensor_dict.get("process_temperature", 35.0) - raw_sensor_dict.get("air_temperature", 25.0)
            elif f_name == "power_watts":
                val = raw_sensor_dict.get("torque", 40.0) * (raw_sensor_dict.get("rotational_speed", 1500.0) * (2.0 * np.pi / 60.0))
            elif f_name == "strain_index":
                val = raw_sensor_dict.get("torque", 40.0) * raw_sensor_dict.get("tool_wear", 20.0)
            else:
                continue

            status = "Nominal"
            if val < bench["nominal_min"]:
                status = "Below Nominal"
            elif val > bench["nominal_max"]:
                status = "Above Nominal / Stress"

            comparison[f_name] = {
                "metric": cls.FEATURE_FRIENDLY_NAMES.get(f_name, (f_name, ""))[0],
                "actual": round(val, 2),
                "nominal_range": f"{bench['nominal_min']} - {bench['nominal_max']} {bench['unit']}",
                "status": status,
                "is_abnormal": status != "Nominal",
            }

        return {
            "model_name": model_name,
            "base_value": round(base_val, 4),
            "attributions": attributions,
            "top_drivers": top_positive,
            "engineering_summary": "\n".join(narratives),
            "normal_vs_abnormal": comparison,
            "disclaimer": cls.DISCLAIMER,
        }

    @classmethod
    def get_global_feature_importance(cls, model_name: str = "Random Forest") -> List[Dict[str, Any]]:
        """
        Compute mean absolute SHAP feature importance across the reference background dataset.
        """
        background = cls.get_background_data(max_samples=100)
        explainer = cls.get_explainer(model_name)
        preprocessor = cls.get_preprocessor()

        shap_vals = explainer.shap_values(background)
        if isinstance(shap_vals, list):
            phi = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        elif len(np.shape(shap_vals)) == 3:
            phi = shap_vals[:, :, 1]
        else:
            phi = shap_vals

        mean_abs = np.mean(np.abs(phi), axis=0)
        total_mean = float(np.sum(mean_abs)) or 1.0

        results = []
        for i, col in enumerate(preprocessor.feature_names):
            friendly, unit = cls.FEATURE_FRIENDLY_NAMES.get(col, (col, ""))
            val = float(mean_abs[i])
            pct = round((val / total_mean) * 100.0, 1)
            results.append({
                "feature": col,
                "display_name": friendly,
                "mean_abs_shap": round(val, 4),
                "global_importance_pct": pct,
                "unit": unit,
            })

        results.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        return results
