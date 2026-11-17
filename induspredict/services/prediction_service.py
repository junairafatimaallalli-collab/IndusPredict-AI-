"""
IndusPredict AI - Failure Prediction & Anomaly Classification Service.
Coordinates inference across Random Forest, XGBoost, LightGBM, and Logistic Regression.
Computes probabilistic failure risk, uncertainty bounds, and physical failure modes,
persisting prediction logs in SQLite.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sqlalchemy.orm import selectinload

from induspredict.core.config import (
    MODELS_DIR,
    RISK_CRITICAL,
    RISK_HIGH,
    RISK_NORMAL,
    RISK_WARNING,
)
from induspredict.core.logger import get_logger
from induspredict.core.utils import utc_now
from induspredict.db.models import Machine, ModelEvaluation, ModelVersion, Prediction
from induspredict.db.session import get_db_session
from induspredict.ml.preprocessor import DataPreprocessor

logger = get_logger("services.prediction")


class PredictionService:
    """Service providing runtime failure prediction, uncertainty estimation, and model benchmarks."""

    _models_cache: Dict[str, Any] = {}
    _preprocessor: Optional[DataPreprocessor] = None

    MODEL_FILENAMES = {
        "Random Forest": "random_forest.joblib",
        "XGBoost": "xgboost.joblib",
        "LightGBM": "lightgbm.joblib",
        "Logistic Regression": "logistic_regression.joblib",
    }

    @classmethod
    def get_preprocessor(cls) -> DataPreprocessor:
        if cls._preprocessor is None:
            cls._preprocessor = DataPreprocessor()
        return cls._preprocessor

    @classmethod
    def load_model(cls, model_name: str = "Random Forest") -> Any:
        """Load trained model artifact from cache or disk."""
        if model_name in cls._models_cache:
            return cls._models_cache[model_name]

        filename = cls.MODEL_FILENAMES.get(model_name, "random_forest.joblib")
        filepath = MODELS_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Trained model artifact '{filename}' not found in {MODELS_DIR}. Train models first.")

        model = joblib.load(filepath)
        cls._models_cache[model_name] = model
        logger.info(f"Loaded ML model: {model_name} from {filepath}")
        return model

    @classmethod
    def predict_machine_failure(
        cls,
        raw_sensor_input: Dict[str, float],
        machine_code: str = "CMP-01",
        model_name: str = "Random Forest",
    ) -> Dict[str, Any]:
        """
        Execute machine failure classification.
        Returns failure probability, uncertainty estimate, risk category,
        likely failure mode, and top contributing physical factors.
        """
        model = cls.load_model(model_name)
        preprocessor = cls.get_preprocessor()

        # 1. Scale and engineer features
        X_scaled = preprocessor.transform_single(raw_sensor_input)

        # 2. Probability Prediction
        probabilities = model.predict_proba(X_scaled)[0]
        failure_prob = float(probabilities[1])

        # 3. Uncertainty Estimation (Epistemic / Ensemble variance)
        if hasattr(model, "estimators_"):
            # Random Forest: estimate standard deviation across individual tree predictions
            tree_preds = [tree.predict_proba(X_scaled)[0, 1] for tree in model.estimators_]
            uncertainty_std = float(np.std(tree_preds))
        else:
            # For linear/boosting: margin heuristic based on distance from decision threshold 0.5
            margin = abs(failure_prob - 0.5)
            uncertainty_std = float(0.15 * (1.0 - margin * 2))

        # 4. Stratify into Risk Category
        if failure_prob < 0.25:
            risk_cat = RISK_NORMAL
        elif failure_prob < 0.50:
            risk_cat = RISK_WARNING
        elif failure_prob < 0.75:
            risk_cat = RISK_HIGH
        else:
            risk_cat = RISK_CRITICAL

        # 5. Physics-Based Likely Failure Mode Identification
        air_t = float(raw_sensor_input.get("air_temperature_c", 25.0)) + 273.15
        proc_t = float(raw_sensor_input.get("process_temperature_c", 35.0)) + 273.15
        speed = float(raw_sensor_input.get("rotational_speed", 1500.0))
        torque = float(raw_sensor_input.get("torque", 40.0))
        tool_wear = float(raw_sensor_input.get("tool_wear", 0.0))

        delta_t = proc_t - air_t
        power_w = torque * speed * (2.0 * np.pi / 60.0)
        strain = torque * tool_wear

        likely_failure_mode = "None (Nominal)"
        if failure_prob >= 0.25:
            modes = []
            if tool_wear >= 200.0:
                modes.append("Tool Wear Failure (TWF)")
            if delta_t < 8.6 and speed < 1380.0:
                modes.append("Heat Dissipation Failure (HDF)")
            if strain > 11000.0:
                modes.append("Overstrain Failure (OSF)")
            if power_w < 3500.0 or power_w > 9000.0:
                modes.append("Power Overload/Underload (PWF)")

            if modes:
                likely_failure_mode = " / ".join(modes)
            else:
                likely_failure_mode = "Kinematic/Thermal Wear Anomaly"

        # 6. Local Feature Contribution Analysis
        feature_names = preprocessor.feature_names
        # Approximate feature importance product with scaled magnitude
        feature_weights = getattr(model, "feature_importances_", None)
        if feature_weights is None and hasattr(model, "coef_"):
            feature_weights = np.abs(model.coef_[0])
        if feature_weights is None:
            feature_weights = np.ones(len(feature_names)) / len(feature_names)

        local_impacts = np.abs(X_scaled[0]) * feature_weights
        sum_impacts = max(1e-5, float(np.sum(local_impacts)))
        normalized_impacts = {
            feature_names[i]: round(float(local_impacts[i] / sum_impacts), 3)
            for i in range(len(feature_names))
        }
        # Sort top 5 features
        top_contributions = dict(sorted(normalized_impacts.items(), key=lambda item: item[1], reverse=True)[:5])

        # 7. Persist to SQLite
        now = utc_now()
        with get_db_session() as session:
            machine = session.query(Machine).filter(Machine.machine_code == machine_code).first()
            if machine:
                pred_record = Prediction(
                    machine_id=machine.id,
                    timestamp=now,
                    model_name=model_name,
                    model_version="1.0.0",
                    failure_probability=round(failure_prob, 4),
                    risk_category=risk_cat,
                    confidence_score=round(1.0 - min(0.5, uncertainty_std), 4),
                    predicted_failure_mode=likely_failure_mode,
                    contributing_features=top_contributions,
                    status="active",
                )
                session.add(pred_record)
                session.commit()

        return {
            "machine_code": machine_code,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "model_version": "1.0.0",
            "failure_probability": round(failure_prob, 4),
            "failure_percentage": round(failure_prob * 100.0, 2),
            "uncertainty_std": round(uncertainty_std, 4),
            "uncertainty_percentage": round(uncertainty_std * 100.0, 2),
            "risk_category": risk_cat,
            "likely_failure_mode": likely_failure_mode,
            "confidence_score": round(1.0 - min(0.5, uncertainty_std), 3),
            "top_contributions": top_contributions,
            "physical_parameters": {
                "delta_t_k": round(delta_t, 2),
                "mechanical_power_w": round(power_w, 1),
                "strain_index": round(strain, 1),
            },
        }

    @staticmethod
    def get_model_benchmark_comparisons() -> List[Dict[str, Any]]:
        """Retrieve real evaluation benchmarks from SQLite for all trained algorithms."""
        with get_db_session() as session:
            evaluations = (
                session.query(ModelEvaluation)
                .options(selectinload(ModelEvaluation.model_version))
                .order_by(ModelEvaluation.evaluated_at.desc())
                .all()
            )

            seen_models = set()
            results = []
            for ev in evaluations:
                m_name = ev.model_version.model_name if ev.model_version else "Unknown"
                if m_name in seen_models:
                    continue
                seen_models.add(m_name)

                cm = {}
                if ev.confusion_matrix_json:
                    try:
                        cm = json.loads(ev.confusion_matrix_json)
                    except Exception:
                        pass

                results.append({
                    "model_name": m_name,
                    "algorithm": ev.model_version.algorithm if ev.model_version else "",
                    "dataset": ev.dataset_name,
                    "accuracy": ev.accuracy,
                    "precision": ev.precision,
                    "recall": ev.recall,
                    "f1_score": ev.f1_score,
                    "roc_auc": ev.roc_auc,
                    "pr_auc": ev.pr_auc,
                    "mcc": ev.mcc,
                    "confusion_matrix": cm,
                    "evaluated_at": ev.evaluated_at.strftime("%Y-%m-%d %H:%M:%S"),
                })

            return results

    @staticmethod
    def get_prediction_history(machine_code: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve historical failure predictions from SQLite."""
        with get_db_session() as session:
            query = (
                session.query(Prediction)
                .options(selectinload(Prediction.machine))
                .order_by(Prediction.timestamp.desc())
            )
            if machine_code:
                query = query.join(Machine).filter(Machine.machine_code == machine_code)

            preds = query.limit(limit).all()
            return [
                {
                    "id": p.id,
                    "machine_code": p.machine.machine_code if p.machine else "N/A",
                    "timestamp": p.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_name": p.model_name,
                    "model_version": p.model_version,
                    "failure_prob": round(p.failure_probability * 100.0, 1),
                    "risk_category": p.risk_category,
                    "confidence": round(p.confidence_score * 100.0, 1),
                    "failure_mode": p.predicted_failure_mode or "None",
                    "top_feature": list(p.contributing_features.keys())[0] if p.contributing_features else "None",
                }
                for p in preds
            ]
