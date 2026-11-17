"""
IndusPredict AI - XAI Service.
Interfaces the UI with SHAP explainability calculations and SQLite prediction records.
"""

from typing import Any, Dict, List, Optional
from sqlalchemy.orm import selectinload

from induspredict.core.logger import get_logger
from induspredict.db.models import Machine, Prediction
from induspredict.db.session import get_db_session
from induspredict.ml.explainability import SHAPExplainer

logger = get_logger("services.xai")


class XAIService:
    """Service providing local and global explainability workflows."""

    @classmethod
    def explain_active_prediction(
        cls,
        raw_sensor_dict: Dict[str, float],
        model_name: str = "Random Forest",
    ) -> Dict[str, Any]:
        """Generate real-time local SHAP explanation for given sensor readings."""
        return SHAPExplainer.explain_prediction(raw_sensor_dict, model_name=model_name)

    @classmethod
    def explain_historical_prediction(cls, prediction_id: int) -> Dict[str, Any]:
        """Retrieve a stored prediction from SQLite and compute its SHAP explanation."""
        with get_db_session() as session:
            pred = session.query(Prediction).options(selectinload(Prediction.machine)).filter(Prediction.id == prediction_id).first()
            if not pred:
                raise ValueError(f"Prediction ID {prediction_id} not found in database.")

            # Fallback sensor reconstruction from prediction record
            raw_sensors = {
                "air_temperature": 25.0,
                "process_temperature": 35.0,
                "rotational_speed": 1500.0,
                "torque": 40.0,
                "tool_wear": 20.0,
                "machine_type": "M",
            }
            if pred.contributing_features and isinstance(pred.contributing_features, dict):
                # Apply any stored specific features if available
                for k, v in pred.contributing_features.items():
                    if k in raw_sensors:
                        raw_sensors[k] = float(v)

            exp = SHAPExplainer.explain_prediction(raw_sensors, model_name=pred.model_name)
            exp["prediction_metadata"] = {
                "id": pred.id,
                "machine_code": pred.machine.machine_code if pred.machine else "UNKNOWN",
                "timestamp": pred.timestamp.strftime("%Y-%m-%d %H:%M"),
                "failure_probability": pred.failure_probability,
                "risk_category": pred.risk_category,
                "predicted_failure_mode": pred.predicted_failure_mode,
            }
            return exp

    @classmethod
    def get_global_importance(cls, model_name: str = "Random Forest") -> List[Dict[str, Any]]:
        """Retrieve ranked global feature importance across background data."""
        return SHAPExplainer.get_global_feature_importance(model_name=model_name)
