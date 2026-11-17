"""
IndusPredict AI - Model Training and Evaluation Pipeline.
Trains Random Forest, XGBoost, LightGBM, and Logistic Regression on UCI AI4I 2020.
Calculates Precision, Recall, F1, PR-AUC, ROC-AUC, MCC, and Confusion Matrices.
Persists model artifacts and registers evaluation benchmarks in SQLite.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
import xgboost as xgb

from induspredict.core.config import MODELS_DIR
from induspredict.core.logger import get_logger
from induspredict.core.utils import utc_now
from induspredict.db.models import ModelEvaluation, ModelVersion
from induspredict.db.session import get_db_session
from induspredict.ml.preprocessor import DataPreprocessor

logger = get_logger("ml.trainer")


class FailurePredictionTrainer:
    """Manages training, hyperparameter tuning, model checkpointing, and evaluation."""

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models_dir = MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train_and_evaluate_all(self, csv_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Train all four algorithms, evaluate on the identical held-out test partition,
        save model artifacts to disk, and store benchmark evaluations in the database.
        """
        logger.info("Initiating model training on UCI AI4I 2020 Predictive Maintenance dataset...")

        X_train, X_test, y_train, y_test, meta = self.preprocessor.fit_transform(csv_path=csv_path)

        # Class imbalance calculation: ratio of negative to positive samples
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        scale_pos_weight = float(n_neg / max(1, n_pos))
        logger.info(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")

        # Model Definitions
        model_configs = {
            "Logistic Regression": {
                "instance": LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                    C=1.0,
                ),
                "algorithm": "LogisticRegression",
                "filename": "logistic_regression.joblib",
                "hyperparameters": {"class_weight": "balanced", "max_iter": 1000, "C": 1.0},
            },
            "Random Forest": {
                "instance": RandomForestClassifier(
                    n_estimators=100,
                    max_depth=12,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
                "algorithm": "RandomForest",
                "filename": "random_forest.joblib",
                "hyperparameters": {"n_estimators": 100, "max_depth": 12, "class_weight": "balanced"},
            },
            "XGBoost": {
                "instance": xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.08,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=42,
                ),
                "algorithm": "XGBoost",
                "filename": "xgboost.joblib",
                "hyperparameters": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.08, "scale_pos_weight": scale_pos_weight},
            },
            "LightGBM": {
                "instance": lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.08,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    verbose=-1,
                ),
                "algorithm": "LightGBM",
                "filename": "lightgbm.joblib",
                "hyperparameters": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.08, "scale_pos_weight": scale_pos_weight},
            },
        }

        all_results = {}

        with get_db_session() as session:
            for model_name, cfg in model_configs.items():
                logger.info(f"Training {model_name}...")
                model = cfg["instance"]
                model.fit(X_train, y_train)

                # Predictions and probabilities on held-out test partition
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                # Comprehensive industrial metrics
                acc = float(accuracy_score(y_test, y_pred))
                prec = float(precision_score(y_test, y_pred, zero_division=0))
                rec = float(recall_score(y_test, y_pred, zero_division=0))
                f1 = float(f1_score(y_test, y_pred, zero_division=0))
                roc_auc = float(roc_auc_score(y_test, y_prob))
                pr_auc = float(average_precision_score(y_test, y_prob))
                mcc = float(matthews_corrcoef(y_test, y_pred))

                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                cm_dict = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

                # Save artifact to disk
                artifact_file = self.models_dir / cfg["filename"]
                joblib.dump(model, artifact_file)

                eval_metrics = {
                    "accuracy": round(acc, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1_score": round(f1, 4),
                    "roc_auc": round(roc_auc, 4),
                    "pr_auc": round(pr_auc, 4),
                    "mcc": round(mcc, 4),
                    "confusion_matrix": cm_dict,
                }
                all_results[model_name] = eval_metrics
                logger.info(f"{model_name} evaluated -> F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, MCC: {mcc:.4f}")

                # Register in database model_versions table
                mv = session.query(ModelVersion).filter(ModelVersion.model_name == model_name).first()
                if not mv:
                    mv = ModelVersion(
                        model_name=model_name,
                        version="1.0.0",
                        model_type="Classifier",
                        algorithm=cfg["algorithm"],
                        hyperparameters_json=json.dumps(cfg["hyperparameters"]),
                        artifact_path=str(artifact_file),
                        metrics_json=json.dumps(eval_metrics),
                        is_active=True,
                        created_at=utc_now(),
                    )
                    session.add(mv)
                    session.flush()
                else:
                    mv.metrics_json = json.dumps(eval_metrics)
                    mv.artifact_path = str(artifact_file)
                    mv.hyperparameters_json = json.dumps(cfg["hyperparameters"])

                # Store detailed evaluation entry
                evaluation = ModelEvaluation(
                    model_version_id=mv.id,
                    dataset_name="UCI AI4I 2020 Predictive Maintenance",
                    accuracy=acc,
                    precision=prec,
                    recall=rec,
                    f1_score=f1,
                    roc_auc=roc_auc,
                    pr_auc=pr_auc,
                    mcc=mcc,
                    confusion_matrix_json=json.dumps(cm_dict),
                    evaluated_at=utc_now(),
                )
                session.add(evaluation)

            session.commit()

        logger.info("All 4 failure classification models trained and registered successfully.")
        return all_results
