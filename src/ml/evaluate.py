# src/ml/evaluate.py
"""Model benchmarking, ROC-AUC / PR-AUC curves, and financial cost matrix evaluation."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, confusion_matrix, classification_report
import sys
from pathlib import Path
import joblib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.helpers import calculate_expected_loss
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor

logger = get_logger("ModelEvaluator")

def evaluate_models(df: pd.DataFrame = None) -> dict:
    """
    Evaluates Champion LightGBM and Baseline Logistic Regression.
    Returns metrics, curves, and financial loss statistics.
    """
    if df is None:
        loader = DataLoader()
        df = loader.load_raw_dataframe(max_rows=50000)

    preprocessor = DataPreprocessor.load(Config.PREPROCESSOR_PATH)
    lgb_model = joblib.load(Config.CHAMPION_MODEL_PATH)
    lr_model = joblib.load(Config.BASELINE_MODEL_PATH)

    X = preprocessor.transform(df)
    y = df["TARGET"]

    # Champion LightGBM Predictions
    lgb_probs = lgb_model.predict_proba(X)[:, 1]
    
    # Baseline LR Predictions
    X_base = preprocessor.prepare_baseline_matrix(X, is_train=False)
    lr_probs = lr_model.predict_proba(X_base)[:, 1]

    # Metrics
    auc_lgb = roc_auc_score(y, lgb_probs)
    pr_auc_lgb = average_precision_score(y, lgb_probs)
    
    auc_lr = roc_auc_score(y, lr_probs)
    pr_auc_lr = average_precision_score(y, lr_probs)

    # ROC Curves
    fpr_lgb, tpr_lgb, _ = roc_curve(y, lgb_probs)
    fpr_lr, tpr_lr, _ = roc_curve(y, lr_probs)

    # PR Curves
    prec_lgb, rec_lgb, _ = precision_recall_curve(y, lgb_probs)
    prec_lr, rec_lr, _ = precision_recall_curve(y, lr_probs)

    # Financial Cost Optimization
    loss_lgb = calculate_expected_loss(y, lgb_probs, threshold=0.5)
    loss_lr = calculate_expected_loss(y, lr_probs, threshold=0.5)

    results = {
        "dataset_size": len(df),
        "default_rate": float(np.mean(y)),
        "champion_lgb": {
            "roc_auc": float(auc_lgb),
            "pr_auc": float(pr_auc_lgb),
            "expected_loss": loss_lgb,
            "fpr": fpr_lgb[::max(1, len(fpr_lgb)//100)].tolist(),
            "tpr": tpr_lgb[::max(1, len(tpr_lgb)//100)].tolist(),
            "precision": prec_lgb[::max(1, len(prec_lgb)//100)].tolist(),
            "recall": rec_lgb[::max(1, len(rec_lgb)//100)].tolist()
        },
        "baseline_lr": {
            "roc_auc": float(auc_lr),
            "pr_auc": float(pr_auc_lr),
            "expected_loss": loss_lr,
            "fpr": fpr_lr[::max(1, len(fpr_lr)//100)].tolist(),
            "tpr": tpr_lr[::max(1, len(tpr_lr)//100)].tolist(),
            "precision": prec_lr[::max(1, len(prec_lr)//100)].tolist(),
            "recall": rec_lr[::max(1, len(rec_lr)//100)].tolist()
        },
        "cost_savings": float(loss_lr["total_cost"] - loss_lgb["total_cost"])
    }

    logger.info(f"Evaluation complete. Champion ROC-AUC: {auc_lgb:.4f} vs Baseline: {auc_lr:.4f}")
    return results

if __name__ == "__main__":
    evaluate_models()
