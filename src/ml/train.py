# src/ml/train.py
"""Model training pipeline for Champion LightGBM and Baseline Logistic Regression."""
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
from pathlib import Path
import joblib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import Config
from src.utils.logger import get_logger
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor

logger = get_logger("ModelTrainer")

def train_all_models(sample_size: int = None):
    """
    Trains Champion LightGBM (with scale_pos_weight) and Baseline Logistic Regression.
    Evaluates with Stratified 5-Fold Cross-Validation and persists all model artifacts.
    """
    start_time = time.time()
    Config.ensure_directories()

    logger.info("=" * 70)
    logger.info("STARTING CREDIT RISK MODEL TRAINING & BENCHMARK PIPELINE")
    logger.info(f"Class Imbalance Strategy: scale_pos_weight={Config.SCALE_POS_WEIGHT} & Stratified {Config.N_FOLDS}-Fold CV")
    logger.info("=" * 70)

    # 1. Load Data
    loader = DataLoader()
    df_raw = loader.load_raw_dataframe(max_rows=sample_size)
    logger.info(f"Loaded {len(df_raw):,} records. Target distribution: {df_raw['TARGET'].value_counts().to_dict()}")

    # 2. Fit Preprocessor & Extract Features
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df_raw)
    preprocessor.save()

    # 3. Stratified K-Fold Setup
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
    
    # Track Out-of-Fold Predictions
    oof_preds_lgb = np.zeros(len(X))
    oof_preds_lr = np.zeros(len(X))
    
    # Feature importances accumulator
    feature_importances = np.zeros(len(preprocessor.feature_names))

    # Pre-scale for logistic regression baseline on a sampled/full subset
    logger.info("Preparing baseline scaled feature matrix...")
    X_baseline = preprocessor.prepare_baseline_matrix(X, is_train=True)

    # Fold Iteration
    fold_lgb_aucs = []
    fold_lr_aucs = []

    logger.info(f"Beginning Stratified {Config.N_FOLDS}-Fold Cross Validation...")
    
    best_lgb_model = None
    best_fold_auc = -1.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        f_start = time.time()
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # --- A. Train Champion LightGBM ---
        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "scale_pos_weight": Config.SCALE_POS_WEIGHT,
            "learning_rate": 0.05,
            "n_estimators": 400,
            "max_depth": 6,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": Config.RANDOM_STATE + fold,
            "n_jobs": -1,
            "verbose": -1
        }
        
        clf_lgb = lgb.LGBMClassifier(**lgb_params)
        
        # Fit with early stopping callback if supported
        clf_lgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
        )
        
        val_probs_lgb = clf_lgb.predict_proba(X_val)[:, 1]
        oof_preds_lgb[val_idx] = val_probs_lgb
        auc_lgb = roc_auc_score(y_val, val_probs_lgb)
        pr_auc_lgb = average_precision_score(y_val, val_probs_lgb)
        fold_lgb_aucs.append(auc_lgb)
        feature_importances += (clf_lgb.feature_importances_ / Config.N_FOLDS)

        if auc_lgb > best_fold_auc:
            best_fold_auc = auc_lgb
            best_lgb_model = clf_lgb

        # --- B. Train Baseline Logistic Regression ---
        X_tr_lr, y_tr_lr = X_baseline[train_idx], y.iloc[train_idx]
        X_va_lr, y_va_lr = X_baseline[val_idx], y.iloc[val_idx]
        
        clf_lr = LogisticRegression(
            class_weight="balanced",
            max_iter=300,
            random_state=Config.RANDOM_STATE,
            solver="lbfgs",
            n_jobs=-1
        )
        clf_lr.fit(X_tr_lr, y_tr_lr)
        val_probs_lr = clf_lr.predict_proba(X_va_lr)[:, 1]
        oof_preds_lr[val_idx] = val_probs_lr
        auc_lr = roc_auc_score(y_va_lr, val_probs_lr)
        fold_lr_aucs.append(auc_lr)

        logger.info(
            f"Fold {fold}/{Config.N_FOLDS} completed in {time.time() - f_start:.1f}s | "
            f"LightGBM ROC-AUC: {auc_lgb:.4f} (PR-AUC: {pr_auc_lgb:.4f}) | "
            f"Baseline LR ROC-AUC: {auc_lr:.4f}"
        )

    # Overall OOF Metrics
    total_auc_lgb = roc_auc_score(y, oof_preds_lgb)
    total_pr_lgb = average_precision_score(y, oof_preds_lgb)
    total_auc_lr = roc_auc_score(y, oof_preds_lr)
    total_pr_lr = average_precision_score(y, oof_preds_lr)

    logger.info("=" * 70)
    logger.info(f"OOF LightGBM Champion ROC-AUC: {total_auc_lgb:.4f} | PR-AUC: {total_pr_lgb:.4f}")
    logger.info(f"OOF Logistic Regression ROC-AUC: {total_auc_lr:.4f} | PR-AUC: {total_pr_lr:.4f}")
    logger.info(f"Performance Lift over Baseline: +{(total_auc_lgb - total_auc_lr):.4f} AUC points")
    logger.info("=" * 70)

    # 4. Save Final Production Models
    logger.info("Training champion model on full dataset for production deployment...")
    final_lgb = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        scale_pos_weight=Config.SCALE_POS_WEIGHT,
        learning_rate=0.05,
        n_estimators=int(best_lgb_model.best_iteration_ if hasattr(best_lgb_model, 'best_iteration_') and best_lgb_model.best_iteration_ else 350),
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=Config.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    final_lgb.fit(X, y)

    # Save Models
    joblib.dump(final_lgb, Config.CHAMPION_MODEL_PATH)
    logger.info(f"Champion LightGBM model saved to {Config.CHAMPION_MODEL_PATH}")

    final_lr = LogisticRegression(class_weight="balanced", max_iter=300, random_state=Config.RANDOM_STATE, solver="lbfgs")
    final_lr.fit(X_baseline, y)
    joblib.dump(final_lr, Config.BASELINE_MODEL_PATH)
    logger.info(f"Baseline Logistic Regression model saved to {Config.BASELINE_MODEL_PATH}")

    # 5. Save Feature Importance
    importance_df = pd.DataFrame({
        "feature": preprocessor.feature_names,
        "importance": feature_importances
    }).sort_values(by="importance", ascending=False)
    
    importance_dict = importance_df.to_dict(orient="records")
    with open(Config.FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(importance_dict, f, indent=2)
    logger.info(f"Top 5 predictive features: {[f['feature'] for f in importance_dict[:5]]}")

    # 6. Save Evaluation Metrics Summary
    from src.utils.helpers import calculate_expected_loss
    loss_lgb = calculate_expected_loss(y, oof_preds_lgb, threshold=0.5)
    loss_lr = calculate_expected_loss(y, oof_preds_lr, threshold=0.5)

    metrics_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_records": int(len(X)),
        "default_rate_pct": float(np.mean(y) * 100.0),
        "scale_pos_weight": Config.SCALE_POS_WEIGHT,
        "champion_lightgbm": {
            "cv_roc_auc_mean": float(np.mean(fold_lgb_aucs)),
            "cv_roc_auc_std": float(np.std(fold_lgb_aucs)),
            "oof_roc_auc": float(total_auc_lgb),
            "oof_pr_auc": float(total_pr_lgb),
            "fold_aucs": [float(a) for a in fold_lgb_aucs],
            "expected_portfolio_loss": loss_lgb
        },
        "baseline_logistic_regression": {
            "cv_roc_auc_mean": float(np.mean(fold_lr_aucs)),
            "cv_roc_auc_std": float(np.std(fold_lr_aucs)),
            "oof_roc_auc": float(total_auc_lr),
            "oof_pr_auc": float(total_pr_lr),
            "fold_aucs": [float(a) for a in fold_lr_aucs],
            "expected_portfolio_loss": loss_lr
        },
        "financial_cost_savings": float(loss_lr["total_cost"] - loss_lgb["total_cost"]),
        "training_duration_seconds": round(time.time() - start_time, 2)
    }

    with open(Config.METRICS_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Model evaluation scorecard saved to {Config.METRICS_PATH}")
    logger.info(f"Training pipeline finished in {metrics_summary['training_duration_seconds']}s.")

    return metrics_summary

if __name__ == "__main__":
    train_all_models()
