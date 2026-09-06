# src/data/preprocessor.py
"""Feature engineering, ratio calculations, encoding, and data transformation pipeline."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("DataPreprocessor")

class DataPreprocessor:
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.num_imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.median_values: Dict[str, float] = {}

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates banking domain ratios and cleans anomalous entries.
        """
        df = df.copy()

        # Handle anomalous 365243 days employed (often used for retired/unemployed)
        if "DAYS_EMPLOYED" in df.columns:
            df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
            df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace({365243: np.nan})

        # Age in years
        if "DAYS_BIRTH" in df.columns:
            df["AGE_YEARS"] = np.abs(df["DAYS_BIRTH"]) / 365.25

        # Years employed
        if "DAYS_EMPLOYED" in df.columns:
            df["YEARS_EMPLOYED"] = np.abs(df["DAYS_EMPLOYED"]) / 365.25

        # 1. PAYMENT_RATE: Speed of loan capital amortization
        if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
            df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1e-6)

        # 2. INCOME_CREDIT_PERC: Earning capacity relative to loan principal
        if "AMT_INCOME_TOTAL" in df.columns and "AMT_CREDIT" in df.columns:
            df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / (df["AMT_CREDIT"] + 1e-6)

        # 3. ANNUITY_INCOME_PERC: Debt-Service-to-Income / DTI ratio
        if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1e-6)

        # 4. DAYS_EMPLOYED_PERC: Proportion of adult life spent in active employment
        if "DAYS_EMPLOYED" in df.columns and "DAYS_BIRTH" in df.columns:
            df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1e-6)

        # 5. INCOME_PER_PERSON: Per-capita disposable income
        if "AMT_INCOME_TOTAL" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
            df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"].fillna(1.0).clip(lower=1.0))

        # 6. External Sources Ratios (Composite bureau indicators)
        ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
        if ext_cols:
            df["EXT_SOURCES_MEAN"] = df[ext_cols].mean(axis=1)
            df["EXT_SOURCES_MIN"] = df[ext_cols].min(axis=1)
            df["EXT_SOURCES_MAX"] = df[ext_cols].max(axis=1)
            df["EXT_SOURCES_STD"] = df[ext_cols].std(axis=1).fillna(0)

        # 7. Credit to Goods Price ratio
        if "AMT_CREDIT" in df.columns and "AMT_GOODS_PRICE" in df.columns:
            df["CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1e-6)

        # 8. Social circle delinquency ratio
        if "DEF_30_CNT_SOCIAL_CIRCLE" in df.columns and "OBS_30_CNT_SOCIAL_CIRCLE" in df.columns:
            df["SOCIAL_CIRCLE_DEF_RATIO"] = df["DEF_30_CNT_SOCIAL_CIRCLE"] / (df["OBS_30_CNT_SOCIAL_CIRCLE"] + 1e-6)

        return df

    def select_model_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Selects primary predictive features and separates target.
        """
        target = df["TARGET"] if "TARGET" in df.columns else None
        
        # Exclude ID and target from feature matrix
        drop_cols = ["SK_ID_CURR", "TARGET"]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        
        X = df[feature_cols].copy()
        return X, target

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fits encoders and transforms training dataset.
        """
        logger.info("Starting feature engineering and preprocessor fitting...")
        df_engineered = self.engineer_features(df)
        X, y = self.select_model_features(df_engineered)

        self.categorical_cols = list(X.select_dtypes(include=["object", "category"]).columns)
        self.numeric_cols = list(X.select_dtypes(include=[np.number]).columns)

        # Label encode categorical columns
        for col in self.categorical_cols:
            le = LabelEncoder()
            # Replace NaNs with 'Missing'
            X[col] = X[col].astype(str).fillna("Missing")
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le

        # Record medians for UI defaults and single applicant inference
        for col in self.numeric_cols:
            self.median_values[col] = float(X[col].median(skipna=True))

        self.feature_names = list(X.columns)
        
        # Fit baseline imputer and scaler
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        X_mat = X.copy().values
        X_imp = self.num_imputer.fit_transform(X_mat)
        self.scaler.fit(X_imp)

        logger.info(f"Preprocessor fit complete. Feature count: {len(self.feature_names)} ({len(self.numeric_cols)} numeric, {len(self.categorical_cols)} categorical)")
        
        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms new applicant data using fitted encoders.
        """
        df_engineered = self.engineer_features(df)
        X, _ = self.select_model_features(df_engineered)

        # Ensure all fitted features exist
        for col in self.feature_names:
            if col not in X.columns:
                if col in self.categorical_cols:
                    X[col] = "Missing"
                else:
                    X[col] = self.median_values.get(col, 0.0)

        # Encode categoricals safely
        for col in self.categorical_cols:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                classes = set(le.classes_)
                X[col] = X[col].astype(str).fillna("Missing")
                # Handle unseen labels by mapping to most frequent / first class
                X[col] = X[col].apply(lambda s: s if s in classes else le.classes_[0])
                X[col] = le.transform(X[col])

        # Align columns
        X = X[self.feature_names].copy()
        return X

    def prepare_baseline_matrix(self, X: pd.DataFrame, is_train: bool = False) -> np.ndarray:
        """
        Imputes NaNs and standardizes features for Logistic Regression baseline.
        """
        X_mat = X.copy().values
        if self.num_imputer is None or self.scaler is None:
            self.num_imputer = SimpleImputer(strategy="median")
            self.scaler = StandardScaler()
            X_imp = self.num_imputer.fit_transform(X_mat)
            return self.scaler.fit_transform(X_imp)

        X_imp = self.num_imputer.transform(X_mat)
        X_scaled = self.scaler.transform(X_imp)
        return X_scaled

    def save(self, filepath: Optional[Path] = None):
        """Saves fitted preprocessor state."""
        save_path = filepath or Config.PREPROCESSOR_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        logger.info(f"Preprocessor saved to {save_path}")

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "DataPreprocessor":
        """Loads fitted preprocessor state."""
        load_path = filepath or Config.PREPROCESSOR_PATH
        if not load_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {load_path}")
        preprocessor = joblib.load(load_path)
        logger.info(f"Preprocessor loaded from {load_path}")
        return preprocessor
