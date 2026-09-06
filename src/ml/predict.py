# src/ml/predict.py
"""Applicant scoring, credit score mapping (300-850), and Basel III risk tiering."""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import joblib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.helpers import prob_to_credit_score, fico_to_risk_tier
from src.data.preprocessor import DataPreprocessor

logger = get_logger("RiskPredictor")

class RiskPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads trained LightGBM model and preprocessor."""
        if Config.CHAMPION_MODEL_PATH.exists() and Config.PREPROCESSOR_PATH.exists():
            self.model = joblib.load(Config.CHAMPION_MODEL_PATH)
            self.preprocessor = DataPreprocessor.load(Config.PREPROCESSOR_PATH)
            logger.info("Loaded Champion LightGBM model and preprocessor.")
        else:
            logger.warning("Model artifacts not found. Please train models first via train_all_models().")

    def predict_applicant(self, applicant_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculates default probability, FICO-scaled credit score, and risk tier for an applicant.
        """
        if self.model is None or self.preprocessor is None:
            self._load_artifacts()
            if self.model is None or self.preprocessor is None:
                raise RuntimeError("Model artifacts not available. Please run train.py first.")

        if isinstance(applicant_data, dict):
            df_in = pd.DataFrame([applicant_data])
        else:
            df_in = applicant_data.copy()

        # Transform features
        X_trans = self.preprocessor.transform(df_in)

        # Get calibrated probabilities
        probs = self.model.predict_proba(X_trans)[:, 1]
        prob_default = float(probs[0])
        
        # Credit score mapping (300 to 850)
        credit_score = prob_to_credit_score(prob_default)
        risk_band, recommendation, color, badge = fico_to_risk_tier(credit_score, prob_default)

        # Extract top key feature values for the applicant
        applicant_features = X_trans.iloc[0].to_dict()

        return {
            "default_probability": round(prob_default, 4),
            "default_probability_pct": round(prob_default * 100.0, 2),
            "credit_score": credit_score,
            "risk_band": risk_band,
            "recommendation": recommendation,
            "theme_color": color,
            "badge_text": badge,
            "applicant_features": applicant_features,
            "raw_input": df_in.iloc[0].to_dict()
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scores an entire batch DataFrame of applicants.
        """
        if self.model is None or self.preprocessor is None:
            self._load_artifacts()

        df_out = df.copy()
        X_trans = self.preprocessor.transform(df)
        probs = self.model.predict_proba(X_trans)[:, 1]
        
        scores = [prob_to_credit_score(p) for p in probs]
        bands = [fico_to_risk_tier(s, p)[0] for s, p in zip(scores, probs)]
        recs = [fico_to_risk_tier(s, p)[1] for s, p in zip(scores, probs)]

        df_out["PROB_DEFAULT"] = np.round(probs, 4)
        df_out["CREDIT_SCORE"] = scores
        df_out["RISK_BAND"] = bands
        df_out["RECOMMENDATION"] = recs
        
        return df_out
