# src/ml/explain.py
"""SHAP TreeExplainer local attribution & automated underwriter credit narrative generator."""
import numpy as np
import pandas as pd
import shap
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import Config
from src.utils.logger import get_logger
from src.data.preprocessor import DataPreprocessor

logger = get_logger("ModelExplainer")

# Friendly names and interpretation units for top financial features
FEATURE_HUMAN_NAMES = {
    "EXT_SOURCES_MEAN": "Composite Bureau Credit Score (EXT_SOURCES_MEAN)",
    "EXT_SOURCE_3": "External Credit Agency 3 Score",
    "EXT_SOURCE_2": "External Credit Agency 2 Score",
    "EXT_SOURCE_1": "External Credit Agency 1 Score",
    "EXT_SOURCES_MIN": "Worst Bureau Agency Score",
    "EXT_SOURCES_MAX": "Best Bureau Agency Score",
    "PAYMENT_RATE": "Loan Capital Amortization Rate (Annuity / Credit)",
    "ANNUITY_INCOME_PERC": "Debt-to-Income / DTI Ratio (Annuity / Income)",
    "INCOME_CREDIT_PERC": "Earning Capacity Ratio (Income / Credit)",
    "DAYS_EMPLOYED_PERC": "Proportion of Adult Life Employed",
    "DAYS_EMPLOYED": "Employment Duration (Days)",
    "DAYS_BIRTH": "Applicant Age (Days)",
    "AGE_YEARS": "Applicant Age (Years)",
    "YEARS_EMPLOYED": "Employment History (Years)",
    "AMT_CREDIT": "Total Loan Credit Amount ($)",
    "AMT_ANNUITY": "Monthly Loan Annuity ($)",
    "AMT_INCOME_TOTAL": "Total Annual Income ($)",
    "AMT_GOODS_PRICE": "Underlying Asset / Goods Price ($)",
    "DAYS_ID_PUBLISH": "Days Since Identity Document Issued",
    "DAYS_REGISTRATION": "Days Since Municipal Registration",
    "DAYS_LAST_PHONE_CHANGE": "Days Since Last Phone Number Change",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Delinquency Count in 30-Day Social Circle",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Delinquency Count in 60-Day Social Circle",
    "REGION_RATING_CLIENT": "Regional Credit Risk Rating",
    "REGION_RATING_CLIENT_W_CITY": "Regional Credit Risk Rating with City",
    "NAME_EDUCATION_TYPE": "Highest Level of Education",
    "NAME_INCOME_TYPE": "Income / Employment Source",
    "OCCUPATION_TYPE": "Occupation Category",
    "CODE_GENDER": "Gender Identifier",
    "FLAG_OWN_CAR": "Vehicle Ownership",
    "FLAG_OWN_REALTY": "Real Estate Ownership"
}

class ModelExplainer:
    def __init__(self):
        self.model = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """Initializes the SHAP TreeExplainer on the champion LightGBM model."""
        try:
            if Config.CHAMPION_MODEL_PATH.exists() and Config.PREPROCESSOR_PATH.exists():
                self.model = joblib.load(Config.CHAMPION_MODEL_PATH)
                self.preprocessor = DataPreprocessor.load(Config.PREPROCESSOR_PATH)
                # Use TreeExplainer for exact fast Shapley attribution
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Initialized SHAP TreeExplainer for LightGBM model.")
        except Exception as e:
            logger.warning(f"Notice during SHAP explainer initialization: {e}")

    def explain_applicant(self, applicant_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes local SHAP values and formats waterfall components and plain-English narrative.
        """
        if self.explainer is None:
            self._init_explainer()
            if self.explainer is None:
                raise RuntimeError("SHAP explainer could not be loaded. Please ensure models are trained.")

        X_trans = self.preprocessor.transform(applicant_df)
        feature_names = self.preprocessor.feature_names

        # Compute SHAP values
        shap_values_raw = self.explainer.shap_values(X_trans)
        
        # Handle binary classification output formats (array or list)
        if isinstance(shap_values_raw, list):
            sv = shap_values_raw[1][0] if len(shap_values_raw) > 1 else shap_values_raw[0][0]
        elif len(shap_values_raw.shape) == 3:
            sv = shap_values_raw[0, :, 1]
        elif len(shap_values_raw.shape) == 2:
            sv = shap_values_raw[0]
        else:
            sv = shap_values_raw

        # Expected baseline value
        base_val = float(self.explainer.expected_value[1]) if isinstance(self.explainer.expected_value, (list, np.ndarray)) else float(self.explainer.expected_value)

        # Build feature contributions list
        contributions = []
        for i, col in enumerate(feature_names):
            val = X_trans.iloc[0][col]
            shap_val = float(sv[i])
            human_name = FEATURE_HUMAN_NAMES.get(col, col)
            contributions.append({
                "feature": col,
                "display_name": human_name,
                "value": val,
                "shap_value": shap_val,
                "abs_shap": abs(shap_val),
                "is_risk_driver": shap_val > 0  # Positive SHAP pushes default risk UP
            })

        # Sort by absolute impact
        contributions_sorted = sorted(contributions, key=lambda x: x["abs_shap"], reverse=True)

        # Top 3 Risk Escalators (Positive SHAP = Higher Default Risk)
        risk_escalators = [c for c in contributions_sorted if c["shap_value"] > 0][:3]
        
        # Top 3 Risk Mitigators (Negative SHAP = Lower Default Risk / Creditworthy)
        risk_mitigators = [c for c in contributions_sorted if c["shap_value"] < 0][:3]

        # Generate Plain-English Underwriter Bullet Points
        underwriter_bullets = self._generate_plain_english_summary(risk_escalators, risk_mitigators, applicant_df.iloc[0])

        # Top 10 features for Waterfall plot
        top_10 = contributions_sorted[:10]

        return {
            "base_value": base_val,
            "total_prediction_margin": float(base_val + np.sum(sv)),
            "contributions_top10": top_10,
            "all_contributions": contributions_sorted,
            "risk_escalators": risk_escalators,
            "risk_mitigators": risk_mitigators,
            "underwriter_summary_bullets": underwriter_bullets
        }

    def _generate_plain_english_summary(self, escalators: List[Dict], mitigators: List[Dict], raw_row: pd.Series) -> List[str]:
        """
        Translates raw mathematical SHAP scores into professional banking underwriter narratives.
        """
        bullets = []

        # Escalators (Adverse Factors)
        if escalators:
            bullets.append("**Primary Adverse Risk Drivers (Factors Increasing Default Risk):**")
            for esc in escalators:
                feat = esc["feature"]
                val = esc["value"]
                if "EXT_SOURCE" in feat:
                    bullets.append(f"• **Weak Bureau Score**: Bureau rating `{feat}` is low ({val:.3f}), indicating adverse credit bureau history across partner bureaus.")
                elif feat == "PAYMENT_RATE":
                    bullets.append(f"• **High Debt Amortization Speed**: Monthly annuity burden `{val:.4f}` relative to credit indicates high short-term liquidity strain.")
                elif feat == "ANNUITY_INCOME_PERC":
                    bullets.append(f"• **Elevated Debt-to-Income (DTI)**: Loan payments consume {val*100:.1f}% of total annual income.")
                elif feat == "DAYS_BIRTH" or feat == "AGE_YEARS":
                    bullets.append(f"• **Age Demographics**: Younger applicant profile correlates with higher historical default volatility.")
                elif feat == "DAYS_EMPLOYED" or feat == "YEARS_EMPLOYED":
                    bullets.append(f"• **Short Employment Tenure**: Shorter job tenure reflects higher vulnerability to income disruption.")
                elif "DEF_" in feat:
                    bullets.append(f"• **Social Circle Delinquency**: Observed historical payment delays in applicant's geographic or social network.")
                else:
                    bullets.append(f"• **{esc['display_name']}**: Recorded value ({val}) adversely impacts default likelihood by +{esc['shap_value']:.3f} log-odds.")

        # Mitigators (Favorable Factors)
        if mitigators:
            bullets.append("**Primary Mitigating Strengths (Factors Supporting Creditworthiness):**")
            for mit in mitigators:
                feat = mit["feature"]
                val = mit["value"]
                if "EXT_SOURCE" in feat:
                    bullets.append(f"• **Strong Bureau Ratings**: Bureau indicator `{feat}` is strong ({val:.3f}), reflecting established repayment discipline.")
                elif feat == "PAYMENT_RATE":
                    bullets.append(f"• **Manageable Amortization**: Repayment schedule ({val:.4f}) is comfortably paced over the loan term.")
                elif feat == "INCOME_CREDIT_PERC":
                    bullets.append(f"• **Robust Earning Power**: Total income represents strong coverage ({val*100:.1f}%) against requested credit.")
                elif feat == "DAYS_EMPLOYED" or feat == "YEARS_EMPLOYED":
                    bullets.append(f"• **Stable Employment History**: Established professional tenure provides reliable cash-flow security.")
                elif "FLAG_OWN_REALTY" in feat or "FLAG_OWN_CAR" in feat:
                    bullets.append(f"• **Collateral Asset Backing**: Ownership of real estate or vehicle provides secondary balance sheet comfort.")
                else:
                    bullets.append(f"• **{mit['display_name']}**: Recorded value ({val}) improves applicant score by {abs(mit['shap_value']):.3f} log-odds.")

        return bullets
