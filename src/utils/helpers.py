# src/utils/helpers.py
"""Financial calculation, formatting, and risk scoring helpers."""
from typing import Tuple, Dict, Any

def prob_to_credit_score(prob_default: float) -> int:
    """
    Maps calibrated default probability [0, 1] to FICO-like credit score [300, 850].
    Score = round(850 - (P_default * 550))
    """
    p = max(0.0, min(1.0, float(prob_default)))
    score = int(round(850.0 - (p * 550.0)))
    return max(300, min(850, score))

def fico_to_risk_tier(score: int, prob_default: float = None) -> Tuple[str, str, str, str]:
    """
    Returns (Risk Band, Recommendation, CSS Theme Color, Badge Text).
    Low Risk: Score 750-850, P < 0.07 -> Auto-Approve (Green)
    Medium Risk: Score 600-749, 0.07 <= P <= 0.20 -> Manual Underwriting (Amber)
    High Risk: Score 300-599, P > 0.20 -> Decline / Restructure (Rose/Red)
    """
    if prob_default is not None:
        if prob_default < 0.07:
            return "LOW RISK", "Auto-Approve: Prime rate pricing, instant digital disbursal.", "#10B981", "[LOW RISK] [AUTO-APPROVE]"
        elif prob_default <= 0.20:
            return "MEDIUM RISK", "Manual Underwriting: Income verification, collateral requirement.", "#F59E0B", "[MEDIUM RISK] [MANUAL REVIEW]"
        else:
            return "HIGH RISK", "Decline / Restructure: High-risk tier, reject unsecured credit.", "#EF4444", "[HIGH RISK] [DECLINE]"
            
    if score >= 750:
        return "LOW RISK", "Auto-Approve: Prime rate pricing, instant digital disbursal.", "#10B981", "[LOW RISK] [AUTO-APPROVE]"
    elif score >= 600:
        return "MEDIUM RISK", "Manual Underwriting: Income verification, collateral requirement.", "#F59E0B", "[MEDIUM RISK] [MANUAL REVIEW]"
    else:
        return "HIGH RISK", "Decline / Restructure: High-risk tier, reject unsecured credit.", "#EF4444", "[HIGH RISK] [DECLINE]"

def format_currency(val: float, symbol: str = "$") -> str:
    """Formats numeric values into clean currency strings."""
    if val is None or val != val:  # NaN check
        return "N/A"
    return f"{symbol}{val:,.2f}"

def format_percent(val: float) -> str:
    """Formats decimal proportions into percentages."""
    if val is None or val != val:
        return "N/A"
    return f"{val * 100.0:.2f}%"

def calculate_expected_loss(y_true, y_prob, threshold: float = 0.5, cost_fn: float = 10000.0, cost_fp: float = 1000.0) -> Dict[str, float]:
    """
    Calculates portfolio financial impact based on asymmetric banking cost matrix.
    """
    import numpy as np
    y_true = np.array(y_true)
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    default_losses = fn * cost_fn
    friction_cost = fp * cost_fp
    
    return {
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "total_cost": float(total_cost),
        "default_losses": float(default_losses),
        "friction_cost": float(friction_cost),
        "avg_cost_per_applicant": float(total_cost / len(y_true)) if len(y_true) > 0 else 0.0
    }
