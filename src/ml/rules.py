# src/ml/rules.py
"""Automated Credit Policy Business Rule Extractor using Decision Tree Induction."""
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import Config
from src.utils.logger import get_logger
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor

logger = get_logger("RuleEngine")

class RuleEngine:
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.tree_model: Optional[DecisionTreeClassifier] = None
        self.rules: List[Dict[str, Any]] = []
        self.rule_feature_names: List[str] = [
            "EXT_SOURCES_MEAN",
            "ANNUITY_INCOME_PERC",
            "PAYMENT_RATE",
            "INCOME_CREDIT_PERC",
            "DAYS_EMPLOYED_PERC",
            "AGE_YEARS",
            "REGION_RATING_CLIENT"
        ]

    def fit_and_extract_rules(self, df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """
        Fits a shallow decision tree on core banking domain ratios to induce transparent policy rules.
        """
        if df is None:
            loader = DataLoader()
            df = loader.load_raw_dataframe(max_rows=50000)

        preprocessor = DataPreprocessor()
        df_eng = preprocessor.engineer_features(df)

        # Filter available features
        feats = [f for f in self.rule_feature_names if f in df_eng.columns]
        X = df_eng[feats].copy()
        
        # Fill missing values with median for tree extraction
        for col in feats:
            X[col] = X[col].fillna(X[col].median())
            
        y = df_eng["TARGET"].values

        # Fit decision tree with class weight balanced
        clf = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=150,
            class_weight="balanced",
            random_state=Config.RANDOM_STATE
        )
        clf.fit(X, y)
        self.tree_model = clf

        # Extract rules from tree structure
        self.rules = self._extract_rules_from_tree(clf, feats, X.values, y)
        
        # Save rules to disk
        with open(Config.RULES_PATH, "w") as f:
            json.dump(self.rules, f, indent=2)
            
        logger.info(f"Extracted {len(self.rules)} credit policy rules from decision tree.")
        return self.rules

    def _extract_rules_from_tree(self, clf: DecisionTreeClassifier, feature_names: List[str], X: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
        """Traverses the decision tree structure to extract IF-THEN rules with empirical portfolio statistics."""
        tree_ = clf.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        # Get true leaf assignment for all training records
        leaf_indices = clf.apply(X)
        rules = []

        def recurse(node, conditions):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]

                # Left branch (<= threshold)
                left_cond = conditions + [f"{name} <= {threshold:.4f}"]
                recurse(tree_.children_left[node], left_cond)

                # Right branch (> threshold)
                right_cond = conditions + [f"{name} > {threshold:.4f}"]
                recurse(tree_.children_right[node], right_cond)
            else:
                # Leaf node: compute true empirical counts across empirical samples
                mask = (leaf_indices == node)
                total_samples = int(np.sum(mask))
                if total_samples > 0:
                    defaulters = int(np.sum(y[mask]))
                    default_rate = defaulters / total_samples
                else:
                    total_samples = int(tree_.n_node_samples[node])
                    val = tree_.value[node][0]
                    defaulters = int(val[1] / max(1e-5, np.sum(val)) * total_samples)
                    default_rate = defaulters / total_samples if total_samples > 0 else 0.08

                # Determine Calibrated Basel III Risk Tier
                if default_rate < 0.06:
                    tier = "LOW RISK"
                    action = "Auto-Approve: Prime pricing, instant digital disbursal"
                    badge_color = "#10B981"
                elif default_rate <= 0.15:
                    tier = "MEDIUM RISK"
                    action = "Manual Underwriting: Secondary income verification & debt check"
                    badge_color = "#F59E0B"
                else:
                    tier = "HIGH RISK"
                    action = "Decline / Restructure: High delinquency tier, reject unsecured loan"
                    badge_color = "#EF4444"

                rule_text = " AND ".join(conditions) if conditions else "ALL APPLICANTS"

                rules.append({
                    "rule_id": f"RULE-{len(rules) + 1:02d}",
                    "condition": rule_text,
                    "leaf_samples": total_samples,
                    "leaf_defaulters": defaulters,
                    "empirical_default_rate_pct": round(default_rate * 100.0, 2),
                    "assigned_tier": tier,
                    "recommended_action": action,
                    "badge_color": badge_color
                })

        recurse(0, [])
        # Sort rules from safest to highest risk
        rules_sorted = sorted(rules, key=lambda r: r["empirical_default_rate_pct"])
        for idx, r in enumerate(rules_sorted, 1):
            r["rule_id"] = f"RULE-{idx:02d}"
        return rules_sorted

    def evaluate_applicant_rules(self, applicant_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Checks which credit policy rules match the applicant profile.
        """
        if not self.rules:
            if Config.RULES_PATH.exists():
                with open(Config.RULES_PATH, "r") as f:
                    self.rules = json.load(f)
            else:
                self.fit_and_extract_rules()

        # Compute applicant financial ratios if missing
        app = dict(applicant_dict)
        inc = float(app.get("AMT_INCOME_TOTAL", 150000.0))
        cred = float(app.get("AMT_CREDIT", 500000.0))
        ann = float(app.get("AMT_ANNUITY", 25000.0))
        birth = float(app.get("DAYS_BIRTH", -14000.0))
        emp = float(app.get("DAYS_EMPLOYED", -2000.0))
        ext1 = float(app.get("EXT_SOURCE_1", 0.5))
        ext2 = float(app.get("EXT_SOURCE_2", 0.5))
        ext3 = float(app.get("EXT_SOURCE_3", 0.5))

        app["ANNUITY_INCOME_PERC"] = ann / (inc + 1e-5)
        app["PAYMENT_RATE"] = ann / (cred + 1e-5)
        app["INCOME_CREDIT_PERC"] = inc / (cred + 1e-5)
        app["DAYS_EMPLOYED_PERC"] = abs(emp) / (abs(birth) + 1e-5)
        app["AGE_YEARS"] = abs(birth) / 365.25
        app["EXT_SOURCES_MEAN"] = (ext1 + ext2 + ext3) / 3.0
        app["REGION_RATING_CLIENT"] = float(app.get("REGION_RATING_CLIENT", 2.0))

        matching_rules = []
        for rule in self.rules:
            cond = rule["condition"]
            if cond == "ALL APPLICANTS":
                matching_rules.append(rule)
                continue

            try:
                # Safe evaluation environment
                local_vars = {k: float(v) for k, v in app.items() if isinstance(v, (int, float))}
                py_expr = cond.replace(" AND ", " and ")
                if eval(py_expr, {"__builtins__": None}, local_vars):
                    matching_rules.append(rule)
            except Exception:
                continue

        return matching_rules
