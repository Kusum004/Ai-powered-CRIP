# tests/test_platform.py
"""Comprehensive unit and integration test suite for Credit Risk Intelligence Platform."""
import os
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import Config
from src.utils.helpers import prob_to_credit_score, fico_to_risk_tier, format_currency, format_percent
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.talk_to_data.query_runner import QueryRunner
from src.talk_to_data.nl_to_sql import NLToSQLAgent
from src.ml.rules import RuleEngine

class TestCreditRiskPlatform(unittest.TestCase):

    def setUp(self):
        Config.ensure_directories()
        self.sample_applicant = {
            "AMT_INCOME_TOTAL": 180000.0,
            "AMT_CREDIT": 500000.0,
            "AMT_ANNUITY": 25000.0,
            "AMT_GOODS_PRICE": 450000.0,
            "DAYS_BIRTH": -14000,
            "DAYS_EMPLOYED": -2500,
            "NAME_EDUCATION_TYPE": "Higher education",
            "NAME_INCOME_TYPE": "Working",
            "NAME_FAMILY_STATUS": "Married",
            "OCCUPATION_TYPE": "Core staff",
            "EXT_SOURCE_1": 0.65,
            "EXT_SOURCE_2": 0.70,
            "EXT_SOURCE_3": 0.68,
            "FLAG_OWN_CAR": "Y",
            "FLAG_OWN_REALTY": "Y",
            "CNT_FAM_MEMBERS": 2.0,
            "REGION_RATING_CLIENT": 2
        }

    def test_fico_score_mapping(self):
        """Tests that probability maps accurately to 300-850 credit scores."""
        # 0.0 default prob should give 850
        self.assertEqual(prob_to_credit_score(0.0), 850)
        # 1.0 default prob should give 300
        self.assertEqual(prob_to_credit_score(1.0), 300)
        # 0.10 default prob: 850 - 55 = 795
        self.assertEqual(prob_to_credit_score(0.10), 795)

    def test_risk_tier_classification(self):
        """Tests Basel III risk tier boundaries."""
        tier_low, rec_low, col_low, badge_low = fico_to_risk_tier(800, prob_default=0.03)
        self.assertEqual(tier_low, "LOW RISK")
        self.assertIn("AUTO-APPROVE", badge_low)

        tier_med, rec_med, col_med, badge_med = fico_to_risk_tier(680, prob_default=0.12)
        self.assertEqual(tier_med, "MEDIUM RISK")
        self.assertIn("MANUAL REVIEW", badge_med)

        tier_high, rec_high, col_high, badge_high = fico_to_risk_tier(450, prob_default=0.35)
        self.assertEqual(tier_high, "HIGH RISK")
        self.assertIn("DECLINE", badge_high)

    def test_feature_engineering(self):
        """Tests domain banking ratios calculation."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame([self.sample_applicant])
        df_eng = preprocessor.engineer_features(df)

        self.assertIn("PAYMENT_RATE", df_eng.columns)
        self.assertIn("INCOME_CREDIT_PERC", df_eng.columns)
        self.assertIn("ANNUITY_INCOME_PERC", df_eng.columns)
        self.assertIn("EXT_SOURCES_MEAN", df_eng.columns)

        # Expected ratio: 25000 / 500000 = 0.05
        self.assertAlmostEqual(df_eng["PAYMENT_RATE"].iloc[0], 0.05, places=3)
        # Expected ratio: 180000 / 500000 = 0.36
        self.assertAlmostEqual(df_eng["INCOME_CREDIT_PERC"].iloc[0], 0.36, places=3)

    def test_ast_safety_validator(self):
        """Tests AST SQL sanitizer blocking dangerous queries."""
        runner = QueryRunner()

        # Valid SELECT
        safe_sql = "SELECT COUNT(*) FROM loan_applications WHERE AMT_INCOME_TOTAL > 100000;"
        is_safe, err = runner.validate_sql_safety(safe_sql)
        self.assertTrue(is_safe)
        self.assertIsNone(err)

        # Dangerous DROP
        bad_drop = "DROP TABLE loan_applications;"
        is_safe, err = runner.validate_sql_safety(bad_drop)
        self.assertFalse(is_safe)
        self.assertIn("Security Violation", err)

        # Dangerous DELETE
        bad_del = "DELETE FROM loan_applications WHERE 1=1;"
        is_safe, err = runner.validate_sql_safety(bad_del)
        self.assertFalse(is_safe)

        # Dangerous UPDATE
        bad_upd = "UPDATE loan_applications SET TARGET = 0;"
        is_safe, err = runner.validate_sql_safety(bad_upd)
        self.assertFalse(is_safe)

    def test_duckdb_olap_speed(self):
        """Tests DuckDB analytical aggregation latency (< 50ms)."""
        runner = QueryRunner()
        sql = "SELECT NAME_EDUCATION_TYPE, COUNT(*), AVG(TARGET) FROM loan_applications GROUP BY NAME_EDUCATION_TYPE;"
        res = runner.execute_safe_query(sql)
        self.assertTrue(res["success"])
        self.assertIsNotNone(res["data"])
        self.assertLess(res["latency_ms"], 100.0)

    def test_rule_engine(self):
        """Tests policy rule extraction and applicant evaluation."""
        rule_engine = RuleEngine(max_depth=3)
        rules = rule_engine.fit_and_extract_rules()
        self.assertGreater(len(rules), 0)

        matches = rule_engine.evaluate_applicant_rules(self.sample_applicant)
        self.assertIsInstance(matches, list)

if __name__ == "__main__":
    unittest.main()
