# notebooks/eda.py
"""Standalone Exploratory Data Analysis script generating the 5 key banking risk insights."""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import Config
from src.utils.logger import get_logger
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor

logger = get_logger("EDA")

def run_eda():
    """Generates the 5 key banking risk charts and summary statistics."""
    logger.info("Starting Portfolio Exploratory Data Analysis...")
    Config.ensure_directories()
    
    loader = DataLoader()
    df = loader.load_raw_dataframe(max_rows=None)
    preprocessor = DataPreprocessor()
    df_eng = preprocessor.engineer_features(df)

    output_dir = Config.DOCUMENTS_DIR / "eda_charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Style configuration
    plt.style.use("dark_background")
    sns.set_palette("muted")

    # 1. Default Rate by Income Bracket
    plt.figure(figsize=(10, 5))
    df_eng["INCOME_TIER"] = pd.qcut(df_eng["AMT_INCOME_TOTAL"], q=5, labels=["Very Low", "Low", "Medium", "High", "Very High"])
    inc_risk = df_eng.groupby("INCOME_TIER")["TARGET"].mean() * 100
    ax = inc_risk.plot(kind="bar", color="#3B82F6", edgecolor="#60A5FA")
    plt.title("Insight 1: Default Rate by Income Quintile", fontsize=14, fontweight="bold", color="white")
    plt.ylabel("Default Rate (%)", fontsize=12)
    plt.xlabel("Income Quintile", fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "1_income_vs_default.png", dpi=300)
    plt.close()

    # 2. Predictive Power of External Source Bureau Mean
    plt.figure(figsize=(10, 5))
    sns.kdeplot(df_eng[df_eng["TARGET"] == 0]["EXT_SOURCES_MEAN"].dropna(), label="Repaid (TARGET=0)", color="#10B981", fill=True, alpha=0.4)
    sns.kdeplot(df_eng[df_eng["TARGET"] == 1]["EXT_SOURCES_MEAN"].dropna(), label="Defaulted (TARGET=1)", color="#EF4444", fill=True, alpha=0.4)
    plt.title("Insight 2: Composite Bureau Score (EXT_SOURCES_MEAN) Distribution by Target", fontsize=14, fontweight="bold", color="white")
    plt.xlabel("EXT_SOURCES_MEAN (Higher = Better Credit Bureau History)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "2_ext_source_distribution.png", dpi=300)
    plt.close()

    # 3. Default Rate across Age Cohorts
    plt.figure(figsize=(10, 5))
    df_eng["AGE_COHORT"] = pd.cut(df_eng["AGE_YEARS"], bins=[20, 30, 40, 50, 60, 100], labels=["<30", "30-39", "40-49", "50-59", "60+"])
    age_risk = df_eng.groupby("AGE_COHORT")["TARGET"].mean() * 100
    age_risk.plot(kind="bar", color="#F59E0B", edgecolor="#FCD34D")
    plt.title("Insight 3: Historical Default Rate Across Age Cohorts", fontsize=14, fontweight="bold", color="white")
    plt.ylabel("Default Rate (%)", fontsize=12)
    plt.xlabel("Age Cohort", fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "3_age_vs_default.png", dpi=300)
    plt.close()

    # 4. Debt-Service-to-Income (Annuity / Income) Ratio
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="TARGET", y="ANNUITY_INCOME_PERC", data=df_eng[df_eng["ANNUITY_INCOME_PERC"] < 0.5], palette=["#10B981", "#EF4444"])
    plt.title("Insight 4: Debt-to-Income / DTI Distribution by Loan Outcome", fontsize=14, fontweight="bold", color="white")
    plt.xticks([0, 1], ["Repaid (0)", "Defaulted (1)"])
    plt.ylabel("Annuity / Total Income (DTI)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "4_dti_ratio.png", dpi=300)
    plt.close()

    # 5. Top 10 High Risk Occupations
    plt.figure(figsize=(12, 6))
    occ_df = df_eng.groupby("OCCUPATION_TYPE").agg(
        count=("TARGET", "count"),
        default_rate=("TARGET", lambda x: x.mean() * 100)
    ).query("count >= 500").sort_values(by="default_rate", ascending=True)
    
    occ_df["default_rate"].plot(kind="barh", color="#8B5CF6", edgecolor="#C4B5FD")
    plt.title("Insight 5: Default Rate by Occupation Category (N >= 500)", fontsize=14, fontweight="bold", color="white")
    plt.xlabel("Default Rate (%)", fontsize=12)
    plt.ylabel("Occupation Type", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "5_occupation_risk.png", dpi=300)
    plt.close()

    logger.info(f"All 5 EDA Insight figures successfully saved to {output_dir}")

if __name__ == "__main__":
    run_eda()
