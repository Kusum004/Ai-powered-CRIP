# src/utils/config.py
"""Central configuration settings for Credit Risk Platform."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables (.env with .env.example fallback)
env_file = BASE_DIR / ".env"
if not env_file.exists():
    env_file = BASE_DIR / ".env.example"
load_dotenv(dotenv_path=env_file, override=True)

class Config:
    # Directories
    BASE_DIR = BASE_DIR
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    DOCUMENTS_DIR = BASE_DIR / "documents"
    SQL_DIR = BASE_DIR / "sql"
    
    # Data Files
    RAW_DATA_PATH = DATA_DIR / "application_train.csv"
    METADATA_PATH = DATA_DIR / "HomeCredit_columns_description.csv"
    DUCKDB_PATH = DATA_DIR / "credit_risk.duckdb"
    
    # Model Artifacts
    CHAMPION_MODEL_PATH = MODELS_DIR / "lgb_champion.joblib"
    BASELINE_MODEL_PATH = MODELS_DIR / "lr_baseline.joblib"
    PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
    METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"
    RULES_PATH = MODELS_DIR / "decision_rules.json"
    FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"
    
    # ML Hyperparameters
    SCALE_POS_WEIGHT = 11.387  # 282686 / 24825
    RANDOM_STATE = 42
    N_FOLDS = 5
    EARLY_STOPPING_ROUNDS = 50
    
    # Risk Thresholds
    LOW_RISK_PROB = 0.07     # Score >= 750 (Auto-Approve)
    HIGH_RISK_PROB = 0.20    # Score < 600 (Decline / Restructure)
    
    # Cost Matrix ($)
    COST_FALSE_NEGATIVE = 10000.0  # Loss on default
    COST_FALSE_POSITIVE = 1000.0   # Opportunity cost of lost good customer
    
    # LLM Provider Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama-3.3-70b-versatile")
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.SQL_DIR.mkdir(parents=True, exist_ok=True)
