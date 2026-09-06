# src/data/loader.py
"""DuckDB in-memory OLAP database manager and data ingestion pipeline."""
import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("DataLoader")

_DB_CONN: Optional[duckdb.DuckDBPyConnection] = None

def get_db_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Returns a singleton DuckDB connection."""
    global _DB_CONN
    if _DB_CONN is None:
        Config.ensure_directories()
        # Use in-memory connection for ultra-fast OLAP speed
        _DB_CONN = duckdb.connect(database=":memory:", read_only=False)
        logger.info("Initialized DuckDB in-memory database instance.")
    return _DB_CONN

class DataLoader:
    def __init__(self, raw_data_path: Optional[Path] = None):
        self.raw_data_path = Path(raw_data_path) if raw_data_path else Config.RAW_DATA_PATH
        self.conn = get_db_connection()
        self._is_ingested = False

    def ingest_to_duckdb(self, force_reload: bool = False) -> duckdb.DuckDBPyConnection:
        """
        Ingests the application_train.csv directly into DuckDB table 'loan_applications'
        using DuckDB's multi-threaded C++ CSV reader.
        """
        if self._is_ingested and not force_reload:
            return self.conn

        if not self.raw_data_path.exists():
            logger.info("Raw dataset CSV not found on disk (e.g. Cloud deployment). Generating realistic benchmark dataset...")
            df_synth = self._generate_benchmark_df(n_samples=5000)
            self.conn.register("df_synth", df_synth)
            self.conn.execute("CREATE OR REPLACE TABLE loan_applications AS SELECT * FROM df_synth;")
        else:
            logger.info(f"Ingesting {self.raw_data_path.name} into DuckDB OLAP engine...")
            
            # Load schema SQL if exists
            schema_file = Config.SQL_DIR / "schema.sql"
            if schema_file.exists():
                with open(schema_file, "r", encoding="utf-8") as f:
                    schema_sql = f.read()
                # Split and execute individual DDL statements safely
                for statement in schema_sql.split(";"):
                    stmt = statement.strip()
                    if stmt and not stmt.startswith("--"):
                        try:
                            self.conn.execute(stmt)
                        except Exception as e:
                            logger.warning(f"Schema statement notice: {e}")

            # Ingest CSV directly into loan_applications table or view
            csv_path_str = str(self.raw_data_path).replace("\\", "/")
            query = f"""
                CREATE OR REPLACE TABLE loan_applications AS 
                SELECT * FROM read_csv_auto('{csv_path_str}', header=True, sample_size=20000);
            """
            self.conn.execute(query)
            
            # Re-apply analytical views
            if schema_file.exists():
                with open(schema_file, "r", encoding="utf-8") as f:
                    for statement in f.read().split(";"):
                        stmt = statement.strip()
                        if stmt and "VIEW" in stmt.upper():
                            try:
                                self.conn.execute(stmt)
                            except Exception as e:
                                logger.warning(f"View re-creation notice: {e}")

        count = self.conn.execute("SELECT COUNT(*) FROM loan_applications;").fetchone()[0]
        logger.info(f"Successfully ingested {count:,} records into DuckDB 'loan_applications' table.")
        self._is_ingested = True
        return self.conn

    def _generate_benchmark_df(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generates realistic synthetic loan dataset matching schema for cloud deployment."""
        import numpy as np
        np.random.seed(42)
        target = np.random.choice([0, 1], size=n_samples, p=[0.9193, 0.0807])
        
        income = np.random.lognormal(mean=11.8, sigma=0.6, size=n_samples).clip(25000, 2000000)
        credit = income * np.random.uniform(1.5, 6.0, size=n_samples)
        annuity = credit / np.random.uniform(12, 36, size=n_samples)
        goods_price = credit * np.random.uniform(0.85, 1.0, size=n_samples)
        
        age_years = np.random.uniform(21, 68, size=n_samples)
        days_birth = - (age_years * 365.25).astype(int)
        
        emp_years = np.random.uniform(0.5, 35, size=n_samples)
        days_employed = - (emp_years * 365.25).astype(int)
        
        ext_1 = np.random.beta(5, 5, size=n_samples)
        ext_2 = np.random.beta(5, 5, size=n_samples)
        ext_3 = np.random.beta(5, 5, size=n_samples)
        
        educations = ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary", "Academic degree"]
        incomes = ["Working", "Commercial associate", "State servant", "Pensioner"]
        occupations = ["Core staff", "Managers", "Laborers", "Sales staff", "Drivers", "Accountants", "High skill tech staff"]
        family = ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
        
        df = pd.DataFrame({
            "SK_ID_CURR": np.arange(100000, 100000 + n_samples),
            "TARGET": target,
            "NAME_CONTRACT_TYPE": np.random.choice(["Cash loans", "Revolving loans"], size=n_samples, p=[0.9, 0.1]),
            "CODE_GENDER": np.random.choice(["M", "F"], size=n_samples, p=[0.35, 0.65]),
            "FLAG_OWN_CAR": np.random.choice(["Y", "N"], size=n_samples, p=[0.34, 0.66]),
            "FLAG_OWN_REALTY": np.random.choice(["Y", "N"], size=n_samples, p=[0.69, 0.31]),
            "CNT_CHILDREN": np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.7, 0.2, 0.08, 0.02]),
            "AMT_INCOME_TOTAL": np.round(income, 2),
            "AMT_CREDIT": np.round(credit, 2),
            "AMT_ANNUITY": np.round(annuity, 2),
            "AMT_GOODS_PRICE": np.round(goods_price, 2),
            "NAME_INCOME_TYPE": np.random.choice(incomes, size=n_samples),
            "NAME_EDUCATION_TYPE": np.random.choice(educations, size=n_samples, p=[0.24, 0.70, 0.03, 0.02, 0.01]),
            "NAME_FAMILY_STATUS": np.random.choice(family, size=n_samples),
            "NAME_HOUSING_TYPE": np.random.choice(["House / apartment", "With parents", "Municipal apartment", "Rented apartment"], size=n_samples),
            "REGION_POPULATION_RELATIVE": np.random.uniform(0.001, 0.07, size=n_samples),
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed,
            "DAYS_REGISTRATION": - np.random.uniform(0, 15000, size=n_samples),
            "DAYS_ID_PUBLISH": - np.random.uniform(0, 7000, size=n_samples).astype(int),
            "OWN_CAR_AGE": np.random.uniform(1, 25, size=n_samples),
            "OCCUPATION_TYPE": np.random.choice(occupations, size=n_samples),
            "CNT_FAM_MEMBERS": np.random.choice([1.0, 2.0, 3.0, 4.0], size=n_samples),
            "REGION_RATING_CLIENT": np.random.choice([1, 2, 3], size=n_samples, p=[0.1, 0.7, 0.2]),
            "REGION_RATING_CLIENT_W_CITY": np.random.choice([1, 2, 3], size=n_samples, p=[0.1, 0.7, 0.2]),
            "ORGANIZATION_TYPE": np.random.choice(["Business Entity Type 3", "Self-employed", "Other", "Government"], size=n_samples),
            "EXT_SOURCE_1": ext_1,
            "EXT_SOURCE_2": ext_2,
            "EXT_SOURCE_3": ext_3,
            "DEF_30_CNT_SOCIAL_CIRCLE": np.random.choice([0.0, 1.0, 2.0], size=n_samples, p=[0.85, 0.12, 0.03]),
            "DEF_60_CNT_SOCIAL_CIRCLE": np.random.choice([0.0, 1.0], size=n_samples, p=[0.92, 0.08]),
            "DAYS_LAST_PHONE_CHANGE": - np.random.uniform(0, 4000, size=n_samples),
            "AMT_REQ_CREDIT_BUREAU_YEAR": np.random.choice([0.0, 1.0, 2.0, 3.0], size=n_samples, p=[0.6, 0.25, 0.1, 0.05])
        })
        return df

    def load_raw_dataframe(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Loads dataset into Pandas DataFrame."""
        if not self.raw_data_path.exists():
            return self._generate_benchmark_df(n_samples=5000)

        logger.info(f"Loading raw DataFrame from {self.raw_data_path.name} (max_rows={max_rows})...")
        if max_rows:
            df = pd.read_csv(self.raw_data_path, nrows=max_rows)
        else:
            df = pd.read_csv(self.raw_data_path)
        logger.info(f"Loaded DataFrame with shape: {df.shape}")
        return df

    def execute_query(self, query: str) -> pd.DataFrame:
        """Executes a SQL query against DuckDB and returns a Pandas DataFrame."""
        if not self._is_ingested:
            self.ingest_to_duckdb()
        return self.conn.execute(query).df()

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Calculates core portfolio executive KPIs."""
        if not self._is_ingested:
            self.ingest_to_duckdb()
            
        summary_query = """
            SELECT 
                COUNT(*) AS total_loans,
                SUM(TARGET) AS total_defaulters,
                ROUND(AVG(TARGET) * 100.0, 2) AS overall_default_rate_pct,
                ROUND(AVG(AMT_INCOME_TOTAL), 2) AS avg_applicant_income,
                ROUND(AVG(AMT_CREDIT), 2) AS avg_credit_amount,
                ROUND(AVG(AMT_ANNUITY), 2) AS avg_monthly_annuity,
                ROUND(SUM(AMT_CREDIT), 2) AS total_portfolio_exposure
            FROM loan_applications;
        """
        res = self.conn.execute(summary_query).df().to_dict(orient="records")[0]
        return res

    def get_column_descriptions(self) -> Dict[str, str]:
        """Loads column descriptions dictionary from metadata CSV."""
        meta_path = Config.METADATA_PATH
        if meta_path.exists():
            df_meta = pd.read_csv(meta_path, encoding="latin1")
            # Map Row to Description
            desc_map = {}
            for _, row in df_meta.iterrows():
                col = str(row.get("Row", "")).strip()
                desc = str(row.get("Description", "")).strip()
                if col:
                    desc_map[col] = desc
            return desc_map
        return {}
