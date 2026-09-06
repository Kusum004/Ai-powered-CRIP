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
            raise FileNotFoundError(f"Raw dataset not found at {self.raw_data_path}")

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

    def load_raw_dataframe(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Loads dataset into Pandas DataFrame."""
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw dataset not found at {self.raw_data_path}")

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
