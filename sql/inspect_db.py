# sql/inspect_db.py
"""Interactive CLI tool to inspect the DuckDB database, schema, views, and preview data."""
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import DataLoader, get_db_connection

def inspect_database():
    print("=" * 70)
    print("DUCKDB DATABASE & OLAP TABLE INSPECTION CONSOLE")
    print("=" * 70)
    
    loader = DataLoader()
    conn = loader.ingest_to_duckdb()
    
    # 1. List Tables and Views
    print("\n[1] Tables & Views Available in DuckDB:")
    tables_df = conn.execute("SHOW TABLES;").df()
    print(tables_df.to_string(index=False))
    
    # 2. Total Records
    count = conn.execute("SELECT COUNT(*) FROM loan_applications;").fetchone()[0]
    print(f"\n[2] Total Records in 'loan_applications': {count:,} rows")
    
    # 3. Column Schema
    print("\n[3] Column Schema Sample (First 15 Columns):")
    schema_df = conn.execute("DESCRIBE loan_applications;").df().head(15)
    print(schema_df.to_string(index=False))
    
    # 4. Preview First 5 Records
    print("\n[4] Data Preview (First 5 Rows):")
    preview_df = conn.execute("""
        SELECT 
            SK_ID_CURR, TARGET, CODE_GENDER, AMT_INCOME_TOTAL, 
            AMT_CREDIT, AMT_ANNUITY, NAME_EDUCATION_TYPE, OCCUPATION_TYPE
        FROM loan_applications 
        LIMIT 5;
    """).df()
    print(preview_df.to_string(index=False))
    
    # 5. Pre-built Analytical View Preview
    print("\n[5] Analytical View Preview (v_education_risk_summary):")
    view_df = conn.execute("SELECT * FROM v_education_risk_summary;").df()
    print(view_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Inspection complete. You can run custom SQL via 'DataLoader().execute_query(...)'.")
    print("=" * 70)

if __name__ == "__main__":
    inspect_database()
