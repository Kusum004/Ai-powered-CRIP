# src/talk_to_data/query_runner.py
"""AST SQL safety validator, latency-benchmarked DuckDB executor, and self-healing query runner."""
import time
import sqlparse
import pandas as pd
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger
from src.data.loader import get_db_connection, DataLoader

logger = get_logger("QueryRunner")

FORBIDDEN_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
    "EXEC", "EXECUTE", "ATTACH", "COPY", "PRAGMA", "GRANT", "REVOKE",
    "REPLACE", "CREATE"
}

class QueryRunner:
    def __init__(self):
        self.conn = get_db_connection()
        self.loader = DataLoader()
        # Ensure DuckDB table exists
        self.loader.ingest_to_duckdb()

    def validate_sql_safety(self, sql_query: str) -> Tuple[bool, Optional[str]]:
        """
        Parses SQL using AST tokenization to ensure query is strictly a read-only SELECT statement.
        """
        cleaned_sql = sql_query.strip().rstrip(";")
        if not cleaned_sql:
            return False, "Query cannot be empty."

        parsed = sqlparse.parse(cleaned_sql)
        if not parsed:
            return False, "Unable to parse SQL syntax."

        if len(parsed) > 1:
            return False, "Multiple SQL statements are not permitted for security reasons."

        stmt = parsed[0]
        stmt_type = stmt.get_type().upper()

        if stmt_type != "SELECT":
            return False, f"Security Violation: Only SELECT statements are allowed. (Found: {stmt_type})"

        # Check all tokens for forbidden write/modify operations
        for token in stmt.flatten():
            val = token.value.upper()
            if val in FORBIDDEN_KEYWORDS:
                # Check if it's not a harmless column alias (e.g. AS drop)
                if token.ttype in (sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DDL, sqlparse.tokens.Keyword.DML):
                    return False, f"Security Violation: Forbidden keyword '{val}' detected in query."

        return True, None

    def execute_safe_query(self, sql_query: str, max_rows: int = 500) -> Dict[str, Any]:
        """
        Validates, executes, and benchmarks execution time against DuckDB OLAP engine.
        """
        start_time = time.perf_counter()
        
        # Clean any accidental markdown backticks
        clean_sql = sql_query.strip()
        if clean_sql.startswith("```sql"):
            clean_sql = clean_sql[6:]
        elif clean_sql.startswith("```"):
            clean_sql = clean_sql[3:]
        if clean_sql.endswith("```"):
            clean_sql = clean_sql[:-3]
        clean_sql = clean_sql.strip().rstrip(";")

        # Safety AST check
        is_safe, err_msg = self.validate_sql_safety(clean_sql)
        if not is_safe:
            return {
                "success": False,
                "error": err_msg,
                "sql": clean_sql,
                "latency_ms": 0.0,
                "data": None,
                "row_count": 0
            }

        try:
            # Execute in DuckDB
            result_df = self.conn.execute(clean_sql).df()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # Cap output rows for memory
            capped_df = result_df.head(max_rows)

            logger.info(f"Executed safe SQL query in {elapsed_ms:.2f}ms | Rows returned: {len(result_df)}")

            return {
                "success": True,
                "error": None,
                "sql": clean_sql,
                "latency_ms": round(elapsed_ms, 2),
                "latency_badge": f"Executed in {elapsed_ms:.1f} ms",
                "data": capped_df,
                "row_count": len(result_df)
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"DuckDB Query execution failed ({elapsed_ms:.2f}ms): {e}")
            return {
                "success": False,
                "error": str(e),
                "sql": clean_sql,
                "latency_ms": round(elapsed_ms, 2),
                "data": None,
                "row_count": 0
            }
