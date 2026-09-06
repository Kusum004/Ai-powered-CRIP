# src/talk_to_data/__init__.py
"""Talk-to-Data NL-to-SQL Copilot with DuckDB OLAP execution and multi-provider LLM support."""
from src.talk_to_data.nl_to_sql import NLToSQLAgent
from src.talk_to_data.query_runner import QueryRunner
from src.talk_to_data.prompt_templates import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES

__all__ = ["NLToSQLAgent", "QueryRunner", "SYSTEM_PROMPT", "FEW_SHOT_EXAMPLES"]
