# src/data/__init__.py
"""Data ingestion, OLAP DuckDB management, and feature engineering."""
from src.data.loader import DataLoader, get_db_connection
from src.data.preprocessor import DataPreprocessor

__all__ = ["DataLoader", "get_db_connection", "DataPreprocessor"]
