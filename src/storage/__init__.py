"""
src/storage/__init__.py

Package exports for storage module.
"""

from .results import (
    TestResult,
    TestRun,
    ResultsStorage,
    generate_run_id,
)

__all__ = [
    "TestResult",
    "TestRun",
    "ResultsStorage",
    "generate_run_id",
]