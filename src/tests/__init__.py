"""
src/tests/__init__.py

Package exports for the tests module.
"""

from .test_cases import (
    TestCase,
    TestCategory,
    ScoringMethod,
    get_all_tests,
    get_tests_by_category,
    get_test_by_id,
)

from .scoring import (
    score_response,
    score_exact,
    score_numeric,
    score_rubric,
    score_format,
    explain_score,
)

__all__ = [
    # Test case classes and enums
    "TestCase",
    "TestCategory", 
    "ScoringMethod",
    # Test retrieval functions
    "get_all_tests",
    "get_tests_by_category",
    "get_test_by_id",
    # Scoring functions
    "score_response",
    "score_exact",
    "score_numeric",
    "score_rubric",
    "score_format",
    "explain_score",
]