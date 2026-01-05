"""
src/tests/scoring.py

Functions to score model responses against expected answers.

SCORING PHILOSOPHY:
- Scores are always 0.0 to 1.0 (0 = completely wrong, 1 = completely correct)
- Scoring should be DETERMINISTIC - same input always gives same score
- Scoring should be LENIENT where appropriate - "636" in "The answer is 636" is correct
- Scoring should handle edge cases gracefully

SCORING METHODS:
1. EXACT - Check if expected answer appears in response (case-insensitive)
2. NUMERIC - Extract numbers and check if any match expected (with tolerance)
3. RUBRIC - Custom logic for specific questions
4. FORMAT - Check if response follows format constraints
5. CREATIVE_STRUCTURE - Check word/sentence/line counts for creative writing
6. CODE_CHECK - Check for keywords in code responses
"""

import re
from typing import Optional
from .test_cases import TestCase, ScoringMethod


def score_response(test: TestCase, response: str) -> float:
    """
    Score a model's response for a given test.
    
    This is the main entry point for scoring. It dispatches to the
    appropriate scoring function based on the test's scoring_method.
    
    Args:
        test: The TestCase that was run
        response: The model's response text
    
    Returns:
        Float between 0.0 and 1.0
    """
    if not response or response.startswith("ERROR:"):
        return 0.0
    
    method = test.scoring_method
    
    # Handle Enum types (old tests)
    if hasattr(method, 'value'):
        method_val = method.value
    else:
        method_val = method
    
    # Convert to string for comparison
    if isinstance(method_val, str):
        method_str = method_val.lower()
        
        if method_str == "exact":
            return score_exact(response, test.expected)
        elif method_str == "numeric":
            return score_numeric(response, test.expected)
        elif method_str == "rubric":
            return score_rubric(response, test)
        elif method_str == "format":
            return score_format(response, test)
        elif method_str == "creative_structure":
            return score_creative_structure(response, test.id, test.metadata or {})
        elif method_str == "code_check":
            return score_code_check(response, test.id, test.metadata or {})
        else:
            return 0.0
    
    # Fallback for Enum comparison
    if method == ScoringMethod.EXACT:
        return score_exact(response, test.expected)
    elif method == ScoringMethod.NUMERIC:
        return score_numeric(response, test.expected)
    elif method == ScoringMethod.RUBRIC:
        return score_rubric(response, test)
    elif method == ScoringMethod.FORMAT:
        return score_format(response, test)
    else:
        return 0.0


def score_exact(response: str, expected: str) -> float:
    """
    Score based on exact match.
    
    Checks if the expected answer appears anywhere in the response.
    This is lenient - "The answer is Paris" matches expected "Paris".
    
    Args:
        response: Model's response text
        expected: The expected answer
    
    Returns:
        1.0 if match found, 0.0 otherwise
    
    Examples:
        >>> score_exact("The capital of France is Paris.", "Paris")
        1.0
        >>> score_exact("I think it might be London", "Paris")
        0.0
        >>> score_exact("PARIS", "Paris")  # Case insensitive
        1.0
    """
    if not expected:
        return 0.0
    
    # Clean both strings
    response_clean = response.lower().strip()
    expected_clean = expected.lower().strip()
    
    # Check if expected is contained in response
    if expected_clean in response_clean:
        return 1.0
    
    # Also check without punctuation (handles "Yes." vs "yes")
    response_alpha = re.sub(r'[^a-z0-9\s]', '', response_clean)
    expected_alpha = re.sub(r'[^a-z0-9\s]', '', expected_clean)
    
    if expected_alpha in response_alpha:
        return 1.0
    
    return 0.0


def score_numeric(response: str, expected: str, tolerance: float = 0.01) -> float:
    """
    Score based on numeric match with tolerance.
    
    Extracts all numbers from the response and checks if any match
    the expected value within the given tolerance.
    
    Args:
        response: Model's response text
        expected: The expected numeric answer (as string)
        tolerance: Relative tolerance (0.01 = 1%)
    
    Returns:
        1.0 if matching number found, 0.0 otherwise
    
    Examples:
        >>> score_numeric("The answer is 636", "636")
        1.0
        >>> score_numeric("Approximately 36.0 dollars", "36")
        1.0
        >>> score_numeric("I think 37", "36")  # Outside tolerance
        0.0
    """
    if not expected:
        return 0.0
    
    try:
        expected_num = float(expected)
    except ValueError:
        return 0.0
    
    # Extract all numbers from response
    # This regex matches integers and decimals, including negative
    numbers = re.findall(r'-?\d+\.?\d*', response)
    
    if not numbers:
        return 0.0
    
    for num_str in numbers:
        try:
            num = float(num_str)
            
            # Check exact match first
            if num == expected_num:
                return 1.0
            
            # Check within tolerance
            # For expected=0, use absolute tolerance
            if expected_num == 0:
                if abs(num) < tolerance:
                    return 1.0
            else:
                relative_diff = abs(num - expected_num) / abs(expected_num)
                if relative_diff <= tolerance:
                    return 1.0
                    
        except ValueError:
            continue
    
    return 0.0


def score_rubric(response: str, test: TestCase) -> float:
    """
    Score using custom rubric logic for specific questions.
    
    Some questions don't have simple exact answers and need
    custom logic to evaluate. This function handles those cases.
    
    Args:
        response: Model's response text
        test: The TestCase (used to identify which rubric to apply)
    
    Returns:
        Float between 0.0 and 1.0
    """
    resp_lower = response.lower().strip()
    test_id = test.id
    
    # reason_001: Syllogism - correct answer is "no, we cannot conclude"
    if test_id == "reason_001" or test_id == "reason_adv_001":
        # Check for negative indicators
        if any(word in resp_lower for word in ["no", "cannot", "can't", "not valid", "invalid"]):
            return 1.0
        if "yes" in resp_lower and "cannot" not in resp_lower:
            return 0.0
        return 0.0
    
    # reason_004: Tom/Mary/John puzzle - correct answer is "yes"
    if test_id == "reason_004":
        # The answer is "yes" - Mary must be either married or unmarried
        # If married: married Mary looks at unmarried John
        # If unmarried: married Tom looks at unmarried Mary
        # Either way, a married person is looking at an unmarried person
        if "yes" in resp_lower and "cannot" not in resp_lower and "not" not in resp_lower:
            return 1.0
        return 0.0
    
    # reason_008: Feathers vs steel - they weigh the same (both are a pound)
    if test_id == "reason_008":
        if any(word in resp_lower for word in ["same", "equal", "neither", "both"]):
            return 1.0
        # Penalize if they say one is heavier
        if "feathers" in resp_lower and "heavier" in resp_lower:
            return 0.0
        if "steel" in resp_lower and "heavier" in resp_lower:
            return 0.0
        return 0.0
    
    # reason_adv_002: Water jug problem
    if test_id == "reason_adv_002":
        # Check for key concepts: fill, pour, gallon
        key_words = ["fill", "pour", "gallon"]
        matches = sum(1 for word in key_words if word in resp_lower)
        if matches >= 2:
            return 1.0
        elif matches == 1:
            return 0.5
        return 0.0
    
    # common_001: Two doors puzzle
    if test_id == "common_001":
        # Check for key concepts: other guard, opposite
        if ("other" in resp_lower or "opposite" in resp_lower) and "guard" in resp_lower:
            return 1.0
        return 0.0
    
    # common_003: Monopoly
    if test_id == "common_003":
        if "monopoly" in resp_lower or "game" in resp_lower or "board" in resp_lower:
            return 1.0
        return 0.0
    
    # Default: try exact match with expected
    if test.expected:
        return score_exact(response, test.expected)
    
    return 0.0


def score_format(response: str, test: TestCase) -> float:
    """
    Score based on format compliance.
    
    Checks if the response follows the format constraints
    specified in the test prompt.
    
    Args:
        response: Model's response text
        test: The TestCase (used to identify format requirements)
    
    Returns:
        1.0 if format is correct, 0.0 otherwise, 0.5 for partial
    """
    resp = response.strip()
    test_id = test.id
    
    # instr_001: List exactly 3 colors
    if test_id == "instr_001":
        # Count items (lines or comma-separated)
        lines = [l.strip() for l in resp.split('\n') if l.strip()]
        if len(lines) == 3:
            return 1.0
        # Try comma-separated
        items = [i.strip() for i in resp.split(',') if i.strip()]
        if len(items) == 3:
            return 1.0
        return 0.0
    
    # instr_002: Single word response
    if test_id == "instr_002":
        words = resp.split()
        if len(words) == 1:
            return 1.0
        return 0.0
    
    # instr_003: Exactly 5 words
    if test_id == "instr_003":
        words = resp.split()
        if len(words) == 5:
            return 1.0
        return 0.0
    
    # instr_004: Just a digit 1-10
    if test_id == "instr_004":
        # Remove any whitespace and check if it's a single number 1-10
        clean = resp.strip()
        try:
            num = int(clean)
            if 1 <= num <= 10:
                return 1.0
        except ValueError:
            pass
        return 0.0
    
    # instr_005: Exactly 4 words
    if test_id == "instr_005":
        words = resp.split()
        if len(words) == 4:
            return 1.0
        return 0.0
    
    # instr_006: All caps
    if test_id == "instr_006":
        # Check if all alphabetic characters are uppercase
        alpha_chars = [c for c in resp if c.isalpha()]
        if alpha_chars and all(c.isupper() for c in alpha_chars):
            return 1.0
        return 0.0
    
    # instr_007: Only 'yes' or 'no'
    if test_id == "instr_007":
        clean = resp.lower().strip().rstrip('.')
        if clean in ['yes', 'no']:
            return 1.0
        return 0.0
    
    # instr_008: Numbers 10-6 descending, comma-separated
    if test_id == "instr_008":
        # Extract numbers
        numbers = re.findall(r'\d+', resp)
        try:
            nums = [int(n) for n in numbers]
            if nums == [10, 9, 8, 7, 6]:
                return 1.0
        except ValueError:
            pass
        return 0.0
    
    # Default
    return 0.0


def score_creative_structure(response: str, test_id: str, metadata: dict) -> float:
    """
    Score creative writing tests based on structure.
    
    Args:
        response: Model's response text
        test_id: Test identifier
        metadata: Dict containing expected structure (sentences, words, lines)
    
    Returns:
        Float between 0.0 and 1.0
    """
    # Check for expected sentences
    if "expected_sentences" in metadata:
        expected = metadata["expected_sentences"]
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if len(sentences) == expected:
            # Check max words if specified
            if "max_words" in metadata:
                word_count = len(response.split())
                if word_count <= metadata["max_words"]:
                    return 1.0
                else:
                    return 0.7  # Right structure, too long
            return 1.0
        return 0.0
    
    # Check for expected words
    if "expected_words" in metadata:
        expected = metadata["expected_words"]
        words = response.strip().split()
        
        if len(words) == expected:
            return 1.0
        elif abs(len(words) - expected) <= 2:  # Within 2 words tolerance
            return 0.7
        return 0.0
    
    # Check for expected lines (e.g., haiku)
    if "expected_lines" in metadata:
        expected = metadata["expected_lines"]
        # Try both / separator and newlines
        lines = [l.strip() for l in response.split('/') if l.strip()]
        if len(lines) != expected:
            lines = [l.strip() for l in response.split('\n') if l.strip()]
        
        if len(lines) == expected:
            return 1.0
        return 0.0
    
    return 0.5  # Default partial credit


def score_code_check(response: str, test_id: str, metadata: dict) -> float:
    """
    Score coding tests by checking for key elements.
    
    Args:
        response: Model's response text
        test_id: Test identifier
        metadata: Dict containing keywords to check for
    
    Returns:
        Float between 0.0 and 1.0
    """
    if "keywords" not in metadata:
        return 0.5
    
    keywords = metadata["keywords"]
    resp_lower = response.lower()
    
    # Count how many keywords are present
    found = 0
    for keyword in keywords:
        if keyword.lower() in resp_lower or keyword in response:
            found += 1
    
    # Score based on percentage of keywords found
    percentage = found / len(keywords) if keywords else 0
    
    if percentage >= 0.75:  # 75%+ of keywords
        return 1.0
    elif percentage >= 0.5:  # 50%+ of keywords
        return 0.7
    elif percentage >= 0.25:  # 25%+ of keywords
        return 0.4
    else:
        return 0.0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def explain_score(test: TestCase, response: str, score: float) -> str:
    """
    Generate a human-readable explanation of why a response got its score.
    
    Useful for debugging and for the dashboard.
    
    Args:
        test: The TestCase
        response: The model's response
        score: The score that was assigned
    
    Returns:
        Human-readable explanation string
    """
    if score == 1.0:
        return "✅ Correct"
    elif score == 0.0:
        if test.expected:
            return f"❌ Expected '{test.expected}' but got '{response[:50]}...'"
        else:
            return f"❌ Response did not meet format requirements"
    else:
        return f"⚠️ Partial credit ({score:.1%})"


# Test the scoring functions
if __name__ == "__main__":
    print("Testing scoring functions...")
    
    # Test exact scoring
    assert score_exact("The answer is Paris", "Paris") == 1.0
    assert score_exact("London", "Paris") == 0.0
    assert score_exact("PARIS", "Paris") == 1.0
    print("✅ Exact scoring works")
    
    # Test numeric scoring
    assert score_numeric("The answer is 636", "636") == 1.0
    assert score_numeric("Approximately 36.0", "36") == 1.0
    assert score_numeric("I think 100", "36") == 0.0
    print("✅ Numeric scoring works")
    
    print("\nAll scoring tests passed!")